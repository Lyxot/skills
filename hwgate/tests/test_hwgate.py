"""Tests for hwgate's admission control.

Each test gets its own HWGATE_STATE_DIR and a small core budget, so nothing here
touches the machine's real slots.
"""

import fcntl
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import pytest

HWGATE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hwgate"


def wait_for_barrier(gate, slot, deadline=15.0):
    barrier = gate.leases.parent / (slot + ".barrier")
    until = time.time() + deadline
    while time.time() < until:
        if barrier.exists() and not lock_is_free(barrier):
            return barrier
        time.sleep(0.05)
    raise AssertionError("no job raised the {} barrier".format(slot))


def lock_is_free(path):
    with open(path, "r+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
    return True


def hold_lease(path, seconds=30):
    """Hold a planted lease as its job would, returning once the lock is really held.

    Until then the lease looks finished, and the next admission deletes it.
    """
    holder = subprocess.Popen([sys.executable, "-c",
                               "import fcntl,sys,time; f=open(sys.argv[1]); fcntl.flock(f, fcntl.LOCK_SH);"
                               " time.sleep({})".format(seconds), str(path)])
    until = time.time() + 10
    while lock_is_free(path):
        assert time.time() < until, "the planted lease was never locked"
        time.sleep(0.02)
    return holder


def load_hwgate():
    """The script as a module, for contracts no end-to-end run can reach on demand."""
    loader = importlib.machinery.SourceFileLoader("hwgate_module", str(HWGATE))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def reserved_cores(path):
    """Cores a lease claims; a lease being written or removed claims nothing yet."""
    try:
        return int(json.loads(path.read_text()).get("cpus", 0))
    except (OSError, ValueError, TypeError):
        return 0


@pytest.fixture
def gate(tmp_path):
    """Runs hwgate against a private state dir, and cleans up what it started."""
    started = []

    hidden = ("HWGATE_", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")

    def environ(**extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(hidden)}
        env["HWGATE_STATE_DIR"] = str(tmp_path / "state")
        env["HWGATE_CPU_BUDGET"] = "8"
        # The machine's real heat would otherwise decide whether a test's job is
        # admitted at all, and the suite runs while other agents load the Mac.
        env["HWGATE_FORCE_THERMAL"] = "0"
        for key, value in extra.items():
            env.pop(key, None) if value is None else env.update({key: str(value)})
        return env

    def run(*args, **extra):
        return subprocess.run(
            [sys.executable, str(HWGATE)] + list(args),
            env=environ(**extra), capture_output=True, text=True, timeout=60,
        )

    def background(*args, **extra):
        # Taken first: a job that starts before the baseline is read would be in it.
        before = running_leases()
        proc = subprocess.Popen(
            [sys.executable, str(HWGATE)] + list(args),
            env=environ(**extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        started.append(proc)
        wait_until_running(before, deadline=20.0)
        return proc

    def running_leases():
        """The leases that name a started job, so their reservation is final."""
        names = set()
        for lease in leases.glob("*.json") if leases.exists() else ():
            try:
                if "pgid" in json.loads(lease.read_text()):
                    names.add(lease.name)
            except (OSError, ValueError):
                pass
        return names

    def wait_until_running(before, deadline):
        until = time.time() + deadline
        while time.time() < until:
            if running_leases() - before:
                return
            time.sleep(0.05)
        raise AssertionError("the background job just started never took a reservation")

    leases = tmp_path / "state" / "leases"
    run.environ = environ
    run.background = background
    run.leases = leases
    yield run
    for proc in started:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_an_undeclared_cpu_job_is_charged_its_share_of_the_pool(gate):
    """Not every core: the slot holds three jobs, so it is charged a third of them.

    Charging it the whole budget would refuse the next declared job on a slot that
    still has room; charging it nothing would tell that job the machine is idle.
    """
    gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    assert "budget: 2/8 cores declared" in gate("status").stdout


def test_a_budget_smaller_than_the_seats_still_charges_a_core(gate):
    """A share that rounds to zero would let any number of undeclared jobs fit the budget."""
    gate.background("run", "--needs", "cpu", "--", "sleep", "20", HWGATE_CPU_BUDGET=2)
    assert "budget: 1/2 cores declared" in gate("status", HWGATE_CPU_BUDGET=2).stdout


def test_the_core_budget_defaults_to_the_performance_cores(gate):
    """Every other test sets the budget, so this is the only one that reads the machine."""
    p_cores = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                             capture_output=True, text=True).stdout.strip()
    status = gate("status", HWGATE_CPU_BUDGET=None).stdout
    assert "budget: 0/{} cores declared".format(p_cores) in status, status


def test_oversized_cpus_is_rejected_without_waiting(gate):
    started = time.time()
    result = gate("run", "--needs", "cpu", "--cpus", "99", "--", "true")
    assert result.returncode == 2
    assert "more than the 8 cores" in result.stderr
    assert time.time() - started < 5
    # Zero is not a small declaration: read as one, it would quietly become "undeclared".
    assert gate("run", "--needs", "cpu", "--cpus", "0", "--", "true").returncode == 2


def test_declared_cpu_jobs_share_the_slot(gate):
    gate.background("run", "--needs", "cpu", "--cpus", "4", "--", "sleep", "20")
    result = gate("run", "--needs", "cpu", "--cpus", "4", "--max-wait", "3", "--", "echo", "ran")
    assert result.returncode == 0
    assert "ran" in result.stdout


def test_cpu_budget_refuses_the_overflow(gate):
    gate.background("run", "--needs", "cpu", "--cpus", "6", "--", "sleep", "20")
    result = gate("run", "--needs", "cpu", "--cpus", "4", "--max-wait", "0.2", "--", "true")
    assert result.returncode == 75
    assert "cpu budget: 6 of 8 cores declared, this job needs 4 (shared by pid" in result.stderr


def test_undeclared_jobs_share_a_slot(gate):
    """Work that does not ask for speed runs alongside other work."""
    gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    assert "cpu: 2 of 3 jobs" in gate("status").stdout
    assert "budget: 4/8 cores declared" in gate("status").stdout


def test_a_declared_job_fits_beside_undeclared_ones_until_the_budget_is_full(gate):
    """Seats and cores are separate limits, and the tighter one decides."""
    gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    fits = gate("run", "--needs", "cpu", "--cpus", "6", "--max-wait", "0.2", "--", "echo", "ran")
    assert fits.returncode == 0, fits.stderr
    gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    result = gate("run", "--needs", "cpu", "--cpus", "6", "--max-wait", "0.2", "--", "true")
    assert result.returncode == 75
    assert "cpu budget: 4 of 8 cores declared, this job needs 6" in result.stderr


def test_a_slot_holds_no_more_than_three_jobs(gate):
    """Sharing is not unlimited: past three, waiting beats thrashing."""
    for _ in range(3):
        gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    result = gate("run", "--needs", "gpu", "--max-wait", "0.2", "--", "true")
    assert result.returncode == 75
    assert "gpu slot is full: 3 of 3 jobs" in result.stderr
    assert "gpu: 3 of 3 jobs" in gate("status").stdout


def test_status_leaves_an_abandoned_job_running(gate):
    """Reporting is read-only: a command that kills things on a look is a trap."""
    proc = gate.background("run", "--needs", "cpu", "--cpus", "8", "--timeout", "1", "--", "sleep", "20")
    lease = json.loads(next(gate.leases.iterdir()).read_text())
    os.kill(proc.pid, signal.SIGKILL)  # the wrapper, so nothing enforces the timeout
    proc.wait()  # and reap it, so the pid is really gone rather than a zombie
    time.sleep(1.5)
    try:
        gate("status")
        os.killpg(lease["pgid"], 0)
        # Admission is where an abandoned job past its deadline is reaped.
        assert gate("run", "--needs", "cpu", "--cpus", "8", "--max-wait", "5", "--", "true").returncode == 0
        with pytest.raises(ProcessLookupError):
            os.killpg(lease["pgid"], 0)
    finally:
        try:
            os.killpg(lease["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("content", ["[]", '"words"', "17", '{"slots": []}', "{ truncated"])
def test_a_lease_that_is_not_a_readable_record_counts_as_the_whole_machine(gate, content):
    gate("status")
    broken = gate.leases / "4242-broken.json"
    broken.write_text(content)
    holder = hold_lease(broken, 30)
    try:
        status = gate("status")
        assert status.returncode == 0
        assert "budget: 8/8 cores declared" in status.stdout
        refused = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "true")
        assert refused.returncode == 75
        assert "cpu budget: 8 of 8 cores declared" in refused.stderr, refused.stderr
    finally:
        holder.kill()
        holder.wait()


def test_a_lease_claiming_a_negative_amount_cannot_free_up_budget(gate):
    gate("status")
    (gate.leases / "4242-neg.json").write_text(json.dumps({"cpus": -100, "slots": ["cpu"]}))
    holder = hold_lease(gate.leases / "4242-neg.json", 30)
    try:
        gate.background("run", "--needs", "cpu", "--cpus", "8", "--", "sleep", "20")
        refused = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "true")
        assert refused.returncode == 75
        assert "cpu budget: 8 of 8 cores declared" in refused.stderr
    finally:
        holder.kill()
        holder.wait()


def test_an_unusable_budget_variable_is_reported_not_raised(gate):
    result = gate("run", "--needs", "gpu", "--", "true", HWGATE_CPU_BUDGET="not a number")
    assert result.returncode == 2
    assert "HWGATE_CPU_BUDGET='not a number' is unusable" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_live_lease_from_another_job_is_counted_and_left_alone(gate):
    gate("status")
    foreign = gate.leases / "4242-abc.json"
    foreign.write_text(json.dumps({"cpus": 2, "started": time.time(), "cmd": "other"}))
    holder = hold_lease(foreign, 30)
    try:
        assert gate("run", "--needs", "cpu", "--cpus", "6", "--max-wait", "2", "--", "true").returncode == 0
        refused = gate("run", "--needs", "cpu", "--cpus", "7", "--max-wait", "0.2", "--", "true")
        assert refused.returncode == 75
        assert "cpu budget: 2 of 8 cores declared, this job needs 7" in refused.stderr
        assert json.loads(foreign.read_text())["cpus"] == 2
        # It names no slots, so it is read as holding both: two more jobs fill the slot.
        for _ in range(2):
            gate.background("run", "--needs", "gpu", "--", "sleep", "20")
        full = gate("run", "--needs", "gpu", "--max-wait", "0.2", "--", "true")
        assert full.returncode == 75
        assert "gpu slot is full: 3 of 3 jobs" in full.stderr, full.stderr
    finally:
        holder.kill()
        holder.wait()


def test_gpu_jobs_share_the_gpu(gate):
    gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    result = gate("run", "--needs", "gpu", "--max-wait", "3", "--", "echo", "ran")
    assert result.returncode == 0
    assert "ran" in result.stdout


def test_only_a_stable_job_keeps_a_slot_to_itself(gate):
    """--stable is now the only way to be alone on the machine."""
    gate.background("run", "--needs", "gpu", "--stable", "--", "sleep", "20", HWGATE_COOLDOWN=0)
    result = gate("run", "--needs", "gpu", "--max-wait", "0.2", "--", "true")
    assert result.returncode == 75
    assert "gpu slot busy (held by pid" in result.stderr


def test_declared_cpus_reach_the_job_as_thread_limits(gate):
    result = gate("run", "--needs", "cpu", "--cpus", "3", "--", "sh", "-c",
                  "echo $HWGATE_CPUS $OMP_NUM_THREADS $RAYON_NUM_THREADS")
    assert result.stdout.split() == ["3", "3", "3"]


def test_a_thread_limit_the_caller_chose_is_left_alone(gate):
    result = gate("run", "--needs", "cpu", "--cpus", "3", "--", "sh", "-c", "echo $OMP_NUM_THREADS",
                  OMP_NUM_THREADS=1)
    assert result.stdout.strip() == "1"


def test_heat_stops_a_new_job_sharing_with_one_admitted_while_cool(gate):
    gate.background("run", "--needs", "cpu", "--cpus", "1", "--", "sleep", "20")
    result = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "true",
                  HWGATE_FORCE_THERMAL=1)
    assert result.returncode == 75
    assert "one heavy job at a time" in result.stderr


def test_a_stable_job_takes_every_slot(gate):
    """Only --stable makes a cpu job wait for the gpu slot as well."""
    gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    shares = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "echo", "ran")
    assert shares.returncode == 0
    alone = gate("run", "--needs", "cpu", "--stable", "--max-wait", "0.2", "--", "true",
                 HWGATE_COOLDOWN=0)
    assert alone.returncode == 75
    assert "gpu slot busy" in alone.stderr and "--stable runs alone" in alone.stderr


def test_a_nested_job_is_not_held_back_by_a_barrier(gate):
    """Whoever raised it may be waiting on this job's parent, which cannot end.

    The waiter here is held up by an unrelated sharer, which is enough to show
    the exemption; the cycle it protects against needs the waiter to be behind
    the parent itself.
    """
    gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    waiter = subprocess.Popen(
        [sys.executable, str(HWGATE), "run", "--needs", "gpu,cpu", "--stable",
         "--max-wait", "25", "--", "true"],
        env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    parent = None
    try:
        wait_for_barrier(gate, "gpu")
        parent = subprocess.Popen(
            [sys.executable, str(HWGATE), "run", "--needs", "cpu", "--cpus", "4", "--",
             sys.executable, str(HWGATE), "run", "--needs", "gpu", "--max-wait", "8",
             "--", "echo", "nested ran"],
            env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
        out, _ = parent.communicate(timeout=40)
        assert parent.returncode == 0
        assert "nested ran" in out
    finally:
        for proc in (waiter, parent):
            if proc is not None:
                proc.kill()
                proc.wait()


def test_a_stable_job_on_a_full_budget_waits_on_the_lock_and_raises_its_barrier(gate):
    """Refused by the budget instead, it would raise no barrier, and sharers declaring
    cores could keep it out for as long as they kept arriving."""
    for _ in range(2):
        gate.background("run", "--needs", "cpu", "--cpus", "4", "--", "sleep", "20")
    waiter = gate("run", "--needs", "cpu", "--stable", "--max-wait", "0.2", "--", "true",
                  HWGATE_COOLDOWN=0)
    assert waiter.returncode == 75
    assert "waiting for the whole cpu slot" in waiter.stderr, waiter.stderr
    assert "cpu budget" not in waiter.stderr


def test_a_job_waiting_for_the_whole_gpu_slot_holds_sharers_back_too(gate):
    gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    waiter = subprocess.Popen(
        [sys.executable, str(HWGATE), "run", "--needs", "gpu", "--stable",
         "--max-wait", "25", "--", "true"],
        env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        wait_for_barrier(gate, "gpu")
        refused = gate("run", "--needs", "gpu", "--max-wait", "0.2", "--", "true")
        assert refused.returncode == 75
        assert "is waiting for the whole gpu slot and goes first" in refused.stderr
    finally:
        waiter.kill()
        waiter.wait()


def test_no_barrier_is_raised_when_no_sharer_is_in_the_way(gate):
    """Two jobs that both want the machine alone owe each other nothing."""
    gate.background("run", "--needs", "cpu", "--stable", "--", "sleep", "20", HWGATE_COOLDOWN=0)
    result = gate("run", "--needs", "cpu", "--stable", "--max-wait", "0.2", "--", "true",
                  HWGATE_COOLDOWN=0)
    assert result.returncode == 75
    assert "waiting for the whole" not in result.stderr
    # Raising one creates the file, so its absence is the durable proof. Both slots are
    # checked: this job asks for both, and it is refused on whichever is tried first.
    for slot in ("cpu", "gpu"):
        assert not (gate.leases.parent / (slot + ".barrier")).exists(), slot


def test_a_job_waiting_for_a_whole_slot_is_not_starved_by_sharers(gate):
    holder = gate.background("run", "--needs", "cpu", "--cpus", "4", "--", "sleep", "20")
    waiter = subprocess.Popen(
        [sys.executable, str(HWGATE), "run", "--needs", "cpu", "--stable",
         "--max-wait", "20", "--", "true"],
        env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        wait_for_barrier(gate, "cpu")
        # The stream of sharers that would otherwise keep the slot busy forever.
        refused = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "true")
        assert refused.returncode == 75
        assert "goes first" in refused.stderr
    finally:
        holder.terminate()
        holder.wait()
        assert waiter.wait(timeout=30) == 0
        waiter.stdout.close()
        waiter.stderr.close()


def test_a_killed_job_gives_its_budget_back(gate):
    proc = gate.background("run", "--needs", "cpu", "--cpus", "8", "--", "sleep", "20")
    lease = json.loads(next(gate.leases.iterdir()).read_text())
    os.kill(proc.pid, signal.SIGKILL)
    os.killpg(lease["pgid"], signal.SIGKILL)
    result = gate("run", "--needs", "cpu", "--cpus", "8", "--max-wait", "5", "--", "echo", "ran")
    assert result.returncode == 0
    assert list(gate.leases.iterdir()) == []


def test_a_nested_job_reuses_the_slots_it_already_holds(gate):
    """It runs straight through, so it never takes a reservation of its own."""
    result = gate("run", "--needs", "cpu", "--cpus", "4", "--", sys.executable, str(HWGATE),
                  "run", "--needs", "cpu", "--cpus", "4", "--max-wait", "3", "--", "sh", "-c",
                  "ls $HWGATE_STATE_DIR/leases | wc -l", HWGATE_CPU_BUDGET=4)
    assert result.returncode == 0
    assert result.stdout.strip() == "1"


def test_a_nested_job_does_not_inflate_what_the_machine_reports(gate):
    gate.background("run", "--needs", "cpu", "--cpus", "4", "--", sys.executable, str(HWGATE),
                    "run", "--needs", "gpu,cpu", "--cpus", "4", "--", "sleep", "20")
    # background() returns on the parent's lease; the nested job takes the gpu slot after.
    until = time.time() + 10
    while "gpu: 1 of 3 jobs" not in gate("status").stdout:
        assert time.time() < until, "the nested job never took its gpu seat"
        time.sleep(0.05)
    assert "budget: 4/8 cores declared" in gate("status").stdout


def test_a_declared_core_count_claims_the_cpu_slot_too(gate):
    """A job from an older hwgate holds the slot and has no lease to be counted."""
    gate("status")
    lock = open(gate.leases.parent / "cpu.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = gate("run", "--needs", "gpu", "--cpus", "1", "--max-wait", "0.2", "--", "true")
        assert result.returncode == 75
        assert "cpu slot busy" in result.stderr
    finally:
        lock.close()


def test_admission_never_hands_out_more_than_the_budget(gate):
    """Two jobs must not both fit into the same free core."""
    jobs = [subprocess.Popen([sys.executable, str(HWGATE), "run", "--needs", "cpu", "--cpus", "1",
                              "--max-wait", "10", "--", "sleep", "1"],
                             env=gate.environ(HWGATE_CPU_BUDGET=3),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(12)]
    try:
        peak = 0
        while any(job.poll() is None for job in jobs):
            peak = max(peak, sum(reserved_cores(p) for p in gate.leases.glob("*.json")))
            assert peak <= 3, "handed out {} cores of a 3-core budget".format(peak)
        assert peak >= 2, "the run never overlapped, so it proves nothing"
    finally:
        for job in jobs:
            job.kill()
            job.wait()


def test_status_reports_the_budget_and_the_sharers(gate):
    assert "budget: 0/8 cores declared" in gate("status").stdout
    gate.background("run", "--needs", "gpu,cpu", "--cpus", "5", "--", "sleep", "20")
    status = gate("status").stdout
    assert "budget: 5/8 cores declared" in status
    assert "cpu: 1 of 3 jobs" in status
    assert "gpu: 1 of 3 jobs" in status
    assert status.count("5 cores, pid ") == 2


def test_status_says_so_when_it_cannot_read_the_budget(gate):
    gate("status")
    lock = open(gate.leases.parent / "admit.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "budget: unknown" in gate("status").stdout
    finally:
        lock.close()


def test_a_job_inside_a_gated_job_is_recognised_without_its_environment(gate, tmp_path):
    """A scrubbed environment must not turn a nested job into a deadlock.

    The parent has to be admitted before the barrier goes up, or it would wait
    out the sharer and the nested job would never meet a barrier at all.
    """
    go = tmp_path / "go"
    parent = subprocess.Popen(
        [sys.executable, str(HWGATE), "run", "--needs", "cpu", "--cpus", "4", "--", "sh", "-c",
         "while [ ! -f {} ]; do sleep 0.1; done; exec "
         "env -u HWGATE_HELD -u HWGATE_HELD_CPUS "
         "{} {} run --needs cpu "
         "--cpus 1 --max-wait 8 -- echo inner ran".format(go, sys.executable, HWGATE)],
        env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    waiter = None
    try:
        for _ in range(200):
            if any("pgid" in p.read_text() for p in gate.leases.glob("*.json") if p.is_file()):
                break
            time.sleep(0.05)
        waiter = subprocess.Popen(
            [sys.executable, str(HWGATE), "run", "--needs", "cpu", "--stable",
             "--max-wait", "40", "--", "true"],
            env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
        wait_for_barrier(gate, "cpu")
        go.write_text("")
        out, err = parent.communicate(timeout=40)
        assert parent.returncode == 0, err
        assert "inner ran" in out
    finally:
        for proc in (parent, waiter):
            if proc is not None:
                proc.kill()
                proc.wait()


def test_a_barrier_names_who_is_waiting(gate):
    gate.background("run", "--needs", "cpu", "--cpus", "4", "--", "sleep", "20")
    waiter = subprocess.Popen(
        [sys.executable, str(HWGATE), "run", "--needs", "cpu", "--stable",
         "--max-wait", "25", "--", "true"],
        env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        wait_for_barrier(gate, "cpu")
        refused = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "true")
        assert refused.returncode == 75
        assert "pid {} ".format(waiter.pid) in refused.stderr
        assert "is waiting for the whole cpu slot and goes first" in refused.stderr
        assert "pid {} ".format(waiter.pid) in gate("status").stdout
    finally:
        waiter.kill()
        waiter.wait()


def test_a_core_budget_of_zero_is_refused_rather_than_read_as_one(gate):
    result = gate("run", "--needs", "cpu", "--", "true", HWGATE_CPU_BUDGET=0)
    assert result.returncode == 2
    assert "HWGATE_CPU_BUDGET=" in result.stderr and "at least 1" in result.stderr


def test_a_nested_job_cannot_take_more_cores_than_the_budget(gate):
    """Its parent's reservation is subtracted, but what it declares beyond that is not free."""
    gate.background("run", "--needs", "cpu", "--cpus", "4", "--", "sleep", "20")
    result = gate("run", "--needs", "cpu", "--cpus", "4", "--", sys.executable, str(HWGATE),
                  "run", "--needs", "gpu,cpu", "--cpus", "8", "--max-wait", "0.2",
                  "--", "echo", "over budget")
    assert result.returncode == 75
    assert "cpu budget: 8 of 8 cores declared, this job needs 4" in result.stderr
    assert "over budget" not in result.stdout


@pytest.mark.parametrize("parent", [
    pytest.param(["run", "--needs", "gpu"], id="parent-shares-the-slot"),
    pytest.param(["run", "--needs", "gpu", "--stable"], id="parent-holds-every-slot-whole"),
])
def test_stable_refuses_to_run_inside_any_gated_job(gate, parent):
    """Nesting skips acquisition, and acquisition is where --stable gets its guarantees.

    A parent holding the slot whole is no better than one sharing it: the nested job
    still never takes the other slot, never waits for nominal heat, and is never
    watched for throttling, so it would report numbers it did not earn.
    """
    result = gate(*parent, "--", sys.executable, str(HWGATE),
                  "run", "--needs", "cpu", "--stable", "--max-wait", "2", "--", "echo", "measured",
                  HWGATE_COOLDOWN=0)
    assert result.returncode == 2
    assert "--stable cannot run inside another gated job" in result.stderr
    assert "measured" not in result.stdout


def test_stable_runs_normally_when_nothing_encloses_it(gate):
    """The refusal must not reach a --stable job that is nobody's child."""
    result = gate("run", "--needs", "gpu", "--stable", "--max-wait", "3", "--", "echo", "measured",
                  HWGATE_COOLDOWN=0)
    assert result.returncode == 0
    assert "measured" in result.stdout
    assert "stable run:" in result.stderr


def test_a_nested_job_is_charged_what_it_declares_beyond_its_parent(gate):
    """It cannot be made to wait for its own parent, but it can be counted."""
    result = gate("run", "--needs", "cpu", "--cpus", "2", "--", sys.executable, str(HWGATE),
                  "run", "--needs", "cpu", "--cpus", "8", "--", sys.executable, str(HWGATE), "status")
    assert result.returncode == 0, result.stderr
    assert "8/8 cores" in result.stdout, result.stdout
    assert "charging the difference" in result.stderr
    # The extra charge is cores only: one job is still one seat, or two such jobs fill a slot.
    assert "cpu: 1 of 3 jobs" in result.stdout, result.stdout


def test_the_charge_for_a_nested_job_holds_the_next_job_back(gate):
    """The point of counting it: a job that *can* wait is told the truth."""
    gate.background("run", "--needs", "cpu", "--cpus", "2", "--", sys.executable, str(HWGATE),
                    "run", "--needs", "cpu", "--cpus", "8", "--", "sleep", "20")
    # background() returns on the parent's lease; the child's charge is written after it.
    until = time.time() + 10
    while sum(map(reserved_cores, gate.leases.glob("*.json"))) < 8:
        assert time.time() < until, "the nested job's extra charge never reached the ledger"
        time.sleep(0.05)
    result = gate("run", "--needs", "gpu", "--cpus", "4", "--max-wait", "0.2", "--", "echo", "ran")
    assert result.returncode == 75
    assert "cpu budget: 8 of 8 cores declared" in result.stderr
    assert "ran" not in result.stdout


def test_a_nested_job_whose_extra_cores_fit_still_runs(gate):
    """The charge must not turn into a refusal on the one path that cannot wait."""
    result = gate("run", "--needs", "cpu", "--cpus", "2", "--", sys.executable, str(HWGATE),
                  "run", "--needs", "cpu", "--cpus", "5", "--", "echo", "ran")
    assert result.returncode == 0, result.stderr
    assert "ran" in result.stdout


def test_a_nested_job_sizes_its_thread_pools_to_its_own_declaration(gate):
    """Thread counts hwgate gave the enclosing job would oversubscribe this one."""
    result = gate("run", "--needs", "cpu", "--cpus", "8", "--", sys.executable, str(HWGATE),
                  "run", "--needs", "cpu", "--cpus", "2", "--", "sh", "-c",
                  "echo $HWGATE_CPUS $OMP_NUM_THREADS $RAYON_NUM_THREADS")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["2", "2", "2"]


def test_the_nested_shortcut_still_hands_the_job_what_it_declared(gate):
    result = gate("run", "--needs", "cpu", "--", sys.executable, str(HWGATE),
                  "run", "--needs", "cpu", "--cpus", "2", "--", "sh", "-c",
                  "echo $HWGATE_CPUS $OMP_NUM_THREADS")
    assert result.returncode == 0
    assert result.stdout.split() == ["2", "2"]


def test_stable_is_refused_from_a_scrubbed_environment_inside_a_job(gate):
    """The ledger answers when the variables are gone, as it does for barriers."""
    result = gate("run", "--needs", "cpu", "--", "sh", "-c",
                  "exec env -u HWGATE_HELD -u HWGATE_HELD_CPUS "
                  "{} {} run --needs gpu --stable --max-wait 2 -- echo measured"
                  .format(sys.executable, HWGATE), HWGATE_COOLDOWN=0)
    assert result.returncode == 2
    assert "--stable cannot run inside another gated job" in result.stderr
    assert "measured" not in result.stdout


def test_stable_is_refused_by_an_inherited_variable_with_no_lease_behind_it(gate):
    """A shell outliving its job still says it holds a slot, and is taken at its word.

    Nothing in the ledger corroborates it, so this is the one case where the two
    signals disagree. Refusing costs a message; admitting costs measurements taken
    beside whatever that shell is still running.
    """
    result = gate("run", "--needs", "gpu", "--stable", "--max-wait", "2", "--", "echo", "measured",
                  HWGATE_HELD="cpu", HWGATE_COOLDOWN=0)
    assert result.returncode == 2
    assert "--stable cannot run inside another gated job" in result.stderr


def test_three_undeclared_jobs_fit_the_seats_the_slot_offers(gate):
    """The share must round down, or the budget refuses the last seat the slot allows.

    Eight cores over three seats is two each: three jobs then declare six and fit. At
    three each they would declare nine, and the third would be refused for the budget
    while a seat sat free, which contradicts the limit the slot advertises.
    """
    for _ in range(2):
        gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    third = gate("run", "--needs", "cpu", "--max-wait", "2", "--", "echo", "ran")
    assert third.returncode == 0, third.stderr
    assert "ran" in third.stdout


def test_the_cpu_slot_also_stops_at_three_jobs(gate):
    """The seat limit is a property of a slot, not something the gpu path does."""
    for _ in range(3):
        gate.background("run", "--needs", "cpu", "--cpus", "1", "--", "sleep", "20")
    result = gate("run", "--needs", "cpu", "--cpus", "1", "--max-wait", "0.2", "--", "true")
    assert result.returncode == 75
    assert "cpu slot is full: 3 of 3 jobs" in result.stderr


def test_a_lease_that_cannot_be_read_occupies_a_seat(gate):
    """Conservative on seats as well as on cores: a torn read must not free a seat.

    Asserted through admission rather than through `status`, because what matters is
    that the seat is not handed out twice.
    """
    for _ in range(2):
        gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    damaged = gate.leases / "damaged.json"
    damaged.write_text("{ truncated")
    holder = hold_lease(damaged, 20)
    try:
        # Declares nothing, so the budget cannot answer and only the seat count can.
        result = gate("run", "--needs", "gpu", "--max-wait", "0.2", "--", "true")
        assert result.returncode == 75
        assert "gpu slot is full: 3 of 3 jobs" in result.stderr
    finally:
        holder.kill()
        holder.wait()


def test_a_barrier_follows_the_slot_that_is_actually_blocking(gate):
    """A barrier is held only while that slot is in the way, and is dropped otherwise.

    The waiter is alive and still waiting throughout, so this covers the stretch the
    kernel's cleanup on exit cannot: a barrier left on a slot that has gone quiet
    would keep new jobs off it for as long as the waiter runs.
    """
    sharer = gate.background("run", "--needs", "gpu", "--", "sleep", "3")
    waiter = subprocess.Popen(
        [sys.executable, str(HWGATE), "run", "--needs", "gpu,cpu", "--stable",
         "--max-wait", "25", "--", "true"],
        env=gate.environ(HWGATE_COOLDOWN=0), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        wait_for_barrier(gate, "gpu")
        # Admitted while the barrier sits on the other slot, and now it is the blocker.
        gate.background("run", "--needs", "cpu", "--", "sleep", "20")
        sharer.wait(timeout=20)
        wait_for_barrier(gate, "cpu")
        assert lock_is_free(gate.leases.parent / "gpu.barrier"), "the gpu barrier outlived its sharer"
        assert waiter.poll() is None, "the waiter stopped waiting, so this proves nothing"
    finally:
        waiter.kill()
        waiter.wait()


def test_a_job_joining_a_busy_slot_is_told_who_else_is_on_it(gate):
    """A shared run is slower than a lone one, and nothing else in the output says so.

    The line has to name the jobs, not just count them: the reason to read it is to
    decide whether to wait for them or accept the contention.
    """
    gate.background("run", "--needs", "gpu,cpu", "--cpus", "2", "--", "sleep", "20")
    gate.background("run", "--needs", "gpu,cpu", "--cpus", "3", "--", "sleep", "21")
    result = gate("run", "--needs", "gpu,cpu", "--cpus", "1", "--max-wait", "5", "--", "echo", "ran")
    assert result.returncode == 0, result.stderr
    assert "ran" in result.stdout
    assert "gpu: 3 of 3 seats taken, alongside pid " in result.stderr
    # The core figure counts this job too, or it understates what the slot is carrying.
    assert "cpu: 3 of 3 seats taken, 6 of 8 cores declared, alongside pid " in result.stderr
    # Each of the two other jobs, on each of the two slots.
    assert result.stderr.count("sleep 20") == 2 and result.stderr.count("sleep 21") == 2, result.stderr


def test_a_job_that_has_a_slot_to_itself_is_told_nothing(gate):
    """Every gated job would otherwise carry a line saying it shares with nobody."""
    alone = gate("run", "--needs", "gpu,cpu", "--cpus", "2", "--", "true")
    assert "seats taken" not in alone.stderr, alone.stderr
    gate.background("run", "--needs", "cpu", "--", "sleep", "20")
    other_slot = gate("run", "--needs", "gpu", "--max-wait", "5", "--", "true")
    assert "seats taken" not in other_slot.stderr, other_slot.stderr


def full_pipe():
    """A pipe whose buffer is already full, so the next write to it blocks."""
    read_end, write_end = os.pipe()
    os.set_blocking(write_end, False)
    try:
        while True:
            os.write(write_end, b"x" * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(write_end, True)
    return read_end, write_end


@pytest.mark.parametrize("stalled", [
    # Long enough that its lease is still there to be seen once it has started.
    pytest.param(["run", "--needs", "gpu", "--", "sleep", "2"], id="joining-a-busy-slot"),
    pytest.param(["run", "--needs", "gpu", "--stable", "--max-wait", "30", "--", "true"],
                 id="raising-a-barrier"),
])
def test_a_stderr_nobody_is_reading_holds_nothing_up(gate, stalled):
    """A caller that reads a captured stderr only at the end can leave the pipe full.

    A write that waited for it would hold whatever hwgate held at that moment: the admit
    lock, stopping every hwgate on the machine, or a barrier, turning away every new job
    on that slot. Neither the job itself nor anyone else may wait on that reader.
    """
    holder = gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    read_end, write_end = full_pipe()
    proc = subprocess.Popen([sys.executable, str(HWGATE)] + stalled, env=gate.environ(HWGATE_COOLDOWN=0),
                            stdout=subprocess.DEVNULL, stderr=write_end)
    os.close(write_end)
    barrier = gate.leases.parent / "gpu.barrier"
    try:
        # Establish that it met the holder: it took a seat beside it, or raised a barrier.
        until = time.time() + 10
        while (len(list(gate.leases.glob("*.json"))) < 2
               and not (barrier.exists() and not lock_is_free(barrier))):
            assert proc.poll() is None, "it finished without meeting the holder"
            assert time.time() < until, "it never met the holder"
            time.sleep(0.02)
        holder.terminate()
        result = gate("run", "--needs", "gpu", "--max-wait", "8", "--", "echo", "ran")
        assert result.returncode == 0, result.stderr
        assert "ran" in result.stdout
        assert proc.wait(timeout=10) == 0
    finally:
        proc.kill()
        proc.wait()
        os.close(read_end)


@pytest.mark.parametrize("ahead", [
    pytest.param(["run", "--needs", "gpu", "--", "sleep", "20"], id="admitted-beside-a-sharer"),
    pytest.param(["run", "--needs", "gpu", "--stable", "--", "sleep", "1.5"], id="admitted-after-waiting"),
])
def test_a_stderr_whose_reader_has_gone_does_not_cost_the_run(gate, ahead):
    """hwgate's messages are only information, whichever one finds the reader gone."""
    gate.background(*ahead, HWGATE_COOLDOWN=0)
    read_end, write_end = os.pipe()
    os.close(read_end)
    proc = subprocess.run([sys.executable, str(HWGATE), "run", "--needs", "gpu", "--max-wait", "15", "--",
                           "sh", "-c", "echo ran; exit 3"],
                          env=gate.environ(), stdout=subprocess.PIPE, stderr=write_end, text=True,
                          timeout=30)
    os.close(write_end)
    assert "ran" in proc.stdout
    # The job's own status, not the 120 Python reports when stderr cannot be flushed at exit.
    assert proc.returncode == 3

def test_a_refusal_names_the_jobs_on_the_slot_not_a_stale_holder_record(gate):
    """A whole-slot job killed with its wrapper leaves its record behind; sharers never
    overwrite it, so a message trusting it would send someone after a dead pid."""
    gate("status")
    (gate.leases.parent / "gpu.json").write_text(json.dumps(
        {"child_pid": 999999, "started": time.time(), "cmd": "a stable job long gone"}))
    for _ in range(3):
        gate.background("run", "--needs", "gpu", "--", "sleep", "20")
    result = gate("run", "--needs", "gpu", "--max-wait", "0.2", "--", "true")
    assert result.returncode == 75
    assert "long gone" not in result.stderr, result.stderr
    assert result.stderr.count("sleep 20") >= 3


def test_no_signal_to_a_group_crashes_on_the_eperm_macos_gives_for_exited_members(monkeypatch):
    """macOS answers EPERM, not ESRCH, for a group left with only exited members.

    That is a group with nothing to kill. Treated as an error, it crashed the wrapper
    after the job had finished, or an admission that was reaping an abandoned job.
    It depends on how fast launchd reaps, so it is pinned here rather than raced for.
    """
    module = load_hwgate()

    def refuse(pgid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(module.os, "killpg", refuse)
    module.kill_group(12345, signal.SIGKILL)
    # Past its deadline, wrapper gone, group reported alive: the reaper signals it.
    module.reap_abandoned({"deadline": 1, "wrapper_pid": 999999, "pgid": 12345,
                           "child_pid": 999998}, "gpu job")


def test_nothing_is_written_while_the_admit_lock_is_held(monkeypatch, tmp_path):
    """Even a bounded write would be waited out by every hwgate on the machine."""
    module = load_hwgate()
    monkeypatch.setattr(module, "STATE_DIR", str(tmp_path))
    written = []
    monkeypatch.setattr(module.os, "write", lambda fd, data: written.append(
        (data, lock_is_free(tmp_path / "admit.lock"))) or len(data))
    with module.admit_lock() as locked:
        assert locked
        module.log("under the lock")
    module.log("after it")
    # Each message is written, in order, and only once the lock is free.
    assert written == [(b"hwgate: under the lock\n", True), (b"hwgate: after it\n", True)]


def test_status_runs_nothing_external_while_holding_the_admit_lock(monkeypatch, tmp_path, capsys):
    """An unreadable lease is charged the whole budget, which may ask sysctl; under the
    lock, a slow answer would be waited out by every hwgate on the machine."""
    module = load_hwgate()
    monkeypatch.setattr(module, "STATE_DIR", str(tmp_path))
    monkeypatch.delenv("HWGATE_CPU_BUDGET", raising=False)
    (tmp_path / "leases").mkdir()
    (tmp_path / "admit.lock").touch()
    torn = tmp_path / "leases" / "4242-torn.json"
    torn.write_text("{ truncated")
    asked = []
    monkeypatch.setattr(module, "sysctl", lambda name: asked.append(lock_is_free(tmp_path / "admit.lock")) or 12)
    holder = hold_lease(torn)
    try:
        module.cmd_status(None)
    finally:
        holder.kill()
        holder.wait()
    assert "budget: 12/12 cores declared" in capsys.readouterr().out
    assert asked == [True]


@pytest.mark.parametrize("recorded, now, reaped", [
    pytest.param("Wed Oct  1 21:00:00 2026", "Wed Oct  1 21:00:00 2026", True, id="identified"),
    pytest.param("Wed Oct  1 21:00:00 2026", "Wed Oct  1 22:00:00 2026", False, id="pid-reused"),
    pytest.param(None, None, False, id="unknown-both-times"),
    pytest.param("Wed Oct  1 21:00:00 2026", None, False, id="unknown-now"),
])
def test_an_abandoned_job_is_reaped_only_when_ps_identifies_it(monkeypatch, recorded, now, reaped):
    """Two start times ps could not give are not a match: the pid may belong to anyone."""
    module = load_hwgate()
    killed = []
    monkeypatch.setattr(module, "pid_alive", lambda pid: pid != 111)
    monkeypatch.setattr(module, "group_alive", lambda pgid: True)
    monkeypatch.setattr(module, "process_start", lambda pid: now)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: killed.append(pgid))
    module.reap_abandoned({"deadline": 1, "wrapper_pid": 111, "pgid": 222, "child_pid": 222,
                           "child_start": recorded}, "gpu job")
    assert killed == ([222] if reaped else [])
