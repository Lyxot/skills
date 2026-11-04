import importlib.machinery
import importlib.util
import os
import pathlib
import subprocess
import sys
import time
import types

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(SCRIPTS / name))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def fake_run(outputs):
    """subprocess.run answering each command by its program name, recording the calls."""
    calls = []

    def run(cmd, **_):
        calls.append(cmd)
        out, code = outputs(cmd)
        return types.SimpleNamespace(stdout=out, stderr="", returncode=code)

    return run, calls


PING6 = """PING6(56=40+8+8 bytes) fe80::aa%bridge0 --> ff02::1
16 bytes from fe80::aa%bridge0, icmp_seq=0 hlim=64 time=0.130 ms
16 bytes from fe80::bb%bridge0, icmp_seq=0 hlim=64 time=0.413 ms
16 bytes from fe80::aa%bridge0, icmp_seq=1 hlim=64 time=0.135 ms
16 bytes from fe80::bb%bridge0, icmp_seq=1 hlim=64 time=0.400 ms
"""
IFCONFIG = "bridge0: flags=8863<UP>\n\tinet6 fe80::aa%bridge0 prefixlen 64 secured scopeid 0xd\n"


def answers(extra=""):
    return lambda cmd: (PING6 + extra, 0) if "ping6" in cmd[0] else (IFCONFIG, 0)


def test_the_server_is_the_one_other_host_answering_on_the_link(monkeypatch):
    helper = load("tbshare-mount")
    run, calls = fake_run(answers())
    monkeypatch.setattr(helper.subprocess, "run", run)
    assert helper.link_address("bridge0") == "fe80::bb"
    assert calls[0] == ["/sbin/ping6", "-c", "3", "-i", "0.5", "-I", "bridge0", "ff02::1"]


def test_two_other_hosts_on_the_link_are_refused_rather_than_guessed(monkeypatch):
    helper = load("tbshare-mount")
    run, _ = fake_run(answers("16 bytes from fe80::cc%bridge0, icmp_seq=1 hlim=64 time=0.4 ms\n"))
    monkeypatch.setattr(helper.subprocess, "run", run)
    with pytest.raises(SystemExit):
        helper.link_address("bridge0")


def test_replies_from_another_interface_or_only_from_itself_find_no_server(monkeypatch):
    helper = load("tbshare-mount")
    only_self = "16 bytes from fe80::aa%bridge0, icmp_seq=0\n16 bytes from fe80::dd%en0, icmp_seq=0\n"
    run, _ = fake_run(lambda cmd: (only_self, 0) if "ping6" in cmd[0] else (IFCONFIG, 0))
    monkeypatch.setattr(helper.subprocess, "run", run)
    assert helper.link_address("bridge0") is None


def test_the_user_side_sees_the_link_reach_the_server_only_through_another_host(monkeypatch):
    tb = load("tbshare")
    run, _ = fake_run(answers())
    monkeypatch.setattr(tb.subprocess, "run", run)
    assert tb.link_reaches({"interface": "bridge0"})
    run, _ = fake_run(lambda cmd: ("16 bytes from fe80::aa%bridge0, icmp_seq=0\n", 0) if "ping6" in cmd[0]
                      else (IFCONFIG, 0))
    monkeypatch.setattr(tb.subprocess, "run", run)
    assert not tb.link_reaches({"interface": "bridge0"})
    # Two other hosts: the helper would refuse to pick one, so the share must not be unmounted for it.
    run, _ = fake_run(answers("16 bytes from fe80::cc%bridge0, icmp_seq=1 hlim=64 time=0.4 ms\n"))
    monkeypatch.setattr(tb.subprocess, "run", run)
    assert not tb.link_reaches({"interface": "bridge0"})


def fake_mount(monkeypatch, helper, tmp_path, body):
    fake = tmp_path / "mount_smbfs"
    fake.write_text("#!/bin/sh\n" + body)
    fake.chmod(0o755)
    monkeypatch.setattr(helper, "MOUNT_SMBFS", str(fake))
    monkeypatch.setattr(os, "setgroups", lambda groups: None)
    # Root-only steps; the tests run as the user the child runs as.
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)


def test_the_password_is_typed_at_the_prompt_with_echo_off(monkeypatch, tmp_path):
    helper = load("tbshare-mount")
    seen = tmp_path / "seen"
    # Leaves echo as it finds it, so only the helper keeps the password from coming back.
    fake_mount(monkeypatch, helper, tmp_path, "printf 'Password for server: ' > /dev/tty\nIFS= read -r pw < /dev/tty\n"
               "printf '%s|%s|%s\\n' \"$pw\" \"$1\" \"$2\" > {}\necho mounted\n".format(seen))
    handed = []
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: handed.append(("chown", uid, gid)))
    real_fchmod = os.fchmod
    monkeypatch.setattr(os, "fchmod", lambda fd, mode: handed.append(("chmod", mode)) or real_fchmod(fd, mode))
    code, output = helper.mount_as(os.getuid(), os.getgid(), "//u@169.254.1.1/S", "/Volumes/S", "s3cret pass")
    assert code == 0, output
    # The terminal stays root's: a user's process that could open it could read the password.
    assert handed == [("chown", 0, 0), ("chmod", 0o600)]
    assert seen.read_text() == "s3cret pass|//u@169.254.1.1/S|/Volumes/S\n"
    assert "s3cret" not in output and "<password>" not in output and "mounted" in output


def test_a_password_the_child_prints_is_scrubbed(monkeypatch, tmp_path):
    helper = load("tbshare-mount")
    fake_mount(monkeypatch, helper, tmp_path, "printf 'Password: ' > /dev/tty\nIFS= read -r pw\necho \"bad $pw\"\nexit 77\n")
    code, output = helper.mount_as(os.getuid(), os.getgid(), "//u@h/S", "/Volumes/S", "s3cret pass")
    assert code == 77 and "s3cret" not in output and "bad <password>" in output


def test_the_mount_url_carries_the_account_but_not_the_password(monkeypatch, capsys):
    helper = load("tbshare-mount")
    calls = []
    monkeypatch.setattr(helper.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "501")
    monkeypatch.setenv("SUDO_GID", "20")
    monkeypatch.setattr(helper.sys, "argv", ["tbshare-mount"])
    monkeypatch.setattr(helper, "load", lambda: ({"share": "My Share", "user": "a@b",
                                                  "interface": "bridge0", "mountpoint": "/Volumes/S"}, "pw:1"))
    monkeypatch.setattr(helper, "link_address", lambda interface: "fe80::bb")
    monkeypatch.setattr(helper, "prepare_mountpoint", lambda path, uid, gid: False)
    monkeypatch.setattr(helper, "mount_as", lambda *a: calls.append(a) or (0, ""))
    helper.main()
    assert calls == [(501, 20, "//a%40b@[fe80::bb%25bridge0]/My%20Share", "/Volumes/S", "pw:1")]


def test_a_mount_that_never_prompts_is_killed_at_the_deadline(monkeypatch, tmp_path):
    helper = load("tbshare-mount")
    fake = tmp_path / "mount_smbfs"
    # Ignores the hangup a closed terminal sends, so only the kill ends it.
    fake.write_text("#!/bin/sh\ntrap '' HUP\nsleep 5\n")
    fake.chmod(0o755)
    monkeypatch.setattr(helper, "MOUNT_SMBFS", str(fake))
    monkeypatch.setattr(helper, "MOUNT_DEADLINE", 1.0)
    monkeypatch.setattr(os, "setgroups", lambda groups: None)
    # Root-only steps; the tests run as the user the child runs as.
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)
    started = time.monotonic()
    code, _ = helper.mount_as(os.getuid(), os.getgid(), "//u@h/S", "/Volumes/S", "pw")
    assert code == -9 and time.monotonic() - started < 3


@pytest.mark.parametrize("path", ["/tmp/S", "/Volumes/a/b", "/Volumes/.."])
def test_the_mountpoint_must_sit_directly_under_volumes(path):
    helper = load("tbshare-mount")
    with pytest.raises(SystemExit):
        helper.prepare_mountpoint(path, os.getuid(), os.getgid())


def test_the_helper_takes_no_arguments(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPTS / "tbshare-mount"), "anything"], capture_output=True, text=True)
    assert result.returncode == 1 and "takes no arguments" in result.stderr


def test_the_link_is_the_interface_whose_counter_grew_by_the_read(monkeypatch, tmp_path):
    tb = load("tbshare")
    probe = tmp_path / "big"
    probe.write_bytes(b"\0" * (4 << 20))
    counts = iter([{"en0": 100, "bridge0": 0}, {"en0": 200, "bridge0": 4 << 20}])
    monkeypatch.setattr(tb, "interface_bytes", lambda: next(counts))
    result = tb.measure({"mountpoint": str(tmp_path), "interface": "bridge0"}, 4 << 20)
    assert result["link"] == "bridge0" and result["bytes"] == 4 << 20


def test_received_bytes_are_read_from_rows_with_and_without_a_hardware_address(monkeypatch):
    tb = load("tbshare")
    table = """Name       Mtu   Network       Address            Ipkts Ierrs     Ibytes    Opkts Oerrs     Obytes  Coll
lo0        16384 <Link#1>                      289326435     0 436838699682 289326436     0 436838701030     0
lo0        16384 127           127.0.0.1       289326435     - 436838699682 289326436     - 436838701030     -
bridge0    1500  <Link#20>   36:c5:e4:44:97:40  9409159     0  753408712 37263714     0 145724045216     0
bridge0    1500  169.254       169.254.156.85     9409159     -  753408712 37263714     - 145724045216     -
"""
    monkeypatch.setattr(tb.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=table))
    assert tb.interface_bytes() == {"lo0": 436838699682, "bridge0": 753408712}


def test_a_cached_read_is_retried_on_another_file(monkeypatch, tmp_path):
    tb = load("tbshare")
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"\0" * (4 << 20))
    # The first file was cached: no counter moved. The second came over bridge0.
    counts = iter([{"bridge0": 0}, {"bridge0": 0}, {"bridge0": 0}, {"bridge0": 4 << 20}])
    monkeypatch.setattr(tb, "interface_bytes", lambda: next(counts))
    monkeypatch.setattr(tb.random, "shuffle", lambda items: None)
    result = tb.measure({"mountpoint": str(tmp_path), "interface": "bridge0"}, 4 << 20)
    assert result["link"] == "bridge0" and result["file"].endswith("/b")


def test_an_empty_read_is_not_a_measurement(monkeypatch, tmp_path):
    tb = load("tbshare")
    (tmp_path / "a").write_bytes(b"\0" * (4 << 20))
    monkeypatch.setattr(tb, "timed_read", lambda path, nbytes, offset: (0, 0.1))
    monkeypatch.setattr(tb, "interface_bytes", lambda: {"en0": 0})
    with pytest.raises(tb.Problem) as e:
        tb.measure({"mountpoint": str(tmp_path), "interface": "bridge0"}, 4 << 20)
    assert e.value.code == 1


def test_reads_that_all_came_from_cache_are_inconclusive_not_a_wrong_link(monkeypatch, tmp_path):
    tb = load("tbshare")
    for name in ("a", "b", "c"):
        (tmp_path / name).write_bytes(b"\0" * (4 << 20))
    monkeypatch.setattr(tb, "interface_bytes", lambda: {"en0": 0, "bridge0": 0})
    with pytest.raises(tb.Problem) as e:
        tb.measure({"mountpoint": str(tmp_path), "interface": "bridge0"}, 4 << 20)
    assert e.value.code == 1 and "cache" in str(e.value)


@pytest.mark.parametrize("link, mbps, code", [("bridge0", 900, 0), ("bridge0", 499, 4), ("en0", 900, 4)])
def test_the_share_passes_only_over_the_link_at_speed(link, mbps, code):
    tb = load("tbshare")
    assert tb.verdict({"interface": "bridge0"}, {"link": link, "mbps": mbps})[0] == code


def test_a_stalled_read_gives_up_instead_of_hanging(monkeypatch, tmp_path):
    tb = load("tbshare")
    (tmp_path / "a").write_bytes(b"\0" * 16)
    monkeypatch.setattr(tb, "READ_DEADLINE", 1)
    monkeypatch.setattr(tb.os, "read", lambda fd, n: time.sleep(5) or b"")
    started = time.monotonic()
    with pytest.raises(tb.Problem) as e:
        tb.timed_read(str(tmp_path / "a"), 16)
    assert e.value.code == 1 and "no progress" in str(e.value) and time.monotonic() - started < 3


def test_the_probe_is_the_first_file_large_enough(tmp_path):
    tb = load("tbshare")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "small").write_bytes(b"x")
    (tmp_path / "b").mkdir()
    with open(tmp_path / "b" / "large", "wb") as f:
        f.truncate(2 << 20)
    (tmp_path / ".index").mkdir()
    with open(tmp_path / ".index" / "store", "wb") as f:
        f.truncate(2 << 20)
    assert list(tb.probe_files(str(tmp_path), 1 << 20)) == [str(tmp_path / "b" / "large")]


CONF = {"mountpoint": "/Volumes/S", "interface": "bridge0"}


def ensure_against(monkeypatch, tmp_path, *, mounted, status_code, link="en0", reaches=True, busy=(),
                   helper_code=0, helper_err="", allowed=True):
    tb = load("tbshare")
    calls = []
    monkeypatch.setattr(tb.os.path, "ismount", lambda p: mounted)
    monkeypatch.setattr(tb, "status", lambda conf: (status_code, "status", {"link": link}))
    monkeypatch.setattr(tb, "link_reaches", lambda conf: reaches)
    monkeypatch.setattr(tb, "open_files", lambda mp: list(busy))
    monkeypatch.setattr(tb, "HELPER", str(tmp_path / "helper"))
    (tmp_path / "helper").write_text("")

    rules = "User u may run the following commands on h:\n    (ALL) ALL\n" + (
        "    (root) NOPASSWD: {} \"\"\n".format(tmp_path / "helper") if allowed else "")

    def run(cmd, **_):
        if cmd[:3] == ["sudo", "-n", "-l"]:
            return types.SimpleNamespace(stdout=rules, stderr="", returncode=0)
        calls.append(cmd[0])
        code = helper_code if cmd[0] == "sudo" else 0
        return types.SimpleNamespace(stdout="mounted", stderr=helper_err, returncode=code)

    monkeypatch.setattr(tb.subprocess, "run", run)
    return tb, calls


def test_ensure_leaves_a_share_on_another_link_alone_while_files_are_open(monkeypatch, tmp_path):
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=4, busy=["python3"])
    with pytest.raises(tb.Problem) as e:
        tb.ensure(CONF)
    assert e.value.code == 5 and calls == []


def test_ensure_remounts_an_idle_share_on_another_link(monkeypatch, tmp_path):
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=4)
    assert tb.ensure(CONF) == (4, "status")
    assert calls == ["diskutil", "sudo"]


def test_ensure_leaves_a_slow_share_already_on_the_link_mounted(monkeypatch, tmp_path):
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=4, link="bridge0")
    assert tb.ensure(CONF) == (4, "status") and calls == []


def test_ensure_keeps_a_share_on_another_link_when_the_link_does_not_reach_the_server(monkeypatch, tmp_path):
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=4, reaches=False)
    with pytest.raises(tb.Problem) as e:
        tb.ensure(CONF)
    assert e.value.code == 4 and "cable" in str(e.value) and calls == []


def test_ensure_does_nothing_to_a_share_already_on_the_link(monkeypatch, tmp_path):
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=0)
    assert tb.ensure(CONF)[0] == 0 and calls == []


def test_ensure_without_the_helper_says_how_to_set_it_up(monkeypatch, tmp_path):
    tb, _ = ensure_against(monkeypatch, tmp_path, mounted=False, status_code=0, helper_code=1,
                           helper_err="sudo: a password is required", allowed=False)
    with pytest.raises(tb.Problem) as e:
        tb.ensure(CONF)
    assert e.value.code == 6 and "install-tbshare-helper.sh" in str(e.value)


def test_ensure_never_unmounts_a_share_the_helper_could_not_mount_back(monkeypatch, tmp_path):
    """Allowed only with a password ((ALL) ALL) is not allowed: agents cannot type it."""
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=4, allowed=False)
    with pytest.raises(tb.Problem) as e:
        tb.ensure(CONF)
    assert e.value.code == 6 and calls == []


def test_a_failed_remount_says_the_share_is_now_unmounted(monkeypatch, tmp_path):
    tb, calls = ensure_against(monkeypatch, tmp_path, mounted=True, status_code=4, helper_code=1,
                               helper_err="tbshare-mount: mount_smbfs exited 77: Authentication error")
    with pytest.raises(tb.Problem) as e:
        tb.ensure(CONF)
    assert e.value.code == 1 and "not mounted now" in str(e.value) and calls == ["diskutil", "sudo"]


def test_open_files_names_the_processes_holding_the_share(monkeypatch):
    tb = load("tbshare")
    lsof = "p123\ncpython3\nn/Volumes/S/a\np456\ncmlx_lm\nn/Volumes/S/b\np789\ncpython3\nn/Volumes/S/c\n"
    monkeypatch.setattr(tb.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=lsof))
    assert tb.open_files("/Volumes/S") == ["mlx_lm", "python3"]


@pytest.mark.parametrize("uid, mode, private, ok", [(0, 0o100644, False, True), (501, 0o100644, False, False),
                                                    (0, 0o100664, False, False), (0, 0o100600, True, True),
                                                    (0, 0o100640, True, False)])
def test_config_and_secret_must_be_root_owned_and_the_secret_root_only(monkeypatch, tmp_path, uid, mode, private, ok):
    helper = load("tbshare-mount")
    path = tmp_path / "f"
    path.write_text("x")
    monkeypatch.setattr(helper.os, "lstat", lambda p: types.SimpleNamespace(st_uid=uid, st_mode=mode))
    if ok:
        helper.root_owned(str(path), private=private)
    else:
        with pytest.raises(SystemExit):
            helper.root_owned(str(path), private=private)


def test_an_unreadable_probe_is_skipped(monkeypatch, tmp_path):
    tb = load("tbshare")
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"\0" * (4 << 20))
    (tmp_path / "a").chmod(0)
    # "a" fails to open after its first counter read; "b" is read between the next two.
    counts = iter([{"bridge0": 0}, {"bridge0": 0}, {"bridge0": 4 << 20}])
    monkeypatch.setattr(tb, "interface_bytes", lambda: next(counts))
    monkeypatch.setattr(tb.random, "shuffle", lambda items: None)
    result = tb.measure({"mountpoint": str(tmp_path), "interface": "bridge0"}, 4 << 20)
    assert result["file"].endswith("/b")


def test_a_slow_read_that_keeps_moving_is_timed_not_called_stalled(monkeypatch, tmp_path):
    tb = load("tbshare")
    (tmp_path / "a").write_bytes(b"\0" * (4 << 20))
    monkeypatch.setattr(tb, "READ_DEADLINE", 1)
    monkeypatch.setattr(tb, "CHUNK", 1 << 20)
    real_read = os.read
    monkeypatch.setattr(tb.os, "read", lambda fd, n: time.sleep(0.6) or real_read(fd, n))
    got, seconds = tb.timed_read(str(tmp_path / "a"), 4 << 20)
    assert got == 4 << 20 and seconds > 2
