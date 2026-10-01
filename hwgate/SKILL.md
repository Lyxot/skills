---
name: hwgate
description: Run GPU and heavy CPU work through the machine-wide hwgate scheduler so agents on this machine take turns and back off when the Mac is hot. Use before running anything that touches MLX or Metal (`*_metal.py` tests, training, inference, benchmarks) or loads many cores (`pytest -n`, builds, CPU-bound scripts). Also use when a job is stuck waiting for a slot, or to see who is holding the GPU.
---

# hwgate

Several agents share this Mac. `hwgate` admits heavy jobs only when the machine can take them, and cleans up after them.

## Usage

| Workload | Command |
|---|---|
| MLX or Metal: `*_metal.py` tests, training, inference, `mlx.launch` | `hwgate run --needs gpu -- <cmd>` |
| Many-core CPU work: `pytest -n`, builds, CPU-bound scripts | `hwgate run --needs cpu --cpus <n> -- <cmd>` |
| Parallel Metal tests | `hwgate run --needs gpu,cpu -- <cmd>` |
| Anything whose result is a speed (benchmarks, perf comparisons) | add `--stable` |

Not needed for git, reading files, `uv sync`, or single-process non-MLX tests.

Always pass `--max-wait` shorter than your own tool timeout:

```bash
hwgate run --needs gpu --max-wait 300 -- python -m pytest tests/test_x_metal.py -q
hwgate run --needs cpu --cpus 4 --max-wait 300 -- python -m pytest -n 4 tests/
```

## How admission works

- **Up to three jobs share each slot.** A fourth waits. A job that joins a busy slot prints the jobs already on it; quote that line if you report a duration from a shared run.
- **`--stable` runs alone.** It takes every slot and starts only after 60s at nominal heat. If it is throttled for longer than `--throttle-budget` (default 10s), it is killed with exit 76. On success it prints a `--stable run:` frequency summary; quote it next to benchmark numbers. Put it on the outermost `hwgate` call only, and never use it for correctness tests.
- **`--cpus <n>`** charges `n` of the core budget (the 12 performance cores, or `HWGATE_CPU_BUDGET`) and exports `HWGATE_CPUS` and `OMP_NUM_THREADS` and friends. Pass the same `n` to `pytest -n`, not `auto`. A job that declares nothing is charged a third of the budget.
- **There is no memory flag.** GPU memory here is wired and can't be swapped, so a job past it fails rather than slowing down. Size large models yourself, and use `--stable` to have the GPU to yourself.
- **Heat:** at fair, one heavy job at a time. At serious or worse, nothing new starts until 60s after it clears. Running jobs are never stopped for heat.
- **Timeout:** `--timeout` (default 7200s) kills the job's whole process group, as does the job exiting. `--timeout 0` disables the timeout and the cleanup.
- **Nesting:** a gated job that calls `hwgate` again reuses its own slots.

`hwgate status` shows the thermal state, the declared budget, and every job holding or waiting for a slot. `hwgate --help` has the details.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| the job's own | the job ran | read its output |
| 75 | `--max-wait` ran out; nothing started | check `hwgate status`, retry later |
| 124 | the job hit `--timeout` | it hung, or needs a longer `--timeout` |
| 2 | bad usage, `--cpus` over the budget, or nested `--stable` | fix the request |
| 76 | `--stable` job aborted for throttling | discard partial numbers and rerun later; it waits for the cooldown itself |

`hwgate:` lines on stderr say why a job is waiting and who holds the slot.

## Rules

- Never run gated work outside `hwgate` because the slot is busy. Wait or retry.
- Never under-declare `--cpus`; nothing enforces it, so a wrong number oversubscribes the machine for everyone.
- Never set `HWGATE_FORCE_THERMAL` or `HWGATE_STATE_DIR` outside hwgate's own tests.
- Don't wrap commands that leave a background process behind (`sh -c 'server &'`): it is killed when the job returns.
- Kill only your own job, by the pid from `hwgate status`. Don't delete lock files.
