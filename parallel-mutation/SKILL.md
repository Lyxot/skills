---
name: parallel-mutation
description: Mutation-check a Python test suite in parallel without ever writing the source tree — each mutant runs in a worker's own copy of the repository. Use when added or changed tests need to be shown to actually catch the defect they guard, when asked whether a suite really protects some code, before asking for review of a change with new tests, or when a sequential mutation loop is too slow. Also covers the mechanical "weaken" sweep that finds decisions no test depends on.
---

# parallel-mutation

A mutation check asks whether a test would notice if the code it guards were wrong: break the code on purpose, run the suite, and the test must fail. `scripts/mutate_parallel.py` runs many such checks at once, each in a separate copy of the repository, and never edits the repository itself.

Why copies: a runner that writes mutants into the real tree and restores them afterwards cannot be parallelized — two workers restore each other's mutants as if they were the original — and an interrupted run leaves a mutant in the source. With copies, a run can be killed at any time and the tree cannot be damaged.

## Two modes

| Mode | Question it answers | Input |
|---|---|---|
| `run` | Does each guarantee I stated have a test that catches its violation? | a file of hand-written mutations, each naming the test that must fail |
| `weaken` | Which decisions in this file can change without any test noticing? | source files; mutants are generated (comparison flips, `and`/`or`, dropped `not`, `True`/`False`, `0`/`1`) |

Use both. `run` only covers guarantees someone thought to state; `weaken` needs nothing from you and finds the ones nobody stated. Both are cheap compared with a reviewer finding the same gaps, so run them before asking for review.

## Running it

```bash
S=<this skill's directory>/scripts/mutate_parallel.py

python $S run \
    --root <repository or worktree> \
    --test "python -m pytest tests/test_x.py -q -p no:cacheprovider" \
    --mutations path/to/mutations.py

python $S weaken \
    --root <repository or worktree> \
    --test "python -m pytest tests/test_x.py -q -p no:cacheprovider" \
    pkg/module_a.py pkg/module_b.py
```

- Use the Python interpreter of the project's environment, both to launch the script and inside `--test`.
- `--test` runs inside each copy with `PYTHONPATH` set to that copy, so the copy's package shadows an editable install. `run` mode needs pytest with its default short summary (`FAILED`/`ERROR` lines) to know which test failed; `weaken` works with any command that exits non-zero on failure. Pass only the test files that matter: every mutant runs the whole command.
- `--workers` defaults to 8. CPU-bound suites scale close to the core count; suites that share one GPU scale less (8 workers gave about 4× on a GPU-bound suite). If the project has a time budget for validation, trim `--test` to the relevant files before adding workers.
- `--exclude` leaves paths out of the copies (`.git`, `.venv`, caches and `node_modules` by default). Build products the suite imports, such as a compiled extension inside the package, must stay in.
- A run is heavy: `--workers` suites at once. If the machine has a shared scheduler or other convention for heavy jobs, run it through that. Never run it at the same time as a benchmark on the same machine.

Before any verdict counts, the runner proves the copies are sound, and stops if either check fails:

1. every copy passes the suite unmutated — a red baseline makes every mutant look caught;
2. every copy fails when a sentinel error is appended to a mutated file — otherwise the suite is importing that file from somewhere else (usually an editable install of the original checkout), and every mutant would look survived.

## Writing hand-written mutations

A mutations file is Python defining `MUTATIONS`, a list of `(label, path relative to --root, old text, new text, name of the test that must fail)`:

```python
MUTATIONS = [
    (
        "an expired token is still accepted",
        "auth/tokens.py",
        "if token.expires_at <= now:\n        raise Expired(token)",
        "if token.expires_at < now:\n        raise Expired(token)",
        "test_a_token_expires_at_its_deadline",
    ),
]
```

- **The old text must appear exactly once** in its file; a mutation whose anchor is missing or ambiguous is reported `SKIP` and counts as a failure. Include enough surrounding lines to make it unique.
- **Use the weakest edit that breaks the guarantee**, not the most destructive one. Deleting a whole function fails every test and proves nothing about the one you named.
- **Write the mutation before the test.** If you cannot state the edit that breaks it, you do not yet know what the test is for.
- Name the test that owns the guarantee. Other tests failing too is fine and is shown as `also failed`; the named one not failing is a `MISS`.
- Keep the mutations file outside the repository's shipped code, wherever the project keeps scratch or analysis files.

## Reading the output

| Line | Meaning | What to do |
|---|---|---|
| `OK` | the named test failed | nothing |
| `MISS` | the named test passed with the code broken | the test does not guard that guarantee — strengthen it, or name the test that does |
| `SKIP` | the anchor is not in the file exactly once | fix the mutation; the code moved |
| `caught` | some test failed on a generated weakening | nothing |
| `SURVIVED` | no test noticed a generated weakening | triage it (below) |
| `<the suite did not run>` among failures | the mutant broke collection or import | counts as caught; if every mutant shows it, the `--test` command is wrong |

`weaken` line numbers refer to the file as the runner parsed and re-printed it (comments dropped). See them with `python $S show-base --root <root> <file> --lines 120 160`.

Triage every survivor into exactly one of:

- **Real gap** — a decision that changes what a caller sees, with no test: add a test or a `run` mutation for it.
- **Dead decision** — a default nothing passes, a branch nothing reaches: prefer deleting the decision over testing it.
- **Equivalent** — the flip reaches the same result another way (a fast path, a message nobody reads, a value computed beside a refusal and never used): record why and leave it. Once triaged, these are the accepted residue; only a survivor outside that list is news.

Do not write a test for every survivor. A test pinned to an implementation detail is not a guarantee.

## Pitfalls that produce wrong verdicts

- **A baseline taken through the code under test agrees with the defect.** If a test snapshots state with the same function it is testing, a mutant that corrupts that function corrupts the snapshot identically and survives. Construct the expected value, or assert the invariant against the object itself.
- **One-element fixtures cannot tell "first" from "all".** A set, list or mapping in a fixture needs at least two members, arranged so a walk that stops early misses the one that matters.
- **A tool that reports nothing may have run nothing.** Check that the number of verdicts matches the number of mutations; this runner prints the counts and fails on a mutant it could not generate.
- **Do not import a runner to read its mutation list.** Keep mutations in their own file, as above.
