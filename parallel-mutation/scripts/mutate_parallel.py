#!/usr/bin/env python3
"""Run mutation checks in parallel, each mutant in a worker's own copy of the repository.

Two modes:

  run     hand-written mutations from a file: each must make its named test fail.
  weaken  mechanical one-decision rewrites of source files: report the ones no test notices.

The source tree is only ever read. Every copy must pass the suite unmutated and must catch a
sentinel mutant before any verdict is reported.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import os
import pathlib
import queue
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

COLLECTION = "<the suite did not run>"
# Put at the top of the file, not the bottom: a script ending in sys.exit(main()) never
# reaches a trailing statement, so an appended sentinel passes and the check that the copy
# is the file under test silently proves nothing.
SENTINEL = 'raise RuntimeError("mutation sentinel: this copy must fail")\n'
DEFAULT_EXCLUDES = (".git", "__pycache__", ".venv", ".pytest_cache", "node_modules")


# ---------------------------------------------------------------------------------------------
# Running the suite in a copy


def failures(tree: pathlib.Path, test: list[str], timeout: float) -> set[str]:
    """Names of the tests that failed in ``tree``; COLLECTION if the suite did not run."""
    try:
        run = subprocess.run(
            test,
            cwd=tree,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(tree), os.environ.get("PYTHONPATH")]))},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"<timed out>"}
    out = run.stdout + run.stderr
    failed = {
        line.split(" ")[1].split(" - ")[0].split("[")[0].split("::")[-1]
        for line in out.splitlines()
        if line.startswith(("FAILED ", "ERROR "))
    }
    # A mutant that stops the suite from collecting fails nothing by name; calling that
    # "nothing failed" would report it as a survivor.
    if run.returncode not in (0, 1) or (run.returncode == 1 and not failed):
        failed.add(COLLECTION)
    return failed


class Pool:
    """``workers`` copies of ``root``, each checked before use, handed out one job at a time."""

    def __init__(self, root, test, workers, workdir, excludes, timeout, pristine=None):
        self.root, self.test, self.timeout = root, test, timeout
        self.workdir = pathlib.Path(tempfile.mkdtemp(prefix="pool-", dir=workdir))
        self.trees = []
        excl = [arg for e in excludes for arg in ("--exclude", e)]
        for i in range(workers):
            tree = workdir / f"copy-{i}"
            subprocess.run(["rsync", "-a", "--delete", *excl, f"{root}/", f"{tree}/"], check=True)
            for rel, text in (pristine or {}).items():
                (tree / rel).write_text(text)
            self.trees.append(tree)
        self.free: queue.Queue[pathlib.Path] = queue.Queue()
        for tree in self.trees:
            self.free.put(tree)
        self.executor = concurrent.futures.ThreadPoolExecutor(workers)

    def in_tree(self, tree, rel, source):
        path = tree / rel
        original = path.read_text()
        try:
            path.write_text(source)
            return failures(tree, self.test, self.timeout)
        finally:
            path.write_text(original)

    def check(self, sentinel_rel: str) -> None:
        base = list(self.executor.map(lambda t: failures(t, self.test, self.timeout), self.trees))
        if any(base):
            raise SystemExit(f"a copy fails before any mutation, so no verdict would mean anything: {sorted(next(b for b in base if b))}")
        text = (self.trees[0] / sentinel_rel).read_text()
        # Every copy, not merely every slot: a copy that imported the source tree instead of
        # itself would pass here only if it were never asked.
        caught = list(self.executor.map(lambda t: self.in_tree(t, sentinel_rel, SENTINEL + text), self.trees))
        if not all(caught):
            raise SystemExit(
                f"a copy did not fail with {sentinel_rel} broken: the suite does not import it from the copy "
                "(an editable install shadowing PYTHONPATH?) or never imports it at all"
            )

    def run(self, jobs):
        """Yield (key, failed tests) for each (key, relative path, mutated source), as they finish."""

        def one(key, rel, source):
            tree = self.free.get()
            try:
                return key, self.in_tree(tree, rel, source)
            finally:
                self.free.put(tree)

        futures = [self.executor.submit(one, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()

    def close(self):
        self.executor.shutdown(wait=True)
        shutil.rmtree(self.workdir, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Hand-written mutations


def load_mutations(path: pathlib.Path) -> list[tuple[str, str, str, str, str]]:
    """``MUTATIONS`` from a Python file: (label, relative path, old, new, test that must fail)."""
    sys.path.insert(0, str(path.parent))
    return list(runpy.run_path(str(path), run_name="mutations")["MUTATIONS"])


def cmd_run(args, pool_args) -> int:
    root = pool_args["root"]
    mutations = load_mutations(args.mutations)
    bad, jobs = 0, []
    for label, rel, old, new, target in mutations:
        text = (root / rel).read_text()
        if text.count(old) != 1:
            print(f"SKIP  {label}: anchor appears {text.count(old)} times in {rel}", flush=True)
            bad += 1
            continue
        jobs.append(((label, target), rel, text.replace(old, new)))
    if not jobs:
        print("nothing to run")
        return 1
    pool = Pool(**pool_args)
    try:
        pool.check(jobs[0][1])
        verdicts = dict(pool.run(jobs))
    finally:
        pool.close()
    for label, rel, old, new, target in mutations:
        if (label, target) not in verdicts:
            continue
        failed = verdicts[(label, target)]
        hit = target in failed
        bad += not hit
        others = sorted(failed - {target})
        print(
            f"{'OK  ' if hit else 'MISS'}  {label}\n        caught by {target}: {hit}"
            + (f"; also failed: {', '.join(others)}" if others else ""),
            flush=True,
        )
    print(f"\n{len(mutations) - bad}/{len(mutations)} mutations caught by their test")
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------
# Mechanical weakening

_CMP = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}


def variants(tree: ast.AST):
    """Every one-node weakening of ``tree``, as (node, what it does, how to apply it)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
            swap = _CMP[type(node.ops[0])]
            yield node, f"{type(node.ops[0]).__name__} -> {swap.__name__}", lambda n, s=swap: setattr(n, "ops", [s()])
        elif isinstance(node, ast.BoolOp):
            other = ast.Or if isinstance(node.op, ast.And) else ast.And
            yield node, f"{type(node.op).__name__} -> {other.__name__}", lambda n, o=other: setattr(n, "op", o())
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield node, "drop not", None
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield node, f"{node.value} -> {not node.value}", lambda n: setattr(n, "value", not n.value)
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1):
            yield node, f"{node.value} -> {1 - node.value}", lambda n: setattr(n, "value", 1 - n.value)


def weakened(source: str, index: int) -> tuple[str, str]:
    """``source`` with its ``index``-th weakening applied, and a description of it."""
    tree = ast.parse(source)
    for i, (node, what, apply) in enumerate(variants(tree)):
        if i != index:
            continue
        if apply is None:  # dropping `not` puts the operand where the negation was
            for parent in ast.walk(tree):
                for name, value in ast.iter_fields(parent):
                    if value is node:
                        setattr(parent, name, node.operand)
                    elif isinstance(value, list):
                        value[:] = [node.operand if v is node else v for v in value]
        else:
            apply(node)
        return ast.unparse(tree), f"line {node.lineno}: {what}"
    raise IndexError(index)


def cmd_weaken(args, pool_args) -> int:
    root = pool_args["root"]
    survivors = 0
    for rel in args.files:
        # Mutants are cut from the unparsed file (unparsing drops comments and reformats), so
        # every copy starts from it too and line numbers refer to it; see --show-base.
        base = ast.unparse(ast.parse((root / rel).read_text()))
        total = sum(1 for _ in variants(ast.parse(base)))
        jobs = []
        for index in range(total):
            source, what = weakened(base, index)
            if source != base:
                jobs.append(((index, what), rel, source))
        print(f"{rel}: {len(jobs)} weakenings", flush=True)
        pool = Pool(**pool_args, pristine={rel: base})
        try:
            pool.check(rel)
            verdicts = dict(pool.run(jobs))
        finally:
            pool.close()
        for (index, what), failed in sorted(verdicts.items()):
            if failed:
                print(f"caught    {rel} {what} ({len(failed)} tests)", flush=True)
            else:
                survivors += 1
                print(f"SURVIVED  {rel} {what}", flush=True)
    print(f"\n{survivors} survivors")
    return 1 if survivors else 0


def cmd_show_base(args) -> int:
    base = ast.unparse(ast.parse((args.root / args.file).read_text())).splitlines()
    lo, hi = args.lines or (1, len(base))
    for n in range(max(lo, 1), min(hi, len(base)) + 1):
        print(f"{n:>5}  {base[n - 1]}")
    return 0


# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(q):
        q.add_argument("--root", type=pathlib.Path, required=True, help="repository to copy; never written")
        q.add_argument("--test", required=True, help='suite command run inside each copy, e.g. "python -m pytest tests/test_x.py -q -p no:cacheprovider"')
        q.add_argument("--workers", type=int, default=8)
        q.add_argument("--timeout", type=float, default=900, help="seconds per suite run")
        q.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDES), help="path pattern not copied (repeatable)")

    r = sub.add_parser("run", help="hand-written mutations, each with the test that must catch it")
    common(r)
    r.add_argument("--mutations", type=pathlib.Path, required=True, help="Python file defining MUTATIONS")
    w = sub.add_parser("weaken", help="one-decision rewrites of source files; report the ones no test notices")
    common(w)
    w.add_argument("files", nargs="+", help="source files relative to --root")
    s = sub.add_parser("show-base", help="print the unparsed file that weaken's line numbers refer to")
    s.add_argument("--root", type=pathlib.Path, required=True)
    s.add_argument("file")
    s.add_argument("--lines", type=int, nargs=2, metavar=("FROM", "TO"))
    args = p.parse_args(argv)

    if args.mode == "show-base":
        return cmd_show_base(args)
    root = args.root.resolve()
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mutation-"))
    pool_args = dict(
        root=root, test=shlex.split(args.test), workers=args.workers,
        workdir=workdir, excludes=args.exclude, timeout=args.timeout,
    )
    start = time.monotonic()
    try:
        return (cmd_run if args.mode == "run" else cmd_weaken)(args, pool_args)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"wall {time.monotonic() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
