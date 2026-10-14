#!/usr/bin/env python3
"""Strip stray whitespace from Hugging Face cache ref files (older runs of the downloader wrote a trailing newline).

usage: repair_cache_refs.py [--cache-dir DIR] [--dry-run]
Only ref files whose stripped content is a 40-character commit with a matching snapshot folder are rewritten.
"""
import argparse, os, re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cache-dir", default = os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
parser.add_argument("--dry-run", action = "store_true")
args = parser.parse_args()
fixed = skipped = 0
for ref in sorted(Path(args.cache_dir).resolve().glob("*--*/refs/**/*")):
    if not ref.is_file():
        continue
    text = ref.read_text()
    commit = text.strip()
    if text == commit:
        continue
    repo = ref.parent
    while repo.name != "refs":
        repo = repo.parent
    repo = repo.parent
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not (repo / "snapshots" / commit).is_dir():
        print(f"skip   {repo.name} {ref.name}: {text!r}")
        skipped += 1
        continue
    if not args.dry_run:
        ref.write_text(commit)
    print(f"{'would fix' if args.dry_run else 'fixed'}  {repo.name} {ref.name}")
    fixed += 1
print(f"{fixed} ref(s) {'to fix' if args.dry_run else 'fixed'}, {skipped} skipped")
