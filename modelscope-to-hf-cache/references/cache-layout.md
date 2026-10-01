# Cache invariants and troubleshooting

## Resulting layout

For `ORG/REPO`, the cache directory is `models--ORG--REPO`:

```text
models--ORG--REPO/
├── blobs/
├── refs/
│   ├── main          # current Hugging Face commit
│   └── modelscope    # downloaded ModelScope commit
└── snapshots/
    ├── <hf-commit>/
    └── <modelscope-commit>/
```

Snapshot entries are relative symlinks into `blobs/`. Hugging Face LFS blob names are SHA-256 values; ordinary files use Git blob IDs. ModelScope-only metadata is retained only in its source snapshot. A file whose contents are identical on both services reuses one blob.

## Important distinctions

- ModelScope normally names its default branch `master`; Hugging Face normally exposes `main`. Their commit IDs are not interchangeable.
- `refs/main` must contain the Hugging Face commit when Hugging Face tools are expected to resolve the cached model normally.
- ModelScope may add `configuration.json` or use a different `.gitattributes`. Do not place those differences inside the Hugging Face snapshot.
- Enumerate ModelScope tree entries recursively. Root-only API results can omit assets or nested tokenizer files.
- Compute ordinary Git blobs with `git hash-object --no-filters`; global line-ending filters must not change the object ID.
- Preserve an empty LFS field with a sentinel such as `-` when parsing tab-separated metadata in shell.

## Verification

Use the commit stored in `refs/main`, not a branch name that may change during validation:

```bash
hf cache verify ORG/REPO \
  --revision "$(cat models--ORG--REPO/refs/main)" \
  --cache-dir /path/to/huggingface/hub \
  --fail-on-missing-files \
  --fail-on-extra-files
```

Also compare both refs to live metadata and inspect only the target repository for `.incomplete` files. Do not use broad cache deletion commands to repair a single repository. Resume its content-addressed staging file or rebuild only the affected snapshot links.
