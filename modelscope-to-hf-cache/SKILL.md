---
name: modelscope-to-hf-cache
description: Download a model from ModelScope into the native Hugging Face Hub cache layout while preserving an exact Hugging Face revision. Use when ModelScope is the requested transfer source but tools must later load the model from the Hugging Face cache. Do not use for ordinary Hugging Face-only downloads or arbitrary local model directories.
---

# ModelScope to Hugging Face Cache

Use the bundled downloader when the same repository ID exists on ModelScope and Hugging Face:

```bash
scripts/download_modelscope_to_hf_cache.sh --dry-run ORG/REPO
scripts/download_modelscope_to_hf_cache.sh ORG/REPO
```

The script recursively downloads the ModelScope revision, stores content-addressed blobs, creates a ModelScope snapshot referenced by `refs/modelscope`, builds the exact current Hugging Face snapshot, sets `refs/main` to the Hugging Face commit, and runs strict `hf cache verify`. It reuses identical files and fetches only content differences from the Hugging Face endpoint.

## Workflow

1. Resolve the intended repository ID. Do not silently substitute a similarly named quantization or owner.
2. Run `--dry-run` and compare the reported download size with current free space. Keep the default 20 GiB reserve unless the user explicitly chooses another value.
3. Run the downloader. For several repositories, process them sequentially or in small batches after accounting for their combined space and network load.
4. Confirm `refs/main` and `refs/modelscope` still match the live commits, no `.incomplete` files remain, and strict verification succeeded.
5. Report repository names, cache sizes, sources, verification status, and remaining disk space.

Useful options:

```bash
scripts/download_modelscope_to_hf_cache.sh \
  --cache-dir /path/to/huggingface/hub \
  --hf-endpoint https://hf.rimuru.work \
  --reserve-gib 20 \
  ORG/REPO
```

`--hf-endpoint` selects the Hugging Face-compatible endpoint used for authoritative metadata differences and verification; ModelScope remains the primary source.

If the same repository ID does not exist on Hugging Face, stop and ask whether the user wants an HF-like ModelScope-only cache or a mapped Hugging Face repository. Those outcomes cannot provide the same `refs/main` and strict Hub verification guarantees.

Do not remove existing snapshots, cache repositories, or unrelated incomplete downloads unless the user explicitly requests cleanup. For cache invariants and troubleshooting, read [references/cache-layout.md](references/cache-layout.md).
