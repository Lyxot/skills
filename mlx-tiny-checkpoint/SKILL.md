---
name: mlx-tiny-checkpoint
description: Answer "does unsloth-zoo support <architecture>", "will this train", or "can we test this without the download" by materializing a shrunken but otherwise real checkpoint and running the true load and training path, then checking the things that actually discriminate — which adapters were wrapped, which ones moved, whether the tensor names match the real repo. Use whenever a checkpoint is too large for the machine, mlx-vlm just added a family, or a load/training failure needs reproducing. Also use before concluding an architecture works from reading a config or building a model in memory: that never reaches from_pretrained, which is where loader and patch-selection bugs live. And use it before believing a green training run — a tiny model's loss curve looks identical whether or not the vision tower, the adapters, or the images were involved at all.
---

# MLX tiny checkpoint

The technique is easy to reinvent; the traps are not. Most of this file is about
which evidence to trust, because the default evidence — a training run that
completes with a falling loss — is nearly content-free on a random-init model.

## Settle whether it fits before doing any work

Free, from `config.json` and `du`: params ≈ `vocab*hidden*2 + layers*(attention +
MLP)`, and for MoE multiply the expert MLP by `num_experts`, which usually
dominates everything else. Two bytes per param at bf16, plus optimizer and
activation headroom. If the real model fits in RAM and is already in the cache,
say so and use it — this skill is for when it does not.

## Two tiers

**Tier 1, build from config.** `mlx_vlm`/`mlx_lm` construct the whole module tree
from `config.json` alone. Shrink the config and build to answer: does the
architecture exist in the installed backend, what modules does it hold, does
patch detection fire on it.

**Tier 2, materialize a checkpoint.** Write the shrunken config, random weights,
and a real tokenizer to a directory, then run `FastMLXModel.from_pretrained` and
`MLXTrainer` against it. Tier 1 answers none of this: routing decisions,
allow-lists keyed on `model_type`, processor plumbing, and MRoPE position
handling all live behind the loader. An architecture that builds fine in tier 1
still fails at load with `ValueError: Model type <arch> not supported` — that has
happened, which is why tier 2 exists.

```bash
S=~/.agents/skills/mlx-tiny-checkpoint/scripts

# 1. Build only, no disk written. --repo pulls config, tokenizer and index from
#    Hugging Face or ModelScope -- metadata only, never weights. Refuses a
#    full-size config rather than being killed allocating the real tree.
python $S/materialize_tiny_checkpoint.py --repo org/name --out /tmp/t \
    --overrides shrink.json --build-only

# 2. Materialize. A ModelScope URL works in place of org/name; --config and
#    --tokenizer still take local paths when the repo is already on disk.
python $S/materialize_tiny_checkpoint.py --repo org/name --out /tmp/tiny \
    --overrides shrink.json

# 3. Train through the real path. Run every mode that applies.
python $S/smoke_train.py --checkpoint /tmp/tiny --mode text
python $S/smoke_train.py --checkpoint /tmp/tiny --mode text-only   # Studio's text-dataset path
python $S/smoke_train.py --checkpoint /tmp/tiny --mode image       # if it has a vision tower
```

Writing `shrink.json` is the only per-architecture work: see
`references/shrinking.md`. A shrink that flattens the structure under test
answers a question about a different model — if the claim concerns a hybrid
attention interleave, keep enough layers for the pattern to repeat.

## Evidence worth reporting

**Which adapters were wrapped, and which moved.** `smoke_train.py` prints both
per group and fails when they disagree with the mode. Targeting flags like
`finetune_vision_layers` match module *names*: a tower named unusually is skipped
in silence, with no error and a perfect-looking loss curve. Two separate
architectures have shipped that way. If vision adapters were wrapped but none
moved, the image never reached the loss and the decoder trained alone.

**Tensor names against the real repo.** `--repo` fetches
`model.safetensors.index.json`, and the materializer then reports how many of
the built model's name patterns appear in the real checkpoint. Random weights
are named by construction, so this is the only check that catches a wrong
backend or a missing HF→MLX remap. Expect a few misses for real fusions —
mlx_lm folds per-expert MLPs into `switch_mlp`, so those names exist nowhere in
the published checkpoint. A near-total mismatch is the signal: it usually means
the checkpoint was built by mlx_vlm (which prefixes the decoder
`language_model.`) for a model the loader will open with mlx_lm, and the loader
reports that as an architecture-support gap rather than a naming one.

**The loss is the weakest signal, so read it last.** On a tiny random-init model
unrelated variants agree to four decimals; CE versus CCE, patched versus not, all
look the same. `ln(vocab_size)` is a floor only for text the model cannot predict
from repetition, which is why the harness trains on diverse sentences — a
repeated phrase scores *below* the floor legitimately and looks exactly like the
real failure it is meant to catch (an embedding narrower than a token id the
processor emits, gathering past the table and returning a number anyway).

**Vocab has to clear the vision token ids.** GLM's image token sits above its
tokenizer's vocab. The script sizes the embedding from both, so do not override
`vocab_size` yourself. Likewise keep the published token ids whenever a real
processor is in play: the processor emits the real id and the image splice
matches on exactly that value.

**Mutation-check the conclusion.** After a fix, remove it and confirm the harness
reproduces the original error verbatim. A harness that passes both ways proves
nothing. The same move tests a routing hypothesis without editing the repo:
monkeypatch the allow-list in-process, run, and treat before/after as the check.

**Run from a directory that is not the workspace parent.** From `~/Github/unsloth`,
`import unsloth` resolves to the directory as a namespace package and shadows the
real one. `/tmp` is fine.

## Constraints

Never download weights — that is the point, and the fetcher enforces it by
name and by size rather than trusting the caller. Metadata is different: config,
tokenizer, chat template and the shard index are small, and they are exactly
what the load path reads, so `--repo` pulls them without asking. They land in
`~/.cache/mlx-tiny-checkpoint/`, deliberately not the HF cache, where a snapshot
holding no weights would look like a real checkpoint to anything else.

A same-family tokenizer already on disk also works via `--tokenizer`, since only
the vocabulary and chat template matter — but prefer the real one when the repo
is reachable. Families that pad `vocab_size` above their tokenizer, or put
control ids above every ordinary token, are common enough that guessing costs
more than fetching.

Keep the checkpoints out of the workspace tree. `/tmp` or a
`tempfile.TemporaryDirectory` is right; they are disposable.
