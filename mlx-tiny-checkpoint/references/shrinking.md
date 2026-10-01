# Writing `shrink.json`

Overrides are dot-paths into `config.json`, applied before the model is built:

```json
{
  "text_config.hidden_size": 256,
  "text_config.num_hidden_layers": 6,
  "text_config.layer_types": ["linear_attention", "full_attention"],
  "vision_config.depth": 2
}
```

A path whose *parent* is missing is a typo and errors — a silently ignored override would leave the checkpoint full size. A new leaf is allowed, because shrinking often has to state a field the published config leaves implicit.

## What to shrink

Aim for tens of millions of parameters. The script estimates before allocating and refuses above 1B, so start large and follow the number it reports.

- **Depth and width**: `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`. Vision towers use `depth` and their own `hidden_size`.
- **Experts** dominate a MoE checkpoint: `n_routed_experts` / `num_experts` and `moe_intermediate_size`. Cutting 512 experts to 8 does more than every other field combined.
- **`max_position_embeddings`** down to a few thousand, or a rope table can outweigh the model.

## What must stay consistent

The config is a set of interlocking shapes, and the failure mode is a construction error rather than a wrong answer — so read the error and fix the relationship it names.

- **Per-layer lists must match the new depth.** `layer_types`, `mlp_layer_types`, `indexer_types` and their kin are one entry per layer. Keep the *mix*: if the real model interleaves linear attention with full attention every fourth layer, keep at least one of each, or the shrunken model never exercises the path under test.
- **Index lists into layers** — `kda_layers`, `full_attn_layers`, `ple_layer_ids`, `deepstack_visual_indexes` — must stay inside the new range and agree with `layer_types`.
- **Head dims** must divide as the real config does. Fused kernels often want a multiple of 32; a head dim of 128 is a safe default when the original is larger.
- **The vision tower's output must match the decoder's input**: `vision_config.out_hidden_size` equals `text_config.hidden_size`.
- **Leave `vocab_size` alone.** The materializer sets it from the tokenizer and the vision token ids.
- **Leave the token ids alone** when a real processor will run. Rewriting `image_token_id` to a small number is only safe for `--build-only`.

## Deriving one for a new architecture

Read the published `config.json` and the backend's model file side by side — `mlx_vlm/models/<arch>/` — and shrink only the fields the constructor reads. Fields the constructor never touches are noise in the shrink file. When a construction error names a shape, it names the relationship you broke; fix that one field rather than guessing.
