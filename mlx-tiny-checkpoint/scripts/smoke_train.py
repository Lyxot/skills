#!/usr/bin/env python3
"""LoRA-train a materialized checkpoint through the real unsloth-zoo path.

Exercises FastMLXModel.from_pretrained -> get_peft_model -> MLXTrainer, which is
where loader and patch-selection bugs live. A config-only build never reaches it.

The loss is the weakest evidence here and is reported last on purpose: on a tiny
random-init model the logits are near zero, so the loss sits at the uniform floor
whatever the model does, and unrelated variants agree to four decimals. What
actually discriminates is which adapters were wrapped and which ones moved.

  smoke_train.py --checkpoint DIR [--mode text|text-only|image] [--steps 4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Diverse rows, ragged on purpose: ragged so a batch cannot read the shape the
# previous one left, diverse because ln(vocab) only bounds the loss for text the
# model cannot predict from repetition. A repeated phrase scores below the floor
# on random weights and looks exactly like an out-of-range embedding gather.
_SENTENCES = [
    "The harbour master logged three arrivals before the fog closed in.",
    "Sediment cores from the delta record an abrupt change in rainfall.",
    "She rewrote the proof to avoid the axiom of choice entirely.",
    "Copper prices fell after the smelter announced an unplanned outage.",
    "The cartilage samples were fixed, sectioned, and stained overnight.",
    "Traffic on the eastern viaduct is restricted to a single lane.",
    "His grandmother kept the recipe written on the back of a ticket stub.",
    "Radio interference peaked shortly after the satellite passed overhead.",
]


def text_rows(n):
    return [{"text": " ".join(_SENTENCES[(i + k) % len(_SENTENCES)]
                              for k in range(1 + i % 4))} for i in range(n)]


def image_rows(n):
    from PIL import Image

    img = Image.new("RGB", (112, 112), (128, 64, 32))
    return [{
        "images": [img],
        "messages": [
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": _SENTENCES[i % 8]}]},
            {"role": "assistant", "content": [{"type": "text",
                                               "text": _SENTENCES[(i + 3) % 8]}]},
        ],
    } for i in range(n)]


def _group(name: str) -> str:
    lowered = name.lower()
    if any(k in lowered for k in ("merger", "projector", "connector", "aligner")):
        return "projector"
    if any(k in lowered for k in ("vision", "visual", "vit", "image_tower")):
        return "vision"
    return "language"


def _lora_params(model):
    """The LoRA B leaves. Only B starts at zero, so only B stays put when no gradient
    reaches it: A drifts under weight decay whether or not the image ever met the loss,
    and counting it hides a vision tower that trained nothing."""
    from mlx.utils import tree_flatten

    return {name: value for name, value in tree_flatten(model.trainable_parameters())
            if "lora_b" in name.lower()}


def run(checkpoint: Path, mode: str, steps: int, use_cce: bool, seq_len: int):
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    load_kwargs = {"text_only": True} if mode == "text-only" else {}
    if os.environ.get("SWEEP_TRUST_REMOTE_CODE") == "1":
        load_kwargs["trust_remote_code"] = True
    model, tokenizer = FastMLXModel.from_pretrained(
        str(checkpoint), max_seq_length=seq_len, load_in_4bit=False, **load_kwargs)
    peft_kwargs = {"finetune_vision_layers": True} if mode == "image" else {}
    model = FastMLXModel.get_peft_model(
        model, r=4, lora_alpha=8, max_seq_length=seq_len,
        use_gradient_checkpointing="mlx", **peft_kwargs)

    before = {n: mx.array(v) for n, v in _lora_params(model).items()}
    rows = image_rows(2 * steps) if mode == "image" else text_rows(2 * steps)
    trainer_kwargs = {"processor": tokenizer} if mode == "image" else {}
    with tempfile.TemporaryDirectory() as out:
        trainer = MLXTrainer(
            model=model, tokenizer=tokenizer, train_dataset=rows, **trainer_kwargs,
            args=MLXTrainingConfig(
                max_steps=steps, per_device_train_batch_size=1,
                gradient_accumulation_steps=1, learning_rate=1e-3, logging_steps=1,
                max_seq_length=seq_len, seed=3407, output_dir=out, use_cce=use_cce,
                compile=False, gradient_checkpointing=True, report_to="none"),
        )
        trainer.train()

    after = _lora_params(model)
    wrapped, moved = {}, {}
    for name, value in after.items():
        g = _group(name)
        wrapped[g] = wrapped.get(g, 0) + 1
        if name in before and float(mx.abs(value - before[name]).max()) > 0:
            moved[g] = moved.get(g, 0) + 1
    losses = [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]
    return losses, wrapped, moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="text", choices=("text", "text-only", "image"),
                    help="text: default load path; text-only: text_only=True, the path "
                         "Studio takes for a text dataset; image: multimodal with images")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    cfg = json.loads((checkpoint / "config.json").read_text())
    vocab = (cfg.get("text_config") or cfg).get("vocab_size") or cfg.get("vocab_size")

    results = {}
    for use_cce in (False, True):
        losses, wrapped, moved = run(checkpoint, args.mode, args.steps, use_cce, args.seq_len)
        if not losses or any(l != l for l in losses):
            sys.exit(f"cce={use_cce}: NaN or missing loss: {losses}")
        results[use_cce] = losses
        print(f"  cce={str(use_cce):5s} losses={losses}", flush=True)

    groups = sorted(set(wrapped) | set(moved))
    print("\nLoRA wrapped: " + ", ".join(f"{g}={wrapped.get(g, 0)}" for g in groups))
    print("LoRA moved:   " + ", ".join(f"{g}={moved.get(g, 0)}" for g in groups))

    # The real evidence. A targeting flag that matches no module name is silently
    # ignored -- no error, no warning, and a loss curve that looks perfect.
    if not moved.get("language"):
        sys.exit("no language adapter changed: training did not reach the decoder")
    if args.mode == "image" and not wrapped.get("vision"):
        sys.exit("finetune_vision_layers wrapped no vision module. The flag matches "
                 "module names, so a tower named differently is skipped in silence. "
                 "Re-run with explicit target_modules covering this tower's linears "
                 "before believing any vision result.")
    if args.mode == "image" and not moved.get("vision"):
        sys.exit("vision adapters were wrapped but none changed: the image is not "
                 "reaching the loss, so the decoder trained alone")

    floor = math.log(vocab)
    first = results[False][0]
    print(f"\nln(vocab_size={vocab}) = {floor:.2f}; first loss {first:.2f}")
    if first < floor - 0.5:
        sys.exit(f"loss {first:.2f} is well under the uniform floor {floor:.2f}. On "
                 f"diverse text that means the run is not measuring what it appears "
                 f"to -- check the embedding covers every id the processor emits.")
    if first < floor:
        print("note: marginally under the floor, which repetitive text can do legitimately")
    print(f"CE vs CCE max divergence: "
          f"{max(abs(a - b) for a, b in zip(results[False], results[True])):.4f}")
    print("OK")


main()
