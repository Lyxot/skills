#!/usr/bin/env python3
"""Materialize a shrunken but otherwise real MLX checkpoint on disk.

Writes config.json, randomly initialized safetensors weights, and a real
tokenizer/processor, so code under test takes the same load path a downloaded
checkpoint would. Weights are never downloaded; --repo fetches only the
metadata files, which are small and are what the load path actually reads.

  materialize_tiny_checkpoint.py --repo <org/name|hub URL> --out DIR \
      [--overrides shrink.json] [--build-only]
  materialize_tiny_checkpoint.py --config <config.json|repo-dir> --out DIR \
      --tokenizer <dir|hf-cache-repo-dir> [--overrides shrink.json] [--build-only]
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

# Ids the processor emits verbatim. The embedding has to be wider than the
# largest of them or the image splice gathers out of range and the run "passes"
# while reading whatever lies past the table.
TOKEN_ID_KEYS = (
    "image_token_id", "image_token_index", "video_token_id", "video_token_index",
    "audio_token_id", "audio_token_index", "vision_start_token_id",
    "vision_end_token_id", "image_start_token_id", "image_end_token_id",
)
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
    "processor_config.json", "image_preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer.jsonl", "vocab.txt", "spiece.model",
    "sentencepiece.bpe.model",
)
# Repos whose tokenizer or processor is remote code name it in an auto_map and ship the
# module beside the config. Without it the tiny checkpoint cannot be opened at all, which
# reads as "family unsupported" when nothing about the family was tested.
TOKENIZER_SUFFIXES = (".py", ".tiktoken")


METADATA_FILES = ("config.json", "generation_config.json", "configuration.json",
                  "preprocessor_config.json") + TOKENIZER_FILES
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
                   ".onnx", ".msgpack", ".h5", ".npz", ".mlx")
METADATA_CACHE = Path.home() / ".cache" / "mlx-tiny-checkpoint"
MAX_METADATA_BYTES = 128 << 20  # a tokenizer.json runs ~20MB; a shard never fits


def is_metadata(name: str) -> bool:
    """Metadata only. The size cap and the suffix list are the same rule twice:
    a weight shard must not be reachable through a filename that looks benign."""
    if "/" in name:
        return False
    if name.endswith(".index.json"):  # names the shards; contains none of them
        return True
    if name.endswith(WEIGHT_SUFFIXES):
        return False
    return name in METADATA_FILES or name.endswith(TOKENIZER_SUFFIXES)


def fetch_metadata(repo: str, revision: str | None) -> Path:
    """Download a repo's metadata files and return the directory holding them.

    Kept out of the HF cache on purpose: a snapshot with no weights in it looks
    like a real checkpoint to anything else that goes looking.
    """
    url = re.match(r"https?://(?:www\.)?(modelscope\.cn|huggingface\.co)/(?:models/)?"
                   r"([^/]+/[^/?#]+)", repo)
    if url:
        source, repo = ("modelscope" if "modelscope" in url.group(1) else "hf"), url.group(2)
    else:
        source = None
    repo = repo.strip("/")

    errors = []
    for candidate in ([source] if source else ["hf", "modelscope"]):
        try:
            names = (_hf_list(repo, revision) if candidate == "hf"
                     else _modelscope_list(repo, revision))
        except Exception as e:
            errors.append(f"{candidate}: {type(e).__name__}: {e}")
            continue
        wanted = sorted(n for n in names if is_metadata(n))
        if not any(n.startswith("config") for n in wanted):
            errors.append(f"{candidate}: no config.json among {len(names)} files")
            continue
        # Named after the source that answered, so the same repo reached by URL
        # and by org/name shares one directory instead of caching twice.
        out = METADATA_CACHE / candidate / repo.replace("/", "--")
        out.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            target = out / name
            if target.exists():
                continue
            if candidate == "hf":
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(repo, name, revision=revision)
                shutil.copy2(src, target)
            else:
                _download(f"https://modelscope.cn/models/{repo}/resolve/"
                          f"{revision or 'master'}/{name}", target)
        print(f"metadata from {candidate}:{repo} -> {out}\n  {', '.join(wanted)}")
        return out
    sys.exit("could not fetch metadata:\n  " + "\n  ".join(errors))


def _hf_list(repo: str, revision):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo, revision=revision)


def _modelscope_list(repo: str, revision):
    with urllib.request.urlopen(
            f"https://modelscope.cn/api/v1/models/{repo}/repo/files"
            f"?Revision={revision or 'master'}", timeout=60) as r:
        payload = json.load(r)
    return [f["Path"] for f in payload.get("Data", {}).get("Files", [])]


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as r:
        size = int(r.headers.get("Content-Length") or 0)
        if size > MAX_METADATA_BYTES:
            sys.exit(f"{target.name} is {size / 1e6:.0f}MB; metadata files are small, "
                     f"so this is probably weights. Refusing.")
        data = r.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES:
        sys.exit(f"{target.name} exceeds the metadata size cap. Refusing.")
    target.write_bytes(data)


def resolve_snapshot(path: Path) -> Path:
    """Accept a plain directory or a `models--org--repo` HF cache entry."""
    if (path / "snapshots").is_dir():
        snapshots = sorted((path / "snapshots").iterdir())
        if not snapshots:
            sys.exit(f"no snapshot under {path}")
        return snapshots[-1]
    return path


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Dot-path assignment: {"text_config.num_hidden_layers": 4}."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        target = cfg
        for key in parents:
            if not isinstance(target.get(key), dict):
                sys.exit(f"override path {dotted!r} does not exist in the config")
            target = target[key]
        # A new leaf is fine -- shrinking often has to state a field the
        # published config leaves implicit. A wrong parent is a typo, and a
        # typo that silently did nothing would leave the checkpoint full size.
        target[leaf] = value
    return cfg


def estimated_params(cfg: dict) -> float:
    """Rough parameter count from the config, before anything is allocated.

    Construction allocates the whole random parameter tree, so a full-size
    config does not fail gracefully -- the process is killed. Estimating first
    turns that into a message telling you to shrink.
    """
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    g = lambda *keys: next((text[k] for k in keys if isinstance(text.get(k), int)), 0)
    hidden, layers = g("hidden_size"), g("num_hidden_layers", "depth")
    inter = g("moe_intermediate_size", "intermediate_size") or hidden
    experts = g("n_routed_experts", "num_experts", "num_local_experts") or 1
    return g("vocab_size") * hidden + layers * (4 * hidden ** 2 + 3 * hidden * inter * experts)


def remapped(package: str, arch: str) -> str:
    """The module a backend builds `model_type` with: both keep a MODEL_REMAPPING
    (`phi4-siglip` -> `phi4_siglip`), and importing the raw name instead reports an
    implemented family as missing."""
    try:
        table = importlib.import_module(f"{package}.utils").MODEL_REMAPPING
    except Exception:
        table = {}
    default = arch.lower() if package == "mlx_vlm" else arch
    return table.get(arch.lower(), table.get(arch, default))


def _import_arch(package: str, arch: str):
    module = f"{package}.models.{remapped(package, arch)}"
    try:
        return importlib.import_module(module), module
    except ModuleNotFoundError as e:
        if e.name != module:  # the family exists; one of its own imports is missing
            raise
        return None, module


def _build_lm(cfg: dict, arch: str):
    pkg, module = _import_arch("mlx_lm", arch)
    if pkg is None:
        return None
    return pkg.Model(pkg.ModelArgs.from_dict(cfg)), module


def _build_vlm(cfg: dict, arch: str):
    pkg, module = _import_arch("mlx_vlm", arch)
    if pkg is None:
        return None
    # The same steps mlx_vlm.utils.load_model takes, so a config that builds here is one
    # the loader builds too: defaulted sub-configs, then every module config converted.
    from mlx_vlm.utils import update_module_configs
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("text_config", cfg.pop("llm_config", {}))
    cfg.setdefault("vision_config", {})
    cfg.setdefault("audio_config", {})
    model_cfg = update_module_configs(
        pkg.ModelConfig.from_dict(cfg), pkg, cfg,
        ["text", "vision", "perceiver", "projector", "audio"])
    return pkg.Model(model_cfg), module


def build_model(cfg: dict):
    """Instantiate from config alone. Raises if the architecture cannot build.

    Backend choice decides the tensor names: mlx_vlm wraps the decoder under a
    `language_model.` prefix, mlx_lm does not. Writing a text-only checkpoint
    with mlx_vlm's names makes the loader reject every weight and report the
    mismatch as an architecture-support gap, which is a lie about the model.
    """
    arch = cfg.get("model_type")
    if not arch:
        sys.exit("config.json has no model_type")
    multimodal = any(isinstance(cfg.get(k), dict)
                     for k in ("vision_config", "audio_config"))
    order = (_build_vlm, _build_lm) if multimodal else (_build_lm, _build_vlm)
    for build in order:
        built = build(cfg, arch)
        if built is not None:
            return built
    sys.exit(f"neither mlx_lm nor mlx_vlm has a module for model_type {arch!r} "
             f"(looked for {remapped('mlx_lm', arch)!r} and {remapped('mlx_vlm', arch)!r})")


def randomize_zero_parameters(model, std: float) -> list[str]:
    """Give every all-zero float parameter small random values.

    Constructors zero-initialise learned gates and residual scales (cross-attention gates,
    per-layer lambdas, some embeddings). A trained checkpoint has real values there; a
    random one keeps the zeros, which cuts the signal path: logits stop depending on the
    input and adapters behind the gate get no gradient. That fails the load probe and the
    adapter-movement check for a reason no real checkpoint has.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    changed = []
    for name, value in tree_flatten(model.parameters()):
        if (mx.issubdtype(value.dtype, mx.floating) and value.size
                and not mx.any(value != 0).item()):
            changed.append((name, (mx.random.normal(value.shape) * std).astype(value.dtype)))
    if changed:
        model.update(tree_unflatten(changed))
        mx.eval(model.parameters())
    return [name for name, _ in changed]


def report_name_overlap(names, index_path: Path) -> None:
    """Compare built tensor names against the real checkpoint's index.

    Random weights are named by construction, so this is the one check that can
    still catch a wrong backend or a missing HF->MLX remap. Layer indices differ
    after shrinking, so compare the shapes of the names rather than the names.
    """
    try:
        real = json.loads(index_path.read_text()).get("weight_map") or {}
    except Exception:
        return
    strip = lambda s: re.sub(r"\.\d+\.", ".N.", s)
    built, published = {strip(n) for n in names}, {strip(n) for n in real}
    missing = sorted(built - published)
    print(f"  name overlap with {index_path.name}: "
          f"{len(built & published)}/{len(built)} patterns matched")
    if missing:
        print(f"  not in the published checkpoint: {', '.join(missing[:4])}"
              f"{' ...' if len(missing) > 4 else ''}")


def vocab_floor(tokenizer_dir: Path, cfg: dict) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    floor = max(len(tok), max(tok.get_vocab().values()) + 1)
    # Published vocab_size is often padded above the tokenizer; keeping the
    # padding costs one embedding row per token and keeps the shrink faithful.
    # eos may be a list, and control ids can sit above every ordinary token.
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    for holder in (cfg, text):
        floor = max(floor, holder.get("vocab_size") or 0)
        for key in TOKEN_ID_KEYS + ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = holder.get(key)
            for i in (value if isinstance(value, list) else [value]):
                if isinstance(i, int):
                    floor = max(floor, i + 1)
    return floor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",
                    help="org/name or a Hugging Face / ModelScope model URL. Fetches "
                         "config, tokenizer and index metadata -- never weights -- and "
                         "uses it for both --config and --tokenizer")
    ap.add_argument("--revision", help="branch, tag or commit for --repo")
    ap.add_argument("--config",
                    help="config.json, or a directory/HF cache entry holding one")
    ap.add_argument("--out", required=True, help="checkpoint directory to write")
    ap.add_argument("--tokenizer",
                    help="local directory or HF cache entry to copy tokenizer/processor from")
    ap.add_argument("--overrides", help="JSON file of dot-path config overrides")
    ap.add_argument("--max-params", type=float, default=1e9,
                    help="refuse to build above this estimated parameter count")
    ap.add_argument("--allow-large", action="store_true",
                    help="build anyway; expect the process to be killed on a real config")
    ap.add_argument("--build-only", action="store_true",
                    help="build from config and report; write nothing")
    ap.add_argument("--zero-init-std", type=float, default=0.2,
                    help="std of the values written over all-zero float parameters; "
                         "0 keeps the constructor's zeros")
    ap.add_argument("--seed", type=int, default=0,
                    help="random-weight seed; fixed so a rebuild reproduces the same checkpoint")
    args = ap.parse_args()

    import mlx.core as mx
    mx.random.seed(args.seed)

    fetched = None
    if args.repo:
        fetched = fetch_metadata(args.repo, args.revision)
        args.config = args.config or str(fetched / "config.json")
        args.tokenizer = args.tokenizer or str(fetched)
    elif not args.config:
        ap.error("pass --repo, or --config for a local config.json")

    config_path = Path(args.config).expanduser()
    if config_path.is_dir():
        config_path = resolve_snapshot(config_path) / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg.pop("quantization_config", None)  # random weights are not quantized
    if args.overrides:
        cfg = apply_overrides(cfg, json.loads(Path(args.overrides).read_text()))

    out = Path(args.out).expanduser()
    if not args.build_only:
        if not args.tokenizer:
            sys.exit("--tokenizer is required unless --build-only")
        out.mkdir(parents=True, exist_ok=True)
        src = resolve_snapshot(Path(args.tokenizer).expanduser())
        copied = [f for f in TOKENIZER_FILES if (src / f).exists()]
        copied += sorted(f.name for f in src.iterdir()
                         if f.is_file() and f.name.endswith(TOKENIZER_SUFFIXES))
        if not any(f.startswith("tokenizer") for f in copied):
            sys.exit(f"no tokenizer files under {src}")
        for name in copied:
            shutil.copy2(src / name, out / name)

        vocab = vocab_floor(out, cfg)
        for holder in (cfg, cfg.get("text_config")):
            if isinstance(holder, dict) and "vocab_size" in holder:
                holder["vocab_size"] = vocab
        cfg.setdefault("vocab_size", vocab)

    estimate = estimated_params(cfg)
    if estimate > args.max_params and not args.allow_large:
        sys.exit(f"config would build ~{estimate / 1e9:.1f}B parameters. Shrink it with "
                 f"--overrides (layer counts, hidden sizes, expert counts) -- construction "
                 f"allocates the whole random tree and the process gets killed. "
                 f"--allow-large overrides this.")

    model, module = build_model(cfg)

    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.eval(model.parameters())
    rezeroed = randomize_zero_parameters(model, args.zero_init_std) if args.zero_init_std else []
    if rezeroed:
        print(f"  randomized {len(rezeroed)} all-zero parameters: {', '.join(rezeroed[:4])}"
              f"{' ...' if len(rezeroed) > 4 else ''}")
    weights = dict(tree_flatten(model.parameters()))
    index = next((f for d in (fetched, config_path.parent) if d
                  for f in sorted(d.glob("*.index.json"))), None)
    if args.build_only:
        print(f"built {module}: {len(weights)} tensors, "
              f"{sum(w.size for w in weights.values()) / 1e6:.1f}M params")
        if index:
            report_name_overlap(weights, index)
        return

    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    mx.save_safetensors(str(out / "model.safetensors"), weights, {"format": "mlx"})
    print(f"materialized {module} -> {out}\n"
          f"  {len(weights)} tensors, {sum(w.size for w in weights.values()) / 1e6:.1f}M params, "
          f"vocab={cfg.get('vocab_size')}\n"
          f"  tokenizer files: {', '.join(copied)}")
    if index:
        report_name_overlap(weights, index)


main()
