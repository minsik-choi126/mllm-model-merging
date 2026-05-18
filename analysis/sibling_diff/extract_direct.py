"""Direct (bypass HF loader) text-backbone extraction for VLMs that use
trust_remote_code custom classes (InternVL, Phi-Vision, etc.).

Reads safetensors directly, applies the same key normalization as our standard
loader (model.language_model.* → model.*, language_model.model.* → model.*, etc.),
saves as a single safetensors with the right tokenizer/config.
"""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TEXT_KEY_PREFIXES = (
    "model.embed_tokens.",
    "model.layers.",
    "model.norm.",
    "model.rotary_emb.",
    "lm_head.",
)


def is_normalized_text_key(key: str) -> bool:
    return key.startswith(TEXT_KEY_PREFIXES)


def normalize(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith("language_model.model."):
        return "model." + key[len("language_model.model."):]
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key[len("language_model.lm_head."):]
    if key == "language_model.output.weight":
        return "lm_head.weight"
    if is_normalized_text_key(key):
        return key
    return None


def collect_weight_map(vlm_dir: Path) -> dict[str, str]:
    """Return {key: full_shard_path}."""
    idx_file = vlm_dir / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.load(open(idx_file))
        return {k: str(vlm_dir / s) for k, s in idx["weight_map"].items()}
    single = vlm_dir / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt") as f:
            return {k: str(single) for k in f.keys()}
    raise FileNotFoundError(vlm_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, help="VLM model dir")
    p.add_argument("--output", required=True, help="output dir for extracted text backbone")
    p.add_argument("--tokenizer-src", required=True,
                   help="dir to copy tokenizer + config from (= base LLM dir, "
                        "or a no-thinking overlay dir for Qwen3)")
    p.add_argument("--config-src", default=None,
                   help="if set, copy config.json from here instead of tokenizer-src "
                        "(use VLM's own llm_config for non-standard IS)")
    args = p.parse_args()

    vlm = Path(args.vlm)
    dst = Path(args.output)
    tok_src = Path(args.tokenizer_src)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"vlm:        {vlm}")
    print(f"tokenizer:  {tok_src}")
    print(f"output:     {dst}")

    # 1. Collect all VLM keys, map to text-side, group by shard
    vlm_map = collect_weight_map(vlm)
    text_keys = []
    for k in vlm_map:
        if normalize(k) is not None:
            text_keys.append(k)
    print(f"  vlm total keys: {len(vlm_map)}, text-side keys: {len(text_keys)}")

    by_shard: dict[str, list[str]] = {}
    for k in text_keys:
        by_shard.setdefault(vlm_map[k], []).append(k)

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            for k in keys:
                sd[normalize(k)] = f.get_tensor(k)
    print(f"  extracted {len(sd)} tensors")

    # 2. Save weights as single safetensors
    out_st = dst / "model.safetensors"
    save_file(sd, str(out_st), metadata={"format": "pt"})
    print(f"  wrote {out_st}")

    # 3. Copy tokenizer files from tok_src
    for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                 "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json",
                 "chat_template.jinja"):
        sp = tok_src / name
        if sp.exists() and sp.is_file() and not sp.is_symlink():
            shutil.copy2(sp, dst / name)
        elif sp.exists() and sp.is_symlink() and sp.resolve().is_file():
            shutil.copy2(sp.resolve(), dst / name)

    # 4. Copy config.json (from --config-src if set, else from tokenizer-src)
    cfg_src = Path(args.config_src) if args.config_src else tok_src
    cfg_path = cfg_src / "config.json"
    if cfg_path.exists():
        if cfg_path.is_symlink():
            shutil.copy2(cfg_path.resolve(), dst / "config.json")
        else:
            shutil.copy2(cfg_path, dst / "config.json")
    # generation_config
    gen_path = tok_src / "generation_config.json"
    if gen_path.exists():
        target = gen_path.resolve() if gen_path.is_symlink() else gen_path
        if target.is_file():
            shutil.copy2(target, dst / "generation_config.json")

    print(f"\nDone: {dst}")


if __name__ == "__main__":
    main()
