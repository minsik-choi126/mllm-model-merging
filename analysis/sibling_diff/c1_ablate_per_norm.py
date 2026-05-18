"""C1 ablation — build three overlays, each killing ONLY one norm type's sink
amplifiers (top-K γ channels → layer mean). Lets us disentangle which sink
stage (input_layernorm / q_norm / k_norm) carries the IFEval-critical signal.

The combined-kill overlay (already built) caused IFEval -44.4 pt. Now compare:
  - kill only k_norm                          (head_dim, 128 ch)
  - kill only input_layernorm                 (hidden,   4096 ch)
  - kill only q_norm                          (head_dim, 128 ch)
"""

from __future__ import annotations
import json, shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

SRC = Path("/131_data/geeho/minsik/Qwen3-8B")
TPL_NOTHINK = Path("/131_data/geeho/minsik/Qwen3-8B-nothink")
TOP_K = 10

VARIANTS = {
    "knorm":  ["self_attn.k_norm"],
    "lnnorm": ["input_layernorm"],
    "qnorm":  ["self_attn.q_norm"],
}

def load_full() -> dict[str, torch.Tensor]:
    idx = json.load(open(SRC / "model.safetensors.index.json"))
    sd, shards = {}, {}
    for k, s in idx["weight_map"].items():
        shards.setdefault(s, []).append(k)
    for s, keys in shards.items():
        with safe_open(str(SRC / s), framework="pt") as f:
            for k in keys:
                sd[k] = f.get_tensor(k)
    return sd

def patch(sd: dict[str, torch.Tensor], norms: list[str]) -> int:
    n_changed = 0
    for li in range(36):
        for nm in norms:
            key = f"model.layers.{li}.{nm}.weight"
            if key not in sd:
                continue
            g = sd[key].to(torch.float32)
            mean = g.mean().item()
            _, idxs = torch.topk(g.abs(), TOP_K)
            new_g = g.clone()
            for j in idxs:
                new_g[j.item()] = mean
            sd[key] = new_g.to(sd[key].dtype)
            n_changed += 1
    return n_changed

def write_overlay(sd: dict[str, torch.Tensor], dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"})
    for name in ("config.json", "generation_config.json"):
        if (SRC / name).exists():
            shutil.copy2(SRC / name, dst / name)
    for name in ("tokenizer.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json"):
        src = SRC / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dst / name)
    # Patched (no-thinking) tokenizer_config.json
    shutil.copy2(TPL_NOTHINK / "tokenizer_config.json", dst / "tokenizer_config.json")

def main():
    print("loading Qwen3-8B...")
    sd_base = load_full()
    print(f"  {len(sd_base)} tensors loaded\n")

    for variant, norms in VARIANTS.items():
        sd = {k: v.clone() for k, v in sd_base.items()}
        n = patch(sd, norms)
        dst = Path(f"/131_data/geeho/minsik/Qwen3-8B-nosink-{variant}-nothink")
        write_overlay(sd, dst)
        print(f"[{variant}] patched {n} {norms[0]} tensors → {dst}")

    print("\nDone. Run IFEval on each overlay path separately.")

if __name__ == "__main__":
    main()
