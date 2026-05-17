"""C1 — Sufficiency test: kill sink in clean Qwen3-8B → does IFEval crash?

If sink encoding is causally necessary for instruction-following, then
*deliberately* destroying it in a clean base LLM should produce an IFEval
crash. This is the cleanest sufficiency proof.

Method:
- Load Qwen3-8B base.
- For each layer, identify top-K channels of input_layernorm γ (the sink amplifier
  channels, e.g. ch.923 with γ=28). Same for k_norm γ.
- Replace those γ values with the *layer mean* (kills amplification, keeps
  channel existing).
- Save as new model + symlink tokenizer from Qwen3-8B-nothink (thinking off).

Output: /131_data/geeho/minsik/Qwen3-8B-nosink-nothink/
"""

from __future__ import annotations
import json, os, shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

SRC = Path("/131_data/geeho/minsik/Qwen3-8B")
TPL_NOTHINK = Path("/131_data/geeho/minsik/Qwen3-8B-nothink")  # patched tokenizer
DST = Path("/131_data/geeho/minsik/Qwen3-8B-nosink-nothink")
TOP_K_PER_LAYER = 10  # kill top-10 amplifier channels per layer per norm

def main():
    DST.mkdir(parents=True, exist_ok=True)

    # Load all tensors from sharded source into a single dict
    idx = json.load(open(SRC / "model.safetensors.index.json"))
    sd: dict[str, torch.Tensor] = {}
    shards: dict[str, list[str]] = {}
    for k, s in idx["weight_map"].items():
        shards.setdefault(s, []).append(k)
    for s, keys in shards.items():
        with safe_open(str(SRC / s), framework="pt") as f:
            for k in keys:
                sd[k] = f.get_tensor(k)
    print(f"loaded {len(sd)} tensors")

    # Identify and patch γ amplifiers
    modified = []
    for li in range(36):
        for norm_key in (f"model.layers.{li}.input_layernorm.weight",
                         f"model.layers.{li}.self_attn.q_norm.weight",
                         f"model.layers.{li}.self_attn.k_norm.weight"):
            if norm_key not in sd:
                continue
            g = sd[norm_key].to(torch.float32)
            mean = g.mean().item()
            # top-K by absolute value (we want big amplifiers, almost all positive but handle sign)
            _, idxs = torch.topk(g.abs(), TOP_K_PER_LAYER)
            new_g = g.clone()
            for j in idxs:
                new_g[j.item()] = mean
            sd[norm_key] = new_g.to(sd[norm_key].dtype)
            modified.append((norm_key, idxs.tolist(), [g[j].item() for j in idxs], mean))
    print(f"patched {len(modified)} γ tensors")
    print("sample modifications:")
    for k, ix, og, mn in modified[:6]:
        print(f"  {k}: replaced ch {ix} (γ={[f'{v:.2f}' for v in og]}) → {mn:.3f}")

    # Save as single safetensors
    out_st = DST / "model.safetensors"
    save_file(sd, str(out_st), metadata={"format": "pt"})
    print(f"wrote {out_st}")

    # Copy config & generation_config from SRC; tokenizer (with patched chat_template) from TPL_NOTHINK
    for name in ("config.json", "generation_config.json"):
        if (SRC / name).exists():
            shutil.copy2(SRC / name, DST / name)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json"):
        src_path = TPL_NOTHINK / name if (TPL_NOTHINK / name).exists() and not (TPL_NOTHINK / name).is_symlink() \
                   else SRC / name
        if src_path.exists() and src_path.is_file():
            shutil.copy2(src_path, DST / name)
    # tokenizer_config.json must be the PATCHED one (no thinking)
    if (TPL_NOTHINK / "tokenizer_config.json").exists():
        # That file is a real file (not symlink) in nothink overlay
        shutil.copy2(TPL_NOTHINK / "tokenizer_config.json", DST / "tokenizer_config.json")

    print(f"\noverlay ready at: {DST}")
    print(f"summary: killed top-{TOP_K_PER_LAYER} γ channels in input_layernorm + q_norm + k_norm of all layers.")


if __name__ == "__main__":
    main()
