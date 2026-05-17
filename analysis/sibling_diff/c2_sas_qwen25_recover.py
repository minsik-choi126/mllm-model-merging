"""C2 — Necessity test: SAS-restore sink columns of Qwen2.5-VL-LM from base Qwen2.5
LLM, then measure IFEval recovery.

Hypothesis: IFEval drop in VLM is *caused by* corruption of sink-encoding weights.
If we restore JUST the sink-relevant columns of W_q/W_k/W_v from the base LLM
(leaving the rest of the VLM weights as-is), IFEval should recover toward the
base LLM number while vision-side weights remain mostly intact.

Identification of sink-relevant columns per layer:
  sink_cols_l = top-K hidden-dim channels by input_layernorm γ at layer l.
  (input_ln γ is mostly preserved by VL training so we use VLM's own γ.)

Restoration:
  For sub in {q_proj, k_proj, v_proj}:
      W_VLM[l, sub][:, sink_cols_l]  ←  W_LLM[l, sub][:, sink_cols_l]

The result keeps ~ (1 - K/hidden) fraction of VLM's projection weights but
restores the sink-relevant input-channel routing.

Output: /131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen25vl_7b_lm_sas/
"""

from __future__ import annotations
import json, os, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SRC_LLM = Path("/131_data/geeho/minsik/Qwen2.5-7B-Instruct")
SRC_VLM = Path("/131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen25vl_7b_lm")
DST     = Path("/131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen25vl_7b_lm_sas")
NUM_LAYERS = 28
TOP_K_PER_LAYER = 36  # ~ top 1% of hidden_size=3584; reasonable amplifier band


def load_all_sharded(path: Path) -> dict[str, torch.Tensor]:
    idxf = path / "model.safetensors.index.json"
    if idxf.exists():
        idx = json.load(open(idxf))
        sd = {}
        shards = {}
        for k, s in idx["weight_map"].items():
            shards.setdefault(s, []).append(k)
        for s, keys in shards.items():
            with safe_open(str(path / s), framework="pt") as f:
                for k in keys:
                    sd[k] = f.get_tensor(k)
        return sd
    p = path / "model.safetensors"
    if p.exists():
        out = {}
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    raise FileNotFoundError(path)


def main():
    DST.mkdir(parents=True, exist_ok=True)

    print("loading base LLM (Qwen2.5-7B-Instruct)...")
    sd_llm = load_all_sharded(SRC_LLM)
    print(f"  {len(sd_llm)} tensors")
    print("loading VLM-LM (extracted)...")
    sd_vlm = load_all_sharded(SRC_VLM)
    print(f"  {len(sd_vlm)} tensors")

    sd_out = {k: v.clone() for k, v in sd_vlm.items()}

    restored_pairs = []
    for li in range(NUM_LAYERS):
        ln_key = f"model.layers.{li}.input_layernorm.weight"
        g = sd_llm[ln_key].to(torch.float32)
        # top-K by γ magnitude (sink amplifier channels)
        _, sink_cols = torch.topk(g.abs(), TOP_K_PER_LAYER)
        sink_cols = sink_cols.sort().values  # sorted indices for readability

        for sub in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight"):
            key = f"model.layers.{li}.{sub}"
            W_llm = sd_llm[key]
            W_vlm = sd_vlm[key]
            # restore those columns (input_dim axis = last)
            W_new = W_vlm.clone()
            W_new[:, sink_cols] = W_llm[:, sink_cols].to(W_vlm.dtype)
            sd_out[key] = W_new
            restored_pairs.append((li, sub, len(sink_cols)))

    print(f"\nrestored sink columns in {len(restored_pairs)} (layer, sub) pairs")
    print(f"per layer: top-{TOP_K_PER_LAYER} of hidden_size=3584  "
          f"({100*TOP_K_PER_LAYER/3584:.2f}% of input dim)")
    print(f"sink-col examples (layer 0): top-{TOP_K_PER_LAYER} by input_ln γ:")
    g0 = sd_llm["model.layers.0.input_layernorm.weight"].to(torch.float32)
    _, cols0 = torch.topk(g0.abs(), TOP_K_PER_LAYER)
    cols0 = cols0.sort().values
    print(f"  {cols0.tolist()[:20]}...")
    print(f"  their γ values:  {[f'{g0[c].item():.3f}' for c in cols0.tolist()[:10]]}")

    out_st = DST / "model.safetensors"
    save_file(sd_out, str(out_st), metadata={"format": "pt"})
    print(f"\nwrote {out_st}")

    # Copy config + tokenizer from the extracted VLM-LM (already has correct config)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "special_tokens_map.json",
                 "added_tokens.json", "chat_template.jinja",
                 "generation_config.json"):
        sp = SRC_VLM / name
        if sp.exists() and sp.is_file():
            shutil.copy2(sp, DST / name)
    print(f"overlay ready at: {DST}")


if __name__ == "__main__":
    main()
