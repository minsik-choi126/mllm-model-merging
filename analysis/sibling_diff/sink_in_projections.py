"""Check whether Qwen2.5 (which has NO q_norm/k_norm) encodes attention-sink
amplification directly in the W_k / W_q projection weights, and how VL training
perturbs that encoding vs Qwen3's structurally separated γ encoding.

For each model and each layer, compute the L2 norm of every output row of W_k
(i.e. the gain applied to each K head_dim channel by the projection).
Heavy-tailed per-row-norm = "this channel will be amplified after projection"
= functional analog of γ_k in q_norm/k_norm models.

Then compare:
  - row-norm distribution (heavy-tailed?)
  - VL-induced shift in amplifier rows vs normal rows
  - persistence of amplifier rows across layers
"""

from __future__ import annotations
import csv, json, os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


PAIRS = {
    "qwen25": {
        "llm": "/131_data/geeho/minsik/Qwen2.5-7B-Instruct",
        "vlm_lm": "/131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen25vl_7b_lm",
        "num_layers": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
    "qwen3": {
        "llm": "/131_data/geeho/minsik/Qwen3-8B",
        "vlm_lm": "/131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen3vl_8b_lm",
        "num_layers": 36,
        "num_kv_heads": 8,
        "head_dim": 128,
    },
}

OUT_DIR = Path(__file__).resolve().parent


def _resolve(path):
    p = Path(path)
    if p.is_file():
        with safe_open(str(p), framework="pt") as f:
            return {k: str(p) for k in f.keys()}
    idx_file = p / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.load(open(idx_file))
        return {k: str(p / s) for k, s in idx["weight_map"].items()}
    single = p / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt") as f:
            return {k: str(single) for k in f.keys()}
    raise FileNotFoundError(path)


def _load(path, key):
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(torch.float32)


def per_output_row_norm(W):
    """W has shape (out, in). Return per-row L2 norm."""
    return W.norm(dim=1).numpy()


def main():
    rows = []
    summary_lines = []

    for name, conf in PAIRS.items():
        llm_map = _resolve(conf["llm"])
        vlm_map = _resolve(conf["vlm_lm"])
        L = conf["num_layers"]
        kvH = conf["num_kv_heads"]
        d = conf["head_dim"]
        print(f"\n=== {name}: L={L}, kv_heads={kvH}, head_dim={d} ===")

        # W_k shape: (kvH * d, hidden_size) = (out, in)
        # rows are arranged as: head_0 dims 0..d-1, head_1 dims 0..d-1, ...
        # We'll fold to (kvH, d) and average across heads to get a per-channel signal.
        # But also keep raw per-row for amplifier identification.
        for li in range(L):
            key = f"model.layers.{li}.self_attn.k_proj.weight"
            W_llm = _load(llm_map[key], key)
            W_vlm = _load(vlm_map[key], key)
            rn_llm = per_output_row_norm(W_llm)  # shape (kvH*d,)
            rn_vlm = per_output_row_norm(W_vlm)
            assert rn_llm.shape[0] == kvH * d, f"shape mismatch: {rn_llm.shape} vs {kvH*d}"
            # average across heads per head_dim channel (gives canonical channel signal)
            rn_llm_h = rn_llm.reshape(kvH, d)
            rn_vlm_h = rn_vlm.reshape(kvH, d)
            # combine: for each (head, channel) we have a per-output-row norm
            for h in range(kvH):
                for ch in range(d):
                    rows.append({
                        "pair": name,
                        "layer_idx": li,
                        "head": h,
                        "channel": ch,
                        "row_norm_llm": float(rn_llm_h[h, ch]),
                        "row_norm_vlm": float(rn_vlm_h[h, ch]),
                        "abs_delta": float(abs(rn_vlm_h[h, ch] - rn_llm_h[h, ch])),
                        "rel_delta": float(abs(rn_vlm_h[h, ch] - rn_llm_h[h, ch]) / max(rn_llm_h[h, ch], 1e-12)),
                    })

    # CSV
    csv_path = OUT_DIR / "k_proj_row_norms.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")

    # Summary stats per pair
    print("\n" + "="*70)
    print("Row-norm distribution per pair (= per-output-channel K gain)")
    print("="*70)
    for name in PAIRS:
        sub = [r for r in rows if r["pair"] == name]
        rn = np.array([r["row_norm_llm"] for r in sub])
        print(f"\n{name}: n={len(sub)}  min={rn.min():.3f}  max={rn.max():.3f}  mean={rn.mean():.3f}")
        for q in (50, 75, 90, 95, 99, 99.5, 99.9, 99.95):
            print(f"  p{q:>5.2f}: {np.percentile(rn, q):.3f}")
        thr = rn.mean() * 3.0  # "amplifier" threshold relative to mean
        n_amp = (rn > thr).sum()
        print(f"  rows with norm > 3× mean (={thr:.3f}): {n_amp}/{len(sub)} ({100*n_amp/len(sub):.2f}%)")
        thr2 = rn.mean() * 5.0
        n_amp2 = (rn > thr2).sum()
        print(f"  rows with norm > 5× mean (={thr2:.3f}): {n_amp2}/{len(sub)} ({100*n_amp2/len(sub):.2f}%)")

    # Per-channel persistence: which (head, channel) indices appear most as amplifiers
    print("\n" + "="*70)
    print("Channel-index persistence of amplifier rows across layers")
    print("(channels appearing > 3× mean in many layers = persistent sink-encoder)")
    print("="*70)
    for name in PAIRS:
        sub = [r for r in rows if r["pair"] == name]
        rn_all = np.array([r["row_norm_llm"] for r in sub])
        mean = rn_all.mean()
        amps = [r for r in sub if r["row_norm_llm"] > 3 * mean]
        ch_count = Counter((r["head"], r["channel"]) for r in amps)
        print(f"\n{name}: {len(amps)} amplifier rows total. Top (head, channel) persistence:")
        for (h, ch), n in ch_count.most_common(12):
            layers_at = sorted(set(r["layer_idx"] for r in amps if r["head"]==h and r["channel"]==ch))
            print(f"  head={h} ch={ch:>3d}  appears in {n} layers  (layers: {layers_at[:8]}{'...' if len(layers_at)>8 else ''})")

    # VL shift differential
    print("\n" + "="*70)
    print("VL-induced shift: amplifier rows vs normal rows")
    print("="*70)
    for name in PAIRS:
        sub = [r for r in rows if r["pair"] == name]
        rn_all = np.array([r["row_norm_llm"] for r in sub])
        mean = rn_all.mean()
        amp = [r for r in sub if r["row_norm_llm"] > 3 * mean]
        rest = [r for r in sub if r["row_norm_llm"] <= 3 * mean]
        a_abs = np.array([r["abs_delta"] for r in amp]) if amp else np.array([0.0])
        a_rel = np.array([r["rel_delta"] for r in amp]) if amp else np.array([0.0])
        r_abs = np.array([r["abs_delta"] for r in rest])
        r_rel = np.array([r["rel_delta"] for r in rest])
        print(f"\n{name}:")
        print(f"  amplifiers (row_norm > 3× mean):  n={len(amp):>4d}  "
              f"mean |Δ|={a_abs.mean():.4f}  mean rel_Δ={a_rel.mean():.4f}")
        print(f"  rest                              n={len(rest):>4d}  "
              f"mean |Δ|={r_abs.mean():.4f}  mean rel_Δ={r_rel.mean():.4f}")
        if len(amp) > 0:
            print(f"  rel_Δ ratio (amp/rest): {a_rel.mean() / max(r_rel.mean(), 1e-12):.3f}")


if __name__ == "__main__":
    main()
