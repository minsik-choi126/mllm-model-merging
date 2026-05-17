"""Per-(layer, sub-module) weight-diff geometry between VLM text-backbone and base LLM.

Two pairs:
  qwen25: Qwen2.5-VL-7B-Instruct (extracted LM)  vs  Qwen2.5-7B-Instruct      [28 layers]
  qwen3 : Qwen3-VL-8B-Instruct   (extracted LM)  vs  Qwen3-8B                 [36 layers]

For every shared tensor key, computes
    norm_llm   = ||W_LLM||_F
    norm_delta = ||W_VLM_LM - W_LLM||_F
    rel_diff   = norm_delta / max(norm_llm, eps)

Outputs (next to this script):
    diff_qwen25.csv, diff_qwen3.csv
    figures/heatmap_rel_diff.png    — 2-panel layer x sub_module heatmap, log color scale
    figures/per_sub_relative.png    — mean rel_diff per sub_module, side-by-side bars
    figures/depth_curves.png        — rel_diff vs relative depth, per sub_module
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open


PAIRS = {
    "qwen25": {
        "llm": "/131_data/geeho/minsik/Qwen2.5-7B-Instruct",
        "vlm_lm": "/131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen25vl_7b_lm",
        "num_layers": 28,
    },
    "qwen3": {
        "llm": "/131_data/geeho/minsik/Qwen3-8B",
        "vlm_lm": "/131_data/geeho/minsik/code/mllm-model-merging/cache/extracted/qwen3vl_8b_lm",
        "num_layers": 36,
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
FIG_DIR = OUT_DIR / "figures"


def _resolve_safetensors_files(path: str) -> dict[str, str]:
    """Map every key in a model to the absolute path of its safetensors shard."""
    p = Path(path)
    if p.is_file():
        with safe_open(str(p), framework="pt") as f:
            return {k: str(p) for k in f.keys()}
    idx_file = p / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.load(open(idx_file))
        return {k: str(p / shard) for k, shard in idx["weight_map"].items()}
    single = p / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt") as f:
            return {k: str(single) for k in f.keys()}
    raise FileNotFoundError(f"No safetensors found at {path}")


def _iter_grouped_by_shard(key_to_shard: dict[str, str]):
    """Yield (shard_path, [keys]) so each shard is opened once."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for k, s in key_to_shard.items():
        grouped[s].append(k)
    for shard, keys in grouped.items():
        yield shard, sorted(keys)


def parse_layer_sub(key: str):
    """Return (layer_idx or None, sub_module). sub_module collapses .weight/.bias."""
    if key == "model.embed_tokens.weight":
        return None, "embed_tokens"
    if key == "model.norm.weight":
        return None, "final_norm"
    if key == "lm_head.weight":
        return None, "lm_head"
    if not key.startswith("model.layers."):
        return None, key
    rest = key[len("model.layers."):]
    li_str, sub_path = rest.split(".", 1)
    li = int(li_str)
    if sub_path.endswith(".weight"):
        sub = sub_path[: -len(".weight")]
    elif sub_path.endswith(".bias"):
        sub = sub_path[: -len(".bias")] + ".bias"
    else:
        sub = sub_path
    return li, sub


def compute_pair_diff(name: str, conf: dict) -> list[dict]:
    llm_map = _resolve_safetensors_files(conf["llm"])
    vlm_map = _resolve_safetensors_files(conf["vlm_lm"])
    common = sorted(set(llm_map) & set(vlm_map))
    assert len(common) == len(llm_map) == len(vlm_map), (
        f"key mismatch for {name}: {len(common)} common vs {len(llm_map)}/{len(vlm_map)}"
    )

    L = conf["num_layers"]
    rows: list[dict] = []

    seen = 0
    for llm_shard, keys in _iter_grouped_by_shard(llm_map):
        vlm_groups: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            vlm_groups[vlm_map[k]].append(k)

        with safe_open(llm_shard, framework="pt") as f_llm:
            for vlm_shard, sub_keys in vlm_groups.items():
                with safe_open(vlm_shard, framework="pt") as f_vlm:
                    for k in sub_keys:
                        w_llm = f_llm.get_tensor(k).to(torch.float32)
                        w_vlm = f_vlm.get_tensor(k).to(torch.float32)
                        delta = w_vlm - w_llm
                        norm_llm = float(w_llm.norm().item())
                        norm_delta = float(delta.norm().item())
                        rel = norm_delta / max(norm_llm, 1e-12)
                        li, sub = parse_layer_sub(k)
                        rows.append(
                            {
                                "pair": name,
                                "key": k,
                                "layer_idx": li if li is not None else -1,
                                "rel_depth": (li / max(L - 1, 1)) if li is not None else -1.0,
                                "sub_module": sub,
                                "shape": "x".join(str(d) for d in w_llm.shape),
                                "numel": int(w_llm.numel()),
                                "norm_llm": norm_llm,
                                "norm_delta": norm_delta,
                                "rel_diff": rel,
                            }
                        )
                        seen += 1
                        if seen % 50 == 0:
                            print(f"  [{name}] {seen}/{len(common)}: {k}  rel={rel:.3e}")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------- plotting ----------

SUB_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
]


def make_heatmap(all_rows: list[dict]):
    pairs = sorted({r["pair"] for r in all_rows})
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.5 * len(pairs), 6.5), squeeze=False)
    axes = axes[0]

    layered = [r for r in all_rows if r["layer_idx"] >= 0 and r["rel_diff"] > 0]
    all_log = np.log10([r["rel_diff"] for r in layered])
    vmin, vmax = float(np.quantile(all_log, 0.02)), float(np.quantile(all_log, 0.98))

    for ax, pname in zip(axes, pairs):
        sub_rows = [r for r in all_rows if r["pair"] == pname and r["layer_idx"] >= 0]
        L = max(r["layer_idx"] for r in sub_rows) + 1
        present_subs = [s for s in SUB_ORDER if any(r["sub_module"] == s for r in sub_rows)]
        H = np.full((len(present_subs), L), np.nan)
        for r in sub_rows:
            if r["sub_module"] in present_subs:
                i = present_subs.index(r["sub_module"])
                j = r["layer_idx"]
                v = r["rel_diff"]
                H[i, j] = np.log10(v) if v > 0 else np.nan

        im = ax.imshow(H, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(present_subs)))
        ax.set_yticklabels(present_subs, fontsize=8)
        ax.set_xticks(range(0, L, max(1, L // 14)))
        ax.set_xlabel("layer index")
        ax.set_title(f"{pname}  (L={L})  log10(rel_diff)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Per-(layer, sub-module) ||W_VLM-LM - W_LLM||_F / ||W_LLM||_F  (log10)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "heatmap_rel_diff.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def make_per_sub_bars(all_rows: list[dict]):
    pairs = sorted({r["pair"] for r in all_rows})
    by = defaultdict(lambda: {p: [] for p in pairs})
    for r in all_rows:
        if r["layer_idx"] >= 0:
            by[r["sub_module"]][r["pair"]].append(r["rel_diff"])

    subs = [s for s in SUB_ORDER if s in by]
    means = {p: np.array([np.mean(by[s][p]) if by[s][p] else np.nan for s in subs]) for p in pairs}
    stds = {p: np.array([np.std(by[s][p]) if by[s][p] else 0.0 for s in subs]) for p in pairs}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(subs))
    w = 0.4
    colors = {"qwen25": "#d97757", "qwen3": "#4c72b0"}
    for i, p in enumerate(pairs):
        ax.bar(x + (i - 0.5) * w, means[p], width=w, yerr=stds[p], label=p,
               color=colors.get(p, None), capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("mean rel_diff (layers)")
    ax.set_title("Mean relative weight-diff per sub-module")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "per_sub_relative.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def make_depth_curves(all_rows: list[dict]):
    pairs = sorted({r["pair"] for r in all_rows})
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.5 * len(pairs), 5), squeeze=False, sharey=True)
    axes = axes[0]

    pivot_subs = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    for ax, pname in zip(axes, pairs):
        rows = [r for r in all_rows if r["pair"] == pname and r["layer_idx"] >= 0]
        for sub in pivot_subs:
            xs, ys = [], []
            for r in sorted([r for r in rows if r["sub_module"] == sub], key=lambda r: r["layer_idx"]):
                xs.append(r["rel_depth"])
                ys.append(r["rel_diff"])
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3, label=sub, linewidth=1.2)
        ax.set_title(pname)
        ax.set_xlabel("relative depth l/(L-1)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("rel_diff (log)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle("Weight-diff vs relative depth per sub-module", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "depth_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def summary_print(all_rows: list[dict]):
    for p in sorted({r["pair"] for r in all_rows}):
        rows = [r for r in all_rows if r["pair"] == p]
        layered = [r for r in rows if r["layer_idx"] >= 0]
        rels = np.array([r["rel_diff"] for r in layered])
        print(f"\n=== {p} summary ===")
        print(f"  layered tensors: {len(layered)}")
        print(f"  rel_diff:  mean={rels.mean():.3e}  median={np.median(rels):.3e}  max={rels.max():.3e}")
        top = sorted(layered, key=lambda r: r["rel_diff"], reverse=True)[:10]
        print(f"  top-10 by rel_diff:")
        for r in top:
            print(f"    L{r['layer_idx']:>2}  {r['sub_module']:30s}  rel={r['rel_diff']:.3e}")
        by_sub = defaultdict(list)
        for r in layered:
            by_sub[r["sub_module"]].append(r["rel_diff"])
        print(f"  by sub_module (mean rel_diff):")
        for sub in sorted(by_sub, key=lambda s: -np.mean(by_sub[s])):
            print(f"    {sub:30s}  {np.mean(by_sub[sub]):.3e}  (n={len(by_sub[sub])})")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for name, conf in PAIRS.items():
        print(f"\n=== Computing diff for {name} ===")
        rows = compute_pair_diff(name, conf)
        write_csv(rows, OUT_DIR / f"diff_{name}.csv")
        print(f"  wrote {OUT_DIR / f'diff_{name}.csv'}  ({len(rows)} rows)")
        all_rows.extend(rows)

    print("\n=== plotting ===")
    make_heatmap(all_rows)
    make_per_sub_bars(all_rows)
    make_depth_curves(all_rows)

    summary_print(all_rows)


if __name__ == "__main__":
    main()
