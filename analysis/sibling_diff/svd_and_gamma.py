"""Beyond Frobenius magnitude: ΔW spectral structure + γ channel analysis.

A1. For every shared (layer, sub_module) projection between W_LLM and W_VLM_LM,
    compute spectral metrics of ΔW = W_VLM_LM - W_LLM:
      - sigma_1                 = top singular value of ΔW
      - frobenius_norm          = ||ΔW||_F  (matches diff_geometry.py)
      - stable_rank             = ||ΔW||_F^2 / sigma_1^2  (concentration in top-1 direction)
      - effective_rank          = exp( H({sigma_i^2 / S}) ),  S = sum sigma_i^2
                                  (computed over top-k singular values; lower-bound proxy)
      - sigma_max_alignment     = | <u_1, v_1_W_LLM> | where (u_1, v_1) are top singular vectors
                                  of ΔW; v_1_W_LLM is W_LLM's top right-singular vector.
                                  Tells whether ΔW writes into the direction W_LLM already used.

C3. For Qwen3 q_norm / k_norm (Qwen3-only modules), pull γ (head_dim=128) per layer
    for base LLM and VLM-LM. Save:
      - csv with per-channel values and deltas
      - figure: γ_q / γ_k across layers, base vs VL, plus per-channel rel_diff distribution

Outputs in this directory:
    svd_metrics.csv
    gamma_qk_qwen3.csv
    figures/svd_stable_rank.png         (per-sub stable rank vs depth, two pairs)
    figures/svd_top_alignment.png       (alignment with W_LLM's top singular dir)
    figures/gamma_channels_qwen3.png    (γ_q / γ_k overlaid per layer, base vs VL)
    figures/gamma_delta_qwen3.png       (heatmap layer x channel, log|γ_VL - γ_base|)
"""

from __future__ import annotations

import csv
import json
import math
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
SVD_TOP_K = 64  # top-k singular values for effective-rank estimate


def _resolve(path: str) -> dict[str, str]:
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
    raise FileNotFoundError(path)


def _load(path: str, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(torch.float32)


def parse_layer_sub(key: str):
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


def is_projection(sub: str) -> bool:
    """Spectral metrics only meaningful for 2-D projections."""
    return sub in {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }


def randomized_svd_top_k(M: torch.Tensor, k: int, niter: int = 2):
    """Return (sigma[:k], u[:, :k], v[:, :k]) via torch.svd_lowrank (niter small)."""
    k_eff = min(k, *M.shape) - 1
    if k_eff <= 8 or min(M.shape) < k + 10:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return S[:k_eff], U[:, :k_eff], Vh[:k_eff, :].T
    U, S, V = torch.svd_lowrank(M, q=k + 6, niter=niter)
    return S[:k_eff], U[:, :k_eff], V[:, :k_eff]


def top_right_singular_vector(M: torch.Tensor, n_iter: int = 8) -> torch.Tensor:
    """Power iteration for the top right-singular vector of M (much cheaper than SVD)."""
    # v_1 of M satisfies (M^T M) v = sigma_1^2 v
    n = M.shape[1]
    v = torch.randn(n)
    v = v / v.norm()
    for _ in range(n_iter):
        v = M.T @ (M @ v)
        v = v / (v.norm() + 1e-30)
    return v


def effective_rank(sigmas: torch.Tensor) -> float:
    """exp( H(p) ) where p_i = sigma_i^2 / sum(sigma_j^2)."""
    s2 = sigmas.pow(2)
    p = s2 / s2.sum().clamp_min(1e-30)
    p = p[p > 0]
    H = -(p * p.log()).sum().item()
    return float(math.exp(H))


# ---------------- A1: SVD on ΔW ----------------

def compute_svd_metrics() -> list[dict]:
    """Streaming: append each row to CSV immediately so a crash leaves partial output."""
    csv_path = OUT_DIR / "svd_metrics.csv"
    fields = ["pair", "key", "layer_idx", "rel_depth", "sub_module", "shape",
              "fro_delta", "sigma_1_delta", "stable_rank_delta",
              f"effective_rank_top{SVD_TOP_K}_delta", "top_align_with_W_LLM"]
    f_csv = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(f_csv, fieldnames=fields)
    csv_w.writeheader(); f_csv.flush()
    rows: list[dict] = []
    for name, conf in PAIRS.items():
        llm_map = _resolve(conf["llm"])
        vlm_map = _resolve(conf["vlm_lm"])
        common = sorted(set(llm_map) & set(vlm_map))
        L = conf["num_layers"]
        proj_keys = [k for k in common if is_projection(parse_layer_sub(k)[1])]
        print(f"=== {name}: {len(proj_keys)} projection tensors ===", flush=True)
        for i, key in enumerate(proj_keys):
            try:
                W_llm = _load(llm_map[key], key)
                W_vlm = _load(vlm_map[key], key)
                D = W_vlm - W_llm
                sD, uD, vD = randomized_svd_top_k(D, SVD_TOP_K, niter=2)
                sigma_1 = float(sD[0].item())
                fro = float(D.norm().item())
                stable_rank = (fro ** 2) / (sigma_1 ** 2 + 1e-30)
                eff_rank = effective_rank(sD)
                # alignment via cheap power iteration on W_llm (not full SVD)
                v1_llm = top_right_singular_vector(W_llm, n_iter=8)
                v1_D = vD[:, 0]
                alignment = float(torch.abs(v1_llm.flatten() @ v1_D.flatten()).item())
                del W_llm, W_vlm, D, sD, uD, vD, v1_llm
            except Exception as e:
                print(f"  [{name}] {i+1}/{len(proj_keys)} {key} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            li, sub = parse_layer_sub(key)
            row = {
                "pair": name, "key": key, "layer_idx": li,
                "rel_depth": li / max(L - 1, 1), "sub_module": sub,
                "shape": "x".join(str(d) for d in (vD.shape if False else [])) or "",
                "fro_delta": fro, "sigma_1_delta": sigma_1,
                "stable_rank_delta": stable_rank,
                f"effective_rank_top{SVD_TOP_K}_delta": eff_rank,
                "top_align_with_W_LLM": alignment,
            }
            rows.append(row)
            csv_w.writerow(row); f_csv.flush()
            print(f"  [{name}] {i+1}/{len(proj_keys)}  L{li:>2} {sub:25s}  "
                  f"σ1={sigma_1:.2e}  sr={stable_rank:>7.2f}  er={eff_rank:>5.2f}  align={alignment:.3f}", flush=True)
    f_csv.close()
    return rows


# ---------------- C3: γ channels for Qwen3 q_norm/k_norm ----------------

def compute_gamma_qwen3() -> list[dict]:
    """γ_q / γ_k per layer for Qwen3 base vs VLM-LM (Qwen2.5 has no q_norm/k_norm)."""
    conf = PAIRS["qwen3"]
    llm_map = _resolve(conf["llm"])
    vlm_map = _resolve(conf["vlm_lm"])
    rows: list[dict] = []
    for sub in ("q_norm", "k_norm"):
        for L in range(conf["num_layers"]):
            key = f"model.layers.{L}.self_attn.{sub}.weight"
            if key not in llm_map or key not in vlm_map:
                continue
            g_llm = _load(llm_map[key], key)
            g_vlm = _load(vlm_map[key], key)
            d = (g_vlm - g_llm).abs()
            for ch in range(g_llm.shape[0]):
                rows.append({
                    "sub_module": sub,
                    "layer_idx": L,
                    "channel": int(ch),
                    "gamma_llm": float(g_llm[ch].item()),
                    "gamma_vlm": float(g_vlm[ch].item()),
                    "abs_delta": float(d[ch].item()),
                    "rel_delta": float((d[ch] / max(abs(g_llm[ch].item()), 1e-12)).item()),
                })
    return rows


# ---------------- plotting ----------------

def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


SUB_PROJ_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def plot_svd_panel(rows: list[dict], metric: str, fname: str, ylabel: str, ylog: bool = False):
    pairs = sorted({r["pair"] for r in rows})
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.5 * len(pairs), 5), squeeze=False, sharey=True)
    axes = axes[0]
    for ax, pname in zip(axes, pairs):
        for sub in SUB_PROJ_ORDER:
            data = sorted([r for r in rows if r["pair"] == pname and r["sub_module"] == sub],
                          key=lambda r: r["layer_idx"])
            if not data:
                continue
            xs = [r["rel_depth"] for r in data]
            ys = [r[metric] for r in data]
            ax.plot(xs, ys, marker="o", markersize=3, label=sub, linewidth=1.2)
        ax.set_title(pname)
        ax.set_xlabel("relative depth l/(L-1)")
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle(ylabel)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{fname}")


def plot_gamma_channels(rows: list[dict]):
    """For Qwen3: overlay γ_LLM vs γ_VLM per channel for each layer."""
    layers = sorted({r["layer_idx"] for r in rows})
    L = len(layers)
    for sub in ("q_norm", "k_norm"):
        # subplot grid: ~6 cols
        n = L
        cols = 6
        rows_g = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_g, cols, figsize=(cols * 2.4, rows_g * 1.6), squeeze=False)
        ymin = min(r["gamma_llm"] for r in rows if r["sub_module"] == sub)
        ymax = max(max(r["gamma_llm"], r["gamma_vlm"]) for r in rows if r["sub_module"] == sub)
        for i, li in enumerate(layers):
            ax = axes[i // cols][i % cols]
            data = sorted([r for r in rows if r["sub_module"] == sub and r["layer_idx"] == li],
                          key=lambda r: r["channel"])
            xs = [r["channel"] for r in data]
            y_llm = [r["gamma_llm"] for r in data]
            y_vlm = [r["gamma_vlm"] for r in data]
            ax.plot(xs, y_llm, color="#1f77b4", linewidth=0.7, alpha=0.9, label="LLM")
            ax.plot(xs, y_vlm, color="#d62728", linewidth=0.7, alpha=0.9, label="VLM-LM")
            ax.set_title(f"L{li}", fontsize=7)
            ax.set_ylim(ymin * 0.95, ymax * 1.05)
            ax.tick_params(labelsize=5)
            if i == 0:
                ax.legend(fontsize=6, loc="best")
        # blank trailing
        for j in range(n, rows_g * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"Qwen3 γ ({sub}): per-channel, base LLM vs Qwen3-VL-LM", fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / f"gamma_channels_qwen3_{sub}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_gamma_delta_heatmap(rows: list[dict]):
    """Heatmap: layer x channel, color = |γ_VL − γ_LLM| / |γ_LLM|."""
    for sub in ("q_norm", "k_norm"):
        layers = sorted({r["layer_idx"] for r in rows if r["sub_module"] == sub})
        channels = sorted({r["channel"] for r in rows if r["sub_module"] == sub})
        H = np.full((len(layers), len(channels)), np.nan)
        for r in rows:
            if r["sub_module"] != sub:
                continue
            i = layers.index(r["layer_idx"])
            j = channels.index(r["channel"])
            H[i, j] = r["rel_delta"]
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(H, aspect="auto", cmap="magma")
        ax.set_xlabel("γ channel (head_dim)")
        ax.set_ylabel("layer")
        ax.set_title(f"Qwen3 γ ({sub}): per-channel relative VL-induced shift")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        fig.tight_layout()
        out = FIG_DIR / f"gamma_delta_qwen3_{sub}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def summary_print_svd(rows: list[dict]):
    print()
    for p in sorted({r["pair"] for r in rows}):
        sub_rows = [r for r in rows if r["pair"] == p]
        print(f"=== {p} ===")
        by_sub = defaultdict(list)
        for r in sub_rows:
            by_sub[r["sub_module"]].append(r)
        for sub in SUB_PROJ_ORDER:
            xs = by_sub.get(sub, [])
            if not xs:
                continue
            sr = np.array([r["stable_rank_delta"] for r in xs])
            er = np.array([r[f"effective_rank_top{SVD_TOP_K}_delta"] for r in xs])
            al = np.array([r["top_align_with_W_LLM"] for r in xs])
            print(f"  {sub:25s}  stable_rank mean={sr.mean():.2f}  eff_rank mean={er.mean():.2f}  top_align mean={al.mean():.3f}  (n={len(xs)})")


def summary_print_gamma(rows: list[dict]):
    print()
    for sub in ("q_norm", "k_norm"):
        ssub = [r for r in rows if r["sub_module"] == sub]
        if not ssub:
            continue
        abs_d = np.array([r["abs_delta"] for r in ssub])
        rel_d = np.array([r["rel_delta"] for r in ssub])
        print(f"=== Qwen3 {sub} γ summary ===")
        print(f"  n channels x layers: {len(ssub)}")
        print(f"  abs_delta:  mean={abs_d.mean():.3e}  max={abs_d.max():.3e}")
        print(f"  rel_delta:  mean={rel_d.mean():.3e}  max={rel_d.max():.3e}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- A1: SVD spectrum of ΔW per (layer, projection) ---")
    svd_rows = compute_svd_metrics()
    write_csv(svd_rows, OUT_DIR / "svd_metrics.csv")
    print(f"wrote svd_metrics.csv  ({len(svd_rows)} rows)")
    plot_svd_panel(svd_rows, "stable_rank_delta", "svd_stable_rank.png",
                   "stable rank of ΔW", ylog=False)
    plot_svd_panel(svd_rows, f"effective_rank_top{SVD_TOP_K}_delta",
                   "svd_effective_rank.png",
                   f"effective rank (top-{SVD_TOP_K} approx)", ylog=False)
    plot_svd_panel(svd_rows, "top_align_with_W_LLM", "svd_top_alignment.png",
                   "|<v_1(ΔW), v_1(W_LLM)>|  (0=orthog, 1=parallel)", ylog=False)
    summary_print_svd(svd_rows)

    print("\n--- C3: γ channels for Qwen3 q_norm / k_norm ---")
    g_rows = compute_gamma_qwen3()
    write_csv(g_rows, OUT_DIR / "gamma_qk_qwen3.csv")
    print(f"wrote gamma_qk_qwen3.csv  ({len(g_rows)} rows)")
    plot_gamma_channels(g_rows)
    plot_gamma_delta_heatmap(g_rows)
    summary_print_gamma(g_rows)


if __name__ == "__main__":
    main()
