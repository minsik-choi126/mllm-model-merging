"""Direct measurement of T = sink logit gap, per (layer, head), for Qwen3-8B and
Qwen2.5-7B-Instruct. Feeds into the V19/V20 verification of the math note's
bound prediction.

Method:
  1. Forward-pass each model on a small set of IFEval-style calibration prompts.
  2. Extract per-(layer, head, query_pos) attention weights.
  3. For each query, define sink position p* = argmax over key positions of
     the attention weight, and define
        T = log(attention[p*] / second_max_attention)
     (this is the logit gap between sink and the next-best key position, by
     properties of softmax)
  4. Aggregate T across (heads, queries) per layer → per-layer median T.
  5. Save CSV + per-layer plot.

Output:
  analysis/sibling_diff/sink_T_qwen3.csv
  analysis/sibling_diff/sink_T_qwen25.csv
  analysis/sibling_diff/figures/sink_T_by_layer.png
"""

from __future__ import annotations
import csv, math, os, json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"

MODELS = {
    "qwen3":  {"path": "/131_data/geeho/minsik/Qwen3-8B-nothink",       "L": 36, "kvh": 8,  "h": 32},
    "qwen25": {"path": "/131_data/geeho/minsik/Qwen2.5-7B-Instruct",    "L": 28, "kvh": 4,  "h": 28},
}

# Short, varied calibration prompts (mix of IFEval-style instructions)
PROMPTS = [
    "Write exactly two sentences about the moon.",
    "List five colors. Use commas to separate them.",
    "Translate to French: The cat sat on the mat.",
    "What is 17 + 29? Answer with a single number.",
    "Describe a sunny day in one paragraph.",
    "Explain photosynthesis in three sentences.",
    "Name the seven days of the week in order.",
    "What is the capital of Japan? Answer in one word.",
    "Write a haiku about the ocean.",
    "List three benefits of regular exercise.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Convert 100 fahrenheit to celsius. Show the formula.",
    "Write a json object with name and age fields for a sample person.",
    "Explain why the sky is blue using simple language.",
    "Give three examples of mammals that live in water.",
]


@torch.no_grad()
def measure(name: str, conf: dict, prompts: list[str], gpu: int = 0):
    print(f"\n=== Measuring T for {name} on GPU {gpu} ===")
    tok = AutoTokenizer.from_pretrained(conf["path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        conf["path"], dtype=torch.bfloat16, device_map=f"cuda:{gpu}",
        attn_implementation="eager",  # need explicit attention to extract weights
        trust_remote_code=True,
    )
    model.eval()

    # accumulate per-(layer) lists of T values
    Ts_per_layer = [[] for _ in range(conf["L"])]
    sink_pos_per_layer = [[] for _ in range(conf["L"])]
    sink_mass_per_layer = [[] for _ in range(conf["L"])]

    for pi, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(f"cuda:{gpu}")
        out = model(**ids, output_attentions=True)

        # out.attentions: tuple of length L, each shape (1, n_heads, seq_len, seq_len)
        for li, attn in enumerate(out.attentions):
            # average across heads for cleaner per-layer signal (but keep all for distribution)
            attn = attn[0].float().cpu().numpy()  # (n_heads, S, S)
            n_heads, S, _ = attn.shape
            # for each query position q (excluding the very first which has trivial attention)
            for q in range(1, S):
                for h in range(n_heads):
                    # row: attention from query q over keys k=0..q
                    row = attn[h, q, :q + 1]
                    if row.sum() < 0.99:
                        continue  # bad row (shouldn't happen with eager attn)
                    sink = int(np.argmax(row))
                    sink_mass = float(row[sink])
                    # second-max attention (excluding sink)
                    row2 = row.copy()
                    row2[sink] = 0
                    second = float(row2.max())
                    if second < 1e-10:
                        continue
                    T = math.log(sink_mass / second)
                    Ts_per_layer[li].append(T)
                    sink_pos_per_layer[li].append(sink)
                    sink_mass_per_layer[li].append(sink_mass)
        if (pi + 1) % 5 == 0:
            print(f"  [{name}] {pi + 1}/{len(prompts)} prompts done")

    del model
    torch.cuda.empty_cache()

    # Per-layer aggregates
    rows = []
    for li in range(conf["L"]):
        ts = Ts_per_layer[li]
        if not ts:
            continue
        arr = np.array(ts)
        sm = np.array(sink_mass_per_layer[li])
        sp = np.array(sink_pos_per_layer[li])
        rows.append({
            "layer_idx": li,
            "n_obs": len(arr),
            "T_mean":   float(arr.mean()),
            "T_median": float(np.median(arr)),
            "T_p25":    float(np.percentile(arr, 25)),
            "T_p75":    float(np.percentile(arr, 75)),
            "T_p95":    float(np.percentile(arr, 95)),
            "sink_mass_median": float(np.median(sm)),
            "sink_mass_mean":   float(sm.mean()),
            "frac_sink_at_bos": float((sp == 0).mean()),
        })

    csv_path = SCRIPT_DIR / f"sink_T_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {csv_path}")
    return rows


def plot_T(rows_by_model: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in rows_by_model.items():
        xs = [r["layer_idx"] for r in rows]
        ys = [r["T_median"] for r in rows]
        lo = [r["T_p25"] for r in rows]
        hi = [r["T_p75"] for r in rows]
        axes[0].plot(xs, ys, marker="o", label=name)
        axes[0].fill_between(xs, lo, hi, alpha=0.15)

        ys_mass = [r["sink_mass_median"] for r in rows]
        axes[1].plot(xs, ys_mass, marker="o", label=name)
    axes[0].set_xlabel("layer index")
    axes[0].set_ylabel("sink logit gap T (nats, median)")
    axes[0].set_title("Per-layer sink logit gap (median ± IQR across heads × queries)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("layer index")
    axes[1].set_ylabel("sink attention mass (median)")
    axes[1].set_title("Per-layer attention mass at sink position")
    axes[1].grid(alpha=0.3); axes[1].legend()
    out = FIG_DIR / "sink_T_by_layer.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main(gpu: int = 0):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, conf in MODELS.items():
        results[name] = measure(name, conf, PROMPTS, gpu=gpu)
    plot_T(results)
    print("\nDone. Use sink_T_*.csv to plug T into the bound check V19.")


if __name__ == "__main__":
    import sys
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(gpu)
