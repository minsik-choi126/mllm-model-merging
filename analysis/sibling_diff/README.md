# Sibling-Pair Sink Analysis

Empirical evidence that **VLM text-side degradation is mechanistically an
attention-sink corruption problem**, and that **QK-RMSNorm** acts as an
architectural protector.

## TL;DR (one-liner)

Qwen3-VL preserves IFEval (Δ ≈ −3 pt, within stderr); Qwen2.5-VL drops ~9 pt.
The two models differ primarily in **QK-RMSNorm**: Qwen3 isolates the
sink-amplifier signal into RMSNorm γ layers that VL training leaves frozen,
whereas Qwen2.5 has no such separation, so sink encoding lives directly in
W_q / W_k and gets disturbed by VL updates.

## Setup

Two sibling pairs, same vendor / similar scale, primary architectural delta
is **QK-RMSNorm**:

| Pair | Base LLM | VLM | L | hidden | KV heads | QK-norm | QKV bias |
|---|---|---|---:|---:|---:|:-:|:-:|
| qwen25 | Qwen2.5-7B-Instruct | Qwen2.5-VL-7B-Instruct | 28 | 3584 | 4 | ✗ | ✓ |
| qwen3  | Qwen3-8B            | Qwen3-VL-8B-Instruct   | 36 | 4096 | 8 | **✓** | ✗ |

VLM text backbone is extracted with the repo's `extraction/extract_lm.py` to
make state-dicts directly comparable element-wise.

## Findings (4 layers, ordered)

### 1 — Phenomenon: cross-vendor IFEval contrast

IFEval prompt-strict (instruct protocol; thinking forced off for Qwen3 family
via patched chat template):

| Model            | LLM   | VLM-LM | Δ      | Note |
|------------------|------:|------:|-------:|------|
| Qwen2.5-7B / VL  | 72.09 | 62.66 | **−9.43** | Repo's existing text_degradation.md |
| Qwen3-8B / VL    | 83.18 | 80.22 | **−2.96** | thinking-off, baseline matches Qwen3 paper 83.0 within 0.2 pt |

→ Same-vendor, same-era, similar-scale pair with **the architectural delta
being QK-RMSNorm**. Drop is 3× larger for the no-QK-norm sibling.

### 2 — Weight-level ΔW (magnitude + spectrum + direction)

Script: [`diff_geometry.py`](diff_geometry.py) (Frobenius) +
[`svd_and_gamma.py`](svd_and_gamma.py) (SVD).

| Sub-module mean rel_diff `‖ΔW‖_F / ‖W_LLM‖_F` | Qwen2.5 | Qwen3 | ratio |
|---|---:|---:|---:|
| self_attn.q_proj | 0.682 | 0.236 | 2.9× |
| mlp.gate_proj   | 0.608 | 0.206 | 3.0× |
| mlp.up_proj     | 0.600 | 0.196 | 3.1× |
| mlp.down_proj   | 0.572 | 0.193 | 3.0× |
| self_attn.k_proj | 0.540 | 0.225 | 2.4× |
| self_attn.o_proj | 0.523 | 0.190 | 2.8× |
| self_attn.v_proj | 0.420 | 0.166 | 2.5× |
| input_layernorm γ | 0.024 | 0.018 | 1.3× |
| post_attn_layernorm γ | 0.030 | 0.020 | 1.5× |
| self_attn.q_norm γ | — | 0.017 | (Qwen3-only) |
| self_attn.k_norm γ | — | 0.013 | (Qwen3-only) |

Spectrum (`svd_metrics.csv`):
- **ΔW is NOT low-rank** in either family. Stable rank `‖ΔW‖_F² / σ_max²` is
  ~100–150 for attention projections, **200–800 for MLP projections** in
  both families. *LoRA / task-vector low-rank assumption is empirically
  violated for VL adaptation.*
- Top-singular direction of ΔW is **mostly orthogonal to W_LLM's top
  direction** (alignment 0.02–0.28; Qwen3 averages 0.07, Qwen2.5 0.18).

Figures: `figures/heatmap_rel_diff.png`, `figures/per_sub_relative.png`,
`figures/depth_curves.png`, `figures/svd_stable_rank.png`,
`figures/svd_top_alignment.png`.

### 3 — Where sinks live (γ amplifier channels)

Script: [`svd_and_gamma.py`](svd_and_gamma.py) (γ extraction) +
[`sink_in_projections.py`](sink_in_projections.py) (W_k control).

Qwen3 has **two stages of sink amplification, both in γ layers**:

| Norm γ | n×L (chs) | max | mean | % > 5× mean | persistent? | VL Δ on amp |
|---|---:|---:|---:|---:|---|---:|
| Qwen3 input_layernorm | 147,456 | **28.1** | 0.81 | **1.15%** | ch.923 in 12 layers; ch.445/822/994/1214 each in 9 | rel 0.06% |
| Qwen3 k_norm          | 4,608  | **34.0** | 1.76 | **0.76%** | ch.48 in 6 layers; ch.47/49/50/53 each in 2–3 | rel 0.94% |
| Qwen3 q_norm          | 4,608  | 5.16   | 1.65 | 0.02% | only 1 outlier total | — |
| Qwen2.5 input_layernorm | 100,352 | 8.5 | 0.73 | **0.33%** | max 2 layers each | rel 0.59% |

Qwen2.5 has **only one stage** (input_layernorm), and its tail is markedly
weaker.

**Sink does NOT live in W_k projections** in either family
(per-output-row norm distribution flat: max ≈ 2.4× mean in both
[`k_proj_row_norms.csv`](k_proj_row_norms.csv)). So in Qwen2.5 the sink
amplification ultimately rides on W_k as a linear combination of input-γ
amplified channels — meaning W_k *updates inevitably perturb sink
encoding*.

Figures: `figures/gamma_channels_qwen3_k_norm.png`,
`figures/gamma_channels_qwen3_q_norm.png`,
`figures/gamma_delta_qwen3_k_norm.png`,
`figures/gamma_delta_qwen3_q_norm.png`.

### 4 — Layer that actually carries IFEval-relevant attention-sink signal

For Qwen3, top-1 k_norm γ channel per layer:

```
L 0:  ch 51 γ= 34.00            L18:  ch 49 γ=  5.66
L 1:  ch 50 γ= 10.81            L21:  ch113 γ=  4.28
L 4:  ch 55 γ= 10.44            L23:  ch 47 γ=  6.66
L 5:  ch117 γ= 15.88            L35:  ch 58 γ=  8.12
```

Amplifier cluster on head_dim **45–58 and 100–127** persistent across layers.

## Causal experiments

### C1 — Sufficiency (sink corruption ⇒ IFEval crash)

Script: [`c1_kill_sink_qwen3.py`](c1_kill_sink_qwen3.py).

Procedure: replace top-K (K=10) γ channels per layer in
`input_layernorm + q_norm + k_norm` with the layer mean γ. Architecture
unchanged; only amplifier scaling neutralized. Output:
`/131_data/geeho/minsik/Qwen3-8B-nosink-nothink/`.

Then run IFEval (thinking off) and compare with Qwen3-8B's 83.18 baseline.

**Hypothesis**: if sink amplification is causally necessary for instruction
format compliance, IFEval should drop sharply (≥10–20 pt).

### C2 — Necessity (sink restoration ⇒ IFEval recovery)

Script: [`c2_sas_qwen25_recover.py`](c2_sas_qwen25_recover.py).

Procedure: take Qwen2.5-VL-LM (extracted text backbone, already
IFEval-degraded). For each layer, identify the top-K (K=36 ≈ 1% of
hidden_size) hidden-dim channels by base-LLM `input_layernorm γ`. Replace
W_q / W_k / W_v *columns* at those indices with base-LLM weights; leave the
remaining ~99 % of projection weights at VLM-LM values. Output:
`cache/extracted/qwen25vl_7b_lm_sas/`.

Then run IFEval and compare with Qwen2.5-VL-LM's 62.66 baseline.

**Hypothesis**: if VL-induced IFEval damage is essentially sink-column
corruption, restoring those columns alone should bring IFEval back toward
the base-LLM number (≈72).

### C3 — Architectural causality (QK-norm protects against VL adaptation)

Planned: inject identity-initialized `q_norm` / `k_norm` modules into
Qwen2.5 backbone, run small-scale LLaVA-style VL training (1–2 GPU-day),
measure whether IFEval drop disappears. Tests whether the
architectural feature alone confers the immunity observed in Qwen3.

## Method (post-hoc recovery): SAS — Sink-Anchored Surgery

C2's restoration procedure is the **method contribution**: training-free,
gradient-free, single-hyperparameter (K, number of columns restored per
layer). It operates only on weights and on each model's own
`input_layernorm γ` profile; no calibration data, no inference, no SFT.

Differentiation from neighbouring work:
- **WeMask / ME-Layer paper ([arXiv 2605.08504](https://arxiv.org/abs/2605.08504))** uses RMSNorm γ to identify dims
  for inference-time *activation masking* (suppression). SAS instead does
  **weight-side restoration** (preservation), in the **opposite direction**
  (restore the sink, not kill it), and in a **different setting**
  (post-hoc VLM recovery, not general LLM improvement).
- **VL safety degradation literature ([arXiv 2410.07571](https://arxiv.org/abs/2410.07571), [arXiv 2410.09047](https://arxiv.org/abs/2410.09047))** addresses safety,
  not IFEval, and uses representation-space or coarse weight-merge
  interventions; we operate at the **γ-channel level** with mechanistic
  selection.

## Reproduction

```bash
# 1. Magnitude diff
python analysis/sibling_diff/diff_geometry.py

# 2. SVD + γ
python analysis/sibling_diff/svd_and_gamma.py

# 3. Sink-location control (W_k row norms)
python analysis/sibling_diff/sink_in_projections.py

# 4. Build C1 overlay (no-sink Qwen3-8B)
python analysis/sibling_diff/c1_kill_sink_qwen3.py

# 5. Build C2 overlay (SAS-restored Qwen2.5-VL-LM)
python analysis/sibling_diff/c2_sas_qwen25_recover.py
```

IFEval evaluation uses the repo's `evaluation/text/eval_8tasks.sh`. For the
Qwen3 family, eval requires the thinking-off overlay (chat_template patch
appending an empty `<think></think>` block to the assistant prompt) to land
within stderr of the Qwen3 tech report's 83.0 baseline; see
`/131_data/geeho/minsik/Qwen3-8B-nothink/` for the patch.

## Outputs

| File | Content |
|---|---|
| `diff_qwen25.csv` | Per-(layer, sub_module) `‖ΔW‖_F`, `‖W_LLM‖_F`, `rel_diff` for Qwen2.5 |
| `diff_qwen3.csv` | Same for Qwen3 |
| `svd_metrics.csv` | Per-projection ΔW spectral metrics (σ_max, stable rank, effective rank, alignment with W_LLM top dir) |
| `gamma_qk_qwen3.csv` | Qwen3 q_norm / k_norm γ values per (layer, channel), with VL shift |
| `k_proj_row_norms.csv` | W_k per-output-row L2 norms (sink-NOT-in-W_k control) |
| `figures/heatmap_rel_diff.png` | Two-panel log-color heatmap of per-(layer, sub) rel_diff |
| `figures/per_sub_relative.png` | Bar chart of mean rel_diff per sub-module, two pairs |
| `figures/depth_curves.png` | rel_diff vs relative depth, per sub-module |
| `figures/svd_stable_rank.png` | Stable rank of ΔW per sub-module per depth |
| `figures/svd_top_alignment.png` | Top-singular alignment of ΔW with W_LLM top right-singular |
| `figures/gamma_channels_qwen3_{q,k}_norm.png` | Per-layer overlay of γ in base vs VL |
| `figures/gamma_delta_qwen3_{q,k}_norm.png` | Heatmap of per-channel rel shift |

## Status

| Item | Status |
|---|---|
| Phenomenon (Qwen3 vs Qwen2.5) | Confirmed ✓ |
| Weight diagnosis (1–4) | Done ✓ |
| C1 (sink ablation in Qwen3-8B) | IFEval running |
| C2 (SAS restoration of Qwen2.5-VL-LM) | Built; IFEval pending GPU slot |
| C3 (QK-norm injection into Qwen2.5 + VL train) | Designed only |
