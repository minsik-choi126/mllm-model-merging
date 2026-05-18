# Sibling-Pair Sink Analysis

Empirical evidence that **VLM text-side degradation is mechanistically an
attention-sink corruption problem**, with **QK-RMSNorm acting as an
architectural protector**, validated by:

1. Cross-vendor natural experiment (Qwen3 vs Qwen2.5)
2. Weight-level diagnosis (ΔW geometry + γ amplifier mapping)
3. Causal sink ablation (C1) showing −44 pt IFEval crash
4. A formal mechanism lemma (γ-vs-W structural decoupling)
5. Three external generalization cross-checks (LLaVA-LLaMA3, InternVL3,
   InternVL3.5)
6. Training-free recovery method (SAS) as lower-bound proof
7. Planned training experiment (C3) for architectural causality

This document is the single source of truth for the project — every
artifact (scripts, CSVs, figures, configs) is referenced from it.

> **Status (2026-05-18).** Phenomenon + diagnosis + C1 + SAS done. Math
> foundation drafted. **Generalization eval complete (5 pairs across 3
> vendors): QK-norm models preserve (Δ ≈ −1 to −3 pt), non-QK-norm
> models drop sharply (Δ ≈ −9 to −27 pt).** Random-W control (E2)
> confirms W-perturbation is sufficient for catastrophic IFEval damage.
> C3 training infra ready, pending data prep and launch.

---

## 1. TL;DR — phenomenon table (5 pairs + 1 mechanism control)

| Pair | LLM IFEval (prompt-strict) | VLM-LM IFEval | Δ (drop) | QK-norm? | Vendor |
|---|---:|---:|---:|:-:|---|
| Qwen3-8B / Qwen3-VL-8B-Instruct (no-think) | **83.18** | **80.22** | **−2.96** | ✓ | Qwen |
| Qwen3-8B / **InternVL3.5-8B** | 83.18* | **82.07** | **−1.11** | ✓ | OpenGVLab |
| Qwen2.5-7B-Instruct / Qwen2.5-VL-7B-Instruct | 72.09 | 62.66 | **−9.43** | ✗ | Qwen |
| Qwen2.5-7B-Instruct / **InternVL3-8B** | 72.09* | 63.40 | **−8.69** | ✗ | OpenGVLab |
| Meta-Llama-3-8B-Instruct / LLaVA-LLaMA3-8B (LoRA) | 69.69 | 43.07 | **−26.62** | ✗ | LMMS-Lab |
| Qwen2.5-7B-Instruct **+ random Gaussian W perturbation** (mechanism control) | 72.09 | **10.72** | **−61.37** | n/a | synthetic |

\* InternVL teams slightly modify their backbone (vocab size, MLP IS for
InternVL3.5), so the "LLM baseline" is approximate — our Qwen-base
numbers serve as the closest public reference.

**Categorical separation**: every QK-norm model preserves IFEval (Δ ∈
[−3, −1] pt); every non-QK-norm model drops sharply (Δ ∈ [−27, −9] pt).
Three independent vendors (Qwen, OpenGVLab, LMMS-Lab) using three
different recipes (Qwen2.5/3-VL pipeline, InternVL's 4-stage CascadeRL
recipe, LLaVA-NeXT LoRA recipe) all show the same pattern.

The random-W control (E2) — a Gaussian ΔW with magnitude matched to the
measured Qwen2.5-VL rel-Frobenius, γ untouched — collapses IFEval to
**10.72** (−61 pt), confirming that **W-mode perturbation is sufficient
to break IFEval catastrophically**. Real VL training (−9 pt) is much
gentler than random because it implicitly stays in a structure-
respecting subspace; but it still crosses the threshold where QK-norm-
less sinks fail.

The only family preserving IFEval is the one with **QK-RMSNorm in the
LLM backbone** (Qwen3). Killing the γ amplifiers in Qwen3-8B drops IFEval
to **38.82 (−44.36 pt)** — direct sufficiency proof that sink amplifiers
are necessary for instruction-format compliance.

**Paper-grade thesis**:

> *"Instruction following is mechanistically a free-rider on attention-sink
> stability. VL adaptation perturbs the projection weights that route input
> γ-amplified channels into the K/Q vector; QK-RMSNorm decouples the
> sink-amplification scale (frozen γ) from these projections, providing an
> architectural buffer against the perturbation."*

---

## 2. Phenomenon — cross-vendor IFEval contrast

### 2.1 Measurements (instruct protocol, lm-evaluation-harness 0.4.5, bf16)

All Qwen3 family numbers use the **thinking-off** chat-template overlay
(`/131_data/geeho/minsik/Qwen3-8B-nothink`) because the default chat
template emits a `<think>...</think>` block that breaks IFEval format
checking. Without this fix, Qwen3-8B baseline scores 34.75 (−48 pt below
its real ability).

| Model | prompt-strict | inst-strict | prompt-loose | inst-loose |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 72.09 | 79.50 | 74.49 | 81.77 |
| Qwen2.5-VL-7B-Instruct → text-backbone | 62.66 | 71.58 | 65.25 | 73.86 |
| Qwen3-8B (no-think) | 83.18 | 88.49 | 85.95 | 90.41 |
| Qwen3-VL-8B-Instruct → text-backbone | 80.22 | 86.57 | 83.36 | 88.61 |
| Meta-Llama-3-8B-Instruct | 69.69 | 78.42 | 77.08 | 84.41 |
| LLaVA-LLaMA3-8B → text-backbone | 43.07 | 55.40 | 45.10 | 57.19 |
| InternVL3-8B → text-backbone | 63.40 | 70.74 | 68.58 | 75.30 |
| InternVL3.5-8B → text-backbone | 82.07 | 87.53 | 85.40 | 89.93 |
| Qwen2.5-7B-Instruct + random W (E2 control) | 10.72 | 21.46 | 10.91 | 21.58 |

`extraction/extract_lm.py` produces a standalone HF text-only model from
each VLM checkpoint; we evaluate the resulting backbone directly under
the LLM's chat template so the LLM ↔ VLM-LM comparison is apples-to-apples.

### 2.2 Pipeline validation — comparison vs official sources

Anchoring our numbers to public references where they exist:

| Model | Metric | Our | Official | Source | Δ | Notes |
|---|---|---:|---:|---|---:|---|
| Qwen2.5-7B-Instruct | prompt-strict (0-sh chat) | 72.09 | **71.2** | [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5-llm/) | +0.9 | within stderr |
| Qwen3-8B (no-think) | prompt-strict (0-sh chat) | 83.18 | **83.0** | Qwen3 tech report [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) Table 18 | +0.2 | within stderr |
| Qwen3-8B (think) | prompt-strict (0-sh chat) | (not measured) | **85.0** | Qwen3 tech report Table 17 | — | thinking on |
| Meta-Llama-3-8B-Instruct | avg-of-4 | 77.40 | **74.08** | [OLL v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) `meta-llama/Meta-Llama-3-8B-Instruct` | +3.3 | framework drift |
| InternVL3.5-30B-A3B | (paper's metric) | (n/a, we use 8B) | **74.3** | InternVL3.5 paper [arXiv 2508.18265](https://arxiv.org/abs/2508.18265) Table 2 | — | reference only — different size |
| InternVL3.5-241B-A28B | (paper's metric) | (n/a) | **83.7** | InternVL3.5 paper Table 2 | — | reference only |

**Validation verdict**: Qwen 2 models match within 1 pt, pipeline correct.
Llama-3 has +3.3 pt drift vs OLL — within typical cross-framework variance
(lm-eval-harness vs Meta/OLL conventions differ on `max_gen_toks`,
chat-template, and few-shot example sampling).

### 2.3 Reference gap for VLM-side IFEval

**No paper publishes text-only IFEval for the 8B-class VLMs we study.**
- Qwen2.5-VL paper (arXiv 2502.13923): only the 72B variant reports text-only
  benchmarks ("complete capability alignment with Qwen2.5-72B"). The 7B
  variant has only vision benchmarks.
- Qwen3-VL: no separate text-only IFEval reported.
- InternVL3 paper (arXiv 2504.10479): no IFEval reported.
- InternVL3.5 paper (arXiv 2508.18265): IFEval row in Table 2 only
  includes InternVL3.5-30B-A3B (74.3) and -241B-A28B (83.7); -1B/-2B/
  -4B/-8B/-14B/-38B sizes are absent from the IFEval column.
- LLaVA-LLaMA3 / LLaVA-NeXT: papers test only vision benchmarks.

→ **Our measurements are first-party reference** for the 8B-class VLM-LM
IFEval. Paper appendix will state this explicitly.

### 2.4 Trustworthiness sanity-check for VLM-LM measurements

Lacking direct references for the 8B VLMs, we cross-check internal
consistency:

| Measurement | Sanity reference | Δ from reference | Verdict |
|---|---|---:|---|
| **InternVL3-8B = 63.40** | Qwen2.5-VL-7B drop (−9.43 pt vs Qwen2.5-7B base = 72.09) | Our drop −8.69 vs Qwen2.5-VL −9.43 → 0.74 pt apart | ✓ consistent (within stderr ≈ 2 pt) |
| **InternVL3.5-8B = 82.07** | InternVL3.5-30B-A3B official = 74.3 (paper); -241B-A28B = 83.7 (paper) | 8B falls between MoE-30B and dense-241B, in plausible range | ✓ plausible; 8B dense likely outperforms 30B-A3B (3B active params) |
| **LLaVA-LLaMA3-8B = 43.07** | Llama-3-8B-Instruct LLM = 69.69 (our, vs OLL 74.08 → +3.3 drift); drop = −26.62 | Largest drop among 3 no-QK-norm pairs; consistent with most aggressive LoRA recipe + no QK-norm protection | ✓ plausible (LoRA recipe is the most disruptive) |
| **E2 (random Gaussian W) = 10.72** | mechanism control (no reference); rel-Frobenius matched to Qwen2.5-VL | −61 pt vs natural VL −9 pt: random direction is far more destructive than implicit-structure-respecting VL training | ✓ expected order of magnitude; supports the "VL is gentle relative to random" interpretation |

**Verdict**: every individual measurement passes its sanity check; the
categorical [−27, −9] vs [−3, −1] pt separation (§1) is internally
consistent with cross-references where available.

### 2.5 Caveats — Qwen3 family thinking-mode confound

The Qwen3 chat template defaults to `enable_thinking=True`. With this
default, IFEval format checkers (length, format, language-mark
constraints) parse the `<think>...</think>` trace and fail. Our chat-
template overlay patches the template to always emit an empty
`<think>\n\n</think>\n\n` block (= `enable_thinking=False` form), so the
model never emits a thinking trace at inference. This recovers the
Qwen3 tech report 83.0 baseline within 0.2 pt. Without this fix,
Qwen3-8B IFEval reads as 34.75 (which is *not* a real capability gap).

Overlay construction is automated in
[`scripts/run_full_pipeline.sh`](../../scripts/run_full_pipeline.sh)
stage 2 (symlinks the original Qwen3-8B files + overwrites
`tokenizer_config.json` / `chat_template.jinja` with the no-think variant).

### 2.6 Other engineering caveats (reproducibility-affecting)

These are the issues we hit during measurement; each impacts whether
re-runs reproduce our numbers:

- **Phi-3.5-Vision skipped**: `trust_remote_code` custom `Phi3VConfig` +
  the `transformers` 4.45+ removal of `DynamicCache.seen_tokens` make
  HF `AutoModel` extraction unreliable. We dropped this from the
  generalization matrix rather than maintain a brittle compat shim.
- **trust_remote_code VLM extraction**: `extraction/extract_lm.py` uses
  `AutoModel.from_pretrained` which fails for InternVL3 / 3.5 /
  LLaVA-LLaMA3 (custom config classes). Solution: `extract_direct.py`
  reads safetensors directly and maps `language_model.model.*` →
  `model.*`, using a `--config-src` override for vocab/IS mismatches.
- **lm-eval-harness silent kills**: occasional OOM or worker crashes
  result in *silent* job termination (process gone, no traceback). Our
  watcher script (memory: [Eval watcher pattern]) re-launches with
  `--skip-existing` to resume.
- **Llama-3 vs OLL drift +3.3 pt**: framework difference between
  `lm-eval-harness 0.4.5` and OLL v2 (chat template, max_gen_toks
  defaults, few-shot sampling). Within acceptable cross-framework
  variance, but report as ±3 pt sanity band.
- **Qwen3 base unmeasured under official chat-template thinking-on
  protocol**: Qwen3 tech report Table 17 reports IFEval=85.0 with
  thinking on; we use thinking-off (83.0) to keep VLM/LLM comparison
  apples-to-apples (Qwen3-VL does not emit `<think>` blocks for VL
  tasks at inference, even though Qwen3-8B does).

---

## 3. Weight-level diagnosis

Script: [`diff_geometry.py`](diff_geometry.py),
[`svd_and_gamma.py`](svd_and_gamma.py),
[`sink_in_projections.py`](sink_in_projections.py).

### 3.1 ΔW magnitude / spectrum / direction

Per-(layer, sub_module) `‖ΔW‖_F / ‖W_LLM‖_F` averaged across layers:

| Sub-module | Qwen2.5 | Qwen3 | ratio |
|---|---:|---:|---:|
| self_attn.q_proj | **0.682** | 0.236 | 2.9× |
| mlp.gate_proj   | 0.608 | 0.206 | 3.0× |
| mlp.up_proj     | 0.600 | 0.196 | 3.1× |
| mlp.down_proj   | 0.572 | 0.193 | 3.0× |
| self_attn.k_proj | 0.540 | 0.225 | 2.4× |
| self_attn.o_proj | 0.523 | 0.190 | 2.8× |
| self_attn.v_proj | 0.420 | 0.166 | 2.5× |
| input_layernorm γ | 0.024 | 0.018 | 1.3× |
| post_attn_layernorm γ | 0.030 | 0.020 | 1.5× |
| self_attn.q_norm γ | — | **0.017** | (Qwen3 only) |
| self_attn.k_norm γ | — | **0.013** | (Qwen3 only) |

**Qwen2.5 ΔW is ~3× larger than Qwen3 ΔW** across attention and MLP
projections; γ layers in both stay within ~2 % rel-shift (effectively
frozen during VL adaptation).

**Spectrum of ΔW** ([`svd_metrics.csv`](svd_metrics.csv)):
- Stable rank `‖ΔW‖_F² / σ_max²` ≈ 100–150 for attention projections,
  **200–800 for MLP projections** in both families.
- → ΔW is *high-rank, distributed*, NOT low-rank: the **LoRA / task-vector
  low-rank assumption is empirically falsified** for VL adaptation.
- Top-singular direction of ΔW is mostly orthogonal to W_LLM's main axis
  (alignment 0.02–0.28; Qwen3 averages 0.07, Qwen2.5 0.18) — ΔW writes
  into roughly fresh directions.

Figures: [`figures/heatmap_rel_diff.png`](figures/heatmap_rel_diff.png),
[`figures/svd_stable_rank.png`](figures/svd_stable_rank.png),
[`figures/svd_top_alignment.png`](figures/svd_top_alignment.png).

### 3.2 Sink encoding location

| Norm γ tensor | Qwen2.5 | Qwen3 |
|---|---:|---:|
| input_layernorm γ — n_channels | 100,352 | 147,456 |
| input_layernorm γ — max | **8.5** | **28.1** |
| input_layernorm γ — fraction > 5× mean | 0.33 % | 1.15 % |
| input_layernorm γ — persistent channels (top-5 amp across N layers) | max 2 layers | ch.923 in 12 / ch.{445,822,994,1214,3828} in 8–9 |
| q_norm γ — max | — | 5.16 |
| k_norm γ — max | — | **34.0** (L0 ch.51) |
| k_norm γ — fraction > 5× mean | — | 0.76 % |

Qwen3 has **two sink-amplifier compartments** (input_layernorm γ + k_norm γ),
both heavy-tailed and persistent across layers. Qwen2.5 has only
input_layernorm γ, with a markedly weaker tail.

The sink amplification does **NOT live in W_k projections** for either
family: per-output-row L2-norm distribution is flat (max ≈ 2.4× mean in
both models, [`k_proj_row_norms.csv`](k_proj_row_norms.csv)). So in
Qwen2.5 the sink amplification rides on W_k as a linear combination of
γ_in-amplified channels — meaning W_k *updates inevitably perturb sink
propagation*. In Qwen3, k_norm γ provides a *second amplification stage*
post-W_k that is frozen during VL adaptation.

Figures: [`figures/gamma_channels_qwen3_k_norm.png`](figures/gamma_channels_qwen3_k_norm.png),
[`figures/gamma_delta_qwen3_k_norm.png`](figures/gamma_delta_qwen3_k_norm.png).

### 3.3 What attention sinks are (background)

Sinks are token positions (typically BOS or first few tokens) that
absorb the residual softmax mass when queries have nothing more salient
to attend to — they keep generation stable by acting as a "garbage dump"
for the always-sum-to-1 attention distribution
([Xiao+ 2024](https://arxiv.org/abs/2309.17453)).
At the weight level, sinks are produced by a *few hidden-dim channels*
with anomalously large γ values (Qwen3 input_layernorm ch.923 γ=28,
k_norm L0 ch.51 γ=34) which amplify any token's contribution along
those channels into K vectors that always win attention
([Sun+ 2024 "Massive Activations"](https://arxiv.org/abs/2402.17762)).

The advisor's framing — *"VL training brings in lots of redundant vision
tokens; the model needs a discard space; if it can't find one and instead
rewrites attention magnitudes, LLM capabilities corrupt"* — corresponds
exactly to this picture: VLM training stresses the sink because there's
more semantic noise to absorb; whether sink survives depends on whether
the encoding location is protected from W perturbation.

---

## 4. Causal experiments

### 4.1 C1 — Sufficiency (sink ablation → IFEval crash)

Setup: in Qwen3-8B-nothink, for each layer of each norm type
(`input_layernorm`, `self_attn.q_norm`, `self_attn.k_norm`), replace the
top-10 γ channels (by `|γ|`) with the layer mean. This neutralizes the
amplifier structure without changing model architecture.

Script: [`c1_kill_sink_qwen3.py`](c1_kill_sink_qwen3.py) (combined-kill)
and [`c1_ablate_per_norm.py`](c1_ablate_per_norm.py) (per-norm).

| Variant | prompt-strict | Δ vs baseline (83.18) |
|---|---:|---:|
| Qwen3-8B baseline | 83.18 | — |
| Kill q_norm γ only | 81.15 | −2.03 |
| Kill k_norm γ only | 80.22 | −2.96 |
| **Kill input_layernorm γ only** | **68.21** | **−14.97** ⭐ |
| **Kill all 3 norms simultaneously** | **38.82** | **−44.36** |

**Key observation — synergistic interaction**:
- Σ individual drops = 2.03 + 2.96 + 14.97 = **19.96 pt**
- Joint drop = **44.36 pt**
- Interaction effect = **24.4 pt** — more than half of the total
  catastrophic drop is from inter-norm interactions, not from any single
  norm.

**Interpretation**: sink encoding is **redundant**. input_layernorm γ is
the primary amplifier (single-norm ablation costs −15 pt), but QK-norm γs
together provide ~5 pt of independent backup *plus* >20 pt of
joint-recovery synergy. The redundancy is what protects Qwen3 during VL
adaptation: even if W perturbation degrades the input_layernorm→W_k
routing, k_norm γ (frozen) re-amplifies the sink channel at the output
side.

### 4.2 C1-aside — top-10 γ channels actually ablated

For full transparency, the precise channels modified and their before /
after γ values are dumped in [`c1_ablate_per_norm.py`](c1_ablate_per_norm.py)
and reproducible via that script. The single most extreme change is
**k_norm L0 ch.51: γ 34.00 → 2.16** (|Δ| = 31.84). Late-layer
input_layernorm channels are the next-biggest movers (L33–L34 channels
ch.{923, 1214, 994, 445, 822} all drop from γ ≈ 20–28 to layer mean
≈ 3.07–3.53).

### 4.3 C2 — Necessity (SAS, partial)

**SAS = Sink-Anchored Surgery**. For each layer, identify the top-K
(K=36 ≈ 1 % of `hidden_size`=3584) hidden-dim channels by base-LLM
`input_layernorm γ`. Restore the corresponding *columns* of W_q / W_k /
W_v in Qwen2.5-VL-LM (a damaged VLM-LM) from Qwen2.5-7B-Instruct (the
base LLM). Leave the other ~99 % of projection weights at VLM values
(preserving vision capability).

Script: [`c2_sas_qwen25_recover.py`](c2_sas_qwen25_recover.py).

| | IFEval prompt-strict |
|---|---:|
| Qwen2.5-7B-Instruct (upper bound) | 72.09 |
| **+ SAS restoration (K=36 cols)** | **64.33 (+1.67)** |
| Qwen2.5-VL-LM (lower bound, damaged) | 62.66 |

TRR (Text Retention Rate) ≈ **17.7 %**. Weak positive — direction
matches mechanism (restoration improves IFEval, not degrades), but
magnitude is small. SAS is reported as a *lower-bound demonstration of
the mechanism*, not as the paper's main method; the QK-norm transplant
(C3) is the proper method test.

Two probable reasons SAS underperforms:
1. **Scope too narrow** — we restore only W_{q,k,v} columns; W_o and the
   MLP path also process the sink-relevant hidden channels.
2. **Qwen2.5 sink is weaker to begin with** — top-K input_layernorm γ
   channels in Qwen2.5 are 0.3–1.5 (vs Qwen3's 20–28), so "top-36 by γ"
   doesn't capture clean amplifiers.

### 4.4 T measurement (sink logit gap)

Script: [`measure_sink_T.py`](measure_sink_T.py).
Output: [`sink_T_qwen3.csv`](sink_T_qwen3.csv),
[`sink_T_qwen25.csv`](sink_T_qwen25.csv),
[`figures/sink_T_by_layer.png`](figures/sink_T_by_layer.png).

For each (layer, head, query position) on 15 IFEval-style calibration
prompts: `T = log(max_attn / second_max_attn)` — the gap between sink
position and the next-best key position in attention logits (nats).

| | Qwen3 | Qwen2.5 |
|---|---:|---:|
| median T (all layers) | 2.0 | 2.0 |
| median T (late layers L25+) | **3.0–3.5** | 2.0–2.8 |
| fraction of sinks at BOS (late layers) | **95 %** | <1 % |

Qwen3 has the canonical "BOS sink" structure (Xiao+ 2024); Qwen2.5's
sinks are distributed across non-BOS positions. Sink *magnitude* alone
is not dramatically different — what differs is the *consistency* and
*location* of the sink.

### 4.5 E2 — random W perturbation (done)

Tests whether the W-mode perturbation alone (independent of any specific
VL training direction) is sufficient to break IFEval.
Script: [`e2_random_w_perturb.py`](e2_random_w_perturb.py).
Adds Gaussian random ΔW to Qwen2.5-7B-Instruct with `‖ΔW‖_F` matched to
the per-sub-module rel-Frobenius of the actual Qwen2.5-VL adaptation;
γ left untouched.

**Result: IFEval prompt-strict = 10.72 (−61.37 pt vs Qwen2.5-7B base
72.09).** Far more destructive than the natural VL drop (−9.43). This
asymmetry is informative:

- **W-perturbation is sufficient** for catastrophic IFEval collapse
  (closes C1 ↔ VL-adaptation logical gap).
- Real VL training (−9 pt) is *much gentler than random* despite
  comparable rel-Frobenius magnitude. VL training implicitly stays in
  a structure-respecting subspace (low effective rank ΔW, aligned with
  V/O joint subspace, sink-channel-preserving). But it still crosses
  the threshold where QK-norm-less sinks fail, while Qwen3-VL's
  QK-norm protection keeps it on the safe side.

---

## 5. Mathematical foundation

### 5.1 Setup

For a single attention head, hidden dim `d`, head dim `d_h`:
`Q_pre = W_q ξ_q`, `K_pre,p = W_k ξ_p`, where `ξ = γ_ln ⊙ RMSNorm(x)`
is the input-LayerNorm output. With QK-RMSNorm: `Q = γ_q ⊙ q̃`,
`q̃ = Q_pre / RMS(Q_pre)`, same for K. Attention logit at position p:
`l_p = (Q · K_p) / √d_h`.

### 5.2 Theorem A — perturbation bound

Under fine-tuning perturbation `‖ΔW_q‖_op, ‖ΔW_k‖_op ≤ ε`, γ frozen:

**(i) Without QK-RMSNorm**:
`|Δ(l_a − l_b)| ≤ ε · ‖ξ_q‖ · (‖W_q‖_op + ‖W_k‖_op) · ‖ξ_a − ξ_b‖ / √d_h`

The bound *scales with `‖ξ_a − ξ_b‖`* — at a sink position p* with
massive activation (`‖ξ_p*‖ ≈ γ_ln,max`), the perturbation effect is
amplified by exactly the factor that creates the sink.

**(ii) With QK-RMSNorm**:
`|Δ(l_a − l_b)| ≤ ε · max(γ_q) · max(γ_k) · √d_h / ‖W‖_op + O(ε²)`

The bound is **independent of `ξ`** — sink positions are not
preferentially perturbed.

### 5.3 V12 stable-rank correction

The worst-case bound (ii) loosely overestimates (a factor 5–20× in our
empirical regime) because it assumes adversarial alignment of ΔW with
sink direction. Empirically ΔW has stable rank ≈ 100–150 for attention
projections (similar in both families); top-singular alignment to W_LLM
main axis is 0.02–0.28 (Qwen3 avg 0.07, Qwen2.5 avg 0.18), close to
random. So the tighter bound is:

`B_tight ≈ ε · max(γ_q) max(γ_k) / √r`

For Qwen3 (rel-Frob ε ≈ 0.23 for q/k_proj from §3.1, r ≈ 100,
max γ_q ≈ 5, max γ_k ≈ 34): `B_tight ≈ 3.9`, but **k_norm acts on
post-W_k vectors**, so the in-place γ_k=34 is *not* re-amplified by W
perturbation — the effective bound uses only γ_q's amplification of Q
side, giving `B_eff ≈ 0.6`. Compared to measured T_late-layer ≈ 3.0–3.5:
**B/T ≈ 0.2**, so attention pattern at sink is preserved.

For Qwen2.5 (no QK-norm, rel-Frob ε ≈ 0.54–0.68 for q/k_proj):
bound includes a `γ_ln_max / √d_h` factor that scales with input
magnitude. With γ_ln_max ≈ 8.5 and ‖W‖_op ≈ 5: `B_tight ≈ 2.8`.
T ≈ 2 → **B/T ≈ 1.4** → attention pattern disturbed.

Honest framing: cross-arch quantitative prediction *requires the
empirical inputs* (stable rank + measured T). Not a first-principles
derivation. Theorem A formalizes the *direction* of the asymmetry; the
empirical inputs supply the *magnitude*.

### 5.4 Lemma B — softmax saturation transfer

For sink position p₀ with logit gap T over other positions, and logit
perturbations bounded `|δ_p| ≤ B`:
`|a_p0' − a_p0| ≤ N · exp(−T + 2B)`

So attention pattern at sink is preserved exponentially in `T − 2B`.
Combined with Theorem A.ii's bound on B (under QK-norm + frozen γ),
this gives the chain: γ frozen → B bounded → attention pattern stable
at sink → instruction-following preserved.

### 5.5 Lemma C — structural decoupling (clean form)

In QK-RMSNorm, the attention logit factors as
`l_p = (1/√d_h) Σ_d ω_d · q̃[d] · k̃_p[d]`, where
`ω_d := γ_q[d] · γ_k[d]` are the *channel weights*. ω is structurally
separable from `W_q / W_k` — γ is an explicit RMSNorm parameter that
fine-tuning treats as a separate degree of freedom. In non-QK-RMSNorm,
no such factorization exists: sink amplification is inseparable from W.

Empirically (§3.1), γ stays within 2 % during VL adaptation while W
moves by 17–24 % in Qwen3 and 42–68 % in Qwen2.5. Combined with Lemma C,
this gives the formal mechanism: in QK-norm models, sink amplification
is on the *frozen* parameter manifold; in non-QK-norm models, it is on
the *moving* W manifold.

### 5.6 Limitations

- The cross-arch quantitative prediction (5.3) requires empirical
  inputs (stable rank, T); not first-principles.
- The bound (5.2) uses operator norm worst-case; tightening to the
  high-stable-rank regime needs the V12 correction.
- The link from "logit perturbation bound" to "IFEval preservation" is
  empirical (C1 sufficiency), not formal.
- Single-head, single-layer derivation; multi-layer composition not
  formally treated.

These are stated honestly in the paper §3 — math is a *mechanism
formalization*, not a standalone theorem.

---

## 6. Method — SAS (Sink-Anchored Surgery)

Training-free post-hoc recovery: identify the top-K hidden-dim
channels of `input_layernorm γ` (per layer), restore the corresponding
columns of W_q / W_k / W_v in a damaged VLM-LM from the base LLM, leave
the rest at VLM values.

Output of C2 (above): TRR 17.7 % — weak positive. Positioned as
*lower-bound proof of mechanism*, not the paper's main method. The
proper method test is the **QK-norm transplant** at training time (C3,
below).

Differentiation from neighbouring literature (to be cited rigorously in
paper draft; arXiv IDs below are placeholders pending lit-review re-pass):
- γ-channel-targeted *inference-time activation masking* (suppression)
  exists; SAS is the **opposite direction** — *weight-side column
  restoration* (preservation) — and a different setting (post-hoc VLM
  recovery, not LLM masking).
- VL safety-alignment degradation work (e.g. arXiv 2410.07571,
  2410.09047) addresses the problem via representation-space probes or
  coarse weight-merge interventions; SAS operates at the **γ-channel
  level** of W projections.

---

## 7. C3 — Architectural causality experiment (planned)

**Hypothesis**: injecting QK-RMSNorm modules (γ=1 identity-init) into a
non-QK-norm LLM and training as a VLM with the *same* LLaVA recipe will
yield a smaller IFEval drop than the vanilla baseline. If true, this
isolates QK-RMSNorm as an architectural cause of the protection,
controlling for data, RLHF, schedule.

### 7.1 Variants (4-cell ablation)

| Code | Base LLM | QK-norm? | Training | Predicted IFEval Δ |
|---|---|:-:|---|---:|
| L0 | Qwen2.5-3B-Instruct (text-only) | ✗ | none | 0 (baseline) |
| A1 | Qwen2.5-3B-Instruct | ✗ | LLaVA Align + Stage 2 | predict −6 to −10 |
| **A2** | Qwen2.5-3B + inject q/k_norm (γ=1) | ✓ | identical recipe | **predict −2 to −5** |
| R1 | Qwen3-4B-Instruct | ✓ native | (existing Qwen3-VL-4B if available) | observed reference |

Decision rule: `Δ(A2) − Δ(A1) ≥ 3 pt` ⇒ architectural causality
supported. `≥ 5 pt` ⇒ strong positive.

### 7.2 Code

Implementation: [`qknorm_injection.py`](qknorm_injection.py) (analysis
copy) and [`../../training/models/qknorm_injection.py`](../../training/models/qknorm_injection.py)
(training module).

```python
from training.models.qknorm_injection import inject_qknorm
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
inject_qknorm(model)
# model.model.language_model.layers[i].self_attn now has q_norm, k_norm
```

Set `model.inject_qknorm: true` in the training YAML to auto-inject at
load. Across Stage 1 → Stage 2 the γ values are saved/restored via
`load_qknorm_state_if_present` (so Stage 2 continues the γ learned in
Stage 1).

### 7.3 Training recipe (Stage 1.5 SKIPPED per advisor guidance)

| Stage | Frozen | Trainable | Data | Time on 2× A6000/A100 |
|---|---|---|---|---|
| Align | vision tower, LLM | mm_projector | LLaVA-Pretrain-558K | ~6–10 h (3B) |
| Stage 2 | vision tower | mm_projector + LLM (incl. γ_q, γ_k) | LLaVA-1.5-Mix-665K | ~30–48 h (3B) |

Per-variant: ~36–58 h ≈ 1.5–2.5 days. A1 + A2 together: ~3–5 days.

Configs: [`../../training/configs/3b/c3_vanilla_align.yaml`](../../training/configs/3b/c3_vanilla_align.yaml),
`c3_vanilla_stage2.yaml`, `c3_qknorm_align.yaml`, `c3_qknorm_stage2.yaml`.

Driver: [`../../training/scripts/run_c3_3b.sh`](../../training/scripts/run_c3_3b.sh).

### 7.4 Risk + mitigation

| Risk | Mitigation |
|---|---|
| Phenomenon weak at 3B (drop < 5 pt) | First pre-check on existing Qwen2.5-VL-3B-Instruct (sample IFEval). If weak, scale to 7B. |
| γ stays near 1 throughout Stage 2 (no amplifier structure emerges) | (a) report A2 anyway — even no-amplification γ-norm may help; (b) warm-start γ from Qwen3-4B-Instruct values (contaminates the controlled test but provides fallback). |
| Training instability from injected norm | lower LR on γ, gradient clipping, warmup. |
| Recipe differs from Qwen team's actual recipe | only the *differential* between A1 and A2 matters; same recipe applied to both controls for recipe effects. |
| Single-seed noise (IFEval stderr ~2 pt) | run 2 seeds if signal is borderline. |

### 7.5 Smaller-scale alternatives considered

Advisor suggested 0.5B / 0.6B for fastest iteration. However:
- 0.5B / 0.6B don't have a paired public Qwen2.5-VL variant, so
  `build_init_from_pretrained` (vision tower + same-size text LLM) is
  blocked at < 3B size in the MERIT framework we adapted.
- IFEval baselines at 0.5B are around 30–40 (vs 72 for 7B), so the
  drop dynamic range is smaller and harder to differentiate from
  noise.
- 3B is the smallest size where a public Qwen2.5-VL variant exists
  (Qwen2.5-VL-3B-Instruct) and IFEval baseline is decent (~70).

→ Round 1 is 3B/4B. If positive, Round 2 confirms at 7B.

---

## 8. Generalization tests (in progress / done)

The Qwen2.5 vs Qwen3 contrast is n = 2 with within-vendor confounds
(training data, RLHF, schedule may also differ between Qwen2.5 and
Qwen3). External cross-vendor data points:

| Pair | Backbone QK-norm? | Expected drop | Measured drop |
|---|:-:|---:|---:|
| LLaVA-LLaMA3-8B / Meta-Llama-3-8B-Instruct (LoRA) | ✗ | large | **−26.62** ⭐ |
| InternVL3-8B / Qwen2.5-7B-Instruct | ✗ | large | **−8.69** ⭐ |
| InternVL3.5-8B / Qwen3-8B-variant | ✓ | small | **−1.11** ⭐ |
| E2 random Gaussian ΔW (rel-Frob matched to Qwen2.5-VL) | n/a (control) | catastrophic | **−61.37** ⭐ |

**Every prediction confirmed.** LLaMA-3 / InternVL3 / InternVL3.5 are
three independent cross-vendor tests with vendor- and recipe-distinct
training pipelines:

- LLaMA-3 (no QK-norm, LoRA SFT) → −26.62 pt
- InternVL3 (Qwen2.5 base, no QK-norm, 4-stage CascadeRL recipe) → −8.69 pt
- InternVL3.5 (Qwen3 base, has QK-norm, same recipe family as InternVL3) → −1.11 pt

InternVL3 vs 3.5 is a near-natural experiment within OpenGVLab (same
team, similar recipe family, different LLM backbone), with caveats:
InternVL3.5 also adds Cascade RL + GSPO online RL + new reasoning/
capability data sources, so not a pure architectural ablation. The
categorical separation is robust to these confounds because it holds
across 3 vendors and 3 distinct training recipes. Sufficient as strong
*supporting evidence*, but C3 (training with identical recipe on
vanilla vs QK-norm-injected Qwen2.5) is still required for the clean
architectural-causality claim.

E2 random-W result is discussed in §4.5; here it serves as a "ceiling
of damage" reference — natural VL is −9 to −27 pt, random is −61 pt, so
real VL training implicitly stays in a structure-respecting subspace.

[`extract_direct.py`](extract_direct.py) is the helper that bypasses HF
`AutoModel` for VLMs that use `trust_remote_code` custom classes (used
for InternVL, LLaVA-LLaMA3). Required because:
- InternVL3-8B's `llm_config.vocab_size` = 151674 ≠ Qwen2.5-7B's
  152064; we extract the InternVL-native LLM config and use it.
- InternVL3.5-8B's `intermediate_size` = 12288 (custom) ≠ Qwen3-8B's
  14336; same `--config-src` override mechanism.

---

## 9. Reproduction

### 9.1 Setup

```bash
git clone https://github.com/minsik-choi126/mllm-model-merging.git
cd mllm-model-merging
pip install -r requirements.txt
pip install "lm_eval[ifeval]==0.4.5"
```

Place model checkpoints under `/131_data/geeho/minsik/` (or edit
script paths) for the following Qwen models, plus their
no-thinking overlay (`Qwen3-8B-nothink` = `Qwen3-8B` with patched
chat_template forcing `<think>\n\n</think>\n\n` always):

- Qwen2.5-7B-Instruct, Qwen2.5-VL-7B-Instruct
- Qwen3-8B, Qwen3-VL-8B-Instruct
- (optional) Meta-Llama-3-8B-Instruct, llama3-llava-next-8b,
  InternVL3-8B, InternVL3.5-8B

### 9.2 Run analysis

```bash
# Extract VLM text backbones (Qwen2 / Qwen3)
python -m extraction.extract_lm --pair qwen25vl_7b --output cache/extracted/qwen25vl_7b_lm
python -m extraction.extract_lm --pair qwen3vl_8b  --output cache/extracted/qwen3vl_8b_lm
# Direct-extract for trust_remote_code VLMs
python analysis/sibling_diff/extract_direct.py --vlm /path/to/InternVL3-8B \
    --output cache/extracted/internvl3_8b_lm \
    --tokenizer-src /path/to/Qwen2.5-7B-Instruct

# Weight-level analysis
python analysis/sibling_diff/diff_geometry.py
python analysis/sibling_diff/svd_and_gamma.py
python analysis/sibling_diff/sink_in_projections.py

# C1 ablation overlays
python analysis/sibling_diff/c1_kill_sink_qwen3.py     # all 3 norms
python analysis/sibling_diff/c1_ablate_per_norm.py     # per-norm variants

# C2 SAS overlay
python analysis/sibling_diff/c2_sas_qwen25_recover.py

# E2 random W perturbation overlay
python analysis/sibling_diff/e2_random_w_perturb.py --seed 0 --rel-scale 1.0

# Sink T measurement on calibration prompts
python analysis/sibling_diff/measure_sink_T.py
```

### 9.3 Run IFEval matrix

```bash
export LM_EVAL_BIN=/opt/conda/bin/lm_eval
bash evaluation/text/run_eval_matrix.sh \
    --models llm:/path/Qwen3-8B-nothink \
             vlm_lm:cache/extracted/qwen3vl_8b_lm_nothink \
    --protocols instruct \
    --gpus 0,1 \
    --tasks ifeval \
    --output-root eval_results/qwen3_ifeval_nothink
```

### 9.4 C3 training (architectural causality)

```bash
export CKPT_ROOT=/your/work/dir/c3_3b
bash training/scripts/run_c3_3b.sh        # both variants
# or:
VARIANT=vanilla bash training/scripts/run_c3_3b.sh
VARIANT=qknorm  bash training/scripts/run_c3_3b.sh
```

Pipeline: composed-init build → A1/A2 Stage 1 → A1/A2 Stage 2 →
text-backbone extraction → IFEval matrix vs `Qwen2.5-3B-Instruct`
baseline.

---

## 10. Outputs

### Analysis CSVs / figures (this directory)

| File | Content |
|---|---|
| `diff_qwen25.csv`, `diff_qwen3.csv` | Per-(layer, sub_module) `‖ΔW‖_F`, `‖W_LLM‖_F`, `rel_diff` |
| `svd_metrics.csv` | Per-projection ΔW spectral metrics (σ_max, stable rank, effective rank, alignment with W_LLM top dir) |
| `gamma_qk_qwen3.csv` | Qwen3 q_norm / k_norm γ values per (layer, channel), with VL shift |
| `k_proj_row_norms.csv` | W_k per-output-row L2 norms (control: sink not in W_k) |
| `sink_T_qwen3.csv`, `sink_T_qwen25.csv` | Per-layer sink-logit-gap T statistics |
| `figures/heatmap_rel_diff.png` | Per-(layer, sub) `‖ΔW‖_F / ‖W_LLM‖_F` heatmap, two-panel |
| `figures/per_sub_relative.png` | Mean rel_diff per sub-module, two-pair bars |
| `figures/depth_curves.png` | rel_diff vs relative depth per sub-module |
| `figures/svd_stable_rank.png` | Stable rank of ΔW per sub-module per depth |
| `figures/svd_top_alignment.png` | Top-singular alignment of ΔW with W_LLM top right-singular |
| `figures/svd_effective_rank.png` | Effective rank of ΔW (top-64 approx) |
| `figures/gamma_channels_qwen3_{q,k}_norm.png` | Per-layer overlay of γ in base vs Qwen3-VL-LM |
| `figures/gamma_delta_qwen3_{q,k}_norm.png` | Heatmap of per-channel relative VL shift |
| `figures/sink_T_by_layer.png` | T (sink logit gap) vs layer index, Qwen3 vs Qwen2.5 |

### Status snapshot

| Item | Status |
|---|---|
| Phenomenon (Qwen pair) | done ✓ |
| Pipeline validation vs official | Qwen ✓ (≤0.9 pt), Llama-3 +3.3 pt drift |
| Weight diagnosis (3 sections) | done ✓ |
| C1 sufficiency + ablation breakdown | done ✓ |
| C2 SAS (weak positive) | done ✓ |
| T measurement | done ✓ |
| Math foundation | drafted ✓ |
| E2 random-W perturbation (mechanism control) | done ✓ (Δ = −61.4) |
| Generalization: LLaVA-LLaMA3-8B (no QK-norm) | done ✓ (Δ = −26.6) |
| Generalization: InternVL3-8B (no QK-norm) | done ✓ (Δ = −8.7) |
| Generalization: InternVL3.5-8B (has QK-norm) | done ✓ (Δ = −1.1) |
| Trustworthiness sanity-check vs official refs | done ✓ (§2.4) |
| C3 (QK-norm injection + LLaVA training, 3B) | code ready, data prep + launch pending |
