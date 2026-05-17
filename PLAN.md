# Plan — extending the text-degradation study to a defensible paper

> **Status note (2026-05-17)**: The project has bifurcated. The original
> E-Pull-centric plan (Sections 1–4 below) is on hold. The current active
> trajectory is the **sink-mechanism investigation** documented in
> [`analysis/sibling_diff/`](analysis/sibling_diff/) and summarized at the
> bottom of this file (**"Active trajectory: sink mechanism"**). The sibling
> pair Qwen3 / Qwen2.5 contrast, the weight-side diagnosis, and the two
> causal experiments (sink ablation, sink restoration) are the new core
> story.

The Stage A primary measurement (Qwen2.5-VL-7B → text backbone vs LLM-Instruct,
+ E-Pull merge) is in progress. Three open gaps need closing before the
study is paper-ready: **phenomenon coverage, absolute-number sanity, and
baseline positioning**. This document tracks them.

---

## 1. Phenomenon coverage — open-recipe models

### Why
Qwen2.5-VL-7B and Phi-3.5-Vision recipes are closed. LLaVA-LLaMA3-8B is the
only open-recipe model in the current set, and it's LoRA-trained, which
makes its degradation pattern partially reversible by not merging the LoRA
adapter back. To show the phenomenon is *general* (not an artifact of any
one training stack) we need at least one more **fully-open-recipe, full-FT**
data point.

### What

#### 1a. Add LLaVA-OneVision family
LLaVA-OneVision (Li et al. 2024, [arXiv:2408.03326](https://arxiv.org/abs/2408.03326))
- Backbone: Qwen2-7B (not Qwen2.5)
- Recipe + data fully open
- Frontier-ish scale, but small enough to extract and evaluate locally
- Concrete model: `lmms-lab/llava-onevision-qwen2-7b-ov` (or `-si`)

Action:
- Add registry entry `llava_onevision_qwen2_7b` in `extraction/models.yaml`
  - vlm: `lmms-lab/llava-onevision-qwen2-7b-ov`
  - llm: `Qwen/Qwen2-7B-Instruct`
- Verify the extraction pipeline handles its key prefixes (likely
  `model.language_model.*`-style; if different, extend
  `extraction.loader.normalize_text_backbone_state_dict`)
- Run text-degradation eval and add to `results/text_degradation.md`

#### 1b. LLaVA-LLaMA3 — clarify which checkpoint
The current registry uses `lmms-lab/llama3-llava-next-8b` (LLaVA-NeXT
revision). The user-suggested reference is `xtuner/llava-llama-3-8b`
(canonical LoRA recipe). These are different training runs.

Action:
- Either swap the registry entry to `xtuner/llava-llama-3-8b` (closer to the
  canonical "open LoRA recipe") **or** add a second pair
  `llava_llama3_8b_xtuner` and report both
- For the LoRA-trained one, also evaluate the **un-merged-LoRA** variant
  (base LM as-is, LoRA at inference only): text degradation should be zero
  there *by construction*, vision quality should be lower than full FT.
  This is itself a baseline (see §3b).

#### 1c. Per-model recipe metadata
For each pair, record in `models.yaml`:
- Training data scale (text-only vs image-text token counts)
- LM tuning regime (full FT / LoRA-rank / frozen)
- Connector type (MLP / Q-Former / cross-attn)

Helps interpret per-model degradation patterns and predict where E-Pull
should help most.

---

## 2. Eval sanity — absolute-number reconciliation, multi-framework

### Why
Multi-framework reproductions are how the paper Appendix defends "we tried
several conventions, here's what each produced, here's why they differ".
The diligence itself is a paper asset; field-wide framework drift is well
known.



### Why
Our current numbers are reported as deltas, but the absolute LLM scores
must be defensible against external references. lm-eval-harness 0.4.5
systematically underestimates Qwen-blog numbers by 0–13 pt due to framework
drift (OpenCompass / TIGER / Google-IFEval-reference vs lm-eval). We need
to (i) document the drift and (ii) verify our pipeline against an
independent reference (Open LLM Leaderboard v2).

### What — current state

| Task | Qwen blog | OLL v2 | Ours | Verdict |
|---|---:|---:|---:|---|
| gsm8k_cot | 91.6 | — (not in v2) | 78.92 | framework drift; cannot reference OLL |
| ifeval prompt-strict | 71.2 | — (avg-of-4 only) | 72.09 | within ±1 pt of Qwen blog ✓ |
| ifeval avg(strict) | — | **75.85** | 75.80 | matches OLL ✓ |
| gpqa cot 0-shot | 36.4 | **29.11** | 29.29 | matches OLL exactly ✓ |
| mmlu_pro | 56.3 | **42.87** | _pending_ | use OLL as cross-check |
| BBH | — | 53.94 | (we don't run) | OLL only |

### Action

- [x] Run **two protocols in parallel** (community-default vs instruct-aware) on all 3 models
      → `eval_8tasks.sh --protocol {default,instruct}` queue
- [x] Write `evaluation/text/protocol_audit.md` documenting:
  - per-task lm-eval flags + n-shot + chat-template choice (per protocol)
  - reference numbers (Qwen blog, OLL v2)
  - the framework-drift gap with sources
- [ ] **OpenCompass cross-check** for the 3 tasks with biggest current drift
  (`gsm8k_cot`, `gpqa_diamond_cot_zeroshot`, `mmlu_pro`). Goal: produce a
  third column ("OpenCompass on our hardware") in the audit tables. If
  OpenCompass reproduces Qwen-blog numbers within 1-2 pt, the framework
  drift hypothesis closes — paper Appendix gets a clean triangle (Qwen
  blog ↔ OLL v2 ↔ ours).
- [ ] Add a `Qwen/Qwen2.5-7B` (base, **non-Instruct**) baseline run too — lets
  us decompose `(VLM-LM - LLM-Instruct)` into
  `(VLM-LM - base) - (LLM-Instruct - base)`
- [ ] In paper, frame absolute numbers as "reproducible under lm-eval-harness
  0.4.5 under two protocols (community-default and instruct-aware), with an
  OpenCompass cross-check on the most-drift-prone tasks. Where Qwen-published
  numbers (OpenCompass) and ours (lm-eval) differ, the gap is reproducible
  framework drift, not a setup bug."

---

## 3. Baselines — what E-Pull must position against

This is the most consequential gap. Without these, a reviewer can dismiss
E-Pull as a solution chasing a problem that practitioners already handle by
simpler means.

### 3a. Joint text-SFT during VL training
*The first thing practitioners try.*

References:
- VILA: On Pre-training for Visual Language Models ([arXiv:2312.07533](https://arxiv.org/pdf/2312.07533))
- Cambrian-1 ([arXiv:2406.16860](https://arxiv.org/pdf/2406.16860))

Both papers explicitly study text-SFT mixing ratios.

**Plan**:
- Use a small base — Qwen2-1.5B or LLaMA-3.2-1B
- Run VL adaptation with text-SFT ratios `{0%, 5%, 10%, 25%, 50%}` of token
  budget
- Measure (text-degradation, vision-quality) per ratio → pareto curve
- Argument we want to support:
  - mixing text-SFT *can* preserve text, but
  - silent regressions on procedural tasks (GSM8K, IFEval) persist even at
    high ratios — i.e., joint SFT doesn't fully solve the problem; AND
  - high text-SFT ratios trade off against vision capability — there's no
    free lunch, which is exactly the gap E-Pull (post-hoc, no retraining)
    fills

**Compute estimate**: 1.5B model × ~1B image-text tokens × 5 settings ≈
~5–10 A6000-days. Plan a small mock-budget run first; scale up if needed.

### 3b. Frozen-LM / LoRA-only training
*The pre-2024 mitigation.*

References:
- VILA (also discusses LM-frozen vs LM-tuned)
- LLaVA LoRA recipe ([github](https://github.com/haotian-liu/LLaVA/blob/main/docs/LoRA.md))
- Frozen-backbone VLM ablation ([OpenReview 1tZbq88f27](https://openreview.net/forum?id=1tZbq88f27))

**Plan**:
- For LLaVA-LLaMA3-8B (LoRA): evaluate **without** merging LoRA into base —
  text scores should match Llama-3-8B-Instruct exactly (sanity check); vision
  scores will be lower than full-FT. This is a baseline for free.
- Optionally train one frozen-LM VLM at small scale and compare.
- Argument: LoRA-only / frozen-LM avoid degradation **by construction**, but
  Cambrian-1 shows they don't reach full-FT vision quality. E-Pull is
  positioned in between — full-FT vision + recovered text.

### 3c. Model-merging baselines
*The actual peers of E-Pull.*

References:
- RegMean (Jin et al. 2023) — what E-Pull degenerates to under uniform routing
- Task arithmetic, TIES, DARE — generic merging
- Naive average `(LLM + VLM-LM) / 2`

**Plan**:
- Implement these as `method/baselines/*.py` (RegMean we can derive from
  the same covariance pipeline; others are weight-only)
- Run on the same Qwen2.5 pair, measure text retention + vision retention
- Required claim for the paper: E-Pull strictly dominates RegMean and naive
  averaging on the (text, vision) frontier

### 3d. What E-Pull must beat
| Baseline | Text retention | Vision retention | Cost |
|---|---|---|---|
| Naive avg | low | medium | free |
| RegMean | medium | medium | per-task covariance |
| TIES / DARE | medium | medium | free |
| Joint text-SFT, low-ratio | medium-high | high | retraining |
| Joint text-SFT, high-ratio | high | medium | retraining |
| LoRA-only / frozen-LM | high (free) | low-medium | retraining |
| **E-Pull (ours)** | **high** | **high (= full FT)** | **per-task covariance, no retraining** |

The story we want: **E-Pull achieves the (text-high, vision-high) corner
post-hoc, without any retraining and without the silent regressions that
joint-SFT has**.

---

## 4. Concrete experiment stages

### Stage A — primary measurement (in progress)
- [x] Implement E-Pull (orthogonal FG joint diag + entropy gate + closed form)
- [x] 8 self-tests at machine precision (closed-form identities, regression,
      symmetry, validation)
- [x] Merge: `Qwen2.5-7B-Instruct + extracted VLM-LM(qwen25vl_7b) → epull`
- [ ] LLM / VLM-LM / merged 8-task eval (running)
- [ ] Vision eval on base VLM
- [ ] Vision eval on "merged text backbone spliced into VLM" (the actual
      published artifact; see notes in §A.1 below)

### Stage B — extend phenomenon coverage
- [ ] Add `llava_onevision_qwen2_7b` registry entry
- [ ] Download + extract LLaVA-OneVision text backbone
- [ ] LLaVA-OneVision text-degradation 8-task
- [ ] LLaVA-OneVision E-Pull merge + 8-task
- [ ] (optional) re-run LLaVA-LLaMA3 with `xtuner/llava-llama-3-8b`
- [ ] Update `results/text_degradation.md` with both new rows

### Stage C — joint text-SFT ablation
- [ ] Pick small base (Qwen2-1.5B / LLaMA-3.2-1B)
- [ ] Set up VL adaptation training pipeline (LLaVA-style) with configurable
      text-SFT ratio
- [ ] Run 5 ratios × eval × compute pareto
- [ ] Document in `experiments/joint_sft/`

### Stage D — LoRA / frozen-LM
- [ ] Evaluate LLaVA-LLaMA3 with LoRA un-merged (free baseline)
- [ ] (optional) train frozen-LM VLM at small scale
- [ ] Document in `experiments/lora_frozen/`

### Stage E — model-merging baselines
- [ ] Implement RegMean cleanly (already derivable from covariance pipeline)
- [ ] Implement TIES, DARE, naive avg (weight-only)
- [ ] Run all on Qwen2.5 pair
- [ ] Add to `method/baselines/` + `results/comparison.md`

### Stage F — writeup
- [ ] Section 4: phenomenon (Stages A + B)
- [ ] Section 5: method (already drafted)
- [ ] Section 6: experiments
  - 6.1 main result: E-Pull vs all baselines on Qwen2.5-VL
  - 6.2 generalization: LLaVA-OneVision (Stage B)
  - 6.3 baseline comparisons: text-SFT pareto + LoRA / frozen + merging baselines
  - 6.4 ablation: jacobi sweeps, alpha, calibration size
- [ ] Section 7: discussion / limitations

---

## A. Notes / risks

### A.1 Vision-side experiment scope
The merged artifact in Stage A is a *text-only* model. To claim "E-Pull
preserves vision capability" we have to splice the merged text-backbone
weights back into the VLM (replacing the original VLM's text-tower) and
run the 6-task vision eval on that spliced model. The
`evaluation/vision/` infrastructure handles this once we write a tiny
splicer.

### A.2 Compute budgets (rough)
- Stage B: 1× 7B extract + 1× E-Pull merge + 14 evals ≈ 1 GPU-day
- Stage C: 5× small-model trainings ≈ 5–10 GPU-days
- Stage D: optional frozen-LM training, 1–2 GPU-days
- Stage E: 4 baselines × 1 merge each + evals ≈ 1 GPU-day
- Total above-and-beyond Stage A: ~10–15 GPU-days

### A.3 Framework-drift caveat in the paper
All scores in the paper will be reported under **lm-eval-harness 0.4.5**
with the per-task protocols in `evaluation/text/eval_8tasks.sh`. The known
0–13 pt drift vs Qwen blog is documented but not corrected — chasing
absolute parity with OpenCompass would obscure the LLM↔VLM-LM↔merged
deltas, which are what the paper claims. Where OLL v2 has a number, we
cross-check (currently: gpqa exact match, ifeval avg-of-strict matches).

---

## Active trajectory: sink mechanism (2026-05-17 onward)

The discovery that **Qwen3-VL preserves IFEval (Δ ≈ −3 pt) while Qwen2.5-VL
collapses (Δ ≈ −9 pt)** under matched protocols led to a reframing: the
paper's central scientific claim is now about **why** this asymmetry exists,
not just **how** to merge text capability back.

### Thesis

> *Instruction-following loss in VLMs is mechanistically an attention-sink
> corruption problem. VL training perturbs the small subset of weights that
> carry sink-amplifier signal; QK-RMSNorm architecturally isolates that
> signal into γ layers that VL updates leave frozen.*

### Paper arc

1. **Phenomenon** — Qwen3 vs Qwen2.5 cross-vendor natural experiment.
2. **Diagnosis** — weight magnitude (3× larger ΔW in Qwen2.5), spectrum
   (high-rank, *not* low-rank as LoRA assumes), sink encoding location
   (input_layernorm γ + k_norm γ in Qwen3; input_layernorm γ only and
   weaker in Qwen2.5; *not* in W_k projection in either).
3. **Method (SAS — Sink-Anchored Surgery)** — post-hoc, training-free,
   single-hyperparameter weight surgery: restore the top-K hidden-dim
   columns of W_q / W_k / W_v (selected by input_layernorm γ) from the base
   LLM. ~1% of weights touched, vision-side weights otherwise intact.
4. **Causal evidence** — three experiments. See
   [`analysis/sibling_diff/README.md`](analysis/sibling_diff/README.md).

### Stages

| | Status |
|---|---|
| **S1**. Confirm Qwen3-VL no-degradation (IFEval, thinking-off) | Done (80.22 vs 83.18; Δ −2.96 pt) |
| **S2**. Weight-side diagnosis (4 figures, two CSVs) | Done |
| **S3**. C1 sufficiency: kill γ amplifiers in Qwen3 → IFEval | Running |
| **S4**. C2 necessity: SAS-restore Qwen2.5-VL-LM → IFEval | Built; pending GPU |
| **S5**. C3 architectural causality: QK-norm inject + small VL train | Designed |
| **S6**. Generalize SAS to Phi-3.5-Vision, LLaVA-OneVision, InternVL | Pending |
| **S7**. Method baselines: TIES/DARE/RegMean/E-Pull on the IFEval recovery task | Pending |
| **S8**. Writeup | Pending |

### Novelty position vs neighbouring work

Documented in detail in
[`analysis/sibling_diff/README.md`](analysis/sibling_diff/README.md#method-post-hoc-recovery-sas--sink-anchored-surgery).
Headline differentiation: prior work using RMSNorm γ
([WeMask / arXiv 2605.08504](https://arxiv.org/abs/2605.08504)) does
*activation masking at inference time, in the suppress direction, for
general LLM improvement*. SAS does *weight-side column restoration,
post-hoc, in the preserve direction, for VLM-LM recovery*.
