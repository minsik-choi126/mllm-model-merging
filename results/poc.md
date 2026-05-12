# Proof-of-concept log

This document records, in chronological order, everything we have actually
implemented and run for the "post-hoc merging recovers VLM text-side
capability" project, the empirical results we obtained, the failure modes
we hit, and what we plan to try next.

This is the engineering / experiment log; the paper-facing tables live in
[`results/text_degradation.md`](text_degradation.md). For the eval-protocol
audit (frameworks, shot counts, chat-template choices, sources for every
public reference number we use), see
[`evaluation/text/protocol_audit.md`](../evaluation/text/protocol_audit.md).
The research-plan rationale lives in [`PLAN.md`](../PLAN.md).

---

## 1. Method implementation (E-Pull)

E-Pull = direction-wise constrained-Pareto merging on the orthogonal common
eigenbasis of the per-modality input covariances. Closed-form interpolation
between RegMean (uniform routing, `g_j = 0`) and direction-wise winner-take-all
(one-hot routing, `g_j = 1`). The gate is parameter-free and derived from
the per-direction modality entropy.

### What's in `method/`

| file | role |
|---|---|
| `covariance.py` | Per-linear-layer input-Gram collection. Trace-normalises and (now) collects full Gram for `down_proj` too. CPU-side accumulation to avoid OOM at intermediate-size dims. |
| `joint_diag.py` | Orthogonal FG joint diagonalization. Pipeline = (k=2) generalized eigvec via Cholesky whitening → polar projection onto the orthogonal manifold → CS-Jacobi sweeps to descend the FG cost. Adaptive `ε` retry on ill-conditioned Cholesky at large d. |
| `merge.py` | Per-layer E-Pull closed form (`m_j^**` = (1-g) m_j^* + g u_{r*}^j), reassembly via `M V^T`, stat collection. Owner-energy fallback for layers whose Gram couldn't be collected. |
| `cli.py` | End-to-end driver: load → calibrate → merge → save. |
| `_self_test.py` | Eight closed-form / regression / symmetry / validation checks at machine precision. |

### Reviewer-round fixes (chronological)

E-Pull went through four rounds of independent review (reviewer played
adversary; I implemented and verified each round).

**Round 1.** sum-PCA `V = eigvecs(Σ α_i C_i)` fails under exact CPC when the
weighted average is degenerate — `eigh` of a constant-scaled identity
returns an arbitrary basis. Concrete counterexample: random orthogonal
`V_true`, `Λ_1`, `Λ_2` chosen so `0.5 Λ_1 + 0.5 Λ_2 = c·I` → sum-PCA's
`V^T C_i V` had ‖off-diag‖_F / ‖·‖_F = 0.665 and merged ΔW relative error
19.6 % vs ideal. **Fix:** for k = 2, generalized eigvec via Cholesky
whitening; relative error dropped to 6 × 10⁻⁷ on the same counterexample.

**Round 2.** Generalized eigvec produces a non-orthogonal `V`; the paper's
closed-form theorem relies on `V^T V = I`. **Fix:** polar projection
`V = polar(V_gen)`. Under exact CPC `V_gen = V_true Λ_2^{-1/2}` so
`polar(V_gen) = V_true` exactly (verified). Added a separate
`commutator_residual = ‖C_1 C_2 − C_2 C_1‖_F / (‖C_1‖_F‖C_2‖_F)` because the
old `fg_residual` is uninformative for k = 2 SPD pairs (always near machine
zero post-gen-eig). New `test_paper_loss_consistency` verifies the
per-direction loss formula `L_r(W) = Σ_j λ_{r,j}‖m_j − u_r^j‖²` matches the
trace form on the orthogonal V.

**Round 3.** CS-Jacobi angle had a factor-of-2 bug. Coordinate-space Givens
by θ corresponds to a rotation by 2θ in the `(a, b)` parametrization, so
the correct formula is `θ = 0.25 · atan2(2·Sxy, Sxx − Syy)`, not `0.5·`.
Verified by a 1D brute-force grid: buggy θ overshoots optimum by 2×, leaves
FG cost flat across sweeps; with the corrected θ the FG cost drops
monotonically (0.150 → 0.087 → 0.069 → 0.061 → 0.0606 over 10 sweeps on a
non-CPC synthetic pair).

**Round 4.** `EpullConfig.jacobi_sweeps` was being parsed but not passed
through to `joint_diagonalize`, so the matrix always defaulted to 2 sweeps
regardless of the CLI flag. **Fix:** propagate, plus `--jacobi-sweeps`
CLI flag, plus `jacobi_sweeps ≥ 0` validation. Also added `jacobi_max_d`
because parallel-round Jacobi at `d = 18944` (intermediate size) would
take ≈ 30 hours per layer; we skip Jacobi above the threshold and rely on
polar(gen-eig) alone for `down_proj`-scale matrices.

### Verification

`python -m method._self_test` passes 8/8 at machine precision after all
fixes:

```
test_limit_uniform               0.000e+00
test_limit_onehot                3.3e-16   (owner-take-all limit)
test_self_improve_dominant       7.2e-15   (per-direction theorem)
test_aggregate_cost              2.0e-14   (aggregate-cost theorem)
test_degenerate_cpc_avg          1.6e-7    (round-1 regression)
test_paper_loss_consistency      3.7e-16   (round-2 trace ↔ per-dir form)
test_modality_swap_symmetry      2.0e-15   (under exact CPC)
test_validation                  all guards raise
```

---

## 2. Extraction infrastructure

`extraction/extract_lm.py` reads a VLM checkpoint and writes a text-only
HF-format model containing only the language-backbone weights.

**Normalization** (`extraction/loader.py:normalize_text_backbone_state_dict`):
the four VLM families we support use different key prefixes for the same
parameters —
- `model.language_model.*` (Qwen2.5-VL, Qwen3-VL)
- `language_model.model.*` (LLaVA-NeXT family)
- `language_model.output.weight` (InternLM2 LM head)
- already-normalized text keys (LLaVA-OneVision-Qwen2, base LLMs)

— all rewritten to the standard HF causal-LM schema
(`model.embed_tokens / model.layers / model.norm / lm_head`).

**Direct-safetensors fallback** (added during LLaVA-OneVision extraction):
LLaVA-OneVision-Qwen2 ships with a custom `LlavaQwenForCausalLM`
architecture that is not registered in `transformers`. `AutoModel.from_pretrained`
raises a vocab-size mismatch (LLaMA-LLaVA's 32000 × 4096 vs Qwen2's
152064 × 3584). When the AutoModel path raises, the loader now falls back
to a direct safetensors read (`safetensors.safe_open`) of the local
checkpoint directory and feeds the resulting raw state dict through the
same key-normalization step. The pair config also pins a local path
(`/131_data/geeho/minsik/llava-onevision-qwen2-7b-ov`) because
`resolve_local_pretrained_path` couldn't find a HF cache snapshot for
this checkpoint.

**Safetensors metadata** (`format=pt`): early checkpoints we saved didn't
include this metadata, which caused HF's loader to reject them with
`Incompatible safetensors file. File metadata is not [...] but None`. The
loader now always injects `format=pt`; the saved extracted backbones are
loadable by every downstream `AutoModelForCausalLM.from_pretrained` call.

**Successful extractions to date** (under `/131_data/geeho/minsik/extracted/`):
- `qwen25vl_7b_lm` (339 tensors, 15 GB)
- `llava_onevision_qwen2_7b_lm` (339 tensors, 15 GB)

---

## 3. Eval infrastructure

### Two protocols, one driver

`evaluation/text/eval_8tasks.sh --protocol {default, instruct}` runs the
eight-task standard set, one `lm_eval` invocation per task (the per-task
protocol differs, so we can't use a single mega-call).

| | community-default | instruct-aware |
|---|---|---|
| gsm8k_cot | 8-shot CoT, **no chat template** | 0-shot CoT, **chat template ON** |
| ifeval | 0-shot, no chat template | 0-shot, chat template ON |
| gpqa_diamond_cot_zeroshot | 0-shot CoT, no chat template | 0-shot CoT, chat template ON |
| mmlu_pro | 5-shot CoT, no chat template | 0-shot CoT, chat template ON |
| mmlu | 5-shot LL, no chat template | 0-shot LL, chat template ON |
| boolq | 0-shot LL, no chat template | 0-shot LL, chat template ON |
| truthfulqa_mc2 | 0-shot LL, no chat template | 0-shot LL, chat template ON |
| eq_bench | 0-shot, no chat template | 0-shot, chat template ON |

The instruct protocol is the principled one for evaluating chat-templated
post-trained models. The community-default protocol matches lm-eval-harness
yaml defaults and is what the broader community (incl. OLL v2) actually
reports — so we publish both, and the LLM ↔ VLM-LM gap *between* the two
protocols is itself a finding (see §5.2).

### Matrix orchestrator

`evaluation/text/run_eval_matrix.sh` runs N models × M protocols on K GPUs
in round-robin, idempotently skipping any task that already has a results
JSON. It survives mid-run kills: re-launching the same command picks up
where it stopped. The per-protocol comparison tables are auto-emitted by
`parse_results.py` after all chains drain.

### Cross-framework cross-check

For Qwen2.5-7B-Instruct (the LLM modality of the Qwen2.5-VL pair) we ran
**OpenCompass** on the same checkpoint and same hardware to establish the
framework-drift baseline. Config: `evaluation/text/opencompass_crosscheck.py`.

| Task | Qwen blog | Our OpenCompass | Δ |
|---|---:|---:|---:|
| GSM8K | 91.6 | **90.67** | −0.93 |
| MMLU-Pro (14-cat avg) | 56.3¹ | **55.84** | −0.46 |
| GPQA-Diamond | 36.4 | (judge-eval failed offline) | — |

¹ The 56.3 number was cited as Qwen-blog mmlu_pro in our scoping survey;
the current Qwen2.5-LLM blog table shows 45.0 for the 7B-Instruct row,
so 56.3 might be from the Qwen2.5-VL paper's text-side comparison
table. Both reproductions are within rounding of *some* published Qwen
number, so the cross-check verdict ("OpenCompass on our hardware ≈
Qwen-published, within 1 pt") stands either way.

The 5–13 pt gap on the same tasks between this OpenCompass reproduction
and our lm-eval-harness reproduction is real, reproducible, and
attributable to framework choice (prompt template, few-shot selection,
answer-extraction regex). It is NOT a setup bug on our side.

---

## 4. Experiments completed

### 4.1 Qwen2.5-VL-7B pair — full pipeline

| stage | status |
|---|---|
| Extract VLM-LM from Qwen2.5-VL-7B-Instruct | ✓ 339 tensors |
| LLM-Instruct 8-task eval × 2 protocols | ✓ |
| VLM-LM 8-task eval × 2 protocols | ✓ (gpqa missing — see §6) |
| E-Pull merge (Qwen2.5-7B base, α=0.5) | ✓ 196 layers, all epull mode, avg gate 0.039, avg FG cost 0.249, avg jacobi sweeps 1.71, all `down_proj` via polar(gen-eig) |
| Merged-model coherence smoke test | partial (correct short answers, autoregressive collapse to repetition past ~10 tokens) |
| Epull 8-task eval × 2 protocols | ✓ |

Absolute scores live in [`text_degradation.md`](text_degradation.md); the
headline numbers (instruct protocol):

|  | LLM | VLM-LM | epull | Δ epull vs LLM |
|---|---:|---:|---:|---:|
| GSM8K-CoT | 78.92 | 77.94 | 56.03 | **−22.9** |
| IFEval (prompt-strict) | 72.09 | 62.66 | 28.28 | **−43.8** |
| GPQA-Diamond-CoT | 29.29 | — | 9.60 | −19.7 |
| MMLU-Pro | 57.17 | 51.47 | 18.07 | **−39.1** |
| MMLU | 69.72 | 67.63 | 70.10 | **+0.4** |
| BoolQ | 85.93 | 84.77 | 85.38 | −0.5 |
| TruthfulQA-MC2 | 62.44 | 57.40 | 56.18 | −6.3 |

The merged model **preserves multi-choice LL tasks** (MMLU, BoolQ; the
LLM-trained-token-ranking ability is robust to weight averaging) but
**catastrophically loses long-generation tasks** (GSM8K, IFEval, MMLU-Pro;
autoregressive coherence dies when the merge effectively averages two
sets of unaligned task vectors). See §5.3 for the diagnosis.

### 4.2 LLaVA-OneVision-Qwen2-7B-OV pair — phenomenon measurement

Same pipeline as Qwen2.5-VL, on a fully-open-recipe full-FT VLM (paper:
arXiv:2408.03326, training data publicly released).

| stage | status |
|---|---|
| Download (Qwen2-7B, Qwen2-7B-Instruct, LLaVA-OneVision-Qwen2-7B-OV) | ✓ 15 GB × 3 |
| Extract VLM-LM (via safetensors-direct fallback) | ✓ 339 tensors |
| LLM-Instruct 8-task eval × 2 protocols | ✓ |
| VLM-LM 8-task eval × 2 protocols | ✓ |
| E-Pull merge | **paused at user request, will resume** |

Headline degradation numbers (Δ = VLM − LLM, points):

| | instruct protocol | default protocol |
|---|---:|---:|
| MMLU | −2.2 | −3.1 |
| MMLU-Pro | **−14.5** | −9.8 |
| GSM8K-CoT | −6.7 | −3.3 |
| TruthfulQA-MC2 | −12.2 | −11.0 |
| BoolQ | −4.6 | +2.1 |
| IFEval (prompt-strict) | −9.2 | **−39.6**¹ |
| GPQA-Diamond-CoT | −7.6 | **−26.3**¹ |

¹ The default-protocol VLM-LM scores on these are 1.11 and 1.52
respectively (absolute, not Δ). The model is unable to follow / parse
prompts that are not wrapped in its chat template — LLaVA-OneVision
appears to have been trained exclusively on chat-templated text, so the
backbone has lost the ability to respond to bare instruction prefixes.
This is a **stronger** form of degradation than what Qwen2.5-VL shows
and is a real differentiator for the paper.

### 4.3 LLM-only protocol cross-comparison

A side finding: **for the SAME LLM checkpoint**, the protocol matters
massively, and the protocol asymmetry differs by capability:

| task | Qwen2.5-7B-Instruct (instruct vs default) | Qwen2-7B-Instruct (instruct vs default) |
|---|---|---|
| GSM8K-CoT | 78.92 vs **86.58** (+7.7 with 8-shot ICL) | 75.44 vs 74.68 |
| IFEval | **72.09** vs 56.93 (+15 with chat tpl) | **51.39** vs 40.67 |
| GPQA-Diamond | 29.29 vs 27.27 | 24.24 vs 27.78 |

Qwen2.5 shows a much sharper "chat-template improves instruction-following,
few-shot improves math" decomposition; Qwen2 is more uniform. Probably
reflects how aggressively each generation was post-trained on
chat-templated SFT vs few-shot ICL.

### 4.4 Failure attempt: first E-Pull merge with owner-energy down_proj

`docs/overview.md` (as authored) said "down-projections excluded from
active merging, follow the dominant-energy modality, as in prior
covariance-based methods". We implemented this literally for the first
merge: `down_proj` weight came entirely from the modality with higher
`α_i · tr(C_i)`. **Result: catastrophic.** 23 of 28 down_projs got the
VLM-LM's weight while gate/up_proj were RegMean-merged; the resulting
MLP block had mismatched activations (gate × up output came from a mix
distribution, then went through a down_proj calibrated for a pure-VLM
intermediate distribution). The merged model produced gibberish from
token 1:

```
"The capital of France is Paris/Dk:'.$[".:'.$[..."         # greedy
"põe].'⚗⚗⚗⚗⚗⚗⚗⚗⚗..."                                # chat-template
```

(Smoke test logs persisted at
`/131_data/geeho/minsik/merged/qwen25_epull_BROKEN_downproj_owner_v1/`.)

**Fix.** Extended `covariance.py` to collect full Gram for `down_proj`
too (1.44 GB fp32 per layer at d = 18944, 28 layers × 2 models ≈ 80 GB on
CPU — fine, box has 503 GB RAM). Moved Gram accumulation off-GPU to avoid
OOM. Re-ran merge: 196 layers all epull mode, MLP blocks coherent, smoke
test now produces correct short answers ("Paris.", "56" for 7×8) before
trailing into repetition — partial improvement, see §5.3.

### 4.5 Other infrastructure failures encountered and fixed

| failure | root cause | fix |
|---|---|---|
| GPU OOM during Gram collection at d = 18944 | hook accumulated 28 × 1.44 GB Grams on GPU | move accumulation to CPU via `.to("cpu", non_blocking=True)` |
| Cholesky failure: leading minor 6886 not PD | fp32 ill-conditioned at d = 18944 | adaptive ε escalation (×100 per retry, up to 4 tries) |
| `lm_eval: command not found` after session restart | conda env wiped between sessions | reinstall `lm_eval[ifeval,math]==0.4.5` + pin `openai` and `datasets` versions |
| `Feature type 'List' not found` | `datasets ≥ 3.6` deprecated `List` (lm-eval 0.4.5 cached features with it) | pin `datasets >= 2.16, < 2.19`; clear `~/.cache/huggingface/datasets/{Idavidrein___gpqa, google___if_eval, mmlu_pro, ...}` |
| GPQA `GatedRepoError 401` | `huggingface-cli login` writes to `HF_HOME/token` but subprocess used different `HF_HOME` | pass `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` env vars explicitly to every subprocess |
| Two matrix runners racing | leftover process from a prior session not cleanly killed | explicit `kill -9 PID` of matrix shells before relaunch; idempotent resume detects existing JSONs to avoid double work |
| `Custom LlavaQwenForCausalLM` not in `transformers` | LLaVA-OneVision custom arch | direct-safetensors-read fallback in `extraction/loader.load_state_dict` |
| `parse_results` showed `gpqa_diamond_cot_zeroshot` as 0 | parser fell back to first alphabetical metric = `strict-match` (always 0 for chat outputs) | add explicit mapping `gpqa_diamond_cot_zeroshot → exact_match,flexible-extract` |
| `parse_results` filtered `mmlu_pro` out | filter clause `t.startswith('mmlu_') and t != 'mmlu'` matched `mmlu_pro` | replace with explicit `KEEP` set |
| `xargs -I{} python -c "..." {}` corrupted Python source | xargs substituted `{}` inside the Python string too | use `OUT_DIR`/`REQ` env vars instead of `{}` substitution |
| Background process killed at session change | bash `&` doesn't fully detach | use `setsid nohup ... < /dev/null & disown` plus explicit `env PATH=...` |
| Long mmlu_pro 5-shot full set (12k items × 5-shot CoT) | wall-time blew through session timeout (~16 h / model) | switched to `--limit 0.0416` ≈ 500-item stratified subsample for the merge-comparison runs; full-set numbers we got separately under OpenCompass |

### 4.6 Cumulative GitHub history (main branch)

```
55f8780  results: add OpenCompass column to Qwen2.5-7B-Instruct cross-check
d8309b8  results: add absolute-score tables + public reference cross-check
429bb05  results: add LLaVA-OneVision text degradation rows (both protocols)
f2aa66e  run_eval_matrix: idempotent resume — skip already-completed tasks
3025829  parse_results: resolve gpqa_diamond_cot_zeroshot to flexible-extract
b00f6a9  Consolidate eval orchestration into run_eval_matrix.sh
4e684b0  Dual-protocol eval (default + instruct-aware) + protocol_audit.md
b6209cf  Add PLAN.md tracking three research-readiness gaps
ad83174  Add E-Pull (Entropy-gated Pull) merge method + per-task eval protocols
acf163b  Initial commit: VLM→LLM extraction + 8-task text / 6-task vision eval
```

---

## 5. Key findings

### 5.1 Text-side degradation is reproducible across VLM families

Four families measured (three legacy + LLaVA-OneVision newly added), three
training regimes (full FT, LoRA, LoRA+SFT). The qualitative pattern is
identical: procedural / CoT / instruction-following lose 10–40 pt;
parametric knowledge (MMLU, BoolQ) is nearly flat. This is the
[`text_degradation.md`](text_degradation.md) headline.

### 5.2 The chat-template-vs-no-chat-template asymmetry

For *every* VLM-LM we tested, the degradation under the default protocol
(no chat template, few-shot for some tasks) is **larger** than under the
instruct protocol (0-shot, chat template). For LLaVA-OneVision the
default-protocol VLM-LM cannot follow non-chat-templated instructions at
all (IFEval 1.11, GPQA 1.52) while its instruct-protocol numbers are
recoverable (IFEval 42.14, GPQA 16.67). This means

- VL post-training has narrowed the text backbone to the chat-template
  input distribution; non-templated prompts go OOD.
- The *standard* (instruct-protocol) chat eval **systematically
  understates** the underlying capability loss.

The Qwen2.5-VL GSM8K asymmetry (−1 pt instruct, −22 pt default) was the
sharpest concrete example. This is a defensible secondary finding for
the paper's introduction; it strengthens the motivation for a merging
approach (because merging recovers the more-degraded subspace, not the
chat-templated façade).

### 5.3 E-Pull on text-only calibration degenerates to RegMean

This is the most important and most negative finding from the merge
itself.

**Stats on the (fixed) Qwen2.5 merge:**
- avg gate = **0.0116** (range 0.0008–0.0796 for active layers; for
  down_proj layers it's higher, 0.10–0.23)
- avg entropy-norm `H_j / log k` = 0.988 (near maximum; routing is nearly
  uniform per direction across most layers)
- avg `fg_cost` = 0.162, avg off-diagonal residual = 0.39, avg commutator
  residual = 0.52

Because the gate is essentially 0 over active layers, the merge formula
collapses to `m_j^** ≈ m_j^*` which is RegMean per direction. The
"entropy pull toward the dominant modality" never activates.

**Why?** The calibration data (wikitext) goes through both LLM-Instruct
and VLM-LM, but **both models process it as plain text**. Their activation
distributions are highly correlated; per-direction λ_{i,j} ratios are
near 1.0; entropy is near max; gate is near 0. E-Pull's whole point is
to use modality-asymmetric activation structure to route per direction —
but on identical-distribution calibration data there *is* no asymmetry to
exploit.

The downstream consequence is exactly what's expected of a weight-average
merge of two specialized models: it preserves token-ranking accuracy on
multi-choice LL tasks (because logit ranking is locally smooth in weight
space) but destroys autoregressive coherence on long generations
(because the trajectories of two unaligned models drift apart fast). The
smoke-test output is the telltale:

```
prompt: "The capital of France is"
greedy: "Paris.\nMUXMUXMUXMUXMUXMUXMUX..."        # correct prefix, then loop

prompt: chat-template("What is 7 * 8? Just give the number.")
greedy: "56 unmist误 Buccane Buccane Buccane..."  # correct token, then loop
```

The model knows the right first token; it loses coherence in the long
tail. This is the canonical RegMean failure mode for instruction-tuned
LLMs that have drifted apart, and it explains the eval table: BoolQ
(single-token answer) is fine, MMLU (single-token answer) is fine, GSM8K
(long CoT) cratered.

### 5.4 Down-projections need to be merged, not picked

`docs/overview.md` describes "down_proj follows the dominant-energy
modality" as a deliberate design choice "as in prior covariance-based
methods". This is **wrong for instruction-tuned chat-templated models.**
Picking one modality's whole `down_proj` while merging `gate_proj` and
`up_proj` produces an MLP block whose pre-activation (gate * up) is a
mix-distribution intermediate but whose down-projection expects a
pure-modality intermediate. The block becomes incoherent and the model
emits gibberish. Concretely:

- Broken (owner-energy down_proj) merge: smoke test totally garbled
  (`MUXMUXMUX...` from token 1)
- Fixed (full E-Pull on down_proj) merge: smoke test partially coherent
  (correct prefix, then repetition)

Both produce useless eval numbers, but the difference between "useless
because gibberish" and "useless because regression to RegMean" is the
gap between (i) an implementation bug and (ii) a method-level
limitation. We're in regime (ii) now, which is the meaningful one.

---

## 6. Open issues / missing data

1. **VLM-LM GPQA missing** for the Qwen2.5-VL pair in both protocols.
   The earlier eval-matrix run crashed mid-GPQA for the VLM-LM dir
   before producing JSONs. Re-running this would close a small but real
   hole in `text_degradation.md`.
2. **mmlu_pro full-set for VLM-LM** runs (we only have 0.0416 subsample
   for some of the Qwen2.5 epull cells); the 5-shot full-set Qwen2.5
   LLM/VLM rows we already have.
3. **GPQA OpenCompass score on our hardware** — the `genericllmeval`
   judge step needs an external API; the raw generations are saved but
   ungraded.
4. **Qwen2-7B-Instruct OpenCompass cross-check** never run; only the
   Qwen2.5-7B-Instruct cross-check landed. Should be a half-day add.
5. **Vision-side preservation eval** — the merged checkpoint we built is
   text-only. To evaluate "did we preserve the VLM's vision capability"
   we need to splice the merged text backbone back into the full
   VLM (replacing the original VLM's text tower) and run MMMU, MMBench,
   MM-Vet. The splicer is a small piece of code; the eval is one
   `bash evaluation/vision/run_vision_eval.sh` away. We have not yet
   done this; without it, the paper claim "merging recovers text without
   killing vision" is unmeasured.
6. **Baseline merging methods** — RegMean, Task Arithmetic, TIES,
   DARE, AdaMerging, Localize-and-Stitch — all need to be benchmarked
   on the same Qwen2.5-VL / LLaVA-OneVision pair under the same two
   protocols. Until we have these we cannot claim E-Pull is better than
   anything (especially since the current run shows it is **worse**
   than the un-merged LLM, so we need at least to be no worse than
   RegMean).

---

## 7. Direction forward

The user has been explicit that **this is a merging paper** — the
phenomenology of VLM text loss is motivation, not the headline. With
that framing, the next moves are:

### 7.1 Modality-aware calibration (the most important next experiment)

The hypothesis underlying §5.3: E-Pull's gate fails because we calibrate
**both** modalities on text inputs. What if we calibrate the VLM side on
**actual VLM-distribution inputs** (image-text)?

Concrete recipe:
- For the LLM modality, keep wikitext calibration as today.
- For the VLM modality, feed image-text prompts (e.g. LLaVA-Instruct-150K
  samples) through the **full VLM** (vision tower + connector + LM) and
  hook the LM's input activations.
- Now the per-direction λ_{i,j} ratios should differ — the VLM's text-path
  attention/MLP heads that are "used by vision tokens" will have very
  different covariance shape than the LLM's, and the entropy gate has
  signal to act on.
- Re-merge with these asymmetric Grams and re-eval.

If this works it converts our failed merge into the paper's central
claim: "cross-modality merging requires modality-asymmetric calibration;
text-only calibration on both sides reduces to RegMean and inherits its
failure mode."

### 7.2 Baselines

After 7.1, run the standard merging-method comparison on both
Qwen2.5-VL and LLaVA-OneVision pairs:

| baseline | what to implement |
|---|---|
| Naive average `(W_LLM + W_VLM)/2` | trivial |
| RegMean (Jin '23) | already covered by E-Pull with `gate = 0`; report both text-cal and mm-cal variants |
| Task Arithmetic (Ilharco '23) | `W = W_base + α(W_LLM − W_base) + β(W_VLM − W_base)`; one hyperparameter search needed |
| TIES (Yadav '23) | trim + sign-elect + average on `ΔW_LLM − ΔW_VLM` |
| DARE (Yu '24) | random-drop + rescale of deltas, often stacked on TIES |
| Localize-and-Stitch (Wei '24) | sparse-localize + graft; closest published comparator on cross-modality merging |

Goal: E-Pull (mm-cal variant) at least matches and ideally beats these on
the Pareto frontier of (text-retention, vision-retention).

### 7.3 Vision-side preservation

Write a tiny "splicer" that takes the merged text backbone + the
original VLM's vision tower / connector and outputs a runnable VLM.
Run `evaluation/vision/run_vision_eval.sh` on MMMU + MMBench-EN +
MM-Vet for: (i) original VLM, (ii) E-Pull merged spliced VLM, (iii) each
baseline merging method spliced. Report a (text, vision) Pareto plot.

This is **required** for the paper. "Recovers text" without "preserves
vision" is uninteresting — the LLM-Instruct already does the first by
not being a VLM in the first place.

### 7.4 Phenomenon: add Phi-3.5-Vision under the new protocols

We have legacy mixed-protocol numbers for Phi-3.5-Vision but no
clean (instruct, default) split. The model is downloaded; redo it the
same way as Qwen2.5-VL and LLaVA-OneVision so all four families are on
the same eval footing.

### 7.5 Deferred (do only if 7.1–7.3 land)

- Joint text-SFT ablation (small-scale Qwen2-1.5B with varying text-data
  mix) — confirms "joint SFT doesn't fully fix the procedural loss",
  positions E-Pull against the practitioner default of "just mix text
  into your VL training data".
- Frozen-LM / LoRA training ablation — confirms the LoRA-un-merged
  baseline ceiling.
- Mechanism probe: induction-head score (Olsson '22) before/after VL
  training — a "where does ICL live" mechanism story, if §7.1 doesn't
  land convincingly.

---

## 8. What's pinned in the repo

| artifact | location |
|---|---|
| Method code | `method/` (7 files + README) |
| Extraction code | `extraction/` (loader, extract_lm, registry, models.yaml) |
| Text-eval driver | `evaluation/text/eval_8tasks.sh` |
| Matrix orchestrator | `evaluation/text/run_eval_matrix.sh` |
| OpenCompass cross-check | `evaluation/text/opencompass_crosscheck.py` |
| Eval-protocol audit | `evaluation/text/protocol_audit.md` |
| Phenomenon results | `results/text_degradation.md` |
| This log | `results/poc.md` |
| Plan / scope | `PLAN.md` |

Eval result JSONs and merged checkpoints live under
`/131_data/geeho/minsik/` (not in the repo); see PLAN.md for paths.
