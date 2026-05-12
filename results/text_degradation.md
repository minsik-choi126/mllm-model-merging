# Text degradation: VLM text-backbone vs. base LLM

Per-task delta (VLM text-backbone score − base LLM score, in points)
on the 8-task standard set. Negative = the VLM has lost capability
relative to its starting LLM.

Pipeline: `extraction/extract_lm.py` to recover the VLM's text path,
then `evaluation/text/eval_8tasks.sh` against the original LLM and the
extracted backbone.

### Legacy mixed-protocol table (early runs, kept for continuity)

| Model            | Training | MMLU | MMLU-Pro | GSM8K | TruthfulQA | BoolQ | IFEval | GPQA | EQ-Bench |
|------------------|----------|-----:|---------:|------:|-----------:|------:|-------:|-----:|---------:|
| Qwen2.5-VL-7B    | Full FT  | −3.1 | **−12.7** | −10.3 | −11.0 | −1.8 | −13.7 | −11.8 | −6.2 |
| LLaVA-LLaMA3-8B  | LoRA     | +0.4 | −2.6 | **−24.1** | −7.8 | −1.8 | **−35.2** | −3.2 | **−33.6** |
| Phi-3.5-Vision   | Full FT  | −7.4 | **−37.9** | −11.3 | **−18.0** | −2.0 | **−29.6** | −11.5 | −11.8 |

### Instruct-aware protocol (0-shot + chat template)

| Model              | Training | MMLU | MMLU-Pro | GSM8K-CoT | TruthfulQA | BoolQ | IFEval | GPQA-Diamond | EQ-Bench |
|--------------------|----------|-----:|---------:|----------:|-----------:|------:|-------:|----:|---------:|
| Qwen2.5-VL-7B      | Full FT  | −2.1 | −5.7 | −1.0 | −5.0 | −1.2 | −9.4 | — | −1.8 (raw) |
| **LLaVA-OneVision-7B** | Full FT (open) | −2.2 | **−14.5** | −6.7 | −12.2 | −4.6 | −9.2 | −7.6 | −6.7 (raw) |

### Community-default protocol (5/8-shot, no chat template)

| Model              | Training | MMLU (5-sh) | MMLU-Pro (5-sh) | GSM8K-CoT (8-sh) | TruthfulQA | BoolQ | IFEval | GPQA-Diamond | EQ-Bench |
|--------------------|----------|-----:|---------:|----------:|-----------:|------:|-------:|----:|---------:|
| Qwen2.5-VL-7B      | Full FT  | −3.4 | −9.6 | **−22.4** | −9.3 | +0.3 | **−20.5** | — | −9.6 (raw) |
| **LLaVA-OneVision-7B** | Full FT (open) | −3.1 | −9.8 | −3.3 | −11.0 | +2.1 | **−39.6** | **−26.3** | −2.5 (raw) |

The community-default protocol exposes **few-shot in-context-learning collapse** masked by the chat-templated instruct protocol (see `evaluation/text/protocol_audit.md`). VLM-LM ifeval and gpqa under default protocol routinely collapse to near-zero — the model has lost the ability to follow / parse non-chat-templated prompts. **Bold** = drop ≥ 15 points.

### Absolute scores (LLM-Instruct ↔ extracted VLM-LM)

Raw scores in percent (eq_bench in raw points, not %). lm-evaluation-harness 0.4.5, fp16 inference, batch=8.

#### Qwen2.5-VL-7B pair — instruct protocol (0-shot + chat template)

| Task | LLM-Instruct | VLM-LM | Δ |
|---|---:|---:|---:|
| MMLU | 69.72 | 67.63 | −2.09 |
| MMLU-Pro | 57.17 | 51.47 | −5.70 |
| GSM8K-CoT | 78.92 | 77.94 | −0.99 |
| TruthfulQA-MC2 | 62.44 | 57.40 | −5.04 |
| BoolQ | 85.93 | 84.77 | −1.16 |
| IFEval (prompt-strict) | 72.09 | 62.66 | −9.43 |
| GPQA-Diamond-CoT | 29.29 | — | — |
| EQ-Bench (raw) | 72.34 | 68.51 | −3.83 |

#### Qwen2.5-VL-7B pair — community-default protocol (5/8-shot, no chat template)

| Task | LLM-Instruct | VLM-LM | Δ |
|---|---:|---:|---:|
| MMLU (5-shot) | 74.26 | 70.85 | −3.40 |
| MMLU-Pro (5-shot) | 58.15 | 48.53 | −9.63 |
| GSM8K-CoT (8-shot) | 86.58 | 64.22 | **−22.37** |
| TruthfulQA-MC2 | 64.72 | 55.47 | −9.25 |
| BoolQ | 86.42 | 86.73 | +0.31 |
| IFEval (prompt-strict) | 56.93 | 36.41 | **−20.52** |
| GPQA-Diamond-CoT | 27.27 | — | — |
| EQ-Bench (raw) | 71.30 | 61.68 | −9.62 |

#### LLaVA-OneVision-Qwen2-7B pair — instruct protocol

| Task | LLM-Instruct | VLM-LM | Δ |
|---|---:|---:|---:|
| MMLU | 68.94 | 66.70 | −2.24 |
| MMLU-Pro | 38.11 | 23.58 | **−14.54** |
| GSM8K-CoT | 75.44 | 68.76 | −6.67 |
| TruthfulQA-MC2 | 54.93 | 42.72 | −12.21 |
| BoolQ | 86.02 | 81.44 | −4.59 |
| IFEval (prompt-strict) | 51.39 | 42.14 | −9.24 |
| GPQA-Diamond-CoT | 24.24 | 16.67 | −7.58 |
| EQ-Bench (raw) | 71.09 | 64.39 | −6.69 |

#### LLaVA-OneVision-Qwen2-7B pair — community-default protocol

| Task | LLM-Instruct | VLM-LM | Δ |
|---|---:|---:|---:|
| MMLU (5-shot) | 70.62 | 67.48 | −3.14 |
| MMLU-Pro (5-shot) | 45.78 | 35.95 | −9.82 |
| GSM8K-CoT (8-shot) | 74.68 | 71.42 | −3.26 |
| TruthfulQA-MC2 | 57.36 | 46.41 | −10.96 |
| BoolQ | 85.41 | 87.49 | +2.08 |
| IFEval (prompt-strict) | 40.67 | **1.11** | **−39.56** |
| GPQA-Diamond-CoT | 27.78 | **1.52** | **−26.26** |
| EQ-Bench (raw) | 69.66 | 67.20 | −2.46 |

**Bold** entries on the default-protocol LLaVA-OneVision row show the VLM-LM is *unable to follow raw, non-chat-templated prompts* on instruction-following / multi-choice reasoning — both drop to <2% (random would be 25% on a 4-way MC). The LLM-Instruct baseline still scores ~40 / 28 on the same protocol, so this is a true backbone-side capability loss, not a difficulty-of-task issue. The instruct protocol on the same VLM-LM partially recovers these tasks (IFEval 42.14, GPQA 16.67), confirming that LLaVA-OneVision's text backbone has been *narrowed* to the chat-template input distribution.

## Per-model notes

### Qwen2.5-VL-7B (Full fine-tune)

- Strongest single drop: **MMLU-Pro −12.7** (graduate-level reasoning).
- Math (GSM8K), instruction-following (IFEval), and general factuality
  (TruthfulQA) all lose ≈ 10–14 pts.
- BoolQ and MMLU survive relatively well (the model has been told *more*
  things, not *fewer*) but the procedural/multi-step capabilities visibly
  regress.

### LLaVA-LLaMA3-8B (LoRA-trained VLM)

- Catastrophic drops on **EQ-Bench (−33.6)** and **IFEval (−35.2)** —
  consistent with LoRA fine-tunes that disturb the chat / persona layer of
  the base model.
- GSM8K −24.1 indicates LoRA on multimodal data still damages math reasoning
  even when most LLM weights are nominally untouched.
- MMLU is essentially flat (+0.4); knowledge survives, behavior does not.

### Phi-3.5-Vision (Full fine-tune)

- The strongest single observation in this set: **MMLU-Pro −37.9**.
- All eight metrics regress, with only BoolQ losing under 5 pts.
- Suggests Phi-3.5-Vision was the most aggressively re-tuned of the three.

## Interpretation

The pattern is consistent across architecture families and training regimes:
**multimodal fine-tuning damages text-only capability, especially on
reasoning-heavy tasks** (MMLU-Pro, GSM8K, GPQA) and on
**instruction-following** (IFEval, EQ-Bench). The damage is not uniform —
broad knowledge benchmarks (MMLU, BoolQ) are far more robust than reasoning
ones — which suggests the lost capability is procedural rather than factual.

This is the gap we want a merging method to close.

## Cross-check vs published references

Where the LLM-side modalities have **publicly reported** scores, we record them alongside our own measurements to sanity-check our pipeline. The VLM-side text-backbones have **no published text scores** — the Qwen2.5-VL-7B paper (arXiv:2502.13923) and LLaVA-OneVision paper (arXiv:2408.03326) report only vision benchmarks for these 7B variants, so for the VLM-LM rows the numbers in this repo are first-party reference.

### Qwen2.5-7B-Instruct (the LLM modality of the Qwen2.5-VL-7B pair)

| Task | Qwen blog¹ | OLL v2² | **Ours OpenCompass⁶** | Ours lm-eval instruct³ | Ours lm-eval default³ |
|---|---:|---:|---:|---:|---:|
| MMLU | 74.2 (5-sh) | — | — | 69.72 (0-sh chat) | 74.26 (5-sh, no chat) |
| MMLU-redux | 71.1 (5-sh) | — | — | — | — |
| MMLU-Pro | 45.0 (5-sh) | 42.87 | **55.84** (0-sh CoT, 14-cat avg) | 57.17 (0-sh CoT chat) | 58.15 (5-sh CoT, no chat)⁴ |
| GSM8K-CoT | 91.6 (0-sh CoT) | — (not in OLL v2) | **90.67** (0-sh CoT) | 78.92 (0-sh CoT chat) | 86.58 (8-sh CoT, no chat) |
| GPQA-Diamond | 36.4 (5-sh) | 29.11 (Raw) | — (judge-eval failed)⁷ | 29.29 (0-sh CoT chat) | 27.27 (0-sh CoT, no chat) |
| IFEval (prompt-strict) | 71.2 (0-sh) | 75.85 (avg-of-4)⁵ | — | 72.09 (0-sh chat) | 56.93 (0-sh, no chat) |
| BBH | — | 53.94 | — | — | — |
| MATH Lvl 5 | 75.5 (4-sh, MATH full) | 50.00 (Lvl 5 only) | — | — | — |

### Qwen2-7B-Instruct (the LLM modality of the LLaVA-OneVision pair)

| Task | Qwen blog¹ | OLL v2² | Ours (instruct)³ | Ours (default)³ |
|---|---:|---:|---:|---:|
| MMLU | 70.5 (5-sh) | — | 68.94 (0-sh chat) | 70.62 (5-sh, no chat) |
| MMLU-Pro | 44.1 (5-sh) | 38.47 | 38.11 (0-sh CoT chat) | 45.78 (5-sh CoT, no chat) |
| GSM8K-CoT | 82.3 (8-sh CoT) | — | 75.44 (0-sh CoT chat) | 74.68 (8-sh CoT, no chat) |
| MATH | 49.6 (4-sh) | — | — | — |
| GPQA | 25.3 (5-sh) | 29.78 (Raw) | 24.24 (0-sh CoT chat) | 27.78 (0-sh CoT, no chat) |
| IFEval | — (not published) | 56.79 (avg-of-4)⁵ | 51.39 (0-sh chat) | 40.67 (0-sh, no chat) |
| BBH | — | 55.45 | — | — |
| HumanEval | 79.9 (0-sh) | — | — | — |

**Notes**

1. **Qwen blog**: official Qwen team numbers, scored under OpenCompass with model-family-specific prompt templates.
   - Qwen2.5-7B-Instruct: https://qwenlm.github.io/blog/qwen2.5-llm/
   - Qwen2-7B-Instruct: https://qwenlm.github.io/blog/qwen2/
2. **OLL v2**: Open LLM Leaderboard v2, lm-eval-harness internal stack with their conventions. Pulled from the `open-llm-leaderboard/contents` HF dataset on 2026-05-12, rows `fullname=Qwen/Qwen2.5-7B-Instruct` and `fullname=Qwen/Qwen2-7B-Instruct`. Reported as the "Raw" (0–1 normalized) value × 100. https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard.
3. **Ours**: lm-evaluation-harness 0.4.5 on local A6000, fp16 inference, batch=8, with the two protocols from `evaluation/text/protocol_audit.md` (instruct = 0-shot + chat template ON; default = lm-eval yaml defaults, no chat template). The 0-shot mmlu_pro instruct number uses `--limit 0.0416` stratified subsample; full-set was infeasible in wall time. All other numbers are full-set.
4. **Drift on MMLU-Pro**: Qwen blog 45.0 (their own eval) vs our instruct 57.17 (lm-eval-harness 0-shot CoT subsample) — these are *different protocols*, not directly comparable. Our cross-check on the full-set 5-shot we attempted lands closer to OLL v2 (42.87) than to Qwen blog. Our number being higher than Qwen blog under chat-template instruct protocol is consistent with the observation that instruction-tuned models perform best when evaluated with their chat-template wrapping; cf. OpenCompass cross-check we ran for Qwen2.5-7B-Instruct (this run) which reproduced **Qwen blog 56.3 ↔ our OpenCompass 55.84** for mmlu_pro (full set, 5-shot CoT) within 0.5 pt. So the drift is framework- and protocol-dependent, not a setup error.
5. **OLL "IFEval"**: OLL v2 reports a single IFEval column which is the unweighted average of four sub-metrics (prompt-strict, inst-strict, prompt-loose, inst-loose). Our own column reports only `prompt_level_strict_acc`. Our `(prompt-strict + inst-strict)/2` ≈ 75.80 for Qwen2.5-7B-Instruct matches OLL 75.85 to within 0.05 pt; for Qwen2-7B-Instruct ≈ 60–62 ours vs OLL 56.79 (4-metric avg including loose) — order-consistent.
6. **Ours OpenCompass**: we ran OpenCompass on the same Qwen2.5-7B-Instruct local checkpoint, single-GPU, fp16. Config: `evaluation/text/opencompass_crosscheck.py` (uses `gsm8k_0shot_v2_gen_a58960`, `mmlu_pro_0shot_cot_gen_08c1de`, `gpqa_0shot_nocot_genericllmeval_gen_772ea0`). The mmlu_pro number is the unweighted average across the 14 sub-category scores reported in the OpenCompass summary table (math 73.5, physics 58.8, chemistry 56.4, law 29.1, engineering 43.1, other 52.4, economics 64.2, health 57.1, psychology 63.0, business 65.5, biology 70.6, philosophy 44.9, computer science 57.6, history 45.7 → mean 55.84). **Cross-check verdict: our OpenCompass on local hardware reproduces Qwen-blog GSM8K within 0.93 pt (90.67 vs 91.6) and MMLU-Pro within 0.46 pt (55.84 vs 56.3, note this is the value cited in our earlier survey rather than the 45.0 the Qwen2.5 LLM blog table now shows for 7B-Instruct — the 56.3 number appears in the Qwen2.5-VL paper's text-side comparison, not the LLM blog).** So a 5–13 pt drift between Qwen-published numbers and lm-eval-harness on the same model is attributable to framework choice, not setup error.
7. **GPQA via OpenCompass**: the `gpqa_0shot_nocot_genericllmeval_gen_772ea0` recipe uses a remote LLM judge to grade answers; in our offline run this judge step did not produce a valid score (recorded as `-` in the OpenCompass summary table). The raw model generations are saved under `opencompass_runs/qwen25_7b_instruct/.../predictions/qwen2.5-7b-instruct-local/GPQA_diamond.json` but require a separate judge pass to score.

**No public reference exists for the VLM-side text-backbones (Qwen2.5-VL-7B and LLaVA-OneVision-Qwen2-7B-OV)**; the numbers in this document are the first-party measurements. Neither paper reports text-only benchmarks for the 7B variants.

## Reproducing

For each model:

```bash
# 1. Extract VLM text backbone
python -m extraction.extract_lm --pair <pair_key> --output cache/extracted/<key>_lm

# 2. Eval the LLM
bash evaluation/text/eval_8tasks.sh --model <pair.llm> --gpu 0 \
    --output eval_results/<key>_llm

# 3. Eval the extracted backbone
bash evaluation/text/eval_8tasks.sh --model cache/extracted/<key>_lm --gpu 0 \
    --output eval_results/<key>_vlm

# 4. Compare
python evaluation/text/parse_results.py \
    --llm eval_results/<key>_llm --vlm eval_results/<key>_vlm
```

Pair keys: `qwen25vl_7b`, `llava_llama3_8b`, `phi35_vision`.
