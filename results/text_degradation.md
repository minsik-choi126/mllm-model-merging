# Text degradation: VLM text-backbone vs. base LLM

Per-task delta (VLM text-backbone score − base LLM score, in points)
on the 8-task standard set. Negative = the VLM has lost capability
relative to its starting LLM.

Pipeline: `extraction/extract_lm.py` to recover the VLM's text path,
then `evaluation/text/eval_8tasks.sh` against the original LLM and the
extracted backbone.

| Model            | Training | MMLU | MMLU-Pro | GSM8K | TruthfulQA | BoolQ | IFEval | GPQA | EQ-Bench |
|------------------|----------|-----:|---------:|------:|-----------:|------:|-------:|-----:|---------:|
| Qwen2.5-VL-7B    | Full FT  | −3.1 | **−12.7** | −10.3 | −11.0 | −1.8 | −13.7 | −11.8 | −6.2 |
| LLaVA-LLaMA3-8B  | LoRA     | +0.4 | −2.6 | **−24.1** | −7.8 | −1.8 | **−35.2** | −3.2 | **−33.6** |
| Phi-3.5-Vision   | Full FT  | −7.4 | **−37.9** | −11.3 | **−18.0** | −2.0 | **−29.6** | −11.5 | −11.8 |

**Bold** = drop ≥ 15 points.

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
