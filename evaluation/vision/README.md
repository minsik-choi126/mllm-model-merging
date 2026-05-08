# Vision eval (6-task core)

Wrapper around [`VLMEvalKit`](https://github.com/open-compass/VLMEvalKit) for
the 6-task core vision benchmark set.

## Tasks

| Group       | Benchmarks                                                       |
|-------------|------------------------------------------------------------------|
| `core` (6)  | MMBench_DEV_EN_V11, SEEDBench_IMG, POPE, MMMU_DEV_VAL, MathVista_MINI, RealWorldQA |
| `extended`  | core + OCRBench, ChartQA_TEST, AI2D_TEST, HallusionBench, MMVet  |
| `full`      | extended + DocVQA_VAL, SEEDBench2_Plus                           |

Domain coverage of the core set:

- **General perception**: MMBench, SEEDBench
- **Knowledge / reasoning**: MMMU, MathVista, RealWorldQA
- **Hallucination**: POPE

This 6-task core matches the eval set used in the merging-paper baselines
(ACOM-style evaluations).

## Setup

Clone VLMEvalKit and install its requirements in a venv:

```bash
git clone https://github.com/open-compass/VLMEvalKit
cd VLMEvalKit
pip install -e .
```

Point the script at your checkout:

```bash
export VLMEVAL_DIR=/path/to/VLMEvalKit
export PYTHON_BIN=/path/to/.venv/bin/python   # or just `python3` if on PATH
```

## Run

```bash
bash run_vision_eval.sh \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu 0 \
    --tasks core
```

For a merged model directory (must include the VLM's processor / vision
tower), pass the local path:

```bash
bash run_vision_eval.sh --model cache/merged/my_method/vlm_model
```

The output directory is created automatically; pass `--output` to override.

## Notes

- The 6-task `core` set is fully deterministic (no LLM-judge metrics). If
  you switch to `extended` or `full`, MMVet uses a judge model and may
  require OpenAI / Gemini API keys configured via VLMEvalKit's own env vars.
- Run-time on a single H100 ≈ 3–6 hours for `core`, depending on VLM size.
