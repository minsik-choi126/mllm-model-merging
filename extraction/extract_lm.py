"""Extract a VLM's text backbone into a standalone HuggingFace text-only model.

Usage:
    # With an explicit LLM template (config + tokenizer copied from there):
    python -m extraction.extract_lm \
        --vlm Qwen/Qwen2.5-VL-7B-Instruct \
        --output cache/extracted/qwen25vl_7b_lm \
        --llm-template Qwen/Qwen2.5-7B-Instruct

    # Or via a registered pair (template inferred from models.yaml):
    python -m extraction.extract_lm --pair qwen25vl_7b --output cache/extracted/qwen25vl_7b_lm

    # Fallback (no LLM template): config built from VLM's text_config field.
    python -m extraction.extract_lm \
        --vlm OpenGVLab/InternVL2_5-8B \
        --output cache/extracted/internvl25_8b_lm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .loader import (
    _from_pretrained_with_local_fallback,
    load_state_dict,
    save_text_model,
)
from .registry import get_pair


def _save_from_vlm_text_config(
    vlm_id: str,
    output_dir: str,
    state_dict: dict[str, torch.Tensor],
):
    """Fallback save path when no LLM template ID is provided.

    Builds a minimal text-only `config.json` from the VLM config's
    `text_config` field, and uses the VLM's tokenizer.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoTokenizer

    vlm_config = _from_pretrained_with_local_fallback(
        AutoConfig,
        vlm_id,
        trust_remote_code=True,
    )
    text_cfg = vlm_config.to_dict().get("text_config")
    if text_cfg is None:
        raise ValueError(
            "No text_config found in the VLM config. Pass --llm-template instead."
        )

    save_file(state_dict, str(out / "model.safetensors"))

    tokenizer = _from_pretrained_with_local_fallback(
        AutoTokenizer,
        vlm_id,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(out))

    arch = text_cfg.get("architectures") or ["Qwen2ForCausalLM"]
    lm_config = {
        "architectures": arch,
        "model_type": text_cfg.get("model_type", "qwen2"),
        "hidden_size": text_cfg["hidden_size"],
        "intermediate_size": text_cfg["intermediate_size"],
        "num_hidden_layers": text_cfg["num_hidden_layers"],
        "num_attention_heads": text_cfg["num_attention_heads"],
        "num_key_value_heads": text_cfg["num_key_value_heads"],
        "max_position_embeddings": text_cfg.get("max_position_embeddings", 32768),
        "rope_theta": text_cfg.get("rope_parameters", {}).get("rope_theta", 1_000_000.0),
        "rms_norm_eps": text_cfg.get("rms_norm_eps", 1e-6),
        "vocab_size": text_cfg["vocab_size"],
        "hidden_act": text_cfg.get("hidden_act", "silu"),
        "tie_word_embeddings": text_cfg.get("tie_word_embeddings", False),
        "use_sliding_window": text_cfg.get("use_sliding_window", False),
        "sliding_window": text_cfg.get("sliding_window"),
        "torch_dtype": "bfloat16",
        "bos_token_id": text_cfg.get("bos_token_id"),
        "eos_token_id": text_cfg.get("eos_token_id"),
    }

    with open(out / "config.json", "w") as f:
        json.dump(lm_config, f, indent=2)


def extract_lm_from_vlm(
    vlm_id: str,
    output_dir: str,
    llm_template: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Extract a VLM's text backbone into a text-only HuggingFace model directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading text backbone from VLM: {vlm_id}")
    lm_sd = load_state_dict(
        vlm_id,
        dtype=dtype,
        device="cpu",
        normalize_text_backbone=True,
    )
    print(f"  Extracted {len(lm_sd)} text-backbone tensors")

    metadata = {"source_vlm": vlm_id}
    if llm_template:
        metadata["llm_template"] = llm_template
        save_text_model(lm_sd, llm_template, str(out), metadata=metadata)
    else:
        _save_from_vlm_text_config(vlm_id, str(out), lm_sd)

    print(f"Saved LM backbone to: {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Extract LM backbone from a VLM")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--vlm", help="VLM model ID (HF or local path)")
    src.add_argument("--pair", help="Registered pair key (see extraction/models.yaml)")

    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--llm-template",
        default=None,
        help="Text-only template model for config/tokenizer export. "
             "Required for clean LLM↔VLM comparison; ignored when --pair is set "
             "(the registry's `llm` field is used).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    if args.pair:
        pair = get_pair(args.pair)
        vlm_id = pair.vlm
        llm_template = pair.llm
    else:
        vlm_id = args.vlm
        llm_template = args.llm_template

    extract_lm_from_vlm(vlm_id, args.output, llm_template=llm_template, dtype=dtype)


if __name__ == "__main__":
    main()
