"""Build a from-scratch VLM training initialisation by composing two public
pretrained checkpoints:

  * **Vision encoder**: ``Qwen/Qwen2.5-VL-7B-Instruct`` — keep ``visual.*`` weights
    that come with this checkpoint (patch embedding + vision transformer blocks).
  * **Language backbone**: ``Qwen/Qwen2.5-7B-Instruct`` — overwrite the VL
    checkpoint's LLM portion (``model.language_model.*`` and ``lm_head.*``) with
    these *text-only* weights, so the resulting model starts as a pure LM that
    has not yet seen any visual training.
  * **Multimodal projector**: ``model.visual.merger.*`` is reset to a fresh
    random initialisation. The projector is the only module that is randomly
    initialised; everything else is loaded from a public checkpoint.

The output is a self-contained Hugging Face model directory (config + sharded
safetensors + tokenizer + processor + chat template) that you can pass to the
MERIT trainer as ``model.pretrained``.

This is the entry point for reproducing the legacy 3-stage VLM training recipe
(align → stage 1.5 → stage 2) where the VLM is built up from a pure LM rather
than starting from an already-aligned VLM such as ``Qwen2.5-VL-7B-Instruct``.

Usage::

    python -m merit.scripts.build_init_from_pretrained \\
        --vl-ckpt   Qwen/Qwen2.5-VL-7B-Instruct \\
        --lm-ckpt   Qwen/Qwen2.5-7B-Instruct \\
        --output    ./ckpts/init/qwen25vl_text_lm \\
        --dtype     bfloat16

Requires ~16 GB output disk and ~30 GB RAM during composition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


def _str_to_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _reset_module_(module: nn.Module) -> None:
    """Apply standard fresh init to a sub-module in place.

    LayerNorms reset to identity (weight=1, bias=0). Linear layers use
    Kaiming-uniform for weight and zero bias — the same convention PyTorch's
    default Linear init uses.
    """
    for sub in module.modules():
        if isinstance(sub, nn.LayerNorm):
            nn.init.ones_(sub.weight)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.Linear):
            nn.init.kaiming_uniform_(sub.weight, a=5**0.5)
            if sub.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(sub.weight)
                bound = 1 / fan_in**0.5 if fan_in > 0 else 0
                nn.init.uniform_(sub.bias, -bound, bound)


def _replace_lm_weights_(vl_state: dict[str, torch.Tensor], lm_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Overwrite VL state dict's LM portion (``model.language_model.*`` +
    ``lm_head.*``) with weights from the text-only LM checkpoint.

    Returns ``(n_replaced, n_skipped)``. Skipped keys are those present in the
    VL model but absent from the LM checkpoint (raises if any LM weight is
    missing — the two checkpoints must have matching vocab/hidden sizes).
    """
    n_replaced = 0
    missing_in_lm: list[str] = []

    for key in list(vl_state.keys()):
        if key.startswith("model.language_model."):
            src_key = "model." + key[len("model.language_model.") :]
        elif key.startswith("lm_head."):
            src_key = key
        else:
            continue

        if src_key not in lm_state:
            missing_in_lm.append(src_key)
            continue

        if vl_state[key].shape != lm_state[src_key].shape:
            raise ValueError(
                f"shape mismatch: vl[{key}]={tuple(vl_state[key].shape)} "
                f"vs lm[{src_key}]={tuple(lm_state[src_key].shape)}; "
                "the two checkpoints must share vocab and hidden sizes."
            )

        vl_state[key] = lm_state[src_key].to(vl_state[key].dtype)
        n_replaced += 1

    if missing_in_lm:
        raise KeyError(
            f"the LM checkpoint is missing {len(missing_in_lm)} keys that the "
            f"VL model expects (e.g. {missing_in_lm[:3]}); cannot proceed."
        )

    return n_replaced, len(missing_in_lm)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vl-ckpt", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Source for vision tower + model architecture.")
    ap.add_argument("--lm-ckpt", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Source for the language backbone weights (text-only).")
    ap.add_argument("--output", required=True, help="Output directory for the composed checkpoint.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the projector reset so the random init is reproducible.")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[build_init] refusing to overwrite non-empty {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _str_to_dtype(args.dtype)

    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )

    print(f"[build_init] loading VL skeleton: {args.vl_ckpt}")
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_ckpt,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    vl_state = vl_model.state_dict()

    print(f"[build_init] loading LM weights: {args.lm_ckpt}")
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.lm_ckpt,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    lm_state = lm_model.state_dict()

    print("[build_init] overwriting LM portion of the VL skeleton")
    n_replaced, _ = _replace_lm_weights_(vl_state, lm_state)
    print(f"[build_init]   replaced {n_replaced} LLM tensors")

    # Free the LM model — only its state_dict was needed.
    del lm_model, lm_state

    vl_model.load_state_dict(vl_state)
    del vl_state

    # Reset multimodal projector (visual.merger) weights — random init.
    merger = None
    for name, module in vl_model.named_modules():
        if name.endswith("visual.merger"):
            merger = module
            break
    if merger is None:
        raise RuntimeError("could not locate visual.merger submodule in the VL model")
    print(f"[build_init] resetting visual.merger ({sum(p.numel() for p in merger.parameters())/1e6:.1f}M params)")
    _reset_module_(merger)

    print(f"[build_init] saving to {out_dir}")
    vl_model.save_pretrained(out_dir)

    # Carry the VL tokenizer/processor (its chat template + image processor +
    # vision-special tokens) so the saved directory is self-contained.
    AutoTokenizer.from_pretrained(args.vl_ckpt, trust_remote_code=True).save_pretrained(out_dir)
    AutoProcessor.from_pretrained(args.vl_ckpt, trust_remote_code=True).save_pretrained(out_dir)

    print("[build_init] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
