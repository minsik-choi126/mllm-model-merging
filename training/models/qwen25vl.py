"""Qwen2.5-VL backbone wrapper used throughout MERIT.

We do NOT subclass the transformers model. Instead we expose thin helpers that
pull in the official ``Qwen2_5_VLForConditionalGeneration`` / tokenizer /
image processor by name, apply MERIT's freezing policy, and return the raw HF
model object. This keeps downstream code (Trainer, gradient extractor, merge
validation, lmms-eval adapter) decoupled from transformers internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelSpec:
    pretrained: str          # e.g. "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    # Qwen2.5-VL dynamic-resolution bounds (forwarded to AutoProcessor).
    max_pixels: int | None = None
    min_pixels: int | None = None


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def load_qwen_vl(spec: ModelSpec) -> tuple[Any, Any, Any]:
    """Load (model, tokenizer, image_processor) from a Qwen2.5-VL checkpoint."""
    # Import lazily so that `merit` is importable without transformers in CI.
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )

    torch_dtype = _DTYPE_MAP.get(spec.dtype.lower(), torch.bfloat16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        spec.pretrained,
        torch_dtype=torch_dtype,
        trust_remote_code=spec.trust_remote_code,
        attn_implementation=spec.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        spec.pretrained,
        trust_remote_code=spec.trust_remote_code,
        use_fast=True,
    )
    proc_kwargs: dict[str, Any] = dict(trust_remote_code=spec.trust_remote_code)
    if spec.max_pixels is not None:
        proc_kwargs["max_pixels"] = spec.max_pixels
    if spec.min_pixels is not None:
        proc_kwargs["min_pixels"] = spec.min_pixels
    processor = AutoProcessor.from_pretrained(spec.pretrained, **proc_kwargs)

    _patch_qwen25vl_forward_for_ddp_safety()

    return model, tokenizer, processor


def _patch_qwen25vl_forward_for_ddp_safety() -> None:
    """Make Qwen2_5_VLModel.forward DDP-safe for batches that have ``pixel_values``
    but no ``image_token_id`` tokens in ``input_ids``.

    Why this is needed
    ------------------
    The projector (``visual.merger``) is *trainable*. With multi-source data
    that contains a non-trivial fraction of text-only samples (LLaVA-1.5
    mix665k, ~6 % ShareGPT), one DDP rank's micro-batch can end up entirely
    text-only while the other 7 ranks have image-bearing samples. Standard
    forward then takes different code paths per rank — the text-only rank
    skips both ``get_image_features`` and the projector. The next collective
    op (e.g. ``lm_head`` gradient ``ALLREDUCE``) ends up at a different NCCL
    sequence number on the skipped rank vs. the others, producing the exact
    ``NumelIn=1 vs NumelIn=544997376 SeqNum=275407`` desync we hit at step
    ~3923 of stage 2 v1 across takes 5 / 7 / 11.

    The fix mirrors a known DDP-safety trick from internal VLM training code
    (``image_features = image_features[0:0]`` for text-only batches): always
    run the vision encoder + projector, but discard their output before it
    touches
    ``inputs_embeds`` so the language model sees no image content. The
    autograd graph is kept alive via ``image_embeds.sum() * 0`` so backward
    still calls the projector's grad ``ALLREDUCE`` on every rank.

    The collator is responsible for injecting a single small dummy image
    (e.g. 56×56 black) into batches that would otherwise have no
    ``pixel_values`` so this patch has something to call the vision encoder
    on.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel,
    )

    if getattr(Qwen2_5_VLModel.forward, "_merit_ddp_safe", False):
        return  # already patched

    _orig_forward = Qwen2_5_VLModel.forward

    def _patched(self, *args, **kwargs):
        pixel_values = kwargs.get("pixel_values")
        input_ids = kwargs.get("input_ids")
        inputs_embeds_arg = kwargs.get("inputs_embeds")
        image_grid_thw = kwargs.get("image_grid_thw")

        # Only intervene when the standard path is going to take the
        # ``pixel_values is not None`` branch *and* would mismatch on the
        # image-token count check.
        if (
            pixel_values is not None
            and input_ids is not None
            and inputs_embeds_arg is None
        ):
            n_image_tokens = (
                (input_ids == self.config.image_token_id).sum().item()
            )
            if n_image_tokens == 0:
                # Force a vision-tower + projector forward for collective sync.
                image_embeds = self.get_image_features(
                    pixel_values, image_grid_thw
                )
                # ``image_embeds[0:0]`` + masked_scatter trick: drop the
                # features to length 0, then masked_scatter with an
                # all-False mask.
                # masked_scatter is a no-op when both mask and source are
                # empty, but autograd still threads ``image_embeds[0:0]``
                # through the graph so backward calls the projector's grad
                # ALLREDUCE on every rank (value will be zero, but the
                # collective op is what we need for DDP synchronisation).
                image_embeds = image_embeds[0:0]

                inputs_embeds = self.get_input_embeddings()(input_ids)
                mask = input_ids == self.config.image_token_id
                mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    mask_expanded, image_embeds
                )

                # Hand off to standard forward via inputs_embeds so it
                # skips its own pixel_values handling entirely.
                kwargs = dict(kwargs)
                kwargs["pixel_values"] = None
                kwargs["image_grid_thw"] = None
                kwargs["inputs_embeds"] = inputs_embeds
                kwargs["input_ids"] = None

        return _orig_forward(self, *args, **kwargs)

    _patched._merit_ddp_safe = True  # type: ignore[attr-defined]
    Qwen2_5_VLModel.forward = _patched  # type: ignore[assignment]
