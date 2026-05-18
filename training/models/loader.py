"""High-level model loader that honors the MERIT freezing policy per stage.

Usage::

    from training.models import load_merit_model, load_tokenizer_and_processor

    model, tokenizer, processor = load_merit_model(cfg, stage="stage1")
    # stage="stage1" : train mm_projector only (freeze LLM + vision)
    # stage="stage2" : train projector + LLM (freeze vision)
    # stage="branch" : same as stage2, but loaded from a merge-ready init ckpt
    # stage="eval"   : everything frozen

Backbone is pulled from ``cfg.model.pretrained``; ``attn_implementation`` is
resolved by :func:`merit.models.attention.resolve_attn_implementation`.
"""
from __future__ import annotations

from typing import Any, Literal

import torch

from training.models.attention import resolve_attn_implementation
from training.models.projector import projector_parameters
from training.models.qknorm_injection import inject_qknorm
from training.models.qwen25vl import ModelSpec, load_qwen_vl
from training.models.vision_tower import freeze_vision_tower
from training.utils.logging import get_logger

logger = get_logger(__name__)

Stage = Literal["stage1", "stage2", "branch", "eval"]


def _freeze_llm(model: Any) -> int:
    n = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if "language_model" in lname or "model.layers" in lname or "lm_head" in lname:
            if "multi_modal_projector" in lname or "visual.merger" in lname:
                continue
            param.requires_grad_(False)
            n += 1
    return n


def load_merit_model(cfg: dict, stage: Stage = "stage2") -> tuple[Any, Any, Any]:
    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained")
    if pretrained is None:
        raise ValueError("cfg.model.pretrained must be set")

    dtype = model_cfg.get("dtype", "bfloat16")
    attn = resolve_attn_implementation(model_cfg.get("attn_implementation"))
    logger.info(
        f"[load_merit_model] pretrained={pretrained} dtype={dtype} attn={attn} stage={stage}"
    )

    spec = ModelSpec(
        pretrained=pretrained,
        dtype=dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        attn_implementation=attn,
        max_pixels=model_cfg.get("max_pixels"),
        min_pixels=model_cfg.get("min_pixels"),
    )
    model, tokenizer, processor = load_qwen_vl(spec)

    # Optional QK-RMSNorm injection (architectural causality experiment).
    # Adds per-head q_norm/k_norm modules with γ=1 to every attention block of
    # the language model; γ becomes a trainable parameter that the optimizer
    # will adjust during the run, analogous to Qwen3's native QK-norm.
    if model_cfg.get("inject_qknorm"):
        from training.models.qknorm_injection import load_qknorm_state_if_present
        logger.info("[load_merit_model] injecting QK-RMSNorm (γ=1 identity init)")
        inject_qknorm(model)
        n_new = sum(p.numel() for n, p in model.named_parameters()
                    if (n.endswith("q_norm.weight") or n.endswith("k_norm.weight")))
        logger.info(f"[load_merit_model] added {n_new} γ parameters via QK-norm injection")
        # If we're resuming from a checkpoint that already has trained γ values
        # (e.g. Stage 2 continues from Stage 1's output), restore those.
        n_loaded = load_qknorm_state_if_present(model, pretrained)
        if n_loaded:
            logger.info(f"[load_merit_model] restored {n_loaded} pre-trained QK-norm tensors from {pretrained}")

    # Always freeze the vision tower (MERIT only trains projector + LM).
    n_vision = freeze_vision_tower(model)
    logger.info(f"[load_merit_model] froze {n_vision} vision parameters")

    if stage == "stage1":
        n_llm = _freeze_llm(model)
        logger.info(
            f"[load_merit_model] froze {n_llm} LLM parameters (stage1: projector-only)"
        )
        # Ensure projector is trainable.
        for p in projector_parameters(model):
            p.requires_grad_(True)
    elif stage == "eval":
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    # stage2 / branch: vision frozen above, everything else trainable (default)

    # Optional projector freeze — useful when stage 2 follows a dedicated
    # projector-warmup phase whose alignment should not be further updated.
    # Applied after the stage-specific defaults so it composes cleanly.
    if model_cfg.get("freeze_projector"):
        n_proj = 0
        for p in projector_parameters(model):
            p.requires_grad_(False)
            n_proj += 1
        logger.info(f"[load_merit_model] froze {n_proj} projector tensors")

    return model, tokenizer, processor


def load_tokenizer_and_processor(cfg: dict) -> tuple[Any, Any]:
    """Load only (tokenizer, processor) — no GPU memory. The processor is the
    full Qwen2.5-VL AutoProcessor; call ``processor.image_processor`` for the
    image-only path."""
    from transformers import AutoProcessor, AutoTokenizer  # type: ignore[import-not-found]

    model_cfg = cfg.get("model", {})
    pretrained = model_cfg["pretrained"]
    trust = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust, use_fast=True)
    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust)
    return tokenizer, processor
