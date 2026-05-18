"""Resolve the ``attn_implementation`` string from config + hardware.

Priority:
  1. If config explicitly sets ``model.attn_implementation``, use that (unless
     the requested backend is not installed, in which case we fall back).
  2. Otherwise, choose the best backend available on the current GPU:
       - flash_attention_2 (if flash-attn installed and bf16 supported)
       - sdpa             (PyTorch-native, works everywhere on torch>=2.1)
       - eager            (last resort)
"""
from __future__ import annotations

import importlib.util
from typing import Literal

import torch

Implementation = Literal["eager", "sdpa", "flash_attention_2"]


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 8  # Ampere+


def resolve_attn_implementation(
    requested: str | None = None,
    *,
    strict: bool = False,
) -> Implementation:
    """Pick an attention backend. If ``strict`` is True, raise when the requested
    backend is unavailable; otherwise silently fall back."""
    if requested == "flash_attention_2":
        if _flash_attn_available() and _bf16_supported():
            return "flash_attention_2"
        if strict:
            raise RuntimeError(
                "flash_attention_2 requested but flash-attn not installed or GPU does not support bf16"
            )
        return "sdpa"
    if requested in ("eager", "sdpa"):
        return requested  # type: ignore[return-value]
    # requested is None → auto
    if _flash_attn_available() and _bf16_supported():
        return "flash_attention_2"
    return "sdpa"
