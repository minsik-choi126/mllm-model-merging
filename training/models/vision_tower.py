"""Helpers for freezing / accessing the vision encoder.

MERIT does not re-implement the vision encoder — we rely on the one embedded in
Qwen2.5-VL. This module provides two utilities:

  - :func:`freeze_vision_tower`: recursively freezes vision encoder params.
  - :func:`vision_tower_parameters`: enumerates them for optimizer exclusion.
"""
from __future__ import annotations

from torch import nn


_VISION_NAME_PATTERNS = ("visual.", "vision_tower", "vision_model")


def _is_vision_param(name: str) -> bool:
    low = name.lower()
    return any(pat in low for pat in _VISION_NAME_PATTERNS)


def freeze_vision_tower(model: nn.Module) -> int:
    """Set ``requires_grad=False`` on all vision encoder params. Returns count."""
    n = 0
    for name, param in model.named_parameters():
        if _is_vision_param(name) and "merger" not in name.lower():
            param.requires_grad_(False)
            n += 1
    return n


def vision_tower_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if _is_vision_param(name)]
