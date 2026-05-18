"""Thin wrappers over torch.distributed so single-process code paths stay clean."""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def gather_object(obj: Any) -> list[Any]:
    """All-gather arbitrary picklable Python objects onto every rank."""
    if not is_initialized():
        return [obj]
    world_size = get_world_size()
    out: list[Any] = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def synced_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
