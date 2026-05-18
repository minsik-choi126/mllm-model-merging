"""Calibration subset builder for gradient extraction.

Given any :class:`torch.utils.data.Dataset` whose items expose a ``task_id``
field, this utility returns a deterministic per-task subset of ``n`` samples
each, chosen via a stride-``s`` walk followed by uniform subsampling.

The paper uses ``n = 200``, ``s = 5`` at 3B scale.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

from torch.utils.data import Dataset, Subset


def build_calibration_set(
    dataset: Dataset,
    *,
    n: int = 200,
    stride: int = 5,
    seed: int = 0,
) -> Subset:
    """Return a torch Subset containing up to ``n`` samples per ``task_id``.

    Strategy:
      1. Walk the dataset with a stride of ``stride`` (ignores 1-of-``stride`` items).
      2. Bucket indices by task_id.
      3. Uniformly sample up to ``n`` indices per task with a fixed seed.
    """
    if n <= 0 or stride <= 0:
        raise ValueError("n and stride must be positive")

    buckets: dict[str, list[int]] = defaultdict(list)
    length = len(dataset)  # type: ignore[arg-type]
    for i in range(0, length, stride):
        sample = dataset[i]
        tid = sample.get("task_id")
        if tid is None:
            continue
        buckets[tid].append(i)

    rng = random.Random(seed)
    selected: list[int] = []
    for tid in sorted(buckets):
        idxs = buckets[tid]
        if len(idxs) > n:
            idxs = rng.sample(idxs, n)
        selected.extend(sorted(idxs))

    return Subset(dataset, selected)


def group_indices_by_task(dataset: Dataset) -> dict[str, list[int]]:
    """Helper for the gradient extractor: task_id -> ordered list of indices."""
    out: dict[str, list[int]] = defaultdict(list)
    for i in range(len(dataset)):  # type: ignore[arg-type]
        sample = dataset[i]
        tid = sample.get("task_id")
        if tid is None:
            continue
        out[tid].append(i)
    return dict(out)
