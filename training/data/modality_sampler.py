"""Modality-length-grouped sampler.

Replicates the LLaVA / Elva ``group_by_modality_length=True`` behavior so that
samples within a micro-batch share roughly the same sequence length, minimizing
padding waste. Uses length *buckets* instead of exact sort so the ordering is
still stochastic.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class ModalityLengthGroupedSampler(Sampler[int]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        bucket_size_factor: int = 50,
    ):
        if len(lengths) == 0:
            raise ValueError("empty lengths")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.mega = batch_size * world_size * bucket_size_factor

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed)

        n = len(self.lengths)
        indices = torch.randperm(n, generator=g).tolist()

        # Split into megabatches and sort each by length (descending).
        megabatches: list[list[int]] = [
            indices[i : i + self.mega] for i in range(0, n, self.mega)
        ]
        for mb in megabatches:
            mb.sort(key=lambda i: self.lengths[i], reverse=True)

        # Interleave across ranks: take every world_size-th batch.
        flat: list[int] = [i for mb in megabatches for i in mb]
        # Trim to a multiple of (batch_size * world_size)
        trim = (len(flat) // (self.batch_size * self.world_size)) * (
            self.batch_size * self.world_size
        )
        flat = flat[:trim]

        # Rank-local slice
        local = flat[self.rank :: self.world_size]
        return iter(local)

    def __len__(self) -> int:
        n = len(self.lengths)
        trim = (n // (self.batch_size * self.world_size)) * (
            self.batch_size * self.world_size
        )
        return trim // self.world_size
