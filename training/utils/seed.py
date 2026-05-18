"""Deterministic seeding helper."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, cudnn_deterministic: bool = False) -> None:
    """Seed Python, numpy, torch CPU & CUDA. Optionally lock cuDNN."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
