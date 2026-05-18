"""Checkpoint shape / name validation for merge-ready branch fusion.

Before averaging branch checkpoints, we verify that they all expose the same
parameter names and shapes. This catches silent failures where one branch was
trained with a different dtype, a different parallelism config, or a stale
merge-ready init.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


def load_param_names(ckpt_path: str | Path) -> Dict[str, Tuple[int, ...]]:
    """Load parameter name → shape mapping from a HuggingFace-style checkpoint.

    Supports both safetensors and torch state_dict formats, plus sharded
    layouts with ``pytorch_model.bin.index.json`` or
    ``model.safetensors.index.json``.
    """
    path = Path(ckpt_path)
    # Single-file paths
    single_bin = path / "pytorch_model.bin"
    single_st = path / "model.safetensors"
    index_bin = path / "pytorch_model.bin.index.json"
    index_st = path / "model.safetensors.index.json"

    if single_st.exists():
        from safetensors import safe_open  # type: ignore[import-not-found]

        out: Dict[str, Tuple[int, ...]] = {}
        with safe_open(str(single_st), framework="pt") as f:
            for key in f.keys():
                out[key] = tuple(f.get_tensor(key).shape)
        return out

    if single_bin.exists():
        sd = torch.load(single_bin, map_location="cpu", weights_only=True)
        return {k: tuple(v.shape) for k, v in sd.items()}

    if index_st.exists():
        import json  # local import
        from safetensors import safe_open  # type: ignore[import-not-found]

        index = json.loads(index_st.read_text())
        weight_map: Dict[str, str] = index["weight_map"]
        out = {}
        seen_files = set()
        for k, fname in weight_map.items():
            if fname not in seen_files:
                with safe_open(str(path / fname), framework="pt") as f:
                    for kk in f.keys():
                        out[kk] = tuple(f.get_tensor(kk).shape)
                seen_files.add(fname)
        return out

    if index_bin.exists():
        import json

        index = json.loads(index_bin.read_text())
        weight_map = index["weight_map"]
        out = {}
        seen_files = set()
        for k, fname in weight_map.items():
            if fname in seen_files:
                continue
            sd = torch.load(path / fname, map_location="cpu", weights_only=True)
            for kk, vv in sd.items():
                out[kk] = tuple(vv.shape)
            seen_files.add(fname)
        return out

    raise FileNotFoundError(f"no recognizable checkpoint at {path}")


def assert_merge_ready(branch_ckpts: Iterable[str | Path]) -> None:
    """Verify every branch checkpoint has identical param names and shapes.

    Raises :class:`ValueError` with a detailed diff if any mismatch is found.
    """
    branches: List[Path] = [Path(b) for b in branch_ckpts]
    if len(branches) < 2:
        return  # trivially merge-ready

    reference = load_param_names(branches[0])
    ref_keys = set(reference)
    for b in branches[1:]:
        cand = load_param_names(b)
        cand_keys = set(cand)
        missing = ref_keys - cand_keys
        extra = cand_keys - ref_keys
        if missing or extra:
            raise ValueError(
                f"branch {b} differs from {branches[0]}: "
                f"missing={sorted(missing)[:5]}... extra={sorted(extra)[:5]}..."
            )
        # Shape check
        bad_shapes = [
            (k, reference[k], cand[k]) for k in ref_keys if reference[k] != cand[k]
        ]
        if bad_shapes:
            first = bad_shapes[:3]
            raise ValueError(f"branch {b} has mismatched shapes: {first}")
