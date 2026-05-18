"""Unified training CLI.

Usage::

    python -m merit.train.cli --config configs/3b/stage1.yaml
    python -m merit.train.cli --config configs/3b/stage2.yaml \\
        --pretrain-projector ckpts/3b/stage1

Stage selection is implicit: if ``model.tune_mm_mlp_adapter: true``, Stage 1
trainer runs; otherwise Stage 2. Branch training is handled by
``merit.train.branch_trainer.run_branch``.
"""
from __future__ import annotations

import argparse
import sys

from training.train.arguments import dataclasses_from_cfg
from training.utils.hydra_helpers import load_yaml
from training.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merit-train")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument(
        "--pretrain-projector",
        default=None,
        help="Directory containing mm_projector.bin (stage2 only)",
    )
    ap.add_argument(
        "--branch-id",
        type=int,
        default=None,
        help="When set, runs the branch trainer on the given group index",
    )
    args = ap.parse_args(argv)

    cfg_dict = load_yaml(args.config)
    cfg = dataclasses_from_cfg(cfg_dict)

    if args.branch_id is not None:
        from training.train.branch_trainer import run_branch

        run_branch(cfg, branch_id=args.branch_id)
        return 0

    if cfg.model.tune_mm_mlp_adapter:
        from training.train.stage1_trainer import run_stage1

        run_stage1(cfg)
    else:
        from training.train.stage2_trainer import run_stage2

        run_stage2(cfg, pretrain_projector=args.pretrain_projector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
