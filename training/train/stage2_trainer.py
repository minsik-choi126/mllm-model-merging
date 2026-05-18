"""Stage 2: full instruction tuning.

Vision encoder remains frozen; projector + LLM are trained on the full
instruction mixture. Output directory holds the **merge-ready initialization**
``θ⁰`` that subsequent branch training forks from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.data import DataCollatorForMERIT, get_dataset
from training.models import load_merit_model
from training.train.arguments import MeritConfig
from training.train.trainer_utils import build_hf_trainer, dump_effective_config, to_hf_training_args
from training.utils.logging import get_logger
from training.utils.seed import set_seed

logger = get_logger(__name__)


def _load_projector_from_stage1(model: Any, stage1_dir: str | Path) -> int:
    """If the user points stage2 at a stage1 output, pull the projector weights."""
    p = Path(stage1_dir) / "mm_projector.bin"
    if not p.exists():
        return 0
    sd = torch.load(p, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(
        f"[stage2] loaded projector from {p}: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return len(sd)


def run_stage2(cfg: MeritConfig, *, pretrain_projector: str | None = None) -> None:
    set_seed(cfg.train.seed)

    model, tokenizer, processor = load_merit_model(
        {"model": cfg.model.__dict__}, stage="stage2"
    )

    if pretrain_projector:
        _load_projector_from_stage1(model, pretrain_projector)

    train_ds = get_dataset(
        cfg.data.name,
        root=cfg.data.root,
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        allowed_task_ids=cfg.data.allowed_task_ids,
    )
    collator = DataCollatorForMERIT(
        tokenizer=tokenizer,
        processor=processor,
        max_length=cfg.data.max_length,
    )

    hf_args = to_hf_training_args(cfg.train)
    trainer = build_hf_trainer(
        model=model,
        args=hf_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    dump_effective_config(cfg, cfg.train.output_dir)
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    processor.save_pretrained(cfg.train.output_dir)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.save_pretrained(cfg.train.output_dir)
    logger.info(f"[stage2] saved merge-ready init to {cfg.train.output_dir}")
