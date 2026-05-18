"""Stage 1: projector alignment (LLaVA-style).

Trains ONLY the multimodal projector. Vision encoder and LLM are frozen. Writes
a single ``mm_projector.bin`` + tokenizer config to the output directory.

Usage::

    from training.train.stage1_trainer import run_stage1
    run_stage1(cfg)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.data import DataCollatorForMERIT, get_dataset
from training.models import load_merit_model
from training.models.projector import projector_parameters
from training.train.arguments import MeritConfig
from training.train.trainer_utils import build_hf_trainer, dump_effective_config, to_hf_training_args
from training.utils.logging import get_logger
from training.utils.seed import set_seed

logger = get_logger(__name__)


def _projector_only_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    sd = {}
    for name, p in model.named_parameters():
        lname = name.lower()
        if "multi_modal_projector" in lname or "mm_projector" in lname or "visual.merger" in lname:
            sd[name] = p.detach().cpu().clone()
    return sd


def run_stage1(cfg: MeritConfig) -> None:
    set_seed(cfg.train.seed)

    # 1. Model (projector only trainable)
    model, tokenizer, processor = load_merit_model(
        {"model": cfg.model.__dict__}, stage="stage1"
    )

    # 2. Data
    ds_kwargs: dict[str, Any] = {
        "root": cfg.data.root,
        "split": cfg.data.split,
        "max_samples": cfg.data.max_samples,
        "allowed_task_ids": cfg.data.allowed_task_ids,
    }
    train_ds = get_dataset(cfg.data.name, **ds_kwargs)
    collator = DataCollatorForMERIT(
        tokenizer=tokenizer,
        processor=processor,
        max_length=cfg.data.max_length,
    )

    # 3. HF training args
    hf_args = to_hf_training_args(cfg.train)

    # 4. Trainer
    trainer = build_hf_trainer(
        model=model,
        args=hf_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 5. Write effective config for reproducibility
    dump_effective_config(cfg, cfg.train.output_dir)

    # 6. Train
    trainer.train()

    # 7. Save projector-only checkpoint (much smaller than the full model)
    proj_sd = _projector_only_state_dict(model)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(proj_sd, out / "mm_projector.bin")
    logger.info(f"[stage1] saved projector checkpoint to {out / 'mm_projector.bin'}")

    # Also save tokenizer + processor so downstream steps can load them directly.
    tokenizer.save_pretrained(out)
    processor.save_pretrained(out)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.save_pretrained(out)
