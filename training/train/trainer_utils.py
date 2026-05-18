"""Helpers shared by Stage 1 / Stage 2 / Branch trainers."""
from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.train.arguments import MeritConfig, TrainingArgs
from training.utils.io import ensure_dir, safe_json_dump


def _promote_trainable_to_fp32(model: Any) -> int:
    """Cast trainable fp16 parameters to fp32 in-place.

    ``transformers.Trainer`` with ``fp16=True`` relies on ``GradScaler.unscale_``,
    which refuses to operate on fp16 gradients. When the model is loaded in
    ``torch_dtype=float16`` (the V100 path) we must keep the master copy of
    trainable parameters in fp32 so autocast + scaler behave correctly.
    """
    import torch

    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.to(torch.float32)
            n += 1
    return n


def build_hf_trainer(
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
) -> Any:
    """Build a ``transformers.Trainer`` using the kwarg name the installed
    version exposes (``tokenizer=`` on 4.x, ``processing_class=`` on 5.x).

    Also upcasts any trainable fp16 params to fp32 when the user requested
    fp16 mixed-precision training so ``GradScaler.unscale_`` does not trip.
    """
    from transformers import Trainer  # type: ignore[import-not-found]

    if getattr(args, "fp16", False) and not getattr(args, "bf16", False):
        _promote_trainable_to_fp32(model)

    # HF Trainer passes the entire collator output to model.forward(...) when
    # remove_unused_columns=False. Our collator emits ``task_ids`` for downstream
    # grouping / logging; drop it here so model.forward() does not receive an
    # unexpected kwarg. Original collator stays intact for non-HF callers.
    inner_collator = data_collator

    def _drop_task_ids_collator(features):
        out = inner_collator(features)
        out.pop("task_ids", None)
        return out

    sig = inspect.signature(Trainer.__init__)
    kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=_drop_task_ids_collator,
    )
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def to_hf_training_args(train: TrainingArgs) -> Any:
    """Materialize a ``transformers.TrainingArguments`` from MERIT's TrainingArgs.

    Importing ``transformers`` lazily so that ``merit`` remains importable in
    environments where transformers is unavailable (e.g. docs build).
    """
    from transformers import TrainingArguments  # type: ignore[import-not-found]

    kwargs: dict[str, Any] = dict(
        output_dir=train.output_dir,
        seed=train.seed,
        optim=train.optim,
        learning_rate=train.learning_rate,
        weight_decay=train.weight_decay,
        max_grad_norm=train.max_grad_norm,
        warmup_ratio=train.warmup_ratio,
        lr_scheduler_type=train.lr_scheduler_type,
        num_train_epochs=train.num_train_epochs,
        max_steps=train.max_steps,
        per_device_train_batch_size=train.per_device_train_batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        bf16=train.bf16,
        fp16=train.fp16,
        logging_steps=train.logging_steps,
        save_strategy=train.save_strategy,
        save_steps=train.save_steps,
        save_total_limit=train.save_total_limit,
        dataloader_num_workers=train.dataloader_num_workers,
        gradient_checkpointing=train.gradient_checkpointing,
        report_to=train.report_to,
        run_name=train.run_name,
        remove_unused_columns=False,
        deepspeed=train.deepspeed,
        ddp_timeout=train.ddp_timeout,
    )
    return TrainingArguments(**kwargs)


def dump_effective_config(cfg: MeritConfig, out_dir: str) -> None:
    """Write the effective config (post-YAML-merge) to <out_dir>/effective_config.json."""
    out = ensure_dir(out_dir)
    safe_json_dump(
        {
            "model": asdict(cfg.model),
            "data": asdict(cfg.data),
            "train": asdict(cfg.train),
            "merit": asdict(cfg.merit),
        },
        out / "effective_config.json",
    )
