"""Branch training: forks from the merge-ready init θ⁰ and trains on a single group.

Reads ``group_assignment.json`` produced by the preprocess stage, subsets the
dataset to the task_ids belonging to the requested branch, then runs a Stage 2
training loop that starts from ``cfg.merit.init_from`` instead of the raw base
checkpoint. Output lands in ``<branch_output_root>/branch_{k}/``.

No cross-branch communication happens: branches are independent processes
(or sequential runs), and the merge step (``merit.merge``) takes the union.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from training.data import DataCollatorForMERIT, get_dataset
from training.models import load_merit_model
from training.train.arguments import MeritConfig
from training.train.trainer_utils import build_hf_trainer, dump_effective_config, to_hf_training_args
from training.utils.io import safe_json_load
from training.utils.logging import get_logger
from training.utils.seed import set_seed

logger = get_logger(__name__)


def _group_task_ids(assignment_path: str | Path, branch_id: int) -> list[str]:
    data = safe_json_load(assignment_path)
    groups = data["groups"]
    if branch_id < 0 or branch_id >= len(groups):
        raise IndexError(
            f"branch_id={branch_id} out of range for K={len(groups)} groups"
        )
    return groups[branch_id]["task_ids"]


def run_branch(cfg: MeritConfig, *, branch_id: int) -> None:
    if cfg.merit.init_from is None:
        raise ValueError("branch training requires merit.init_from")
    if cfg.merit.group_assignment is None:
        raise ValueError("branch training requires merit.group_assignment")
    if cfg.merit.branch_output_root is None:
        raise ValueError("branch training requires merit.branch_output_root")

    set_seed(cfg.train.seed + branch_id)  # decorrelate branch seeds

    task_ids = _group_task_ids(cfg.merit.group_assignment, branch_id)
    logger.info(
        f"[branch {branch_id}] {len(task_ids)} task_ids: {task_ids[:5]}"
        f"{'...' if len(task_ids) > 5 else ''}"
    )

    # 1. Load model from the merge-ready init (NOT from the base Qwen checkpoint)
    model_cfg = dict(cfg.model.__dict__)
    model_cfg["pretrained"] = cfg.merit.init_from
    model, tokenizer, processor = load_merit_model({"model": model_cfg}, stage="branch")

    # 2. Subset dataset to this group's task_ids
    dataset = get_dataset(
        cfg.data.name,
        root=cfg.data.root,
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        allowed_task_ids=task_ids,
    )
    collator = DataCollatorForMERIT(
        tokenizer=tokenizer,
        processor=processor,
        max_length=cfg.data.max_length,
    )

    # 3. Override output_dir per branch
    branch_out = Path(cfg.merit.branch_output_root) / f"branch_{branch_id}"
    cfg.train.output_dir = str(branch_out)
    cfg.train.run_name = f"{cfg.train.run_name}-b{branch_id}"

    hf_args = to_hf_training_args(cfg.train)
    trainer = build_hf_trainer(
        model=model,
        args=hf_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    dump_effective_config(cfg, cfg.train.output_dir)
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    # Processor files (preprocessor_config.json + chat template) are required for
    # downstream merge + inference; ``save_model`` only writes model weights.
    # In transformers 5.x ``processor.save_pretrained`` writes ``processor_config.json``
    # but not ``preprocessor_config.json`` — save the image processor separately
    # so the merged directory is directly loadable with ``AutoProcessor``.
    processor.save_pretrained(cfg.train.output_dir)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.save_pretrained(cfg.train.output_dir)
    logger.info(f"[branch {branch_id}] saved to {cfg.train.output_dir}")
