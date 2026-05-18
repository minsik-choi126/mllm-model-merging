"""Training argument dataclasses bound from YAML.

Structure mirrors Elva's model/data/training dataclass split but we use YAML
for human-readability. The YAML loader (``merit.utils.hydra_helpers.load_yaml``)
returns a nested dict; :func:`dataclasses_from_cfg` then materializes strongly
typed dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelArgs:
    pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None          # None = auto
    tune_mm_mlp_adapter: bool = False                  # Stage 1 flag
    freeze_llm: bool = False                           # Stage 1 flag
    freeze_projector: bool = False                     # freeze ``visual.merger`` (LM-only fine-tune)
    # Qwen2.5-VL dynamic-resolution bounds: caps the per-image patch count so
    # high-res images don't blow up the sequence length. Bytes per image is
    # roughly H*W (so e.g. 1920*1080 ≈ 2.07M pixels → ≈2580 merged tokens).
    max_pixels: Optional[int] = None                   # None = processor default
    min_pixels: Optional[int] = None                   # None = processor default
    # C3 architectural-causality experiment: inject QK-RMSNorm modules into
    # the LLM's attention layers at load time. When True, loader.py adds
    # per-head q_norm / k_norm (γ=1 init) to every self_attn block.
    inject_qknorm: bool = False


@dataclass
class DataArgs:
    name: str = "vflan136"
    root: str = "data/vision_flan"
    split: str = "train"
    max_length: int = 4096
    max_samples: Optional[int] = None                  # smoke-test cap
    allowed_task_ids: Optional[list[str]] = None       # branch subset


@dataclass
class TrainingArgs:
    output_dir: str = "ckpts/run"
    deepspeed: Optional[str] = None                    # path to zero2.json
    seed: int = 42

    # optimizer
    optim: str = "adamw_torch"                          # any HF-recognised optim (e.g. "adafactor")
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # schedule
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # precision
    bf16: bool = True
    fp16: bool = False

    # distributed
    # NCCL collective-op watchdog timeout (seconds). HF default is 30 min; bump
    # this when a slow shared filesystem can stall a DataLoader worker for
    # longer than that — otherwise an idle rank's all_reduce will time out and
    # the whole job crashes mid-stage.  7200 s (2 h) is a sensible upper
    # bound for the 7B chain on network-mounted shared storage.
    ddp_timeout: int = 1800

    # logging / checkpointing
    logging_steps: int = 1
    save_strategy: str = "epoch"
    save_steps: int = 1000
    save_total_limit: int = 1
    dataloader_num_workers: int = 2
    gradient_checkpointing: bool = True
    report_to: str = "none"
    run_name: str = "merit-run"

    # LLaVA-style modality-length grouping
    group_by_modality_length: bool = True


@dataclass
class MeritArgs:
    """MERIT-specific knobs used by preprocess / branch / merge stages."""

    r: int = 0                          # 0 = joint baseline, >=1 = PCA dims
    init_from: Optional[str] = None     # path to merge-ready init (stage2 ckpt)
    group_assignment: Optional[str] = None  # path to group_assignment.json
    branch_output_root: Optional[str] = None
    calibration_size: int = 200         # n
    calibration_stride: int = 5         # s
    grad_proj_dim: int = 8192           # JL projection dim
    grad_layer_filter: str = "attention"  # "attention" | "all" | regex
    merged_output_dir: Optional[str] = None


@dataclass
class MeritConfig:
    model: ModelArgs = field(default_factory=ModelArgs)
    data: DataArgs = field(default_factory=DataArgs)
    train: TrainingArgs = field(default_factory=TrainingArgs)
    merit: MeritArgs = field(default_factory=MeritArgs)


def _bind(section: dict[str, Any] | None, dc_cls: type) -> Any:
    section = section or {}
    # Keep only recognized fields so unknown YAML keys become errors.
    from dataclasses import fields

    valid = {f.name for f in fields(dc_cls)}
    unknown = set(section) - valid
    if unknown:
        raise ValueError(f"unknown keys for {dc_cls.__name__}: {sorted(unknown)}")
    return dc_cls(**{k: v for k, v in section.items() if k in valid})


def dataclasses_from_cfg(cfg: dict[str, Any]) -> MeritConfig:
    return MeritConfig(
        model=_bind(cfg.get("model"), ModelArgs),
        data=_bind(cfg.get("data"), DataArgs),
        train=_bind(cfg.get("train"), TrainingArgs),
        merit=_bind(cfg.get("merit"), MeritArgs),
    )
