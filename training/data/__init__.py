"""MERIT data package.

Public entry point is :func:`get_dataset`, which dispatches to the
Vision-FLAN 136, 176-source multimodal mixture, or text-only FLAN 66 loaders.
Every dataset exposes a per-sample ``task_id`` (or ``source_id``) field so the
preprocessing stage can aggregate gradients at the dataset-unit level.
"""
from training.data.calibration import build_calibration_set
from training.data.collator import DataCollatorForMERIT
from training.data.conversation import build_qwen_chat
from training.data.modality_sampler import ModalityLengthGroupedSampler
from training.data.registry import get_dataset, list_datasets, register_dataset

# Register built-in datasets.
from training.data import vflan136  # noqa: F401
from training.data import mix176  # noqa: F401
from training.data import flan66  # noqa: F401
from training.data import jsonl  # noqa: F401
from training.data import webdataset_tar  # noqa: F401
from training.data import openai_chat_jsonl  # noqa: F401
from training.data import qa_pairs_webdataset  # noqa: F401

__all__ = [
    "build_calibration_set",
    "build_qwen_chat",
    "DataCollatorForMERIT",
    "ModalityLengthGroupedSampler",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
