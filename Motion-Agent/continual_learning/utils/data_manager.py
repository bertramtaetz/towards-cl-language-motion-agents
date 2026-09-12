"""
Data Manager for O-LoRA Continual Learning

Supports 5-task experiment mode with predefined or random
80/20 splits, mirroring the D-MoLE data manager interface.
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import os
import sys
from pathlib import Path

_MOTION_AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MOTION_AGENT_ROOT / "baselines" / "transfer_learning"))
sys.path.insert(0, str(_MOTION_AGENT_ROOT / "continual_learning" / "common"))

SPLIT_MODE = 'random_80_20'
RANDOM_SEED = 42

# Optional global override for task_splits.json path.
#
# Rationale: we want a single switch that forces *all* continual-learning methods
# (O-LoRA multi, O-LoRA merged, O-LoRA MoE, D-MoLE, baseline eval scripts) to
# consume the same disjoint split file without having to thread a new CLI flag
# through every script.
TASK_SPLITS_ENV = "MOTION_TASK_SPLITS_PATH"

from data_loader import (
    SingleTaskMotionDataset,
    get_single_task_loader,
    TASK_NAMES as TASK_NAMES_5,
    TASK_ORDER as TASK_ORDER_5,
)

try:
    from data_loader_random_split import (
        get_random_split_loader,
        set_task_mode as set_random_split_task_mode,
    )
except ImportError:
    get_random_split_loader = None
    set_random_split_task_mode = None

TASK_NAMES = list(TASK_NAMES_5)
TASK_ORDER = list(TASK_ORDER_5)

TaskFilteredDataset = SingleTaskMotionDataset

__all__ = [
    'TaskFilteredDataset',
    'get_task_loader',
    'TASK_NAMES',
    'TASK_ORDER',
    'get_task_name',
    'set_split_mode',
]


def get_task_name(task_id: int) -> str:
    if 0 <= task_id < len(TASK_NAMES):
        return TASK_NAMES[task_id]
    return f"task_{task_id}"


def set_split_mode(mode: str, seed: int = 42):
    global SPLIT_MODE, RANDOM_SEED
    if mode not in ['predefined', 'random_80_20']:
        raise ValueError(f"Unknown split mode: {mode}")
    SPLIT_MODE = mode
    RANDOM_SEED = seed


def get_task_loader(
    dataset_name,
    split,
    batch_size,
    w_vectorizer,
    task_id,
    num_workers=4,
    unit_length=4,
    task_splits_path=None,
    drop_last=None,
):
    """Get DataLoader for a specific task, routing by split mode."""
    if task_id < 0 or task_id >= len(TASK_NAMES):
        raise ValueError(f"Invalid task_id {task_id}. Must be 0-{len(TASK_NAMES)-1}")

    task_name = TASK_NAMES[task_id]

    # Allow a process-wide split override via environment variable.
    if task_splits_path is None:
        task_splits_path = os.environ.get(TASK_SPLITS_ENV) or None

    if drop_last is None:
        drop_last = (split in ('train', 'trainval'))

    if SPLIT_MODE == 'random_80_20' and get_random_split_loader is not None:
        random_split = 'train' if split in ('train', 'trainval') else 'test'
        return get_random_split_loader(
            task_name=task_name,
            split=random_split,
            batch_size=batch_size,
            w_vectorizer=w_vectorizer,
            unit_length=unit_length,
            num_workers=num_workers,
            task_splits_path=task_splits_path,
            train_ratio=0.8,
            random_seed=RANDOM_SEED,
        )

    loader = get_single_task_loader(
        task_name=task_name,
        split=split,
        batch_size=batch_size,
        w_vectorizer=w_vectorizer,
        unit_length=unit_length,
        num_workers=num_workers,
        task_splits_path=task_splits_path,
    )

    if loader.drop_last != drop_last:
        from torch.utils.data import DataLoader
        from data_loader import collate_fn
        loader = DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

    return loader
