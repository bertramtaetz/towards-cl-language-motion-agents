"""Multi-task Data Loader with Random 80/20 Split (Upper Bound Baseline)

This module mirrors the continual-learning random split protocol used by
`Motion-Agent/continual_learning/utils/data_manager.py` (which itself relies on
`continual_learning/common/data_loader_random_split.py`).

Why this file exists
--------------------
The original multitask baseline (`baselines/train_multitask.py`) used the
predefined split from `baselines/transfer_learning/task_splits.json`.
However, most continual learning experiments default to `split_mode=random_80_20`
which combines all predefined splits and re-splits 80/20 with a seed.

If multitask is trained on predefined but evaluated on random_80_20, results are
not comparable and can look artificially poor.

This loader enables training multitask baselines on the SAME protocol.

Design
------
- Build one RandomSplitMotionDataset per task (train/test) using the existing
  implementation from D-MoLE.
- Concatenate them into a single dataset.
- Provide an optional stratified batch sampler to keep batches balanced across
  tasks when not using DDP.
"""

from __future__ import annotations

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

# NOTE: We cannot do `from continual_learning.d-mole...` because the folder name
# contains a hyphen, and we also cannot do `import data_loader_random_split`
# after modifying sys.path because THIS FILE is also named
# `data_loader_random_split.py`, which causes a self-import / circular import.
#
# Instead we load the D-MoLE implementation from its file path under a unique
# module name.
_MOTION_AGENT_ROOT = Path(__file__).resolve().parents[2]
_DMOLE_RANDOM_SPLIT_PATH = _MOTION_AGENT_ROOT / "continual_learning" / "common" / "data_loader_random_split.py"
if not _DMOLE_RANDOM_SPLIT_PATH.exists():
    raise FileNotFoundError(f"Expected D-MoLE random split loader at: {_DMOLE_RANDOM_SPLIT_PATH}")

_spec = importlib.util.spec_from_file_location(
    "_dmole_data_loader_random_split",
    _DMOLE_RANDOM_SPLIT_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to create module spec for {_DMOLE_RANDOM_SPLIT_PATH}")

_dmole_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dmole_mod)

RandomSplitMotionDataset = _dmole_mod.RandomSplitMotionDataset
TASK_NAMES = _dmole_mod.TASK_NAMES


def collate_fn(batch):
    """Sort batch by sentence length (descending) for efficient RNN packing."""
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class _IndexedDataset(data.Dataset):
    """Wrap a dataset and expose its task_id for each sample."""

    def __init__(self, ds: data.Dataset, task_id: int):
        self.ds = ds
        self.task_id = task_id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Forward the original sample tuple unchanged.
        return self.ds[idx]


@dataclass(frozen=True)
class _SampleRef:
    task_id: int
    local_index: int


class RandomSplitMultiTaskDataset(data.Dataset):
    """Concatenate per-task random-split datasets with task tracking."""

    def __init__(
        self,
        split: str,
        w_vectorizer,
        task_splits_path: str,
        train_ratio: float,
        random_seed: int,
        unit_length: int = 4,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test' for random_80_20, got: {split}")

        self.split = split
        self.task_to_indices: Dict[int, List[int]] = {}

        self._datasets: List[_IndexedDataset] = []
        self._index: List[_SampleRef] = []

        for task_id, task_name in enumerate(TASK_NAMES):
            ds = RandomSplitMotionDataset(
                task_name=task_name,
                split=split,
                w_vectorizer=w_vectorizer,
                task_splits_path=task_splits_path,
                unit_length=unit_length,
                train_ratio=train_ratio,
                random_seed=random_seed,
            )
            wrapped = _IndexedDataset(ds, task_id=task_id)
            self._datasets.append(wrapped)

            self.task_to_indices[task_id] = []
            for j in range(len(wrapped)):
                global_idx = len(self._index)
                self._index.append(_SampleRef(task_id=task_id, local_index=j))
                self.task_to_indices[task_id].append(global_idx)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ref = self._index[idx]
        return self._datasets[ref.task_id][ref.local_index]


class StratifiedBatchSampler(data.Sampler[List[int]]):
    """Ensure each batch contains equal samples from each task.

    Note: this is only used for single-process training because custom
    batch_samplers interact poorly with Accelerate DDP.
    """

    def __init__(self, task_to_indices: Dict[int, Sequence[int]], batch_size: int, drop_last: bool = True):
        self.task_to_indices = {k: list(v) for k, v in task_to_indices.items()}
        self.num_tasks = len(self.task_to_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.samples_per_task = max(1, batch_size // self.num_tasks)
        self.actual_batch_size = self.samples_per_task * self.num_tasks

        min_task_size = min(len(v) for v in self.task_to_indices.values())
        self.num_batches = min_task_size // self.samples_per_task

    def __iter__(self) -> Iterator[List[int]]:
        shuffled = {k: random.sample(v, len(v)) for k, v in self.task_to_indices.items()}
        for batch_idx in range(self.num_batches):
            batch: List[int] = []
            for task_id in range(self.num_tasks):
                start = batch_idx * self.samples_per_task
                end = start + self.samples_per_task
                batch.extend(shuffled[task_id][start:end])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class RoundRobinTaskBatchSampler(data.Sampler[List[int]]):
    """Balanced sampling across tasks even when batch_size < num_tasks.

    Instead of enforcing "one sample per task per batch" (impossible when
    micro-batch < num_tasks without changing effective batch size), this sampler
    yields batches drawn from a single task at a time but cycles tasks in a
    round-robin order.

    Effect:
    - Equal number of batches per task per epoch
    - Preserves the requested batch_size exactly
    - Avoids one task dominating updates when tasks have different dataset sizes

    Note: Used only for single-process training. Custom batch_samplers interact
    poorly with Accelerate DDP.
    """

    def __init__(self, task_to_indices: Dict[int, Sequence[int]], batch_size: int, drop_last: bool = True):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.task_to_indices = {k: list(v) for k, v in task_to_indices.items()}
        self.num_tasks = len(self.task_to_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # To keep tasks balanced, cap to the smallest task.
        min_task_size = min(len(v) for v in self.task_to_indices.values())
        self.num_batches_per_task = min_task_size // self.batch_size
        self.num_batches = self.num_batches_per_task * self.num_tasks

    def __iter__(self) -> Iterator[List[int]]:
        shuffled = {k: random.sample(v, len(v)) for k, v in self.task_to_indices.items()}
        for batch_idx in range(self.num_batches_per_task):
            for task_id in range(self.num_tasks):
                start = batch_idx * self.batch_size
                end = start + self.batch_size
                batch = shuffled[task_id][start:end]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        return self.num_batches


def get_multitask_random_split_loader(
    split: str,
    batch_size: int,
    w_vectorizer,
    unit_length: int,
    num_workers: int,
    task_splits_path: str,
    train_ratio: float,
    random_seed: int,
    stratified: bool,
    distributed: bool,
):
    ds = RandomSplitMultiTaskDataset(
        split=split,
        w_vectorizer=w_vectorizer,
        task_splits_path=task_splits_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
        unit_length=unit_length,
    )

    # IMPORTANT: We provide two *balanced* sampling modes:
    # 1) StratifiedBatchSampler: each batch contains equal samples from each task
    #    (requires batch_size >= num_tasks)
    # 2) RoundRobinTaskBatchSampler: cycles tasks across batches to keep the
    #    number of updates per task balanced even when batch_size < num_tasks.
    #
    # Both are single-process only; Accelerate DDP expects to own the sampler.
    if stratified and split == "train" and not distributed:
        if batch_size >= len(ds.task_to_indices):
            sampler = StratifiedBatchSampler(ds.task_to_indices, batch_size=batch_size, drop_last=True)
            sampling_mode = "stratified_per_batch"
        else:
            sampler = RoundRobinTaskBatchSampler(ds.task_to_indices, batch_size=batch_size, drop_last=True)
            sampling_mode = "round_robin_batches"

        loader = data.DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        # Best-effort metadata for logging/debugging.
        setattr(loader, "sampling_mode", sampling_mode)
        return loader

    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
    setattr(loader, "sampling_mode", "shuffle" if split == "train" else "sequential")
    if distributed and stratified and split == "train":
        # Caller should warn; we keep the loader valid.
        setattr(loader, "sampling_mode", "shuffle_ddp_unbalanced")
    return loader
