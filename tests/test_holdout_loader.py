"""Verify actual loader filtering/splitting, independently of pretrained inference."""
import json
from pathlib import Path

import numpy as np

from continual_learning.common.data_loader_random_split import RandomSplitMotionDataset

ROOT = Path(__file__).resolve().parent.parent


def test_original_random_split(tmp_path):
    (tmp_path / "texts").mkdir()
    (tmp_path / "new_joint_vecs").mkdir()
    ids = [f"sample_{i:03}" for i in range(20)]
    for mid in ids:
        np.save(tmp_path / "new_joint_vecs" / f"{mid}.npy", np.zeros((64, 263), dtype=np.float32))
        (tmp_path / "texts" / f"{mid}.txt").write_text("a person jumps#a/DET person/NOUN jumps/VERB#0.0#0.0\n")
    manifest = json.loads((ROOT / "benchmark/splits/holdout10/tasks.json").read_text())
    task = next(iter(manifest["tasks"].values()))
    task.update(train=ids, val=[], test=[])
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(manifest))
    common = dict(task_name=task["name"], w_vectorizer=None,
                  task_splits_path=str(path), data_root=str(tmp_path), random_seed=42)
    train = RandomSplitMotionDataset(split="train", **common)
    test = RandomSplitMotionDataset(split="test", **common)
    indices = np.arange(20)
    np.random.RandomState(42).shuffle(indices)
    assert train.name_list == [ids[i] for i in indices[:16]]
    assert test.name_list == [ids[i] for i in indices[16:]]
    assert set(train.name_list).isdisjoint(test.name_list)