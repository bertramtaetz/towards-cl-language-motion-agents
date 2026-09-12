"""Explicit repository-local resources shared by all experiment entrypoints."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = Path(os.environ.get("MOTION_PRETRAINED_ROOT", ROOT / "pretrained")).resolve()
DATA_ROOT = Path(os.environ.get("MOTION_DATA_ROOT", ROOT / "datasets/HumanML3D")).resolve()
TASK_SPLITS = Path(os.environ.get("MOTION_TASK_SPLITS_PATH", ROOT / "benchmark/splits/holdout10/tasks.json")).resolve()
BACKBONE = str(PRETRAINED / "gemma-2-2b-it")
os.environ.setdefault("MOTION_NLG_ROOT", str(PRETRAINED / "nlg"))


def task_order():
    """Use manifest insertion order, not obsolete hard-coded cluster labels."""
    with TASK_SPLITS.open() as stream:
        tasks = json.load(stream)["tasks"]
    return [(task["name"], task["cluster_id"]) for task in tasks.values()]