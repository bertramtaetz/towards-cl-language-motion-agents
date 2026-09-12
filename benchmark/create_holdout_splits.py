"""Utility: Create disjoint holdout/CL task split files.

Goal
----
Create a *per-task* holdout split (e.g. 10% of motion_ids per task) for
minimal pretraining, and a complementary CL split (remaining 90%) that is
guaranteed to exclude all holdout motion_ids.

Why this exists
---------------
All continual-learning methods in this repository ultimately draw motion_ids
from `baselines/transfer_learning/task_splits.json` (either directly via
predefined train/val/test or indirectly via `random_80_20` which unions all
splits and re-splits).

By generating two split files with the same schema we can:
1) Pretrain a "minimal motion interface" on holdout motion_ids.
2) Run the entire CL + baseline pipeline on the remaining motion_ids.

Disjointness guarantee
----------------------
We enforce *global* disjointness:
If a motion_id is selected for holdout for any task, it is removed from ALL
tasks in the CL split file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class SplitPaths:
    input_path: Path
    out_holdout: Path
    out_cl: Path


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _gather_task_motion_ids(task_info: dict) -> List[str]:
    all_ids: List[str] = []
    for k in ("train", "val", "test"):
        all_ids.extend(task_info.get(k, []) or [])
    return _unique_preserve_order(all_ids)


def _sample_holdout(ids: List[str], ratio: float, seed: int) -> List[str]:
    if not ids:
        return []
    if ratio <= 0:
        return []
    if ratio >= 1:
        return list(ids)

    import numpy as np

    n = len(ids)
    n_holdout = max(1, int(round(n * ratio)))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n_holdout, replace=False)
    idx = sorted(int(i) for i in idx)
    return [ids[i] for i in idx]


def make_disjoint_splits(
    src: dict,
    holdout_ratio: float,
    seed: int,
) -> Tuple[dict, dict]:
    """Return (holdout_json, cl_json)."""

    if "tasks" not in src or not isinstance(src["tasks"], dict):
        raise ValueError("Input JSON must contain a top-level 'tasks' dict")

    tasks: Dict[str, dict] = src["tasks"]

    per_task_holdout: Dict[str, Set[str]] = {}
    global_holdout: Set[str] = set()

    # 1) Select holdout motion_ids per task (using task-specific seed offset).
    for task_key, task_info in tasks.items():
        ids = _gather_task_motion_ids(task_info)
        # IMPORTANT: Use a stable per-task seed offset.
        # Do NOT use Python's built-in hash() (salted per process).
        # task_key is of the form "task_1", "task_2", ...
        try:
            task_idx = int(str(task_key).split("_")[-1])
        except Exception:
            task_idx = 0
        task_seed = seed + (task_idx * 10_000)
        task_holdout = set(_sample_holdout(ids, ratio=holdout_ratio, seed=task_seed))
        per_task_holdout[task_key] = task_holdout
        global_holdout |= task_holdout

    # 2) Build holdout file: keep schema, store holdout ids in train.
    holdout = json.loads(json.dumps(src))  # deep copy via json
    holdout["ordering_rationale"] = (holdout.get("ordering_rationale", "") + "\n\n"
                                    "[Auto-generated] 10% holdout split for minimal pretraining.")
    holdout["holdout_ratio"] = holdout_ratio
    holdout["holdout_seed"] = seed
    holdout["holdout_global_ids"] = sorted(global_holdout)
    for task_key, task_info in holdout["tasks"].items():
        ids = sorted(per_task_holdout.get(task_key, set()))
        task_info["train"] = ids
        task_info["val"] = []
        task_info["test"] = []
        task_info["n_train"] = len(ids)
        task_info["n_val"] = 0
        task_info["n_test"] = 0
        task_info["n_total"] = len(ids)

    # 3) Build CL file: remove global holdout ids from every split list.
    cl = json.loads(json.dumps(src))
    cl["ordering_rationale"] = (cl.get("ordering_rationale", "") + "\n\n"
                               "[Auto-generated] Complementary split excluding holdout motion_ids.")
    cl["excluded_holdout_ratio"] = holdout_ratio
    cl["excluded_holdout_seed"] = seed
    cl["excluded_holdout_global_ids"] = sorted(global_holdout)
    for task_key, task_info in cl["tasks"].items():
        for split_name in ("train", "val", "test"):
            orig = task_info.get(split_name, []) or []
            task_info[split_name] = [mid for mid in orig if mid not in global_holdout]
        task_info["n_train"] = len(task_info.get("train", []) or [])
        task_info["n_val"] = len(task_info.get("val", []) or [])
        task_info["n_test"] = len(task_info.get("test", []) or [])
        task_info["n_total"] = task_info["n_train"] + task_info["n_val"] + task_info["n_test"]

    return holdout, cl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate disjoint holdout/CL task split JSONs")
    p.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "baselines" / "transfer_learning" / "task_splits.json"),
        help="Input task_splits.json",
    )
    p.add_argument(
        "--out-holdout",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "baselines" / "transfer_learning" / "task_splits_holdout10.json"),
        help="Output holdout split JSON",
    )
    p.add_argument(
        "--out-cl",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "baselines" / "transfer_learning" / "task_splits_cl90.json"),
        help="Output CL split JSON",
    )
    p.add_argument("--holdout-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    paths = SplitPaths(
        input_path=Path(args.input).resolve(),
        out_holdout=Path(args.out_holdout).resolve(),
        out_cl=Path(args.out_cl).resolve(),
    )

    src = json.loads(paths.input_path.read_text())
    holdout, cl = make_disjoint_splits(src, holdout_ratio=float(args.holdout_ratio), seed=int(args.seed))

    paths.out_holdout.parent.mkdir(parents=True, exist_ok=True)
    paths.out_cl.parent.mkdir(parents=True, exist_ok=True)
    paths.out_holdout.write_text(json.dumps(holdout, indent=2))
    paths.out_cl.write_text(json.dumps(cl, indent=2))

    # Basic integrity check: ensure disjoint motion_id sets.
    def _all_ids(doc: dict) -> Set[str]:
        out: Set[str] = set()
        for ti in doc.get("tasks", {}).values():
            for s in ("train", "val", "test"):
                out |= set(ti.get(s, []) or [])
        return out

    holdout_ids = _all_ids(holdout)
    cl_ids = _all_ids(cl)
    overlap = holdout_ids & cl_ids
    if overlap:
        raise RuntimeError(f"Split generation failed: overlap detected ({len(overlap)} ids).")

    print("Generated disjoint split files:")
    print(f"  input:   {paths.input_path}")
    print(f"  holdout: {paths.out_holdout}  (ids={len(holdout_ids)})")
    print(f"  cl:      {paths.out_cl}       (ids={len(cl_ids)})")


if __name__ == "__main__":
    main()
