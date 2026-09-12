"""Copy a bounded, valid subset of the original benchmark; never modify input data."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Motion-Agent"))
from continual_learning.common.data_loader_random_split import RandomSplitMotionDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--num-tasks", type=int, choices=range(1, 6), default=2)
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("At least 10 samples are required")
    out = ROOT / "outputs/two_task_smoke"
    source = ROOT / "benchmark/splits/holdout10/tasks.json"
    manifest = json.loads(source.read_text())
    manifest["tasks"] = dict(list(manifest["tasks"].items())[:args.num_tasks])
    manifest["task_names"] = [t["name"] for t in manifest["tasks"].values()]
    for task in manifest["tasks"].values():
        dataset = RandomSplitMotionDataset(task["name"], "train", None,
                    task_splits_path=str(source), data_root=str(args.data_root), train_ratio=1.0)
        ids = dataset.name_list[:args.samples]
        if len(ids) != args.samples:
            raise ValueError(f"Insufficient valid samples: {task['name']}")
        for mid in ids:
            for folder, suffix in [("texts", ".txt"), ("new_joint_vecs", ".npy")]:
                target = out / "data" / folder / (mid + suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.data_root / folder / (mid + suffix), target)
        task.update(train=ids, val=[], test=[], n_train=len(ids), n_val=0, n_test=0, n_total=len(ids))
    manifest["engineering_smoke_only"] = True
    (out / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()