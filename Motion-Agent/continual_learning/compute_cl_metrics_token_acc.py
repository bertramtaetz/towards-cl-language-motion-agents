"""Compute continual learning metrics (ACC/BWT/FWT) from token accuracy.

This is the **canonical** implementation for token-accuracy continual-learning
metrics in this repo.

It reads per-task token accuracy from per-stage evaluation JSONs:

  <exp_dir>/eval_results/after_task{i}_{split}.json

Expected JSON structure:
  {
    "per_task": {
      "task_1_jumping": {"accuracy": 0.123, ...},
      ...
    }
  }

Performance matrix R is built as:
  R[i, j] = accuracy on task j after training stage i

Metrics:
  - BWT = (1/(T-1)) * sum_{j=0}^{T-2} (R[T-1,j] - R[j,j])
  - FWT = (1/(T-1)) * sum_{i=1}^{T-1} R[i-1,i]
  - ACC = (1/T)      * sum_j R[T-1,j]

Note:
  - This computes the CL metrics from the already-produced evaluation matrix.
    It does **not** attempt to harmonize the *inference mechanisms* of different
    methods (adapter switching vs routing), only the CL metric formulas.
"""


from __future__ import annotations
from repo_paths import task_order

import argparse
import json
from pathlib import Path

import numpy as np

TASK_NAMES = [name for name, _ in task_order()]

TASK_DISPLAY = {name: name.split("_", 2)[-1].replace("_", " ").title() for name, _ in task_order()}


def load_accuracy_matrix(exp_dir: Path, split: str) -> np.ndarray:
    n_tasks = len(TASK_NAMES)
    n_stages = n_tasks
    R = np.zeros((n_stages, n_tasks), dtype=np.float64)

    for i in range(n_stages):
        f = exp_dir / "eval_results" / f"after_task{i}_{split}.json"
        if not f.exists():
            raise FileNotFoundError(f"Missing evaluation JSON: {f}")
        data = json.loads(f.read_text())
        per_task = data.get("per_task", {})
        for j, task_name in enumerate(TASK_NAMES):
            if task_name in per_task:
                R[i, j] = float(per_task[task_name].get("accuracy", 0.0))

    return R


def compute_bwt(R: np.ndarray) -> tuple[float, list[float]]:
    n_tasks = R.shape[1]
    bwt_per_task = [float(R[-1, j] - R[j, j]) for j in range(n_tasks - 1)]
    return float(np.mean(bwt_per_task)), bwt_per_task


def compute_fwt(R: np.ndarray) -> tuple[float, list[float]]:
    n_tasks = R.shape[1]
    fwt_per_task = [float(R[i - 1, i]) for i in range(1, n_tasks)]
    return float(np.mean(fwt_per_task)), fwt_per_task


def compute_acc(R: np.ndarray) -> float:
    return float(np.mean(R[-1, :]))


def main() -> None:
    p = argparse.ArgumentParser(description="Compute CL metrics (token accuracy)")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "cl_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    R = load_accuracy_matrix(exp_dir, args.split)
    bwt, bwt_per_task = compute_bwt(R)
    fwt, fwt_per_task = compute_fwt(R)
    acc = compute_acc(R)

    # Console output (kept similar to previous script)
    print("\nTOKEN ACCURACY PERFORMANCE MATRIX  R[stage, task]")
    print("=" * 70)
    header = f"{'Stage':<14}" + "".join(f"{TASK_DISPLAY[t][:11]:>12}" for t in TASK_NAMES)
    print(header)
    print("-" * (14 + 12 * len(TASK_NAMES)))
    for i in range(R.shape[0]):
        row = f"After Task {i}  "
        for j in range(R.shape[1]):
            row += f"{R[i, j] * 100:>12.2f}"
        print(row)
    print("=" * 70)

    print("\nPER-TASK BWT  (R[final,j] - R[j,j])")
    for j in range(len(TASK_NAMES) - 1):
        diag = R[j, j]
        final = R[-1, j]
        diff = final - diag
        sign = "+" if diff >= 0 else ""
        print(
            f"  {TASK_DISPLAY[TASK_NAMES[j]]:<20}  diag={diag*100:.2f}%  "
            f"final={final*100:.2f}%  BWT={sign}{diff*100:.2f}pp"
        )

    print("\nPER-TASK FWT  (R[i-1,i] = zero-shot on task i)")
    for i in range(1, len(TASK_NAMES)):
        fwt_val = R[i - 1, i]
        print(f"  Task {i} ({TASK_DISPLAY[TASK_NAMES[i]]:<20}) zero-shot acc = {fwt_val*100:.2f}%")

    print(f"\n{'='*50}")
    print(f"  ACC (avg final accuracy): {acc*100:.2f}%")
    print(
        f"  BWT (backward transfer):  {bwt*100:+.2f}pp  "
        f"({'forgetting' if bwt < 0 else 'positive transfer'})"
    )
    print(f"  FWT (forward transfer):   {fwt*100:.2f}%  (avg zero-shot)")
    print(f"{'='*50}")

    output = {
        "metric": "token_accuracy",
        "split": args.split,
        "n_stages": int(R.shape[0]),
        "n_tasks": int(R.shape[1]),
        "tasks": TASK_NAMES,
        "performance_matrix": R.tolist(),
        "average": acc,
        "bwt": bwt,
        "bwt_per_task": bwt_per_task,
        "fwt": fwt,
        "fwt_per_task": fwt_per_task,
        "final_per_task": R[-1, :].tolist(),
    }

    out_file = out_dir / f"cl_metrics_token_acc_{args.split}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
