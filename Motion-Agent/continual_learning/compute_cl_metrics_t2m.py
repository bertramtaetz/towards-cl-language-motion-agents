#!/usr/bin/env python3
"""
Shared Continual Learning Metrics for T2M (Text-to-Motion) Direction.

Works for all CL methods: transfer learning, multitask, O-LoRA, D-MoLE.
Reads per-stage evaluation results and computes standardised CL metrics.

T2M quality metrics and their directions:
    fid          (lower is better)
    top1/2/3     (higher is better) -- R-Precision
    diversity    (higher is better)
    mm_dist      (lower is better)

For metrics where lower is better (FID, MM-Dist), values are negated
internally so that BWT < 0 consistently means forgetting across all metrics.

Usage:
    python compute_cl_metrics_t2m.py --exp-dir experiments/transfer_learning/t2m/v1 --method transfer --split test
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse, json, re, sys
import numpy as np
from pathlib import Path

TASK_ORDER = task_order()
TASK_NAMES = [t[0] for t in TASK_ORDER]
TASK_DISPLAY = {name: name.split("_", 2)[-1].replace("_", " ").title() for name, _ in task_order()}
STAGES = [f"after_task{i}" for i in range(5)]
METRIC_NAMES = ["fid", "top1", "top2", "top3", "diversity", "mm_dist"]
HIGHER_IS_BETTER = {"top1", "top2", "top3", "diversity"}
LOWER_IS_BETTER = {"fid", "mm_dist"}

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_results(exp_dir, split="test"):
    results = {}
    for stage in STAGES:
        f = exp_dir / "eval_results" / f"{stage}_{split}.json"
        if f.exists():
            with open(f) as fh: results[stage] = json.load(fh)
        else:
            print(f"  Warning -- missing: {f}")
    return results


def build_perf_matrix(results, metric="fid", negate=False):
    stages_present = [s for s in STAGES if s in results]
    if stages_present != STAGES:
        raise ValueError("Incomplete benchmark: all five consecutive stages are required")
    R = np.zeros((len(stages_present), len(TASK_NAMES)))
    for i, stage in enumerate(stages_present):
        pt = results[stage].get("per_task", results[stage])
        for j, task in enumerate(TASK_NAMES):
            if task in pt and metric in pt[task]:
                v = pt[task][metric]
                if not np.isfinite(v):
                    raise ValueError(f"Non-finite {metric} at {stage}/{task}")
                if negate and metric in LOWER_IS_BETTER: v = -v
                R[i, j] = v
            else:
                raise ValueError(f"Missing {metric} at {stage}/{task}")
    return R, stages_present, TASK_NAMES


def compute_bwt(R):
    n_s, n_t = R.shape
    if n_s < 2: return None, None
    vals = [R[-1, j] - R[j, j] for j in range(n_t - 1) if j < n_s]
    return (float(np.mean(vals)), vals) if vals else (None, None)


def compute_fwt(R):
    n_s, n_t = R.shape
    if n_s < 2: return None, None
    vals = [R[i-1, i] for i in range(1, n_t) if i-1 < n_s]
    return (float(np.mean(vals)), vals) if vals else (None, None)


def compute_acc(R):
    return float(np.mean(R[-1, :])), R[-1, :].tolist()


def plot_heatmap(R, stages, tasks, path, mn, label):
    if not HAS_PLOT: return
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(R, annot=True, fmt=".4f", cmap="RdYlGn",
                xticklabels=[TASK_DISPLAY.get(t, t) for t in tasks],
                yticklabels=[f"After T{i}" for i in range(len(stages))], ax=ax)
    ax.set_title(f"{label} -- {mn.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_curves(R, stages, tasks, path, mn, label):
    if not HAS_PLOT: return
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)); axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, len(tasks)))
    for j, (t, c) in enumerate(zip(tasks, colors)):
        ax = axes[j]
        ax.plot(range(len(stages)), R[:, j], "o-", color=c, lw=2, ms=8)
        ax.axvline(x=j, color="red", ls="--", alpha=.5)
        ax.set_title(TASK_DISPLAY.get(t, t), fontweight="bold")
        ax.set_ylabel(mn.upper()); ax.grid(True, alpha=.3)
    axes[-1].plot(range(len(stages)), R.mean(axis=1), "o-", color="k", lw=2, ms=8)
    axes[-1].set_title("Average", fontweight="bold"); axes[-1].grid(True, alpha=.3)
    plt.suptitle(f"{label}: {mn.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def main():
    p = argparse.ArgumentParser(description="CL Metrics for T2M direction")
    p.add_argument("--exp-dir", required=True)
    p.add_argument(
        "--method",
        required=True,
        help=(
            "CL method identifier. For joined MoE, use 'olora_moe_joined_K{K}' or 'lora_moe_joined_K{K}'. "
            "Legacy aliases (*_hard/*_soft) are accepted but deprecated."
        ),
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "cl_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_labels = {
        "transfer": "Transfer Learning",
        "transfer_multi": "Transfer Learning (Multi)",
        "multitask": "Multitask",
        "olora_multi": "O-LoRA Multi",
        "olora_merged": "O-LoRA Merged",
        "olora_full": "O-LoRA Full (online merge)",
        "olora_moe": "O-LoRA MoE",
        "olora_moe2": "O-LoRA MoE2",
        "dmole": "D-MoLE",
    }

    # Normalize joined MoE method names so K is always encoded.
    method_raw = args.method
    method_norm = method_raw
    mlbl = base_labels.get(method_norm, method_norm)

    m = re.match(r"^(olora_moe_joined|lora_moe_joined)_K(?P<k>\d+)$", method_raw)
    if m:
        k = int(m.group("k"))
        fam = m.group(1)
        method_norm = f"{fam}_K{k}"
        if fam == "olora_moe_joined":
            mlbl = f"O-LoRA MoE Joined (K={k})"
        else:
            mlbl = f"LoRA MoE Joined (K={k}, $\\lambda_{{orth}}=0$)"
    elif method_raw in {"olora_moe_joined_hard", "olora_moe_joined_soft", "lora_moe_joined_hard", "lora_moe_joined_soft"}:
        # Backward-compatible aliases
        k = 1 if method_raw.endswith("_hard") else 2
        fam = method_raw.replace("_hard", "").replace("_soft", "")
        method_norm = f"{fam}_K{k}"
        print(
            f"[cl-metrics] WARNING: deprecated --method '{method_raw}'. "
            f"Use '{method_norm}' instead (joined MoE always encodes K)."
        )
        if fam == "olora_moe_joined":
            mlbl = f"O-LoRA MoE Joined (K={k})"
        else:
            mlbl = f"LoRA MoE Joined (K={k}, $\\lambda_{{orth}}=0$)"
    else:
        # Non-joined methods: keep current behavior.
        mlbl = base_labels.get(method_raw, method_raw)

    print("=" * 70)
    print(f"CL METRICS -- T2M  |  {mlbl}  |  {exp_dir.name}  |  {args.split}")
    print("=" * 70)

    results = load_results(exp_dir, args.split)
    if not results: print("ERROR: No results!"); sys.exit(1)
    print(f"  Loaded {len(results)} stage(s)")

    all_m, perf_raw = {}, {}
    _, stages, tasks = build_perf_matrix(results, "fid")

    for mn in METRIC_NAMES:
        dr = "lower better" if mn in LOWER_IS_BETTER else "higher better"
        R_raw, _, _ = build_perf_matrix(results, mn, negate=False)
        perf_raw[mn] = R_raw.tolist()
        R_n, _, _ = build_perf_matrix(results, mn, negate=True)
        bwt, bwt_v = compute_bwt(R_n)
        fwt, fwt_v = compute_fwt(R_n)
        all_m[mn] = {
            "average_raw": float(np.mean(R_raw[-1, :])),
            "final_per_task_raw": R_raw[-1, :].tolist(),
            "higher_is_better": mn in HIGHER_IS_BETTER,
            "bwt": bwt, "bwt_per_task": [float(v) for v in bwt_v] if bwt_v else None,
            "fwt": fwt, "fwt_per_task": [float(v) for v in fwt_v] if fwt_v else None,
        }
        avg = float(np.mean(R_raw[-1, :]))
        b = f"  BWT={bwt:+.4f}" if bwt is not None else ""
        fw = f"  FWT={fwt:+.4f}" if fwt is not None else ""
        print(f"  {mn:<10} ({dr:<14})  ACC={avg:.4f}{b}{fw}")

        if not args.no_plots and HAS_PLOT:
            plot_heatmap(R_raw, stages, tasks, out_dir / f"heatmap_{mn}.png", mn, mlbl)
            plot_curves(R_raw, stages, tasks, out_dir / f"curves_{mn}.png", mn, mlbl)

    output = {
        "experiment": exp_dir.name,
        "experiment_type": f"{method_norm}_t2m",
        "method": method_norm,
        "method_display": mlbl,
        "split": args.split, "n_stages": len(stages), "n_tasks": len(tasks),
        "stages": stages, "tasks": tasks, "task_display_names": TASK_DISPLAY,
        "metrics": all_m, "performance_matrices_raw": perf_raw,
        "note": "BWT/FWT on normalised values (higher=better); FID/MM-Dist negated so BWT<0 = forgetting.",
    }
    of = out_dir / f"cl_metrics_{args.split}.json"
    with open(of, "w") as f: json.dump(output, f, indent=2, allow_nan=False)
    print("Saved:", of)

    print("\n" + "=" * 70)
    print(f"{'Metric':<12} {'Direction':<16} {'ACC':>10} {'BWT':>10} {'FWT':>10}")
    print("-" * 60)
    for m, d in all_m.items():
        dr = "lower better" if m in LOWER_IS_BETTER else "higher better"
        b = f"{d['bwt']:.4f}" if d["bwt"] is not None else "N/A"
        fw = f"{d['fwt']:.4f}" if d["fwt"] is not None else "N/A"
        print(f"{m:<12} {dr:<16} {d['average_raw']:>10.4f} {b:>10} {fw:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
