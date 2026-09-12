#!/usr/bin/env python3
"""
Shared Continual Learning Metrics for M2T (Motion-to-Text) Direction.

Works identically for all CL methods: O-LoRA Multi-Adapter, O-LoRA Single-Adapter, D-MoLE M2T.
Reads per-stage evaluation results and computes standardized CL metrics.

Evaluated NLG metrics (all higher-is-better):
    bleu4, rouge_l, cider, bleu1, spice, bertscore

CL metrics computed for each NLG metric:
    BWT (Backward Transfer): Measures catastrophic forgetting
    FWT (Forward Transfer):  Measures zero-shot knowledge transfer to new tasks
    ACC (Average):           Final average performance across all tasks

Performance Matrix R:
    R[i,j] = performance on task j after training on tasks 0..i

    Rows:    Training stages (after_task0, after_task1, ...)
    Columns: Tasks (task_1_jumping, task_2_arms_hands, ...)

Expected directory layout:
    <exp-dir>/
        eval_results/
            after_task0_test.json   # Each contains {per_task: {task_name: {metric: val}}}
            after_task1_test.json
            ...
            after_task4_test.json

Usage:
    # O-LoRA Multi-Adapter M2T
    python compute_cl_metrics_m2t.py \\
        --exp-dir experiments/olora_multi/m2t/v1 \\
        --method olora_multi --split test

    # D-MoLE M2T
    python compute_cl_metrics_m2t.py \\
        --exp-dir experiments/dmole/m2t/5tasks_v2 \\
        --method dmole_m2t --split test

    # Transfer Learning M2T
    python compute_cl_metrics_m2t.py \\
        --exp-dir experiments/transfer_learning/m2t/v1 \\
        --method olora_multi --split test
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import re
import numpy as np
from pathlib import Path
import sys

_CL_DIR = Path(__file__).parent.resolve()
_MOTION_AGENT_ROOT = _CL_DIR.parent.resolve()

# 5-task order — matches baselines/transfer_learning/data_loader.py TASK_ORDER
# Intentionally hardcoded to avoid heavy import dependencies (VQ-VAE, paramUtil, etc.)
TASK_ORDER = task_order()

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAMES = [t[0] for t in TASK_ORDER]

TASK_DISPLAY_NAMES = {name: name.split("_", 2)[-1].replace("_", " ").title() for name, _ in task_order()}

STAGES = [f"after_task{i}" for i in range(5)]

# M2T NLG metrics to compute CL metrics for — all higher-is-better
METRIC_NAMES = [
    'bleu4',
    'rouge_l',
    'cider',
    'bleu1',
    'spice',
    'bertscore',
]

METHOD_DISPLAY = {
    'transfer':     'Transfer Learning',
    'multitask':    'Multitask',
    'olora_multi':  'O-LoRA Multi-Adapter',
    'olora_merged': 'O-LoRA Merged',
    'olora_full':   'O-LoRA Full (online merge)',
    'olora_moe':    'O-LoRA MoE',
    'olora_moe2':   'O-LoRA MoE2',
    'olora_single': 'O-LoRA Single-Adapter',
    'dmole_m2t':    'D-MoLE M2T',
}


def _normalize_method_and_label(method_raw: str) -> tuple[str, str]:
    """Normalize method id so joined-MoE always encodes K.

    Returns (method_norm, method_label).
    """
    m = re.match(r"^(olora_moe_joined|lora_moe_joined)_K(?P<k>\d+)$", method_raw)
    if m:
        k = int(m.group("k"))
        fam = m.group(1)
        method_norm = f"{fam}_K{k}"
        if fam == "olora_moe_joined":
            label = f"O-LoRA MoE Joined (K={k})"
        else:
            label = f"LoRA MoE Joined (K={k}, $\\lambda_{{orth}}=0$)"
        return method_norm, label

    if method_raw in {"olora_moe_joined_hard", "olora_moe_joined_soft", "lora_moe_joined_hard", "lora_moe_joined_soft"}:
        k = 1 if method_raw.endswith("_hard") else 2
        fam = method_raw.replace("_hard", "").replace("_soft", "")
        method_norm = f"{fam}_K{k}"
        print(
            f"[cl-metrics] WARNING: deprecated --method '{method_raw}'. "
            f"Use '{method_norm}' instead (joined MoE always encodes K)."
        )
        if fam == "olora_moe_joined":
            label = f"O-LoRA MoE Joined (K={k})"
        else:
            label = f"LoRA MoE Joined (K={k}, $\\lambda_{{orth}}=0$)"
        return method_norm, label

    return method_raw, METHOD_DISPLAY.get(method_raw, method_raw)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(exp_dir: Path, split: str = 'test'):
    """Load evaluation results from all stages.

    Expected file: <exp_dir>/eval_results/after_taskN_<split>.json
    Each file must contain a ``per_task`` dict keyed by task name.
    """
    results = {}
    eval_dir = exp_dir / "eval_results"
    for stage in STAGES:
        result_file = eval_dir / f"{stage}_{split}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[stage] = json.load(f)
        else:
            print(f"  Warning — missing: {result_file}")
    return results


# ---------------------------------------------------------------------------
# Performance matrix
# ---------------------------------------------------------------------------

def build_performance_matrix(results: dict, metric: str = 'bleu4'):
    """Build the performance matrix R[i,j].

    Returns (R, stages_present, task_names).
    """
    n_tasks = len(TASK_NAMES)
    stages_present = [s for s in STAGES if s in results]
    if stages_present != STAGES:
        raise ValueError("Incomplete benchmark: all five consecutive stages are required")
    n_stages = len(stages_present)

    R = np.zeros((n_stages, n_tasks))

    for i, stage in enumerate(stages_present):
        stage_data = results[stage]
        per_task = stage_data.get('per_task', stage_data)

        for j, task in enumerate(TASK_NAMES):
            if task in per_task:
                task_data = per_task[task]
                if metric in task_data:
                    if not np.isfinite(task_data[metric]):
                        raise ValueError(f"Non-finite {metric} at {stage}/{task}")
                    R[i, j] = task_data[metric]
                else:
                    raise ValueError(f"Missing {metric} at {stage}/{task}")
            else:
                raise ValueError(f"Missing task {stage}/{task}")

    return R, stages_present, TASK_NAMES


# ---------------------------------------------------------------------------
# CL metric formulas
# ---------------------------------------------------------------------------

def compute_backward_transfer(R: np.ndarray):
    """Backward Transfer (BWT).

    BWT = (1/(T-1)) * sum_{j=0}^{T-2} (R[T-1, j] - R[j, j])

    For higher-is-better metrics:
        BWT < 0  =>  catastrophic forgetting
        BWT ~ 0  =>  no forgetting
        BWT > 0  =>  positive backward transfer (rare)
    """
    n_stages, n_tasks = R.shape
    if n_stages < 2:
        return None, None

    bwt_values = []
    for j in range(n_tasks - 1):          # exclude last task (nothing trained after it)
        if j < n_stages:
            bwt_j = R[-1, j] - R[j, j]   # final perf − perf right after learning task j
            bwt_values.append(bwt_j)

    if not bwt_values:
        return None, None
    return float(np.mean(bwt_values)), bwt_values


def compute_forward_transfer(R: np.ndarray):
    """Forward Transfer (FWT).

    FWT = (1/(T-1)) * sum_{i=1}^{T-1} R[i-1, i]

    Measures zero-shot performance on task *i* before training it
    (i.e. after training tasks 0..i-1).
    """
    n_stages, n_tasks = R.shape
    if n_stages < 2:
        return None, None

    fwt_values = []
    for i in range(1, n_tasks):
        if i - 1 < n_stages:
            fwt_values.append(R[i - 1, i])

    if not fwt_values:
        return None, None
    return float(np.mean(fwt_values)), fwt_values


def compute_average_accuracy(R: np.ndarray):
    """Average final performance (ACC).

    ACC = (1/T) * sum_{j=0}^{T-1} R[T-1, j]
    """
    final_row = R[-1, :]
    return float(np.mean(final_row)), final_row.tolist()


def compute_forgetting_matrix(R: np.ndarray):
    """Per-cell forgetting: F[i,j] = max(R[0:i+1, j]) - R[i, j]."""
    n_stages, n_tasks = R.shape
    F = np.zeros_like(R)
    for i in range(n_stages):
        for j in range(n_tasks):
            if i >= j:
                F[i, j] = np.max(R[:i + 1, j]) - R[i, j]
    return F


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_performance_heatmap(R, stages, tasks, output_path, metric_name, method_label):
    if not HAS_PLOTTING:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    display_tasks = [TASK_DISPLAY_NAMES.get(t, t) for t in tasks]
    stage_labels = [f"After Task {i}" for i in range(len(stages))]

    sns.heatmap(R, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=display_tasks, yticklabels=stage_labels, ax=ax)

    metric_display = metric_name.replace('_', ' ').upper()
    ax.set_title(f'{method_label} — Performance Matrix: {metric_display}\n'
                 f'R[i,j] = {metric_display} on task j after training stage i',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Training Stage', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_forgetting_curves(R, stages, tasks, output_path, metric_name, method_label):
    if not HAS_PLOTTING:
        return
    n_tasks = len(tasks)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, n_tasks))

    for j, (task, color) in enumerate(zip(tasks, colors)):
        ax = axes[j]
        ax.plot(range(len(stages)), R[:, j], 'o-', color=color, linewidth=2, markersize=8)
        ax.axvline(x=j, color='red', linestyle='--', alpha=0.5, label='trained here')
        ax.set_title(f'Task {j}: {TASK_DISPLAY_NAMES.get(task, task)}',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name.replace('_', ' ').upper())
        ax.set_xlabel('Training Stage')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"T{i}" for i in range(len(stages))])
        ax.grid(True, alpha=0.3)

    # Average in last subplot
    ax = axes[-1]
    ax.plot(range(len(stages)), R.mean(axis=1), 'o-', color='black', linewidth=2, markersize=8)
    ax.set_title('Average Across Tasks', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').upper())
    ax.set_xlabel('Training Stage')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([f"T{i}" for i in range(len(stages))])
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{method_label}: {metric_name.upper()} Over Training Stages',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute CL Metrics for M2T direction (shared across methods)')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Path to experiment directory containing eval_results/')
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        help=(
            "CL method identifier. For joined MoE, use 'olora_moe_joined_K{K}' or 'lora_moe_joined_K{K}'. "
            "Legacy aliases (*_hard/*_soft) are accepted but deprecated."
        ),
    )
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <exp-dir>/cl_metrics)')
    parser.add_argument('--wandb-project', type=str, default='msai-thesis')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating heatmap/curve PNGs')

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    method_norm, method_label = _normalize_method_and_label(args.method)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "cl_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CONTINUAL LEARNING METRICS — M2T DIRECTION")
    print(f"Method:     {method_label}")
    print(f"Experiment: {exp_dir.name}")
    print(f"Split:      {args.split}")
    print(f"Output:     {output_dir}")
    print(f"NLG metrics: {METRIC_NAMES}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load per-stage results
    # ------------------------------------------------------------------
    print("\nLoading evaluation results...")
    results = load_results(exp_dir, args.split)
    if not results:
        print("ERROR: No evaluation results found!")
        print(f"Expected files in: {exp_dir / 'eval_results'}/after_taskN_{args.split}.json")
        sys.exit(1)

    print(f"  Loaded {len(results)} stage(s): {list(results.keys())}")
    if len(results) < 5:
        missing = [s for s in STAGES if s not in results]
        print(f"  WARNING: Missing stages: {missing}")
        print(f"  BWT/FWT will be computed on available stages only.")

    # ------------------------------------------------------------------
    # Compute CL metrics for each NLG metric
    # ------------------------------------------------------------------
    all_metrics = {}
    performance_matrices = {}
    _, stages, tasks = build_performance_matrix(results, 'bleu4')

    for metric_name in METRIC_NAMES:
        print(f"\n{'='*60}")
        print(f"  {metric_name.upper()}")
        print(f"{'='*60}")

        R, _, _ = build_performance_matrix(results, metric=metric_name)
        performance_matrices[metric_name] = R.tolist()


        avg, final_values = compute_average_accuracy(R)
        bwt, bwt_values   = compute_backward_transfer(R)
        fwt, fwt_values   = compute_forward_transfer(R)

        all_metrics[metric_name] = {
            'average':       avg,
            'final_per_task': [float(v) for v in final_values],
            'bwt':           bwt,
            'bwt_per_task':  [float(v) for v in bwt_values] if bwt_values else None,
            'fwt':           fwt,
            'fwt_per_task':  [float(v) for v in fwt_values] if fwt_values else None,
            'higher_is_better': True,
        }

        print(f"  ACC (avg final): {avg:.4f}")
        if bwt is not None:
            sign = '+' if bwt > 0 else ''
            label = '(forgetting!)' if bwt < 0 else ''
            print(f"  BWT:             {sign}{bwt:.4f} {label}")
        if fwt is not None:
            sign = '+' if fwt > 0 else ''
            label = '(positive transfer)' if fwt > 0 else ''
            print(f"  FWT:             {sign}{fwt:.4f} {label}")

        # Plots
        if not args.no_plots and HAS_PLOTTING:
            hm_path = output_dir / f"heatmap_{metric_name}.png"
            plot_performance_heatmap(R, stages, tasks, hm_path, metric_name, method_label)
            cv_path = output_dir / f"curves_{metric_name}.png"
            plot_forgetting_curves(R, stages, tasks, cv_path, metric_name, method_label)
            print(f"  Plots: {hm_path.name}, {cv_path.name}")

    # ------------------------------------------------------------------
    # Save consolidated JSON
    # ------------------------------------------------------------------
    output = {
        'experiment':          exp_dir.name,
        'experiment_type':     f'{method_norm}_m2t',
        'method':              method_norm,
        'method_display':      method_label,
        'split':               args.split,
        'n_stages':            len(stages),
        'n_tasks':             len(tasks),
        'stages':              stages,
        'tasks':               tasks,
        'task_display_names':  TASK_DISPLAY_NAMES,
        'metrics':             all_metrics,
        'performance_matrices': performance_matrices,
    }

    output_file = output_dir / f"cl_metrics_{args.split}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"\nAll CL metrics saved to: {output_file}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY — {method_label} — M2T CL Metrics (all higher-is-better)")
    print("=" * 70)
    print(f"{'Metric':<15} {'ACC (avg)':>10} {'BWT':>10} {'FWT':>10}")
    print("-" * 50)
    for metric_name, m in all_metrics.items():
        avg_s = f"{m['average']:.4f}"
        bwt_s = f"{m['bwt']:.4f}" if m['bwt'] is not None else "N/A"
        fwt_s = f"{m['fwt']:.4f}" if m['fwt'] is not None else "N/A"
        print(f"{metric_name:<15} {avg_s:>10} {bwt_s:>10} {fwt_s:>10}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # W&B (optional)
    # ------------------------------------------------------------------
    if HAS_WANDB and not args.no_wandb:
        run_name = f"{exp_dir.name}_cl_metrics_{args.split}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'exp_name':        exp_dir.name,
                'experiment_type': f'{method_norm}_m2t',
                'method':          method_norm,
                'split':           args.split,
                'n_stages':        len(stages),
                'n_tasks':         len(tasks),
                'metrics_computed': METRIC_NAMES,
            },
            tags=['cl-metrics', method_norm, 'm2t', 'continual-learning', args.split],
        )

        for metric_name, m in all_metrics.items():
            prefix = f"cl/{metric_name}"
            wandb.log({f"{prefix}/average": m['average']})
            if m['bwt'] is not None:
                wandb.log({f"{prefix}/bwt": m['bwt']})
            if m['fwt'] is not None:
                wandb.log({f"{prefix}/fwt": m['fwt']})
            for task, val in zip(tasks, m['final_per_task']):
                wandb.log({f"{prefix}/final_{task}": val})

        for metric_name in METRIC_NAMES:
            if metric_name in performance_matrices:
                R = np.array(performance_matrices[metric_name])
                if not np.all(R == 0):
                    rows = []
                    for i, stage in enumerate(stages):
                        rows.append([stage] + [R[i, j] for j in range(len(tasks))])
                    table = wandb.Table(columns=["Stage"] + tasks, data=rows)
                    wandb.log({f"cl/matrix_{metric_name}": table})

        if not args.no_plots:
            for metric_name in METRIC_NAMES:
                for kind in ('heatmap', 'curves'):
                    p = output_dir / f"{kind}_{metric_name}.png"
                    if p.exists():
                        wandb.log({f"cl/{kind}_{metric_name}": wandb.Image(str(p))})

        wandb.finish()
        print("\nCL metrics logged to W&B")

    print("\nDone.")


if __name__ == '__main__':
    main()
