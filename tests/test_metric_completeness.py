"""Aggregation must never turn missing benchmark results into zero scores."""
import copy
import numpy as np
import pytest
import json
import os
import subprocess
import sys
from pathlib import Path
from continual_learning import compute_cl_metrics_t2m as t2m
from continual_learning import compute_cl_metrics_m2t as m2t


@pytest.mark.parametrize("module,builder,metric", [
    (t2m, t2m.build_perf_matrix, "fid"),
    (m2t, m2t.build_performance_matrix, "bleu4"),
])
def test_complete_and_invalid_matrices(module, builder, metric):
    results = {stage: {"per_task": {task: {metric: float(i + j)}
               for j, task in enumerate(module.TASK_NAMES)}}
               for i, stage in enumerate(module.STAGES)}
    matrix, _, _ = builder(results, metric)
    np.testing.assert_equal(matrix, np.add.outer(np.arange(5), np.arange(5)))
    for kind in ["stage", "task", "metric", "nan"]:
        broken = copy.deepcopy(results)
        stage, task = module.STAGES[0], module.TASK_NAMES[0]
        if kind == "stage": del broken[stage]
        elif kind == "task": del broken[stage]["per_task"][task]
        elif kind == "metric": del broken[stage]["per_task"][task][metric]
        else: broken[stage]["per_task"][task][metric] = float("nan")
        with pytest.raises(ValueError): builder(broken, metric)


def test_t2m_formulas():
    matrix = np.add.outer(np.arange(5), np.arange(5)).astype(float)
    assert t2m.compute_acc(matrix)[0] == 6.0
    assert t2m.compute_bwt(matrix)[0] == 2.5
    # Existing convention: raw pre-task performance, not baseline-subtracted FWT.
    assert t2m.compute_fwt(matrix)[0] == 4.0


def test_t2m_written_metrics(tmp_path):
    root = Path(__file__).resolve().parent.parent
    evaluation = tmp_path / "eval_results"
    evaluation.mkdir()
    for i, stage in enumerate(t2m.STAGES):
        payload = {"per_task": {task: {metric: float(i + j)
                   for metric in t2m.METRIC_NAMES}
                   for j, task in enumerate(t2m.TASK_NAMES)}}
        (evaluation / f"{stage}_test.json").write_text(json.dumps(payload))
    output = tmp_path / "metrics"
    env = dict(os.environ, PYTHONPATH=str(root / "Motion-Agent"), WANDB_MODE="disabled")
    subprocess.run([sys.executable, "-B", str(root / "Motion-Agent/continual_learning/compute_cl_metrics_t2m.py"),
                    "--exp-dir", str(tmp_path), "--method", "olora_multi",
                    "--output-dir", str(output), "--no-plots"], env=env, check=True)
    result = json.loads((output / "cl_metrics_test.json").read_text())
    assert result["n_stages"] == result["n_tasks"] == 5
    assert result["tasks"] == t2m.TASK_NAMES
    assert result["metrics"]["fid"]["average_raw"] == 6.0
    assert result["metrics"]["fid"]["bwt"] == -2.5
    assert result["metrics"]["top1"]["bwt"] == 2.5

def test_m2t_written_metrics(tmp_path):
    root = Path(__file__).resolve().parent.parent
    evaluation = tmp_path / "eval_results"
    evaluation.mkdir()
    for i, stage in enumerate(m2t.STAGES):
        payload = {"per_task": {task: {metric: float(i + j)
                   for metric in m2t.METRIC_NAMES}
                   for j, task in enumerate(m2t.TASK_NAMES)}}
        (evaluation / f"{stage}_test.json").write_text(json.dumps(payload))
    output = tmp_path / "metrics"
    env = dict(os.environ, PYTHONPATH=str(root / "Motion-Agent"), WANDB_MODE="disabled")
    subprocess.run([sys.executable, "-B", str(root / "Motion-Agent/continual_learning/compute_cl_metrics_m2t.py"),
                    "--exp-dir", str(tmp_path), "--method", "olora_multi",
                    "--output-dir", str(output), "--no-plots", "--no-wandb"], env=env, check=True)
    result = json.loads((output / "cl_metrics_test.json").read_text())
    assert result["n_stages"] == result["n_tasks"] == 5
    assert result["metrics"]["bleu4"]["average"] == 6.0
    assert result["metrics"]["bleu4"]["bwt"] == 2.5
    assert result["metrics"]["bleu4"]["fwt"] == 4.0
