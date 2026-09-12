"""Smoke test for joined MoE variants (K=1 vs K=2; orth vs no-orth).

This test intentionally performs a *small but realistic* end-to-end run:

- Uses the default backbone (Gemma2) and normal data pipeline.
- Trains for 1 epoch on trainval with --num-tasks 1 (task_0 only).
- Evaluates K=1 (hard/top1) + K=2 (mixture) into separate output dirs via --checkpoint-dir.
- Repeats for orth regularization on (lambda_orth=0.5) and off (lambda_orth=0).

Expected outputs (per direction, per variant):
  <exp_dir>/eval_results/after_task0_test.json

Run:
  pytest tests/test_olora_moe_joined_variants_smoke.py -v -m integration
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
import torch


_THIS_DIR = Path(__file__).parent.resolve()
_MOTION_AGENT_ROOT = _THIS_DIR.parent.resolve()
_PROJECT_ROOT = _MOTION_AGENT_ROOT.parent

sys.path.insert(0, str(_MOTION_AGENT_ROOT))


def _require_path(p: Path, reason: str) -> None:
    if not p.exists():
        pytest.skip(f"{reason}: missing {p}")


def _run(cmd: list[str], *, cwd: Path) -> None:
    """Run a subprocess and raise a helpful error message on failure."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Keep smoke test fully offline.
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise AssertionError(
            "Command failed (exit code %s):\n%s\n\nOutput:\n%s"
            % (p.returncode, " ".join(cmd), p.stdout)
        )


def _assert_eval_json(exp_dir: Path) -> None:
    jf = exp_dir / "eval_results" / "after_task0_test.json"
    assert jf.exists(), f"Missing eval output JSON: {jf}"
    d = json.loads(jf.read_text())
    assert "per_task" in d, f"Missing 'per_task' in {jf}"

    # Different evaluators historically used different keys.
    # Joined MoE evaluators currently write `average`, while metrics scripts
    # treat it as an aggregate row.
    agg = None
    if "avg" in d:
        agg = d["avg"]
    elif "average" in d:
        agg = d["average"]
    assert isinstance(agg, dict), f"Missing aggregate metrics ('avg' or 'average') in {jf}"
    assert "loss" in agg, f"Missing aggregate loss in {jf}"
    assert "accuracy" in agg, f"Missing aggregate accuracy in {jf}"


def _summarize_eval(exp_dir: Path) -> dict:
    """Return a small summary dict for printing/debugging."""
    jf = exp_dir / "eval_results" / "after_task0_test.json"
    d = json.loads(jf.read_text())
    agg = d.get("avg") or d.get("average") or {}

    # Grab first task entry (since num_tasks=1)
    per_task = d.get("per_task") or {}
    task_key = next(iter(per_task.keys()), None)
    router_diag = None
    if task_key is not None:
        router_diag = (per_task.get(task_key) or {}).get("router")

    out = {
        "loss": float(agg.get("loss", float("nan"))),
        "accuracy": float(agg.get("accuracy", float("nan"))),
    }
    if isinstance(router_diag, dict):
        out.update(
            {
                "unseen_rate": float(router_diag.get("unseen_rate", float("nan"))),
                "router_top1": float(router_diag.get("router_top1", float("nan"))),
                "router_top2": float(router_diag.get("router_top2", float("nan"))),
                "routed_counts": router_diag.get("routed_counts"),
            }
        )
    return out


@pytest.mark.integration
def test_joined_moe_variants_smoke():
    # Gemma2 realistic smoke test requires GPU.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available (Gemma2 smoke test requires GPU)")

    # Pretrained used by usual runs.
    pretrained = Path(os.environ.get("MOTION_SMOKE_CHECKPOINT", _PROJECT_ROOT / "pretrained/v5/motionllm_v5.pth"))
    _require_path(pretrained, "Pretrained MotionLLM checkpoint required")

    # VQ-VAE checkpoint required by MotionLLM.
    vq = _PROJECT_ROOT / "pretrained/ckpt/vqvae.pth"
    _require_path(vq, "VQ-VAE checkpoint required")

    # Dataset required (realistic run).
    data_root = Path(os.environ.get("MOTION_DATA_ROOT", _PROJECT_ROOT / "datasets/HumanML3D"))
    _require_path(data_root / "new_joint_vecs", "HumanML3D dataset required")
    _require_path(data_root / "texts", "HumanML3D dataset required")

    train_script = _MOTION_AGENT_ROOT / "continual_learning" / "olora_moe_joined" / "train_olora_moe_joined.py"
    eval_t2m = _MOTION_AGENT_ROOT / "continual_learning" / "olora_moe_joined" / "evaluate_olora_t2m_moe_joined.py"
    eval_m2t = _MOTION_AGENT_ROOT / "continual_learning" / "olora_moe_joined" / "evaluate_olora_m2t_moe_joined.py"

    for p in (train_script, eval_t2m, eval_m2t):
        _require_path(p, "Joined MoE script required")

    # Keep the smoke test small but realistic.
    # - 1 task (task_0 only)
    # - 1 epoch
    # - router training reduced but still executed
    base_train_flags = [
        sys.executable,
        str(train_script),
        "--num-tasks",
        "1",
        "--start-task",
        "0",
        "--epochs-per-task",
        "1",
        "--train-split",
        "trainval",
        "--router-epochs",
        "1",
        "--router-samples",
        "256",
        "--router-batch-size",
        "64",
        "--no-wandb",
        "--pretrained-path",
        str(pretrained),
        "--split-mode",
        "predefined",
        "--split-seed",
        "42",
        "--olora-target-modules",
        "qv",
        "--gradient-checkpointing",
    ]

    base_eval_flags = [
        "--num-tasks",
        "1",
        "--stage",
        "0",
        "--split",
        "test",
        "--device",
        "cuda:0",
        "--batch-size",
        "8",
        "--skip-generation",
        "--pretrained-path",
        str(pretrained),
        "--split-mode",
        "predefined",
        "--split-seed",
        "42",
        "--olora-target-modules",
        "qv",
    ]

    # Each direction trains separately to keep the test logic simple.
    directions = [
        ("t2m", eval_t2m),
        ("m2t", eval_m2t),
    ]

    keep_dir = os.environ.get("KEEP_SMOKE_DIR", "0") == "1"
    tmp_ctx = None
    if keep_dir:
        root = (
            _PROJECT_ROOT
            / "experiments"
            / "smoke_tests"
            / f"joined_moe_{uuid.uuid4().hex[:8]}"
        )
        root.mkdir(parents=True, exist_ok=True)
        print(f"[smoke] KEEP_SMOKE_DIR=1 -> artifacts kept at: {root}")
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="joined_moe_smoke_")
        root = Path(tmp_ctx.name)

    summaries: list[tuple[str, str, str, dict]] = []
    try:
        for direction, eval_script in directions:
            # -------------------- O-LoRA joined (lambda_orth=0.5) --------------------
            olora_train_dir = root / f"{direction}" / "olora_joined_train"
            olora_k1_dir = root / f"{direction}" / "olora_joined_K1"
            olora_k2_dir = root / f"{direction}" / "olora_joined_K2"
            olora_train_dir.mkdir(parents=True, exist_ok=True)

            _run(
                base_train_flags
                + ["--training-task", direction, "--out-dir", str(olora_train_dir), "--lambda-orth", "0.5"],
                cwd=_MOTION_AGENT_ROOT,
            )

            # Verify stage-0 artifacts
            assert (olora_train_dir / "olora_moe_task_0_best.pth").exists()
            assert (olora_train_dir / "router_task_0.pth").exists()

            # K=1 eval (hard/top1)
            _run(
                [
                    sys.executable,
                    str(eval_script),
                    "--exp-dir",
                    str(olora_k1_dir),
                    "--checkpoint-dir",
                    str(olora_train_dir),
                ]
                + base_eval_flags
                + ["--router-top-k", "1", "--router-infer-mode", "top1"],
                cwd=_MOTION_AGENT_ROOT,
            )
            _assert_eval_json(olora_k1_dir)
            summaries.append((direction, "lambda_orth=0.5", "K1(top1)", _summarize_eval(olora_k1_dir)))

            # K=2 eval (mixture)
            _run(
                [
                    sys.executable,
                    str(eval_script),
                    "--exp-dir",
                    str(olora_k2_dir),
                    "--checkpoint-dir",
                    str(olora_train_dir),
                ]
                + base_eval_flags
                + ["--router-top-k", "2", "--router-infer-mode", "mixture"],
                cwd=_MOTION_AGENT_ROOT,
            )
            _assert_eval_json(olora_k2_dir)
            summaries.append((direction, "lambda_orth=0.5", "K2(mixture)", _summarize_eval(olora_k2_dir)))

            # -------------------- no-orth joined (lambda_orth=0) --------------------
            lora_train_dir = root / f"{direction}" / "lora_joined_train"
            lora_k1_dir = root / f"{direction}" / "lora_joined_K1"
            lora_k2_dir = root / f"{direction}" / "lora_joined_K2"
            lora_train_dir.mkdir(parents=True, exist_ok=True)

            _run(
                base_train_flags
                + ["--training-task", direction, "--out-dir", str(lora_train_dir), "--lambda-orth", "0"],
                cwd=_MOTION_AGENT_ROOT,
            )

            assert (lora_train_dir / "olora_moe_task_0_best.pth").exists()
            assert (lora_train_dir / "router_task_0.pth").exists()

            _run(
                [
                    sys.executable,
                    str(eval_script),
                    "--exp-dir",
                    str(lora_k1_dir),
                    "--checkpoint-dir",
                    str(lora_train_dir),
                ]
                + base_eval_flags
                + ["--router-top-k", "1", "--router-infer-mode", "top1"],
                cwd=_MOTION_AGENT_ROOT,
            )
            _assert_eval_json(lora_k1_dir)
            summaries.append((direction, "lambda_orth=0", "K1(top1)", _summarize_eval(lora_k1_dir)))

            _run(
                [
                    sys.executable,
                    str(eval_script),
                    "--exp-dir",
                    str(lora_k2_dir),
                    "--checkpoint-dir",
                    str(lora_train_dir),
                ]
                + base_eval_flags
                + ["--router-top-k", "2", "--router-infer-mode", "mixture"],
                cwd=_MOTION_AGENT_ROOT,
            )
            _assert_eval_json(lora_k2_dir)
            summaries.append((direction, "lambda_orth=0", "K2(mixture)", _summarize_eval(lora_k2_dir)))

        # Print a compact summary for easier debugging.
        print("\n[smoke] Joined MoE variant summary (stage=0, num_tasks=1, split=test, skip_generation=1)")
        for direction, orth, routing, s in summaries:
            rc = s.get("routed_counts")
            print(
                f"  - {direction:<3} | {orth:<13} | {routing:<18} | "
                f"loss={s['loss']:.4f} acc={s['accuracy']:.4f} "
                f"unseen_rate={s.get('unseen_rate', float('nan')):.3f} routed_counts={rc}"
            )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
