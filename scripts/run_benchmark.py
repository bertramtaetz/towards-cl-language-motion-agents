#!/usr/bin/env python3
"""Run the six retained approaches with original motion-cluster holdout partitions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "olora_multi_adapter": ("olora_multi_adapter", "train_olora.py", "evaluate_olora_{direction}.py"),
    "olora_multi_adapter_merged": ("olora_multi_adapter_merged", "train_olora_merged.py", "evaluate_olora_{direction}_merged.py"),
    "olora_moe": ("olora_moe", "train_olora_moe.py", "evaluate_olora_{direction}_moe.py"),
    "olora_moe_joined": ("olora_moe_joined", "train_olora_moe_joined.py", "evaluate_olora_{direction}_moe_joined.py"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", "--dir", choices=["t2m", "m2t"], required=True)
    p.add_argument("--methods", nargs="+", default=["multi_task", "transfer_learning", *METHODS])
    p.add_argument("--task-splits", type=Path, default=ROOT / "benchmark/splits/holdout10/tasks.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/HumanML3D")
    p.add_argument("--pretrained-root", type=Path, default=ROOT / "pretrained")
    p.add_argument("--pretrained", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-tasks", type=int, choices=range(1, 6), default=5)
    p.add_argument("--lambda-orth", type=float, default=0.5)
    p.add_argument("--smoke-token-only", action="store_true", help="Engineering validation only; skip generation and benchmark aggregation")
    p.add_argument("--router-epochs", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    phase = p.add_mutually_exclusive_group()
    phase.add_argument("--train-only", action="store_true")
    phase.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    args.methods = ["olora_multi_adapter" if m == "olora" else m for m in args.methods]
    if set(args.methods) - {"multi_task", "transfer_learning", *METHODS}:
        p.error("Unknown method; legacy implementations are not included")
    for name in ["task_splits", "data_root", "pretrained_root", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    checkpoint = (args.pretrained or args.pretrained_root / "holdout10" / args.direction / "multitask_final.pth").resolve()
    tasks = list(json.loads(args.task_splits.read_text())["tasks"].values())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled",
               MOTION_TASK_SPLITS_PATH=str(args.task_splits), MOTION_DATA_ROOT=str(args.data_root),
               MOTION_PRETRAINED_ROOT=str(args.pretrained_root),
               PYTHONPATH=str(ROOT / "Motion-Agent"), TOKENIZERS_PARALLELISM="false")
    if not args.dry_run:
        required = [checkpoint, args.data_root / "new_joint_vecs", args.data_root / "texts",
                    args.pretrained_root / "ckpt/vqvae.pth", args.pretrained_root / "gemma-2-2b-it/config.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            p.error("Missing required resources (no legacy fallback):\n" + "\n".join(missing))
        if args.direction == "m2t" and not args.train_only and not args.smoke_token_only:
            p.error("Full M2T generation and SPICE execution have not yet been "
                    "validated (resources are bundled; Java 21 SPICE fails). Evaluation is blocked rather than silently "
                    "omitting publication metrics. See docs/VALIDATION.md.")

    def run(command):
        command = [str(x) for x in command]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT / "Motion-Agent", env=env, check=True)

    launch = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", args.num_gpus,
              "--mixed_precision", "bf16", "--main_process_port", "0"]
    shared = ["--pretrained-path", checkpoint, "--split-mode", "random_80_20", "--split-seed", "42"]
    train = ["--training-task", args.direction, "--batch-size", args.batch_size,
             "--gradient-accumulation-steps", args.gradient_accumulation, "--learning-rate", "1e-4", "--no-wandb"]
    for method in args.methods:
        out = args.output_dir / args.direction / method
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            config.update(checkpoint=str(checkpoint), split_sha256=hashlib.sha256(args.task_splits.read_bytes()).hexdigest())
            (out / "launcher_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if method in METHODS:
            folder, trainer, evaluator = METHODS[method]
            base = Path("continual_learning") / folder
            if not args.eval_only:
                run([*launch, base / trainer, *train, *shared, "--out-dir", out,
                     "--epochs-per-task", args.epochs, "--num-tasks", args.num_tasks,
                     "--train-split", "train", "--lambda-orth", args.lambda_orth,
                     "--olora-target-modules", "full", "--gradient-checkpointing",
                     *(["--router-epochs", args.router_epochs, "--router-samples", 32] if args.smoke_token_only and "moe" in method else [])])
            if not args.train_only:
                run([sys.executable, base / evaluator.format(direction=args.direction), *shared,
                     "--exp-dir", out, "--split", "test", "--device", args.device,
                     "--batch-size", args.batch_size, *( ["--skip-generation"] if args.smoke_token_only else ["--generate-all-stages"]), "--olora-target-modules", "full",
                     *(["--num-tasks", args.num_tasks] if not (method == "olora_multi_adapter" and args.direction == "m2t") else [])])
        else:
            baseline = "multitask" if method == "multi_task" else "transfer"
            if not args.eval_only:
                common = [*train, *shared, "--out-dir", out, "--epochs", args.epochs,
                          "--task-splits-path", args.task_splits, "--use-pretrained-lora"]
                if baseline == "multitask":
                    run([*launch, "baselines/train_multitask.py", *common])
                else:
                    previous = None
                    for i, task in enumerate(tasks[:args.num_tasks]):
                        resume = ["--resume", previous] if previous else []
                        run([*launch, "baselines/train_transfer.py", *common, "--task-id", i,
                             "--task-name", task["name"], *resume])
                        previous = out / f"after_task{i}_{task['name']}.pth"
            if not args.train_only:
                run([sys.executable, f"baselines/evaluate_baselines_{args.direction}.py", *shared,
                     "--method", baseline, "--checkpoint-dir", out, "--output-dir", out,
                     "--split", "test", "--device", args.device, *(["--skip-generation"] if args.smoke_token_only else ["--generate-all-stages"])])
        if not args.train_only and not args.smoke_token_only:
            run([sys.executable, "continual_learning/compute_cl_metrics_token_acc.py", "--exp-dir", out, "--split", "test"])
            metric_method = {"multi_task": "multitask", "transfer_learning": "transfer",
                             "olora_multi_adapter": "olora_multi", "olora_multi_adapter_merged": "olora_merged"}.get(method, method)
            extra = ["--no-wandb"] if args.direction == "m2t" else []
            run([sys.executable, f"continual_learning/compute_cl_metrics_{args.direction}.py",
                 "--exp-dir", out, "--method", metric_method, "--split", "test", *extra])


if __name__ == "__main__":
    main()