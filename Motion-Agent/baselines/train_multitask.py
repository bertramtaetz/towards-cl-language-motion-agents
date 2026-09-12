"""
Multi-Task Learning Training Script for Motion-Agent

Trains a single shared LoRA adapter on all 5 motion-based tasks SIMULTANEOUSLY.
This establishes the UPPER BOUND for continual learning benchmark.

Supports both T2M (text-to-motion) and M2T (motion-to-text) directions
via the --training-task flag.

Default mode uses fresh LoRA (randomly initialized) with only motion token
embeddings loaded from the pretrained checkpoint, matching the transfer
learning baseline for fair comparison.

Tasks (Motion-Based Clustering):
    task_1_jumping    (Cluster 8)
    task_2_arms_hands (Cluster 4)
    task_3_walking    (Cluster 2)
    task_4_gestures   (Cluster 5)
    task_5_sit_stand  (Cluster 14)

Usage:
    # T2M multitask
    accelerate launch --num_processes 4 train_multitask.py \
        --training-task t2m --epochs 20

    # M2T multitask
    accelerate launch --num_processes 4 train_multitask.py \
        --training-task m2t --epochs 20
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import wandb
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
from tqdm import tqdm

MOTION_AGENT_PATH = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(MOTION_AGENT_PATH))

from models.mllm import MotionLLM
from utils.word_vectorizer import WordVectorizer

sys.path.insert(0, str(Path(__file__).parent / "multi_task"))
from data_loader import get_multitask_loader

# Random split multi-task loader (mirrors CL protocol)
sys.path.insert(0, str(Path(__file__).parent / "multi_task"))
try:
    from data_loader_random_split import get_multitask_random_split_loader
except Exception:
    get_multitask_random_split_loader = None


def get_logger(out_dir):
    logger = logging.getLogger("MultiTask-Exp")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_embeddings_only(model, checkpoint_path, device):
    """Load ONLY motion token embeddings and lm_head (no LoRA weights)."""
    print(f"Loading ONLY embeddings from {checkpoint_path}")
    save_dict = torch.load(checkpoint_path, map_location=device)

    lora_keys = [k for k in save_dict if "lora" in k]
    print(f"  Skipping {len(lora_keys)} pre-trained LoRA weights")

    if "embeddings" in save_dict:
        model.llm.get_input_embeddings().weight.data[model.nb_text_tokens:] = save_dict["embeddings"]
    if "lm_head" in save_dict:
        model.llm.lm_head.weight.data[model.nb_text_tokens:] = save_dict["lm_head"]

    print("  Embeddings loaded, LoRA adapters remain fresh")


def compute_token_accuracy(model, data_loader, device):
    model.eval()
    all_losses, all_accs = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", leave=False):
            word_emb, pos_oh, caption, sent_len, motion, m_length, token, name = batch
            motion_tokens = []
            for i in range(motion.shape[0]):
                m_in = motion[i:i+1, :m_length[i], :].to(device)
                tok = model.net.encode(m_in).squeeze(0)
                tok_remapped = torch.from_numpy(
                    model.motion_token_indices[tok.cpu().numpy()]
                ).to(device)
                motion_tokens.append(tok_remapped)
            loss, gen_acc, _, _ = model(caption, motion_tokens)
            all_losses.append(loss.item())
            all_accs.append(gen_acc)
    return np.mean(all_losses), np.mean(all_accs)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Task Learning for Motion-Agent")

    p.add_argument("--training-task", type=str, required=True,
                   choices=["t2m", "m2t"],
                   help="Training direction: t2m (text-to-motion) or m2t (motion-to-text)")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (small regularization; default 0.01)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio for cosine LR schedule (default 0.05)",
    )
    p.add_argument("--save-epochs", type=str, default="20",
                   help="Comma-separated epochs to save checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--train-split", type=str, default="trainval",
                   choices=["train", "trainval"])

    # Early stopping (uses current validation loader)
    p.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping on validation metric",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Patience (epochs) for early stopping (default 3)",
    )
    p.add_argument(
        "--early-stopping-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_acc"],
        help="Metric to monitor for early stopping (default val_loss)",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement to reset patience (default 0.0)",
    )

    # Split protocol alignment (for fair CL comparisons)
    p.add_argument(
        "--split-mode",
        type=str,
        default="random_80_20",
        choices=["predefined", "random_80_20"],
        help=(
            "Data split protocol. 'predefined' uses task_splits.json train/val/test. "
            "'random_80_20' combines all splits and resplits 80/20 with a seed (matches CL default)."
        ),
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used when split-mode is random_80_20",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for random_80_20 split mode (default: 0.8)",
    )

    # Capacity scaling (single-adapter parameter budget baseline)
    p.add_argument(
        "--capacity-multiplier",
        type=int,
        default=1,
        help=(
            "Scale LoRA rank and alpha by this factor for the active direction adapter. "
            "Use 5 to roughly match the parameter budget of 5 per-task adapters."
        ),
    )

    # W&B
    p.add_argument("--wandb-project", type=str, default="motion-agent-multitask")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")

    # Model
    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=64)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false", default=True)

    # Memory optimization
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help=(
            "Enable gradient checkpointing on the base LLM to reduce VRAM usage "
            "(trades compute for memory). Recommended for minimal pretraining."
        ),
    )

    # LoRA capacity alignment (optional)
    # - full: Motion-Agent default (7 modules)
    # - qv:   q_proj/v_proj only (matches O-LoRA adapters)
    p.add_argument(
        "--lora-target-modules-t2m",
        type=str,
        default="full",
        help="LoRA target modules for t2m adapter: 'full', 'qv', or comma-separated module names",
    )
    p.add_argument(
        "--lora-target-modules-m2t",
        type=str,
        default="full",
        help="LoRA target modules for m2t adapter: 'full', 'qv', or comma-separated module names",
    )

    # VQ-VAE
    p.add_argument("--dataname", type=str, default="t2m")
    p.add_argument("--code-dim", type=int, default=512)
    p.add_argument("--nb-code", type=int, default=512)
    p.add_argument("--mu", type=float, default=0.99)
    p.add_argument("--down-t", type=int, default=2)
    p.add_argument("--stride-t", type=int, default=2)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dilation-growth-rate", type=int, default=3)
    p.add_argument("--output-emb-width", type=int, default=512)
    p.add_argument("--vq-act", type=str, default="relu")
    p.add_argument("--vq-norm", type=str, default=None)
    p.add_argument("--quantizer", type=str, default="ema_reset")
    p.add_argument("--beta", type=float, default=1.0)

    # Paths
    project_root = MOTION_AGENT_PATH.parent
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: experiments/multitask/<direction>/v1)")
    p.add_argument("--vq-path", type=str,
                   default=str(PRETRAINED / "ckpt" / "vqvae.pth"))
    p.add_argument("--pretrained-path", type=str,
                   default=str(project_root / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"))
    p.add_argument("--nb-joints", type=int, default=22)
    p.add_argument("--task-splits-path", type=str,
                   default=str(TASK_SPLITS))
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--use-pretrained-lora", action="store_true",
                   help="Load pretrained LoRA (not recommended for CL comparison)")

    # Minimal pretraining (holdout) knobs
    p.add_argument(
        "--init-from-scratch",
        action="store_true",
        help=(
            "Start from base LLM without merging a pretrained motion adapter. "
            "Used for the holdout pretraining stage."
        ),
    )
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional MotionLLM checkpoint (.pth) to load before training. "
            "Useful for pretraining m2t after pretraining t2m on holdout."
        ),
    )
    p.add_argument(
        "--train-motion-interface",
        action="store_true",
        help=(
            "Enable training of motion-token embedding/lm_head rows (masked grads). "
            "Recommended when starting from scratch (Gemma) on the holdout pretraining split."
        ),
    )

    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = str(project_root / "experiments" / "multitask" / args.training_task / "v1")

    return args


def main():
    args = parse_args()

    # Apply capacity scaling to the adapter that is actually trained.
    if args.capacity_multiplier < 1:
        raise ValueError("--capacity-multiplier must be >= 1")
    if args.training_task == "t2m":
        args.lora_r_t2m *= args.capacity_multiplier
        args.lora_alpha_t2m *= args.capacity_multiplier
    else:
        args.lora_r_m2t *= args.capacity_multiplier
        args.lora_alpha_m2t *= args.capacity_multiplier

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.use_bf16 else "no",
        kwargs_handlers=[ddp_kwargs],
    )
    args.device = accelerator.device

    exp_dir = args.out_dir
    if accelerator.is_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        logger = get_logger(exp_dir)

        logger.info("=" * 70)
        logger.info(f"MULTI-TASK LEARNING [{args.training_task.upper()}]")
        logger.info("Upper bound for continual learning comparison")
        logger.info("=" * 70)
        args_dict = vars(args).copy()
        args_dict["device"] = str(args_dict["device"])
        logger.info(json.dumps(args_dict, indent=4))
    else:
        logger = logging.getLogger("MultiTask-Child")
        logger.addHandler(logging.NullHandler())

    # W&B
    if accelerator.is_main_process and not args.no_wandb:
        cfg = vars(args).copy()
        cfg["device"] = str(cfg["device"])
        cfg["num_gpus"] = torch.cuda.device_count()
        run_name = args.wandb_name or f"multitask_{args.training_task}"
        wandb.init(
            project=args.wandb_project, name=run_name, config=cfg,
            tags=["multi-task", args.training_task, "upper-bound", "fresh-lora"],
        )
        wandb.define_metric("batch/*", step_metric="global_step")
        wandb.define_metric("epoch/*", step_metric="epoch")

    w_vectorizer = WordVectorizer(str(MOTION_AGENT_PATH / str(PRETRAINED / 'glove')), "our_vab")

    if accelerator.is_main_process:
        logger.info("Initializing MotionLLM...")
    model = MotionLLM(args)
    model.training_task = args.training_task
    model.llm.set_adapter(args.training_task)

    # Memory optimization (optional)
    if args.gradient_checkpointing:
        # PEFT wraps the base model; enable checkpointing on the underlying transformer.
        model.llm.base_model.model.gradient_checkpointing_enable()
        # Required by HF for gradient checkpointing with some PEFT wrappers.
        model.llm.enable_input_require_grads()
        if accelerator.is_main_process:
            logger.info("Gradient checkpointing enabled (base model)")

    # Optional init checkpoint (load BEFORE merge-then-adapt).
    if args.init_checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Loading init checkpoint: {args.init_checkpoint}")
        model.load_model(args.init_checkpoint)

    if args.train_motion_interface:
        if accelerator.is_main_process:
            logger.info("Enabling motion-token interface training (masked grads)")
        model.enable_motion_token_interface_training()

    # Checkpoint loading: merge-then-adapt pattern (default) OR scratch init.
    if args.init_from_scratch:
        if accelerator.is_main_process:
            logger.info("Mode: INIT-FROM-SCRATCH (no pretrained adapter merge)")
    else:
        if accelerator.is_main_process:
            logger.info(f"Mode: MERGE-THEN-ADAPT (pretrained {args.training_task} LoRA merged into base)")

        if os.path.exists(args.pretrained_path):
            if accelerator.is_main_process:
                logger.info(f"Merging pretrained {args.training_task} adapter: {args.pretrained_path}")
            model.merge_pretrained_adapter(args.pretrained_path, args.training_task)
        model.llm.set_adapter(args.training_task)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if accelerator.is_main_process:
            logger.info(f"Resuming from: {args.resume}")
        model.load_model(args.resume)
        ckpt_name = os.path.basename(args.resume)
        if "epoch_" in ckpt_name:
            try:
                start_epoch = int(ckpt_name.split("epoch_")[1].split(".")[0])
            except ValueError:
                pass

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    save_epochs = [int(e) for e in args.save_epochs.split(",")]
    if accelerator.is_main_process:
        logger.info(f"Save checkpoints at epochs: {save_epochs}")

    # Data
    is_distributed = accelerator.num_processes > 1

    if args.split_mode == "random_80_20":
        if get_multitask_random_split_loader is None:
            raise ImportError(
                "Random split multitask loader is unavailable. "
                "Expected baselines/multi_task/data_loader_random_split.py to be importable."
            )
        # In random_80_20 mode there is no separate val split.
        train_loader = get_multitask_random_split_loader(
            split="train",
            batch_size=args.batch_size,
            w_vectorizer=w_vectorizer,
            unit_length=2 ** args.down_t,
            num_workers=args.num_workers,
            task_splits_path=args.task_splits_path,
            train_ratio=args.train_ratio,
            random_seed=args.split_seed,
            stratified=True,
            distributed=is_distributed,
        )
        val_loader = get_multitask_random_split_loader(
            split="test",
            batch_size=args.batch_size,
            w_vectorizer=w_vectorizer,
            unit_length=2 ** args.down_t,
            num_workers=args.num_workers,
            task_splits_path=args.task_splits_path,
            train_ratio=args.train_ratio,
            random_seed=args.split_seed,
            stratified=False,
            distributed=False,
        )
    else:
        train_loader = get_multitask_loader(
            split=args.train_split,
            batch_size=args.batch_size,
            w_vectorizer=w_vectorizer,
            unit_length=2 ** args.down_t,
            num_workers=args.num_workers,
            stratified=True,
            task_splits_path=args.task_splits_path,
            distributed=is_distributed,
        )
        val_loader = get_multitask_loader(
            split="val",
            batch_size=args.batch_size,
            w_vectorizer=w_vectorizer,
            unit_length=2 ** args.down_t,
            num_workers=args.num_workers,
            stratified=False,
            task_splits_path=args.task_splits_path,
            distributed=False,
        )

    if accelerator.is_main_process:
        logger.info(f"Training: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")
        logger.info(f"Validation: {len(val_loader.dataset)} samples")
        logger.info(f"Split mode: {args.split_mode} (seed={args.split_seed})")
        if hasattr(train_loader, "sampling_mode"):
            logger.info(f"Sampling mode: {getattr(train_loader, 'sampling_mode')}")
        elif args.split_mode == "random_80_20":
            logger.info("Sampling mode: <unknown>")
        if is_distributed and args.split_mode == "random_80_20":
            logger.warning(
                "Distributed training detected: balanced sampling is disabled under DDP; "
                "training will use unbalanced shuffle sampling across tasks. "
                "For balanced multitask sampling, run with --num_processes 1."
            )
        logger.info(f"Capacity multiplier: {args.capacity_multiplier}")
        logger.info(f"LoRA r/alpha (t2m): r={args.lora_r_t2m}, alpha={args.lora_alpha_t2m}")
        logger.info(f"LoRA r/alpha (m2t): r={args.lora_r_m2t}, alpha={args.lora_alpha_m2t}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Cosine LR schedule with warmup (step-based)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * max(1, args.epochs - start_epoch))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if accelerator.is_main_process:
        try:
            lr0 = optimizer.param_groups[0]["lr"]
        except Exception:
            lr0 = None
        logger.info(
            f"LR schedule: cosine(warmup_ratio={args.warmup_ratio}, warmup_steps={warmup_steps}, total_steps={total_steps}, lr0={lr0})"
        )

    # Early stopping state (main process only)
    best_metric = float("inf") if args.early_stopping_metric == "val_loss" else -float("inf")
    patience_left = args.early_stopping_patience

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        batch_losses, batch_accs = [], []
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process, dynamic_ncols=True,
        )
        for batch in pbar:
            word_emb, pos_oh, caption, sent_len, motion, m_length, token, name = batch

            with accelerator.accumulate(model):
                motion_tokens = []
                um = accelerator.unwrap_model(model)
                for i in range(motion.shape[0]):
                    m_in = motion[i:i+1, :m_length[i], :].to(accelerator.device)
                    with torch.no_grad():
                        tok = um.net.encode(m_in).squeeze(0)
                    tok_remapped = torch.from_numpy(
                        um.motion_token_indices[tok.cpu().numpy()]
                    ).to(accelerator.device)
                    motion_tokens.append(tok_remapped)

                loss, gen_acc, _, _ = model(caption, motion_tokens)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                global_step += 1

                if accelerator.is_main_process:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{gen_acc:.4f}"})
                    if not args.no_wandb:
                        wandb.log({"batch/loss": loss.item(), "batch/accuracy": gen_acc,
                                   "global_step": global_step})

        if accelerator.is_main_process:
            dt = time.time() - t0
            avg_loss = np.mean(batch_losses)
            avg_acc = np.mean(batch_accs)
            logger.info(f"Epoch {epoch+1}/{args.epochs} [Train]: "
                        f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={dt:.1f}s")

        # Validation
        if accelerator.is_main_process:
            um = accelerator.unwrap_model(model)
            val_loss, val_acc = compute_token_accuracy(um, val_loader, accelerator.device)
            logger.info(f"Epoch {epoch+1}/{args.epochs} [Val]:   "
                        f"Loss={val_loss:.4f}, Acc={val_acc:.4f}")

            # Early stopping / best checkpoint (uses current val split)
            if args.early_stopping:
                if args.early_stopping_metric == "val_loss":
                    current = val_loss
                    improved = (best_metric - current) > args.early_stopping_min_delta
                else:
                    current = val_acc
                    improved = (current - best_metric) > args.early_stopping_min_delta

                if improved:
                    best_metric = current
                    patience_left = args.early_stopping_patience
                    best_ckpt = os.path.join(exp_dir, "multitask_best.pth")
                    accelerator.unwrap_model(model).save_model(best_ckpt)
                    logger.info(
                        f"New best ({args.early_stopping_metric}={current:.4f}). Saved: {best_ckpt}"
                    )
                else:
                    patience_left -= 1
                    logger.info(
                        f"No improvement on {args.early_stopping_metric} (best={best_metric:.4f}). "
                        f"Patience left: {patience_left}/{args.early_stopping_patience}"
                    )
                    if patience_left <= 0:
                        logger.info("Early stopping triggered. Stopping training.")
                        break

            if not args.no_wandb:
                wandb.log({"epoch/train_loss": avg_loss, "epoch/train_accuracy": avg_acc,
                           "epoch/val_loss": val_loss, "epoch/val_accuracy": val_acc,
                           "epoch/duration_sec": dt, "epoch": epoch})
            model.train()

        if accelerator.is_main_process and (epoch + 1) in save_epochs:
            ckpt = os.path.join(exp_dir, f"multitask_epoch_{epoch+1}.pth")
            accelerator.unwrap_model(model).save_model(ckpt)
            logger.info(f"Saved: {ckpt}")

    # Final checkpoint
    if accelerator.is_main_process:
        final = os.path.join(exp_dir, "multitask_final.pth")
        accelerator.unwrap_model(model).save_model(final)
        logger.info("=" * 50)
        logger.info(f"TRAINING COMPLETE  |  Final: {final}")
        logger.info("=" * 50)
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
