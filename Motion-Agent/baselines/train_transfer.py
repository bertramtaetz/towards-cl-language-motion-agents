"""
Transfer Learning (Sequential Fine-tuning) for Motion-Agent

Trains sequentially on tasks 1->2->3->4->5, establishing the LOWER BOUND
for continual learning experiments. Supports both T2M and M2T directions.

Default mode uses fresh LoRA adapters (randomly initialized) with only
motion token embeddings loaded from the pretrained checkpoint. This ensures
true catastrophic forgetting measurement.

Task Order (motion-based clustering, maximises forgetting):
    Task 0: Jumping (Cluster 8)
    Task 1: Arms/Hands (Cluster 4)
    Task 2: Walking (Cluster 2)
    Task 3: Gestures (Cluster 5)
    Task 4: Sit/Stand (Cluster 14)

Usage:
    # T2M — train task 0 with fresh LoRA
    accelerate launch --num_processes 4 train_transfer.py \
        --training-task t2m --task-id 0 --task-name task_1_jumping

    # M2T — train task 0 with fresh LoRA
    accelerate launch --num_processes 4 train_transfer.py \
        --training-task m2t --task-id 0 --task-name task_1_jumping

    # Resume from a previous task checkpoint
    accelerate launch --num_processes 4 train_transfer.py \
        --training-task t2m --task-id 1 --task-name task_2_arms_hands \
        --resume <checkpoint_dir>/after_task0_task_1_jumping.pth
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import logging
import os
import sys
import time
import re

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
from tqdm import tqdm

MOTION_AGENT_PATH = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(MOTION_AGENT_PATH))

from models.mllm import MotionLLM
from utils.word_vectorizer import WordVectorizer

# Use the same CL data manager as O-LoRA so we can align split protocol.
from continual_learning.utils.data_manager import get_task_loader, set_split_mode

sys.path.insert(0, str(Path(__file__).parent / "transfer_learning"))
from data_loader import TASK_ORDER


def _detect_task_adapters_from_state_dict(state_dict: dict) -> list[str]:
    """Return sorted adapter names like ["task_0", "task_1", ...] found in a checkpoint."""
    adapters: set[str] = set()
    for k in state_dict.keys():
        for part in k.split("."):
            if re.fullmatch(r"task_\d+", part):
                adapters.add(part)
    return sorted(adapters, key=lambda s: int(s.split("_")[1]))


def _ensure_task_adapters_exist(model: MotionLLM, adapter_names: list[str], training_task: str) -> None:
    """Add missing task adapters to model.llm so load_model can fill their weights."""
    if not adapter_names:
        return
    lora_config = model.lora_config_t2m if training_task == "t2m" else model.lora_config_m2t
    existing = set(getattr(model.llm, "peft_config", {}).keys())
    for name in adapter_names:
        if name in existing:
            continue
        model.llm.add_adapter(name, lora_config)


def _copy_adapter_weights(peft_model, src: str, dst: str) -> None:
    """Copy LoRA A/B matrices from src adapter to dst adapter (warm-start)."""
    copied = 0
    for _, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        # PEFT stores adapters in ModuleDicts.
        if src not in module.lora_A or dst not in module.lora_A:
            continue
        module.lora_A[dst].weight.data.copy_(module.lora_A[src].weight.data)
        module.lora_B[dst].weight.data.copy_(module.lora_B[src].weight.data)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"No LoRA modules copied when warm-starting {dst} from {src}.")


def _set_trainable_only_for_adapter(peft_model, adapter_name: str) -> None:
    """Freeze all LoRA params except those belonging to adapter_name."""
    for n, p in peft_model.named_parameters():
        if "lora" not in n:
            continue
        p.requires_grad = adapter_name in n


def get_logger(out_dir, name="Transfer-Exp"):
    logger = logging.getLogger(name)
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


def parse_args():
    p = argparse.ArgumentParser(description="Transfer Learning for Motion-Agent")

    # Direction
    p.add_argument("--training-task", type=str, required=True,
                   choices=["t2m", "m2t"],
                   help="Training direction: t2m (text-to-motion) or m2t (motion-to-text)")

    # Task
    p.add_argument("--task-id", type=int, required=True, help="Task index (0-4)")
    p.add_argument("--task-name", type=str, required=True, help="Task name")

    # Hyperparameters
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=20,
                   help="Save per-epoch checkpoint every N epochs (set high to save only final)")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint from previous task")

    # W&B
    p.add_argument("--wandb-project", type=str, default="motion-agent-transfer")
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
                   help="Checkpoint output directory (default: experiments/transfer_learning/<task>/v1)")
    p.add_argument("--vq-path", type=str,
                   default=str(PRETRAINED / "ckpt" / "vqvae.pth"))
    p.add_argument("--pretrained-path", type=str,
                   default=str(project_root / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"))
    p.add_argument("--nb-joints", type=int, default=22)
    p.add_argument("--task-splits-path", type=str,
                   default=str(TASK_SPLITS))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-split", type=str, default="trainval",
                   choices=["train", "trainval"])

    # Split protocol alignment (optional)
    p.add_argument(
        "--split-mode",
        type=str,
        default="random_80_20",
        choices=["predefined", "random_80_20"],
        help="Data split mode. Use 'random_80_20' to match O-LoRA defaults.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used when split-mode is random_80_20",
    )

    # Fresh LoRA
    p.add_argument("--use-pretrained-lora", action="store_true",
                   help="Load pretrained LoRA (not recommended for CL experiments)")

    # Transfer mode
    # - shared (default): current baseline, one shared adapter per direction (t2m/m2t)
    # - multi:            one adapter per task (task_0..task_4) with warm-start
    p.add_argument(
        "--transfer-mode",
        type=str,
        default="shared",
        choices=["shared", "multi"],
        help="Transfer learning mode: 'shared' adapter per direction (default) or 'multi' per-task adapters.",
    )

    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = str(project_root / "experiments" / "transfer_learning" / args.training_task / "v1")

    return args


def main():
    args = parse_args()

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
        logger = get_logger(exp_dir, f"Transfer-{args.training_task}-Task{args.task_id}")

        logger.info("=" * 70)
        logger.info(f"TRANSFER LEARNING [{args.training_task.upper()}] — "
                     f"Task {args.task_id}: {args.task_name}")
        logger.info("=" * 70)
        logger.info(f"Task Order: {[t[0] for t in TASK_ORDER]}")
        args_dict = vars(args).copy()
        args_dict["device"] = str(args_dict["device"])
        logger.info(json.dumps(args_dict, indent=4))
    else:
        logger = logging.getLogger("Transfer-Child")
        logger.addHandler(logging.NullHandler())

    # W&B
    if accelerator.is_main_process and not args.no_wandb:
        cfg = vars(args).copy()
        cfg["device"] = str(cfg["device"])
        cfg["num_gpus"] = torch.cuda.device_count()
        run_name = args.wandb_name or f"transfer_{args.training_task}_task{args.task_id}_{args.task_name}"
        wandb.init(
            project=args.wandb_project, name=run_name, config=cfg,
            tags=["transfer-learning", args.training_task, f"task{args.task_id}"],
        )
        wandb.define_metric("batch/*", step_metric="global_step")
        wandb.define_metric("epoch/*", step_metric="epoch")

    # Word vectorizer
    w_vectorizer = WordVectorizer(str(MOTION_AGENT_PATH / str(PRETRAINED / 'glove')), "our_vab")

    # Model
    if accelerator.is_main_process:
        logger.info("Initializing MotionLLM...")
    model = MotionLLM(args)
    model.training_task = args.training_task

    if args.transfer_mode == "shared":
        model.llm.set_adapter(args.training_task)
    else:
        # Per-task adapter training uses adapter_override.
        model.adapter_override = None

    if args.transfer_mode == "shared":
        # Checkpoint loading: merge-then-adapt pattern
        # Always merge pretrained adapter into base weights first, then load
        # task-specific LoRA from previous checkpoint if continuing.
        if accelerator.is_main_process:
            logger.info(
                f"Mode: MERGE-THEN-ADAPT (pretrained {args.training_task} LoRA merged into base)"
            )

        if os.path.exists(args.pretrained_path):
            if accelerator.is_main_process:
                logger.info(f"Merging pretrained {args.training_task} adapter: {args.pretrained_path}")
            model.merge_pretrained_adapter(args.pretrained_path, args.training_task)
        model.llm.set_adapter(args.training_task)

        if args.resume and os.path.exists(args.resume):
            if accelerator.is_main_process:
                logger.info(f"Resuming from: {args.resume}")
            model.load_model(args.resume)
        elif args.task_id > 0:
            prev_id = args.task_id - 1
            prev_name = TASK_ORDER[prev_id][0]
            prev_ckpt = os.path.join(exp_dir, f"after_task{prev_id}_{prev_name}.pth")
            if os.path.exists(prev_ckpt):
                if accelerator.is_main_process:
                    logger.info(f"Loading previous task LoRA: {prev_ckpt}")
                model.load_model(prev_ckpt)
    else:
        # transfer_mode == multi
        # We keep the pretrained direction adapter (t2m/m2t) around for O-LoRA comparable
        # evaluation (unseen tasks use direction adapter).
        if accelerator.is_main_process:
            logger.info(
                "Mode: TRANSFER-MULTI (per-task adapters task_0..task_4, warm-start; keep direction adapter for unseen tasks)"
            )

        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_path}")

        # Load pretrained direction adapter weights + embeddings.
        model.load_model(args.pretrained_path)
        model.llm.set_adapter(args.training_task)

        # Load previous stage checkpoint (contains task_0..task_{t-1}) if needed.
        prev_ckpt = None
        if args.resume and os.path.exists(args.resume):
            prev_ckpt = args.resume
        elif args.task_id > 0:
            prev_id = args.task_id - 1
            prev_name = TASK_ORDER[prev_id][0]
            prev_ckpt = os.path.join(exp_dir, f"after_task{prev_id}_{prev_name}.pth")

        if prev_ckpt and os.path.exists(prev_ckpt):
            if accelerator.is_main_process:
                logger.info(f"Loading previous stage checkpoint: {prev_ckpt}")
            sd = torch.load(prev_ckpt, map_location="cpu")
            adapters = _detect_task_adapters_from_state_dict(sd)
            _ensure_task_adapters_exist(model, adapters, args.training_task)
            model.load_model(prev_ckpt)

        # Ensure current adapter exists and warm-start it.
        current_adapter = f"task_{args.task_id}"
        _ensure_task_adapters_exist(model, [current_adapter], args.training_task)

        src_adapter = args.training_task if args.task_id == 0 else f"task_{args.task_id - 1}"
        model.llm.set_adapter(src_adapter)
        _copy_adapter_weights(model.llm, src=src_adapter, dst=current_adapter)
        model.llm.set_adapter(current_adapter)

        # Freeze all other LoRA params.
        _set_trainable_only_for_adapter(model.llm, current_adapter)
        model.adapter_override = current_adapter

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    # Data
    is_distributed = accelerator.num_processes > 1
    # Align split protocol with O-LoRA (random_80_20 by default)
    set_split_mode(args.split_mode, args.split_seed)

    train_loader = get_task_loader(
        dataset_name=args.dataname,
        split=args.train_split,
        batch_size=args.batch_size,
        w_vectorizer=w_vectorizer,
        task_id=args.task_id,
        num_workers=args.num_workers,
        unit_length=2 ** args.down_t,
        task_splits_path=args.task_splits_path,
    )

    # For strict comparability with O-LoRA, skip per-epoch validation.
    # In random_80_20 split mode there is no dedicated val split.
    val_loader = None

    if accelerator.is_main_process:
        logger.info(f"Split mode: {args.split_mode} (seed={args.split_seed})")
        logger.info(f"Train split: {args.train_split}, Train samples: {len(train_loader.dataset)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        batch_losses, batch_accs = [], []
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Task {args.task_id} [{args.task_name}] Epoch {epoch+1}/{args.epochs}",
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
                optimizer.zero_grad()

                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                global_step += 1

                if accelerator.is_main_process:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{gen_acc:.4f}"})
                    if not args.no_wandb:
                        wandb.log({"batch/loss": loss.item(), "batch/accuracy": gen_acc,
                                   "global_step": global_step})

        # Epoch summary
        if accelerator.is_main_process:
            dt = time.time() - t0
            avg_loss = np.mean(batch_losses)
            avg_acc = np.mean(batch_accs)
            logger.info(f"Task {args.task_id} Epoch {epoch+1}/{args.epochs} [Train]: "
                        f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={dt:.1f}s")

        # Validation intentionally skipped for comparability with O-LoRA.
        # If you want it back, implement an evaluation loader for split-mode.
        if accelerator.is_main_process and not args.no_wandb:
            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/train_accuracy": avg_acc,
                "epoch/duration_sec": dt,
                "epoch": epoch,
            })

        # Periodic checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_interval == 0:
            ckpt = os.path.join(exp_dir, f"task{args.task_id}_{args.task_name}_epoch{epoch+1}.pth")
            accelerator.unwrap_model(model).save_model(ckpt)
            logger.info(f"Saved: {ckpt}")

    # Final checkpoint
    if accelerator.is_main_process:
        final = os.path.join(exp_dir, f"after_task{args.task_id}_{args.task_name}.pth")
        accelerator.unwrap_model(model).save_model(final)
        logger.info(f"{'='*50}")
        logger.info(f"Task {args.task_id} ({args.task_name}) complete!  Saved: {final}")
        logger.info(f"{'='*50}")
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
