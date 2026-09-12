"""
O-LoRA Multi-Adapter Training -- Unified T2M / M2T

Implements the exact O-LoRA algorithm from:
"Orthogonal Subspace Learning for Language Model Continual Learning"
Wang et al., EMNLP 2023 Findings

Per-task adapters with orthogonality loss:
    L = L_task + lambda_orth * L_orth
    L_orth = sum_{i<t} ||A_i^T @ A_t||_F^2

CRITICAL -- forward pass uses model.llm() directly, NOT model.forward(),
because model.forward() would call set_adapter(training_task) and override
the per-task O-LoRA adapter.

Usage:
    # T2M
    accelerate launch --num_processes 4 --mixed_precision bf16 train_olora.py \
        --training-task t2m --epochs-per-task 20

    # M2T
    accelerate launch --num_processes 4 --mixed_precision bf16 train_olora.py \
        --training-task m2t --epochs-per-task 20
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
from tqdm import tqdm

MOTION_AGENT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MOTION_AGENT_PATH))

from continual_learning.olora_multi_adapter.olora_manager import MultiAdapterOLoRAManager
from continual_learning.utils.data_manager import (
    get_task_loader, TASK_NAMES, set_split_mode,
)
from models.mllm import MotionLLM
from models.training_utils import process_batch
from utils.word_vectorizer import WordVectorizer


def load_motion_embeddings(model, checkpoint_path, device):
    """Load ONLY motion token embeddings and lm_head (no LoRA weights)."""
    print(f"[O-LoRA] Loading embeddings from {checkpoint_path}")
    save_dict = torch.load(checkpoint_path, map_location=device)
    lora_keys = [k for k in save_dict if "lora" in k.lower()]
    print(f"[O-LoRA] Skipping {len(lora_keys)} pre-trained LoRA weights")
    if "embeddings" in save_dict:
        model.llm.get_input_embeddings().weight.data[model.nb_text_tokens:] = save_dict["embeddings"]
        print(f"[O-LoRA] Loaded embeddings: {save_dict['embeddings'].shape}")
    if "lm_head" in save_dict:
        model.llm.lm_head.weight.data[model.nb_text_tokens:] = save_dict["lm_head"]
        print(f"[O-LoRA] Loaded lm_head: {save_dict['lm_head'].shape}")


def get_logger(out_dir, name="OLoRA"):
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


def verify_freeze_state(model, adapter_name, task_id, logger):
    """Log and verify which parameters are trainable vs frozen after prepare_task()."""
    trainable_lora = []
    frozen_lora = []
    for n, p in model.llm.named_parameters():
        if "lora" not in n:
            continue
        if p.requires_grad:
            trainable_lora.append(n)
        else:
            frozen_lora.append(n)

    logger.info(f"  Trainable LoRA params: {len(trainable_lora)}")
    logger.info(f"  Frozen LoRA params:    {len(frozen_lora)}")

    bad = [n for n in trainable_lora if adapter_name not in n]
    if bad:
        logger.error(f"  BUG: Unexpected trainable LoRA (not {adapter_name}): {bad[:5]}")
        raise RuntimeError(f"Freeze verification failed: {len(bad)} unexpected trainable LoRA params")

    for n, p in model.llm.named_parameters():
        if (".t2m." in n or ".m2t." in n) and p.requires_grad:
            logger.error(f"  BUG: Base adapter param is trainable: {n}")
            raise RuntimeError(f"Base adapter {n} should be frozen")

    emb_frozen = not model.llm.get_input_embeddings().weight.requires_grad
    lmh_frozen = not model.llm.lm_head.weight.requires_grad
    if not emb_frozen or not lmh_frozen:
        logger.warning(f"  embeddings frozen={emb_frozen}, lm_head frozen={lmh_frozen}")

    logger.info(f"  Freeze verification PASSED for task {task_id} (adapter={adapter_name})")


def parse_args():
    p = argparse.ArgumentParser(description="O-LoRA Multi-Adapter Training")

    p.add_argument("--training-task", type=str, required=True,
                   choices=["t2m", "m2t"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--epochs-per-task", type=int, default=20)
    p.add_argument("--start-task", type=int, default=0)
    p.add_argument("--train-split", type=str, default="trainval",
                   choices=["train", "trainval"])
    p.add_argument("--lambda-orth", type=float, default=0.5)
    p.add_argument(
        "--olora-target-modules",
        type=str,
        default="qv",
        help="Target modules for O-LoRA per-task adapters: 'qv', 'full', or comma-separated module names.",
    )
    p.add_argument("--split-mode", type=str, default="random_80_20",
                   choices=["predefined", "random_80_20"])
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--wandb-project", type=str, default="motion-agent-olora")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")

    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=64)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--olora-r", type=int, default=None,
                   help="LoRA rank for O-LoRA per-task adapters (default: matches direction adapter)")
    p.add_argument("--olora-alpha", type=int, default=None,
                   help="LoRA alpha for O-LoRA per-task adapters")
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False,
                   help="Enable gradient checkpointing to save memory (trades compute for memory)")

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

    PROJECT_ROOT = MOTION_AGENT_PATH.parent
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--vq-path", type=str,
                   default=str(PRETRAINED / "ckpt" / "vqvae.pth"))
    p.add_argument("--pretrained-path", type=str,
                   default=str(PROJECT_ROOT / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"))
    p.add_argument("--nb-joints", type=int, default=22)

    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = str(PROJECT_ROOT / "experiments" / "olora_multi" / args.training_task / "v1")

    if args.olora_r is None:
        args.olora_r = args.lora_r_t2m if args.training_task == "t2m" else args.lora_r_m2t
    if args.olora_alpha is None:
        args.olora_alpha = args.lora_alpha_t2m if args.training_task == "t2m" else args.lora_alpha_m2t

    return args


def _resolve_target_modules(mode: str) -> list[str]:
    mode = (mode or "qv").strip().lower()
    if mode in {"qv", "q_proj,v_proj", "q_proj_v_proj"}:
        return ["q_proj", "v_proj"]
    if mode in {"full", "default"}:
        return [
            "o_proj", "q_proj", "up_proj", "v_proj",
            "k_proj", "down_proj", "gate_proj",
        ]
    if "," in mode:
        return [m.strip() for m in mode.split(",") if m.strip()]
    raise ValueError(
        f"Unknown --olora-target-modules={mode}. Use 'qv', 'full', or a comma-separated list."
    )


def main():
    args = parse_args()

    # Normalize olora_target_modules to a list so downstream is consistent.
    args.olora_target_modules = _resolve_target_modules(args.olora_target_modules)

    set_split_mode(args.split_mode, args.split_seed)

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
        logger = get_logger(exp_dir, f"OLoRA-{args.training_task}")

        logger.info("=" * 70)
        logger.info(f"O-LoRA MULTI-ADAPTER [{args.training_task.upper()}] TRAINING")
        logger.info("=" * 70)
        args_dict = vars(args).copy()
        args_dict["device"] = str(args_dict["device"])
        logger.info(json.dumps(args_dict, indent=4))
    else:
        logger = logging.getLogger("OLoRA-Child")
        logger.addHandler(logging.NullHandler())

    if accelerator.is_main_process and not args.no_wandb:
        import wandb
        cfg = vars(args).copy()
        cfg["device"] = str(cfg["device"])
        run_name = args.wandb_name or f"olora_{args.training_task}"
        wandb.init(project=args.wandb_project, name=run_name, config=cfg,
                   tags=["olora", args.training_task])

    w_vectorizer = WordVectorizer(str(MOTION_AGENT_PATH / str(PRETRAINED / 'glove')), "our_vab")

    if accelerator.is_main_process:
        logger.info("Initializing MotionLLM...")
    model = MotionLLM(args)
    model.training_task = args.training_task

    # Merge-then-adapt: bake pretrained LoRA into base weights, add fresh LoRA
    if os.path.exists(args.pretrained_path):
        if accelerator.is_main_process:
            logger.info(f"Merging pretrained {args.training_task} adapter: {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, args.training_task)
    elif accelerator.is_main_process:
        logger.warning(f"Pretrained not found: {args.pretrained_path}")

    if args.gradient_checkpointing:
        model.llm.base_model.model.gradient_checkpointing_enable()
        model.llm.enable_input_require_grads()
        if accelerator.is_main_process:
            logger.info("Gradient checkpointing enabled (base model)")

    olora_manager = MultiAdapterOLoRAManager(
        model=model.llm,
        lora_r=args.olora_r,
        lora_alpha=args.olora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.olora_target_modules,
    )

    if accelerator.is_main_process:
        logger.info(f"O-LoRA Manager: r={args.olora_r}, alpha={args.olora_alpha}, "
                     f"lambda_orth={args.lambda_orth}, targets={args.olora_target_modules}")

    for task_id in range(args.start_task, args.num_tasks):
        if accelerator.is_main_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"TASK {task_id} ({TASK_NAMES[task_id]})")
            logger.info(f"Completed: {olora_manager.get_num_completed_tasks()}")
            logger.info(f"{'='*60}")

        adapter_name = olora_manager.prepare_task(task_id, model.llm)
        model.adapter_override = adapter_name

        if accelerator.is_main_process:
            info = olora_manager.get_trainable_params_info(model.llm)
            logger.info(f"Adapter '{adapter_name}': {info['trainable_params']:,} trainable ({info['trainable_percent']:.2f}%)")
            verify_freeze_state(model, adapter_name, task_id, logger)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

        train_loader = get_task_loader(
            args.dataname, args.train_split, args.batch_size, w_vectorizer,
            task_id, unit_length=2 ** args.down_t,
        )
        if len(train_loader) == 0:
            if accelerator.is_main_process:
                logger.warning(f"Task {task_id} has no data, skipping.")
            continue

        # Re-prepare each task so DDP discovers new adapter parameters
        model_prepared, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader)
        model_prepared.train()

        for epoch in range(args.epochs_per_task):
            batch_losses, batch_task_losses, batch_orth_losses, batch_accs = [], [], [], []
            t0 = time.time()

            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{args.epochs_per_task}",
                        disable=not accelerator.is_main_process, dynamic_ncols=True)

            for batch in pbar:
                word_emb, pos_oh, caption, sent_len, motion, m_length, token, name = batch

                with accelerator.accumulate(model_prepared):
                    um = accelerator.unwrap_model(model_prepared)

                    motion_tokens = []
                    for i in range(motion.shape[0]):
                        m_in = motion[i:i+1, :m_length[i], :].to(accelerator.device)
                        with torch.no_grad():
                            tok = um.net.encode(m_in).squeeze(0)
                        tok_remapped = torch.from_numpy(
                            um.motion_token_indices[tok.cpu().numpy()]
                        ).to(accelerator.device)
                        motion_tokens.append(tok_remapped)

                    task_loss, gen_acc, _, _ = model_prepared(
                        caption=list(caption), motion=motion_tokens,
                    )
                    orth_loss = olora_manager.compute_orthogonality_loss(um.llm)
                    total_loss = task_loss + args.lambda_orth * orth_loss

                    accelerator.backward(total_loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_losses.append(total_loss.item())
                    batch_task_losses.append(task_loss.item())
                    orth_val = orth_loss.item() if isinstance(orth_loss, torch.Tensor) else orth_loss
                    batch_orth_losses.append(orth_val)
                    batch_accs.append(gen_acc)

                    if accelerator.is_main_process:
                        pbar.set_postfix({"loss": f"{total_loss.item():.3f}",
                                          "task": f"{task_loss.item():.3f}",
                                          "orth": f"{orth_val:.4f}",
                                          "acc": f"{gen_acc:.3f}"})

            if accelerator.is_main_process:
                dt = time.time() - t0
                logger.info(
                    f"Task {task_id} Epoch {epoch+1}/{args.epochs_per_task}: "
                    f"Total={np.mean(batch_losses):.4f}, Task={np.mean(batch_task_losses):.4f}, "
                    f"Orth={np.mean(batch_orth_losses):.4f}, Acc={np.mean(batch_accs):.4f}, "
                    f"Time={dt:.1f}s"
                )

        # After task training
        accelerator.wait_for_everyone()
        um = accelerator.unwrap_model(model_prepared)
        olora_manager.complete_task(task_id, um.llm)

        if accelerator.is_main_process:
            ckpt_path = os.path.join(exp_dir, f"olora_multi_task_{task_id}_best.pth")
            um.save_model(ckpt_path)
            logger.info(f"Saved: {ckpt_path}")
            logger.info(f"Task {task_id} complete. Adapters: {olora_manager.get_active_adapters()}")

        accelerator.free_memory()
        del train_loader, optimizer, model_prepared
        gc.collect()
        torch.cuda.empty_cache()


    if accelerator.is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info(f"O-LoRA [{args.training_task.upper()}] TRAINING COMPLETE")
        logger.info(f"Tasks: {olora_manager.get_num_completed_tasks()}, "
                     f"Adapters: {olora_manager.get_active_adapters()}")
        logger.info("=" * 60)
        if not args.no_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
