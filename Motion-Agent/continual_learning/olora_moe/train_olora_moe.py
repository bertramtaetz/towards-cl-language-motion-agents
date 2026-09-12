"""O-LoRA MoE training.

Trains the standard O-LoRA multi-adapter setup (one LoRA adapter per task +
orthogonality regularizer) AND additionally trains an autoencoder router that
enables task-id-free inference.

High-level protocol per task t:
  1) Create/activate adapter `task_{t}`; freeze previous adapters.
  2) Train O-LoRA task adapter with orthogonality loss (same as olora_multi_adapter).
  3) Train router autoencoder for task t on routing embeddings extracted from data.
  4) Update router threshold for unseen detection.
  5) Save checkpoint that includes LoRA weights + router state.

The resulting checkpoints can be evaluated with:
  - evaluate_olora_t2m_moe.py
  - evaluate_olora_m2t_moe.py
"""

from __future__ import annotations

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

MOTION_AGENT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MOTION_AGENT_PATH))

from continual_learning.olora_multi_adapter.olora_manager import MultiAdapterOLoRAManager
from continual_learning.olora_moe.router import AutoencoderRouter
from continual_learning.utils.data_manager import (
    TASK_NAMES,
    get_task_loader,
    set_split_mode,
)
from models.mllm import MotionLLM
from utils.word_vectorizer import WordVectorizer


def get_logger(out_dir: str, name: str = "OLoRA-MoE"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def _resolve_target_modules(mode: str) -> list[str]:
    mode = (mode or "qv").strip().lower()
    if mode in {"qv", "q_proj,v_proj", "q_proj_v_proj"}:
        return ["q_proj", "v_proj"]
    if mode in {"full", "default"}:
        return [
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
        ]
    if "," in mode:
        return [m.strip() for m in mode.split(",") if m.strip()]
    raise ValueError(f"Unknown --olora-target-modules={mode}")


def parse_args():
    p = argparse.ArgumentParser(description="O-LoRA MoE Training")

    p.add_argument("--training-task", type=str, required=True, choices=["t2m", "m2t"])
    p.add_argument("--device", type=str, default="cuda:0")

    # O-LoRA training knobs
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--epochs-per-task", type=int, default=20)
    p.add_argument("--start-task", type=int, default=0)
    p.add_argument("--train-split", type=str, default="trainval", choices=["train", "trainval"])
    p.add_argument("--lambda-orth", type=float, default=0.5)
    p.add_argument(
        "--olora-target-modules",
        type=str,
        default="qv",
        help="Target modules for O-LoRA per-task adapters: 'qv', 'full', or comma-separated list.",
    )

    # Router training knobs
    p.add_argument("--router-hidden-dim", type=int, default=256)
    p.add_argument("--router-lr", type=float, default=1e-3)
    p.add_argument("--router-epochs", type=int, default=50)
    p.add_argument("--router-batch-size", type=int, default=128)
    p.add_argument(
        "--router-embed-source",
        type=str,
        default=None,
        choices=["text", "motion_prefix"],
        help=(
            "Router embedding source for T2M. "
            "- text: mean pooled prompt token embeddings (legacy) "
            "- motion_prefix: mean pooled embeddings of first N motion tokens (two-stage routing)"
        ),
    )
    p.add_argument(
        "--router-prefix-len",
        type=int,
        default=10,
        help="Prefix length (in motion tokens) when --router-embed-source=motion_prefix.",
    )
    p.add_argument(
        "--router-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std added to routing logits (exploration). 0 disables.",
    )
    p.add_argument("--router-threshold-k", type=float, default=2.0)
    p.add_argument("--router-samples", type=int, default=2048, help="How many samples to collect for router training")

    # Experiment
    p.add_argument("--split-mode", type=str, default="random_80_20", choices=["predefined", "random_80_20"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="motion-agent-olora")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)

    # Backbone / LoRA
    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=64)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--olora-r", type=int, default=None)
    p.add_argument("--olora-alpha", type=int, default=None)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)

    # VQ-VAE config (required by MotionLLM ctor)
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
    p.add_argument("--vq-path", type=str, default=str(PRETRAINED / "ckpt" / "vqvae.pth"))

    PROJECT_ROOT = MOTION_AGENT_PATH.parent
    p.add_argument(
        "--pretrained-path",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"),
    )
    p.add_argument("--nb-joints", type=int, default=22)

    args = p.parse_args()

    args.olora_target_modules = _resolve_target_modules(args.olora_target_modules)
    if args.out_dir is None:
        args.out_dir = str(PROJECT_ROOT / "experiments" / "olora_moe" / args.training_task / "v1")

    if args.olora_r is None:
        args.olora_r = args.lora_r_t2m if args.training_task == "t2m" else args.lora_r_m2t
    if args.olora_alpha is None:
        args.olora_alpha = args.lora_alpha_t2m if args.training_task == "t2m" else args.lora_alpha_m2t

    # Router embedding defaults: only change T2M. M2T stays as-is.
    if args.router_embed_source is None:
        args.router_embed_source = "motion_prefix" if args.training_task == "t2m" else "text"

    return args


@torch.no_grad()
def _extract_router_embedding(
    model: MotionLLM,
    captions: list[str],
    motion_tokens: list[torch.Tensor],
    training_task: str,
    device: torch.device,
    *,
    router_embed_source: str,
    router_prefix_len: int,
) -> torch.Tensor:
    """Compute routing embeddings.

    t2m: mean pooled input-token embeddings of the prompt.
    m2t: mean pooled motion-token embeddings.
    """
    if training_task == "t2m":
        if router_embed_source == "motion_prefix":
            # Use first N motion tokens (in LLM vocab index space already) and mean pool their embeddings.
            # Note: motion_tokens are remapped to LLM vocab ids: nb_text_tokens + 0..nb_code-1
            # (see how motion_tokens are built in the training loop).
            prefix_embs = []
            emb_layer = model.llm.get_input_embeddings()
            n = max(1, int(router_prefix_len))
            for mtok in motion_tokens:
                mtok = mtok.to(device)
                prefix = mtok[:n]
                e = emb_layer(prefix)  # [n,H]
                prefix_embs.append(e.mean(dim=0))
            return torch.stack(prefix_embs, dim=0)

        if router_embed_source != "text":
            raise ValueError(f"Unknown router_embed_source for t2m: {router_embed_source}")

        # Build same text prompt as used in process_batch via MotionLLM.forward
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
        )
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        batch_inputs = []
        for c in captions:
            batch_inputs.append(prompt + instruction + f"### Input:\n{c}\n\nResponse: <Motion>")

        tok = model.tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        emb_layer = model.llm.get_input_embeddings()
        emb = emb_layer(input_ids)  # [B, S, H]
        mask = attention_mask.unsqueeze(-1)  # [B, S, 1]
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    if training_task == "m2t":
        # motion_tokens are already in vocab index space (MotionLLM uses remapped indices)
        emb_layer = model.llm.get_input_embeddings()
        pooled = []
        for mtok in motion_tokens:
            mtok = mtok.to(device)
            e = emb_layer(mtok)  # [L, H]
            pooled.append(e.mean(dim=0))
        return torch.stack(pooled, dim=0)

    raise ValueError(f"Unknown training_task={training_task}")


def _train_router_for_task(
    router: AutoencoderRouter,
    task_id: int,
    embeddings: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    accelerator: Accelerator,
    logger: logging.Logger,
) -> None:
    # Ensure router has this expert
    while router.num_active_experts <= task_id:
        router.add_expert()

    router.freeze_prior_autoencoders()
    ae = router.autoencoders[task_id]
    for p in ae.parameters():
        p.requires_grad = True

    ds = torch.utils.data.TensorDataset(embeddings)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=0.01)

    # Prepare with accelerator (single-process still fine)
    ae_p, opt, dl = accelerator.prepare(ae, opt, dl)
    ae_p.train()
    for ep in range(epochs):
        losses = []
        for (x,) in dl:
            opt.zero_grad()
            loss = ae_p.compute_reconstruction_loss(x, reduction="mean")
            accelerator.backward(loss)
            opt.step()
            losses.append(loss.item())
        if accelerator.is_main_process and ((ep + 1) % max(1, epochs // 5) == 0):
            logger.info(f"[router] task={task_id} epoch={ep+1}/{epochs} recon_loss={float(np.mean(losses)):.6f}")

    accelerator.wait_for_everyone()
    # Update threshold using all embeddings
    um_ae = accelerator.unwrap_model(ae_p)
    um_ae.eval()
    router.autoencoders[task_id] = um_ae
    router.update_threshold_for_current_task(embeddings)


def main():
    args = parse_args()
    set_split_mode(args.split_mode, args.split_seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    args.device = accelerator.device

    exp_dir = args.out_dir
    if accelerator.is_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        logger = get_logger(exp_dir, f"OLoRA-MoE-{args.training_task}")
        args_dump = vars(args).copy()
        args_dump["device"] = str(args_dump["device"])
        logger.info(json.dumps(args_dump, indent=2))
    else:
        logger = logging.getLogger("OLoRA-MoE-Child")
        logger.addHandler(logging.NullHandler())

    if accelerator.is_main_process and not args.no_wandb:
        import wandb

        cfg = vars(args).copy()
        cfg["device"] = str(cfg["device"])
        run_name = args.wandb_name or f"olora_moe_{args.training_task}"
        wandb.init(project=args.wandb_project, name=run_name, config=cfg, tags=["olora-moe", args.training_task])

    if accelerator.is_main_process:
        logger.info("Initializing MotionLLM...")
    model = MotionLLM(args)
    model.training_task = args.training_task

    # Data loader helper expects a vectorizer (even though we don't use its values
    # directly in the CL objective). Keep consistent with other scripts.
    w_vectorizer = WordVectorizer(str(MOTION_AGENT_PATH / str(PRETRAINED / 'glove')), "our_vab")

    # Merge pretrained adapter into base weights
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        if accelerator.is_main_process:
            logger.info(f"Merging pretrained {args.training_task} adapter: {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, args.training_task)
    elif accelerator.is_main_process:
        logger.warning(f"Pretrained not found: {args.pretrained_path}")

    if args.gradient_checkpointing:
        model.llm.base_model.model.gradient_checkpointing_enable()
        model.llm.enable_input_require_grads()
        if accelerator.is_main_process:
            logger.info("Gradient checkpointing enabled")

    # O-LoRA manager (per-task adapters)
    olora_manager = MultiAdapterOLoRAManager(
        model=model.llm,
        lora_r=args.olora_r,
        lora_alpha=args.olora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.olora_target_modules,
    )

    # Router (embedding dim = LLM hidden size)
    router = AutoencoderRouter(
        embed_dim=model.llm.config.hidden_size,
        max_experts=args.num_tasks,
        hidden_dim=args.router_hidden_dim,
        top_k=1,
        routing_noise_std=args.router_noise_std,
        threshold_k=args.router_threshold_k,
    ).to(accelerator.device)

    for task_id in range(args.start_task, args.num_tasks):
        if accelerator.is_main_process:
            logger.info("\n" + "=" * 60)
            logger.info(f"TASK {task_id} ({TASK_NAMES[task_id]})")
            logger.info("=" * 60)

        adapter_name = olora_manager.prepare_task(task_id, model.llm)
        model.adapter_override = adapter_name

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

        # Load task data
        # NOTE: dataset_name uses 't2m' for both directions in this repo.
        train_loader = get_task_loader(
            args.dataname,
            args.train_split,
            args.batch_size,
            w_vectorizer,
            task_id,
            unit_length=2 ** args.down_t,
        )

        model_prepared, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        model_prepared.train()

        # ------------------- Train O-LoRA adapter -------------------
        for epoch in range(args.epochs_per_task):
            batch_losses = []
            t0 = time.time()
            pbar = tqdm(
                train_loader,
                desc=f"Task {task_id} Epoch {epoch+1}/{args.epochs_per_task}",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
            )

            for batch in pbar:
                # Batch format: (word_emb, pos_oh, caption, sent_len, motion, m_length, token, name)
                _, _, caption, _, motion, m_length, _, _ = batch

                # Convert motion to tokens via VQ-VAE
                um = accelerator.unwrap_model(model_prepared)
                motion_tokens = []
                for i in range(motion.shape[0]):
                    m_in = motion[i : i + 1, : m_length[i], :].to(accelerator.device)
                    with torch.no_grad():
                        tok = um.net.encode(m_in).squeeze(0)
                    tok_remapped = torch.from_numpy(um.motion_token_indices[tok.cpu().numpy()]).to(accelerator.device)
                    motion_tokens.append(tok_remapped)

                with accelerator.accumulate(model_prepared):
                    task_loss, gen_acc, _, _ = model_prepared(caption=list(caption), motion=motion_tokens)
                    orth_loss = olora_manager.compute_orthogonality_loss(um.llm)
                    total_loss = task_loss + args.lambda_orth * orth_loss

                    accelerator.backward(total_loss)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_losses.append(total_loss.item())
                if accelerator.is_main_process:
                    pbar.set_postfix({"loss": f"{total_loss.item():.3f}", "acc": f"{gen_acc:.3f}"})

            if accelerator.is_main_process:
                logger.info(
                    f"Task {task_id} Epoch {epoch+1}: loss={float(np.mean(batch_losses)):.4f} time={time.time()-t0:.1f}s"
                )

        accelerator.wait_for_everyone()
        um = accelerator.unwrap_model(model_prepared)
        olora_manager.complete_task(task_id, um.llm)

        # ------------------- Train router autoencoder -------------------
        if accelerator.is_main_process:
            logger.info("Collecting router embeddings...")

        # Collect embeddings from a subset of batches
        collected = []
        um.eval()
        for batch in train_loader:
            _, _, caption, _, motion, m_length, _, _ = batch
            # Build motion tokens
            motion_tokens = []
            for i in range(motion.shape[0]):
                m_in = motion[i : i + 1, : m_length[i], :].to(accelerator.device)
                with torch.no_grad():
                    tok = um.net.encode(m_in).squeeze(0)
                tok_remapped = torch.from_numpy(um.motion_token_indices[tok.cpu().numpy()]).to(accelerator.device)
                motion_tokens.append(tok_remapped)

            emb = _extract_router_embedding(
                model=um,
                captions=list(caption),
                motion_tokens=motion_tokens,
                training_task=args.training_task,
                device=accelerator.device,
                router_embed_source=args.router_embed_source,
                router_prefix_len=args.router_prefix_len,
            )
            collected.append(emb.detach().cpu())
            if sum(x.size(0) for x in collected) >= args.router_samples:
                break
        if not collected:
            if accelerator.is_main_process:
                logger.warning("No embeddings collected for router; skipping router training")
        else:
            emb_all = torch.cat(collected, dim=0).to(accelerator.device)
            if accelerator.is_main_process:
                logger.info(f"Router embeddings: {tuple(emb_all.shape)}")

            _train_router_for_task(
                router=router,
                task_id=task_id,
                embeddings=emb_all,
                lr=args.router_lr,
                epochs=args.router_epochs,
                batch_size=args.router_batch_size,
                accelerator=accelerator,
                logger=logger,
            )

        # ------------------- Save checkpoint -------------------
        if accelerator.is_main_process:
            ckpt_path = os.path.join(exp_dir, f"olora_moe_task_{task_id}_best.pth")
            # Save LoRA weights (same format as MotionLLM.save_model())
            um.save_model(ckpt_path)

            # Save router separately (same directory) to avoid changing base ckpt readers.
            router_path = os.path.join(exp_dir, f"router_task_{task_id}.pth")
            torch.save(
                {
                    "router_state": router.state_dict(),
                    "router_num_active": router.num_active_experts,
                    "router_config": {
                        "embed_dim": router.embed_dim,
                        "max_experts": router.max_experts,
                        "hidden_dim": router.hidden_dim,
                        "top_k": router.top_k,
                        "routing_noise_std": router.routing_noise_std,
                        "threshold_k": router.threshold_k,
                        # Extra metadata for T2M two-stage routing. Safe to ignore for M2T.
                        "router_embed_source": str(getattr(args, "router_embed_source", "text")),
                        "router_prefix_len": int(getattr(args, "router_prefix_len", 10)),
                    },
                },
                router_path,
            )
            logger.info(f"Saved: {ckpt_path}")
            logger.info(f"Saved router: {router_path}")

        accelerator.free_memory()
        del train_loader, optimizer, model_prepared
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process and not args.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
