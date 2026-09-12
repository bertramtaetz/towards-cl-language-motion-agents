"""O-LoRA MoE T2M Evaluation -- task-id free routing.

Builds the continual-learning matrix R[stage, task] like the O-LoRA baselines,
but chooses the adapter at inference time via a router (no task id).

Checkpoints expected:
  <exp-dir>/olora_moe_task_{stage}_best.pth
  <exp-dir>/router_task_{stage}.pth

Routing modes (T2M):
  - two_stage_prefix (DEFAULT):
      1) generate a short prefix (N motion tokens) with base adapter 't2m'
      2) route based on the *generated prefix motion-token embeddings*
      3) continue generation / evaluate token-acc with the routed adapter
  - text_ae (legacy):
      route from caption prompt mean pooled *input token embeddings*

Unseen detection:
  - If unseen (all thresholds reject) -> fallback to base adapter 't2m'

Outputs are compatible with compute_cl_metrics_token_acc.py and
compute_cl_metrics_t2m.py.
"""

from __future__ import annotations

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).parent.resolve()
_CL_DIR = _THIS_DIR.parent
_MOTION_AGENT_ROOT = _CL_DIR.parent.resolve()

sys.path.insert(0, str(_MOTION_AGENT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.mllm import MotionLLM
from models.training_utils import process_batch
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.evaluation import evaluation_test
from utils.word_vectorizer import WordVectorizer
from peft import LoraConfig
from continual_learning.utils.data_manager import (
    TASK_NAMES,
    get_task_loader,
    set_split_mode,
)
from continual_learning.olora_moe.router import AutoencoderRouter
from continual_learning.olora_moe.two_stage_t2m import compute_t2m_prefix_embedding_from_indices
from continual_learning.olora_moe.topk_token_inference import (
    MergedAdapterSpec,
    canonical_merge_name,
    compute_causal_lm_per_sample_loss_from_logits,
    compute_motionllm_token_accuracy_from_logits,
    ensure_merged_lora_adapter,
)


T2M_QUALITY_METRICS = ["fid", "top1", "top2", "top3", "diversity", "mm_dist"]


def _compute_utilization_stats(counts: Dict[str, int], expert_names: List[str]) -> dict:
    """Compute simple load/utilization stats for the given experts.

    Stats are computed only over `expert_names` (i.e., excludes base fallback adapters).
    """
    total = float(sum(int(counts.get(k, 0)) for k in expert_names))
    if total <= 0 or len(expert_names) == 0:
        return {
            "util_total": 0,
            "util_entropy": 0.0,
            "util_cv": 0.0,
            "util_max_frac": 0.0,
            "util_num_nonzero": 0,
        }
    freqs = np.array([float(counts.get(k, 0)) for k in expert_names], dtype=np.float64)
    p = freqs / total
    eps = 1e-12
    ent = float(-(p * np.log(p + eps)).sum())
    ent_norm = float(ent / np.log(len(expert_names) + eps)) if len(expert_names) > 1 else 0.0

    mean = float(freqs.mean())
    std = float(freqs.std())
    cv = float(std / (mean + eps))

    return {
        "util_total": int(total),
        "util_entropy": ent_norm,
        "util_cv": cv,
        "util_max_frac": float(p.max()) if p.size else 0.0,
        "util_num_nonzero": int((freqs > 0).sum()),
    }


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


def load_olora_checkpoint(checkpoint_path: str, args, device: torch.device) -> MotionLLM:
    """Load MotionLLM with per-task O-LoRA adapters from a checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    task_adapters = set()
    for key in ckpt.keys():
        for part in key.split("."):
            if part.startswith("task_") and part[5:].isdigit():
                task_adapters.add(part)
    task_adapters = sorted(task_adapters)
    print(f"  Found task adapters: {task_adapters}")

    model = MotionLLM(args).to(device)

    # Merge pretrained t2m then remove adapter so we can add task adapters
    if getattr(args, "pretrained_path", None) and os.path.exists(args.pretrained_path):
        print(f"  Merging pretrained t2m adapter from {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, "t2m")
        model.llm = model.llm.merge_and_unload()

    if task_adapters:
        from peft import get_peft_model as gpm

        task_lora_config = LoraConfig(
            r=args.olora_r,
            lora_alpha=args.olora_alpha,
            target_modules=_resolve_target_modules(args.olora_target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        first = task_adapters[0]
        model.llm = gpm(model.llm, task_lora_config, adapter_name=first)
        for an in task_adapters[1:]:
            model.llm.add_adapter(an, task_lora_config)
        print(f"  Added adapters: {task_adapters}")

    loaded = 0
    for name, param in model.llm.named_parameters():
        if name in ckpt:
            param.data = ckpt[name].to(device, dtype=param.dtype)
            loaded += 1

    if "embeddings" in ckpt:
        model.llm.get_input_embeddings().weight.data[model.nb_text_tokens :] = ckpt["embeddings"].to(device)
        print(f"  Loaded embeddings: {ckpt['embeddings'].shape}")
    if "lm_head" in ckpt:
        model.llm.lm_head.weight.data[model.nb_text_tokens :] = ckpt["lm_head"].to(device)
        print(f"  Loaded lm_head: {ckpt['lm_head'].shape}")

    print(f"  Loaded {loaded} LoRA parameters")
    model._task_adapters = task_adapters
    model.eval()
    return model


def load_router(router_ckpt_path: str, device: torch.device) -> AutoencoderRouter:
    d = torch.load(router_ckpt_path, map_location="cpu")
    cfg = d.get("router_config") or {}
    router = AutoencoderRouter(
        embed_dim=int(cfg.get("embed_dim", 2304)),
        max_experts=int(cfg.get("max_experts", 5)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        top_k=int(cfg.get("top_k", 1)),
        routing_noise_std=float(cfg.get("routing_noise_std", 0.0)),
        threshold_k=float(cfg.get("threshold_k", 2.0)),
    ).to(device)
    router.load_state_dict(d["router_state"])
    router.set_num_experts(int(d.get("router_num_active", router.num_active_experts)))
    router.eval()
    return router


@torch.no_grad()
def compute_text_router_embedding(model: MotionLLM, captions: List[str], device: torch.device) -> torch.Tensor:
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    batch_inputs = [prompt + instruction + f"### Input:\n{c}\n\nResponse: <Motion>" for c in captions]

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
    mask = attention_mask.unsqueeze(-1)
    pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


@torch.no_grad()
def generate_motion_prefix_tokens(
    *,
    model: MotionLLM,
    captions: List[str],
    device: torch.device,
    prefix_len: int,
    base_adapter: str = "t2m",
) -> List[torch.Tensor]:
    """Generate a short motion-token prefix for each caption using `base_adapter`.

    Returns a list of 1D tensors with motion-token indices in [0..nb_code-1]
    (length <= prefix_len per sample).

    Implementation detail:
    We use `transformers.generate()` with `max_new_tokens=prefix_len` and parse
    `outputs.scores` while restricting logits to the motion-token region.
    """

    if prefix_len <= 0:
        prefix_len = 1

    model.llm.set_adapter(base_adapter)
    model.llm.eval()

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    batch_inputs = [prompt + instruction + f"### Input:\n{c}\n\nResponse: <Motion>" for c in captions]

    original_padding_side = model.tokenizer.padding_side
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    tok = model.tokenizer(
        batch_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    model.tokenizer.padding_side = original_padding_side

    # Motion token region in vocab:
    motion_start = int(model.nb_text_tokens)
    motion_end = int(model.nb_text_tokens + model.args.nb_code + 2)

    outputs = model.llm.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(prefix_len),
        num_beams=1,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # If scores are missing for some reason, fall back to empty prefixes.
    if not getattr(outputs, "scores", None):
        return [torch.empty(0, dtype=torch.long) for _ in captions]

    # scores: list[len<=prefix_len] of [B,V]
    scores = torch.stack(list(outputs.scores), dim=0)  # [T,B,V]
    motion_logits = scores[:, :, motion_start:motion_end]  # [T,B,nb_code+2]

    # Avoid generating <Motion> (index 0 in motion space) after the prompt.
    motion_logits[:, :, 0] = -1e9

    motion_space = torch.argmax(motion_logits, dim=-1)  # [T,B] in motion-space indices

    prefixes: List[torch.Tensor] = []
    for b in range(motion_space.size(1)):
        tokens = []
        for t in range(motion_space.size(0)):
            ms = int(motion_space[t, b].item())
            if ms == 1:  # </Motion>
                break
            if ms >= 2:
                tokens.append(ms - 2)
        if not tokens:
            prefixes.append(torch.empty(0, dtype=torch.long))
        else:
            prefixes.append(torch.tensor(tokens[: int(prefix_len)], dtype=torch.long))
    return prefixes


@torch.no_grad()
def generate_t2m_two_stage_single(
    *,
    model: MotionLLM,
    router: AutoencoderRouter,
    caption: str,
    device: torch.device,
    prefix_len: int,
    base_adapter: str = "t2m",
    max_motion_len: int = 200,
) -> torch.Tensor:
    """Two-stage T2M generation for a single caption.

    Stage 1: generate `prefix_len` motion tokens with `base_adapter`.
    Stage 2: route on generated prefix embedding and continue with routed adapter.

    Returns motion-token indices in [0..nb_code-1].
    """

    prefix_len = max(1, int(prefix_len))
    max_motion_len = int(max_motion_len)

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    full_input = prompt + instruction + f"### Input:\n{caption}\n\nResponse: <Motion>"

    input_ids = model.tokenizer.encode(full_input, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    motion_start = int(model.nb_text_tokens)
    motion_end = int(model.nb_text_tokens + model.args.nb_code + 2)

    generated_motion: List[int] = []
    routed_adapter: str | None = None

    model.llm.eval()
    router.eval()
    with torch.no_grad():
        for step in range(max_motion_len):
            # Choose adapter
            if routed_adapter is None:
                model.llm.set_adapter(base_adapter)
            else:
                model.llm.set_adapter(routed_adapter)

            out = model.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            next_logits = out.logits[:, -1, motion_start:motion_end]  # [1, nb_code+2]
            next_logits[:, 0] = -1e9  # prevent <Motion>
            ms = int(torch.argmax(next_logits, dim=-1).item())

            # Append generated token to context
            next_token_id = torch.tensor([[motion_start + ms]], device=device, dtype=torch.long)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

            if ms == 1:
                break
            if ms >= 2:
                generated_motion.append(ms - 2)

            # Route right after we have the prefix.
            if routed_adapter is None and len(generated_motion) >= prefix_len:
                prefix_tokens = torch.tensor(generated_motion[:prefix_len], dtype=torch.long, device=device)
                emb = compute_t2m_prefix_embedding_from_indices(model, prefix_tokens, device=device).unsqueeze(0)
                full_logits, all_losses = router.compute_full_logits_and_losses(emb, add_noise=True)
                topk_idx = torch.topk(full_logits, k=1, dim=-1)[1]
                unseen = router.detect_unseen_from_losses(all_losses)
                if bool(unseen[0].item()):
                    routed_adapter = base_adapter
                else:
                    routed_adapter = f"task_{int(topk_idx[0, 0].item())}"

    if not generated_motion:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.tensor(generated_motion, dtype=torch.long, device=device)


@torch.no_grad()
def generate_t2m_two_stage_single_mixture(
    *,
    model: MotionLLM,
    router: AutoencoderRouter,
    caption: str,
    device: torch.device,
    prefix_len: int,
    router_top_k: int,
    base_adapter: str = "t2m",
    max_motion_len: int = 200,
) -> torch.Tensor:
    """Two-stage T2M generation with Top-K *mixture* decoding (greedy).

    This is an inference-time ensemble over the router's Top-K experts.
    Unlike reranking, it does not require ground truth.

    Implementation notes:
    - Routing is computed once from the generated prefix and then kept fixed.
    - Decoding is greedy on the *mixture* next-token logits.
    - Uses per-expert KV caches so runtime scales ~O(K * T) rather than O(K * T^2).

    Returns motion-token indices in [0..nb_code-1].
    """

    prefix_len = max(1, int(prefix_len))
    max_motion_len = int(max_motion_len)
    router_top_k = max(1, int(router_top_k))

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    full_input = prompt + instruction + f"### Input:\n{caption}\n\nResponse: <Motion>"

    input_ids = model.tokenizer.encode(full_input, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    motion_start = int(model.nb_text_tokens)
    motion_end = int(model.nb_text_tokens + model.args.nb_code + 2)

    model.llm.eval()
    router.eval()

    # ------------------------------------------------------------
    # Stage 1: generate prefix with base adapter (small, so simple loop)
    # ------------------------------------------------------------
    generated_motion: List[int] = []
    ended_early = False
    for _ in range(prefix_len):
        model.llm.set_adapter(base_adapter)
        out = model.llm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        next_logits = out.logits[:, -1, motion_start:motion_end].float()  # [1, nb_code+2]
        next_logits[:, 0] = -1e9  # prevent <Motion>
        ms = int(torch.argmax(next_logits, dim=-1).item())

        next_token_id = torch.tensor([[motion_start + ms]], device=device, dtype=torch.long)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

        if ms == 1:  # </Motion>
            ended_early = True
            break
        if ms >= 2:
            generated_motion.append(ms - 2)

    if not generated_motion:
        # Keep consistent with existing behavior.
        return torch.zeros(1, dtype=torch.long, device=device)

    # If the model already produced </Motion> during the prefix stage, return.
    if ended_early:
        return torch.tensor(generated_motion, dtype=torch.long, device=device)

    # ------------------------------------------------------------
    # Route from prefix embedding
    # ------------------------------------------------------------
    prefix_tokens = torch.tensor(generated_motion[:prefix_len], dtype=torch.long, device=device)
    emb = compute_t2m_prefix_embedding_from_indices(model, prefix_tokens, device=device).unsqueeze(0)
    full_logits, all_losses = router.compute_full_logits_and_losses(emb, add_noise=True)
    unseen = router.detect_unseen_from_losses(all_losses)

    if bool(unseen[0].item()) or router.num_active_experts <= 0:
        expert_names = [base_adapter]
        weights = torch.ones(1, device=device, dtype=torch.float32)
    else:
        k = min(router_top_k, int(router.num_active_experts))
        topk_logits, topk_idx = torch.topk(full_logits, k=k, dim=-1)
        weights = torch.softmax(topk_logits[0], dim=-1).to(device=device, dtype=torch.float32)  # [K]
        expert_names = [f"task_{int(i.item())}" for i in topk_idx[0]]

    # ------------------------------------------------------------
    # Stage 2: greedy mixture decoding with per-expert KV caches
    # ------------------------------------------------------------
    # Initialize cache for each expert by running the full current context once.
    per_expert_cache = {}
    per_expert_next_logits = {}
    for an in expert_names:
        model.llm.set_adapter(an)
        out = model.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
        )
        per_expert_cache[an] = out.past_key_values
        per_expert_next_logits[an] = out.logits[:, -1, motion_start:motion_end].float().squeeze(0)  # [nb_code+2]

    for _ in range(max_motion_len - len(generated_motion)):
        # Mixture over next-token logits in motion space.
        mix = torch.zeros((motion_end - motion_start,), device=device, dtype=torch.float32)
        for w, an in zip(weights, expert_names):
            mix = mix + float(w.item()) * per_expert_next_logits[an]
        mix[0] = -1e9  # prevent <Motion>

        ms = int(torch.argmax(mix, dim=-1).item())
        next_token_id = torch.tensor([[motion_start + ms]], device=device, dtype=torch.long)

        # Append to context
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

        if ms == 1:
            break
        if ms >= 2:
            generated_motion.append(ms - 2)

        # Advance each expert one step with its own cache.
        for an in expert_names:
            model.llm.set_adapter(an)
            out = model.llm(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=per_expert_cache[an],
                return_dict=True,
                use_cache=True,
            )
            per_expert_cache[an] = out.past_key_values
            per_expert_next_logits[an] = out.logits[:, -1, motion_start:motion_end].float().squeeze(0)

    if not generated_motion:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.tensor(generated_motion, dtype=torch.long, device=device)


@torch.no_grad()
def generate_t2m_text_ae_single_mixture(
    *,
    model: MotionLLM,
    router: AutoencoderRouter,
    caption: str,
    device: torch.device,
    router_top_k: int,
    base_adapter: str = "t2m",
    max_motion_len: int = 200,
) -> torch.Tensor:
    """Text-AE T2M generation with Top-K *mixture* decoding (greedy).

    This is the text-based routing variant (legacy `routing_mode=text_ae`).
    We compute the router embedding from the caption prompt once and keep the
    Top-K mixture weights fixed for the entire generation.

    Notes / limitations:
    - Greedy decoding only (no beam search / sampling).
    - Router weights are fixed (no re-routing mid-generation).
    - Uses per-expert KV caches so runtime scales ~O(K * T).

    Returns motion-token indices in [0..nb_code-1].
    """

    max_motion_len = int(max_motion_len)
    router_top_k = max(1, int(router_top_k))

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    full_input = prompt + instruction + f"### Input:\n{caption}\n\nResponse: <Motion>"

    input_ids = model.tokenizer.encode(full_input, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    motion_start = int(model.nb_text_tokens)
    motion_end = int(model.nb_text_tokens + model.args.nb_code + 2)

    model.llm.eval()
    router.eval()

    # ------------------------------------------------------------
    # Route from text prompt embedding
    # ------------------------------------------------------------
    emb = compute_text_router_embedding(model, [caption], device=device)  # [1,H]
    full_logits, all_losses = router.compute_full_logits_and_losses(emb, add_noise=True)
    unseen = router.detect_unseen_from_losses(all_losses)

    if bool(unseen[0].item()) or router.num_active_experts <= 0:
        expert_names = [base_adapter]
        weights = torch.ones(1, device=device, dtype=torch.float32)
    else:
        k = min(router_top_k, int(router.num_active_experts))
        topk_logits, topk_idx = torch.topk(full_logits, k=k, dim=-1)
        weights = torch.softmax(topk_logits[0], dim=-1).to(device=device, dtype=torch.float32)  # [K]
        expert_names = [f"task_{int(i.item())}" for i in topk_idx[0]]

    # ------------------------------------------------------------
    # Greedy mixture decoding with per-expert KV caches
    # ------------------------------------------------------------
    generated_motion: List[int] = []

    per_expert_cache = {}
    per_expert_next_logits = {}
    for an in expert_names:
        model.llm.set_adapter(an)
        out = model.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
        )
        per_expert_cache[an] = out.past_key_values
        per_expert_next_logits[an] = out.logits[:, -1, motion_start:motion_end].float().squeeze(0)  # [nb_code+2]

    for _ in range(max_motion_len):
        mix = torch.zeros((motion_end - motion_start,), device=device, dtype=torch.float32)
        for w, an in zip(weights, expert_names):
            mix = mix + float(w.item()) * per_expert_next_logits[an]
        mix[0] = -1e9  # prevent <Motion>

        ms = int(torch.argmax(mix, dim=-1).item())
        next_token_id = torch.tensor([[motion_start + ms]], device=device, dtype=torch.long)

        # Append to context
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

        if ms == 1:  # </Motion>
            break
        if ms >= 2:
            generated_motion.append(ms - 2)

        # Advance each expert one step with its own cache.
        for an in expert_names:
            model.llm.set_adapter(an)
            out = model.llm(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=per_expert_cache[an],
                return_dict=True,
                use_cache=True,
            )
            per_expert_cache[an] = out.past_key_values
            per_expert_next_logits[an] = out.logits[:, -1, motion_start:motion_end].float().squeeze(0)

    if not generated_motion:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.tensor(generated_motion, dtype=torch.long, device=device)


def _route_adapters(
    model: MotionLLM,
    router: AutoencoderRouter,
    captions: List[str],
    device: torch.device,
    base_adapter: str = "t2m",
) -> Tuple[List[str], Dict[str, float]]:
    emb = compute_text_router_embedding(model, captions, device)
    topk_w, topk_idx, _ = router(emb, return_all_losses=False)
    unseen = router.detect_unseen_task(emb)

    chosen_idx = topk_idx[:, 0].tolist()
    adapter_names = []
    for i, idx in enumerate(chosen_idx):
        if bool(unseen[i].item()):
            adapter_names.append(base_adapter)
        else:
            adapter_names.append(f"task_{idx}")

    stats = {
        "unseen_rate": float(unseen.float().mean().item()),
    }
    return adapter_names, stats


def compute_token_accuracy_taskid_free(
    model: MotionLLM,
    router: AutoencoderRouter,
    loader,
    device: torch.device,
    true_task_id: int,
    base_adapter: str = "t2m",
    routing_mode: str = "two_stage_prefix",
    prefix_len: int = 10,
    router_top_k: int = 1,
    router_infer_mode: str = "top1",
) -> Tuple[float, float, dict]:
    """Token accuracy but adapter is selected via router per sample."""
    model.eval()
    router.eval()

    all_losses: List[float] = []
    all_accs: List[float] = []

    # Router diagnostics
    routed_counter = Counter()
    confusion_counter = Counter()
    unseen_counter = 0

    # top-k routing quality
    eligible_for_task_acc = int(true_task_id) < int(router.num_active_experts)
    router_count = 0
    router_top1_correct = 0
    router_top2_correct = 0
    fallback_on_seen = 0

    for batch in tqdm(loader, desc="token acc (routed)", leave=False):
        _, _, caption, _, motion, m_length, _, _ = batch
        captions = list(caption)

        # Build motion tokens (as in baselines)
        motion_tokens = []
        for i in range(motion.size(0)):
            m = motion[i : i + 1, : m_length[i], :].to(device)
            tok = model.net.encode(m).squeeze(0)
            tok_remapped = torch.from_numpy(model.motion_token_indices[tok.cpu().numpy()]).to(device)
            motion_tokens.append(tok_remapped)

        # Route (compute logits once so top-1/top-2 share the same noise sample)
        if routing_mode == "text_ae":
            emb = compute_text_router_embedding(model, captions, device)
        elif routing_mode == "two_stage_prefix":
            prefixes = generate_motion_prefix_tokens(
                model=model,
                captions=captions,
                device=device,
                prefix_len=int(prefix_len),
                base_adapter=base_adapter,
            )
            emb_list = []
            for p in prefixes:
                # Ensure non-empty to avoid NaNs
                if p.numel() == 0:
                    p = torch.zeros(1, dtype=torch.long)
                emb_list.append(compute_t2m_prefix_embedding_from_indices(model, p.to(device), device=device))
            emb = torch.stack(emb_list, dim=0)
        else:
            raise ValueError(f"Unknown routing_mode={routing_mode}")

        # NOTE: avoid shadowing the Python list `all_losses` used for token-level losses.
        full_logits, router_losses = router.compute_full_logits_and_losses(emb, add_noise=True)

        # Get Top-K indices for inference + Top-2 for diagnostics.
        k_infer = max(1, min(int(router_top_k), int(router.num_active_experts)))
        k_diag = 2 if router.num_active_experts >= 2 else 1
        k_top = max(k_infer, k_diag)

        topk_logits, topk_idx = torch.topk(full_logits, k=min(k_top, router.num_active_experts), dim=-1)
        topk_weights = torch.softmax(topk_logits[:, :k_infer], dim=-1)
        unseen = router.detect_unseen_from_losses(router_losses)

        # Router diagnostics
        for i in range(len(captions)):
            if bool(unseen[i].item()):
                unseen_counter += 1
                routed_counter[base_adapter] += 1
                confusion_counter[base_adapter] += 1
                if eligible_for_task_acc:
                    fallback_on_seen += 1
            else:
                pred_idx = int(topk_idx[i, 0].item())
                pred_name = f"task_{pred_idx}"
                routed_counter[pred_name] += 1
                confusion_counter[pred_name] += 1

                if eligible_for_task_acc:
                    router_count += 1
                    if pred_idx == int(true_task_id):
                        router_top1_correct += 1
                    if topk_idx.size(1) >= 2:
                        in_top2 = int(true_task_id) in {int(topk_idx[i, 0].item()), int(topk_idx[i, 1].item())}
                    else:
                        in_top2 = pred_idx == int(true_task_id)
                    if in_top2:
                        router_top2_correct += 1

        # Build full batch once; all modes use the same tokenization.
        input_ids, targets, attention_mask = process_batch(
            tokenizer=model.tokenizer,
            batch_of_captions=captions,
            max_tgt_len=200,
            batch_of_motions=motion_tokens,
            training_task="t2m",
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        mode = (router_infer_mode or "top1").strip().lower()
        if mode == "top1" or k_infer == 1:
            # Original behavior: pick a single adapter per sample and group for efficiency.
            groups: Dict[str, List[int]] = defaultdict(list)
            for i in range(len(captions)):
                if bool(unseen[i].item()):
                    groups[base_adapter].append(i)
                else:
                    groups[f"task_{int(topk_idx[i, 0].item())}"].append(i)

            for adapter_name, idxs in groups.items():
                model.llm.set_adapter(adapter_name)
                with torch.no_grad():
                    out = model.llm(
                        input_ids=input_ids[idxs],
                        attention_mask=attention_mask[idxs],
                        return_dict=True,
                        labels=targets[idxs],
                    )
                all_losses.append(float(out.loss.item()))
                acc = compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[idxs]).item()
                all_accs.append(float(acc))

        elif mode == "mixture":
            # Mode A: mixture of logits across Top-K experts.
            bsz = input_ids.size(0)

            # Memory note:
            # Mixture mode materializes a (B,S,V) logits tensor. Keeping it in
            # bf16 avoids a large fp32 allocation and reduces OOM risk.
            mix_dtype = torch.bfloat16

            # NOTE: vocab size can differ from config.vocab_size if embeddings were resized.
            vocab_size = int(model.llm.get_output_embeddings().weight.size(0))
            mixture_logits = torch.zeros(
                (bsz, input_ids.size(1), vocab_size),
                device=device,
                dtype=mix_dtype,
            )

            for j in range(k_infer):
                # Evaluate only samples that are not unseen.
                idxs = [i for i in range(bsz) if not bool(unseen[i].item())]
                if not idxs:
                    break
                adapter_names = [f"task_{int(topk_idx[i, j].item())}" for i in idxs]
                unique_adapters = sorted(set(adapter_names))
                for an in unique_adapters:
                    sel = [idxs[ii] for ii, a in enumerate(adapter_names) if a == an]
                    w = topk_weights[sel, j].view(-1, 1, 1).to(dtype=mix_dtype)
                    model.llm.set_adapter(an)
                    with torch.no_grad():
                        out = model.llm(
                            input_ids=input_ids[sel],
                            attention_mask=attention_mask[sel],
                            return_dict=True,
                        )
                    mixture_logits[sel] = mixture_logits[sel] + out.logits.to(dtype=mix_dtype) * w

            # Unseen fall back to base adapter.
            unseen_idxs = [i for i in range(input_ids.size(0)) if bool(unseen[i].item())]
            if unseen_idxs:
                model.llm.set_adapter(base_adapter)
                with torch.no_grad():
                    out = model.llm(
                        input_ids=input_ids[unseen_idxs],
                        attention_mask=attention_mask[unseen_idxs],
                        return_dict=True,
                    )
                mixture_logits[unseen_idxs] = out.logits.to(dtype=mix_dtype)

            # Compute per-sample loss and overall acc from mixture logits.
            per_sample_loss = compute_causal_lm_per_sample_loss_from_logits(logits=mixture_logits, labels=targets)
            all_losses.append(float(per_sample_loss.mean().item()))
            all_accs.append(float(compute_motionllm_token_accuracy_from_logits(logits=mixture_logits, labels=targets).item()))

        elif mode == "rerank":
            # Mode B: pick the expert (from Top-K) with lowest teacher-forced NLL per sample.
            bsz = input_ids.size(0)
            per_sample_best_loss = torch.full((bsz,), float("inf"), device=device)
            per_sample_best_adapter = [base_adapter for _ in range(bsz)]

            # Unseen always uses base adapter.
            seen_idxs = [i for i in range(bsz) if not bool(unseen[i].item())]
            if seen_idxs:
                # Evaluate each j on the seen subset.
                for j in range(k_infer):
                    adapter_names = [f"task_{int(topk_idx[i, j].item())}" for i in seen_idxs]
                    unique_adapters = sorted(set(adapter_names))
                    for an in unique_adapters:
                        sel = [seen_idxs[ii] for ii, a in enumerate(adapter_names) if a == an]
                        model.llm.set_adapter(an)
                        with torch.no_grad():
                            out = model.llm(
                                input_ids=input_ids[sel],
                                attention_mask=attention_mask[sel],
                                return_dict=True,
                            )
                        losses = compute_causal_lm_per_sample_loss_from_logits(logits=out.logits, labels=targets[sel])
                        for offset, sidx in enumerate(sel):
                            l = float(losses[offset].item())
                            if l < float(per_sample_best_loss[sidx].item()):
                                per_sample_best_loss[sidx] = losses[offset]
                                per_sample_best_adapter[sidx] = an

            # Now run one forward per selected adapter to compute token acc and overall loss.
            groups: Dict[str, List[int]] = defaultdict(list)
            for i, an in enumerate(per_sample_best_adapter):
                groups[an].append(i)
            batch_losses = []
            batch_accs = []
            for an, idxs in groups.items():
                model.llm.set_adapter(an)
                with torch.no_grad():
                    out = model.llm(
                        input_ids=input_ids[idxs],
                        attention_mask=attention_mask[idxs],
                        return_dict=True,
                        labels=targets[idxs],
                    )
                batch_losses.append(out.loss.item())
                batch_accs.append(compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[idxs]).item())
            all_losses.append(float(torch.tensor(batch_losses).mean().item()) if batch_losses else 0.0)
            all_accs.append(float(torch.tensor(batch_accs).mean().item()) if batch_accs else 0.0)

        elif mode == "merge":
            # Mode C: create a cached merged adapter per unique Top-K tuple.
            bsz = input_ids.size(0)
            merge_prefix = "_merged_topk"
            groups: Dict[str, List[int]] = defaultdict(list)

            # Use base adapter for unseen.
            for i in range(bsz):
                if bool(unseen[i].item()):
                    groups[base_adapter].append(i)
                    continue
                ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                merged_name = canonical_merge_name(merge_prefix, ex)
                src_names = tuple(f"task_{e}" for e in ex)
                ensure_merged_lora_adapter(
                    peft_model=model.llm,
                    spec=MergedAdapterSpec(merged_adapter_name=merged_name, source_adapter_names=src_names),
                    base_lora_config=model.llm.peft_config[src_names[0]],
                )
                groups[merged_name].append(i)

            for an, idxs in groups.items():
                model.llm.set_adapter(an)
                with torch.no_grad():
                    out = model.llm(
                        input_ids=input_ids[idxs],
                        attention_mask=attention_mask[idxs],
                        return_dict=True,
                        labels=targets[idxs],
                    )
                all_losses.append(float(out.loss.item()))
                all_accs.append(float(compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[idxs]).item()))

        else:
            raise ValueError(f"Unknown --router-infer-mode={router_infer_mode}")

    diag = {
        "unseen_count": int(unseen_counter),
        "unseen_rate": float(unseen_counter / max(1, sum(routed_counter.values()))),
        "routed_counts": dict(routed_counter),
        "confusion_counts": dict(confusion_counter),
        "router_num_active_experts": int(router.num_active_experts),
        "router_true_task_id": int(true_task_id),
        "router_top1": (float(router_top1_correct) / float(router_count) if eligible_for_task_acc and router_count else None),
        "router_top2": (float(router_top2_correct) / float(router_count) if eligible_for_task_acc and router_count else None),
        "fallback_rate_on_seen": (float(fallback_on_seen) / float(sum(routed_counter.values())) if eligible_for_task_acc else None),
    }

    # Utilization stats (experts only)
    expert_names = [f"task_{i}" for i in range(int(router.num_active_experts))]
    diag.update(_compute_utilization_stats(dict(routed_counter), expert_names))
    return float(np.mean(all_losses)), float(np.mean(all_accs)), diag


class RoutedGeneratorWrapper:
    """Adapter-routing wrapper for evaluation_test() motion quality metrics."""

    def __init__(self, model: MotionLLM, router: AutoencoderRouter, device: torch.device):
        self._model = model
        self._router = router
        self._device = device

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        # evaluation_test() calls generate_batch(list(caption))
        results = []
        self._model.eval()
        self._router.eval()
        with torch.no_grad():
            emb = compute_text_router_embedding(self._model, captions, self._device)
            topk_w, topk_idx, _ = self._router(emb)
            unseen = self._router.detect_unseen_task(emb)

            for i, cap in enumerate(captions):
                if bool(unseen[i].item()):
                    self._model.adapter_override = "t2m"
                else:
                    self._model.adapter_override = f"task_{int(topk_idx[i, 0].item())}"
                results.append(self._model.generate(cap))
            self._model.adapter_override = None
        return results


class TwoStageRoutedGeneratorWrapper:
    """Two-stage adapter-routing wrapper for evaluation_test() motion quality metrics."""

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        prefix_len: int = 10,
        base_adapter: str = "t2m",
        max_motion_len: int = 200,
    ):
        self._model = model
        self._router = router
        self._device = device
        self._prefix_len = int(prefix_len)
        self._base_adapter = base_adapter
        self._max_motion_len = int(max_motion_len)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        # evaluation_test() calls generate_batch(list(caption))
        results = []
        self._model.eval()
        self._router.eval()
        for cap in captions:
            mot = generate_t2m_two_stage_single(
                model=self._model,
                router=self._router,
                caption=cap,
                device=self._device,
                prefix_len=self._prefix_len,
                base_adapter=self._base_adapter,
                max_motion_len=self._max_motion_len,
            )
            results.append(mot)
        return results


class TwoStageMixtureGeneratorWrapper:
    """Two-stage mixture adapter-routing wrapper for evaluation_test() motion quality metrics."""

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        prefix_len: int = 10,
        router_top_k: int = 2,
        base_adapter: str = "t2m",
        max_motion_len: int = 200,
    ):
        self._model = model
        self._router = router
        self._device = device
        self._prefix_len = int(prefix_len)
        self._router_top_k = int(router_top_k)
        self._base_adapter = base_adapter
        self._max_motion_len = int(max_motion_len)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        results = []
        self._model.eval()
        self._router.eval()
        for cap in captions:
            mot = generate_t2m_two_stage_single_mixture(
                model=self._model,
                router=self._router,
                caption=cap,
                device=self._device,
                prefix_len=self._prefix_len,
                router_top_k=self._router_top_k,
                base_adapter=self._base_adapter,
                max_motion_len=min(int(max_length), int(self._max_motion_len)),
            )
            results.append(mot)
        return results


class TextAEMixtureGeneratorWrapper:
    """Text-AE mixture adapter-routing wrapper for evaluation_test() motion quality metrics."""

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        router_top_k: int = 2,
        base_adapter: str = "t2m",
        max_motion_len: int = 200,
    ):
        self._model = model
        self._router = router
        self._device = device
        self._router_top_k = int(router_top_k)
        self._base_adapter = base_adapter
        self._max_motion_len = int(max_motion_len)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        results = []
        self._model.eval()
        self._router.eval()
        for cap in captions:
            mot = generate_t2m_text_ae_single_mixture(
                model=self._model,
                router=self._router,
                caption=cap,
                device=self._device,
                router_top_k=self._router_top_k,
                base_adapter=self._base_adapter,
                max_motion_len=min(int(max_length), int(self._max_motion_len)),
            )
            results.append(mot)
        return results


def compute_motion_quality_taskid_free(
    model: MotionLLM,
    router: AutoencoderRouter,
    loader,
    eval_wrapper,
    temp_dir: str,
    device: torch.device,
    routing_mode: str = "two_stage_prefix",
    prefix_len: int = 10,
    router_top_k: int = 1,
    router_infer_mode: str = "top1",
) -> dict:
    mode = (router_infer_mode or "top1").strip().lower()
    use_mixture = (mode == "mixture") and (int(router_top_k) > 1)

    if routing_mode == "text_ae":
        if use_mixture:
            print(f"    Using text_ae mixture decoding (top_k={int(router_top_k)})")
            wrapped = TextAEMixtureGeneratorWrapper(
                model,
                router,
                device,
                router_top_k=int(router_top_k),
            )
        else:
            wrapped = RoutedGeneratorWrapper(model, router, device)
    elif routing_mode == "two_stage_prefix":
        if use_mixture:
            wrapped = TwoStageMixtureGeneratorWrapper(
                model,
                router,
                device,
                prefix_len=prefix_len,
                router_top_k=int(router_top_k),
            )
        else:
            wrapped = TwoStageRoutedGeneratorWrapper(model, router, device, prefix_len=prefix_len)
    else:
        raise ValueError(f"Unknown routing_mode={routing_mode}")
    fid, div, top1, top2, top3, mm_dist = evaluation_test(
        temp_dir,
        loader,
        wrapped,
        eval_wrapper=eval_wrapper,
        draw=False,
        savenpy=False,
    )
    return {
        "fid": float(fid),
        "top1": float(top1),
        "top2": float(top2),
        "top3": float(top3),
        "diversity": float(div),
        "mm_dist": float(mm_dist),
    }


def evaluate_task(
    model: MotionLLM,
    router: AutoencoderRouter,
    task_id: int,
    split: str,
    w_vectorizer,
    args,
    eval_wrapper,
    temp_dir: str | None,
    with_generation: bool,
) -> dict:
    device = torch.device(args.device)
    task_name = TASK_NAMES[task_id]

    print(f"  Task {task_id} ({task_name})")
    loader = get_task_loader(
        dataset_name="t2m",
        split=split,
        batch_size=args.batch_size,
        w_vectorizer=w_vectorizer,
        task_id=task_id,
        num_workers=0,
        unit_length=2 ** args.down_t,
    )
    n_samples = len(loader.dataset)
    print(f"    {n_samples} samples")

    loss, acc, diag = compute_token_accuracy_taskid_free(
        model,
        router,
        loader,
        device,
        true_task_id=task_id,
        base_adapter="t2m",
        routing_mode=args.routing_mode,
        prefix_len=args.prefix_len,
        router_top_k=getattr(args, "router_top_k", 1),
        router_infer_mode=getattr(args, "router_infer_mode", "top1"),
    )
    print(f"    Loss={loss:.4f}, Accuracy={acc:.4f} ({acc*100:.2f}%)")

    result = {
        "task_id": task_id,
        "task_name": task_name,
        "adapter_used": "router(top1)+fallback",
        "n_samples": n_samples,
        "loss": loss,
        "accuracy": acc,
        "router": diag,
    }

    if with_generation and eval_wrapper is not None and temp_dir is not None:
        print("    Generating motions (router)...")
        quality = compute_motion_quality_taskid_free(
            model,
            router,
            loader,
            eval_wrapper,
            temp_dir,
            device,
            routing_mode=args.routing_mode,
            prefix_len=args.prefix_len,
            router_top_k=getattr(args, "router_top_k", 1),
            router_infer_mode=getattr(args, "router_infer_mode", "top1"),
        )
        result.update(quality)
        print(f"    FID={quality['fid']:.4f}, Top1={quality['top1']:.4f}, Div={quality['diversity']:.4f}")

    return result


def evaluate_stage(
    stage: int,
    ckpt_path: Path,
    router_path: Path,
    exp_dir: Path,
    split: str,
    w_vectorizer,
    args,
    eval_wrapper,
    temp_dir: str | None,
    with_generation: bool,
) -> dict:
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"STAGE {stage} -- {ckpt_path.name}")
    print(f"Router: {router_path.name}")
    print(f"{'='*70}")

    model = load_olora_checkpoint(str(ckpt_path), args, device)
    router = load_router(str(router_path), device)

    # Optional override
    if args.router_noise_std is not None:
        router.routing_noise_std = float(args.router_noise_std)

    per_task = {}
    for task_id in range(args.num_tasks):
        per_task[TASK_NAMES[task_id]] = evaluate_task(
            model=model,
            router=router,
            task_id=task_id,
            split=split,
            w_vectorizer=w_vectorizer,
            args=args,
            eval_wrapper=eval_wrapper,
            temp_dir=temp_dir,
            with_generation=with_generation,
        )

    avg = {}
    for k in ["loss", "accuracy"] + T2M_QUALITY_METRICS:
        vals = [per_task[t][k] for t in per_task if k in per_task[t]]
        if vals:
            avg[k] = float(np.mean(vals))

    output = {
        "stage": f"after_task{stage}",
        "checkpoint": str(ckpt_path),
        "router_checkpoint": str(router_path),
        "split": split,
        "evaluation_protocol": "T2M O-LoRA MoE (task-id free routing)",
        "task_order": TASK_NAMES,
        "per_task": per_task,
        "average": avg,
    }

    out_dir = exp_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"after_task{stage}_{split}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_file}")

    del model
    torch.cuda.empty_cache()
    return output


def parse_args():
    p = argparse.ArgumentParser(description="O-LoRA MoE T2M Evaluation")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--generate-all-stages", action="store_true")
    p.add_argument(
        "--routing-mode",
        type=str,
        default="two_stage_prefix",
        choices=["two_stage_prefix", "text_ae"],
        help="T2M routing mode. Default is two_stage_prefix.",
    )
    p.add_argument(
        "--prefix-len",
        type=int,
        default=10,
        help="Number of motion tokens to generate as prefix before routing (two_stage_prefix mode).",
    )
    p.add_argument(
        "--router-noise-std",
        type=float,
        default=None,
        help="Override router noise std at inference time (default: from router checkpoint).",
    )

    p.add_argument(
        "--router-top-k",
        type=int,
        default=1,
        help="Top-K experts to consider at inference time for token-accuracy routing (default: 1).",
    )
    p.add_argument(
        "--router-infer-mode",
        type=str,
        default="top1",
        choices=["top1", "mixture", "rerank", "merge"],
        help="How to use the Top-K experts for token-accuracy evaluation (default: top1).",
    )

    # Must match training
    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=64)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--olora-r", type=int, default=64)
    p.add_argument("--olora-alpha", type=int, default=64)
    p.add_argument("--olora-target-modules", type=str, default="qv")

    p.add_argument("--pretrained-path", type=str, default=str(_MOTION_AGENT_ROOT.parent / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"))
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--split-mode", type=str, default="random_80_20", choices=["predefined", "random_80_20"])
    p.add_argument("--split-seed", type=int, default=42)

    # VQ-VAE config
    p.add_argument("--nb-code", type=int, default=512)
    p.add_argument("--code-dim", type=int, default=512)
    p.add_argument("--output-emb-width", type=int, default=512)
    p.add_argument("--down-t", type=int, default=2)
    p.add_argument("--stride-t", type=int, default=2)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dilation-growth-rate", type=int, default=3)
    p.add_argument("--vq-act", type=str, default="relu")
    p.add_argument("--vq-norm", type=str, default=None)
    p.add_argument("--quantizer", type=str, default="ema_reset")
    p.add_argument("--mu", type=float, default=0.99)
    p.add_argument("--beta", type=float, default=1.0)

    args = p.parse_args()
    args.training_task = "t2m"
    args.nb_joints = 22
    args.dataname = "t2m"
    return args


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    set_split_mode(args.split_mode, args.split_seed)

    stages = [args.stage] if args.stage is not None else list(range(args.num_tasks))
    for s in stages:
        ckpt = exp_dir / f"olora_moe_task_{s}_best.pth"
        rtr = exp_dir / f"router_task_{s}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        if not rtr.exists():
            raise FileNotFoundError(f"Missing router checkpoint: {rtr}")

    print("=" * 70)
    print("O-LoRA MoE T2M Evaluation (task-id free)")
    print("=" * 70)
    print(f"Experiment: {exp_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Stages:     {stages}")
    print(f"Routing:    {args.routing_mode} (prefix_len={args.prefix_len})")

    w_vectorizer = WordVectorizer(str(PRETRAINED / 'glove'), "our_vab")
    with_generation = not args.skip_generation

    eval_wrapper = None
    temp_dir = None
    if with_generation:
        print("Loading EvaluatorModelWrapper for quality metrics...")
        opt_path = str(PRETRAINED / 'checkpoints' / "t2m" / "Comp_v6_KLD005" / "opt.txt")
        wrapper_opt = get_opt(opt_path, args.device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        temp_dir = tempfile.mkdtemp(prefix="olora_moe_t2m_eval_")
        print(f"  Temp dir: {temp_dir}")

    try:
        for s in stages:
            ckpt = exp_dir / f"olora_moe_task_{s}_best.pth"
            rtr = exp_dir / f"router_task_{s}.pth"
            gen_for_stage = with_generation and (args.generate_all_stages or (s == args.num_tasks - 1))
            evaluate_stage(
                stage=s,
                ckpt_path=ckpt,
                router_path=rtr,
                exp_dir=exp_dir,
                split=args.split,
                w_vectorizer=w_vectorizer,
                args=args,
                eval_wrapper=eval_wrapper,
                temp_dir=temp_dir,
                with_generation=gen_for_stage,
            )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp dir: {temp_dir}")


if __name__ == "__main__":
    main()
