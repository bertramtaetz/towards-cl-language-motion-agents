"""O-LoRA MoE (Joined) T2M Evaluation -- task-id free routing.

This is a joined variant of `continual_learning/olora_moe/evaluate_olora_t2m_moe.py`.

Key difference:
  - Routing embedding (T2M) is computed from *caption-only* mean pooled LLM input
    embeddings (no instruction/prompt template), matching the other-branch
    `olora_multi_adapter` MoE experiments.

Unseen detection:
  - If unseen (all thresholds reject) -> fallback to base adapter 't2m'

Outputs are compatible with:
  - continual_learning/compute_cl_metrics_token_acc.py
  - continual_learning/compute_cl_metrics_t2m.py
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
_PROJECT_ROOT = _MOTION_AGENT_ROOT.parent

sys.path.insert(0, str(_MOTION_AGENT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from continual_learning.olora_moe.router import AutoencoderRouter
from continual_learning.olora_moe.topk_token_inference import (
    MergedAdapterSpec,
    canonical_merge_name,
    compute_causal_lm_per_sample_loss_from_logits,
    compute_motionllm_token_accuracy_from_logits,
    ensure_merged_lora_adapter,
    ensure_weighted_merged_lora_adapter,
)
from continual_learning.olora_moe_joined.embeddings import compute_caption_router_embedding
from continual_learning.utils.data_manager import TASK_NAMES, get_task_loader, set_split_mode
from models.evaluator_wrapper import EvaluatorModelWrapper
from models.mllm import MotionLLM
from models.training_utils import process_batch
from options.get_eval_option import get_opt
from peft import LoraConfig
from utils.evaluation import evaluation_test
from utils.word_vectorizer import WordVectorizer


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
def generate_t2m_caption_only_single_mixture(
    *,
    model: MotionLLM,
    router: AutoencoderRouter,
    caption: str,
    device: torch.device,
    router_top_k: int,
    base_adapter: str = "t2m",
    max_motion_len: int = 200,
    text_max_length: int = 128,
) -> torch.Tensor:
    """Caption-only routing + Top-K mixture decoding (greedy).

    This is analogous to `generate_t2m_text_ae_single_mixture` but uses caption-only
    embeddings for routing.

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
    # Route from caption-only embedding
    # ------------------------------------------------------------
    emb = compute_caption_router_embedding(model, [caption], device=device, max_length=int(text_max_length))
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


def compute_token_accuracy_taskid_free(
    model: MotionLLM,
    router: AutoencoderRouter,
    loader,
    device: torch.device,
    true_task_id: int,
    base_adapter: str = "t2m",
    router_top_k: int = 1,
    router_infer_mode: str = "top1",
    router_text_max_length: int = 128,
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
        emb = compute_caption_router_embedding(
            model=model,
            captions=captions,
            device=device,
            max_length=int(router_text_max_length),
        )
        full_logits, router_losses = router.compute_full_logits_and_losses(emb, add_noise=True)

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
            bsz = input_ids.size(0)
            mix_dtype = torch.bfloat16
            vocab_size = int(model.llm.get_output_embeddings().weight.size(0))
            mixture_logits = torch.zeros((bsz, input_ids.size(1), vocab_size), device=device, dtype=mix_dtype)

            for j in range(k_infer):
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
                        out = model.llm(input_ids=input_ids[sel], attention_mask=attention_mask[sel], return_dict=True)
                    mixture_logits[sel] = mixture_logits[sel] + out.logits.to(dtype=mix_dtype) * w

            unseen_idxs = [i for i in range(input_ids.size(0)) if bool(unseen[i].item())]
            if unseen_idxs:
                model.llm.set_adapter(base_adapter)
                with torch.no_grad():
                    out = model.llm(input_ids=input_ids[unseen_idxs], attention_mask=attention_mask[unseen_idxs], return_dict=True)
                mixture_logits[unseen_idxs] = out.logits.to(dtype=mix_dtype)

            per_sample_loss = compute_causal_lm_per_sample_loss_from_logits(logits=mixture_logits, labels=targets)
            all_losses.append(float(per_sample_loss.mean().item()))
            all_accs.append(
                float(compute_motionllm_token_accuracy_from_logits(logits=mixture_logits, labels=targets).item())
            )

        elif mode == "rerank":
            bsz = input_ids.size(0)
            per_sample_best_loss = torch.full((bsz,), float("inf"), device=device)
            per_sample_best_adapter = [base_adapter for _ in range(bsz)]

            seen_idxs = [i for i in range(bsz) if not bool(unseen[i].item())]
            if seen_idxs:
                for j in range(k_infer):
                    adapter_names = [f"task_{int(topk_idx[i, j].item())}" for i in seen_idxs]
                    unique_adapters = sorted(set(adapter_names))
                    for an in unique_adapters:
                        sel = [seen_idxs[ii] for ii, a in enumerate(adapter_names) if a == an]
                        model.llm.set_adapter(an)
                        with torch.no_grad():
                            out = model.llm(input_ids=input_ids[sel], attention_mask=attention_mask[sel], return_dict=True)
                        losses = compute_causal_lm_per_sample_loss_from_logits(logits=out.logits, labels=targets[sel])
                        for offset, sidx in enumerate(sel):
                            l = float(losses[offset].item())
                            if l < float(per_sample_best_loss[sidx].item()):
                                per_sample_best_loss[sidx] = losses[offset]
                                per_sample_best_adapter[sidx] = an

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
                batch_accs.append(
                    compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[idxs]).item()
                )
            all_losses.append(float(torch.tensor(batch_losses).mean().item()) if batch_losses else 0.0)
            all_accs.append(float(torch.tensor(batch_accs).mean().item()) if batch_accs else 0.0)

        elif mode in {"merge", "merge_unweighted"}:
            bsz = input_ids.size(0)
            merge_prefix = "_merged_topk"
            groups: Dict[str, List[int]] = defaultdict(list)

            for i in range(bsz):
                if bool(unseen[i].item()):
                    groups[base_adapter].append(i)
                    continue
                ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                merged_name = canonical_merge_name(merge_prefix, ex)
                src_names = tuple(f"task_{e}" for e in ex)
                merged_name = ensure_merged_lora_adapter(
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
                all_accs.append(
                    float(compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[idxs]).item())
                )

        elif mode == "merge_weighted":
            # Weighted merge is sample-dependent (weights differ per sample), so
            # we overwrite a single temporary merged adapter and score samples
            # sequentially.
            weighted_name = "_merged_topk_weighted"
            bsz = input_ids.size(0)
            for i in range(bsz):
                if bool(unseen[i].item()):
                    an = base_adapter
                else:
                    ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                    src_names = tuple(f"task_{e}" for e in ex)
                    an = ensure_weighted_merged_lora_adapter(
                        peft_model=model.llm,
                        merged_name=weighted_name,
                        source_adapter_names=src_names,
                        weights=topk_weights[i].to(device=device),
                        base_lora_config=model.llm.peft_config[src_names[0]],
                    )

                model.llm.set_adapter(an)
                with torch.no_grad():
                    out = model.llm(
                        input_ids=input_ids[i : i + 1],
                        attention_mask=attention_mask[i : i + 1],
                        return_dict=True,
                        labels=targets[i : i + 1],
                    )
                all_losses.append(float(out.loss.item()))
                all_accs.append(
                    float(
                        compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[i : i + 1]).item()
                    )
                )

        else:
            raise ValueError(f"Unknown --router-infer-mode={router_infer_mode}")

    diag = {
        "unseen_count": int(unseen_counter),
        "unseen_rate": float(unseen_counter / max(1, sum(routed_counter.values()))),
        "routed_counts": dict(routed_counter),
        "confusion_counts": dict(confusion_counter),
        "router_num_active_experts": int(router.num_active_experts),
        "router_true_task_id": int(true_task_id),
        "router_top1": (
            (float(router_top1_correct) / float(router_count) if eligible_for_task_acc and router_count else None)
        ),
        "router_top2": (
            (float(router_top2_correct) / float(router_count) if eligible_for_task_acc and router_count else None)
        ),
        "fallback_rate_on_seen": (
            float(fallback_on_seen) / float(sum(routed_counter.values())) if eligible_for_task_acc else None
        ),
    }
    expert_names = [f"task_{i}" for i in range(int(router.num_active_experts))]
    diag.update(_compute_utilization_stats(dict(routed_counter), expert_names))
    return float(np.mean(all_losses)), float(np.mean(all_accs)), diag


class RoutedGeneratorWrapper:
    """Adapter-routing wrapper for evaluation_test() motion quality metrics."""

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        *,
        text_max_length: int = 128,
    ):
        self._model = model
        self._router = router
        self._device = device
        self._text_max_length = int(text_max_length)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        results = []
        self._model.eval()
        self._router.eval()
        with torch.no_grad():
            emb = compute_caption_router_embedding(
                model=self._model,
                captions=captions,
                device=self._device,
                max_length=self._text_max_length,
            )
            topk_w, topk_idx, all_losses = self._router(emb, return_all_losses=True)
            if all_losses is None:
                unseen = self._router.detect_unseen_task(emb)
            else:
                unseen = self._router.detect_unseen_from_losses(all_losses)

            for i in range(len(captions)):
                if bool(unseen[i].item()):
                    self._model.adapter_override = "t2m"
                else:
                    self._model.adapter_override = f"task_{int(topk_idx[i, 0].item())}"
                results.append(self._model.generate(captions[i]))
            self._model.adapter_override = None
        return results


class CaptionOnlyMixtureGeneratorWrapper:
    """Caption-only routing + Top-K mixture decoding wrapper for evaluation_test()."""

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        *,
        router_top_k: int,
        text_max_length: int = 128,
    ):
        self._model = model
        self._router = router
        self._device = device
        self._router_top_k = int(router_top_k)
        self._text_max_length = int(text_max_length)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        results = []
        self._model.eval()
        self._router.eval()
        for cap in captions:
            mot = generate_t2m_caption_only_single_mixture(
                model=self._model,
                router=self._router,
                caption=cap,
                device=self._device,
                router_top_k=self._router_top_k,
                base_adapter="t2m",
                max_motion_len=int(max_length),
                text_max_length=self._text_max_length,
            )
            results.append(mot)
        return results


class CaptionOnlyMergeUnweightedGeneratorWrapper:
    """Caption-only routing + Top-K *merged adapter* decoding wrapper.

    This corresponds to Appendix "Unweighted merge (implemented)": the selected
    experts are merged into a single adapter (rank K*r) by concatenating LoRA
    factors. Decoding then proceeds with a single KV cache / single trajectory.
    """

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        *,
        router_top_k: int,
        text_max_length: int = 128,
        merge_prefix: str = "_merged_topk",
    ):
        self._model = model
        self._router = router
        self._device = device
        self._router_top_k = int(router_top_k)
        self._text_max_length = int(text_max_length)
        self._merge_prefix = str(merge_prefix)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        results = []
        self._model.eval()
        self._router.eval()
        with torch.no_grad():
            emb = compute_caption_router_embedding(
                model=self._model,
                captions=captions,
                device=self._device,
                max_length=self._text_max_length,
            )
            full_logits, all_losses = self._router.compute_full_logits_and_losses(emb, add_noise=True)
            unseen = self._router.detect_unseen_from_losses(all_losses)

            k_infer = max(1, min(int(self._router_top_k), int(self._router.num_active_experts)))
            if int(self._router.num_active_experts) > 0:
                _, topk_idx = torch.topk(full_logits, k=min(k_infer, self._router.num_active_experts), dim=-1)
            else:
                topk_idx = None

            for i, cap in enumerate(captions):
                if bool(unseen[i].item()) or topk_idx is None:
                    self._model.adapter_override = "t2m"
                else:
                    ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                    merged_name = canonical_merge_name(self._merge_prefix, ex)
                    src_names = tuple(f"task_{e}" for e in ex)
                    merged_name = ensure_merged_lora_adapter(
                        peft_model=self._model.llm,
                        spec=MergedAdapterSpec(merged_adapter_name=merged_name, source_adapter_names=src_names),
                        base_lora_config=self._model.llm.peft_config[src_names[0]],
                    )
                    self._model.adapter_override = merged_name

                results.append(self._model.generate(cap))

            self._model.adapter_override = None

        return results


class CaptionOnlyMergeWeightedGeneratorWrapper:
    """Caption-only routing + Top-K *weighted merged adapter* decoding wrapper.

    This implements Appendix "Weighted merge (conceptual extension)": for each
    sample, we overwrite a single temporary merged adapter whose delta equals
    the router-weighted sum of the selected Top-K deltas.
    """

    def __init__(
        self,
        model: MotionLLM,
        router: AutoencoderRouter,
        device: torch.device,
        *,
        router_top_k: int,
        text_max_length: int = 128,
        merged_name: str = "_merged_topk_weighted",
    ):
        self._model = model
        self._router = router
        self._device = device
        self._router_top_k = int(router_top_k)
        self._text_max_length = int(text_max_length)
        self._merged_name = str(merged_name)

    def __getattr__(self, item):
        return getattr(self._model, item)

    def generate_batch(self, captions: List[str], max_length: int = 200, verbose: bool = False):
        results = []
        self._model.eval()
        self._router.eval()
        with torch.no_grad():
            emb = compute_caption_router_embedding(
                model=self._model,
                captions=captions,
                device=self._device,
                max_length=self._text_max_length,
            )
            full_logits, all_losses = self._router.compute_full_logits_and_losses(emb, add_noise=True)
            unseen = self._router.detect_unseen_from_losses(all_losses)

            k_infer = max(1, min(int(self._router_top_k), int(self._router.num_active_experts)))
            if int(self._router.num_active_experts) > 0:
                topk_logits, topk_idx = torch.topk(full_logits, k=min(k_infer, self._router.num_active_experts), dim=-1)
                topk_w = torch.softmax(topk_logits, dim=-1) if topk_logits.numel() else None
            else:
                topk_idx = None
                topk_w = None

            for i, cap in enumerate(captions):
                if bool(unseen[i].item()) or topk_idx is None or topk_w is None:
                    self._model.adapter_override = "t2m"
                else:
                    ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                    src_names = tuple(f"task_{e}" for e in ex)
                    merged_name = ensure_weighted_merged_lora_adapter(
                        peft_model=self._model.llm,
                        merged_name=self._merged_name,
                        source_adapter_names=src_names,
                        weights=topk_w[i].to(device=self._device),
                        base_lora_config=self._model.llm.peft_config[src_names[0]],
                    )
                    self._model.adapter_override = merged_name

                results.append(self._model.generate(cap))

            self._model.adapter_override = None

        return results


def compute_motion_quality_taskid_free(
    model: MotionLLM,
    router: AutoencoderRouter,
    loader,
    eval_wrapper,
    temp_dir: str,
    device: torch.device,
    router_top_k: int = 1,
    router_infer_mode: str = "top1",
    router_text_max_length: int = 128,
) -> dict:
    mode = (router_infer_mode or "top1").strip().lower()
    use_mixture = (mode == "mixture") and (int(router_top_k) > 1)
    use_merge_unweighted = (mode in {"merge", "merge_unweighted"}) and (int(router_top_k) > 1)
    use_merge_weighted = (mode == "merge_weighted") and (int(router_top_k) > 1)
    if use_mixture:
        print(f"    Using caption-only mixture decoding (top_k={int(router_top_k)})")
        wrapped = CaptionOnlyMixtureGeneratorWrapper(
            model,
            router,
            device,
            router_top_k=int(router_top_k),
            text_max_length=int(router_text_max_length),
        )
    elif use_merge_unweighted:
        print(f"    Using caption-only merged-adapter decoding (unweighted, top_k={int(router_top_k)})")
        wrapped = CaptionOnlyMergeUnweightedGeneratorWrapper(
            model,
            router,
            device,
            router_top_k=int(router_top_k),
            text_max_length=int(router_text_max_length),
        )
    elif use_merge_weighted:
        print(f"    Using caption-only merged-adapter decoding (weighted, top_k={int(router_top_k)})")
        wrapped = CaptionOnlyMergeWeightedGeneratorWrapper(
            model,
            router,
            device,
            router_top_k=int(router_top_k),
            text_max_length=int(router_text_max_length),
        )
    else:
        wrapped = RoutedGeneratorWrapper(model, router, device, text_max_length=int(router_text_max_length))

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
        router_top_k=getattr(args, "router_top_k", 1),
        router_infer_mode=getattr(args, "router_infer_mode", "top1"),
        router_text_max_length=int(getattr(args, "router_text_max_length", 128)),
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
            router_top_k=getattr(args, "router_top_k", 1),
            router_infer_mode=getattr(args, "router_infer_mode", "top1"),
            router_text_max_length=int(getattr(args, "router_text_max_length", 128)),
        )
        result.update(quality)
        print(f"    FID={quality['fid']:.4f}, Top1={quality['top1']:.4f}")

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
    temp_dir,
    with_generation: bool,
) -> dict:
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"STAGE {stage} -- {ckpt_path.name}")
    print(f"Router: {router_path.name}")
    print(f"{'='*70}")

    model = load_olora_checkpoint(str(ckpt_path), args, device)
    router = load_router(str(router_path), device)

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
        "evaluation_protocol": "T2M, O-LoRA MoE Joined (caption-only routing, task-id free)",
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
    p = argparse.ArgumentParser(description="O-LoRA MoE Joined T2M Evaluation")
    p.add_argument("--exp-dir", required=True)
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to load checkpoints/routers from. "
            "If omitted, checkpoints are loaded from --exp-dir. "
            "Useful for shared-training + eval-to-different-output-dirs."
        ),
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--generate-all-stages", action="store_true")
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
        help="Top-K experts to consider at inference time for token-accuracy evaluation (default: 1).",
    )
    p.add_argument(
        "--router-infer-mode",
        type=str,
        default="top1",
        choices=["top1", "mixture", "rerank", "merge", "merge_unweighted", "merge_weighted"],
        help="How to use the Top-K experts for token-accuracy evaluation (default: top1).",
    )
    p.add_argument(
        "--router-text-max-length",
        type=int,
        default=128,
        help="Max length for caption-only tokenization used for routing (default: 128).",
    )

    # Must match training
    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=32)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--olora-r", type=int, default=32)
    p.add_argument("--olora-alpha", type=int, default=32)
    p.add_argument("--olora-target-modules", type=str, default="qv")
    p.add_argument(
        "--pretrained-path",
        type=str,
        default=str(_PROJECT_ROOT / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"),
    )
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
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else exp_dir
    set_split_mode(args.split_mode, args.split_seed)

    stages = [args.stage] if args.stage is not None else list(range(args.num_tasks))
    for s in stages:
        ckpt = ckpt_dir / f"olora_moe_task_{s}_best.pth"
        rtr = ckpt_dir / f"router_task_{s}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        if not rtr.exists():
            raise FileNotFoundError(f"Missing router checkpoint: {rtr}")

    print("=" * 70)
    print("O-LoRA MoE Joined T2M Evaluation (caption-only routing)")
    print("=" * 70)
    print(f"Output dir: {exp_dir}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Stages:     {stages}")
    print(f"Generation: {'enabled (final stage only)' if not args.skip_generation else 'disabled'}")
    print(f"Routing:    caption_only (max_len={int(args.router_text_max_length)})")
    print(f"Top-K token-acc: k={int(args.router_top_k)} mode={args.router_infer_mode}")

    w_vectorizer = WordVectorizer(str(PRETRAINED / 'glove'), "our_vab")

    eval_wrapper = None
    temp_dir = None
    if not args.skip_generation:
        opt_path = str(PRETRAINED / 'checkpoints' / "t2m" / "Comp_v6_KLD005" / "opt.txt")
        wrapper_opt = get_opt(opt_path, args.device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        temp_dir = tempfile.mkdtemp(prefix="olora_moe_joined_t2m_eval_")

    try:
        for s in stages:
            ckpt = ckpt_dir / f"olora_moe_task_{s}_best.pth"
            rtr = ckpt_dir / f"router_task_{s}.pth"
            gen_for_stage = (not args.skip_generation) and (args.generate_all_stages or (s == args.num_tasks - 1))
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

    print("\nDone.")


if __name__ == "__main__":
    main()
