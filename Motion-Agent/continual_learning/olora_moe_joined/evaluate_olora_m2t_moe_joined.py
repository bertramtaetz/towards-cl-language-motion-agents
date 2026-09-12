"""O-LoRA MoE (Joined) M2T Evaluation -- task-id free routing.

Builds the continual-learning matrix R[stage, task] like the O-LoRA M2T
baseline, but chooses the adapter at inference time via a router (no task id).

Checkpoints expected:
  <exp-dir>/olora_moe_task_{stage}_best.pth
  <exp-dir>/router_task_{stage}.pth

Routing:
  - Compute routing embeddings from the *motion tokens* (mean pooled token embeddings)
  - Route to top-1 expert (task_{k})
  - If unseen (all thresholds reject) -> fallback to base adapter 'm2t'

Outputs are compatible with compute_cl_metrics_token_acc.py and
compute_cl_metrics_m2t.py.
"""

from __future__ import annotations

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import os
import re
import sys
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

from models.mllm import MotionLLM
from models.training_utils import process_batch
from peft import LoraConfig
from utils.word_vectorizer import WordVectorizer
from continual_learning.utils.data_manager import (
    TASK_NAMES,
    get_task_loader,
    set_split_mode,
)
from continual_learning.olora_moe.router import AutoencoderRouter
from continual_learning.olora_moe.topk_token_inference import (
    MergedAdapterSpec,
    canonical_merge_name,
    compute_causal_lm_per_sample_loss_from_logits,
    compute_motionllm_token_accuracy_from_logits,
    ensure_merged_lora_adapter,
    ensure_weighted_merged_lora_adapter,
)


# Data root for multi-reference captions (TM2T protocol)
DATA_ROOT = _PROJECT_ROOT / "datasets" / "HumanML3D" / "HumanML3D"
NLG_METRICS = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider", "spice", "bertscore"]


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


try:
    from nlgeval import NLGEval

    HAVE_NLGEVAL = True
except ImportError:
    HAVE_NLGEVAL = False
    print("Warning: nlgeval not installed. NLG metrics will be skipped.")

try:
    from bert_score import score as bert_score_fn

    HAVE_BERTSCORE = True
except ImportError:
    HAVE_BERTSCORE = False
    print("Warning: bert_score not installed. BERTScore will be skipped.")


def _resolve_target_modules(mode: str) -> List[str]:
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

    # Merge pretrained m2t then remove adapter so we can add task adapters
    if getattr(args, "pretrained_path", None) and os.path.exists(args.pretrained_path):
        print(f"  Merging pretrained m2t adapter from {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, "m2t")
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
def compute_motion_router_embedding(model: MotionLLM, motion_tokens: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Mean pool motion token embeddings for routing."""
    emb_layer = model.llm.get_input_embeddings()
    pooled = []
    for mtok in motion_tokens:
        mtok = mtok.to(device)
        e = emb_layer(mtok)  # [L,H]
        pooled.append(e.mean(dim=0))
    return torch.stack(pooled, dim=0)


def compute_token_accuracy_taskid_free(
    model: MotionLLM,
    router: AutoencoderRouter,
    loader,
    device: torch.device,
    true_task_id: int,
    base_adapter: str = "m2t",
    router_top_k: int = 1,
    router_infer_mode: str = "top1",
) -> Tuple[float, float, dict]:
    model.eval()
    router.eval()

    all_losses: List[float] = []
    all_accs: List[float] = []

    routed_counter = Counter()
    confusion_counter = Counter()
    unseen_counter = 0

    eligible_for_task_acc = int(true_task_id) < int(router.num_active_experts)
    router_count = 0
    router_top1_correct = 0
    router_top2_correct = 0
    fallback_on_seen = 0

    for batch in tqdm(loader, desc="token acc (routed)", leave=False):
        _, _, caption, _, motion, m_length, _, name = batch
        captions = list(caption)

        # Encode motion to VQ-VAE tokens
        motion_tokens = []
        for i in range(motion.size(0)):
            m = motion[i : i + 1, : m_length[i], :].to(device)
            with torch.no_grad():
                tok = model.net.encode(m).squeeze(0)
            tok_np = tok.cpu().numpy()
            tok_remapped = torch.from_numpy(model.motion_token_indices[tok_np]).to(device)
            motion_tokens.append(tok_remapped)

        # Route based on motion embedding (compute logits once so top-1/top-2 share the same noise sample)
        emb = compute_motion_router_embedding(model, motion_tokens, device)
        # NOTE: avoid shadowing the Python list `all_losses` used for token-level losses.
        full_logits, router_losses = router.compute_full_logits_and_losses(emb, add_noise=True)
        # Get Top-K indices for inference + Top-2 for diagnostics.
        k_infer = max(1, min(int(router_top_k), int(router.num_active_experts)))
        k_diag = 2 if router.num_active_experts >= 2 else 1
        k_top = max(k_infer, k_diag)

        topk_logits, topk_idx = torch.topk(full_logits, k=min(k_top, router.num_active_experts), dim=-1)
        topk_weights = torch.softmax(topk_logits[:, :k_infer], dim=-1)
        unseen = router.detect_unseen_from_losses(router_losses)

        # Router diagnostics + routed-counts are always computed from top-1.
        for i in range(len(captions)):
            if bool(unseen[i].item()):
                unseen_counter += 1
                routed_counter[base_adapter] += 1
                confusion_counter[base_adapter] += 1
                if eligible_for_task_acc:
                    fallback_on_seen += 1
            else:
                pred_idx = int(topk_idx[i, 0].item())
                an = f"task_{pred_idx}"
                routed_counter[an] += 1
                confusion_counter[an] += 1

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
            training_task="m2t",
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
                    outputs = model.llm(
                        input_ids=input_ids[idxs],
                        attention_mask=attention_mask[idxs],
                        return_dict=True,
                        labels=targets[idxs],
                    )

                all_losses.append(float(outputs.loss.item()))
                all_accs.append(
                    float(compute_motionllm_token_accuracy_from_logits(logits=outputs.logits, labels=targets[idxs]).item())
                )

        elif mode == "mixture":
            # Mode A: mixture of logits across Top-K experts.
            bsz = input_ids.size(0)

            # Memory note:
            # Mixture mode materializes a (B,S,V) logits tensor. Keeping it in
            # bf16 avoids a large fp32 allocation and reduces OOM risk.
            mix_dtype = torch.bfloat16

            vocab_size = int(model.llm.get_output_embeddings().weight.size(0))
            mixture_logits = torch.zeros(
                (bsz, input_ids.size(1), vocab_size),
                device=device,
                dtype=mix_dtype,
            )

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
                        out = model.llm(
                            input_ids=input_ids[sel],
                            attention_mask=attention_mask[sel],
                            return_dict=True,
                        )
                    mixture_logits[sel] = mixture_logits[sel] + out.logits.to(dtype=mix_dtype) * w

            unseen_idxs = [i for i in range(bsz) if bool(unseen[i].item())]
            if unseen_idxs:
                model.llm.set_adapter(base_adapter)
                with torch.no_grad():
                    out = model.llm(
                        input_ids=input_ids[unseen_idxs],
                        attention_mask=attention_mask[unseen_idxs],
                        return_dict=True,
                    )
                mixture_logits[unseen_idxs] = out.logits.to(dtype=mix_dtype)

            per_sample_loss = compute_causal_lm_per_sample_loss_from_logits(logits=mixture_logits, labels=targets)
            all_losses.append(float(per_sample_loss.mean().item()))
            all_accs.append(float(compute_motionllm_token_accuracy_from_logits(logits=mixture_logits, labels=targets).item()))

        elif mode == "rerank":
            # Mode B: pick the expert (from Top-K) with lowest teacher-forced NLL per sample.
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

        elif mode in {"merge", "merge_unweighted"}:
            # Mode C: create a cached merged adapter per unique Top-K tuple.
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
                all_accs.append(float(compute_motionllm_token_accuracy_from_logits(logits=out.logits, labels=targets[idxs]).item()))

        elif mode == "merge_weighted":
            # Weighted merge is sample-dependent, so overwrite a single temporary
            # merged adapter and score samples sequentially.
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
        "router_top1": (float(router_top1_correct) / float(router_count) if eligible_for_task_acc and router_count else None),
        "router_top2": (float(router_top2_correct) / float(router_count) if eligible_for_task_acc and router_count else None),
        "fallback_rate_on_seen": (float(fallback_on_seen) / float(sum(routed_counter.values())) if eligible_for_task_acc else None),
    }

    expert_names = [f"task_{i}" for i in range(int(router.num_active_experts))]
    diag.update(_compute_utilization_stats(dict(routed_counter), expert_names))
    return float(np.mean(all_losses)), float(np.mean(all_accs)), diag


def load_all_references(motion_id: str) -> List[str]:
    text_file = DATA_ROOT / "texts" / f"{motion_id}.txt"
    if not text_file.exists():
        return []
    refs = []
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if parts:
            cap = parts[0].strip()
            if cap:
                refs.append(cap)
    return refs


def compute_bertscore(hypotheses: List[str], references: List[str], device: str) -> float:
    if not HAVE_BERTSCORE or not hypotheses:
        return 0.0
    try:
        _, _, F1 = bert_score_fn(
            hypotheses,
            references,
            lang="en",
            verbose=False,
            device=device if "cuda" in device else None,
        )
        return float(F1.mean().item() * 100)
    except Exception as e:
        print(f"  Warning: BERTScore failed: {e}")
        return 0.0


def generate_caption_with_adapter(model: MotionLLM, motion_np: np.ndarray, adapter_name: str, device: torch.device) -> str:
    """Generate caption with a specific adapter (bypasses MotionLLM.caption())."""
    model.llm.set_adapter(adapter_name)
    model.llm.eval()

    motion_norm = model.normalize(motion_np)
    motion_tensor = torch.from_numpy(motion_norm).float().to(device).unsqueeze(0)
    with torch.no_grad():
        motion_tokens = model.net.encode(motion_tensor).squeeze(0)
    motion_tokens = motion_tokens + model.nb_text_tokens + 2

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
    input_text = "### Input:\n<Motion>" + model.tokenizer.decode(motion_tokens) + "</Motion>\n\nResponse: "
    full_text = prompt + instruction + input_text
    input_ids = model.tokenizer.encode(full_text, return_tensors="pt").to(device)

    with torch.no_grad():
        pred = model.llm.generate(
            input_ids,
            max_length=200,
            num_beams=2,
            pad_token_id=model.tokenizer.pad_token_id,
        )

    pred = pred[0, len(input_ids[0]) :]
    pred_text = model.tokenizer.decode(pred)
    caption = pred_text.split("<eos>")[0].strip()
    caption = re.sub(r"<Motion_\d+>", "", caption)
    caption = re.sub(r"</?Motion>", "", caption)
    return caption.strip()


def _generate_caption_with_merge_unweighted(
    *,
    model: MotionLLM,
    motion_np: np.ndarray,
    expert_indices: List[int],
    device: torch.device,
    merge_prefix: str = "_merged_topk",
) -> str:
    merged_name = canonical_merge_name(merge_prefix, expert_indices)
    src_names = tuple(f"task_{int(e)}" for e in expert_indices)
    merged_name = ensure_merged_lora_adapter(
        peft_model=model.llm,
        spec=MergedAdapterSpec(merged_adapter_name=merged_name, source_adapter_names=src_names),
        base_lora_config=model.llm.peft_config[src_names[0]],
    )
    return generate_caption_with_adapter(model=model, motion_np=motion_np, adapter_name=merged_name, device=device)


def _generate_caption_with_merge_weighted(
    *,
    model: MotionLLM,
    motion_np: np.ndarray,
    expert_indices: List[int],
    weights: torch.Tensor,
    device: torch.device,
    merged_name: str = "_merged_topk_weighted",
) -> str:
    src_names = tuple(f"task_{int(e)}" for e in expert_indices)
    merged_name = ensure_weighted_merged_lora_adapter(
        peft_model=model.llm,
        merged_name=merged_name,
        source_adapter_names=src_names,
        weights=weights.to(device=device),
        base_lora_config=model.llm.peft_config[src_names[0]],
    )
    return generate_caption_with_adapter(model=model, motion_np=motion_np, adapter_name=merged_name, device=device)


@torch.no_grad()
def generate_caption_with_mixture(
    model: MotionLLM,
    motion_np: np.ndarray,
    expert_adapter_names: List[str],
    expert_weights: torch.Tensor,
    device: torch.device,
    max_length: int = 200,
) -> str:
    """Generate caption using a Top-K *mixture* of expert next-token logits (greedy).

    This is inference-only and does not require ground truth.

    Notes:
    - We keep expert weights fixed for the full generation (computed from router).
    - Greedy decoding is used (beam search would require additional bookkeeping).
    - Uses per-expert KV caches so runtime scales ~O(K*T).
    """

    if len(expert_adapter_names) == 0:
        raise ValueError("expert_adapter_names must be non-empty")
    expert_weights = expert_weights.to(device=device, dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum().clamp(min=1e-12)

    model.llm.eval()

    # Build motion tokens in vocab-id space.
    motion_norm = model.normalize(motion_np)
    motion_tensor = torch.from_numpy(motion_norm).float().to(device).unsqueeze(0)
    motion_tokens = model.net.encode(motion_tensor).squeeze(0)
    motion_tokens = motion_tokens + model.nb_text_tokens + 2

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
    input_text = "### Input:\n<Motion>" + model.tokenizer.decode(motion_tokens) + "</Motion>\n\nResponse: "
    full_text = prompt + instruction + input_text
    input_ids = model.tokenizer.encode(full_text, return_tensors="pt").to(device)

    # Initialize each expert cache using the prompt context.
    per_expert_cache: dict[str, object] = {}
    per_expert_next_logits: dict[str, torch.Tensor] = {}

    for an in expert_adapter_names:
        model.llm.set_adapter(an)
        out = model.llm(input_ids=input_ids, return_dict=True, use_cache=True)
        per_expert_cache[an] = out.past_key_values
        per_expert_next_logits[an] = out.logits[:, -1, :].float().squeeze(0)

    generated: List[int] = []
    cur_len = int(input_ids.size(1))

    # Greedy decode until max_length tokens in total sequence.
    while cur_len < int(max_length):
        mix = torch.zeros_like(per_expert_next_logits[expert_adapter_names[0]], dtype=torch.float32)
        for w, an in zip(expert_weights, expert_adapter_names):
            mix = mix + float(w.item()) * per_expert_next_logits[an]

        next_id = int(torch.argmax(mix, dim=-1).item())
        generated.append(next_id)
        cur_len += 1

        # Stop on EOS.
        if next_id == int(model.tokenizer.eos_token_id):
            break

        next_token = torch.tensor([[next_id]], device=device, dtype=torch.long)

        # Advance each expert one token with its cache.
        for an in expert_adapter_names:
            model.llm.set_adapter(an)
            out = model.llm(
                input_ids=next_token,
                past_key_values=per_expert_cache[an],
                return_dict=True,
                use_cache=True,
            )
            per_expert_cache[an] = out.past_key_values
            per_expert_next_logits[an] = out.logits[:, -1, :].float().squeeze(0)

    pred_text = model.tokenizer.decode(torch.tensor(generated, device="cpu"))
    caption = pred_text.split("<eos>")[0].strip()
    caption = re.sub(r"<Motion_\d+>", "", caption)
    caption = re.sub(r"</?Motion>", "", caption)
    return caption.strip()


def evaluate_task(
    model: MotionLLM,
    router: AutoencoderRouter,
    task_id: int,
    stage: int,
    split: str,
    w_vectorizer,
    args,
    nlg_eval,
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

    # --- Token accuracy ---
    loss, acc, diag = compute_token_accuracy_taskid_free(
        model,
        router,
        loader,
        device,
        true_task_id=task_id,
        base_adapter="m2t",
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

    # --- Caption generation + NLG metrics (optional) ---
    if with_generation and nlg_eval is not None:
        print("    Generating captions for NLG metrics...")
        predictions = []
        refs_first = []
        refs_multi = []
        generation_errors = 0

        for batch in tqdm(loader, desc="caption gen (routed)", leave=False):
            _, _, caption, _, motion, m_length, _, name = batch
            captions = list(caption)

            # Encode motion to tokens
            motion_tokens = []
            motion_denorm = []
            for i in range(motion.size(0)):
                m_np = motion[i : i + 1, : m_length[i], :].cpu().numpy()[0]
                m_denorm = model.mean + m_np * model.std
                motion_denorm.append(m_denorm)

                m = motion[i : i + 1, : m_length[i], :].to(device)
                with torch.no_grad():
                    tok = model.net.encode(m).squeeze(0)
                tok_remapped = torch.from_numpy(model.motion_token_indices[tok.cpu().numpy()]).to(device)
                motion_tokens.append(tok_remapped)

            # Route based on motion embedding
            emb = compute_motion_router_embedding(model, motion_tokens, device)
            full_logits, all_losses = router.compute_full_logits_and_losses(emb, add_noise=True)
            unseen = router.detect_unseen_from_losses(all_losses)

            k_infer = max(1, min(int(getattr(args, "router_top_k", 1)), int(router.num_active_experts)))
            topk_logits, topk_idx = torch.topk(full_logits, k=min(k_infer, router.num_active_experts), dim=-1)
            topk_w = torch.softmax(topk_logits, dim=-1) if topk_logits.numel() else None

            for i in range(len(captions)):
                try:
                    if bool(unseen[i].item()):
                        adapter_name = "m2t"
                    else:
                        adapter_name = f"task_{int(topk_idx[i, 0].item())}"

                    mode = (getattr(args, "router_infer_mode", "top1") or "top1").strip().lower()
                    use_mixture = (mode == "mixture") and (k_infer > 1) and (not bool(unseen[i].item()))
                    use_merge_unweighted = (mode in {"merge", "merge_unweighted"}) and (k_infer > 1) and (not bool(unseen[i].item()))
                    use_merge_weighted = (mode == "merge_weighted") and (k_infer > 1) and (not bool(unseen[i].item()))
                    if use_mixture:
                        expert_names = [f"task_{int(topk_idx[i, j].item())}" for j in range(k_infer)]
                        weights = topk_w[i]
                        pred = generate_caption_with_mixture(
                            model,
                            motion_denorm[i],
                            expert_adapter_names=expert_names,
                            expert_weights=weights,
                            device=device,
                            max_length=200,
                        )
                    elif use_merge_unweighted:
                        ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                        pred = _generate_caption_with_merge_unweighted(
                            model=model,
                            motion_np=motion_denorm[i],
                            expert_indices=ex,
                            device=device,
                        )
                    elif use_merge_weighted:
                        ex = [int(topk_idx[i, j].item()) for j in range(k_infer)]
                        pred = _generate_caption_with_merge_weighted(
                            model=model,
                            motion_np=motion_denorm[i],
                            expert_indices=ex,
                            weights=topk_w[i],
                            device=device,
                        )
                    else:
                        pred = generate_caption_with_adapter(model, motion_denorm[i], adapter_name, device)
                    motion_id = str(name[i]) if hasattr(name, "__getitem__") else None
                    refs = load_all_references(motion_id) if motion_id else [captions[i]]
                    if not refs:
                        refs = [captions[i]]
                    if len(refs) > 3:
                        refs = refs[:3]
                    while len(refs) < 3:
                        refs.append(refs[0])

                    predictions.append(pred)
                    refs_first.append(refs[0])
                    refs_multi.append(refs)
                except Exception as e:
                    generation_errors += 1
                    predictions.append("")
                    refs_first.append(captions[i])
                    refs_multi.append([captions[i], captions[i], captions[i]])

        result["num_generation_errors"] = int(generation_errors)

        if predictions:
            refs_transposed = [list(rs) for rs in zip(*refs_multi)]
            scores = nlg_eval.compute_metrics(refs_transposed, predictions)
            result["bleu1"] = scores.get("Bleu_1", 0.0) * 100
            result["bleu2"] = scores.get("Bleu_2", 0.0) * 100
            result["bleu3"] = scores.get("Bleu_3", 0.0) * 100
            result["bleu4"] = scores.get("Bleu_4", 0.0) * 100
            result["rouge_l"] = scores.get("ROUGE_L", 0.0) * 100
            result["cider"] = scores.get("CIDEr", 0.0) * 100
            if "SPICE" in scores:
                result["spice"] = scores["SPICE"] * 100
            if HAVE_BERTSCORE:
                result["bertscore"] = compute_bertscore(predictions, refs_first, args.device)

            print(
                f"    BLEU-4={result.get('bleu4', 0):.2f}, ROUGE-L={result.get('rouge_l', 0):.2f}, "
                f"CIDEr={result.get('cider', 0):.2f}, BERTScore={result.get('bertscore', 0):.2f}"
            )

    return result


def evaluate_stage(
    stage: int,
    ckpt_path: Path,
    router_path: Path,
    exp_dir: Path,
    split: str,
    w_vectorizer,
    args,
    nlg_eval,
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
            stage=stage,
            split=split,
            w_vectorizer=w_vectorizer,
            args=args,
            nlg_eval=nlg_eval,
            with_generation=with_generation,
        )

    # Compute averages
    avg = {}
    for k in ["loss", "accuracy"] + NLG_METRICS:
        vals = [per_task[t][k] for t in per_task if k in per_task[t]]
        if vals:
            avg[k] = float(np.mean(vals))

    output = {
        "stage": f"after_task{stage}",
        "checkpoint": str(ckpt_path),
        "router_checkpoint": str(router_path),
        "split": split,
        "evaluation_protocol": "TM2T (nlgeval multi-ref), O-LoRA MoE Joined (task-id free routing)",
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
    p = argparse.ArgumentParser(description="O-LoRA MoE Joined M2T Evaluation")
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
        help="Top-K experts to consider at inference time for token-accuracy routing (default: 1).",
    )
    p.add_argument(
        "--router-infer-mode",
        type=str,
        default="top1",
        choices=["top1", "mixture", "rerank", "merge", "merge_unweighted", "merge_weighted"],
        help="How to use the Top-K experts for token-accuracy evaluation (default: top1).",
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
        default=str(_MOTION_AGENT_ROOT.parent / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"),
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
    args.training_task = "m2t"
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
    print("O-LoRA MoE Joined M2T Evaluation (task-id free)")
    print("=" * 70)
    print(f"Output dir: {exp_dir}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Stages:     {stages}")
    print(f"Generation: {'enabled (final stage only)' if not args.skip_generation else 'disabled'}")

    w_vectorizer = WordVectorizer(str(PRETRAINED / 'glove'), "our_vab")

    nlg_eval = None
    with_generation = not args.skip_generation
    if with_generation and HAVE_NLGEVAL:
        disable_spice = str(os.environ.get("MOTION_AGENT_DISABLE_SPICE", "0")).strip() in {"1", "true", "yes"}

        metrics_to_omit = [
            "METEOR",
            "EmbeddingAverageCosineSimilarity",
            "SkipThoughtCS",
            "VectorExtremaCosineSimilarity",
            "GreedyMatchingScore",
        ]
        if disable_spice:
            metrics_to_omit.append("SPICE")
            print("Loading NLGEval (BLEU, ROUGE-L, CIDEr; SPICE disabled via MOTION_AGENT_DISABLE_SPICE=1)...")
        else:
            print("Loading NLGEval (BLEU, ROUGE-L, CIDEr, SPICE)...")

        nlg_eval = NLGEval(metrics_to_omit=metrics_to_omit)
    elif with_generation and not HAVE_NLGEVAL:
        print("Warning: NLGEval unavailable — NLG metrics will be skipped.")

    for s in stages:
        ckpt = ckpt_dir / f"olora_moe_task_{s}_best.pth"
        rtr = ckpt_dir / f"router_task_{s}.pth"
        gen_for_stage = with_generation and (args.generate_all_stages or (s == args.num_tasks - 1))
        evaluate_stage(
            stage=s,
            ckpt_path=ckpt,
            router_path=rtr,
            exp_dir=exp_dir,
            split=args.split,
            w_vectorizer=w_vectorizer,
            args=args,
            nlg_eval=nlg_eval,
            with_generation=gen_for_stage,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
