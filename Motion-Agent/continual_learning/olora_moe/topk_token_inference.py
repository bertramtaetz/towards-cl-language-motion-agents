"""Top-K token-accuracy inference helpers for O-LoRA MoE.

This module extends the original O-LoRA MoE evaluation with optional Top-K
expert utilization strategies. It is intentionally **evaluation-only** and is
designed so that the default settings reproduce the original behavior.

Currently implemented here:
  - Creating *merged* adapters (mode C) by concatenating LoRA A/B matrices.

Notes on PEFT:
  - PEFT does not provide a stable built-in API for weighted adapter merging in
    this repo's environment.
  - However, LoRA deltas add linearly, so we can build an *exact* merged LoRA
    adapter representing the sum of K task adapters by concatenating their
    low-rank factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig


@dataclass(frozen=True)
class MergedAdapterSpec:
    """Specifies which adapters to merge into a single merged adapter."""

    merged_adapter_name: str
    source_adapter_names: tuple[str, ...]


def _iter_lora_modules(peft_model):
    """Yield (module_name, module) for modules that look like PEFT LoRA layers."""

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # PEFT stores adapters in ModuleDicts.
            if isinstance(getattr(module, "lora_A"), torch.nn.ModuleDict) and isinstance(
                getattr(module, "lora_B"), torch.nn.ModuleDict
            ):
                yield name, module


def _infer_adapter_rank_and_alpha(
    *,
    peft_model,
    adapter_name: str,
    base_lora_config: LoraConfig,
) -> tuple[int, int]:
    """Infer LoRA rank and alpha from *actual* PEFT module tensors.

    Why:
      In this codebase, evaluation sometimes loads checkpoints into a model
      whose CLI-provided LoRA config does not match the checkpoint rank. Because
      we assign `param.data = ckpt_tensor` directly, PEFT's stored `peft_config`
      can become stale (e.g. r=32) while the live LoRA parameters have a
      different shape (e.g. r=64).

    We therefore infer:
      - rank r from `lora_A[adapter].weight.shape[0]`
      - effective alpha from the layer's scaling if available

    Returns:
        (r, alpha)
    """

    adapter_name = str(adapter_name)
    # Default fallback: config values (may be stale).
    r_fallback = int(getattr(base_lora_config, "r"))
    alpha_fallback = int(getattr(base_lora_config, "lora_alpha"))

    for _, module in _iter_lora_modules(peft_model):
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        A = module.lora_A[adapter_name].weight
        r = int(A.shape[0])

        # Prefer the *effective* scaling used by PEFT at runtime.
        scaling = None
        if hasattr(module, "scaling") and isinstance(getattr(module, "scaling"), dict):
            scaling = module.scaling.get(adapter_name)

        if scaling is None:
            # Some PEFT versions store alpha in a dict.
            if hasattr(module, "lora_alpha") and isinstance(getattr(module, "lora_alpha"), dict):
                alpha = module.lora_alpha.get(adapter_name)
                if alpha is not None:
                    return r, int(alpha)
            return r, alpha_fallback

        # scaling = alpha / r  => alpha = scaling * r
        alpha_eff = int(round(float(scaling) * float(r)))
        alpha_eff = max(1, alpha_eff)
        return r, alpha_eff

    # If we couldn't find any LoRA module containing this adapter, fall back.
    return r_fallback, alpha_fallback


def _pick_compatible_merged_name(
    *,
    peft_model,
    merged_name: str,
    expected_rank: int,
) -> str:
    """Return a merged adapter name whose *live* tensor shapes match expected_rank.

    PEFT 0.15 in this repo lacks adapter deletion. If an adapter exists with the
    same name but wrong shape, we must create a new name and leave the old one
    in place.
    """

    merged_name = str(merged_name)

    def rank_matches(name: str) -> bool:
        for _, module in _iter_lora_modules(peft_model):
            if name in module.lora_A:
                return int(module.lora_A[name].weight.shape[0]) == int(expected_rank)
        return False

    # If name doesn't exist (or exists but no LoRA modules found), keep it.
    if not (hasattr(peft_model, "peft_config") and merged_name in getattr(peft_model, "peft_config")):
        return merged_name

    if rank_matches(merged_name):
        return merged_name

    # Stable alternative name keyed by expected rank.
    alt = f"{merged_name}__r{int(expected_rank)}"
    if hasattr(peft_model, "peft_config") and alt in getattr(peft_model, "peft_config") and rank_matches(alt):
        return alt
    return alt


def ensure_merged_lora_adapter(
    *,
    peft_model,
    spec: MergedAdapterSpec,
    base_lora_config: LoraConfig,
) -> str:
    """Ensure a merged adapter exists on the given PEFT model.

    The merged adapter is an *exact* sum of the LoRA deltas from the
    `source_adapter_names`.

    Implementation:
      - Create a new LoRA adapter with rank r_merge = K * r_base.
      - For every LoRA module: concatenate A matrices along the rank dimension
        and concatenate B matrices along the rank dimension.

    Args:
        peft_model: A PEFT model (e.g. PeftModelForCausalLM).
        spec: Defines target merged adapter name and source adapters.
        base_lora_config: The LoRA config used for the individual source
            adapters (same target_modules, dropout, etc.).

    Returns:
        The merged adapter name.

        Note: if an adapter with the requested name already exists but has the
        wrong rank (shape mismatch), a new name will be created and returned.
    """

    requested_name = str(spec.merged_adapter_name)
    src = tuple(spec.source_adapter_names)
    if len(src) == 0:
        raise ValueError("source_adapter_names must be non-empty")

    k = len(src)

    # Infer true rank/alpha from the live source adapter tensors.
    r_src, alpha_src = _infer_adapter_rank_and_alpha(
        peft_model=peft_model,
        adapter_name=src[0],
        base_lora_config=base_lora_config,
    )
    expected_rank = int(r_src) * int(k)
    expected_alpha = int(alpha_src) * int(k)

    merged_name = _pick_compatible_merged_name(
        peft_model=peft_model,
        merged_name=requested_name,
        expected_rank=expected_rank,
    )

    # Ensure the adapter exists (create if missing).
    if not (hasattr(peft_model, "peft_config") and merged_name in getattr(peft_model, "peft_config")):
        merged_cfg = LoraConfig(
            r=int(expected_rank),
            lora_alpha=int(expected_alpha),
            target_modules=list(base_lora_config.target_modules),
            lora_dropout=float(base_lora_config.lora_dropout),
            bias=str(base_lora_config.bias),
            task_type=base_lora_config.task_type,
        )
        peft_model.add_adapter(merged_name, merged_cfg)

    # Overwrite the randomly initialized merged LoRA matrices.
    touched = 0
    with torch.no_grad():
        for _, module in _iter_lora_modules(peft_model):
            # Only touch modules where all source adapters exist.
            if not all((s in module.lora_A and s in module.lora_B) for s in src):
                continue
            if merged_name not in module.lora_A or merged_name not in module.lora_B:
                # Should not happen, but keep safe.
                continue

            A_list = [module.lora_A[s].weight.data for s in src]  # each [r, in]
            B_list = [module.lora_B[s].weight.data for s in src]  # each [out, r]

            A_cat = torch.cat(A_list, dim=0)  # [k*r, in]
            B_cat = torch.cat(B_list, dim=1)  # [out, k*r]

            # Defensive shape check (helps diagnose stale adapters / config drift).
            if tuple(module.lora_A[merged_name].weight.shape) != tuple(A_cat.shape):
                raise RuntimeError(
                    "Merged adapter shape mismatch for lora_A: "
                    f"dest={tuple(module.lora_A[merged_name].weight.shape)} vs src_cat={tuple(A_cat.shape)}. "
                    f"requested_name={requested_name!r} resolved_name={merged_name!r} src={src}"
                )
            if tuple(module.lora_B[merged_name].weight.shape) != tuple(B_cat.shape):
                raise RuntimeError(
                    "Merged adapter shape mismatch for lora_B: "
                    f"dest={tuple(module.lora_B[merged_name].weight.shape)} vs src_cat={tuple(B_cat.shape)}. "
                    f"requested_name={requested_name!r} resolved_name={merged_name!r} src={src}"
                )

            module.lora_A[merged_name].weight.data.copy_(A_cat)
            module.lora_B[merged_name].weight.data.copy_(B_cat)
            touched += 1

    if touched == 0:
        raise RuntimeError(
            f"Failed to build merged adapter {merged_name!r}: no LoRA modules contained all source adapters {src}."
        )

    return merged_name


def ensure_weighted_merged_lora_adapter(
    *,
    peft_model,
    merged_name: str,
    source_adapter_names: Sequence[str],
    weights: torch.Tensor,
    base_lora_config: LoraConfig,
) -> str:
    """Ensure a *weighted* merged adapter exists and matches the given weights.

    This implements Appendix Eq. (weighted merge) by scaling each expert's LoRA
    delta before concatenation.

    We keep A matrices unchanged and scale B matrices:
      A_merge = cat(A_j)
      B_merge = cat(w_j * B_j)

    This yields:
      DeltaW_merge = sum_j w_j * DeltaW_j

    Important notes:
    - PEFT 0.15.0 in this repo does not provide adapter deletion APIs, so this
      function supports *overwriting* an existing merged adapter (same name)
      each call.
    - Because weights are input-dependent, callers typically reuse a constant
      merged_name (e.g. "_merged_topk_weighted") and overwrite its matrices for
      each sample.
    """

    src = tuple(str(s) for s in source_adapter_names)
    if len(src) == 0:
        raise ValueError("source_adapter_names must be non-empty")

    w = weights.detach()
    if w.ndim != 1:
        raise ValueError(f"weights must be 1D tensor of shape [K], got {tuple(w.shape)}")
    if w.numel() != len(src):
        raise ValueError(f"weights must have same length as source_adapter_names (K={len(src)}), got {int(w.numel())}")

    # Normalize weights defensively (router softmax should already sum to 1).
    w = w.to(dtype=torch.float32)
    w = w / w.sum().clamp(min=1e-12)

    # Ensure the merged adapter exists (if not, create it).
    k = len(src)

    # Infer true rank/alpha from live tensors for robustness.
    r_src, alpha_src = _infer_adapter_rank_and_alpha(
        peft_model=peft_model,
        adapter_name=src[0],
        base_lora_config=base_lora_config,
    )
    expected_rank = int(r_src) * int(k)
    expected_alpha = int(alpha_src) * int(k)

    requested_name = str(merged_name)
    merged_name = _pick_compatible_merged_name(
        peft_model=peft_model,
        merged_name=requested_name,
        expected_rank=expected_rank,
    )

    if not (hasattr(peft_model, "peft_config") and merged_name in getattr(peft_model, "peft_config")):
        merged_cfg = LoraConfig(
            r=int(expected_rank),
            lora_alpha=int(expected_alpha),
            target_modules=list(base_lora_config.target_modules),
            lora_dropout=float(base_lora_config.lora_dropout),
            bias=str(base_lora_config.bias),
            task_type=base_lora_config.task_type,
        )
        peft_model.add_adapter(merged_name, merged_cfg)

    # Overwrite merged LoRA matrices according to current weights.
    touched = 0
    with torch.no_grad():
        for _, module in _iter_lora_modules(peft_model):
            if not all((s in module.lora_A and s in module.lora_B) for s in src):
                continue
            if merged_name not in module.lora_A or merged_name not in module.lora_B:
                continue

            # NOTE: A: [r, in], B: [out, r]
            A_list = [module.lora_A[s].weight.data for s in src]
            B_list = [module.lora_B[s].weight.data for s in src]
            A_cat = torch.cat(A_list, dim=0)
            # Scale each B by weight w_j (broadcast over rows/out dim).
            B_scaled = [Bj * float(wj.item()) for Bj, wj in zip(B_list, w)]
            B_cat = torch.cat(B_scaled, dim=1)

            if tuple(module.lora_A[merged_name].weight.shape) != tuple(A_cat.shape):
                raise RuntimeError(
                    "Weighted merged adapter shape mismatch for lora_A: "
                    f"dest={tuple(module.lora_A[merged_name].weight.shape)} vs src_cat={tuple(A_cat.shape)}. "
                    f"requested_name={requested_name!r} resolved_name={merged_name!r} src={src}"
                )
            if tuple(module.lora_B[merged_name].weight.shape) != tuple(B_cat.shape):
                raise RuntimeError(
                    "Weighted merged adapter shape mismatch for lora_B: "
                    f"dest={tuple(module.lora_B[merged_name].weight.shape)} vs src_cat={tuple(B_cat.shape)}. "
                    f"requested_name={requested_name!r} resolved_name={merged_name!r} src={src}"
                )

            module.lora_A[merged_name].weight.data.copy_(A_cat)
            module.lora_B[merged_name].weight.data.copy_(B_cat)
            touched += 1

    if touched == 0:
        raise RuntimeError(
            f"Failed to build weighted merged adapter {merged_name!r}: no LoRA modules contained all source adapters {src}."
        )

    return merged_name


def canonical_merge_name(prefix: str, expert_indices: Sequence[int]) -> str:
    """Create a stable merged-adapter name for a set of experts."""

    tup = tuple(int(x) for x in expert_indices)
    return prefix + "__" + "_".join(str(x) for x in tup)


def compute_causal_lm_mean_loss_from_logits(*, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute HF-style causal LM loss (mean over non-ignored tokens).

    This matches `AutoModelForCausalLM(..., labels=labels).loss` behavior.
    """

    # shift so tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss


def compute_causal_lm_per_sample_loss_from_logits(*, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sample mean NLL for causal LM logits.

    Returns shape [B]. Ignores label tokens == -100.
    """

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)

    per_token = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))

    mask = shift_labels.ne(-100)
    denom = mask.sum(dim=1).clamp(min=1)
    per_sample = (per_token * mask.float()).sum(dim=1) / denom
    return per_sample


def compute_motionllm_token_accuracy_from_logits(*, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute token-accuracy as implemented in existing eval scripts.

    Motion-Agent's evaluation uses:
      chosen = argmax(logits)[:, 1:-1]
      lbls   = labels[:, 2:]
    and ignores lbls == -100.

    Returns scalar tensor (float).
    """

    chosen = torch.max(logits, dim=-1)[1][:, 1:-1]
    lbls = labels[:, 2:]
    correct = (chosen.reshape(-1) == lbls.reshape(-1)).long()
    valid_mask = (lbls != -100).reshape(-1)
    acc = (correct & valid_mask).sum().float() / (valid_mask.sum().float() + 1.0)
    return acc
