"""Shared embedding helpers for `olora_moe_joined`.

Important: These are *not* external embedding models (e.g. SBERT). They are mean-pooled
token input embeddings of the MotionLLM's LLM backbone.
"""

from __future__ import annotations

from typing import List

import torch

from models.mllm import MotionLLM


@torch.no_grad()
def compute_caption_router_embedding(
    model: MotionLLM,
    captions: List[str],
    device: torch.device,
    *,
    max_length: int = 128,
) -> torch.Tensor:
    """Compute caption-only mean-pooled LLM input embeddings.

    Matches the other-branch `olora_multi_adapter` MoE experiments:
      - tokenize *captions only* (no instruction/prompt template)
      - max_length=128
      - mean pool LLM input embeddings with attention mask
    """
    tok = model.tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    emb_layer = model.llm.get_input_embeddings()
    emb = emb_layer(input_ids)  # [B,S,H]
    mask = attention_mask.unsqueeze(-1).float()  # [B,S,1]
    pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled
