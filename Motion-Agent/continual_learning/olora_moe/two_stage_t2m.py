"""Helpers for two-stage routing in T2M (prefix -> route -> continue).

We keep these helpers in a dedicated module so they can be reused by:
  - evaluate_olora_t2m_moe.py
  - router_diagnostics.py (optional)
  - unit tests

The central concept is the *motion-prefix embedding*:
  - given motion token indices in [0..nb_code-1]
  - map them to LLM vocab IDs for <Motion_i>
  - mean-pool the LLM input embeddings over the prefix

This matches how routing embeddings are trained in train_olora_moe.py when
--training-task=t2m and --router-embed-source=motion_prefix.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_t2m_prefix_embedding_from_indices(
    model,
    motion_token_indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compute mean-pooled motion-prefix embedding for T2M routing.

    Args:
        model: MotionLLM-like object with fields:
          - nb_text_tokens: number of original text tokens (before adding motion tokens)
          - llm.get_input_embeddings(): embedding layer
        motion_token_indices: 1D tensor of ints in [0..nb_code-1]
        device: device for computation

    Returns:
        1D tensor [H] (embedding dim).
    """

    if motion_token_indices.dim() != 1:
        motion_token_indices = motion_token_indices.reshape(-1)
    if motion_token_indices.numel() == 0:
        # caller should avoid this, but keep a safe behavior
        motion_token_indices = torch.zeros(1, dtype=torch.long, device=device)
    motion_token_indices = motion_token_indices.to(device=device, dtype=torch.long)

    # Motion tokens in vocab:
    #   <Motion>   -> nb_text_tokens + 0
    #   </Motion>  -> nb_text_tokens + 1
    #   <Motion_i> -> nb_text_tokens + 2 + i
    vocab_ids = int(model.nb_text_tokens) + 2 + motion_token_indices

    emb_layer = model.llm.get_input_embeddings()
    e = emb_layer(vocab_ids)  # [L,H]
    return e.mean(dim=0)
