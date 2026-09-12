"""Unit tests for T2M two-stage routing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

MOTION_AGENT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(MOTION_AGENT_PATH))

from continual_learning.olora_moe.two_stage_t2m import compute_t2m_prefix_embedding_from_indices


class _DummyEmbLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(vocab_size * hidden, dtype=torch.float32).reshape(vocab_size, hidden))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]


class _DummyLLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self._emb = _DummyEmbLayer(vocab_size, hidden)

    def get_input_embeddings(self):
        return self._emb


class _DummyModel:
    def __init__(self, nb_text_tokens: int, nb_code: int, hidden: int):
        # minimal MotionLLM-like surface
        self.nb_text_tokens = nb_text_tokens
        self.args = type("Args", (), {"nb_code": nb_code})()
        # reserve space for: text tokens + <Motion>, </Motion>, <Motion_i>
        vocab_size = nb_text_tokens + 2 + nb_code
        self.llm = _DummyLLM(vocab_size=vocab_size, hidden=hidden)


def test_compute_t2m_prefix_embedding_vocab_offset_and_mean_pool():
    device = torch.device("cpu")
    nb_text_tokens = 100
    nb_code = 8
    hidden = 4
    model = _DummyModel(nb_text_tokens=nb_text_tokens, nb_code=nb_code, hidden=hidden)

    # motion token indices in [0..nb_code-1]
    motion_idx = torch.tensor([0, 3, 7], dtype=torch.long)
    emb = compute_t2m_prefix_embedding_from_indices(model, motion_idx, device=device)
    assert emb.shape == (hidden,)

    # Expected vocab ids: nb_text_tokens + 2 + idx
    vocab_ids = nb_text_tokens + 2 + motion_idx
    expected = model.llm.get_input_embeddings()(vocab_ids).mean(dim=0)
    assert torch.allclose(emb, expected)
