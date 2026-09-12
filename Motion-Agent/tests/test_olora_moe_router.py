"""Unit tests for the O-LoRA MoE AutoencoderRouter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

MOTION_AGENT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(MOTION_AGENT_PATH))

from continual_learning.olora_moe.router import AutoencoderRouter, TaskAutoencoder


def test_threshold_update_matches_mean_plus_k_std():
    ae = TaskAutoencoder(input_dim=4, hidden_dim=8)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    k = 2.5
    ae.update_threshold(losses, k=k)
    expected = losses.mean() + k * losses.std()
    assert torch.isclose(ae.threshold, expected)


def test_detect_unseen_from_losses_all_reject():
    router = AutoencoderRouter(embed_dim=4, max_experts=3)
    router.set_num_experts(2)
    # set thresholds small so large losses reject
    router.autoencoders[0].threshold.fill_(0.1)
    router.autoencoders[1].threshold.fill_(0.1)
    all_losses = torch.tensor(
        [
            [1.0, 1.2, float("inf")],
            [0.2, 2.0, float("inf")],
        ]
    )
    unseen = router.detect_unseen_from_losses(all_losses)
    assert unseen.dtype == torch.bool
    assert unseen.tolist() == [True, True]


def test_noise_can_change_routing_choice_distribution():
    torch.manual_seed(0)
    router = AutoencoderRouter(embed_dim=4, max_experts=2, routing_noise_std=0.0)
    router.set_num_experts(2)

    # Make both experts have identical recon losses by zeroing weights.
    for i in range(2):
        for p in router.autoencoders[i].parameters():
            p.data.zero_()
        router.autoencoders[i].threshold.fill_(float("inf"))

    x = torch.randn(256, 4)

    # No noise -> deterministic tie-breaking of torch.topk (implementation-defined but stable)
    router.routing_noise_std = 0.0
    w0, idx0, _ = router(x)
    uniq0 = set(idx0[:, 0].tolist())
    assert len(uniq0) == 1

    # With noise -> should sample both experts at least once.
    router.routing_noise_std = 5.0
    w1, idx1, _ = router(x)
    uniq1 = set(idx1[:, 0].tolist())
    assert uniq1.issubset({0, 1})
    assert len(uniq1) == 2
