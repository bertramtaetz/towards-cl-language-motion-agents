"""Autoencoder router for O-LoRA MoE.

We reuse the reconstruction-loss routing idea from D-MoLE:
  - One small MLP autoencoder per task/expert
  - Route to expert with minimum reconstruction loss
  - Detect unseen inputs via per-expert threshold (mean + k*std)

Unlike D-MoLE, O-LoRA MoE uses *PEFT adapters* as experts (task_0, task_1, ...)
and therefore routing ultimately resolves to an adapter name.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAutoencoder(nn.Module):
    """2-layer MLP autoencoder for a single task."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

        # mean/std/threshold are buffers so they get saved in state_dict.
        self.register_buffer("threshold", torch.tensor(float("inf")))
        self.register_buffer("mean_recon_loss", torch.tensor(0.0))
        self.register_buffer("std_recon_loss", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def compute_reconstruction_loss(self, x: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        recon, _ = self.forward(x)
        if reduction == "none":
            return F.mse_loss(recon, x, reduction="none").mean(dim=-1)
        return F.mse_loss(recon, x, reduction=reduction)

    def update_threshold(self, recon_losses: torch.Tensor, k: float = 2.0) -> None:
        mean_loss = recon_losses.mean()
        std_loss = recon_losses.std()
        self.mean_recon_loss.copy_(mean_loss)
        self.std_recon_loss.copy_(std_loss)
        self.threshold.copy_(mean_loss + k * std_loss)

    def is_within_threshold(self, recon_loss: torch.Tensor) -> torch.Tensor:
        return recon_loss <= self.threshold


class AutoencoderRouter(nn.Module):
    """Autoencoder-based router.

    Returns routing decisions over experts indexed [0..num_active_experts-1].
    """

    def __init__(
        self,
        embed_dim: int,
        max_experts: int = 5,
        hidden_dim: int = 256,
        top_k: int = 1,
        routing_noise_std: float = 0.0,
        threshold_k: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_experts = max_experts
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.routing_noise_std = float(routing_noise_std)
        self.threshold_k = threshold_k

        self.num_active_experts = 0
        self.autoencoders = nn.ModuleList(
            [TaskAutoencoder(embed_dim, hidden_dim) for _ in range(max_experts)]
        )
        self.register_buffer("trained_mask", torch.zeros(max_experts, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Expert lifecycle
    # ------------------------------------------------------------------
    def add_expert(self) -> int:
        if self.num_active_experts >= self.max_experts:
            raise ValueError(f"Maximum experts ({self.max_experts}) reached")
        idx = self.num_active_experts
        self.num_active_experts += 1
        self.trained_mask[idx] = True
        return idx

    def set_num_experts(self, num_experts: int) -> None:
        self.num_active_experts = min(int(num_experts), self.max_experts)
        self.trained_mask[: self.num_active_experts] = True
        self.trained_mask[self.num_active_experts :] = False

    def get_current_autoencoder(self) -> TaskAutoencoder:
        if self.num_active_experts == 0:
            raise ValueError("No active experts")
        return self.autoencoders[self.num_active_experts - 1]

    def freeze_prior_autoencoders(self) -> None:
        for i in range(self.num_active_experts - 1):
            for p in self.autoencoders[i].parameters():
                p.requires_grad = False

    def unfreeze_all_autoencoders(self) -> None:
        for ae in self.autoencoders:
            for p in ae.parameters():
                p.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        if self.num_active_experts == 0:
            return []
        return list(self.get_current_autoencoder().parameters())

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def compute_full_logits_and_losses(
        self,
        embedding: torch.Tensor,
        *,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing logits and reconstruction losses.

        Returns:
          full_logits: [B, max_experts] where inactive experts are -inf.
          all_losses:  [B, max_experts] where inactive experts are +inf.

        Notes:
        - Logits are negative reconstruction loss.
        - Optional Gaussian noise is added to *active* logits (MoE-style exploration).
        """
        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        bsz = embedding.size(0)
        device = embedding.device

        # Losses
        all_losses = torch.full((bsz, self.max_experts), float("inf"), device=device)
        for i in range(self.num_active_experts):
            all_losses[:, i] = self.autoencoders[i].compute_reconstruction_loss(
                embedding, reduction="none"
            )

        # Logits (active)
        active_losses = all_losses[:, : self.num_active_experts]
        logits_active = -active_losses

        if add_noise and self.routing_noise_std > 0:
            logits_active = logits_active + torch.randn_like(logits_active) * self.routing_noise_std

        full_logits = torch.full((bsz, self.max_experts), float("-inf"), device=device)
        full_logits[:, : self.num_active_experts] = logits_active
        return full_logits, all_losses

    def forward(
        self,
        embedding: torch.Tensor,
        return_all_losses: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Route to top-k experts using negative reconstruction loss as logits."""
        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        bsz = embedding.size(0)
        device = embedding.device

        if self.num_active_experts == 0:
            weights = torch.ones(bsz, 1, device=device)
            indices = torch.zeros(bsz, 1, dtype=torch.long, device=device)
            return weights, indices, (None if not return_all_losses else None)

        full_logits, all_losses = self.compute_full_logits_and_losses(embedding, add_noise=True)

        k = min(int(self.top_k), int(self.num_active_experts))
        topk_logits, topk_indices = torch.topk(full_logits, k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        if return_all_losses:
            return topk_weights, topk_indices, all_losses
        return topk_weights, topk_indices, None

    def compute_reconstruction_loss(self, embedding: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)
        if self.num_active_experts == 0:
            return torch.tensor(0.0, device=embedding.device)
        if task_idx is None:
            task_idx = self.num_active_experts - 1
        if task_idx < 0 or task_idx >= self.num_active_experts:
            return torch.tensor(0.0, device=embedding.device)
        return self.autoencoders[task_idx].compute_reconstruction_loss(embedding, reduction="mean")

    @torch.no_grad()
    def update_threshold_for_current_task(self, embeddings: torch.Tensor) -> None:
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        if self.num_active_experts == 0:
            return
        ae = self.get_current_autoencoder()
        losses = ae.compute_reconstruction_loss(embeddings, reduction="none")
        ae.update_threshold(losses, k=self.threshold_k)

    @torch.no_grad()
    def detect_unseen_task(self, embedding: torch.Tensor) -> torch.Tensor:
        """True if *all* experts reject the input via their thresholds."""
        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        bsz = embedding.size(0)
        device = embedding.device
        if self.num_active_experts == 0:
            return torch.ones(bsz, dtype=torch.bool, device=device)

        is_unseen = torch.ones(bsz, dtype=torch.bool, device=device)
        for i in range(self.num_active_experts):
            ae = self.autoencoders[i]
            losses = ae.compute_reconstruction_loss(embedding, reduction="none")
            within = ae.is_within_threshold(losses)
            is_unseen = is_unseen & (~within)
        return is_unseen

    @torch.no_grad()
    def detect_unseen_from_losses(self, all_losses: torch.Tensor) -> torch.Tensor:
        """Compute unseen mask from precomputed reconstruction losses.

        all_losses must have shape [B, max_experts] (as returned by
        compute_full_logits_and_losses). Only losses for active experts are used.
        """
        if self.num_active_experts == 0:
            return torch.ones(all_losses.size(0), dtype=torch.bool, device=all_losses.device)

        is_unseen = torch.ones(all_losses.size(0), dtype=torch.bool, device=all_losses.device)
        for i in range(self.num_active_experts):
            ae = self.autoencoders[i]
            within = all_losses[:, i] <= ae.threshold
            is_unseen = is_unseen & (~within)
        return is_unseen

    def get_routing_stats(self) -> dict:
        stats = {
            "num_active_experts": int(self.num_active_experts),
            "max_experts": int(self.max_experts),
            "top_k": int(self.top_k),
            "hidden_dim": int(self.hidden_dim),
        }
        for i in range(self.num_active_experts):
            ae = self.autoencoders[i]
            stats[f"task_{i}_threshold"] = float(ae.threshold.item())
            stats[f"task_{i}_mean_loss"] = float(ae.mean_recon_loss.item())
            stats[f"task_{i}_std_loss"] = float(ae.std_recon_loss.item())
        return stats


# Backward compatible alias
MotionExpertRouter = AutoencoderRouter
