"""O-LoRA MoE (olora-MoE): task-id free routing over per-task O-LoRA adapters.

This module combines:
  - O-LoRA multi-adapter training (one LoRA adapter per task + orthogonality regularizer)
  - A Mixture-of-Experts style router that selects the adapter from the input
    (task-id free inference)

Design goals:
  - Keep the strong CL performance of O-LoRA multi-adapter (ACC/BWT/FWT)
  - Avoid the performance degradation of the merged-adapter approach
  - Do NOT require task IDs at inference time

Implementation note:
  We keep per-task PEFT adapters (adapter_name='task_{t}') and use an
  autoencoder-based router (reconstruction loss) to predict the best adapter.
"""

from .router import AutoencoderRouter

__all__ = ["AutoencoderRouter"]
