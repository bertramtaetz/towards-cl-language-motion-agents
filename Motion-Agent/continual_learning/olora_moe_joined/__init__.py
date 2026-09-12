"""Joined variant of O-LoRA MoE.

This method reuses the AutoencoderRouter + unseen-threshold fallback logic from
`continual_learning.olora_moe`, but changes the T2M routing embedding to match the
caption-only embedding strategy used in the other branch experiments.
"""
