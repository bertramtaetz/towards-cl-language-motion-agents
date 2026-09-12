"""O-LoRA Multi-Adapter (Merged) Implementation.

This variant trains like O-LoRA multi-adapter (one adapter per task + orthogonality
regularizer), but *evaluates/infers* with all learned task adapters merged into the
base model weights. After merging, inference does not require task labels because
there is no adapter switching.
"""

__all__ = []
