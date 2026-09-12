"""O-LoRA Multi-Adapter (Merged) Training.

Training is identical to `continual_learning/olora_multi_adapter/train_olora.py`.

We keep a dedicated entrypoint so experiment orchestration can treat this as a
separate method, while checkpoints remain compatible with the merged evaluation
scripts in this folder.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Reuse the canonical O-LoRA multi-adapter training implementation.
_MOTION_AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MOTION_AGENT_ROOT))


def main() -> int:
    from continual_learning.olora_multi_adapter.train_olora import main as _main

    _main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
