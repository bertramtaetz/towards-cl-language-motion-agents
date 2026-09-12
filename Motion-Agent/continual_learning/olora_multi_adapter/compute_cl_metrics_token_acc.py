"""DEPRECATED: compute CL metrics (token accuracy).

This script has been moved to:

  continual_learning/compute_cl_metrics_token_acc.py

Reason:
  Token-accuracy CL metrics (ACC/BWT/FWT) should be computed via a single,
  method-agnostic canonical implementation to avoid ambiguity when comparing
  results across baselines, O-LoRA, and D-MoLE.

This file remains as a thin wrapper for backward compatibility.
"""

from __future__ import annotations

import sys

from continual_learning.compute_cl_metrics_token_acc import main


if __name__ == "__main__":
    # Preserve old CLI entrypoint.
    sys.exit(main())
