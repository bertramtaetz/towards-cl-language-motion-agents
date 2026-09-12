"""Report installation/resource readiness; never declares paper reproduction."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["t2m", "m2t", "benchmark-generation", "pretraining"], default="t2m")
    parser.add_argument("--cuda", action="store_true", help="Run a small real forward/backward GPU check")
    parser.add_argument("--nlg", action="store_true", help="Run real NLG scorers; writes outputs/smoke/nlg_metrics.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/HumanML3D")
    args = parser.parse_args()
    failures = []
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        failures.append("Not using repository-local .venv")
    commands = [[sys.executable, "-B", str(ROOT / "scripts/resources.py"), "verify", "--profile", args.profile],
                [sys.executable, "-B", str(ROOT / "scripts/verify_repository.py"), "--data-root", str(args.data_root.resolve())]]
    if args.cuda:
        commands.append([sys.executable, "-B", "-c", "import torch; x=torch.ones(4,device='cuda',requires_grad=True); (x*x).sum().backward(); assert torch.isfinite(x.grad).all(); print(torch.__version__, torch.cuda.get_device_name())"])
    if args.nlg:
        commands.append([sys.executable, "-B", str(ROOT / "scripts/smoke_nlg.py")])
    for command in commands:
        try:
            result = subprocess.run(command, cwd=ROOT, timeout=600)
            if result.returncode:
                failures.append("Failed: " + " ".join(command))
        except subprocess.TimeoutExpired:
            failures.append("Timed out: " + " ".join(command))
    if args.profile == "m2t":
        failures.append("Unified M2T evaluation remains blocked pending validated generation/SPICE")
    if args.profile == "pretraining":
        failures.append("Pretraining checkpoint lineage remains unverified")
    print(json.dumps({"status": "NOT READY" if failures else "REQUESTED CHECKS PASSED",
                      "failures": failures, "cuda_checked": args.cuda, "nlg_checked": args.nlg,
                      "publication_reproduction_validated": False}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())