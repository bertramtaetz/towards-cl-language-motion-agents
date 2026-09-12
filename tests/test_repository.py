"""Fast tests which do not load pretrained weights or require CUDA."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def test_python_compilation():
    for folder in ["Motion-Agent", "benchmark", "scripts", "tests"]:
        for path in (ROOT / folder).rglob("*.py"):
            compile(path.read_text(), str(path), "exec")


def test_canonical_splits():
    spec = importlib.util.spec_from_file_location("verify", ROOT / "scripts/verify_repository.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_splits(ROOT / "benchmark/splits/holdout10")


def test_regeneration(tmp_path):
    subprocess.run([sys.executable, "-B", str(ROOT / "benchmark/create_holdout_splits.py"),
                    "--input", str(ROOT / "benchmark/splits/holdout10/source.json"),
                    "--out-holdout", str(tmp_path / "holdout.json"),
                    "--out-cl", str(tmp_path / "tasks.json"),
                    "--holdout-ratio", "0.10", "--seed", "42"], check=True)
    for name in ["tasks.json", "holdout.json"]:
        assert json.loads((tmp_path / name).read_text()) == json.loads(
            (ROOT / "benchmark/splits/holdout10" / name).read_text())


def test_all_method_dry_runs():
    for direction in ["t2m", "m2t"]:
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_benchmark.py"),
                                 "--dir", direction, "--dry-run"], capture_output=True, text=True, check=True)
        assert "--split-mode random_80_20" in result.stdout
        assert "--generate-all-stages" in result.stdout
        assert "continual_learning/olora/" not in result.stdout
        assert "olora_multi_adapter/train_olora.py" in result.stdout