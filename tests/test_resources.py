"""Offline tests of safe release-resource handling."""
import importlib.util
import hashlib
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("resources", Path(__file__).resolve().parents[1] / "scripts/resources.py")
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def test_reject_unsafe_paths(tmp_path):
    for name in ["../escape", "/tmp/escape"]:
        with pytest.raises(ValueError):
            resources.destination(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        resources.destination(tmp_path, "link/file")


def test_preserve_valid_file_and_reject_unconfigured_download(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"known")
    entry = dict(path="weights", size=5, sha256=hashlib.sha256(b"known").hexdigest(), url=None)
    resources.download(tmp_path, entry)
    assert path.read_bytes() == b"known"
    assert resources.valid(path, entry)
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        resources.download(tmp_path, entry)
    assert path.read_bytes() == b"known"