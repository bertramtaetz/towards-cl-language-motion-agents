#!/usr/bin/env python3
"""Dependency-free checks of source preservation, assets and canonical splits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_splits(directory, assignments=None):
    tasks = json.loads((directory / "tasks.json").read_text())
    holdout = json.loads((directory / "holdout.json").read_text())
    excluded = set(holdout["holdout_global_ids"])
    assert excluded == set(tasks["excluded_holdout_global_ids"])
    assert [t["cluster_id"] for t in tasks["tasks"].values()] == [8, 4, 2, 5, 14]
    used = set()
    for task in tasks["tasks"].values():
        ids = [mid for split in ("train", "val", "test") for mid in task[split]]
        assert len(ids) == len(set(ids))
        assert excluded.isdisjoint(ids)
        assert used.isdisjoint(ids)
        used.update(ids)
    return used | excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hashes", action="store_true", help="Local extraction audit only")
    parser.add_argument("--asset-hashes", action="store_true")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    count = 0
    for folder in ["Motion-Agent", "benchmark", "scripts", "tests"]:
        for path in (ROOT / folder).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
            compile(path.read_text(), str(path), "exec")
            count += 1
        assert not any(p.is_symlink() for p in (ROOT / folder).rglob("*"))
    assert not (ROOT / "Motion-Agent/continual_learning/olora").exists()
    for directory in sorted((ROOT / "benchmark/splits").iterdir()):
        ids = verify_splits(directory)
        if args.data_root:
            missing = [mid for mid in ids if not (args.data_root / "texts" / f"{mid}.txt").is_file()
                       or not (args.data_root / "new_joint_vecs" / f"{mid}.npy").is_file()]
            assert not missing, missing[:10]
        print("PASS split checks:", directory.name)
    if args.source_hashes:
        for entry in json.loads((ROOT / "docs/source_manifest.json").read_text()):
            assert sha256(Path(entry["source"])) == entry["sha256"], entry["source"]
        print("PASS selected source files unchanged")
    if args.asset_hashes:
        for entry in json.loads((ROOT / "pretrained/manifest.json").read_text()):
            path = ROOT / "pretrained" / entry["path"]
            assert not path.is_symlink()
            assert path.stat().st_size == entry["size"]
            assert sha256(path) == entry["sha256"], str(path)
        print("PASS copied asset hashes")
    print(f"PASS Python parsing: {count} files")


if __name__ == "__main__":
    main()