"""Inspect, verify, or download release resources without checkpoint substitution.

Only maintainer-approved HTTPS file URLs belong in resources.json. Google Drive
HTML/share links are not direct file URLs; HTML responses fail checksum validation.
Archives are deliberately not extracted by this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def destination(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe resource path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Resource escapes repository: {relative}")
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"Symlink resource path: {relative}")
    return path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def valid(path, entry):
    return (path.is_file() and entry.get("sha256") is not None
            and path.stat().st_size == entry.get("size")
            and digest(path) == entry["sha256"])


def download(root, entry):
    target = destination(root, entry["path"])
    if valid(target, entry):
        return
    url = entry.get("url")
    if not url or not url.startswith("https://") or not entry.get("sha256") or entry.get("size") is None:
        raise ValueError("No approved direct HTTPS URL, size and checksum; install manually")
    # Never overwrite an existing file of unknown provenance.
    if target.exists():
        raise ValueError(f"Existing invalid file must be reviewed manually: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".resource-", dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("Refusing non-HTTPS redirect")
            count = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                count += len(block)
                if count > entry["size"]:
                    raise ValueError("Response exceeds expected size (possibly a download error page)")
                out.write(block)
        if not valid(temporary, entry):
            raise ValueError("Downloaded bytes do not match the manifest")
        # Exclusive creation prevents overwriting a file created during the download.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "verify", "download"])
    parser.add_argument("--profile", choices=["all", "t2m", "m2t", "benchmark-generation", "pretraining"], default="all")
    args = parser.parse_args()
    entries = json.loads((ROOT / "resources.json").read_text())["files"]
    failures = []
    for entry in entries:
        if args.profile != "all" and args.profile not in entry["profiles"]:
            continue
        try:
            path = destination(ROOT, entry["path"])
            if args.action == "list":
                print(entry["path"], "present" if path.is_file() else "missing",
                      "download configured" if entry.get("url") else "manual / release pending")
                continue
            if args.action == "download":
                download(ROOT, entry)
            if not valid(path, entry):
                raise ValueError("missing, unverified, wrong size or wrong checksum")
            print("PASS", entry["path"])
        except (ValueError, OSError) as exc:
            failures.append(entry["path"])
            print("NOT READY", entry["path"], str(exc))
    print(f"{len(failures)} resource failures; file integrity alone does not establish runtime readiness")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())