"""Package explicitly reviewed resources only; never uploads files."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_file(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'Unsafe path: {name}')
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Not a local file: {name}')
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f'Symlink not allowed: {name}')
    return path


def validate_bundle(root, name):
    policy = json.loads((root / 'distribution_policy.json').read_text())
    bundle = policy['bundles'].get(name)
    if not bundle or bundle.get('status') != 'approved' or not bundle.get('review', '').strip():
        raise ValueError('Bundle has no documented redistribution approval')
    if not bundle.get('files') or not bundle.get('notices'):
        raise ValueError('Explicit resources and accompanying notices are required')
    manifest = {f['path']: f for f in json.loads((root / 'resources.json').read_text())['files']}
    paths = []
    for item in bundle['files']:
        if any(item.startswith(prefix) for prefix in policy.get('upstream_only', [])):
            raise ValueError(f'Upstream-only resource: {item}')
        path = local_file(root, item)
        record = manifest.get(item)
        if not record or path.stat().st_size != record['size'] or digest(path) != record['sha256']:
            raise ValueError(f'Resource manifest mismatch: {item}')
        paths.append(path)
    paths.extend(local_file(root, item) for item in bundle['notices'])
    paths.extend(local_file(root, item) for item in ['resources.json', 'distribution_policy.json'])
    return list(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true')
    group.add_argument('--bundle')
    args = parser.parse_args()
    policy = json.loads((ROOT / 'distribution_policy.json').read_text())
    if args.list:
        print(json.dumps(policy, indent=2))
        return
    if not args.bundle or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.bundle):
        parser.error('Bundle name must contain only letters, digits, hyphens and underscores')
    try:
        paths = validate_bundle(ROOT, args.bundle)
    except ValueError as exc:
        parser.error(str(exc))
    out = ROOT / 'release_bundles'
    out.mkdir(exist_ok=True)
    archive = out / f'{args.bundle}.tar.gz'
    if archive.exists():
        parser.error('Archive already exists; use a new version')
    staging = out / f'{args.bundle}.partial'
    try:
        with tarfile.open(staging, 'w:gz') as tar:
            for path in paths:
                tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        staging.rename(archive)
    finally:
        staging.unlink(missing_ok=True)
    archive.with_suffix(archive.suffix + '.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    print(archive)


if __name__ == '__main__':
    main()