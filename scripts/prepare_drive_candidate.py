"""Prepare local review archives, not approved public distributions or uploads.

The approved-release gate in package_resources.py remains unchanged. This tool
builds a separately labelled candidate for inspection before licensing approval.
"""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from package_resources import digest, local_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'motion-agent-resource-v1'
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing candidate: {out}')
    records = json.loads((ROOT / 'resources.json').read_text())['files']
    groups = {
        'holdout-initialization-v1': [r for r in records if r['path'] in {
            'pretrained/holdout10/t2m/multitask_final.pth',
            'pretrained/holdout10/m2t/multitask_final.pth'}],
        'motion-resources-v1': [r for r in records if
            r['path'] == 'pretrained/ckpt/vqvae.pth' or
            r['path'].startswith(('pretrained/checkpoints/', 'pretrained/glove/'))],
    }
    if len(groups['holdout-initialization-v1']) != 2 or not groups['motion-resources-v1']:
        raise SystemExit('Incomplete candidate resource selection')
    for entries in groups.values():
        for entry in entries:
            path = local_file(ROOT, entry['path'])
            if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
                raise SystemExit(f'Resource mismatch: {path}')
    out.mkdir()
    shutil.copyfile(ROOT / 'docs/DRIVE_BUNDLE_README.md', out / 'BUNDLE_README.md')
    shutil.copyfile(ROOT / 'resources.json', out / 'resources.json')
    inventory = {'status': 'local_candidate_pending_redistribution_review', 'bundles': {}}
    checksums = []
    for name, entries in groups.items():
        archive = out / f'{name}.tar.gz'
        partial = out / f'{name}.partial'
        with tarfile.open(partial, 'w:gz', compresslevel=1) as tar:
            for entry in entries:
                tar.add(local_file(ROOT, entry['path']), arcname=entry['path'], recursive=False)
            # Keep notices under a bundle namespace: extraction must not overwrite code.
            for notice in ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/LICENSING.md',
                           'Motion-Agent/LICENSE', 'distribution_policy.json', 'resources.json']:
                path = local_file(ROOT, notice)
                tar.add(path, arcname=f'bundle_metadata/{name}/{notice}', recursive=False)
            tar.add(out / 'BUNDLE_README.md', arcname=f'bundle_metadata/{name}/BUNDLE_README.md')
        partial.rename(archive)
        checksum = digest(archive)
        line = f'{checksum}  {archive.name}\n'
        (out / f'{archive.name}.sha256').write_text(line)
        checksums.append(line)
        inventory['bundles'][name] = {
            'archive': archive.name, 'sha256': checksum,
            'archive_bytes': archive.stat().st_size,
            'resource_bytes': sum(e['size'] for e in entries),
            'files': entries,
        }
        print(f'Prepared {archive}', flush=True)
    (out / 'SHA256SUMS').write_text(''.join(checksums))
    (out / 'bundle_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print('Local candidate only: obtain redistribution approval before public upload.')


if __name__ == '__main__':
    main()