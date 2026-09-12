import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('packaging_tool', Path(__file__).resolve().parents[1] / 'scripts/package_resources.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def fixture(root):
    (root / 'weight').write_bytes(b'test')
    (root / 'LICENSE').write_text('fixture notice')
    (root / 'resources.json').write_text(json.dumps({'files': [{'path': 'weight', 'size': 4, 'sha256': hashlib.sha256(b'test').hexdigest()}]}))
    policy = {'bundles': {'test': {'status': 'approved', 'review': 'fixture only', 'files': ['weight'], 'notices': ['LICENSE']}}, 'upstream_only': []}
    (root / 'distribution_policy.json').write_text(json.dumps(policy))
    return policy


def test_approved_files_include_notices(tmp_path):
    fixture(tmp_path)
    assert {p.name for p in tool.validate_bundle(tmp_path, 'test')} == {'weight', 'LICENSE', 'resources.json', 'distribution_policy.json'}


@pytest.mark.parametrize('case', ['unknown', 'modified', 'missing_notice', 'upstream', 'unapproved'])
def test_release_guards(tmp_path, case):
    policy = fixture(tmp_path)
    if case == 'modified':
        (tmp_path / 'weight').write_bytes(b'bad!')
    if case == 'missing_notice':
        policy['bundles']['test']['notices'] = []
    if case == 'upstream':
        policy['upstream_only'] = ['weight']
    if case == 'unapproved':
        policy['bundles']['test']['status'] = 'review'
    (tmp_path / 'distribution_policy.json').write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        tool.validate_bundle(tmp_path, 'absent' if case == 'unknown' else 'test')


def test_external_paths_rejected(tmp_path):
    with pytest.raises(ValueError):
        tool.local_file(tmp_path, '../outside')