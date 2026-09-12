import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke', ROOT / 'scripts/smoke_all_methods.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def results(tmp_path, stages):
    directory = tmp_path / 'eval_results'
    directory.mkdir()
    for stage in stages:
        (directory / f'after_task{stage}_test.json').write_text(json.dumps({
            'per_task': {task: {'loss': 1.0, 'accuracy': .5} for task in ['a', 'b']}}))


def test_sequential_matrix(tmp_path):
    results(tmp_path, [0, 1])
    assert smoke.validate_results(tmp_path, ['a', 'b'], 'olora_moe')['accuracy_matrix'] == [[.5, .5], [.5, .5]]


def test_multitask_is_not_sequential(tmp_path):
    results(tmp_path, [1])
    assert smoke.validate_results(tmp_path, ['a', 'b'], 'multi_task')['stages'] == [1]
    with pytest.raises(FileNotFoundError):
        smoke.validate_results(tmp_path, ['a', 'b'], 'transfer_learning')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 2, True])
def test_invalid_accuracy_fails(tmp_path, value):
    results(tmp_path, [1])
    path = tmp_path / 'eval_results/after_task1_test.json'
    data = json.loads(path.read_text())
    data['per_task']['a']['accuracy'] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        smoke.validate_results(tmp_path, ['a', 'b'], 'multi_task')