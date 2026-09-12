"""Two-task train/reload/token-evaluation checks; never publication metrics."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
METHODS = ['multi_task', 'transfer_learning', 'olora_multi_adapter',
           'olora_multi_adapter_merged', 'olora_moe', 'olora_moe_joined']


def validate_results(directory, tasks, method):
    stages = [1] if method == 'multi_task' else [0, 1]
    matrix = []
    for stage in stages:
        path = directory / 'eval_results' / f'after_task{stage}_test.json'
        result = json.loads(path.read_text())
        if set(result['per_task']) != set(tasks):
            raise ValueError(f'{path}: expected exactly the two smoke tasks')
        row = []
        for task in tasks:
            values = result['per_task'][task]
            for key in ['loss', 'accuracy']:
                value = values[key]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f'{path}: invalid {task}/{key}')
            if not 0 <= values['accuracy'] <= 1:
                raise ValueError(f'{path}: accuracy outside [0, 1]')
            row.append(values['accuracy'])
        matrix.append(row)
    return {'stages': stages, 'task_names': tasks, 'accuracy_matrix': matrix,
            'engineering_smoke_only': True, 'generation_metrics': 'NOT TESTED'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--directions', nargs='+', choices=['t2m', 'm2t'], default=['t2m', 'm2t'])
    p.add_argument('--timeout', type=int, default=1800, help='Seconds per train or evaluation phase')
    args = p.parse_args()
    root = ROOT / 'outputs/two_task_smoke'
    tasks_file = root / 'tasks.json'
    manifest = json.loads(tasks_file.read_text())
    tasks = [v['name'] for v in manifest['tasks'].values()]
    if len(tasks) != 2 or not manifest.get('engineering_smoke_only'):
        p.error('Run prepare_smoke.py --num-tasks 2 first')
    report = {'engineering_smoke_only': True, 'generation_metrics': 'NOT TESTED', 'runs': {}}
    report_file = root / 'report.json'
    failed = False
    for direction in args.directions:
        for method in args.methods:
            key = f'{direction}/{method}'
            entry = report['runs'][key] = {}
            common = [sys.executable, '-B', str(ROOT / 'scripts/run_benchmark.py'),
                      '--dir', direction, '--methods', method, '--num-tasks', '2',
                      '--epochs', '1', '--router-epochs', '1', '--batch-size', '2',
                      '--gradient-accumulation', '1', '--smoke-token-only',
                      '--task-splits', str(tasks_file), '--data-root', str(root / 'data'),
                      '--output-dir', str(root / 'runs')]
            for phase in ['train', 'eval']:
                log = root / f'{direction}_{method}_{phase}.log'
                try:
                    with log.open('w') as stream:
                        subprocess.run([*common, f'--{phase}-only'], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       check=True, timeout=args.timeout)
                    entry[phase] = 'PASSED'
                    if phase == 'eval':
                        entry['metrics'] = validate_results(root / 'runs' / direction / method, tasks, method)
                except Exception as error:
                    entry[phase] = 'FAILED'
                    entry['error'] = str(error)
                    failed = True
                    break
                finally:
                    report_file.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(key, entry, flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())