# Two-task engineering tests

Run commands from the repository root using its own `.venv`. Install dependencies
and place the direction-specific holdout initialization and other resources as
described in the installation guide. No source repository is imported at runtime.

## Prepare and run all six methods

```bash
.venv/bin/python -B scripts/prepare_smoke.py --data-root datasets/HumanML3D --num-tasks 2 --samples 20
.venv/bin/python -B scripts/smoke_all_methods.py
```

Preparation independently copies 20 valid motions/captions for each of jumping
and arms/hands into `outputs/two_task_smoke/data`. The manifest contains exactly
two tasks. Runtime splitting preserves `random_80_20`, seed 42 (16 train and 4
test motions per task). Tests use one epoch, batch size 2, accumulation 1 and one
router-training epoch. Methods/directions run sequentially. Each train/evaluation
phase has a 1,800-second timeout; override with `--timeout`.

## Invoke each method individually

Each command below trains, reloads and evaluates both directions:

```bash
.venv/bin/python -B scripts/smoke_all_methods.py --methods multi_task
.venv/bin/python -B scripts/smoke_all_methods.py --methods transfer_learning
.venv/bin/python -B scripts/smoke_all_methods.py --methods olora_multi_adapter
.venv/bin/python -B scripts/smoke_all_methods.py --methods olora_multi_adapter_merged
.venv/bin/python -B scripts/smoke_all_methods.py --methods olora_moe
.venv/bin/python -B scripts/smoke_all_methods.py --methods olora_moe_joined
```

Append `--directions t2m` or `--directions m2t` to select one direction.

## Phases and dry-run

For any method above, use the underlying launcher for phase selection:

```bash
.venv/bin/python -B scripts/run_benchmark.py --dir t2m --methods olora_moe --num-tasks 2 --epochs 1 --router-epochs 1 --batch-size 2 --gradient-accumulation 1 --smoke-token-only --task-splits outputs/two_task_smoke/tasks.json --data-root outputs/two_task_smoke/data --output-dir outputs/two_task_smoke/runs --dry-run
```

Replace `--dry-run` with `--train-only` or `--eval-only`; change `--dir` and
`--methods` as needed. Evaluation requires the matching trained checkpoints.

## Outputs and interpretation

`outputs/two_task_smoke/report.json` records separate train/evaluation status
and token-accuracy matrices. Logs are `<direction>_<method>_<phase>.log` in the
same directory; checkpoints and raw `eval_results` JSON are under
`runs/<direction>/<method>`. A nonzero exit indicates failure; inspect the log.
Each invocation replaces the report with the selected suite's results.

Sequential approaches require two stages evaluated on both tasks (2×2 matrix).
Multi-task has one jointly trained model (1×2 matrix), not invented CL stages.
Validation requires exact task identities, finite losses and accuracies in [0,1].

**These are token-only engineering tests in both directions. They do not validate
FID, text-generation quality, SPICE, or full benchmark scores.** M2T full
generation remains blocked by SPICE runtime compatibility. Strict full-benchmark
aggregation is not run or relaxed. See [VALIDATION.md](VALIDATION.md) for execution status.

## Recorded execution

On September 12, 2026 all six methods passed in both directions (12 runs) on
the RTX 5090 using the repository-local environment. See
[two_task_results.json](two_task_results.json) for the validated token-accuracy matrices.


[Back to README](../README.md)
