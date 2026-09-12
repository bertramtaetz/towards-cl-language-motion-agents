# Original motion-cluster holdout benchmark

The authoritative source is `msai-thesis`, not the master snapshot. Task order:
jumping (8), arms/hands (4), walking (2), gestures (5), sit/stand (14).
Canonical manifests are `benchmark/splits/holdout10/{source,holdout,tasks}.json`.
Each original task has 560 IDs. The generator selects 10% with seed 42 plus the
source task-index offset, removes the global holdout from all downstream tasks,
and preserves source ordering. The loader then filters valid motions/captions,
shuffles with NumPy RandomState(42), and uses floor(0.8*N) training samples.
Do not treat the JSON train/val/test lists as the effective runtime partitions.

## Regenerate and verify

```bash
.venv/bin/python -B benchmark/create_holdout_splits.py --input benchmark/splits/holdout10/source.json --out-holdout outputs/regenerated/holdout.json --out-cl outputs/regenerated/tasks.json --holdout-ratio 0.10 --seed 42
.venv/bin/python -B scripts/verify_repository.py --data-root datasets/HumanML3D
.venv/bin/python -B -m pytest -q
```

The copied motion feature/clustering/selection implementation is under
`benchmark/motion_clustering`. It extracts 1052 statistical features (mean, std,
min, max of 263 coordinates) and uses Ward clustering. Outputs default to
`outputs/motion_clustering`; set `MOTION_DATA_ROOT` for input. Full re-clustering
has not been verified against canonical cluster identities. Canonical manifest
regeneration, rather than a new clustering run, is the reproducible benchmark path.

## Invoke methods individually

Run from the repository root:

```bash
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --methods multi_task
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --methods transfer_learning
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --methods olora_multi_adapter
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --methods olora_multi_adapter_merged
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --methods olora_moe
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --methods olora_moe_joined
```

Append `--dry-run` to inspect commands, `--train-only` to train, or `--eval-only`
to evaluate existing checkpoints. For M2T replace `--dir t2m` with `--dir m2t`
and use `--train-only`; full M2T generation remains blocked by SPICE validation.
Outputs are under `outputs/<direction>/<method>` unless `--output-dir` is supplied.

## Two-task engineering tests

See [SMOKE_TESTS.md](SMOKE_TESTS.md) for all-method and individual two-task commands in both
directions, phase selection, output checks and limitations. These token-only
checks do not run full benchmark aggregation. FWT retains the source mean
pre-task score convention (not baseline-subtracted).


[Back to README](../README.md)
