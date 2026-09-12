# Validation: original-source holdout benchmark

Updated September 12, 2026. This report supersedes earlier benchmark reports.
Only the original `msai-thesis` motion-cluster holdout benchmark is supported.

## Verified
- All 24 tests passed; one integration test excluded. See tests_local_env.txt.
- All copied Python files parse; installer and runner shell syntax passes.
- Canonical 10% holdout regeneration matches the copied JSON exactly.
- Selected source hashes and copied asset hashes pass (repository_checks.txt).
- All referenced dataset files exist in the inspected source dataset. They are
  not automatically installed at the destination's default dataset location.
- Direction-specific minimal holdout initialization weights are copied under
  pretrained/holdout10/t2m and pretrained/holdout10/m2t; no missing initialization
  is required for the current benchmark.
- The actual loader passes an independent deterministic 80/20 split test using
  seed 42, including sample ordering and train/test disjointness.
- Smoke preparation successfully copied 20 valid examples per task into local
  outputs; the loader uses 16 training and four test examples per task.
- Real T2M multi-adapter training completed one epoch on one task with the
  holdout initialization and saved a checkpoint. See holdout_training_t2m.txt.
- All execution used the destination .venv (PyTorch 2.10.0/CUDA 12.8).

## Not yet validated
- Generation-quality smoke cycles across all methods remain untested. The new
  two-task suite validates token metrics only. Older generation logs used a
  different initialization and are historical evidence only.
- Full clustering regeneration and correspondence to publication tables.
- SPICE compatibility: Java 21 failed; full M2T evaluation remains guarded.
- Full dataset installation at the default destination, clean-machine install,
  public model downloads, and full benchmarks.

Metric fixture tests verify aggregate JSON and reject incomplete/non-finite
inputs. These tests are not evidence that all generation paths work. Source
FWT remains mean pre-task performance, not baseline-subtracted transfer.
Nothing has been pushed; source repositories were not modified.

## Two-task GPU suite — September 12, 2026

All 12 method/direction combinations passed training, checkpoint reload, and
token-metric JSON validation using the destination environment and the original
direction-specific holdout initialization. Evidence: [two_task_results.json](two_task_results.json)
and [two_task_execution.txt](two_task_execution.txt); reproduction commands: [SMOKE_TESTS.md](SMOKE_TESTS.md).

| Method | T2M | M2T | Matrix |
|---|---|---|---|
| multi_task | PASSED | PASSED | 1×2 joint-model results |
| transfer_learning | PASSED | PASSED | 2×2 |
| olora_multi_adapter | PASSED | PASSED | 2×2 |
| olora_multi_adapter_merged | PASSED | PASSED | 2×2 |
| olora_moe | PASSED | PASSED | 2×2 |
| olora_moe_joined | PASSED | PASSED | 2×2 |

Each task contains 16 training and four test motions. One training epoch and one
router epoch exercise the pipeline, not convergence or benchmark quality.
Both MoE methods saved task routers and performed routed evaluation. Validation
checks exact task identities, finite losses, finite accuracies within [0,1],
and all required stages. Generation metrics are explicitly NOT TESTED.


[Back to README](../README.md)
