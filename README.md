# Towards Continual Motion-Language Agents

Companion implementation for arXiv:2606.30266. This extraction uses the original
`msai-thesis` motion-cluster holdout benchmark, not a claim of exact paper reproduction.
The legacy O-LoRA implementation is excluded.

## Available and validated

Six approaches are included: multi-task, transfer learning, O-LoRA multi-adapter,
merged multi-adapter, MoE and joined MoE. The code includes canonical holdout
manifests, split regeneration, motion clustering, resource checks and a unified runner.

**All six methods passed two-task training, checkpoint reload and token-metric
JSON validation in both directions (12 combinations).** Sequential methods produce
2×2 stage/task matrices; multi-task produces a 1×2 joint-model result. Checks cover
exact task identities, required stages, finite losses and accuracies in [0,1].
These are **token-only engineering checks**, not generation-quality measurements.
See [recorded validation](docs/VALIDATION.md) for evidence and scope.

## Install

Run from the cloned repository root (all commands use its isolated environment):

```bash
bash scripts/install.sh
.venv/bin/python -B -m pytest -q
```

Linux, Python 3.12, PyTorch 2.10.0/CUDA 12.8, Transformers 4.44.2 and PEFT 0.15.0
are tested locally, including RTX 5090 forward/backward execution.

## Data and models

Place prepared HumanML3D `texts/` and `new_joint_vecs/` under `datasets/HumanML3D/`,
or supply `--data-root`. Obtain HumanML3D under its original dataset terms.
See [data and model preparation](docs/DATA_AND_MODELS.md) for required files and placement.
Weights are obtained separately and are excluded from Git. Never commit datasets, environments or weights.

```bash
.venv/bin/python -B scripts/resources.py verify --profile t2m
.venv/bin/python -B scripts/check_setup.py --profile t2m --cuda
```

## First run: two-task smoke suite

After preparing data and weights, run:

```bash
.venv/bin/python -B scripts/prepare_smoke.py --data-root datasets/HumanML3D --num-tasks 2 --samples 20
.venv/bin/python -B scripts/smoke_all_methods.py
```

To train, reload and evaluate just one method in one direction:

```bash
.venv/bin/python -B scripts/smoke_all_methods.py --methods olora_moe --directions t2m
```

Results are written to `outputs/two_task_smoke/report.json`. See the
[smoke-test guide](docs/SMOKE_TESTS.md) for individual commands for all six methods.

## Benchmark commands

```bash
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --dry-run
bash Motion-Agent/scripts/run_all_methods.sh --dir t2m --train-only
bash Motion-Agent/scripts/run_all_methods.sh --dir m2t --train-only
```

Methods: `multi_task`, `transfer_learning`, `olora_multi_adapter` (alias `olora`),
`olora_multi_adapter_merged`, `olora_moe`, `olora_moe_joined`. Select with `--methods`.
The runner selects the matching direction checkpoint automatically and uses
`random_80_20`, seed 42. Use the token-only smoke workflow below for validated M2T evaluation.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/INSTALLATION.md) | Environment, dependencies and diagnostics |
| [Data and models](docs/DATA_AND_MODELS.md) | Dataset layout and resource profiles |
| [Pretrained resources](pretrained/README.md) | Weight placement and provenance |
| [Benchmark](docs/BENCHMARK.md) | Regeneration and individual method commands |
| [Smoke tests](docs/SMOKE_TESTS.md) | Two-task suite, phases and result checks |
| [Validation](docs/VALIDATION.md) | Recorded checks and validation scope |

Maintainers: [extraction plan](create_repo_from_existing_repo.md) and
[source/provenance audit](docs/EXTRACTION_AUDIT.md).

## Repository layout

```text
.
├── Motion-Agent/
│   ├── baselines/              # Multi-task and transfer learning
│   ├── continual_learning/     # Four LoRA approaches and shared metrics
│   ├── models/                 # Motion-language model and VQ-VAE
│   └── scripts/                # Unified shell entrypoint
├── benchmark/                  # Motion clustering and holdout manifests
├── scripts/                    # Install, resources, launch and smoke tools
├── tests/                      # Repository and metric validation
├── docs/                       # Detailed guides and validation evidence
├── pretrained/                 # Local resources; weights excluded from Git
├── datasets/                   # User-provided data (not tracked)
├── outputs/                    # Generated results (not tracked)
├── resources.json              # Resource profiles and checksums
├── requirements.txt
└── LICENSE
```

## Citation

If you use this implementation, please cite:

```bibtex
@misc{taetz2026continualmotionlanguage,
  title         = {Towards Continual Motion-Language Agents: LoRA Variants for Incremental Motion Understanding and Generation},
  author        = {Taetz, Bertram and Albuquerque Cosme da Silva, Hugo and Bleser-Taetz, Gabriele},
  year          = {2026},
  eprint        = {2606.30266},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2606.30266}
}
```

## License

Original project contributions are provided under the **[MIT License](LICENSE)**,
copyright 2026 bertramtaetz. Third-party components retain their accompanying
licenses and notices. Model weights and datasets are governed by their respective
terms; the root MIT license does not replace those terms.

See [third-party notices](THIRD_PARTY_NOTICES.md), [licensing and upstream model links](docs/LICENSING.md), and [Google Drive packaging](docs/GOOGLE_DRIVE.md).
