# Data and model distribution

Keep code, manifests, tests and documentation in the existing GitHub repository.
Keep weights out of Git. Publish versioned Google Drive bundles with SHA-256
checksums, provenance, release identifier and exact extraction layout. URLs in
`resources.json` intentionally remain unset until actual approved downloads exist.

Required direction initialization:
- `pretrained/holdout10/t2m/multitask_final.pth`
- `pretrained/holdout10/m2t/multitask_final.pth`

Both were independently copied from the original minimal_holdout10_seed42_r0.10
experiment and hash-verified. They are not renamed original MotionLLM weights.
Other resources: `pretrained/gemma-2-2b-it`, `ckpt/vqvae.pth`, `checkpoints`, `glove`,
`roberta-large` and `nlg`. The optional original `ckpt/motionllm.pth` is not the
benchmark default. Semantic embedding weights are not required.

Prefer separately versioned initialization, motion-evaluation and NLG bundles.
Download upstream Gemma and RoBERTa under their own terms with pinned revisions;
review all redistribution terms before making any local copies public.
`resources.json` drives both download and hash verification. Profiles select
workflow requirements; no tool silently substitutes a different initialization.

```bash
.venv/bin/python -B scripts/resources.py list --profile t2m
.venv/bin/python -B scripts/resources.py verify --profile t2m
```

Prepared HumanML3D belongs in `datasets/HumanML3D/{texts,new_joint_vecs}`.
The full dataset is not bundled. The source-machine dataset was checked read-only;
use the smoke preparation command to copy a small independent subset for testing.


[Back to README](../README.md)

Before mirroring or redistributing resources, read [licensing and upstream acquisition](LICENSING.md) and [Google Drive packaging](GOOGLE_DRIVE.md). Local copies are not redistribution approvals.
