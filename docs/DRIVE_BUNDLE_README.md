# Motion-Agent resource bundle v1

**Local release candidate: redistribution review is pending. Do not publicly
upload these candidate archives until the maintainer has completed the review
in the repository's licensing guide.** Included code notices do not license the
weights. Add all required asset-specific agreements/notices and rebuild approved
archives before public distribution. This file must be updated to reflect that
completed review and the final compatible code commit/tag.

Intended download folder:
<https://drive.google.com/drive/folders/1sZdxi83YWXWYwtpFwXYBrOUNqPiBb42P?usp=drive_link>

## Contents

- `holdout-initialization-v1.tar.gz`: separate T2M/M2T holdout initializations.
- `motion-resources-v1.tar.gz`: VQ-VAE, motion evaluators/configuration,
  normalization arrays and GloVe files listed in the resource manifest.
- Individual `.sha256` files and `SHA256SUMS`: archive integrity checks.
- `resources.json`: full code resource manifest (includes resources not bundled).
- `bundle_inventory.json`: exact payload, per-file hashes and actual sizes.
- `bundle_metadata/` inside archives: provenance/policy snapshots and notices,
  kept separate from the code so extraction does not overwrite its manifests.

These resources belong to the original motion-cluster holdout workflow. They
are neither the excluded semantic benchmark initialization nor trained final
checkpoints for each method. Gemma, RoBERTa, NLG/Java and HumanML3D are not bundled.
Obtain them using the code repository's data/model and licensing guides.

## Install downloaded resources

Use a checkout of the corresponding code release. The commands below assume
you are in its root and downloaded the files into its ignored `downloads/`
directory. Download the two archives directly; if Drive wraps multiple downloads
in a ZIP, unpack that wrapper into `downloads/` first.

```bash
# Verify both archive downloads; do not extract if this fails.
(cd downloads && sha256sum -c SHA256SUMS)

# Inspect before extracting: entries should be under pretrained/ or bundle_metadata/.
tar -tzf downloads/holdout-initialization-v1.tar.gz
tar -tzf downloads/motion-resources-v1.tar.gz

# Protect existing files: use a fresh checkout or relocate conflicting resources.
tar --keep-old-files -xzf downloads/holdout-initialization-v1.tar.gz
tar --keep-old-files -xzf downloads/motion-resources-v1.tar.gz

# After installing the environment and obtaining the upstream backbone:
.venv/bin/python -B scripts/resources.py verify --profile t2m
.venv/bin/python -B scripts/check_setup.py --profile t2m --cuda
```

The extracted model layout is:

```text
pretrained/
├── holdout10/t2m/multitask_final.pth
├── holdout10/m2t/multitask_final.pth
├── ckpt/vqvae.pth
├── checkpoints/
└── glove/
```

These two archives alone are not a complete environment or dataset. Resource
verification reports separately required upstream files until they are installed.
Follow the repository's smoke-test guide after setup. Never load checkpoints from
an untrusted source; checksums detect corruption, not the trustworthiness of the publisher.