# Preparing Google Drive downloads

Do not upload the entire local `pretrained/` tree. Upload only bundles approved
under the [licensing policy](LICENSING.md). No binary bundle is approved yet.

Local upload candidate folder: `motion-agent-resource-v1/` (excluded from Git).

Designated [Google Drive folder](https://drive.google.com/drive/folders/1sZdxi83YWXWYwtpFwXYBrOUNqPiBb42P?usp=drive_link).
This link identifies the owner's folder; it does not establish that downloads
are uploaded, public, or licensed for redistribution.

The local candidate contains both archives below, archive checksum files,
`SHA256SUMS`, a matching `resources.json`, `bundle_inventory.json`, a policy
snapshot, third-party notices, and `BUNDLE_README.md`. Its model payloads are
verified against the resource manifest. Candidate preparation does not approve
redistribution. Complete the review and required notices before copying the
folder contents to public Drive. The approved-release packager remains gated.

## Installing downloaded archives

Download both archives and `SHA256SUMS` into the same directory. In that directory:

```bash
sha256sum -c SHA256SUMS
tar -tzf holdout-initialization-v1.tar.gz
tar -tzf motion-resources-v1.tar.gz
```

After inspecting trusted archive contents, change to the companion repository
root and extract each archive (replace the download paths):

```bash
tar --keep-old-files -xzf /absolute/download/path/holdout-initialization-v1.tar.gz
tar --keep-old-files -xzf /absolute/download/path/motion-resources-v1.tar.gz
.venv/bin/python scripts/resources.py verify --profile t2m
```

The archives restore `pretrained/holdout10/`, `pretrained/ckpt/vqvae.pth`,
`pretrained/checkpoints/`, and `pretrained/glove/`; they do not overwrite code or
repository manifests. `--keep-old-files` reports an error instead of replacing
existing files. Verify existing resources before deciding whether to replace
them. Install the environment and separate upstream resources according to
[Installation](INSTALLATION.md) and [Licensing](LICENSING.md). For M2T use
`--profile m2t`; the bundles alone do not satisfy upstream model/NLG dependencies.
Then follow [Two-task smoke tests](SMOKE_TESTS.md).

| Candidate bundle | Repository-relative payload | Release condition |
|---|---|---|
| `holdout-initialization-v1` | `pretrained/holdout10/t2m/multitask_final.pth`, `pretrained/holdout10/m2t/multitask_final.pth` | Checkpoint provenance/derivative review and accompanying Gemma terms/notices |
| `motion-resources-v1` | `pretrained/ckpt/vqvae.pth`, manifest-listed `pretrained/checkpoints/` and `pretrained/glove/` files | Individual asset redistribution review and notices |

Obtain Gemma and RoBERTa upstream using the links in [LICENSING.md](LICENSING.md).
Keep NLG/Java binaries out of the initial Drive release. Do not upload HumanML3D,
unused GTE weights, environments, caches, smoke outputs or original initialization
as a substitute for the holdout initialization.

## Packaging

Record each approved bundle in `distribution_policy.json` with `status` equal to
`approved`, a nonempty `review`, an explicit `files` list of resource-manifest
paths and a nonempty `notices` list of repository-relative license/notice files.
The list must enumerate files, not directories or globs. Do not set approval
without completing its licensing obligations.

```bash
.venv/bin/python scripts/package_resources.py --list
# Once a reviewed bundle exists:
.venv/bin/python scripts/package_resources.py --bundle holdout-initialization-v1
```

The tool writes into ignored `release_bundles/`: an archive with repository-root
paths, an archive SHA-256 file, and includes the matching resource manifest,
policy and required notices inside the archive. Upload the archive and checksum
along with a short bundle README naming the code tag, extraction procedure,
provenance, licenses, archive/extracted sizes and verified workflow.

Users download manually, verify with `sha256sum -c`, inspect the archive and
extract into the repository root, then run `scripts/resources.py verify` using
the repository environment. Existing valid local assets should be preserved.
The resource downloader handles direct individual HTTPS files, not Drive share
pages or archives; do not place an archive share link in per-file URL fields.

[Back to README](../README.md)