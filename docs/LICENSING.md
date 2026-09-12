# Licensing and upstream acquisition

Reviewed: September 12, 2026. This inventory is a conservative release policy,
not a legal certification. Local possession and successful tests do not establish
redistribution permission. Use exact version/revision terms when releasing files.

| Local resource (repository-relative) | Upstream / applicable terms | Initial distribution policy |
|---|---|---|
| `pretrained/gemma-2-2b-it/` | [model](https://huggingface.co/google/gemma-2-2b-it), [Gemma terms](https://ai.google.dev/gemma/terms) | Obtain upstream at the manifest's revision; do not mirror initially |
| `pretrained/holdout10/` and `pretrained/ckpt/motionllm.pth` | Gemma-derived initialization; [Gemma terms](https://ai.google.dev/gemma/terms) plus training/source provenance | Review checkpoint contents and rights before any upload; not MIT weights |
| `pretrained/roberta-large/` | [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large), upstream identifies MIT | Prefer pinned upstream download; this is not a claim that mirroring is prohibited |
| `pretrained/ckpt/vqvae.pth` | [Motion-Agent](https://github.com/szqwu/Motion-Agent), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) | Verify exact weight provenance and permissions separately from code licensing |
| `pretrained/checkpoints/` | [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [Motion-Agent](https://github.com/szqwu/Motion-Agent) | Review each evaluator weight, normalization array and configuration before mirroring |
| `pretrained/glove/` | [GloVe](https://nlp.stanford.edu/projects/glove/), upstream motion evaluation preparation | Trace these processed vocabulary files and retain relevant terms; do not treat source-code licensing as weight permission |
| `pretrained/nlg/` | [SPICE](https://github.com/peteanderson80/SPICE), [CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html), [METEOR](https://www.cs.cmu.edu/~alavie/METEOR/) | Upstream acquisition; mixed-license JARs/data must not be blanket-relicensed as MIT |
| `datasets/` | [HumanML3D acquisition](https://github.com/EricGuo5513/HumanML3D), including its underlying data requirements | User acquisition, not part of Google Drive release |

## Gemma-derived checkpoint release

Review Section 3.1 of the linked Gemma agreement before distribution. It requires
the agreement to accompany distributions, applicable use restrictions and notice
to recipients, prominent modification notices, and the prescribed Notice text.
Download the authoritative agreement and reproduce its required Notice directly
from that agreement in the approved bundle. This guide does not replace either.
Do not assume that saving only adapters removes model-derivative obligations.
Also review the training data and ownership of the fine-tuning contributions.

## Java and other nested dependencies

[Stanford's licensing description](https://nlp.stanford.edu/software/corenlp.shtml)
distinguishes its GPL code from the composite distribution. Inspect the actual
3.6.0 distribution and each included library: current upstream descriptions alone
do not settle historical-version compliance. If redistribution is approved,
preserve license texts, notices and fulfill applicable source-distribution/source
availability requirements; attribution alone is insufficient. METEOR, SPICE,
CoreNLP models and supporting JARs need individual review.

## Release gate

`distribution_policy.json` intentionally has no approved binary bundles initially.
Record exact file paths, checksums (in `resources.json`), review rationale and
included license/notice files before adding an approved bundle. The packaging
tool rejects missing approvals, missing notices, external paths and altered files.
Approval is a maintainer decision based on evidence, not a checkbox that grants
rights. Retain [third-party code notices](../THIRD_PARTY_NOTICES.md).

[Back to README](../README.md)