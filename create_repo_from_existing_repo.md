# Extraction plan: original-source holdout benchmark

Approved scope: build directly in this existing Git repository; never modify either
source tree, reuse a source environment, or push automatically. Source authority:
`../msai-thesis`. Exclude the master semantic/pretraining benchmark entirely.

- [x] Retain six requested methods; exclude legacy olora.
- [x] Extract shared data loader into a neutral common module.
- [x] Copy original motion-cluster manifests and holdout generator.
- [x] Copy feature extraction, Ward clustering and task-selection source.
- [x] Default to random_80_20 seed 42 and direction-specific minimal holdout weights.
- [x] Copy independent T2M/M2T weights; record hashes and provenance.
- [x] Verify canonical holdout regeneration and all referenced dataset files.
- [x] Keep a destination-local .venv; provide installer, resource checks and smoke preparation.
- [ ] Verify full clustering regeneration against canonical assignments.
- [ ] Complete train/reload/generation/metrics smoke tests for every method/direction.
- [ ] Resolve SPICE Java compatibility and unblock full M2T evaluation safely.
- [ ] Validate clean-machine installation and public downloads.
- [ ] Run full benchmarks and establish relation to paper results.

Validation evidence is in docs/VALIDATION.md. Historical earlier-benchmark smoke
logs are not evidence for this benchmark. Missing metrics must never become zeros;
partial runs must never be labelled complete benchmarks. Large resources and all
outputs remain Git-ignored. Review provenance/licenses before publication.
