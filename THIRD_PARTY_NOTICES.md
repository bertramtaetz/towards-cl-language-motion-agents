# Third-party notices

Original project contributions are under the [root MIT license](LICENSE).
This does not relicense copied code, model weights, model derivatives, data,
or evaluation resources. Preserve nested notices when redistributing code.

| Component | Provenance and notices | Scope |
|---|---|---|
| Motion-Agent-derived code | [szqwu/Motion-Agent](https://github.com/szqwu/Motion-Agent); [original MIT notice](Motion-Agent/LICENSE), copyright 2025 szqwu | Copied/adapted models, data utilities and evaluation infrastructure |
| Locally extracted continual-learning code | [source inventory](docs/source_manifest.json) | Inventory records local provenance, not proof of contributor ownership; maintainers must have authority to license their contributions |
| Vendored nlg-eval | [Maluuba/nlg-eval](https://github.com/Maluuba/nlg-eval); [Microsoft MIT notice](third_party/nlg-eval/LICENSE.md) | Includes local modifications; nested components retain their own notices |
| COCO caption evaluation components | [nested notice](third_party/nlg-eval/nlgeval/pycocoevalcap/license.txt), [BLEU notice](third_party/nlg-eval/nlgeval/pycocoevalcap/bleu/LICENSE) | Do not infer the licenses of downloaded Java binaries from the Python wrapper license |

The companion adapts source paths, task-manifest selection, orchestration,
metric validation and resource handling. These are modifications, not original
upstream releases. Installed Python dependencies retain the licenses in their
distributions; a requirements file is not a replacement for those notices.

For models, datasets, Java resources, and release decisions see
[licensing and acquisition](docs/LICENSING.md) and
[Google Drive packaging](docs/GOOGLE_DRIVE.md).