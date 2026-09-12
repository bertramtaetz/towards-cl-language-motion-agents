# Installation

All commands run from the companion Git root. Do not use the source project's
environment. Linux/Python 3.12 is the tested configuration.

```bash
bash scripts/install.sh
.venv/bin/python -B -m pytest -q
.venv/bin/python -B scripts/verify_repository.py
```

The installer creates `.venv` if absent, selects PyTorch 2.10.0 CUDA 12.8 wheels,
installs requirements and the independent local NLG package, and checks package
dependencies. It needs internet access. The existing environment has been tested;
this installer is not yet validated on a fresh machine. The environment freeze
in [environment_validated.txt](environment_validated.txt) is an audit artifact, not a portable lockfile.

RTX 5090 forward/backward operations passed with this stack. Older PyTorch 2.2.2
could not execute on that GPU. CPU unit tests neither load full models nor
validate GPU or publication reproduction.

## Resources and readiness

Follow [DATA_AND_MODELS.md](DATA_AND_MODELS.md), then:

```bash
.venv/bin/python -B scripts/resources.py verify --profile t2m
.venv/bin/python -B scripts/check_setup.py --profile t2m --cuda
```

Verification checks bytes, not model compatibility. Setup checks additionally
validate canonical dataset IDs and optional real CUDA execution. It returns nonzero for missing resources or dataset files. All profile paths are repository-local by default.

For NLG diagnostics (writes a metric file only on success):

```bash
.venv/bin/python -B scripts/smoke_nlg.py
```

Java 21 fails the legacy SPICE scorer because a required JavaScript engine is
unavailable. A replacement runtime is not yet validated. Installing Java alone
does not resolve this release blocker. Do not disable SPICE and call the result
a complete benchmark. The launcher guards full M2T generation evaluation. Token-only M2T engineering
evaluation is supported by the documented two-task smoke suite.

Use [VALIDATION.md](VALIDATION.md) for the authoritative tested/untested status.

[Back to README](../README.md)
