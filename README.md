# Healthcare ML Reliability — code-first edition

Original research software for causal feature extraction, isolated label-free monitoring interfaces, artificial delayed-label replay and record-cluster statistics. This edition contains software and invented test fixtures only. It makes no empirical model-performance or clinical-effectiveness claim.

## What is included

- The original `healthcare_reliability` Python modules, copied without changing the research implementation.
- A standalone synthetic-only test suite for feature availability, endpoint boundaries, payload validation, empirical KS arithmetic, release ordering and record-balanced uncertainty.
- Pinned dependencies and original software instructions. No network service or deployment stack is implemented.

The full research portfolio, raw/source files, trained models, learned detector references/cutoffs, cohort memberships, empirical aggregates, figures and original study configuration are deliberately absent. Historical artifact-backed loaders remain in the library and fail closed when their required artifacts are missing. They are not part of this edition's supported verification workflow.

## Install and verify

Tested using CPython 3.14.6 on macOS ARM64. Other environments are unverified.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/run_checks.py
```

No Git repository, credentials, dataset download or ignored artifact is needed. The tests do not train a predictive model or calibrate a detector. They validate invented examples; passing them is not clinical validation or reproduction of the withheld empirical study. See [reproduction details](docs/REPRODUCING.md).

## Interfaces

| Concern | Module |
| --- | --- |
| Current/past prefix features | [features.py](healthcare_reliability/features.py) |
| Retrospective endpoint rules | [endpoints.py](healthcare_reliability/endpoints.py) |
| Label-free value objects and statistics | [monitoring.py](healthcare_reliability/monitoring.py), [detectors.py](healthcare_reliability/detectors.py) |
| Monitor-before-release lifecycle | [release.py](healthcare_reliability/release.py) |
| Record-balanced loss and cluster comparison | [evaluation.py](healthcare_reliability/evaluation.py) |
| Access/membership guards | [partitions.py](healthcare_reliability/partitions.py), [access.py](healthcare_reliability/access.py) |

Artificial release steps are not real time or measured clinical reporting delay. Label-free distribution statistics cannot establish predictive harm without outcome-supported assessment. Medical variable names and clinical annotation concepts are not licensed by the software grant.

Original software and associated original usage instructions are [MIT licensed](LICENSE). Third-party libraries retain their own [notices](THIRD_PARTY_NOTICES.md). The [scope notice](NOTICE.md) explains the omitted research/data materials and the unresolved official PhysioNet license notices. This proposed edition has not been committed or published.
