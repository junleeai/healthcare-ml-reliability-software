# Healthcare ML Reliability and Drift Monitoring

Hospital models can encounter different measurement patterns, missing values or patient populations after development. These changes can affect predictions. Actual outcomes—the labels needed to evaluate those predictions—may arrive later. This creates a gap between noticing a data change and knowing whether prediction quality has worsened.

This Python library provides components for monitoring input and prediction distributions, simulating delayed label release and evaluating predictions after labels become available. The public edition includes source modules and synthetic tests. It does not reproduce the complete empirical study.

## What the software does

| Capability | Included implementation |
| --- | --- |
| Construct features from available observations | [features.py](healthcare_reliability/features.py) builds feature vectors from a supplied prefix of current and past observations. |
| Calculate distribution-change statistics | [detectors.py](healthcare_reliability/detectors.py) includes empirical Kolmogorov–Smirnov (KS) calculations. |
| Keep labels separate from monitoring inputs | [monitoring.py](healthcare_reliability/monitoring.py) defines input summaries, coverage and prediction scores; its monitoring schema rejects outcome fields. |
| Simulate delayed outcome availability | [release.py](healthcare_reliability/release.py) runs the monitoring callback before releasing due labels to the review callback. |
| Compare prediction loss across records | [evaluation.py](healthcare_reliability/evaluation.py) calculates record-balanced and hour-weighted loss and resamples whole records for uncertainty estimates. |
| Check retrospective endpoints and access rules | [endpoints.py](healthcare_reliability/endpoints.py), [partitions.py](healthcare_reliability/partitions.py) and [access.py](healthcare_reliability/access.py) provide endpoint, membership and access helpers. |

## Technology stack

Python, NumPy and SciPy, with scikit-learn and joblib dependencies for the included model-related modules. Tests use pytest. [requirements.txt](requirements.txt) pins the dependencies.

## Install and run the synthetic tests

Tested with **CPython 3.14.6 on macOS ARM64**. Other environments are unverified. From the repository root:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/run_checks.py
```

The suite contains **21 test cases** using invented inputs. It checks feature availability, endpoint boundaries, label-free inputs, KS arithmetic, loss weighting, record resampling, release ordering and frozen-state guards. No dataset download, model training or detector calibration is needed. See the [reproduction guide](docs/REPRODUCING.md) for coverage and compatibility details.

## Walkthrough: records and hours give different averages

Loss measures prediction error; lower is better. The existing `test_record_and_hour_weighting_are_distinct` in [the synthetic suite](tests/test_synthetic_core.py) uses two scored records: one with a mean loss of **0.1 over one hour**, and one with **0.9 over nine hours**. A third record has no eligible hours and is excluded from both averages.

The following Python example uses the same invented inputs. Run it with the installed environment's `.venv/bin/python` from the repository root:

```python
from healthcare_reliability.evaluation import RecordLoss, estimate
from healthcare_reliability.partitions import RecordKey

records = [
    RecordLoss(RecordKey("A", "p1.psv"), 0.1, 1, 0, "invented-digest"),
    RecordLoss(RecordKey("A", "p2.psv"), 8.1, 9, 0, "invented-digest"),
    RecordLoss(RecordKey("A", "p3.psv"), 0.0, 0, 0, "invented-digest"),
]
result = estimate(records)
print(result.scored_records)               # 2
print(round(result.patient_balanced, 2))   # 0.5
print(round(result.hour_weighted, 2))      # 0.82
```

Giving each scored record equal weight produces **0.5**. Weighting by eligible hours produces **0.82**, because the longer record contributes more. These are arithmetic checks on supplied losses, not measured model performance. The record keys are invented identifiers; the example opens no patient files.

## Supported use and limitations

The synthetic suite and walkthrough run independently of the private research checkout.

Some retained study APIs require unavailable inputs: pinned membership and protocol files, fitted models or learned detector references and cutoffs. Those loaders reject missing or invalid inputs; the complete study workflow is unsupported here. No patient data, empirical tables or figures are included.

A distribution alert does not establish that prediction quality worsened; that assessment requires outcomes. Replay delays of **0, 3 or 6 steps** are artificial, not measured clinical reporting delays or calendar time. The synthetic tests do not establish clinical effectiveness or safety.

The legacy `evaluation.secondary_margin` helper uses a patient-balanced interval. It should not be used as an hour-weighted harm criterion. The [compatibility note](docs/REPRODUCING.md#compatibility-and-reproduction-limits) explains the distinction and the supported loss checks.

## Development approach

I directed the project, reviewed the Codex-generated work and made project decisions. **Codex wrote most of the implementation code.**

## Licensing and attribution

The original software, synthetic tests and software-use instructions are publicly available under the [MIT License](LICENSE). Third-party libraries retain their own [license terms and notices](THIRD_PARTY_NOTICES.md).

The software was developed as part of work involving Reyna et al., *Early Prediction of Sepsis from Clinical Data*, PhysioNet/CinC 2019 v1.0.0. The [scope and attribution notice](NOTICE.md) contains the official citation and distinguishes software rights from data, terminology and research-output rights. It also documents the unresolved conflict between PhysioNet's CC BY 4.0 listing and its versioned ODbL license file; this edition does not select one as authoritative. The full research portfolio remains unpublished pending clarification of those rights.
