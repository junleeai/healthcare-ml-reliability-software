# Running the synthetic software checks

This guide covers the public software edition. The checks use invented observations, record identifiers, prediction values and label streams. They require no files from the private research checkout.

## Install and verify

From the repository root, run:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/run_checks.py
```

These are the same commands as the [README](../README.md#install-and-run-the-synthetic-tests). The pinned dependencies cover the included library imports and tests. Plotting, acquisition and full-study reporting scripts are not included.

The tested environment is **CPython 3.14.6 on macOS ARM64**. Pins do not guarantee identical wheels or binaries across platforms. Windows uses a different virtual-environment executable path; other platforms need their own installation and test verification.

## What the tests check

[scripts/run_checks.py](../scripts/run_checks.py) runs the included [synthetic suite](../tests/test_synthetic_core.py), currently **21 cases** including parameterized cases. The runner blocks attempts through `Path.open` to read raw `.psv` files beneath this edition's `data/raw` directory. It is a targeted guard, not a general filesystem sandbox.

| Area | Behavior checked |
| --- | --- |
| Features | Changing observations outside a selected prefix does not change its features; missing values and observed zeros have distinct coverage. |
| Endpoints | First-positive censoring, prediction-horizon boundaries and post-onset exclusion follow the implemented rules. |
| Monitoring inputs | Outcome-related extra fields and duplicate record identifiers are rejected. |
| Distribution statistics | The empirical KS calculation matches the scaled SciPy statistic for the supplied arrays. |
| Loss and uncertainty | Record and hour weighting differ; paired whole-record resampling retains a constant loss difference; independent comparisons reject overlapping keys. |
| Delayed release | The monitor callback runs before outcome review at each supported artificial delay. |
| Access and state | A changed frozen-state fingerprint and an unauthorized record role raise errors. |

The tests use fixed prediction values. They do not train a model, learn detector references or select cutoffs. The pass count describes this public suite, not the larger private research suite.

## Follow the README example

The [walkthrough](../README.md#walkthrough-records-and-hours-give-different-averages) reproduces the inputs to `test_record_and_hour_weighting_are_distinct`. The expected outputs are **2 scored records**, **0.5 record-balanced loss** and **0.82 hour-weighted loss**. All values and identifiers are invented.

To run just that existing test:

```sh
.venv/bin/python -m pytest -q tests/test_synthetic_core.py::test_record_and_hour_weighting_are_distinct
```

The delayed-release test, `test_monitor_runs_before_artificial_release`, separately checks callback order for **0, 3 and 6 artificial steps**. Its `seal()` call flushes pending labels before the final order assertion. This checks software ordering, not real clinical label latency or monitoring accuracy.

## Compatibility and reproduction limits

The source retains study-specific APIs, but the standalone checks cover only the synthetic paths described above. Loading the original protocol, cohort memberships, fitted models or detector references requires artifacts that are not supplied. Full benchmark execution and empirical figure reconstruction are unsupported in this edition.

**Legacy loss-helper warning:** `evaluation.secondary_margin` uses `comparison.patient_interval`, which describes record-balanced loss. The completed study used a separately specified hour-weighted adapter in `benchmark_statistics`. The two are not interchangeable. The public loss tests exercise `estimate` and `compare`; they do not validate `secondary_margin` as an empirical harm classifier or provide clinically validated margins or thresholds.

Replay steps are artificial ordering units, not timestamps. Label-free distribution statistics describe changes in monitored inputs or scores; outcome labels are needed to assess predictive deterioration. Passing synthetic checks does not reproduce the empirical study or establish clinical effectiveness or safety.

See [NOTICE.md](../NOTICE.md) for software/data scope and attribution, and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for dependency notices.
