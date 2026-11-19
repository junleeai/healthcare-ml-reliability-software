# Reproducing the code-first software checks

Run the README installation commands from this edition's root. No files from the full research checkout should be copied in to make these checks pass. The dependency list covers the included library imports and tests; plotting, network acquisition and full-study reporting dependencies have been omitted because their scripts/results are not shipped.

The runner checks only `tests/test_synthetic_core.py` in this edition and blocks actual raw-PSV access under its own data root. Fixtures use invented arrays, generic record identifiers, fixed prediction values and artificial label streams. No model is trained and no reference/threshold is learned. The software can exercise a monitor-before-release lifecycle without claiming real monitoring performance.

The full research suite is not included and its pass count must not be claimed for this edition. Artifact-dependent study commands, empirical figure reconstruction and private model loading are unsupported here. The underlying library modules remain unchanged, so pinned real-artifact loaders continue to require unavailable, checksum-verified inputs rather than being weakened for the release.

This is a source-inspection/software-contract edition, not a promise that every historical API is supported independently. In particular, `evaluation.secondary_margin` retains its historical patient-interval convention; the completed study used the separately specified hour-weighted adapter in `benchmark_statistics`. Neither empirical materiality classification nor clinical thresholds are offered by this edition. Its loss checks exercise `estimate` and `compare`; no frozen helper has been silently corrected or promoted.

The intended checks cover causal-prefix invariance, first-positive censoring, horizon boundaries, strict label-free payloads, duplicate rejection, empirical KS calculations, paired bootstrap arithmetic, artificial feedback ordering and frozen-state failure. Any displayed numbers are invented fixture values, not patient results.

Version pins do not guarantee platform-independent wheel or binary identity. The tested host is CPython 3.14.6 on macOS ARM64; Windows uses a different venv executable path. A different platform needs its own installation/test verification. See the [scope notice](../NOTICE.md) and [dependency notices](../THIRD_PARTY_NOTICES.md).
