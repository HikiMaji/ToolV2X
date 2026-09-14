# Structured driver consolidated final fix report

Date: 2026-09-15

Status: COMPLETE for the two Important findings in `final-review.md`. This wave used only synthetic CPU fixtures. It did not run real training, initialize real resources, execute real GoT/MTR, download data, run the repository-wide suites, push, or change the accepted `structured_inputs.py` module-size ruling.

## Changes

### Runtime history audit and numeric training

`src/planning/train_structured_driver.py::_task_rows` no longer compares `task.inputs.ego_history_read_paths` with `plans[0].prepared.ego_history_used.read_paths`. They serve different contracts: the task field preserves the physical files attempted by the runtime loader, while the prepared field is reconstructed solely from the three numeric feature arrays and intentionally has `read_paths=[]` so paths do not enter model input or semantic identity.

No history validation was removed. Export still calls `_check_features`, and saved-row validation in `fit` still reaches the same check. The shared collator verifies that `ego_pose_history`, `ego_pose_history_valid`, and `ego_pose_history_times` are all present together, have the required shapes/dtypes/finite values, and equal the prepared states, mask, and times. Input snapshots continue to copy the full source task, including its actual nonempty history audit paths.

`tests/test_structured_training.py` now has a runtime-style fixture with eleven nonempty attempted pose paths and all three real numeric history arrays. Its focused test verifies the intended empty prepared-path field, exact array/mask/value agreement, successful three-row export, a real tiny-network one-step optimizer fit, and preservation of the nonempty task audit list in `fit/inputs/task_000000.json`.

### Legacy list-window compatibility

`src/planning/evidence.py::new_ledger` now enumerates `prediction_objects` in the source order already checked by `validate_prediction`, instead of redundantly finding each object ID with a NumPy-only comparison. This preserves the original v1 object/field order and removes the invalid assumption that `track_ids` is an array.

The opt-in history branch converts each selected `valid` row and `time_seconds` to NumPy only where `.tolist()` and `.sum()` are needed. List-valued `track_ids`, `states`, `valid`, `scores`, and `time_seconds`, all accepted by the causal-window validator, therefore work in both default v1 and history-enabled v2. Paths remain absent from the ledger and model features.

`tests/test_evidence_ledger.py` now checks that a fully list-valued default window returns exactly the same v1 ledger as its array-valued equivalent. A separate test checks full list-valued opt-in history content, masks, and times.

## TDD evidence

Focused RED before production changes:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_evidence_ledger.EvidenceLedgerTests.test_default_v1_list_window_matches_array_window \
  test_evidence_ledger.EvidenceLedgerTests.test_opt_in_history_accepts_full_list_window \
  test_structured_training.StructuredTrainingTests.test_runtime_history_audit_exports_and_fits_from_numeric_arrays
```

Observed result: 3 tests, 3 errors. Both ledger tests failed at `np.flatnonzero(w['track_ids'] == obj['track_id'])[0]` with `IndexError`. The training test failed at `_task_rows` with `ValueError: ego history audit paths differ from actual task`.

The same command after the minimal production changes:

```text
...
----------------------------------------------------------------------
Ran 3 tests in 1.880s

OK
```

## Scoped verification

Directly covering suites, using the required model environment:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_evidence_ledger test_structured_inputs test_structured_training
```

Result:

```text
..........................................................
----------------------------------------------------------------------
Ran 58 tests in 19.299s

OK
```

Syntax compilation:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m py_compile \
  src/planning/evidence.py src/planning/train_structured_driver.py \
  tests/test_evidence_ledger.py tests/test_structured_training.py
```

Result: exit 0, no output.

Repository whitespace check:

```text
git diff --check
```

Result: exit 0, no output.

The parent agent retains responsibility for the single planned fresh repository-wide lightweight and model-environment suite run after scoped review.
