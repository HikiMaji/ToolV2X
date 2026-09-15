# Task 1 report: request-decision cost accounting

## Outcome

Task 1 adds one compute-accounting contract for every actual request decision or request preflight, including episodes without a control configuration. New episodes identify that contract with `toolv2x_compute_accounting_v1`; each attempted decision stage records a `control` cost event using `toolv2x_control_compute_v2`, with a stage number, measured duration when known, and an explicit completion flag.

The evaluator keeps completed `control_seconds` and the existing non-overlapping `total_compute_seconds` meaning. Missing or incomplete decision time makes those complete values unknown, while `known_cost` retains measured partial durations. Historical episodes with explicit v2 control timing remain readable when they predate the episode-level accounting identity. Historical decision events without usable timing are no longer interpreted as zero.

## Owned changes and APIs

- `src/planning/method_episode.py`
  - Added `COMPUTE_ACCOUNTING_VERSION = 'toolv2x_compute_accounting_v1'`.
  - Added `DECISION_TIMING_VERSION = 'toolv2x_control_compute_v2'`.
  - `run_task_episode(...)` now emits the episode accounting identity independent of `control_spec`.
  - An actual decision/preflight creates its incomplete control event before work begins. Policy/preflight/dispatch failures retain known elapsed time with `complete=False`; successful or forced STOP closes the same event with `complete=True`.
  - The policy-visible `cost_ledger` remains the true pre-decision prefix: the current unfinished audit event stays in the episode but is not presented as prior spent cost.
  - Progress persistence remains outside measured decision time. Prefix reuse requires the same accounting contract and preserves prior stage events without duplication.
- `src/evaluation/framework.py`
  - `_method_cost(task)` remains the shared reader. It validates the accounting identity and decision timing identity, derives expected decision stages from actual event history, distinguishes a measured zero from missing timing, and preserves known partial cost.
  - `evaluate_method_task(...)` publishes the episode accounting identity.
  - `evaluate_method(...)` reports the observed accounting-version counts and labels only uniformly current archives with `toolv2x_measured_stages_v4`; mixed or legacy inputs are explicit.
  - `METHOD_TIMING_SCOPE` documents exclusions: model loading, process management, unrecorded executor bookkeeping, persistence I/O, and network transport. Generation and peer/receiver model times remain nested diagnostics and are not added twice.
- `tests/test_decision_accounting.py`
  - Covers None and explicit-feedback policy timing, forced no-call STOP, policy exception, dispatch failure, progress interruption, prefix reuse, exact repeat, historical missing timing, and historical explicit timing.
- `tests/test_method_evaluation.py`
  - Updates expected totals to include the newly measured decision intervals.
- `scripts/check_review.py`
  - Registers the torch-free accounting test module in the lightweight review set.

Shared consumers inspected: `src/planning/query_data.py` and `src/planning/bundle_data.py` already route through `_method_cost` and reject unknown or inconsistent terminal costs. They require no duplicate version logic or file changes.

The unrelated `docs/driver_readiness_stage_a_2026_09_15.md` draft is outside Task 1 ownership and is excluded from the commit.

## RED/GREEN chronology

The inherited Task 1 ledger records the initial eight-test RED run as 3 failures, 4 errors, and 1 pass. It specifically reproduced the old None-control zero accounting and missing accounting events on failure/interruption paths. The prior implementer then recorded the first fix as 8/8 GREEN. No archived measurements were rewritten.

During handoff verification, these two incorrect package-qualified invocations failed before collection because `tests/` is not a Python package. They changed no files and are retained here as known command REDs:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v tests.test_decision_accounting
Result: 1 import error, ModuleNotFoundError for tests.test_decision_accounting.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v tests.test_method_episode tests.test_method_evaluation tests.test_method_controls.ControlTests tests.test_budget_review tests.test_query_data tests.test_query_value tests.test_bundle_data tests.test_control_bundle
Result: 8 import errors, one for each package-qualified module.
```

Corrected handoff checks use bare module names because `tests` is placed directly on `PYTHONPATH`.

The first broad old-regression command ran to completion, but its tool session identifier was lost at the initial 30-second yield, so its final exit and total count could not be recovered. Its visible output showed every method-episode, method-evaluation, method-control, and budget-review case passing before the query modules. Those modules were not rerun after their passing output was already confirmed.

The separately recoverable query runs then exposed the same shared regression:

```text
test_query_data
Result before final fix: 22 tests in 62.214s; 16 passed and 6 errored at the strict decision-boundary equality check.

test_query_value
Result before final fix: 25 tests in 146.824s; 11 passed and 14 errored through the same query-data decision-boundary check.
```

Root cause: the new unfinished control event correctly existed in the episode before decision work, but `decision_state(episode)` also copied that current event into the policy-visible `cost_ledger`. Online states therefore contained one more event than reconstruction from the actual pre-action prefix. A focused assertion reproduced this for both None and feedback controls with 2 subtest failures. The minimum fix constructs the policy state from the cost-event prefix immediately before the current audit event; no archive equality rule was weakened.

## Verification

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v test_decision_accounting
Final result after the shared-state fix: 8 tests passed in 0.266s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -q test_query_data.TargetTests.test_control_grammars_keep_frozen_mask_and_bind_old_union_requests test_query_data.TargetTests.test_first_failure_uses_failure_loss_without_teacher_or_fake_children test_query_data.TargetTests.test_first_targets_follow_frozen_teacher_even_when_stop_is_better test_query_data.TargetTests.test_teacher_selects_before_gt_is_read_and_stop_only_charges_first_request test_query_data.TargetTests.test_terminal_supervision_preserves_both_improvement_and_harm test_query_data.TargetTests.test_terminal_targets_use_incremental_cost_and_stop_zero
Result after the shared-state fix: 6 tests passed in 26.840s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v test_query_value.DatasetTests.test_training_rows_join_actual_online_states_and_keep_recording_holdout
Result after the shared-state fix: 1 test passed in 14.192s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -q test_bundle_data
Result: 12 tests passed in 64.633s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -q test_control_bundle
Result: 15 tests passed in 0.210s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m py_compile src/planning/method_episode.py src/evaluation/framework.py tests/test_decision_accounting.py tests/test_method_evaluation.py scripts/check_review.py
Result: exit 0.

git diff --check
Result: exit 0.
```

The complete repository suite remains deferred to final integration as required.

## Concerns and review boundary

- `control_seconds` includes local candidate construction, policy selection, and final request dispatch validation. It excludes progress persistence and transport execution.
- A recognized, complete event may measure actual zero; an absent, incomplete, unversioned, or unknown-version event cannot establish zero.
- `known_cost.total_compute_seconds` is a measured lower bound when any component is unknown. It is not presented as a complete total.
- Exact-repeat setup generations that do not attempt a request decision are not charged as control stages; only its final actual decision is charged.
- This task changes derived evaluation of historical missing decision timing from zero/complete to unknown/incomplete. It does not modify archived episode bytes or claim new method performance.
- Independent spec/quality review is still required by the task gate after this owned commit; the root integrator will perform that review before starting the dependent task.

## Independent review fix round 1

The review identified two reader-only gaps. Both were fixed without changing the current episode writer, archived measurements, strict archive equality, or query/bundle consumers.

- A legacy terminal failure with `error.stage == 'request_preflight_or_policy'` and a terminal failed event now contributes that event's stage to the expected control stages. If its control event is absent, `control_seconds` is unknown, `known_cost.control_seconds` remains 0, and `cost_issues` includes `missing control cost stage`. The explicit measured-zero forced STOP remains 0 and complete.
- Archive summary counting now maps a non-string accounting identity to the stable category `invalid_non_string`. `_method_cost` still reports `unknown compute accounting version`, so the row remains an invalid artifact instead of aborting the complete archive evaluation.

Exact RED/GREEN checks:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v test_decision_accounting.DecisionAccountingTests.test_policy_exception_before_decision_append_keeps_known_incomplete_prefix
RED: 1 test failed in 0.011s because legacy control_seconds was 0 instead of None.
GREEN: 1 test passed in 0.011s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v test_method_evaluation.MethodArchiveTests.test_non_string_accounting_version_is_retained_as_invalid_category
First fixture correction: an empty list was swallowed by the old falsy fallback and produced the wrong legacy category.
RED with a nonempty list: 1 test failed in 0.016s because Counter raised TypeError for the unhashable list.
GREEN: 1 test passed in 0.027s.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest -v test_decision_accounting test_method_evaluation
Covering result: 25 tests passed in 1.089s.
```

No query suite, model test, real model, data, or training command was run in this fix round.
