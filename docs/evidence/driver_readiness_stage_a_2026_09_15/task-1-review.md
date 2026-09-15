# Task 1 independent review: request-decision cost accounting

Reviewed the packaged net diff against named base `sdd/readiness-accounting-base`, subject `Account for every actual request decision`, together with the Task 1 brief, exact global constraints, implementer report, and review policy. This review did not recreate the diff, modify code, run real models/data/training, or repeat the reported suites.

## Verdicts

- **Spec compliance: CHANGES REQUESTED.** The new writer satisfies the current-episode accounting path, but the shared reader still maps one explicitly required historical missing-duration case to numeric zero.
- **Code quality: CHANGES REQUESTED.** The writer-side event lifecycle is small and coherent, and version constants are shared rather than duplicated. The reader's incomplete inference and one archive-summary robustness issue need correction or explicit triage.

## Findings

### Important — A legacy pre-decision policy/preflight failure is interpreted as zero control time

**Location:** `src/evaluation/framework.py:177-190`; missing regression at `tests/test_decision_accounting.py:74-87` and `tests/test_decision_accounting.py:147-169`.

`expected_control` is derived only from persisted `decision` events and already-present control-cost events. A legacy None-control episode that failed in `request_preflight_or_policy` before the decision append has neither. `_method_cost` therefore supplies an empty value list and returns `control_seconds == 0`, even though the failed phase proves that decision/preflight work started and its duration is missing. `cost_complete` is false because the old episode cost is incomplete, but the field itself is still indistinguishable from the real measured-zero budget STOP, and `cost_issues` is empty.

A bounded synthetic probe against the reviewed code produced:

```text
status=policy_error
error_stage=request_preflight_or_policy
decision_events=[]
control_seconds=0
known_control_seconds=0
cost_complete=False
cost_issues=[]
```

This violates the brief's requirements to distinguish missing from actual zero and to reject missing decision duration as unknown. It also leaves the report's claim that every actual decision/preflight is represented too broad for historical archives.

**Minimal resolution:** when a persisted failure phase unambiguously proves that request preflight/policy began, add that failed stage to the expected control stages even if the legacy archive has no `decision` or control event. Return `control_seconds=None`, keep the known prefix at zero, and preferably record a missing-control-stage issue. Add a regression by taking the pre-decision policy-failure fixture, removing the new accounting identity and control event to reproduce the old None-control shape, and distinguish it from the explicit zero-duration no-call STOP.

### Minor — An unhashable accounting-version value can abort archive summary generation

**Location:** `src/evaluation/framework.py:528-537`.

`_method_cost` treats an unknown JSON accounting version as an issue, but `evaluate_method` later feeds the raw value to `Counter`. A malformed persisted list or object is a legal JSON value but is unhashable, so the batch evaluator raises instead of retaining the row as an invalid artifact and reporting the observed invalid identity.

**Minimal resolution:** validate or normalize the exported accounting-version value to a stable string category before counting, while preserving the per-row unknown-version issue. Add a small archive-reader case if this is fixed now; otherwise record it for final-review triage.

## Verified coverage and report accuracy

The diff does establish an episode-level `toolv2x_compute_accounting_v1` identity independently of control kind. It creates the incomplete control event before candidate construction/policy work, closes that same event after dispatch validation, keeps the event out of the policy-visible prior-cost prefix, excludes persistence callback time, requires the accounting identity for prefix reuse, and skips exact-repeat setup stages that make no decision. The evaluator retains numeric durations from incomplete recognized events in `known_cost`, makes their complete totals unknown, preserves historical explicit v2 timing, and documents non-overlapping timing exclusions. Query and bundle consumers inspected through their named `_method_cost` interface already reject missing or inconsistent terminal costs; no additional consumer edit is required for the normal contract.

The implementer report matches the packaged files and the current-path tests, but its historical-unknown conclusion does not cover the Important case above. The report clearly records that the broad old-regression result was unrecoverable, that full query suites exposed the shared state-prefix regression, that only focused affected cases plus bundle checks were rerun after the fix, and that the complete repository suite remains deferred. I did not treat the unrecoverable broad run as passing evidence.

The added module is included in the lightweight review runner and contains no direct torch/model import; the runner retains its model-dependency guard. No out-of-scope source, role/control semantics, capacities, archived measurements, or research artifacts appear in the packaged diff.

## Validation boundary for root

Only the bounded legacy policy-failure reader probe above was run because it resolved a concrete uncovered doubt. After the Important finding is fixed, rerun the focused decision-accounting and affected query/evaluation checks. Root still owns the required final complete lightweight/model-environment regressions in the declared external-resource environment; the incomplete large-suite evidence in the Task 1 report is not a substitute for that final validation.
