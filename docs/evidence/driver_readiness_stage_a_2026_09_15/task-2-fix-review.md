# Task 2 fix round 1 independent review

Scope was limited to `task-2-fix-review.diff`, the appended Task 2 report, and the single Important finding in `task-2-review.md`. I did not repeat the reported tests or reopen the rest of Task 2.

## Verdicts

- **Spec compliance: PASS for fix round 1.** The provider-acquired versus receiver-derived distinction required by the original finding is now explicit and traceable without changing ledger or protocol semantics.
- **Code quality: PASS for fix round 1.** The fix reuses the existing ledger categories, keeps the neutral remote union clearly named, and adds focused regression coverage without a second provenance registry.

## Finding resolution

### Original Important — RESOLVED

`src/evaluation/structured.py:88-104` now derives provider receipt fields only from `ledger['acquired_fields']` and receiver derivations only from `ledger['derived_fields']`. It computes known/new/previous collections independently for both classes. `src/evaluation/structured.py:163-187` publishes those six provenance-specific collections and their matching counts; `previously_acquired_remote_refs` is acquired-only. The combined collections remain available under neutral names, including `previously_known_remote_refs`, so the public audit no longer assigns an acquired meaning to receiver-derived forecasts.

The regressions at `tests/test_structured_admission.py:192-237` verify a P-local stage where the receipt-acquired anchor/history remain separate from the newly receiver-derived forecast, followed by a direct F acquisition. The reference-only case at `tests/test_structured_admission.py:256-269` verifies that a new receipt with no new fields leaves both provenance-specific new collections empty while preserving their previous known fields.

The updated Task 2 report accurately lists the new keys and counts, explains the neutral union, records the focused RED/GREEN chronology, and preserves the no-real-model/no-full-regression boundary.

## New Critical/Important findings

None introduced by the fix diff.

## Validation boundary

I did not rerun the reported two-test fix check or 23-test fix-round regression. Root retains responsibility for the final complete lightweight and model-environment regressions and for later cross-task readiness review.

## Assessment

**Fix round 1: PASS.** The original Important finding is resolved, and the scoped fix introduces no new Critical or Important issue.
