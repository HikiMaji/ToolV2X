# Task 2 independent review: derived structured evidence-use audit

Reviewed the packaged net diff against named base `sdd/readiness-admission-base..HEAD`, subject `Audit direct and indirect structured evidence use`, together with the binding Task 2 brief, global constraints, implementer report, review policy, and the named shared interfaces. This review did not recreate the diff, mutate code or git state, run real inference/training/data work, or repeat the reported passing suites.

## Verdicts

- **Spec compliance: CHANGES REQUESTED.** The audit covers admission, direct/indirect use, coverage, capacities, prior dependencies and plan-to-plan changes, but its new/previous remote collections do not preserve the required distinction between provider-acquired evidence and receiver-derived evidence.
- **Code quality: CHANGES REQUESTED.** The helper is otherwise compact, read-only, validation-first and well integrated. One public collection currently gives receiver-derived fields an acquired/new-remote meaning that consumers cannot disambiguate from the audit row.

## Strengths

- `audit_structured_episode` first delegates to the existing numeric episode validator, then derives its report from the validated ledger snapshots and prepared admission metadata. It does not create a second receipt registry or read models, labels, files, tokenizers or future data.
- Direct remote/local primary fields come from actual `use='tensor'` groups, while coverage reports retain full refs, tensor locations, valid observation steps, forecast modes/timepoints, contexts and sources. Dependency collections intentionally overlap, and `indirect_only_*` performs the required subtraction.
- First-plan tensor/prior/output changes remain undefined; later plans compare the actual archived tensors, prior input and waypoints. Exact repeats, refinements, reference-only receipts and missing outputs are represented without fabricated zero baselines.
- Integration is limited to numeric interaction v2; old GoT dispatch still emits `structured_audit=None`. The new test module is registered in the torch-free review runner.

## Issues

### Critical

None.

### Important — Receiver-derived forecasts are reported as newly acquired remote fields

**Location:** `src/evaluation/structured.py:88-95`, `src/evaluation/structured.py:154-176`; missing semantic assertion at `tests/test_structured_admission.py:192-209`.

`known_remote_refs`, `new_remote_refs` and `previously_acquired_remote_refs` are all built from `ledger['acquired_fields'] + ledger['derived_fields']`. The original ledger already keeps those categories distinct: acquired records bind provider response values to receipt IDs, while derived records have `origin='receiver_derived'` and purchased parent refs. Combining them makes `previously_acquired_remote_refs` semantically false and leaves a standalone audit row unable to tell which newly listed fields were actually purchased from the provider.

A single bounded synthetic P-local probe against the reviewed code confirmed the impact. Its paid receipt contained only `anchor` and `history`, while `new_remote_refs` contained `anchor`, `history`, and the receiver-derived `forecast`. The implementer report explicitly discloses this union, but that disclosure does not satisfy the brief's acquired-versus-direct/indirect audit requirement.

**Minimal resolution:** reuse the existing ledger/admission split and publish separate acquired and receiver-derived known/new/previous collections with matching counts. Keep `known_remote_refs` as a neutral union if it is useful, but make `previously_acquired_remote_refs` refer only to `acquired_fields` and ensure the corresponding new-acquired collection excludes `derived_fields`. Do not add another ledger. Extend the existing P-local test to assert that receipt-acquired anchor/history and the receiver-derived forecast remain distinguishable in the public audit.

### Minor

None.

## Report accuracy and validation boundary

Apart from the Important semantic issue, the Task 2 report accurately describes the packaged files, public API, test fixture boundary, reported commands and unexecuted full-suite boundary. The report does not claim real method utility, training success, CUDA parity or closed-loop evidence.

Only the bounded P-local provenance probe described above was run because it resolved the concrete uncovered doubt; no reported passing test was repeated. After the Important fix, rerun the focused structured-admission and affected method-evaluation checks. Root still owns the final complete lightweight and model-environment regressions in the declared external-resource environment. Stage A Tasks 3 and 4 were not implemented or reviewed here, so this verdict does not establish their readiness or the final cross-task research boundary.

## Assessment

**Ready to proceed? With fixes.** The implementation is structurally sound and most of the required audit is present, but the acquired/derived conflation changes the meaning of a core public provenance collection. Correct that split and add the focused regression before passing the Task 2 review gate.
