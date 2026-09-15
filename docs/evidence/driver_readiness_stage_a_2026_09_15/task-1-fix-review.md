# Task 1 fix round 1 independent review

Scope was limited to `task-1-fix-review.diff`, the appended Task 1 report, and the two findings in `task-1-review.md`. I did not repeat the reported tests or broaden this into a repository review.

## Verdict

- **Original Important — RESOLVED.** `src/evaluation/framework.py:179-195` now recognizes a terminal `request_preflight_or_policy` failure as an expected control stage even when a legacy archive has neither a `decision` event nor a control-cost event. The missing event therefore contributes `None`, keeps known control cost at zero, and adds `missing control cost stage`. The regression at `tests/test_decision_accounting.py:89-95` constructs the precise old None-control shape, while the existing measured-zero forced STOP assertion remains intact.
- **Original Minor — RESOLVED.** `src/evaluation/framework.py:533-535` maps `None` to `legacy_unspecified`, retains strings verbatim, and maps every non-string JSON value to the stable `invalid_non_string` category before `Counter`. `_method_cost` still marks that row with `unknown compute accounting version`. The archive-level regression at `tests/test_method_evaluation.py:318-332` verifies that evaluation completes, the row remains invalid, and the summary records the normalized category.
- **New Critical/Important findings in the fix diff: NONE.**

## Quality and report check

The Important fix is narrowly tied to the failure phase that proves preflight/policy execution began and also requires the terminal event to be `failed`; it does not infer control work for unrelated stages. Existing control events remain deduplicated through the stage set, and the new missing-stage issue is consistent with downstream consumers that already reject incomplete terminal accounting.

The Minor fix preserves the semantic distinction between absent and invalid identity and avoids using archive-controlled compound values as counter keys. It does not weaken per-row version validation or allow an invalid archive to receive the current v4 scope label.

The appended report accurately describes both code changes and records the focused RED/GREEN chronology plus the combined result of 25 covering tests. I did not independently rerun those tests. No query suite, model test, real model, data, or training work is claimed for this fix round.

## Scoped conclusion

**Fix round 1: PASS for both original findings.** Task 1 may proceed past this review gate with the prior requirement unchanged: root owns the final complete lightweight/model-environment regressions and broader integration validation.
