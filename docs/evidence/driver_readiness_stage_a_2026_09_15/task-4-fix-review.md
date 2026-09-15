# Task 4 fix round 1 independent re-review

Reviewed only `task-4-fix-review.diff` against the two Important findings in `task-4-review.md`, together with the updated `task-4-report.md`. No passed test was rerun, and no code, config, documentation, git state or research resource was modified.

## Verdicts

- **Spec compliance: PASS.** Both previously reported Important gaps are resolved within the requested Task 4 scope.
- **Code quality: PASS.** The fixes are focused, reuse the existing validator path, and introduce no new Critical or Important issue.

## Resolution evidence

- `scripts/prepare_driver_readiness.py:211-249` now tracks `sample_id`, `(scene, local_frame)` and `(scene, g)` independently while retaining the existing `sample_id == scene:local_frame` relation. A changed-`g` duplicate alias and a same-`(scene, g)` alias can no longer preserve counts and endpoint selection while bypassing the stable-identity contract. `tests/test_driver_readiness_preparation.py:87-99` covers both exact mutations.
- `scripts/prepare_driver_readiness.py:331-349` makes all three source inputs optional at argument parsing, invokes portable `validate_package(out)` when `--validate-only` has no source copy, permits only optional `--source-frames` for exact local comparison, and still requires the complete three-input set for package preparation. It rejects preparation-only metadata arguments in validation mode rather than giving them misleading semantics.
- `docs/structured_driver_readiness_2026_09_15.md:16` and `:46-56` publish the retained zero-private-input validation command and accurately limit the optional source-copy comparison: it does not certify provenance from the external index. `tests/test_driver_readiness_preparation.py:101-111` covers both portable and optional-comparison CLI forms and asserts the reported comparison boundary.

## Findings

### Critical

None.

### Important

None.

### Minor

None.

## Recorded verification and limits

The recorded `task-4-fix-green.log` / `.exit` show the two focused regressions passing with exit 0. The recorded `task-4-fix-final.log` / `.exit` show all eight preparation tests passing with exit 0. These logs were inspected without repeating the tests. The earlier 36 method/budget checks and full suites were intentionally not rerun.

This PASS is limited to the Task 4 fix diff and closes the two original review findings. Root still owns the whole-branch review and complete regressions; the preparation package remains `prepared_not_executed` and does not establish real load identity, eligible labels, training, collection, performance or closed-loop evidence.

## Assessment

**Ready to proceed: yes, for Task 4.**
