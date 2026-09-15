# Task 4 independent spec and quality review

Reviewed `task-4-brief.md`, `global-constraints.md`, `task-4-preflight.md`, `task-4-report.md`, `review-policy.md`, and the complete packaged `task-4-review.diff` against named base `sdd/readiness-preparation-base..HEAD`. The packaged diff covers every changed Task 4 code/config/doc/test file except the deliberately omitted 3.2 MB `frames.jsonl`; direct byte comparison confirmed that file equals the controller-whitelisted `source_causal_frames.jsonl`. The packaged `source_metadata.json` and `acceptance_frames.json` likewise equal their controller inputs. No arrays, labels, checkpoints, providers, models or real experiment resources were read or run.

## Verdicts

- **Spec compliance: CHANGES REQUESTED.** The prepared package and lifecycle boundary substantially match the brief, but the validator does not enforce the explicitly required duplicate-ID rule and the only documented validation command is not portable after the batch-private controller files are removed.
- **Code quality: CHANGES REQUESTED.** The implementation is focused and reuses the required APIs, but the identity key weakens a named invariant and the CLI exposes an existing portable library path only through three unnecessarily mandatory preparation arguments.

## Confirmed strengths

- `scripts/prepare_driver_readiness.py:48-176` reconstructs the complete default observations-only `StructuredDriverSpec`, the reviewed `ExecutionSpec`, both explicit Task 3 v2 configs through `_config`, feedback/one-shot controls through `normalize_control`, and the real request through `episode_bundle`. The recorded constructor result is an actual 1,813-byte UTF-8 request with aggregate outer cap 133,120 and candidate IDs `initial`, `slower`, `constant_motion`; the existing `task-4-final-contract.log` records the successful resource-free check.
- `configs/structured_driver_readiness_v1/readiness.json:2-224` freezes the ten recording roles, 2,987/108 full-population counts, 16/4 first/last acceptance identities, known 40/0 exclusions, four Ego/P/F/PF conditions, two-call/three-attempt caps, shared observations-only driver, bootstrap contingency, metrics and B/C/D gates. It keeps lifecycle `prepared_not_executed`, null runtime identity, `ready_to_run_method_spec=false`, and every collection/training/inference/provider/model execution flag false.
- `src/planning/method_episode.py:23-63` adds only `f_current` and `p_current_f_current`, keeps the three existing diagnostic IDs and chooses current-mode actions from `available_actions` using actual receipt count. `src/planning/run_framework.py:454-490` consumes the same exported tuple in both whitelists. The existing `p_current_f_change` remains the actual-revision route, and `diagnostic_conditional_v1` remains the built-in one-shot continuation registered at `src/planning/run_framework.py:573-580`.
- `docs/structured_driver_readiness_2026_09_15.md:1-44` is candid about unknown eligible labels, real loads, output paths/capacity, runtime/storage cost, training, acquisition coverage, evidence use, adapted quality and closed-loop behavior. It also correctly states that bootstrap is conditional, at most three epochs, recollection-only, and not a warm start for the fresh full adaptation.

## Issues

### Critical

None.

### Important — Duplicate physical/frame identities can bypass validation by changing `g`

**Location:** `scripts/prepare_driver_readiness.py:211-245`; missing mutation coverage at `tests/test_driver_readiness_preparation.py:41-83`.

`validate_frames` defines `seen` identity as `(sample_id, scene, g, local_frame)`. Because `sample_id` is separately required to equal `scene:local_frame`, a second row with the same valid `sample_id`/scene/local frame but a different `g` produces a different tuple and is accepted. Conversely, two different sample aliases in one scene may point to the same `g` and are also accepted. This violates the brief's explicit duplicate-ID failure requirement and lets a malformed replacement row preserve population counts and endpoint selection.

A bounded in-memory mutation probe against the actual validator replaced one interior row at a time while preserving counts and acceptance endpoints. Both `duplicate_sample_changed_g` and `same_scene_g_alias` were accepted.

**Minimal resolution:** track stable `sample_id` uniqueness and `(scene, g)` uniqueness independently (the existing `sample_id == scene:local_frame` relation already binds the local-frame alias). Add focused portable-validator mutations for both cases; do not broaden this into a general source-provenance framework.

### Important — The documented portable validation command requires batch-private source files

**Location:** `scripts/prepare_driver_readiness.py:327-336`, `docs/structured_driver_readiness_2026_09_15.md:46-55`; missing CLI coverage in `tests/test_driver_readiness_preparation.py`.

`validate_package(package)` already implements the documented portable structural validation when `source_frames` is absent. The CLI nevertheless marks `--source-frames`, `--acceptance-preview`, and `--source-metadata` as required for every invocation. The only public command points all three at `.superpowers/sdd/...`, which is batch-private and scheduled to disappear. Running the documented package path with only `--validate-only` exits during argument parsing, so users cannot invoke the promised portable check from the retained package and helper.

**Minimal resolution:** require all three source arguments only in preparation mode. In `--validate-only` mode, accept no source arguments and call `validate_package(out)`; optionally accept `--source-frames` to request the explicitly limited exact local whitelist comparison. Do not describe that optional file comparison as independent source-index provenance. Publish the zero-private-input command and add one focused CLI test.

### Minor

None.

## Validation and cross-task limits

The recorded `task-4-final-tests.log` / `.exit` show 36/36 focused preparation, method-episode and budget tests passing with exit 0; `task-4-final-contract.log` / `.exit` show the reported package counts, 1,813/133,120 budget result, exact local source comparison and portable function check with exit 0. Those passing checks were read, not rerun. Only the two bounded probes above were executed to resolve concrete uncovered doubts; neither read arrays/labels nor invoked a model/provider.

After the two Important fixes, rerun only the focused preparation tests and contract command before root's broader gate. Root owns the final whole-branch review and complete lightweight/model-environment regressions. This review confirms a preparation contract and existing-metadata copy, not genuine bootstrap-label eligibility, runnable method/load identity, training, collection, performance, paper superiority or closed-loop safety.

## Assessment

**Ready to proceed? With fixes.** The scientific/lifecycle boundary, frozen configs, deterministic source selection, fixed diagnostic extension and one-shot budget construction are otherwise consistent with Task 4. The two Important validation gaps must be closed before passing the Task 4 gate.
