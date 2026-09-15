# Task4 incremental implementation report

Status: initial implementation and fix round committed; both Important findings from independent Task4 review fixed and verified. Scope remains limited to the Task4-owned config/package, helper/test, diagnostic selector/CLI whitelist and lightweight review registration. Root-owned Stage A report/plan/docs are untouched and unstaged.

## Confirmed existing interfaces

- Task3 full v2 config is strict through `train_structured_driver._config`: version `toolv2x_structured_training_v2`, complete default observations-only `StructuredDriverSpec`, selected-stage/refinement objective, frame-condition weighting, explicit validation cadence and invalid-rate/prefix-L2/earliest selection.
- `ExecutionSpec` is the reviewed `real_smoke_v1` literal: request 16384 bytes, response 65536 bytes, episode 196608 bytes, with all geometry/dynamics fields present.
- Actual one-shot construction is `normalize_control` then `episode_bundle`; the real built-in continuation registration is `diagnostic_conditional_v1`. The prepared control uses `episode_aggregate`, initial/slower/constant_motion, three candidates, 0.5 slowdown, 2048-byte wrapper reserve, 8-field/1536-byte summary and self-refinement within three driver calls.
- Fixed collection policies are `stop`, `p_current`, `f_current`, `p_current_f_current`. Existing `p_current_f_change` remains unchanged and requests change mode only after an actual receipt/revision path.
- Whitelisted metadata contains 3095 accepted physical-train rows: 2987 train and 108 validation across eight/two physical recording roles. Acceptance preview is 20 identities: 16 train and 4 validation, first/last eligible source-order frame per physical recording.
- Source paths are audit aids, never semantic identity. No actual driver runtime load identity exists in Stage A, so the preparation explicitly keeps it null and does not publish a runnable `method_run_spec`.

## RED evidence

Command: `PYTHONPATH=src:tests /root/miniconda3/bin/python -m unittest test_driver_readiness_preparation test_method_episode.MethodEpisodeTests.test_fixed_diagnostic_ids_schedule_only_current_receipt_count -v`.

`task-4-red.log` and `task-4-red.exit` record exit 1: the package/helper and exported diagnostic IDs were absent. A second focused RED proved an unavailable fixed F action incorrectly returned F instead of STOP before the availability guard was added.

## Files currently drafted

- `src/planning/method_episode.py`: one exported diagnostic-ID tuple and fixed F-current/P-current-then-F-current routing based only on actual receipt count and legal available actions.
- `src/planning/run_framework.py`: both diagnostic whitelists consume that shared tuple.
- `scripts/prepare_driver_readiness.py`: one resource-free preparation/validation helper using actual Task3 `_config`, `StructuredDriverSpec`, `ExecutionSpec`, `normalize_control`, `episode_bundle`, UTF-8 `encode` and bundle-envelope validation.
- `tests/test_method_episode.py`, `tests/test_driver_readiness_preparation.py`: selector and package mutation checks.

## Prepared package and validation

`configs/structured_driver_readiness_v1/` contains one 3,095-row `frames.jsonl`, the 20-row acceptance identity JSON, the controller metadata snapshot, complete Task3 full-adaptation and expected-bootstrap v2 configs, and `readiness.json`. The validator reconstructs the actual full default `StructuredDriverSpec`, validates the exact `real_smoke_v1` `ExecutionSpec`, calls Task3 `_config` for both v2 training files, checks the frozen roles/counts/exclusions, and rejects altered roles, fields, source rows, population counts, acceptance selection, P mode and budgets.

The prepared full config is seed 7, AdamW lr 0.0001 / weight decay 0.01, batch size 2, 20 epochs, refinement depth 1, save every 25 processed batches, validate every 100 plus initial/final, CUDA as a future execution choice, selected-stage/refinement objective, frame-condition weighting and invalid-rate/prefix-L2/earliest selection. The bootstrap expected config changes only to three epochs and interval 8. It explicitly does not certify 16 eligible labels: real preflight must calculate eligible rows and save a complete concrete interval `ceil(eligible_train_rows/2)` before any start.

The actual public one-shot constructor produced 1,813 UTF-8 request bytes with candidate IDs initial/slower/constant_motion and aggregate outer response cap 133,120 bytes. Request and permitted response fit the common 16,384/65,536/196,608 `ExecutionSpec`; no provider was called. `diagnostic_conditional_v1` is the actual existing registered continuation ID. Feedback and one-shot controls preserve the same shared specs and evidence rules.

## Final verification

- `task-4-final-tests.log` / `.exit`: 36/36 pass, exit 0. This includes all preparation tests, the full affected old `test_method_episode` module and existing `test_budget_review` constructor/budget controls.
- `task-4-final-contract.log` / `.exit`: exit 0 for py_compile, exact full local source comparison, deterministic 20-frame selection, actual constructor budget validation, portable structural validation, lightweight `check_review` registration, no torch/transformers import and `git diff --check`.
- Staged diff contains only the 13 Task4-owned package/code/test/doc files. Root README/STATUS/GitHub review/plan/Stage A report changes remain unstaged.

## Unknown real-readiness boundaries

No source arrays, labels, checkpoints, driver/predictor model load, provider runtime, training, inference or real collection were accessed or executed. The external source-index provenance is locally cross-checked only through the controller-whitelisted JSON copy; portable checks establish structure/count/selection, not origin from the workstation index. Actual eligible bootstrap labels, output capacity, runtime/storage cost, load identities, finite real initial plans, requested acquisition coverage, direct evidence use, adapted quality and closed-loop behavior remain unknown. The package therefore has lifecycle `prepared_not_executed`, null runtime identity and no runnable method spec. Full adaptation starts fresh after recollection; no warm-start API or changed-config exact resume is promised.

Commit subject: `Freeze shared driver preparation and acceptance settings`.

## Fix round 1

Review found two Important gaps. First, the old composite duplicate key could accept the same stable sample/local-frame alias with a changed global `g`, or two local aliases with the same `(scene, g)`. `validate_frames` now independently enforces unique `sample_id`, `(scene, local_frame)` and `(scene, g)` while retaining the existing exact row schema and `sample_id == scene:local_frame` relation. No manifest field or provenance mechanism was added.

Second, the CLI previously required all three private preparation inputs even for `--validate-only`. Portable validation now accepts only the retained package path plus `--validate-only`; `--source-frames` is optional for the explicitly limited exact copy comparison. The acceptance preview and source metadata remain required together with source frames only when creating a new package. The public document now shows the zero-private-input command and says the optional comparison does not certify external-index provenance.

Fix RED: `task-4-fix-red.log` / `.exit` records exit 1 with both stable-identity collision probes accepted and both public CLI forms rejected by argument parsing. Fix GREEN: `task-4-fix-green.log` / `.exit` records both focused tests passing. Final fix verification: `task-4-fix-final.log` / `.exit` records the complete eight-test preparation module passing, py_compile and `git diff --check`, exit 0. Per controller instruction, the earlier 36 method/budget targets and full suites were not repeated. No resource work occurred.

Fix commit subject: `Fix preparation identities and portable validation`.
