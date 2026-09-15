# Stage B real execution evidence

Read [the result and stopping boundary](../../structured_driver_acceptance_2026_09_15.md) first.

- `summary.json`: all five complete 80-task batches, actual request counts, validation and verified checkpoint lineage.
- `bounded_bootstrap_result.json`: the terminal three-epoch cap; full adaptation remains unstarted.
- `physics_diagnosis.json`: per-frame raw trajectory dynamics and separate offline label dynamics. Invalid outputs remain failures.
- `coverage.json`, `preflight.json`, `concrete_training_v2.json`: actual independent label coverage and the complete fixed 16/4 bootstrap config.
- `execution_record.json`: model environment and source/run boundary.
- Python files: copies of local operational wrappers, not new production APIs. Their relative-root assumptions apply to the original output location below; do not execute these copied files in this evidence directory.

Original logs, literal source snapshots, tasks, model provenance, inputs and complete checkpoints remain at `/root/autodl-tmp/ToolV2X/outputs/structured_acceptance_2026_09_15_v1/`. The first failed pretraining invocation performed zero updates; the captured successful first epoch and later exact resumes form one 24-update lineage. Initial tensor-consistency errors are retained and not claimed fixed.

No production code or frozen preparation spec changed. The prior 346/448 test counts are historical Stage A verification, not new test results for this execution. This package establishes a failed bounded readiness gate, not P/F consumption, method benefit or paper superiority.
