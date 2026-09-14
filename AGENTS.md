# ToolV2X repository guidance

Start with `README.md`, `docs/github_review.md`, and `docs/structured_driver_implementation_2026_09_15.md`.
`docs/evidence_adaptation.md` records the retained GoT evidence path.
`docs/framework_design.md` describes the approved target architecture; it is not a completion report.
Older documents and `src/probe`/`src/oracle` retain historical experiments. Some probe validation utilities are still imported by the current tools, so inspect callers before removing them.

## Checks and resources

- Resource-independent review: `python scripts/check_review.py` after installing `requirements-review.txt`.
- Full local integration: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'` in the model environment described in the README.
- The full suite requires external checkpoint/tokenizer files. A lightweight PASS does not mean the full model was run.
- Model weights, full datasets and repeated run snapshots are intentionally external. Do not treat archived absolute paths as portable input manifests.
- The user supplies large downloads from GitHub, GitHubusercontent, GitHubassets and Hugging Face. On 2026-09-14 the user explicitly authorized direct downloads smaller than 100 MB; enforce that limit when size is unknown. Use official sources, not mirror workarounds. Access to the explicitly assigned project repository is authorized separately.
- Preserve old evidence files. New runs use new output directories and describe their own code/configuration.

## Code Review Rules

1. Trace actual time-t input access. Flag future observations or labels entering online planning, peer evidence available before a query, missing history silently filled from the future, and confusion between physical dataset splits and recording-level research holdout. Offline labels may read future poses; follow the caller rather than flagging every future read.
2. Check P/F semantics and fair comparisons: common coordinates and timestamps, full source context before ROI filtering, P local recomputation versus F at equal information, retention of remote-only targets and separate modes/sources, and actual request/response bytes. Distinguish prompt rounding/truncation from wire compression.
3. Check the real driving/supervision path: in `q8_q9` mode Q9 must use the generated Q8 parent; explicit `direct` mode uses the `Trajectory` task and must not contain or execute Q8. Missing parents must never silently switch the mode. Invalid answers remain failures, only target answer tokens receive loss, and point/prompt tokens stay masked. Parsed trajectories, finite forward loss and historical checker PASS do not establish task quality, successful training, CUDA parity or sequential-query benefit.

4. The opt-in numeric driver uses `toolv2x_interaction_v2` and a full `StructuredDriverSpec`. Check that P histories and F modes enter the shared numeric model, that actual prior plans carry field dependencies, and that exact-repeat versus same-evidence refinement preserves the intended input. Numeric outputs are not Q9 text; bill all driving attempts separately from language generation.
5. Numeric training must use independent offline labels, recording-level train/validation isolation, and the same structured collator/network as inference. No target may fill a missing prior. Checkpoint initialization is explicitly untrained; synthetic gradient/resume tests are not dataset training or method evidence.
