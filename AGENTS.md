# ToolV2X repository guidance

Start with `README.md`, `docs/github_review.md`, and `docs/evidence_adaptation.md`.
`docs/framework_design.md` describes the approved target architecture; it is not a completion report.
Older documents and `src/probe`/`src/oracle` retain historical experiments. Some probe validation utilities are still imported by the current tools, so inspect callers before removing them.

## Checks and resources

- Resource-independent review: `python scripts/check_review.py` after installing `requirements-review.txt`.
- Full local integration: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'` in the model environment described in the README.
- The full suite requires external checkpoint/tokenizer files. A lightweight PASS does not mean the full model was run.
- Model weights, full datasets and repeated run snapshots are intentionally external. Do not treat archived absolute paths as portable input manifests.
- The user supplies downloads from GitHub, GitHubusercontent, GitHubassets and Hugging Face. Do not work around this via mirrors or redirects. Access to the explicitly assigned project repository is authorized separately.
- Preserve old evidence files. New runs use new output directories and describe their own code/configuration.

## Code Review Rules

1. Trace actual time-t input access. Flag future observations or labels entering online planning, peer evidence available before a query, missing history silently filled from the future, and confusion between physical dataset splits and recording-level research holdout. Offline labels may read future poses; follow the caller rather than flagging every future read.
2. Check P/F semantics and fair comparisons: common coordinates and timestamps, full source context before ROI filtering, P local recomputation versus F at equal information, retention of remote-only targets and separate modes/sources, and actual request/response bytes. Distinguish prompt rounding/truncation from wire compression.
3. Check the real driving/supervision path: in `q8_q9` mode Q9 must use the generated Q8 parent; explicit `direct` mode uses the `Trajectory` task and must not contain or execute Q8. Missing parents must never silently switch the mode. Invalid answers remain failures, only target answer tokens receive loss, and point/prompt tokens stay masked. Parsed trajectories, finite forward loss and historical checker PASS do not establish task quality, successful training, CUDA parity or sequential-query benefit.
