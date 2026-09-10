# GitHub review preparation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to execute the approved upload scope in this session.

**Goal:** Publish the first reviewable ToolV2X source and selected evidence to the user-supplied HikiMaji/ToolV2X repository.

**Architecture:** Keep the existing research modules and historical artifacts. Bundle the locally available upstream source/configuration needed to inspect the actual calls, make resource locations configurable, and provide a NumPy/SciPy-only review check beside the original integration tests.

**Tech Stack:** Python 3.8–3.11, unittest, NumPy, SciPy, Git; the existing integration environment additionally uses PyTorch and the original LLaVA dependencies.

**Spec:** The upload inventory approved by the user's repository and credential handoff on 2026-09-10; current research scope remains `docs/framework_design.md` and `docs/evidence_adaptation.md`.

## Global constraints

- Two vehicles and P/F only; no training, router, RSU, I, or new method claims.
- Preserve generated Q8 as Q9 input; future labels remain offline.
- No model weights, full datasets, paper PDFs/full texts, download caches, credentials, or repeated run snapshots in the Git tree.
- GitHub/Hugging Face resource downloads remain the user's responsibility. This repository handoff authorizes Git access to the named destination.
- Keep existing artifacts unchanged and distinguish old execution evidence from new packaging checks.
- Do not record file-content digests. Verify copied bytes directly and use paths, configuration names, and dates.
- No force push or remote visibility change.

## Task 1: Source and resource portability

Files: `src/common/v2v4real_meta.py`, `src/prediction/cmp_adapter.py`, `src/planning/v2vgot.py`, `src/planning/adaptation_data.py`, `tests/test_review_portability.py`, `vendor/`.

- [x] Re-run the existing 58 tests before edits.
- [x] Add a regression check that `TOOLV2X_V2VGOT_ROOT` changes the actual localization file read; run it before changing the loader and confirm the assertion fails.
- [x] Copy the local original LLaVA package, native offline label source, CMP YAML/intention points, reference dataset source, and corresponding licenses. Record each source path and local modifications.
- [x] Use a sibling data repository as the default and environment variables for external resources; use bundled source/configuration for the current model adapters.
- [x] Re-run the new check and the full original integration suite.

## Task 2: Review entry points and selected evidence

Files: `README.md`, `AGENTS.md`, `.gitignore`, `requirements-review.txt`, `scripts/check_review.py`, `docs/github_review.md`, `docs/upstream_sources.md`, `tests/test_saved_review_evidence.py`.

- [x] Add checks against the existing saved P/F forecasts, failed compact answer, and generated-parent supervision examples.
- [x] Provide a review runner selecting resource-independent tests explicitly; keep the full integration command separate.
- [x] Whitelist source, Markdown documents and the selected JSON/text artifacts. Retain the failed run as well as successful cases.
- [x] Explain implemented versus planned components, local resource setup, reference source ownership, and the fact that optimizer steps remain zero.

## Task 3: Verify the actual Git tree and publish

- [x] Initialize local Git and connect the supplied empty destination without putting authentication material into Git configuration.
- [x] Audit the staged file list, symlinks, large files, and accidental credentials. Validate selected artifact references and documentation links.
- [x] Export exactly the staged tree to a temporary directory; run lightweight checks there with only review dependencies and unavailable external resource roots.
- [x] Verify unchanged model source/configuration and run the full local integration suite after source edits.
- [x] Commit and push the selected tree to `main`; verify the remote branch matches the local commit without printing revision identifiers.
- [x] Report the published location, checks performed, limits of reproducibility, and token revocation advice.
