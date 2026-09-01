# Decisions

## 2026-09-01 — Server stack: Gradio 5.x web UI + facade CLI

- Context: Needed a browser frontend for image upload → LaTeX inference, plus one CLI to start either frontend or API server.
- Memory: Gradio `>=5.9.1` (latest 5.x line; Gradio 6 exists but pinning major-bump-risk). `build_app()` in `latex_ocr/webui.py` returns a `gr.Blocks` (not an ASGI app); the facade CLI `latex_ocr` (`latex_ocr/cli.py`, click group) calls `demo.launch(...)` directly with `max_file_size=f"{n}mb"`. API server stays FastAPI (`latex-ocr-server` / `latex-ocr api`). `gr.mount_gradio_app(demo, path=...)` signature takes the app first and has no `share` param — avoid.
- Evidence: `latex_ocr/cli.py`, `latex_ocr/webui.py`, `pyproject.toml` scripts/extras.
- Reuse: When extending servers, add options to both `webui`/`api` commands via `_shared_model_options` in cli.py.

## 2026-09-01 — Dependency versions sourced from /root/Deep-Learning-Practices/pyproject.toml

- Context: User asked to keep dep versions consistent with that reference project.
- Memory: Identical pins (torch>=2.9.0 cu128 index, timm>=1.0.22, mlflow>=3.4.0, etc.); pyproject version bumped 0.1.0 → 1.0.0 to match release-1.0.0. `uv sync` cannot complete in this sandbox (PyPI timeouts) but `uv lock` resolved 196 packages and `uv lock --check --offline` passes.
- Evidence: `pyproject.toml`, `uv.lock` (committed lockfile, torch via pytorch-cu128 index).
- Reuse: Regenerate lock with `uv lock` on a networked machine if deps change.

## 2026-09-01 — Large-file guard: 10 MB, with allowlist for pre-existing corpora

- Context: User asked for gitleaks + large-file detection in pre-commit and CI; repo already tracks corpora JSONs up to 93 MB.
- Memory: Single script `.githooks/check-large-files [threshold-mb]` used by both pre-commit (local repo hook) and `.github/workflows/ci.yml`. Scans staged files (commit) or all tracked files (clean index/CI). Allowlist `.githooks/large-files-allowlist` (gitignore-style patterns) exempts `datasets/formula_corpora/*`; the packaged `latex_ocr/models/pretrained/*.pth` (7.1 MB each) are intentionally NOT exempted at the 10 MB threshold.
- Evidence: `.pre-commit-config.yaml` (gitleaks v8.30.1, pre-commit-hooks not added — ruff already fails on ~190 pre-existing issues, lint gates were not requested), `ci.yml` (checkout@v6, gitleaks-action@v3).
- Reuse: Don't add ruff/mypy CI gates without first fixing pre-existing violations; migrate corpora to HF Hub then empty the allowlist.
