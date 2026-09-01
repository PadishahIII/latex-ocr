## 2026-09-01 — Missing [build-system] made project "virtual": no latex-ocr binaries after uv sync

- Context: User followed README install (`uv sync --extra server`) but `.venv/bin/latex-ocr` / `latex-ocr-server` did not exist.
- Memory: `pyproject.toml` lacked a `[build-system]` table, so uv treated the project as virtual (`source = { virtual = "." }` in `uv.lock`) and only installed dependencies — `[project.scripts]` entry points were never installed. Fix: add `requires = ["hatchling"]` / `build-backend = "hatchling.build"`. Follow-on: torchtune 0.6.1 imports `torchao.dtypes.nf4tensor` unconditionally but its metadata doesn't require torchao; torchao ≥0.15 moved that module (ao PR #4256), so pin `torchao<0.15` (0.14.1) alongside torchtune (archived, 0.6.1 final).
- Evidence: `pyproject.toml` (now has [build-system] + torchao pin), `uv.lock`, verified `.venv/bin/latex-ocr --help` exit 0 with api/webui commands.
- Reuse: When `[project.scripts]` binaries are missing after `uv sync`, check for a missing `[build-system]` table first (uv makes the project virtual). If torchtune is touched, keep `torchao<0.15`.

## 2026-09-01 — Per-GPU torch flavor via cu126/cu128 extras, universal lockfile

- Context: User's local GPU is a V100 (sm_70) but repo users may have newer GPUs; asked how to branch local vs default config. Previously `pyproject.toml` hard-pinned `[tool.uv.sources]` torch→cu128 index for everyone.
- Memory: Use uv's PyTorch pattern: torch/torchvision index pins are gated per-extra (`{ index = ..., extra = "cu128" }`), extras `cu126`/`cu128` declared with `conflicts = [[{extra="cu126"},{extra="cu128"}]]`. Default `uv sync` = plain PyPI torch; `--extra cu126` = PyTorch cu126 index (works on V100; PyTorch 2.11+cu128 builds dropped sm_70); `--extra cu128` = cu128 index (Turing+; index tops out at torch 2.11.0). One committed universal `uv.lock` covers all forks — no local files, no `uv.toml` (uv rejects `sources` in `uv.toml`), no `--no-sources` needed. Local machine convention: `uv sync --extra cu126 --extra server`.
- Evidence: `pyproject.toml` [tool.uv] (conflicts, two [[tool.uv.index]], per-extra sources), `uv.lock` (torch 2.13.0 / 2.13.0+cu126 / 2.11.0+cu128 forks), verified syncs: local venv runs torch 2.13.0+cu126, `torch.cuda.is_available()` True, device cc (7,0); `uv sync --extra cu126 --extra cu128` correctly rejected by conflicts guard.
- Reuse: To add a CUDA flavor (e.g. cu130): add `[[tool.uv.index]]` + per-extra source line + matching extra + conflicts entry, then `uv lock`. Never put machine-specific index pins in unconditioned `[tool.uv.sources]`.
## 2026-09-01 — Server stack: Gradio 6.x web UI + facade CLI (supersedes earlier 5.x intent)

- Context: Needed a browser frontend for image upload → LaTeX inference, plus one CLI to start either frontend or API server.
- Memory: Gradio is pinned `>=6,<7`. The 5.x intent was never encoded and the lock resolved 6.26.0, where `gr.Latex` was REMOVED — rendering goes through `gr.Markdown` (KaTeX, default delimiters `$$...$$`), and `Textbox(show_copy_button=...)` became `Textbox(buttons=["copy"])`. `build_app()` returns a `gr.Blocks`; the facade CLI `latex_ocr` calls `demo.launch(...)` with `max_file_size="{n}mb"` (still valid in 6.x, verified). API server stays FastAPI (`latex-ocr-server` / `latex-ocr api`).
- Evidence: `latex_ocr/webui.py` (gr.Markdown + buttons=["copy"]), `pyproject.toml` server extra, smoke test: build_app + launch + GET / → 200 on gradio 6.26.0.
- Reuse: When extending servers, add options to both `webui`/`api` commands via `_shared_model_options` in cli.py. Gradio major bumps: diff component signatures before assuming 5.x APIs survive.
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
