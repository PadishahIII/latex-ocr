# Workflows

## 2026-09-01 — Install, run servers, release model

- Context: Verified entry points after adding webui/facade CLI.
- Memory:
  ```bash
  uv sync --extra server                                  # web UI + API extras
  hf download PadishahIIIXXX/latex-ocr --local-dir models/checkpoints/

  latex-ocr webui --model models/checkpoints/<ckpt>.pth --device cpu   # :7860
  latex-ocr api   --model models/checkpoints/<ckpt>.pth --device cpu   # :8000
  latex-ocr-server --model models/checkpoints/<ckpt>.pth --device cpu  # legacy entry, same as `api`
  ```
- Evidence: `pyproject.toml [project.scripts]`, `latex_ocr/cli.py`.
- Reuse: All three commands accept `--model/--device/--max-length/--beam-size`.

## 2026-09-01 — Pre-commit setup

- Context: Guard commits against secrets and large binaries.
- Memory:
  ```bash
  uv tool install pre-commit
  pre-commit install
  pre-commit run --all-files     # gitleaks + large-file detector
  ```
- Evidence: `.pre-commit-config.yaml`, `.githooks/check-large-files`, `.githooks/large-files-allowlist`.
- Reuse: CI (`.github/workflows/ci.yml`) runs the identical checks; keep both in sync via the shared script.
