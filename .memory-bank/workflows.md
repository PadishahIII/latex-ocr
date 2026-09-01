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

## 2026-09-01 — HF model card for release-1.0.0

- Context: Wrote `models/release-1.0.0/README.md` as a Hugging Face model card (with YAML frontmatter per hub model-card spec).
- Memory: Frontmatter uses `license: mit`, `pipeline_tag: image-to-text`, `base_model: timm/swin_small_patch4_window7_224.ms_in22k` (verified real HF repo, matches `swin_name` in `latex_ocr/trainers/train.py`), `datasets: [PadishahIIIXXX/latex-ocr-dataset]`, and a `model-index` with the plain/styled test-split metrics. Card facts verified in source: vocab 1,122 (line count of `latex_tokenizer.vocab`), input 192×672, beam 4, max length 354 (`latex_ocr/serve.py`), splits 23,868/2,955 (README). Per user instruction: never mention the Debug-8-2 codename in user-facing cards — say "latex-ocr (this release)" in tables instead.
- Evidence: `models/release-1.0.0/README.md`, `latex_ocr/serve.py`, `latex_ocr/trainers/train.py:220,247`.
- Reuse: For future releases, duplicate the release-1.0.0 card, update metrics/params, and keep internal codenames out.

## 2026-09-01 — GitHub push over clash (slow-push fix)

- Context: First push to github.com/PadishahIII/latex-ocr crawled from China mainland behind clash.
- Memory:
  - Repo proxy must be `http://127.0.0.1:7890` (mixed port, ~1.8 MB/s down), NOT `socks5h://127.0.0.1:7891` (~0.6 MB/s). Applied in `.git/config` for http/https/lfs.
  - clash node upload ≈165 KB/s per connection but scales with concurrency (~890 KB/s at 8 streams) → set `lfs.concurrenttransfers 8`, `lfs.transfer.maxretries 8`, `http.version HTTP/1.1`.
  - Raw 97 MB/17 MB/4 MB dataset JSONs + 7.4 MB `.pth` sat in git history (pre-LFS commit 0c3a1cd) → push would upload them twice (git pack + LFS). Purged with `git filter-repo --strip-blobs-bigger-than 2M --force`; threshold 2M chosen so `uv.lock` (1.2 MB) survives. filter-repo removes the `origin` remote — re-add it after.
  - Pre-purge backups: `/tmp/latex-ocr-pre-purge.bundle` (git bundle) + `/tmp/latex-ocr-lfs-backup` (.git/lfs copy).
- Evidence: curl speed tests via 7890/7891/TUN; `git count-objects -vH` (packs: 0, ~47 MiB loose); `git cat-file -s` blob sizes per commit; 8-connection upload test.
- Reuse: Before the first push of any repo migrated to LFS, check `git rev-list --objects --all` for raw big blobs; for clash pushes use HTTP port 7890 + HTTP/1.1 + parallel LFS. If push is still slow, the clash node's upstream is the ceiling — switch nodes, LFS resumes.

## 2026-09-01 — Gitleaks full-history scan gotcha

- Context: `gitleaks detect --log-opts="... B^..H"` failed with `fatal: ambiguous argument` even though both SHAs existed — because `B` (b1978ec) is the repo's root commit, and `root^` doesn't resolve. Gitleaks treats non-empty git stderr as fatal ("scanned ~0 bytes, no leaks found in partial scan") — a silent false-pass.
- Memory:
  ```bash
  # Include the root commit: pass head's history directly
  gitleaks detect --redact -v --exit-code=2 \
    --report-format=sarif --report-path=results.sarif \
    --log-opts="--no-merges --first-parent <head-sha>"
  # Equivalent in-range alternative: git log B H   (two revs, no ^)
  ```
- Evidence: `git rev-list --parents -n1 b1978ec…` → no parent; corrected scan found 2 leaks (generic-api-key) in `recipe/utils/minio_util.py:226,228` (MINIO_ACCESS_KEY/MINIO_SECRET_KEY hardcoded defaults via `os.getenv(..., default)`).
- Reuse: When base commit = root commit, drop the `^..` form. Real secrets exist in minio_util.py defaults — rotate before any push.
