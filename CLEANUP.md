# Public release cleanup

Date: 2026-10-25

## Preservation

- Original branch: `main`, commit `9966cf6` (same as `origin/main`).
- Full snapshot: new branch `dev`, commit `0c611f6`.
- Included all six modified tracked files and all four untracked PDF/PNG files.
- SHA-256 comparison of all 147 tracked/untracked files against `dev` succeeded,
  including the original `results` symlink text. No stash, reset, or force push.
- Ignored Python caches were left alone. The external target of `results` was
  not traversed or deleted. This is a source snapshot, not a backup of external data.
- `main` was fast-forwarded to the snapshot before cleanup, preserving the latest
  SpecViT MAE changes. The complete history remains reachable from both branches;
  this cleanup reduces the current tree, not the historical Git object contents.
- Changes are local; no remote branch was pushed or modified.

## Retained and added

All `GeospatialFM/models`, `datasets`, and `data_process` source files and sensor
YAML remain. Original pretrain/finetune argument definitions remain in their
original paths. `get_task_model` was extracted unchanged from finetuning utilities
into `GeospatialFM/models/loading.py` without training/metric dependencies.

Added JSON generation via `python -m GeospatialFM.configs`, package metadata,
reference dependencies, usage documentation, and offline smoke tests.

Minimal compatibility fixes:

- Pretrain `parse_args` accepts an explicit argument list for config export.
- DINOv3 imports optional xformers only when sparse execution is enabled.
- DOFA stores its default LayerNorm selection as a JSON-compatible string and
  resolves it to the same LayerNorm factory at model construction time.
- Downstream LESSViT configs now default `rank` to 1 (matching the original CLI),
  fixing direct construction without a training argument namespace. Explicit
  checkpoint/CLI rank values remain honored.

## Removed from the main tree

60 files: training/fine-tuning/evaluation runners and utilities, sweep/launch
scripts, notebooks, generated figures, website assets and deployment workflow,
editor config, and the machine-specific results symlink. Every removed file is
available unchanged on `dev`; no datasets or checkpoint targets were deleted.

## Validation and limits

Validation uses Python 3.11 in `/tmp/lessvit-release-venv`, inheriting installed
packages from `sat` and locally overriding `huggingface-hub` to 0.34.4 and
`datasets` to 3.6.0. The existing `sat` environment was not modified. This is not
a full clean-room dependency installation.

Offline checks cover all Python module imports, seven model config JSON round
trips, five sensor exports and SRF matrices, both experiment argument export
modes, overwrite protection, a small LESSViT checkpoint save/reload and encoder
weight load, and synthetic 202/120/60/10-band GeoTIFF loading/collation.

External baseline checkpoints, real datasets, optional GPU sparse execution,
and full training/evaluation were not run. Original model-specific checkpoint
formats and dataset directory layouts still apply.

Results: all four smoke tests passed. Wheel build succeeded; packaged CLI help,
sensor YAML inclusion, and DOFA JSON-to-model construction were checked outside
the repository. `git diff --check` passed.
