# Install recipes

`./install.sh` reads one recipe per backend from `install/<backend>.yaml` and the packs from `install/packs.yaml`.
A recipe is the only place the install steps live: the env-lock header and the research notes may explain *why*,
but the installer runs what is written here.

```yaml
backend: stab3d                 # = the backend id in models.yaml / models.d/*.yaml (its `python:` is envs/<env>)
env: stab3d                     # venv folder under envs/ (default: backend id)
python: "3.12"
torch:                          # installed first, from the PyTorch index matching the driver (cu130 / cu128)
  - torch==2.9.0
  - torchvision==0.24.0
lock: stab3d                    # env-lock/<lock>.txt, installed with `uv pip install -r`
lock_flags: []                  # e.g. [--no-build-isolation]
vendor: [TripoSR, stable-fast-3d]   # scripts/fetch_vendor.sh filters, cloned before the lock (locks may use ./vendor/...)
steps:                          # optional shell steps, run from the repo root after the lock, with
  - uv pip install --no-deps -e ./vendor/stable-fast-3d/texture_baker   #   VIRTUAL_ENV, PY (venv python) and REPO set
check: "import torch, tsr"      # optional python one-liner run in the venv to verify the install
manual:                         # printed at the end, never run (things that need nvcc, a licence, ...)
  - "SF3D texture baking needs the CPU build step above; GPU rasterizer is not available without nvcc."
tested: [linux-aarch64]         # platforms this recipe was actually run on
size_gb: 9                      # rough venv size
shares_env_with: null           # backend id, when this backend reuses another backend's venv (no own install)
```

Rules:
- Steps must be idempotent: re-running `./install.sh` on an installed backend is a no-op or a quick re-check.
- No `sudo`, no edits outside the repo, no downloads of model weights in `steps` (weights come from the
  registry: `./install.sh model <id>` downloads them one repo at a time).
- Platform-specific steps: `steps_linux-aarch64: [...]` / `steps_linux-x86_64: [...]` are run in addition to `steps`.

How `./install.sh` runs a recipe (details that are not obvious from the keys):
- Order: `vendor` clones, `uv venv`, `torch` (from the PyTorch index), `lock`, `steps` (+ platform steps), `check`.
  `steps` run with `bash -euo pipefail`; the extra environment variables are `VIRTUAL_ENV`, `PY`, `REPO` and
  `TORCH_INDEX` (`cu130` or `cu128`), and `PATH` starts with the venv's `bin/`, so a plain `uv pip install ...` targets the venv.
- `lock` files are frozen on cu130. On cu128 the installer drops the torch family, the `nvidia-*` CUDA libraries and
  `triton` from a temporary copy (the `torch` step already brought matching ones) and removes `+cuNNN` suffixes.
  List every torch-index package of the lock (`torch`, `torchaudio`, `torchvision`, `torchcodec`) in `torch:`.
- `lock_flags: [--no-deps]` is right for a complete `uv pip freeze` (nothing left to resolve). If `lock_flags` contains
  `--no-build-isolation`, setuptools and wheel are installed first.
- A recipe with `shares_env_with` runs after the recipe it shares with; its own `vendor`, `steps` and `check` run in that venv.
- A recipe with no `lock`, `torch`, `steps` or `shares_env_with` is manual-only: the installer prints `manual:` and does nothing else.
- An `envs/<env>` that is a symlink is never modified. `./install.sh validate` checks every recipe against this file.
