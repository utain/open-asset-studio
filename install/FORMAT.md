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
