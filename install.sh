#!/usr/bin/env bash
# Open Asset Studio installer: checks git / ffmpeg / uv, then runs scripts/install.py with uv-managed Python 3.12
# (the NVIDIA driver, CUDA version, PyTorch index and disk space are checked there).
#
#   ./install.sh                      install the core, then choose packs of models (default: starter)
#   ./install.sh list [packs|backends|models]
#   ./install.sh doctor [--check]     GPU, driver, CUDA, disk, ffmpeg, HF login, installed / broken backends
#   ./install.sh core
#   ./install.sh pack <name>...       e.g. starter image 3d video sound voice music omni all
#   ./install.sh backend <id>...      one backend (a venv + its upstream source clones)
#   ./install.sh model <model-id>...  the model's backend + its weights (one Hugging Face repo at a time)
#   options:  --weights  also download the weights of packs / backends
#             --dry-run  show what would be done, change nothing
#             --yes      no questions
#             --force    run the recipes again even if installed
#             --torch-index cu130|cu128   override the PyTorch wheel index
# Weights are downloaded with HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1; gated repos stop with instructions.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

die() { echo "error: $*" >&2; exit 3; }

if [ "$(uname -s)" != "Linux" ]; then
  die "Open Asset Studio needs Linux (x86_64 or aarch64) with an NVIDIA GPU. Found: $(uname -s)."
fi

# --- uv (always needed: it provides Python 3.12 and runs the installer) ---------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: uv is not installed. Install it with the official one-liner, then open a new shell and run ./install.sh again:

  curl -LsSf https://astral.sh/uv/install.sh | sh

EOF
  exit 3
fi

# --- git / ffmpeg: needed to install; not for list / doctor / help ----------------------------------------------
need_tools=1
for a in "$@"; do
  case "$a" in list|doctor|validate|help|-h|--help) need_tools=0 ;; esac
done
if [ "$need_tools" = 1 ]; then
  hint="install them with your package manager"
  if command -v apt-get >/dev/null 2>&1; then hint="sudo apt install git ffmpeg"
  elif command -v dnf >/dev/null 2>&1; then hint="sudo dnf install git ffmpeg   (ffmpeg needs RPM Fusion)"
  elif command -v pacman >/dev/null 2>&1; then hint="sudo pacman -S git ffmpeg"
  fi
  missing=()
  for t in git ffmpeg ffprobe; do command -v "$t" >/dev/null 2>&1 || missing+=("$t"); done
  if [ "${#missing[@]}" -gt 0 ] && [[ " $* " != *" --dry-run "* ]]; then
    die "missing: ${missing[*]}. Install: $hint"
  fi
fi

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"
export HF_HUB_DISABLE_XET=1
exec uv run --quiet --no-project --python 3.12 scripts/install.py "$@"
