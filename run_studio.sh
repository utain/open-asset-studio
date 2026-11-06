#!/usr/bin/env bash
# Open Asset Studio: start the core (FastAPI + web UI). Workers and the prompt LLM start on demand.
#   ./run_studio.sh                 # http://127.0.0.1:7860 (add --host 0.0.0.0 for the LAN; no auth)
#   STUDIO_PORT=7861 ./run_studio.sh
#   LLAMA_SERVER=/path/to/llama-server ./run_studio.sh
# Logs: logs/studio.log (core), logs/worker-<backend>.log, logs/prompt_llm.log. Ctrl-C stops everything it started.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export STUDIO_PORT="${STUDIO_PORT:-7860}"
export LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || echo "$HOME/llama.cpp/build/bin/llama-server")}"
PY=envs/core/bin/python
if [ ! -x "$PY" ]; then
  echo "envs/core is missing. Create it with:" >&2
  echo "  uv venv --python 3.12 envs/core && VIRTUAL_ENV=\$PWD/envs/core uv pip install -r env-lock/core.txt" >&2
  exit 1
fi
mkdir -p logs data
echo "Open Asset Studio on port ${STUDIO_PORT} (default host 127.0.0.1; log: logs/studio.log)"
"$PY" -m app.main "$@" 2>&1 | tee -a logs/studio.log
