# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Open Asset Studio: a local, offline web studio (FastAPI core + vanilla-JS SPA) that drives ~175 open-weight
image / 3D / audio / video models through one worker subprocess per backend venv. Built and measured on an
NVIDIA DGX Spark (aarch64, GB10, 121 GB unified memory, CUDA 13); numbers in `docs/research/` are from that machine.

The authoritative references are `docs/CONTRACTS.md` (registry schema, worker protocol, job API, plugin API,
asset meta, frontend module contract, cloud providers, profiles / licence filter) and `docs/DESIGN.md`. Read the
relevant CONTRACTS section before touching an extension point.

## Commands

```bash
./run_studio.sh --host 0.0.0.0           # core on :7860 (LAN, no auth); logs/studio.log. STUDIO_PORT=7861 for another port
envs/core/bin/python -m pytest           # all tests, ~seconds, no GPU / no weights needed
envs/core/bin/python -m pytest tests/test_media_audio.py::test_time_formats   # one file / one test
./install.sh list | doctor               # backends + models installed?, environment check
./install.sh backend <backend> [--weights]   # (re)build one venv from install/<backend>.yaml
./install.sh model <model-id>            # one model: its backend + its weights
```

Run a worker by hand (protocol over stdin/stdout):
```bash
echo '{"op":"run","id":"t","model":{"id":"test-tone","name":"T"},"task":"sfx","params":{"steps":3},"out_dir":"/tmp/x"}' \
  | PYTHONPATH=workers envs/core/bin/python workers/echo_worker.py
```

Always use the venv pythons (`envs/<env>/bin/python`), never the system python. Runtime is offline
(`HF_HUB_OFFLINE=1`); weights must already be in the HF cache.

## Architecture

- **One core process** (`app/main.py`, wired by `app/core.py`): REST + SSE (`/api/events`), serves `app/web/`.
- **Registry** (`app/registry.py`): `models.yaml` then every `models.d/*.yaml` merged in. Defines `tasks`,
  `backends` (venv python path + worker), `models` (tasks, params that drive the generated UI form, presets,
  licence, `memory_gb`, `input_slots`). Each model gets a status: `ready` / `missing_weights` / `missing_env` / `broken`.
- **Jobs** (`app/jobs.py`): one global queue, one GPU job at a time, CPU tools (`gpu=False` handlers) run alongside.
  Before a GPU job, `app/memory.py` checks `/proc/meminfo` MemAvailable and evicts our own resident models
  (prompt LLM first, then LRU). It never touches other users' processes (e.g. ComfyUI).
- **Workers** (`app/workers.py`, `app/procs.py`, `workers/studio_worker.py`): `workers/<backend>_worker.py` runs
  in that backend's venv, started lazily on the first job, talking JSON lines over stdin/stdout. A worker subclasses
  `Backend` (`probe` / `load` / `run` / `unload`) and calls `serve()`. Use `ctx.output` / `ctx.progress` /
  `ctx.check`, and raise `InputError` for user errors. Workers die with the core. fd 1 is reserved for the
  protocol, because the framework redirects prints to stderr (`logs/worker-<backend>.log`).
  Pure-Python helpers that tests import go in a separate `workers/<group>_common.py`.
- **Plugins** (`app/plugins/*.py`, imported at startup): `@handler("name", gpu=...)` for multi-step / CPU jobs
  (`ctx.run_worker(...)` chains worker calls), plus an optional `router` mounted at `/api/x/<plugin>`.
  `providers.py` loads first.
- **Frontend** (`app/web/`): ES modules, **no build step, no CDN** (vendored libs in `web/vendor/`). One file per
  studio in `web/studios/`, built with `S.defineStudio` and usually `S.TaskStudio` from `web/core/sdk.js`.
  Studios hand assets to each other through `receives` / `S.sendTo` action ids, so keep the existing ids.
  Pages must work at 1440 px and 390 px, in dark and light themes.
- **Assets**: `data/projects/<p>/assets/<id>/` (files + `meta.json` with lineage, seed, prompt, licence).
  Exports always write derived files and never modify originals.
- **Cloud** (`app/providers.py`, `app/cloud/`): optional, off by default, `backend: cloud` registry entries.

Core-owned files (listed at the top of `docs/CONTRACTS.md`) should change only through small,
backwards-compatible hooks, each recorded in the CONTRACTS changelog. Everything else is a drop-in file.

## Adding a model group (the usual unit of work)

Recent commits touch the same set of files. Use `git show --stat 482adeb` as a template:
- `models.d/<group>.yaml`: entries with an exact licence (`commercial: false` for non-commercial, `verify: true` if unsure)
  and the measured `memory_gb`
- `workers/<backend>_worker.py` (+ `<group>_common.py` for testable helpers); `app/plugins/<group>.py` if it needs a handler
- `install/<backend>.yaml` recipe (format in `install/FORMAT.md`: idempotent, no sudo, no weights) + `env-lock/<backend>.txt`
  (`uv pip freeze`); upstream repos pinned in `vendor/vendor.lock` (`vendor/` is not committed; `scripts/fetch_vendor.sh`)
- UI in `app/web/studios/<studio>.js` when a new task / tab is needed
- `tests/test_<group>.py`: the registry loads cleanly, tasks and backends exist, every param validates its default,
  and the pure helpers work. No GPU, no weights (pattern: `tests/test_moss_more.py`)
- `docs/research/<group>.md` with measured load / run times and memory; `THIRD_PARTY_NOTICES.md` for new upstream code

Never commit weights, generated media, `data/`, `envs/`, `logs/` or `vendor/` checkouts.

## Running instance

The user often keeps a core running on :7860 (`--host 0.0.0.0`). Edits to `app/web/` only need a browser reload.
Changes to `models.d/` / `prompts/` can be picked up with **Re-check** on the Models page. Changes to `app/*.py`
or plugins need a core restart, and changed worker code takes effect only in a new worker process.
