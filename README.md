# Open Asset Studio

A local, offline web studio for **assets for any medium**: images, 3D models, sound and video from
open-weight models, in one app with a shared asset library, a job queue that keeps one GPU busy safely, and a
built-in prompt improver. Make what a production needs:

* **Games**: transparent RGBA sprites and icons, seamless loops, GLB props, SFX sheets, barks, music.
* **Video and film**: thumbnails, b-roll stills, storyboard frames and concept art, SFX, foley, music beds and
  stings, voice-over, loudness-matched masters.
* **Podcasts**: cover art, intro / outro stings, music beds, narrator and host voices, -16 LUFS delivery.
* **Social and ads**: vertical 9:16 stills, product shots, jingles and hooks, -14 LUFS delivery.

Pick a **profile** (game, video, podcast, social, film or general) in the top bar and the studio puts the right style
presets first, defaults to the right aspect ratio, and exports with the right loudness, sample rate and image size
(see [Profiles and export presets](#profiles-export-presets-and-the-commercial-safe-filter)).

It drives about 100 open-weight models through small per-model worker processes: image generation and editing,
Hunyuan3D image-to-3D, and 65+ audio models covering SFX, ambience, music, songs, TTS, voice design, voice cloning,
voice conversion, singing, speech-to-speech replies and video-to-sound (Foley). The full audio index, with the
licence of each model, is in [docs/research/audio_more/README.md](docs/research/audio_more/README.md).

> **Status: research project.** It was built and measured on one machine, an NVIDIA DGX Spark
> (aarch64, GB10, 121 GB unified memory, CUDA 13). Other Linux + NVIDIA setups should work for most backends, but
> the pinned environments, memory numbers and workarounds in `docs/research/` are specific to that machine.

* Spec: [docs/DESIGN.md](docs/DESIGN.md) · Contracts for studio builders: [docs/CONTRACTS.md](docs/CONTRACTS.md)
  · Setup: [docs/SETUP.md](docs/SETUP.md) · Research and measurements: [docs/research/](docs/research/)

## Model licences: read this first

**This repository's code is Apache-2.0. The models it runs are not.** Many are **non-commercial** (CC-BY-NC,
CPML, research-only and similar), some have regional limits, and some need you to accept terms on Hugging Face.
The licence of every model is in its registry entry, shown on the Models page and stored on every asset it makes.
You are responsible for following the licence of each model, and of each upstream project in `vendor/`
(see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), that you install and use.

## Quick start

Requirements: Linux (x86_64 or aarch64), an NVIDIA GPU with a driver that supports CUDA 12.8+ (CUDA 13 is the
tested one), git, ffmpeg and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
Python 3.12 is installed by uv; nothing is installed system-wide.

```bash
git clone <repository-url> open-asset-studio && cd open-asset-studio && ./install.sh
./run_studio.sh                      # http://127.0.0.1:7860
```

`./install.sh` checks the prerequisites, installs the core, then asks which **packs** of models to install
(default: `starter`, about 30 GB: one image model, sound effects, a voice and image-to-3D, all commercial-OK and
small enough for a 24-32 GB GPU). It downloads model weights one Hugging Face repo at a time and skips what is
already in your cache. Models whose repo is gated tell you to accept the terms and run `hf auth login`.

```bash
./install.sh list                    # packs, backends, models: installed? how big?
./install.sh doctor                  # GPU, driver, CUDA, disk, ffmpeg, HF login, broken backends
./install.sh pack image 3d --weights # more packs (starter image 3d video sound voice music omni all)
./install.sh model qwen-image-2.1    # one model: its backend + its weights
./install.sh backend foley           # one backend only (add --weights for all its models' weights)
./install.sh pack starter --dry-run  # show what would be done; --yes skips the questions
```

The core also starts with nothing installed: the Models page lists every model with its status
(`missing_env` / `missing_weights`) and the `./install.sh model <id>` command that fixes it. The CPU test models
(Diagnostics page) work right away. The recipes behind the installer are `install/<backend>.yaml`
(format: [install/FORMAT.md](install/FORMAT.md)) and `install/packs.yaml`; the manual version of every step is in
[docs/SETUP.md](docs/SETUP.md).

## Run

```bash
./run_studio.sh                      # http://127.0.0.1:7860, logs in logs/studio.log
./run_studio.sh --host 0.0.0.0       # listen on the LAN (no authentication: see SECURITY.md)
STUDIO_PORT=7861 ./run_studio.sh     # another port
LLAMA_SERVER=/path/to/llama-server ./run_studio.sh   # prompt improver binary (default: llama-server on PATH)
```

The core starts in about a second. Model workers start on the first job of their backend and unload / exit
when idle; the prompt LLM (llama-server, 127.0.0.1:7880) starts on the first **✨ Improve** click and sleeps
after 5 minutes. Stopping the core (Ctrl-C) stops every process it started; if the core is killed, its workers
and the LLM die with it (PR_SET_PDEATHSIG + stdin EOF + parent watch).

`HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_XET=1` are set by the script: nothing is downloaded at runtime.

## What is where

```
run_studio.sh            start script (127.0.0.1:7860, logs/studio.log)
models.yaml              model registry: tasks, backends (venvs), models (features, license, memory, params, presets)
models.d/*.yaml          drop-in registry files (merged after models.yaml): sound, tts_light, tts_clone_a/b, music, sfx_more, foley, omni, vc_sing, image, cloud
                         (models may declare `input_slots`, the extra audio / voice / video inputs their worker reads)
settings.json            user overrides, git-ignored (factory defaults: app/config.py + registry): profile, commercial-safe filter, default model per task, memory margin, idle timeouts, LLM config
prompts/*.yaml           prompt-improver templates per prompt_style (core.yaml + studio files)
app/                     FastAPI core (envs/core)
  main.py                REST + SSE API, static web UI
  core.py                wires the services; loads app/plugins/*.py
  registry.py            models.yaml -> capabilities + status (ready / missing_weights / missing_env / broken)
  jobs.py                global queue, one GPU job at a time, CPU tools alongside, cancel, retry, history
  memory.py              MemAvailable checks, loaded-model table, LRU eviction (LLM first)
  workers.py procs.py    worker subprocesses (JSON lines over stdin/stdout), leak-proof spawning
  assets.py media.py     projects/assets on disk, thumbnails, waveforms, LUFS, export zip, safe file serving
  profiles.py licensing.py  use-case profiles + export presets; the commercial-use licence filter
  improver.py            llama-server supervisor + templates + JSON-schema output + checks
  providers.py cloud/    optional cloud APIs: keys (data/secrets.json, 600), Allow-cloud switch, budget, adapters
  plugins/               server-side extensions (job handlers, CPU tools, extra routes)
  web/                   single-page app (ES modules, no build step, vendored libs in web/vendor/)
workers/                 backend workers (studio_worker.py = framework, echo_worker.py = reference backend; one <backend>_worker.py per venv:
                         ttslight*, clonea_*, cloneb_*, music_*, sfxmore, foley, omni_*, vc, vc_diffsinger)
vendor/                  upstream source trees used by those workers (not committed; vendor.lock + scripts/fetch_vendor.sh)
engines/hy3d/app/      Hunyuan3D engine + image helpers shared by the qwen / hy3d / imgtools workers
scripts/                 fetch_vendor.sh
data/                    projects/<p>/assets/<id>/ (files + meta.json), jobs.jsonl, presets.json, exports/
logs/                    studio.log, worker-<backend>.log, prompt_llm.log, probe.log
envs/                    studio-owned venvs: core, audio, qwentts, vllm_omni_overlay and one per backend of the 65 added audio models
                         (see Environments)
env-lock/                pip freeze of every studio-owned env
tests/                   critical tests (pytest)
```

Sound studio tabs: SFX, Ambience, Music, **Song** (full song, or *Sing a score* with DiffSinger), **Voice** (Speak, Design,
Clone, **Convert** = voice conversion, **Reply** = speech-to-speech answer with its text), **Video to sound** (MMAudio,
HunyuanVideo-Foley, ThinkSound; the take card plays the muxed video) and Sheets. Video files (mp4 / mov / webm / mkv)
are a library asset type with a poster and a player. The 65 added audio models: `docs/research/audio_more/README.md`.

Pages: **Home** (profile chooser, quick-start flows, idea box with the improver, recent assets, queue, memory), **Image / 3D /
Sound / Pipelines** (studios), **Library** (filters, search, rate, tag, compare, export, upload), **Models**
(capability matrix, status + fix command, license, defaults per task, backend deep check), **Queue & System**
(jobs, loaded models, workers, prompt LLM, studio settings, logs, self-test), **Providers** (optional cloud APIs: keys, Allow
cloud switch (off by default), budget + confirm threshold, spend, prices, your OpenAI-compatible endpoints,
subscription CLIs; docs/CONTRACTS.md section 8), **Diagnostics** (a full studio page on the CPU test
models: the reference for new studios).
**Style guide** (WORKSPACE group) keeps one look per project: a written look, an avoid list, up to 4 reference images, a palette
and per-kind variants (image, texture, sprite, UI icon, video, audio mood). The core adds it to the prompt of every image, texture,
video and sound job when the job is submitted, so Retry and the Assistant use it too. A "Style" chip in every prompt box shows it and
switches it off for one job; every output records the guide version it was made with. Versioned, with history and restore
(docs/CONTRACTS.md section 11, `docs/research/style_guide.md`).
**Settings** (SYSTEM group) holds every configurable value in one place, studio-wide (`settings.json`) and per project
(`project.json`): profile, commercial-safe filter, export preset, default model per task, prompt backends and art direction,
memory, workers, the prompt LLM and cloud limits. A project's overrides win over the system values, and the page marks
each setting as inherited or overridden (docs/CONTRACTS.md section 10).

## Profiles, export presets and the commercial-safe filter

**Profiles.** The profile button in the top bar (and Home, and Queue & System > Studio settings) selects what you are
making: `game`, `video`, `podcast`, `social`, `film` or `general` (default). It is stored in `settings.json`
(`"profile"`). A profile changes defaults only, never what a model can do:

| profile | style chips first | default aspect | default export |
|---|---|---|---|
| general | file order | model default | original files |
| game | RGBA sprites, loops, 3D-ready, pixel art, icons | model default (1:1) | original files (today's behaviour) |
| video | thumbnail, b-roll, product shot, cinematic, music beds, transitions | 16:9 | 48 kHz 24-bit WAV at -14 LUFS, 1920x1080 PNG |
| podcast | cover art, intro / outro stings, beds, host voice | 1:1 | 44.1 kHz MP3 at -16 LUFS, 3000x3000 JPEG |
| social | social vertical, product shot, ad jingle, hooks | 9:16 | 48 kHz MP3 at -14 LUFS, 1080x1920 JPEG |
| film | storyboard, concept art, score cues, room tone | 16:9 | 48 kHz 24-bit WAV at -23 LUFS, PNG |

**Export presets.** Export (library, studios, asset panel > Export / convert...) can convert audio and images on the
way out: ffmpeg two-pass `loudnorm` to a target LUFS (true peak -1 dBTP), sample rate, bit depth and format
(WAV / MP3 / OGG / FLAC); images are cropped or padded to an aspect ratio, resized and saved as PNG / JPEG / WebP.
Conversions always write **new derived files** (a zip, a download, or new library assets via the `asset.export`
tool); the stored originals are never modified. CPU only.

**Commercial-safe models only.** Every model has a licence badge: *Commercial OK*, *Check licence* (allowed on paper
but flagged `verify`) or *Non-commercial*. The shield button in the top bar (or Models, or Queue & System) turns on
`license.commercial_only`: non-commercial models disappear from every picker and the job API refuses them
(HTTP 403, kind `license`). `license.strict` also refuses *Check licence* models. Details: docs/CONTRACTS.md
"Profiles, export presets, licence filter".

## Add a model

1. Make sure it runs in one of the backend venvs (see `docs/research/*`), and put the weights in the HF cache.
2. Add an entry (in `models.yaml` or a new `models.d/<name>.yaml`):

   ```yaml
   models:
     - id: my-sfx-model
       name: My SFX model
       modality: audio
       tasks: [sfx]
       backend: audio                      # an existing backend (venv + workers/audio_worker.py)
       weights: [{hf: org/my-sfx-model}]
       license: {name: Apache-2.0, commercial: true}
       memory_gb: 8
       prompt_style: sfx
       params:
         - {name: seconds, type: float, min: 1, max: 30, default: 5, unit: s}
         - {name: batch, label: Takes, type: int, min: 1, max: 8, default: 4}
         - {name: seed, type: seed, default: -1}
       presets: {fast: {batch: 2}, balanced: {}, quality: {batch: 8}}
   ```
3. Teach the backend worker to load / run it (`workers/<backend>_worker.py`, `Backend.load` + `Backend.run`).
   A new venv = a new `backends:` entry + a new worker script (copy `workers/echo_worker.py`).
4. Models page -> **Re-check** (and **Deep check** for the backend). The model shows up in the pickers of its tasks.

Full schema, worker protocol, job API, asset meta and the frontend module contract: `docs/CONTRACTS.md`.

## Environments

| env | used for | recreate |
|---|---|---|
| `envs/core` | the FastAPI core, echo worker, CPU tools | `uv venv --python 3.12 envs/core && VIRTUAL_ENV=$PWD/envs/core uv pip install -r env-lock/core.txt` |
| `envs/audio` | Stable Audio 3, MiniMax-Music3, MOSS-SFX, OmniVoice, F5, Orpheus, CLAP, Whisper | recipe in `docs/research/audio_models.md` §3, versions in `env-lock/audio.txt` |
| `envs/qwentts` | Qwen3-TTS (transformers 4.57.3) | `uv venv --python 3.12 envs/qwentts && VIRTUAL_ENV=$PWD/envs/qwentts uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple -r env-lock/qwentts.txt` |
| `envs/vllm_omni`, `envs/vllm_omni_overlay` | vLLM-Omni (Higgs, Fish, Voxtral, CosyVoice); the overlay adds fish-speech, DAC, s3tokenizer | `pip install --target envs/vllm_omni_overlay --no-deps -r env-lock/vllm_omni_overlay.txt` |
| `envs/ttslight`, `envs/ttslight_parler` | Kokoro, Piper, StyleTTS 2; Parler-TTS, MeloTTS | `docs/research/audio_more/tts_light.md`, locks `env-lock/ttslight*.txt` |
| `envs/clonea_{xtts,chatterbox,zonos,openvoice,codec}` | voice-cloning TTS group A | `docs/research/audio_more/tts_clone_a.md`, `env-lock/clonea_*.txt` |
| `envs/cloneb_{index,amphion,hf}` | IndexTTS, MaskGCT / Vevo, Dia / Dia2 / Bark / VibeVoice / CSM | `docs/research/audio_more/tts_clone_b.md`, `env-lock/cloneb_*.txt` |
| `envs/music_{audiocraft,acestep,diffrhythm,yue,heartmula,magenta}` (`music_levo` is declared but not installed) | MusicGen / AudioGen / MAGNeT / Riffusion, ACE-Step, DiffRhythm, YuE, HeartMuLa, Magenta RT | `docs/research/audio_more/music.md`, `env-lock/music_*.txt` |
| `envs/sfxmore` | AudioLDM 1/2, Tango 2, TangoFlux | `docs/research/audio_more/sfx_more.md`, `env-lock/sfxmore.txt` |
| `envs/sao` | Stable Audio Open 1.0 / Small (gated) | `docs/research/audio_more/sfx_more.md`, `env-lock/sao.txt` |
| `envs/foley` | MMAudio, HunyuanVideo-Foley, ThinkSound (needs `ffmpeg` on PATH to mux the video) | `docs/research/audio_more/foley.md`, `env-lock/foley.txt` |
| `envs/omni_{moshi,qwen,glm,minicpm,step,kimi,llama2}` | speech-to-speech / omni models | `docs/research/audio_more/omni.md`, `env-lock/omni_*.txt` |
| `envs/vc`, `envs/vc_diffsinger` | Seed-VC, RVC; DiffSinger | `docs/research/audio_more/vc_sing.md`, `env-lock/vc*.txt` |
| `envs/stab3d`, `envs/shape3d`, `envs/trellis`, `envs/rig` | TripoSR, SF3D, SPAR3D, Zero123++, SV3D (Stability AI); TripoSG, Hunyuan3D-2mini, Step1X-3D; TRELLIS (image to 3D, multi-view); UniRig (auto-rigging) | `docs/research/3d_more.md`, `env-lock/stab3d.txt`, `env-lock/shape3d.txt`, `env-lock/trellis.txt`, `env-lock/rig.txt` |
| `envs/qwen`, `envs/hy3d` | diffusers image models; Hunyuan3D-2.1, Real-ESRGAN, rembg (engine code in `engines/hy3d/`) | `docs/SETUP.md`, `env-lock/qwen.txt`, `env-lock/hy3d.txt` |
| `envs/moss` | MOSS-TTS, MOSS-VoiceGenerator (transformers 5.0.0) | `docs/SETUP.md`, `env-lock/moss.txt` |

Each of these envs has a `fix` / recipe in its report and a pip freeze in `env-lock/`; the Models page shows
`missing_env` with the exact command when one is gone. Upstream code the workers import lives in `vendor/`, which is not committed: `scripts/fetch_vendor.sh` clones each project at its pinned commit from `vendor/vendor.lock`.
MusicGen-melody separates stems with demucs `htdemucs`; its weights come from the HF cache
(`models--adefossez--HTDemucs`), so the melody slot works offline.

## Use it from Claude Code and other MCP clients

The core serves an MCP server at **`http://127.0.0.1:7860/api/x/mcp`** (Streamable HTTP, `app/plugins/mcp.py`). It
starts and stops with the studio, so Claude Code, Claude Desktop, Cursor, VS Code or any MCP client can list models,
search assets, **submit jobs** and get the output files back (local paths + URLs, image previews, live progress).
Jobs go through the same queue, memory checks and commercial-safe filter as the UI; cloud jobs that cost money or
upload inputs are refused unless the client passes `confirm_cost` / `confirm_upload` (the error carries the estimate,
so the client can ask you first).

Tools (28):

| Group | Tools |
| --- | --- |
| Look around | `studio_status`, `list_models` (tasks incl. `rig` and `motion`), `get_model` (params, presets, user presets, input slots), `get_rig` (the bones of a rigged mesh: index, name, parent, position, role, weight share), `preview_rig_edit` (dry run of a `rig.edit`: what the ops would change, nothing written; flow `get_rig` -> `preview_rig_edit` -> `submit_job(tool='rig.edit')` -> `submit_job(task='reskin')`), `get_settings` (groups and the values that differ from the defaults; `group` / `q` / `keys` give label, help, default, range; `project` shows a project's effective values and overrides), `list_tools` (each CPU tool with its inputs and params, e.g. `mesh.flex_preview` takes `{clips, angle}` on a rigged mesh, and `rig.edit` takes `{ops}`: add a tail, move / rename / delete bones, Mixamo / VRM / Unreal names; follow it with `submit_job(task='reskin')` when it added bones), `studio_info` (topics: `profiles` / export presets, `voices`, `video_formats`, `dub_options`, `ready_tasks`, `flows`, `cloud`, `llm`, `comfyui`, `downloads`, `logs`) |
| Make | `improve_prompt` (rough idea to a model-ready prompt, or `list_styles`), `submit_job` (waits up to `wait_s`, default 300 s, max 600), `get_job`, `list_jobs`, `cancel_job`, `retry_job` |
| Library | `list_projects`, `create_project` (`update=true` edits one), `search_assets`, `get_asset` (meshes show `mesh3d`: rigged, joints, animation clip names, attachments such as `skeleton.json` / `rigged_plain.glb`; `lineage=true` adds ancestors and derived assets), `update_asset`, `import_file` (clients on the same machine only), `delete_asset` (needs `confirm: true`; moves it to the project's `.trash`) |
| Style | `get_style_guide` (the project's look, avoid list, references, palette and per-kind variants; `history=true`, `version=n`), `set_style_guide` (`guide` / `patch` / `restore_version`, `base_version` guards against overwriting; every save is a new version), `style_from_assets` (draft a guide from 1-4 finished images, optional `describe` by the prompt LLM), `preview_style` (dry run: the exact prompt, negative and references a model would get). The server adds the guide to every image / texture / video / sound job by itself; `submit_job(style='off')` skips it for one job and the result and `get_asset` show what was applied (`x_style`) |
| Deliver | `export_assets` (export preset, zip with meta and attachments, or an engine-ready sound folder), `save_preset` (user presets per model) |
| Configure | `update_settings` (`scope` system or project + `project`, `values` {dotted.key: value}, `reset` [keys]; validated by the same schema as the Settings page, so file-only settings such as executables are refused; a text backend other than `local` needs `confirm_upload`) |

Left out on purpose: provider API keys, cloud on/off and budget, file-only settings (executables, llama-server arguments), model downloads,
unloading models, the Assistant chats and the Pipelines page's flow editing (read-only through `studio_info`). Change
those in the web UI.

### Client configs

Start the studio first (`./run_studio.sh`); the server name `studio` is free to change. Replace `127.0.0.1` with the
studio machine's address to use it from another computer (needs `./run_studio.sh --host 0.0.0.0`; see Security below).

**Claude Code**: in this repository, `.mcp.json` already registers it (approve it once when `claude` asks).
Anywhere else:

```bash
claude mcp add --transport http studio http://127.0.0.1:7860/api/x/mcp              # this folder only
claude mcp add --transport http --scope user studio http://127.0.0.1:7860/api/x/mcp # every folder
```

or a project's `.mcp.json`:

```json
{
  "mcpServers": {
    "studio": { "type": "http", "url": "http://127.0.0.1:7860/api/x/mcp" }
  }
}
```

**Claude Desktop**: `claude_desktop_config.json` (Settings > Developer > Edit Config) starts local servers over
stdio, so point it at the stdio transport of the same tools (stdlib only, any `python3` >= 3.9):

```json
{
  "mcpServers": {
    "studio": {
      "command": "python3",
      "args": ["/path/to/open-asset-studio/scripts/mcp_server.py"],
      "env": { "STUDIO_URL": "http://127.0.0.1:7860" }
    }
  }
}
```

**Cursor**: `~/.cursor/mcp.json` (all projects) or `<project>/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "studio": { "url": "http://127.0.0.1:7860/api/x/mcp" }
  }
}
```

**VS Code** (Copilot agent mode): `.vscode/mcp.json`:

```json
{
  "servers": {
    "studio": { "type": "http", "url": "http://127.0.0.1:7860/api/x/mcp" }
  }
}
```

**Any other client**: HTTP clients use the URL above. Stdio-only clients run
`python3 /path/to/open-asset-studio/scripts/mcp_server.py`, with `STUDIO_URL` set when the core is not on
`http://127.0.0.1:7860`. The stdio transport needs the studio's files on the same machine only for local file paths
and `import_file`; everything else goes over `STUDIO_URL`.

Check it: `claude mcp list` shows `studio ... ✔ Connected`, or
`curl -s -X POST http://127.0.0.1:7860/api/x/mcp -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'`.

Security: there is no authentication. Anyone who can reach the core's port can submit jobs through MCP, the same as
through the REST API (see SECURITY.md). Requests with a browser `Origin` of another site are refused, and
`import_file`, the only tool that reads a path on the studio's machine, is refused for clients that are not on that
machine.

## Tests

```bash
envs/core/bin/python -m pytest          # ~6 s: file-route path safety, registry status, queue/GPU lock/cancel/eviction, worker IPC + parent death
```

## Licence

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
Third-party code and assets: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are downloaded by you
from their publishers and keep their own licences (shown on the Models page and on every asset).

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).
