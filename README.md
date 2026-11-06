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

Requirements: Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), git, ffmpeg, an NVIDIA GPU with a recent driver.

```bash
uv venv --python 3.12 envs/core && VIRTUAL_ENV=$PWD/envs/core uv pip install -r env-lock/core.txt
./run_studio.sh                      # http://127.0.0.1:7860
```

The core starts with no GPU backends installed: the Models page lists every model with its status
(`missing_env` / `missing_weights`) and the command that fixes it. Install only the backends you want; see
[docs/SETUP.md](docs/SETUP.md). The CPU test models (Diagnostics page) work right away.

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
settings.json            user settings: profile, commercial-safe filter, default model per task, memory margin, idle timeouts, LLM config
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
| `envs/foley` | MMAudio, HunyuanVideo-Foley, ThinkSound (needs `ffmpeg` on PATH to mux the video) | `docs/research/audio_more/foley.md`, `env-lock/foley.txt` |
| `envs/omni_{moshi,qwen,glm,minicpm,step,kimi,llama2}` | speech-to-speech / omni models | `docs/research/audio_more/omni.md`, `env-lock/omni_*.txt` |
| `envs/vc`, `envs/vc_diffsinger` | Seed-VC, RVC; DiffSinger | `docs/research/audio_more/vc_sing.md`, `env-lock/vc*.txt` |
| `envs/qwen`, `envs/hy3d` | diffusers image models; Hunyuan3D-2.1, Real-ESRGAN, rembg (engine code in `engines/hy3d/`) | `docs/SETUP.md`, `env-lock/qwen.txt`, `env-lock/hy3d.txt` |
| `envs/moss` | MOSS-TTS, MOSS-VoiceGenerator (transformers 5.0.0) | `docs/SETUP.md`, `env-lock/moss.txt` |

Each of these envs has a `fix` / recipe in its report and a pip freeze in `env-lock/`; the Models page shows
`missing_env` with the exact command when one is gone. Upstream code the workers import lives in `vendor/`, which is not committed: `scripts/fetch_vendor.sh` clones each project at its pinned commit from `vendor/vendor.lock`.
MusicGen-melody separates stems with demucs `htdemucs`; its weights come from the HF cache
(`models--adefossez--HTDemucs`), so the melody slot works offline.

## Tests

```bash
envs/core/bin/python -m pytest          # ~6 s: file-route path safety, registry status, queue/GPU lock/cancel/eviction, worker IPC + parent death
```

## Licence

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
Third-party code and assets: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are downloaded by you
from their publishers and keep their own licences (shown on the Models page and on every asset).

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).
