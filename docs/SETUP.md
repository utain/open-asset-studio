# Setup

The studio is one small core plus many optional GPU backends. Each backend is its own Python venv under `envs/`,
because the models pin incompatible versions of torch, transformers and diffusers. Install only the backends you
want: the Models page shows every model's status and the command that fixes it.

Tested on an NVIDIA DGX Spark (aarch64, GB10 / sm_121, CUDA 13 driver, no `nvcc`, Python 3.12). Several of the
workarounds below exist only because of that machine (no aarch64 wheels, no CUDA compiler). On x86_64 with a CUDA
toolkit installed, you can usually use the upstream install steps instead.

## 1. Core (required)

```bash
uv venv --python 3.12 envs/core
VIRTUAL_ENV=$PWD/envs/core uv pip install -r env-lock/core.txt
./run_studio.sh                       # http://127.0.0.1:7860
envs/core/bin/python -m pytest -q     # optional: core tests, no GPU needed
```

System tools: `git`, `ffmpeg`/`ffprobe` (waveforms, video to sound), `uv`.

## 2. Upstream source code (`vendor/`)

Most backends import upstream projects that are not on PyPI. They are not committed here. Clone them, each at the
commit it was tested with:

```bash
scripts/fetch_vendor.sh                  # all of vendor/vendor.lock (~40 repos)
scripts/fetch_vendor.sh seed-vc MMAudio  # only the ones you need
```

Each project keeps its own licence. Some are GPL, non-commercial or community licences; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## 3. Backend environments

Every backend in `models.yaml` / `models.d/*.yaml` names its venv (`python:`) and its lock file (`lock:` ->
`env-lock/<lock>.txt`, a `uv pip freeze`). The general recipe:

```bash
B=sfxmore                                             # backend venv name
uv venv --python 3.12 envs/$B
VIRTUAL_ENV=$PWD/envs/$B uv pip install --index-url https://download.pytorch.org/whl/cu130 torch torchaudio  # version from the lock
VIRTUAL_ENV=$PWD/envs/$B uv pip install -r env-lock/$B.txt   # run from the repo root: lock files refer to ./vendor/...
```

The exact, tested commands for each backend (extra pins, packages installed with `--no-deps`, patches applied at
runtime) are in the group reports:

| backends | report |
|---|---|
| `audio`, `qwentts`, `moss`, `vllm_omni` | [research/audio_models.md](research/audio_models.md) §3 |
| `ttslight`, `ttslight_parler` | [research/audio_more/tts_light.md](research/audio_more/tts_light.md) |
| `clonea_*` | [research/audio_more/tts_clone_a.md](research/audio_more/tts_clone_a.md) |
| `cloneb_*` | [research/audio_more/tts_clone_b.md](research/audio_more/tts_clone_b.md) |
| `music_*` | [research/audio_more/music.md](research/audio_more/music.md) |
| `sfxmore` | [research/audio_more/sfx_more.md](research/audio_more/sfx_more.md) |
| `foley` | [research/audio_more/foley.md](research/audio_more/foley.md) |
| `omni_*` | [research/audio_more/omni.md](research/audio_more/omni.md) |
| `vc`, `vc_diffsinger` | [research/audio_more/vc_sing.md](research/audio_more/vc_sing.md) |
| `qwen`, `hy3d`, `imgtools` | below, and the headers of `env-lock/qwen.txt` / `env-lock/hy3d.txt` |

### Image and 3D (`envs/qwen`, `envs/hy3d`)

The image workers and the Hunyuan3D worker share the engine code in `engines/hy3d/app/`. Hunyuan3D-2.1 itself is
cloned to `engines/hy3d/vendor/Hunyuan3D-2.1` by `scripts/fetch_vendor.sh`.

```bash
uv venv --python 3.12 envs/qwen
VIRTUAL_ENV=$PWD/envs/qwen uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.0 torchvision==0.24.0
VIRTUAL_ENV=$PWD/envs/qwen uv pip install -r env-lock/qwen.txt
```

`envs/hy3d`: follow the header of `env-lock/hy3d.txt`. Hunyuan3D's texture stage needs its `custom_rasterizer` CUDA
extension, which needs `nvcc` to build (`engines/hy3d/vendor/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer`), plus a
`bpy` stub module where no Blender wheel exists.

### vLLM-Omni (`envs/vllm_omni`)

Higgs TTS, Fish S2, Voxtral TTS and CosyVoice 3 run through [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
(pinned in `vendor/vendor.lock`). Install it into `envs/vllm_omni` following its README, then add the overlay:
`pip install --target envs/vllm_omni_overlay --no-deps -r env-lock/vllm_omni_overlay.txt`.

## 4. Model weights

Weights come from the Hugging Face cache (`~/.cache/huggingface/hub`). The studio runs with `HF_HUB_OFFLINE=1`
and never downloads at runtime. The Models page shows the download command for each missing model. Some models
are gated: accept their terms on Hugging Face and run `hf auth login` first.

Downloads were unreliable through Xet on the test machine, so `HF_HUB_DISABLE_XET=1` is set everywhere.

## 5. Prompt improver (optional)

The ✨ Improve button runs a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`. Put it on
`PATH` or set `LLAMA_SERVER=/path/to/llama-server`. Set the GGUF model in `settings.json` (`llm`); see
[research/prompt_llm.md](research/prompt_llm.md).

## 6. Optional cloud providers

Off by default. Keys are stored in `data/secrets.json` (mode 600, never committed). See
[CONTRACTS.md](CONTRACTS.md) §8.
