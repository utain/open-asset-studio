# Sound studio: open-weight audio models on the DGX Spark

Research and feasibility notes for the Sound studio. It covers every audio model in `~/.cache/huggingface/hub` as of 2026-11-05.

- **Machine:** DGX Spark: aarch64, GB10 (sm_121), 121 GB unified memory, CUDA 13 driver, no nvcc. Everything runs offline (`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`).
- **Test method:** each model got one tiny real generation. Numbers are from one cold process, so `load_s` and the first generation include warm-up.
- **Memory:** "peak GB" is `torch.cuda.max_memory_allocated` for plain-PyTorch backends. For vLLM backends it is the reservation set in the deploy config.
- **QA proxies:** Whisper-large-v3-turbo transcripts for TTS and LAION-CLAP cosine for SFX and music. See [QA proxy results](#qa-proxy-results).
- **Files:** the test audio was generated in a throwaway scratch directory and is not included. Snippets are in `audio_snippets/`.

> Notation: **NC** = non-commercial only. Thai: ✅ = officially listed, ⚠ = listed but weak or untested, ❌ = not supported.

---

## 1. Capability matrix

### 1a. What each model does

| Model | SFX | Amb. | Music | Song+lyrics | TTS | Clone | Voice design | Emotion / instruct | Languages (Thai?) | SR / ch | Max len | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SA3 small-sfx | ✅ | ✅ | – | – | – | – | – | – | EN prompts | 44.1k st | 120 s | Stability Community (free <$1M rev.) |
| SA3 medium | ✅ | ✅ | ✅ instr. | – | – | – | – | – (+inpaint/continue) | EN prompts | 44.1k st | ~380 s | Stability Community |
| MOSS-SoundEffect-v2 | ✅ | ✅ | short | – | – | – | – | – | EN/ZH prompts | 48k | 30 s | Apache-2.0 |
| MiniMax-Music3 | – | – | ✅ | ✅ vocals | – | – | – | style via caption | lyrics languages undocumented (EN tested), Thai ⚠ unknown | 44.1k st | ~5 min | MiniMax Community (commercial OK, credit in UI, <$20M) |
| MOSS-TTS-v1.5 | – | – | – | – | ✅ | ✅ | – | duration ctrl, IPA/pinyin | 31 incl. **Thai ✅** | 24k | long-form | Apache-2.0 |
| MOSS-VoiceGenerator | – | – | – | – | ✅ | – | ✅ prose | ✅ via description | multilingual (Thai ⚠ untested) | 24k | ~80 s | Apache-2.0 |
| MOSS-TTS-Nano | – | – | – | – | ✅ | ✅ req. | – | – | 20, Thai ❌ | 48k | – | Apache (repo: "not yet licensed") |
| Qwen3-TTS CustomVoice | – | – | – | – | ✅ | – | 9 presets | ✅ free-text `instruct` | 10, Thai ❌ | 24k | – | Apache-2.0 |
| Qwen3-TTS Base | – | – | – | – | ✅ | ✅ | – | – | 10, Thai ❌ | 24k | – | Apache-2.0 |
| OmniVoice | – | – | – | – | ✅ | ✅ | ✅ attributes | non-verbal tags | 600+ incl. **Thai ✅** | 24k | – | Apache-2.0 |
| omnivoice-thai | – | – | – | – | ✅ | ✅ | ✅ attributes | non-verbal tags | **Thai** fine-tune | 24k | – | Apache-2.0 |
| F5-TTS v1 Base | – | – | – | – | ✅ | ✅ req. | – | speed / fixed dur. | EN/ZH | 24k | ~30 s/chunk | **CC-BY-NC weights (NC)** |
| Orpheus 3B ft | – | – | – | – | ✅ | – | 8 presets | `<laugh>` etc. | EN | 24k | – | Apache + Llama 3.2 license |
| Higgs TTS 3 4B | – | – | – | – (`style:singing`) | ✅ | ✅ | – | ✅ 43 tags | 100+ incl. **Thai ✅** | 24k | 8k ctx | **NC** (creator grant) |
| Fish S2 Pro | – | – | – | – | ✅ | ✅ | – | ✅ free-form `[tags]` | 80+, Thai ⚠ tier 3 | 44.1k | – | **NC** (Fish research) |
| Voxtral 4B TTS | – | – | – | – | ✅ | ❌ (no encoder) | 20 presets | – | 9, Thai ❌ | 24k | – | **CC-BY-NC** |
| CosyVoice3 0.5B | – | – | – | – | ✅ | ✅ | – | ✅ instruct, `[breath]` | 9 + ZH dialects, Thai ❌ | 24k | – | Apache-2.0 |
| CLAP htsat-unfused | scorer | | | | | | | | | 48k in | | Apache-2.0 |
| Whisper v3 turbo | ASR | | | | | | | | 99 incl. Thai | 16k in | | MIT |

### 1b. Measured on this box (one tiny generation each)

| Model | venv | Status | Load | Test clip | Gen time | RTF | Peak GPU | QA proxy |
|---|---|---|---|---|---|---|---|---|
| SA3 small-sfx | audio | ✅ | 4.5 s | 4 s laser | 3.8 s (0.16 s sampling) | 0.96 cold | 2.1 GB | CLAP 0.18 |
| SA3 medium | audio | ✅ | 14–19 s | 30 s music / 5 s door | 14.6 s / 10 s | 0.49 / 2.0 cold | 8.7 GB | CLAP 0.76 / 0.42 |
| MOSS-SoundEffect-v2 | audio (or the original SFX env) | ✅ | 38 s | 5 s door | 74 s (100 steps) | 14.8 | 14.9 GB | CLAP 0.37 |
| MiniMax-Music3 | audio | ✅ | 160 s | 20 s song | 188 s | 9.4 | 22.8 GB | CLAP 0.39, Whisper hears verse 1 |
| MOSS-VoiceGenerator | moss | ✅ | 66 s | 3.3 s | 4.6 s | 1.4 | 10.7 GB | ASR exact |
| MOSS-TTS-v1.5 | moss | ✅ | 104–133 s | 3.4 s EN / 8.2 s TH | 23 s / 16 s | 6.8 / 2.0 | 22.8 GB | ASR exact (EN, TH) |
| MOSS-TTS-Nano | – | ❌ missing 88 MB codec | | | | | | |
| Qwen3-TTS CustomVoice | qwen | ✅ | 27 s | 6.4 s | 5.4 s | 0.85 | 4.1 GB | ASR exact |
| Qwen3-TTS Base | qwen | ✅ | 28 s | 3.4 s clone | 3.5 s | 1.0 | 4.3 GB | ASR exact |
| OmniVoice | audio | ✅ | 5.6 s | 2.9 s clone/design, 3.9 s TH | 1.2–1.9 s | 0.3–0.6 | 2.1 GB | ASR exact (EN, TH) |
| omnivoice-thai | audio | ✅ | 5.6 s | 3.9 s TH | 1.2 s | 0.32 | 2.0 GB | ASR exact (TH) |
| F5-TTS | audio | ✅ | 2.2 s | 4.1 s clone | 2.7 s | 0.67 | 1.9 GB | ASR exact, peak 1.16 (clips) |
| Orpheus 3B | audio | ✅ | 40 s | 5.1 s | 41.6 s | 8.1 | 6.4 GB | ASR 12/13 words |
| Higgs TTS 3 | vllm-omni | ✅ | 90–165 s engine | 2.8 s plain / clone | 3.8–4.0 s | 1.4 | 19 + 7 GB reserved | ASR exact |
| Fish S2 Pro | vllm-omni + overlay | ✅ | ~155 s engine | 3.5 s | 24.9 s (1st req, JIT) | ~7 cold | 22 + 7 GB reserved | ASR exact |
| Voxtral TTS | vllm-omni | ✅ presets / ❌ clone | ~85 s engine | 2.9 s | 3.5 s | 1.22 | 17 + 7 GB reserved | ASR exact |
| CosyVoice3 | vllm-omni + overlay | ✅ (3 workarounds) | ~60 s engine | 5.7 s clone | (70 s wall incl. start) | – | 10 + 7 GB reserved | ASR exact |
| CLAP / Whisper | audio | ✅ | ~5 s | 21 clips scored | <1 s/clip | – | ~2 GB | – |

RTF = generation time / audio duration, measured cold on the first request. Warm calls are faster, especially for vLLM (CUDA graphs) and SA3 (about 10 s of first-call overhead). vLLM "peak" is the reserved `gpu_memory_utilization` × 121 GB from my studio deploy configs, not actual use.

## 2. Recommended defaults per task

| Task | Default (commercial-safe) | Alternatives / when to switch |
|---|---|---|
| One-shot SFX, UI sounds | **Stable Audio 3 small-sfx** (8 steps, sub-second warm, 2 GB) | SA3 medium for richer, longer SFX. MOSS-SoundEffect-v2 for Chinese prompts or as a second opinion (slow, about 15× RT at 100 steps). Rank 3–4 seeds with CLAP. |
| Ambience / loops | **SA3 medium** (up to ~380 s, stereo) | SA3 small-sfx up to 120 s; MOSS-SFX up to 30 s. Inpaint with SA3 medium to make seamless loop seams. |
| Instrumental music / BGM | **SA3 medium** (add BPM and key to the prompt) | MiniMax-Music3 with `[instrumental]` sections for long, structured tracks. ACE-Step in a ComfyUI container (not touched here). |
| Song with lyrics and vocals | **MiniMax-Music3** (the only local model with vocals; RTF about 9) | ACE-Step via ComfyUI. Higgs `<|style:singing|>` for short sung lines only. |
| TTS, preset voices + emotion instruction | **Qwen3-TTS CustomVoice** (9 voices, free-text `instruct`, 10 langs) | Higgs TTS 3 (best tag control, NC). Fish S2 Pro free-form tags (NC). Orpheus `<laugh>` (EN). |
| Voice clone | **OmniVoice** (fast, 2 GB, 600+ langs) | MOSS-TTS-v1.5 (higher fidelity, long-form, 23 GB). Qwen3 Base (10 langs). CosyVoice3. F5 / Higgs / Fish are NC. |
| Voice design from a description | **MOSS-VoiceGenerator** (free prose) → clone with MOSS-TTS or OmniVoice | OmniVoice `instruct` (fixed attribute list, EN/ZH-trained). Qwen3 VoiceDesign is not cached. |
| **Thai** speech | **OmniVoice / omnivoice-thai** (clone or design; both exact on Whisper) | MOSS-TTS-v1.5 with `language="th"` (exact, but 52 % silence: trim it). Higgs TTS 3 lists Thai (NC, untested in Thai). |
| QA / ranking | **Whisper large-v3-turbo** (word or character recall), **CLAP** (prompt cosine vs. "silence" and "speech" anchors) | – |

If the studio must stay commercial-safe, hide or flag the **NC** backends: Higgs, Fish, Voxtral and F5 weights. Stability, MiniMax and Orpheus/Llama need attribution or registration.

## 3. Backend → venv map

Four Python environments cover all 21 models. Each backend runs as a worker subprocess `<venv>/bin/python worker_<backend>.py` that talks JSON-lines over stdin/stdout.

| venv | Python / torch | Key pins | Backends |
|---|---|---|---|
| **`envs/audio`** (new, tested) | 3.12, torch 2.9.1+cu130, torchaudio 2.9.1 | transformers 5.22.1, diffusers git `a91eccd`, stable-audio-tools git `3241adb` (`--no-deps`), omnivoice 0.2.1, f5-tts 1.1.22 (`--no-deps`), moss-soundeffect-v2 git `934d682` (`--no-deps`), snac 1.2.1, vocos, pytorch_lightning, descript-audiotools (`--no-deps`) | Stable Audio 3 medium and small-sfx, MiniMax-Music3, MOSS-SoundEffect-v2, OmniVoice, omnivoice-thai, F5-TTS, Orpheus, CLAP, Whisper |
| **`envs/qwen`** (new, tested) | 3.12, torch 2.9.1+cu130 | qwen-tts 0.1.1 → **transformers 4.57.3**, accelerate 1.12 | Qwen3-TTS CustomVoice and Base |
| **`moss`** = `envs/moss` | 3.12, torch 2.9.1+cu130 | **transformers 5.0.0** exactly, editable `moss_tts` (OpenMOSS/MOSS-TTS) | MOSS-TTS-v1.5, MOSS-VoiceGenerator (and MOSS-TTS-Nano once its 88 MB codec is downloaded) |
| **`vllm-omni`** = existing `envs/vllm_omni`, read-only, plus `PYTHONPATH=envs/vllm_omni_overlay` | 3.12, torch 2.11.0+cu130 | vllm 0.25.0, vllm-omni 0.25.0rc2.dev41 (editable source checkout `109942465`), transformers 5.14.1. Overlay (`--target`, `--no-deps`): fish-speech@0f8a12e, descript-audio-codec, descript-audiotools, s3tokenizer, onnx, soxr and small helpers | Higgs TTS 3, Fish S2 Pro, Voxtral TTS, CosyVoice3. Qwen3-TTS, OmniVoice and MOSS-TTS have untested vLLM-Omni paths here too. |

Why they cannot be merged further:

- `qwen-tts` hard-pins transformers 4.57.3.
- MOSS-TTS remote code breaks on transformers > 5.0.0 (`ProcessorMixin` rejects `revision`).
- OmniVoice cloning needs ≥5.3.
- The vLLM stack brings its own torch 2.11. For the app, a dedicated `envs/vllm` is cleaner than borrowing `<vllm-omni checkout>`, but building vLLM for sm_121 / aarch64 is a large separate job, so the plan reuses the working one.

Recipe for `envs/audio` (exact versions in `audio_snippets/lock_audio.txt`):

```bash
cd <repo>/envs && uv venv --python 3.12 audio && export VIRTUAL_ENV=$PWD/audio
echo -e "torch==2.9.1+cu130\ntorchaudio==2.9.1" > /tmp/pin.txt
IDX="--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -c /tmp/pin.txt"
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1 torchaudio==2.9.1
uv pip install --no-deps "stable-audio-tools @ git+https://github.com/Stability-AI/stable-audio-tools@3241adb" \
    v-diffusion-pytorch==0.0.2 f5-tts==1.1.22 descript-audiotools flatten-dict julius ffmpy randomname argbind \
    "moss-soundeffect-v2 @ git+https://github.com/OpenMOSS/MOSS-TTS@934d682#subdirectory=moss_soundeffect_v2"
uv pip install $IDX transformers==5.22.1 accelerate alias-free-torch==0.0.6 dill einops einops-exts torchsde nnAudio \
    PyWavelets safetensors scipy sentencepiece soxr tqdm vector-quantize-pytorch soundfile librosa numpy pytorch_lightning \
    "diffusers @ git+https://github.com/huggingface/diffusers@a91eccd" omnivoice==0.2.1 snac \
    cached_path click ema_pytorch hydra-core pydub pypinyin rjieba tomli torchdiffeq transformers_stream_generator \
    unidecode vocos x_transformers matplotlib datasets wandb importlib_resources tensorboard ftfy imageio regex
```

Recipe for `envs/qwen`:

```bash
uv venv --python 3.12 qwen && VIRTUAL_ENV=$PWD/qwen uv pip install $IDX torch==2.9.1 torchaudio==2.9.1 qwen-tts==0.1.1
```

## 4. Per-model notes

### 4.1 Stable Audio 3: `stabilityai/stable-audio-3-medium` and `stable-audio-3-small-sfx`

The user calls these "Stable Audio 3.1". The model cards and the code only say "Stable Audio 3" (Medium 2B, Small-SFX 0.6B).

- **Tasks.**
  - Medium: music (instrumental), ambience, SFX, and inpainting/continuation (`diffusion_cond_inpaint`).
  - Small-SFX: sound effects.
  - No vocals or lyrics. Prompts are English only.
- **Output.**
  - 44.1 kHz stereo.
  - Max length is set by `sample_size`: Medium 16 777 216 samples ≈ **380 s**, Small-SFX 5 292 032 ≈ **120 s**.
  - `seconds_total` is conditioned 0–384.
- **Parameters.** These are ARC-distilled `rf_denoiser` models.
  - `steps=8`, `cfg_scale=1.0`, `sampler_type="pingpong"` (the only sampler offered for rf_denoiser).
  - `seed`, optional `negative_conditioning`, `apg_scale` (1.0), `cfg_interval`.
  - Inpainting: `inpaint_audio=(sr, tensor)` plus `inpaint_mask_start_seconds` / `inpaint_mask_end_seconds`.
  - `adapt_duration_to_conditioning` is on automatically: it generates `seconds + 6 s` and you cut to `seconds`.
- **License.**
  - Stability AI Community License: free, including commercial use, under USD 1 M annual revenue. Commercial use requires registering at stability.ai/community-license and showing "Powered by Stability AI".
  - Outputs may not be used to train other foundation models.
  - The T5Gemma text encoder adds the Gemma terms.
- **Library.**
  - `stable-audio-tools @ git+https://github.com/Stability-AI/stable-audio-tools@3241adb` (main, 2026-05-26). Only main has `taae_v2`, `t5gemma`, `rf_denoiser` and `pingpong`. The PyPI release 0.0.20 pins torch 2.7.1 and python <3.11.
  - Install it `--no-deps` into py3.12, then add its deps by hand plus `pytorch_lightning`, which the LoRA module imports even for inference.
  - `flash_attn` is optional and falls back to SDPA. A newer `stable-audio-3` package (`StableAudioModel.from_pretrained("medium")`) also exists but was not tested.
- **Measured** (fp16, cold process, first call includes about 10 s of warm-up):
  - Small-SFX: 4 s clip, load 4.5 s, gen 3.8 s (the 8 denoiser steps themselves take 0.16 s), peak 2.1 GB.
  - Medium: 30 s music in 14.6 s, 5 s SFX in 10 s, load 14–19 s, peak 8.7 GB for both lengths.
  - Warm generation is about 1–2 s for short clips.
- **Quality proxies.**
  - Music 30 s: no silence, peak hits 1.0 (0.025 % of samples clipped), so peak-normalise or limit the output.
  - SFX: 28–30 % silence, the expected tail after a one-shot. CLAP results are in the [QA proxy results](#qa-proxy-results) section.
- **Failure modes.** Clipping on loud music. English prompts only. The duration is a hint: the model fades or pads within the window.
- **Worker fit.** Excellent: one process, model resident (2–9 GB), sub-second to a few seconds per job.

### 4.2 `MiniMaxAI/MiniMax-Music3`: songs with lyrics and vocals

- **Tasks.** Full songs with vocals from lyrics plus a style description, up to about 5 min (9 000 frames at 25 fps). Instrumental is possible with `[instrumental]` sections or an instrumental description.
- **Output.** 44.1 kHz stereo from the vocoder. The reference server resamples to 32 kHz.
- **Parameters.**
  - `prompt` (description, up to 5 000 tokens), `lyrics` (structure-tagged), `audio_duration` (an upper bound: the LM can stop earlier), `generator` seed.
  - Flow-matching CFG 1.7 via `pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=...))`. The flow stage ran 120 steps.
- **License.** MiniMax-Music3 Community License.
  - Commercial use is allowed if "MiniMax-Music3" is shown prominently in the product UI.
  - Above USD 20 M annual revenue you need written authorisation.
  - The AUP applies, and hosted services must have safeguards.
- **Library.**
  - `diffusers @ git+https://github.com/huggingface/diffusers@a91eccd` (main, 2026-11-04) with transformers 5.22.1. The model card pins PR commit `dafe3733…`; main now contains it.
  - API: `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")`, `pipe.load_components(...)`.
- **Gotchas (both are in the snippet).**
  1. With `HF_HUB_OFFLINE=1`, diffusers calls `model_info()` for sharded components, so you must pass `local_files_only=True`.
  2. Loading to CPU and then `.to("cuda")` doubles RAM on unified memory. It was OOM-killed twice while other jobs held about 40 GB. Pass `device_map="cuda"`.
  3. The cache has two snapshots: `1b525cc` is `main` and complete (27 GB). `fbdf52f` is an older, partial one (36 KB).
- **Measured.**
  - 20 s song: load 160 s, generation 188 s (RTF 9.4: the AR stage takes about 135 s, the flow stage 51 s), peak 22.8 GB.
  - Output was 20.0 s stereo, 2.7 % silence, peak 0.97, no clipping.
  - Expect a 2–3 min song to take roughly 20–30 min.
- **Worker fit.** OK as a dedicated long-job worker: about 23 GB resident and minutes per job. Report progress and allow cancellation. Do not keep it resident next to other big models.

### 4.3 `OpenMOSS-Team/MOSS-SoundEffect-v2.0`

- **Tasks.** SFX, ambience, creatures, human actions, short music-like clips. Prompts in English or Chinese.
- **Output.** 48 kHz (mono as written by the worker). Up to **30 s**.
- **Parameters.** `prompt`, `negative_prompt`, `seconds` (≤30), `num_inference_steps` (100), `cfg_scale` (4.0), `seed`.
- **License.** Apache-2.0.
- **Library.** `moss_soundeffect_v2` from the OpenMOSS/MOSS-TTS repo (`subdirectory=moss_soundeffect_v2`, commit 934d682).
  - It works in the original SFX env (transformers 4.57.1, diffusers 0.37.1).
  - It also works in **envs/audio** (transformers 5.22.1, diffusers main), installed `--no-deps` plus `descript-audiotools` `--no-deps`, `flatten-dict`, `julius`, `ffmpy`, `randomname`, `argbind`, `importlib_resources`, `tensorboard`, `ftfy` and `imageio`.
- **Measured** (audio env): 5 s at 100 steps took 74 s, load 38 s, peak 14.9 GB.
  - That is slow (RTF 15). Use about 50 steps for drafts.
  - 58 % silence on a door creak, which is plausible for a one-shot.
- **Worker fit.** Good: this is the pattern already proven in the earlier standalone SFX worker.

### 4.4 `OpenMOSS-Team/MOSS-TTS-v1.5` (+ `MOSS-Audio-Tokenizer`)

- **Tasks.** Zero-shot voice cloning from a reference clip, long-form speech, token-level duration control, Pinyin/IPA pronunciation, code-switching. **31 languages including Thai (th)**.
- **Output.** 24 kHz mono, 12.5 tokens/s.
- **Parameters.**
  - `text`, `reference=[codes]` (encoded with `processor.encode_audios_from_wav`), `language` (set it: v1.5 is clearly better with it).
  - `max_new_tokens = 12.5 × seconds`.
  - Sampling: `audio_temperature=1.7`, `audio_top_p=0.8`, `audio_top_k=25`, `audio_repetition_penalty=1.0`, and a seed.
- **License.** Apache-2.0.
- **Library.** `trust_remote_code` with **transformers 5.0.0 exactly**.
  - transformers 5.22.1 fails with `TypeError: Unexpected keyword argument revision` in `ProcessorMixin.__init__`, so it cannot share the audio env.
  - Use `envs/moss` (torch 2.9.1 cu130, editable `moss_tts` from the vendor repo).
  - Disable cuDNN SDPA (`torch.backends.cuda.enable_cudnn_sdp(False)`).
- **Measured.**
  - Load 104–133 s (16 GB of shards plus the 6.7 GB tokenizer), peak 22.7 GB.
  - English clone: 3.4 s audio in 23 s. Thai clone: 8.2 s audio in 16 s.
  - The Thai output is 52 % silence, which is suspicious; see [QA proxy results](#qa-proxy-results).
- **Worker fit.** Good: proven in `tts_worker.py`. Keep it resident only if VRAM allows (about 23 GB).

### 4.5 `OpenMOSS-Team/MOSS-VoiceGenerator`

- **Tasks.** Voice design: speech in a new voice from a natural-language description. No reference is needed. Its output is used as the reference clip for MOSS-TTS or another cloner.
- **Parameters.**
  - `instruction` (description) and `text`. The processor needs `normalize_inputs=True`.
  - Sampling: `audio_temperature=1.5`, `top_p=0.6`, `top_k=50`, `repetition_penalty=1.1`, `max_new_tokens=1000`.
- **License.** Apache-2.0. Same env as MOSS-TTS (transformers 5.0.0).
- **Measured.** Load 66 s, 3.3 s of audio in 4.6 s, peak 10.7 GB.

### 4.6 `OpenMOSS-Team/MOSS-TTS-Nano` (0.1B): not runnable offline

- **Tasks.** Voice cloning (a reference is required) in 20 languages: zh, en, de, es, fr, ja, it, hu, ko, ru, fa, ar, pl, pt, cs, da, sv, el, tr. No Thai. 48 kHz mono.
- **Missing file.** Its codec `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (**88 MB**, one safetensors file) is not in the cache. That counts as a small missing file, so I did not download it.
  - Code path: the upstream `github.com/OpenMOSS/MOSS-TTS-Nano` package (`pip install -e .`, plus `WeTextProcessing`), or vLLM-Omni `examples/offline_inference/text_to_speech/moss_tts_nano`.
- **License.** Apache-2.0 on the card, but the repo README says "treat as not yet licensed for redistribution" until its LICENSE file is published.

### 4.7 `OpenMOSS-Team/MOSS-Audio-Tokenizer`

This is a codec, not a generator. MOSS-TTS-v1.5 and VoiceGenerator load it through remote code: about 6.7 GB on disk and about 12 GB of the MOSS peak.

### 4.8 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` and `-Base`

- **Tasks.**
  - CustomVoice: 9 preset speakers (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) plus a free-text `instruct` for emotion, tone and speed.
  - Base: zero-shot clone from `ref_audio` + `ref_text`, or `x_vector_only_mode=True` without a transcript (lower quality). `create_voice_clone_prompt` builds a reusable speaker prompt.
  - **10 languages**: zh, en, ja, ko, de, fr, ru, pt, es, it. **No Thai.**
  - VoiceDesign is a separate checkpoint that is not cached.
- **Output.** 24 kHz mono.
- **Parameters.** `language` ("Auto" or a name), `speaker`, `instruct`, plus any HF `generate` kwargs (`max_new_tokens`, `temperature`, `top_p`).
- **License.** Apache-2.0.
- **Library.** `qwen-tts==0.1.1`, which **pins transformers==4.57.3** and accelerate 1.12. It therefore gets its own venv, `envs/qwen`.
  - **Gotcha.** transformers 4.57.x `_patch_mistral_regex` calls `model_info()` even when offline, so pass `snapshot_download(repo, local_files_only=True)` (a local path) instead of the repo id.
  - `flash_attn` is not installed, so it uses SDPA.
- **Measured.**
  - CustomVoice: load 27 s, 6.4 s of audio in 5.4 s (RTF 0.85), peak 4.1 GB.
  - Base clone: 3.4 s of audio in 3.5 s, peak 4.3 GB.
  - vLLM-Omni also supports both checkpoints with streaming (`deploy/qwen3_tts.yaml`), but that was not tested.
- **Worker fit.** Excellent.

### 4.9 `k2-fsa/OmniVoice` and `hotdogs/omnivoice-thai`

- **Tasks.**
  - Three modes: clone (`ref_audio` + `ref_text`), voice design (`instruct`: comma-separated attributes from fixed categories) and auto (random voice).
  - Inline non-verbal tags and pinyin/phoneme pronunciation fixes.
  - **600+ languages including Thai.** Voice design is trained on zh/en only and is "unstable" for other languages.
- **Design attributes.**
  - gender: male / female
  - age: child, teenager, young adult, middle-aged, elderly
  - pitch: very low … very high
  - style: whisper
  - accents and Chinese dialects
- **Output.** 24 kHz mono.
- **Parameters.**
  - `num_step=32` (16 is faster), `guidance_scale=2.0`, `t_shift=0.1`.
  - `position_temperature=5.0`, `class_temperature=0.0`, `layer_penalty_factor=5.0`, `denoise=True`.
  - `speed` or `duration` (fixed seconds), `language`, and `preprocess_prompt` / `postprocess_output` (silence trimming).
- **`hotdogs/omnivoice-thai`.**
  - A Thai fine-tune (about 12.6 h of data, 2 speakers, ASR transcripts, about 1.7 k steps).
  - Same API. It has no `audio_tokenizer/`, so omnivoice falls back to `eustlb/higgs-audio-v2-tokenizer`, which is cached.
  - The cache also holds optimizer and scheduler states (about 4 GB of training leftovers).
- **License.** Both are Apache-2.0. The Thai fine-tune's training data licenses are unclear (Thanarit/Thai-Voice-Test7).
- **Library.** `omnivoice==0.2.1`, which needs transformers ≥5.3 for cloning. Installed in **envs/audio**.
- **Measured.** Load 5.6 s, peak about 2 GB, 3–4 s of audio in 1.2–1.9 s (RTF 0.3–0.6). The base and Thai models perform the same.
- **Worker fit.** Excellent. This is the fastest and smallest multilingual cloner.

### 4.10 `SWivid/F5-TTS` (F5TTS_v1_Base)

- **Tasks.** Zero-shot cloning (reference ≤12 s plus transcript), English and Chinese. No instructions or emotion tags. Speed and fixed duration are controllable.
- **Output.** 24 kHz, vocoder `charactr/vocos-mel-24khz` (cached).
- **Parameters.** `nfe_step=32`, `cfg_strength=2`, `sway_sampling_coef=-1`, `speed`, `fix_duration`, `cross_fade_duration`, `target_rms=0.1`, `remove_silence`, `seed`.
- **License.**
  - Code: MIT.
  - **Weights: CC-BY-NC-4.0** (trained on Emilia), so non-commercial.
- **Library.** `f5-tts==1.1.22` installed `--no-deps` into envs/audio, plus its deps minus `torchcodec` and `bitsandbytes`.
  - **Gotcha.** torchaudio 2.9's `load()` needs torchcodec, which has no aarch64 wheel. The snippet monkey-patches `torchaudio.load` to use soundfile.
  - `<F5-TTS checkout>/.venv` (torch 2.10 with a self-built torchcodec 0.10) also works.
- **Measured.** Load 2.2 s, 4.1 s of audio in 2.7 s, peak 1.9 GB.
  - The waveform peak is 1.16 before clipping: normalise the output (or keep `target_rms`).
  - 41 % silence, because the trailing silence is not trimmed by default.

### 4.11 `unsloth/orpheus-3b-0.1-ft`

- **Tasks.** English TTS with 8 preset voices (tara, leah, jess, leo, dan, mia, zac, zoe) and inline emotive tags `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>`. No cloning in the ft model.
- **Output.** SNAC 24 kHz (`hubertsiuzdak/snac_24khz`, cached).
- **License.** Apache-2.0 on the card, but the model is a Llama-3.2-3B fine-tune, so the **Llama 3.2 Community License** also applies (attribution, AUP, 700 M MAU clause).
- **Library.** Plain transformers generate plus snac, in envs/audio. The official `orpheus-speech` package needs vLLM.
  - Prompt tokens: `128259 + tok("voice: text") + 128009 128260`.
  - Output codes come after token 128257: 7 tokens per frame, offset `128266 + k·4096`.
  - Sampling: temperature 0.6, top_p 0.95, repetition_penalty 1.1.
- **Measured.** Load 40 s, 5.1 s of audio in 41.6 s (RTF 8, unbatched HF generate, about 85 tok/s), peak 6.4 GB.
  - vLLM would be about 5–10× faster but is not wired up.

### 4.12 `bosonai/higgs-tts-3-4b` (+ `bosonai/higgs-audio-v2-tokenizer`)

- **Tasks.**
  - Expressive conversational TTS in **100+ languages (Thai listed in the <5 % WER tier)**.
  - Zero-shot clone (`ref_audio` + optional `ref_text`).
  - Inline control tags: 21 emotions, 3 styles, 9 SFX and 10 prosody tags (see §5).
- **Output.** 24 kHz mono, 25 fps, context 8 192 tokens.
- **Parameters.** Clone sampling `temperature=0.8`, `top_k=50`, `max_new_tokens=1024` (the example default is 2048).
- **License.** **Research and non-commercial only.** A "Creator Use Grant" allows monetised podcasts and videos with credit. Embedding it in a product needs a commercial license.
- **Library.** vLLM-Omni `Omni` offline engine (stage 0 Qwen3 talker plus stage 1 codec), using the existing `envs/vllm_omni` read-only.
  - Model type `higgs_multimodal_qwen3`.
  - Use my low-memory copy of `higgs_multimodal_qwen3_low_latency.yaml` (stage 0 at 0.16 = 19 GB, stage 1 at 0.06).
- **Measured.**
  - Engine start 90–165 s (weights 7.6 GB, KV 11.6 GB, CUDA-graph capture).
  - Then 2.8 s of audio in 3.8–4.0 s (first request, RTF 1.4) for both plain and clone.
  - The user's own earlier `vllm-omni serve bosonai/higgs-tts-3-4b` run produced `<vllm-omni checkout>/output.wav`.
- **Worker fit.** Use it as a resident engine: either a long-lived `Omni` object inside the worker, or `vllm-omni serve` on a private port with the worker as an HTTP client. Never start it per job.

### 4.13 `fishaudio/s2-pro`

- **Tasks.**
  - TTS with zero-shot cloning in 80+ languages (Thai is in the "other" tier).
  - **Free-form inline `[tag]` instructions** such as `[whisper]`, `[laughing]`, `[pause]`, `[angry]`, `[pitch up]` or `[professional broadcast tone]` (15 000+ tags seen in training).
- **Output.** 44.1 kHz mono (DAC codec, 10 codebooks at about 21 Hz).
- **License.** **Fish Audio Research License: non-commercial.** Commercial use needs a separate license.
- **Library.** vLLM-Omni (dual-AR `fish_qwen3_omni`). It needs the `fish_speech` DAC module, which the vllm-omni venv lacks.
  - I built a **PYTHONPATH overlay** `envs/vllm_omni_overlay` (`--target`, `--no-deps`: fish-speech@0f8a12e, descript-audio-codec, descript-audiotools and a few small modules) so the read-only venv is not modified.
  - Deploy: my low-memory `fish_qwen3_omni_studio.yaml` (0.18 / 0.06).
- **Measured.** Engine start about 155 s, 3.5 s of audio in 24.9 s for the first request (includes FlashInfer autotune and JIT), peak 0.50, 22 % silence.

### 4.14 `mistralai/Voxtral-4B-TTS-2603`

- **Tasks.**
  - TTS with **20 preset voices** (`voice_embedding/*.pt`: casual, cheerful and neutral × male/female, plus ar, de, es, fr, hi, it, nl and pt voices).
  - 9 languages: en, fr, es, de, it, pt, nl, ar, hi. **No Thai.**
  - **No voice cloning with the open weights.** `encode_waveforms requires encoder weights which are not available in the open-source checkpoint` (verified). Cloning only works on Mistral's hosted API.
- **Output.** 24 kHz. The deploy defaults are temperature 0, `repetition_penalty=1.1`, `cfg_alpha=1.2`, `max_tokens=2048`.
- **License.** **CC-BY-NC-4.0** (inherited from the reference voices): non-commercial.
- **Library.** vLLM-Omni (`voxtral_tts`, mistral tokenizer/config/load format), using my low-memory `voxtral_tts_studio.yaml` (0.14 / 0.06).
  - **Gotcha.** With `HF_HUB_OFFLINE=1`, pass the **local snapshot directory** as `--model`. With the repo id, model-type detection from `params.json` fails ("Could not determine model_type").
  - The clone path also needs `soxr` (added to the overlay), but it then fails on the missing encoder.
- **Measured.** Engine start about 85 s, 2.9 s of audio at RTF 1.22 on the first request, Whisper transcript exact.

### 4.15 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`

- **Tasks.**
  - Zero-shot multilingual and cross-lingual cloning.
  - 9 languages (zh, en, ja, ko, de, es, fr, it, ru) plus 18+ Chinese dialects. **No Thai.**
  - Instruct mode for dialect, emotion, speed and volume (`inference_instruct2`), and inline `[breath]`, `[laughter]` and pinyin fixes.
- **Output.** 24 kHz.
- **License.** Apache-2.0.
- **Library.** vLLM-Omni `cosyvoice3` (talker plus flow/HiFiGAN code2wav), low-memory `cosyvoice3_studio.yaml` (0.08 / 0.06). Three workarounds, all in `vllm_omni_tts.sh`:
  1. The cached `config.json` is `{}`, so the wrapper builds a symlinked model dir with `{"model_type": "cosyvoice3"}`.
  2. It needs `s3tokenizer`, which the overlay adds.
  3. `s3tokenizer.load_model("speech_tokenizer_v3_25hz")` **downloads 925 MB from modelscope** into `~/.cache/s3tokenizer`. My first run started that download; I killed it and deleted the partial file. The file is byte-identical (sha256 `23236a74…`) to the cached `speech_tokenizer_v3.onnx`, so the wrapper symlinks it.
  - TensorRT is missing, so the campplus speaker embedding falls back to ONNX Runtime (fine).
  - The example writes `output_0.wav` labelled **22 050 Hz**, but the model is 24 kHz (`cosyvoice3.yaml: sample_rate: 24000`). Relabel the output or it plays slow and low.
  - The prompt text must be `You are a helpful assistant.<|endofprompt|>` + the reference transcript.
- **Measured.** Wall time 70 s including engine start, 5.7 s of audio, Whisper transcript exact.
- **Alternative.** The upstream FunAudioLLM/CosyVoice repo was not tried: it is heavy, with pinned deps and onnxruntime-gpu.

### 4.16 Helpers: `laion/clap-htsat-unfused` and `openai/whisper-large-v3-turbo`

- **CLAP** (BSD-3 / Apache):
  - Text–audio cosine similarity at 48 kHz. Use it to rank N SFX or music candidates against the prompt, and to compare against "silence" and "speech" anchors to reject failed generations.
  - About 0.6 GB in memory. `transformers.ClapModel` in the audio env; the `main` snapshot `8fa0f1c` has `pytorch_model.bin`.
- **Whisper large-v3-turbo** (MIT):
  - ASR for TTS: transcribe, then word or character recall against the script. Pass `language` for Thai.
  - About 1.6 GB, `pipeline("automatic-speech-recognition")`.
- Both are in `qa_score.py`.

## 5. Prompt formats (for the prompt-improver LLM)

These are the rules to give the prompt-improver LLM for each backend. Anything not listed takes plain natural-language text.

| Backend | Field(s) | Format rules for the LLM |
|---|---|---|
| **Stable Audio 3** | `prompt` (English), `seconds_total` | One comma-separated descriptive sentence: source, action, material, space/reverb, mood. For music add genre, instruments, mood, **"NNN BPM"** and key, e.g. `House music … festival … 124 BPM`. SFX: short and concrete (`chugging train coming into station with horn`). No lyrics. Optional negative prompt. |
| **MOSS-SoundEffect v2** | `prompt` (EN/ZH), `negative_prompt`, `seconds` ≤30 | Natural caption of the scene (`A dog barking loudly in a park.`). Duration goes in `seconds`, not in the text. |
| **MiniMax-Music3** | `prompt`, `lyrics`, `audio_duration` | `prompt` = structured caption in three parts. **Global metadata**: `Genre: … BPM: … Key: …`, emotional arc, scenario. **Vocals**: gender, timbre, style, harmonies (always state the vocal, or it may drift instrumental). **Arrangement**: instruments per section. `lyrics` = section tags on **their own line**: `[intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo] [outro]`. Text after a tag on the same line is dropped. Lyric language is not documented (EN/ZH are safe). Upstream has a `music-caption-rewriter` skill (`npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter`). |
| **Higgs TTS 3** | `input` | Tags are exactly `<|category:value|>`, and only the 43 known tags work. Unknown tags are read aloud or degrade output. **Sentence-start**: `<|emotion:X|>` (affection, amusement, anger, arousal, awe, bitterness, confusion, contemplation, contentment, determination, disgust, elation, enthusiasm, fear, helplessness, longing, pride, relief, sadness, shame, surprise), `<|style:singing/shouting/whispering|>`, `<|prosody:speed_very_slow/speed_slow/speed_fast/speed_very_fast/pitch_low/pitch_high/expressive_high/expressive_low|>`. **Inline**: `<|prosody:pause|>`, `<|prosody:long_pause|>`, and `<|sfx:X|>` + onomatopoeia **with no space** (`<|sfx:laughter|>Haha, …`; X = cough, laughter, crying, screaming, burping, humming, sigh, sniff, sneeze). |
| **Fish S2 Pro** | `text` | Free-form bracket instructions inline at the word: `[whisper]`, `[excited]`, `[laughing]`, `[pause]`, `[short pause]`, `[sigh]`, `[angry]`, `[low voice]`, `[volume up]`, `[pitch up]`, `[professional broadcast tone]` … Keep them short and English. |
| **OmniVoice** (+Thai) | `text`, `instruct`, `language` | `instruct` = comma-separated attributes, **one per category**: gender (male/female), age (child, teenager, young adult, middle-aged, elderly), pitch (very low … very high pitch), style (whisper), accent (e.g. british accent) or Chinese dialect. Not free prose. Inline non-verbals in `text`: `[laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh] [question-ei] [question-yi] [surprise-ah] [surprise-oh] [surprise-wa] [surprise-yo] [dissatisfaction-hnn]`. Pinyin or phoneme overrides are allowed for pronunciation. |
| **Qwen3-TTS CustomVoice** | `text`, `speaker`, `language`, `instruct` | `instruct` = free natural-language direction of emotion, tone and pace (`Very happy.`, `Speak in an incredulous tone, with a hint of panic`). `speaker` ∈ 9 presets. 10 languages, no Thai. |
| **Qwen3-TTS Base** | `text`, `ref_audio`, `ref_text` | No style control; the style comes from the reference clip. |
| **MOSS-VoiceGenerator** | `instruction`, `text` | Rich free-prose voice description (age, gender, timbre, pace, accent, emotion, e.g. *"A deep, gruff, low-pitched adult male voice … slow and heavy delivery"*). The text should be a neutral 1–2 sentence line, used later as a clone reference. |
| **MOSS-TTS v1.5** | `text`, `reference`, `language` | Always set `language` (name, e.g. `"Thai"`/`"French"`, or a code). Pinyin/IPA pronunciation control is supported. Style comes from the reference clip. |
| **Orpheus** | `"voice: text"` | voice ∈ tara, leah, jess, leo, dan, mia, zac, zoe. Inline tags: `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>`. English only. |
| **CosyVoice3** | `text`, `prompt_text` | `prompt_text` = `You are a helpful assistant.<|endofprompt|>` + transcript of the reference. For instruct mode: `You are a helpful assistant. <instruction><|endofprompt|>`. Inline `[breath]`, `[laughter]` and pinyin `[j][ǐ]`. |
| **Voxtral TTS** | `text`, `voice` | Plain text. `voice` ∈ 20 presets (casual/cheerful/neutral × male/female, plus ar/de/es/fr/hi/it/nl/pt …). No tags. |
| **F5-TTS** | `gen_text`, `ref_text` | Plain text. Style comes from the reference. Keep the reference ≤12 s with an exact transcript. |

## 6. Worker / JSON-lines protocol

All backends fit a long-lived worker that reads one JSON job per line on stdin and writes JSON progress and result lines on stdout. Every snippet already ends with `RESULT {...json...}` in that shape. The pattern is the existing `workers/*.py`: load once, loop over jobs, catch per-job exceptions, `torch.cuda.empty_cache()` between jobs.

- **In-process PyTorch** (audio, qwen and moss venvs): load on the first job for that model, keep it resident, and unload on an `{"op":"unload"}` message or after an idle timeout. Approximate resident size:
  - SA3: 2 / 9 GB
  - OmniVoice: 2 GB
  - Qwen3: 4 GB
  - F5: 2 GB
  - Orpheus: 6 GB
  - MOSS-SFX: 15 GB
  - VoiceGenerator: 11 GB
  - MOSS-TTS: 23 GB
  - MiniMax: 23 GB
- **Run MiniMax-Music3 alone.** It takes minutes per song, so report the AR-stage progress (frames out of `25 × duration`) and support cancellation by killing the worker.
- **vLLM-Omni backends** (Higgs, Fish, Voxtral, CosyVoice):
  - Engine start is 60–165 s, so never start one per job.
  - Either keep an `Omni(model, deploy_config=…)` object in the worker (engine stages are child processes, so shut down with `omni.close()`), or run `vllm-omni serve … --omni --deploy-config <studio yaml>` on a private port (one that no other service on the machine uses) with the worker as an HTTP client of `/v1/audio/speech`.
  - Always use the **studio deploy configs**: the stock ones reserve 40–90 % of the GPU (Voxtral's stock 0.8 + 0.1 would starve other GPU tenants such as ComfyUI or the Hunyuan3D engine).
- **Memory policy.** One heavy model at a time. Other tenants share the 121 GB (llama-server about 22 GB, the Hunyuan3D engine, ComfyUI). MiniMax was OOM-killed twice while about 45 GB was in use elsewhere. Gate loads on free memory (`/proc/meminfo` MemAvailable) and load straight to the GPU (`device_map="cuda"`).
- **Offline.** Set `HF_HUB_OFFLINE=1` in every worker environment, and apply the per-backend offline workarounds (§7).

## QA proxy results

Whisper large-v3-turbo transcripts against the target text (word recall; character recall for Thai):

| Clip | Transcript | Recall |
|---|---|---|
| MOSS-VoiceGenerator, MOSS-TTS clone, OmniVoice clone and design, F5, Qwen3 Base, Higgs plain and clone, Fish, Voxtral, CosyVoice3 | "Hold the line. Reinforcements are on the way." | 1.00 |
| Qwen3 CustomVoice (Ryan + instruct) | exact | 1.00 |
| Orpheus (leo, `<sigh>`) | "…through **the** shield…" (said "the" instead of "this") | 0.92 |
| MOSS-TTS Thai, OmniVoice Thai, omnivoice-thai Thai | "สวัสดีครับ ยินดีต้อนรับสู่สนามรบ เตรียมตัวให้พร้อม" | 1.00 |
| MiniMax-Music3 song | "Morning light filtering through the pine. Every quiet street is yours and mine." (the chorus fell past the 20 s cap) | 0.74 |

CLAP cosine, prompt vs. "silence" vs. "speech" anchors (higher prompt and lower anchors are better; absolute values are only comparable within one prompt):

| Clip | prompt | silence | speech |
|---|---|---|---|
| SA3 medium, orchestral battle music | **0.76** | -0.04 | -0.04 |
| SA3 medium, castle door creak | **0.42** | -0.06 | -0.09 |
| MOSS-SFX, same door prompt | 0.37 | 0.14 | 0.03 |
| MiniMax song vs. "acoustic pop … soft female vocal …" | 0.39 | -0.04 | 0.02 |
| SA3 small-sfx, laser blast | 0.18 | -0.13 | 0.04 |

Level checks: no NaNs anywhere.

- SA3 music and F5 reach full scale (peak 1.0; F5 1.16 before clipping), so add a peak normaliser or limiter at -1 dBFS in post.
- MOSS-TTS Thai has 52 % silence and the MOSS-SFX door 58 %. Trim leading and trailing silence in post, which the audio pipeline already does.

## 7. Failures, gotchas and open items

**Did not run**

- **MOSS-TTS-Nano.** Needs `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (88 MB, not cached). It was not downloaded, per the instructions. After `hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`, use the vLLM-Omni `moss_tts_nano` example or the upstream package.
- **Voxtral voice cloning.** Impossible with the open checkpoint (no audio encoder). Presets only.

**Offline and aarch64 gotchas (all fixed in the snippets)**

| # | Backend | Problem | Fix |
|---|---|---|---|
| 1 | MiniMax-Music3 | diffusers calls `model_info()` for sharded components when offline | `load_components(..., local_files_only=True)` |
| 2 | MiniMax-Music3 | CPU load then `.to(cuda)` doubles memory on unified RAM, so it was OOM-killed | `device_map="cuda"` |
| 3 | Qwen3-TTS | transformers 4.57.3 `_patch_mistral_regex` calls `model_info()` when offline | pass `snapshot_download(repo, local_files_only=True)` |
| 4 | Voxtral | model type not detectable from the repo id when offline (`params.json`) | pass the local snapshot dir |
| 5 | CosyVoice3 | cached `config.json` is `{}` | symlinked model dir with `{"model_type":"cosyvoice3"}` |
| 6 | CosyVoice3 | `s3tokenizer` missing; it then downloads a 925 MB ONNX from modelscope | overlay install; symlink the identical cached `speech_tokenizer_v3.onnx` into `~/.cache/s3tokenizer/speech_tokenizer_v3_25hz.onnx` (**this symlink now exists on the machine**) |
| 7 | CosyVoice3 | example labels output 22 050 Hz, the model is 24 kHz | write at 24 kHz |
| 8 | Fish S2 Pro | `fish_speech` DAC module missing from the vllm-omni venv | `envs/vllm_omni_overlay` on `PYTHONPATH` |
| 9 | F5-TTS | torchaudio 2.9 `load()` needs torchcodec (no aarch64 wheel) | monkey-patch `torchaudio.load` to use soundfile, or use `<F5-TTS checkout>/.venv` with its self-built torchcodec |
| 10 | MOSS-TTS family | transformers 5.22 rejects the `revision` kwarg in `ProcessorMixin` | keep transformers 5.0.0 (moss venv) |
| 11 | stable-audio-tools | PyPI 0.0.20 is too old; git main declares python <3.11 and torch 2.7.1 | `--no-deps` git install plus manual deps and `pytorch_lightning` |
| 12 | vLLM-Omni configs | stock deploy YAMLs reserve 40–90 % of GPU memory | `audio_snippets/vllm_omni_deploy/*_studio.yaml` |
| 13 | Snippet naming | a script named `omnivoice.py` or `f5_tts.py` shadows the package | renamed to `run_*.py` |
| 14 | General | torch warns that sm_121 is above its max of sm_120 | harmless, everything ran |

**Not tested or open**

- Higgs and Fish in Thai.
- Warm-request latency for vLLM backends (only cold first requests were measured).
- Qwen3-TTS, OmniVoice and MOSS-TTS through vLLM-Omni (possible streaming gains).
- SA3 inpainting.
- MiniMax-Music3 lyric languages beyond English.
- Orpheus via vLLM.
- The newer `stable-audio-3` package.
- The `hotdogs/omnivoice-thai` snapshot carries about 4 GB of optimizer state that is not needed for inference.
- ACE-Step lives in a separate ComfyUI container and was not touched.

**Side effects on the machine**

- New: `envs/{audio,qwen,vllm_omni_overlay}`.
- One symlink in `~/.cache/s3tokenizer`. A partial 917 MB modelscope download landed there before I killed it; I deleted it.
- Nothing was installed into the existing moss or vllm-omni venvs.
- No containers or ports were touched.

**Snippets** (`audio_snippets/`):

- `_bench.py`: timing, memory and audio stats.
- `stable_audio_3.py`, `minimax_music3.py`, `moss_soundeffect.py`, `moss_tts.py`, `qwen3_tts.py`, `run_omnivoice.py`, `run_f5_tts.py`, `orpheus.py`.
- `vllm_omni_tts.sh` with `vllm_omni_deploy/`.
- `qa_score.py`.
- `lock_audio.txt`, `lock_qwen.txt`.
