# OpenMOSS models beyond the first four

Research and integration notes, 2026-11-06. OpenMOSS (Fudan / MOSI; HF orgs `OpenMOSS-Team`, `fnlp`, `OpenMOSS`) ships a large family of open-weight speech
models. Four were already in the studio (`moss-tts-v1.5`, `moss-voicegen`, `moss-tts-nano`, `moss-sfx-v2`). This note covers the rest that are useful for
media production. **None of the added models is gated** (all Apache-2.0, plain HF downloads, no token needed).

- Registry: `models.d/moss_more.yaml`. Workers: `workers/moss_more_worker.py`, `workers/moss_asr_worker.py`, `workers/moss_audio_worker.py`.
- Machine: DGX Spark (aarch64, GB10, 121 GB unified memory, CUDA 13, no nvcc). Tests ran offline, one model at a time under the shared GPU lock.
- Tests: `tests/test_moss_more.py` (registry, slots, worker helpers). The real generations below were run with the worker protocol by hand.

## What was added

| Model id | Task | What it is | Backend / env |
|---|---|---|---|
| `moss-ttsd-v1` | `tts`, `voice_clone` | MOSS-TTSD v1.0: podcast / dialogue generator, 1-5 speakers, 20 languages (no Thai), up to ~1 h per script | `moss_more` / `envs/moss` (shared, read-only) |
| `moss-tts-realtime` | `tts`, `voice_clone` | MOSS-TTS-Realtime 1.7 B: low-latency streaming-style TTS with voice prompt | `moss_more` / `envs/moss` |
| `moss-tts-local-v1.5` | `tts`, `voice_clone` | MOSS-TTS-Local-Transformer v1.5: 48 kHz stereo codec (MOSS-Audio-Tokenizer-v2), 31 languages including **Thai**, duration control | `moss_more` / `envs/moss` |
| `moss-transcribe-diarize` | `transcribe` (new) | Transcript with speaker labels and timestamps; 50+ languages; audio or video up to ~90 min. Outputs `.txt`, `.srt`, `.json` | `moss_asr` / `envs/moss_asr` |
| `moss-audio-4b-instruct` | `audio_ask` (new) | MOSS-Audio 4B Instruct: ask anything about a clip (caption, speakers, emotion, instruments, events, timestamps) | `moss_audio` / `envs/moss_audio` |

### How the UI sees them

- **Dialogue needs no UI change.** It follows the Dia / VibeVoice convention: tasks `tts` / `voice_clone`, a multi-line `text` script with `[S1] ... [S2] ...` tags,
  and input slots `ref`, `ref2`, `ref3`, `ref4` ("Speaker 1..4 voice"). Each reference needs its transcript (the voice-library `ref_text` is used, else the
  params `ref_text`, `ref_text2`, ...). With no references the model invents the voices (generation mode); with references it continues in those voices
  (continuation mode, the prompt audio is not part of the output).
- **New tasks `transcribe` and `audio_ask` have no Sound-studio tab.** They are registered with `studio: sound` and produce `text` assets; they work through the API,
  the pipeline runner and the assistant. A tab in `app/web/studios/sound.js` was deliberately not added (another agent edits that file).

## Install

Both new venvs follow `install/<backend>.yaml` (see `install/FORMAT.md`); `./install.sh moss_asr moss_audio` does it all. By hand, from the repo root:

```bash
# moss_more: nothing to install, it reuses envs/moss (needs scripts/fetch_vendor.sh MOSS-TTS)

# moss_asr (MOSS-Transcribe-Diarize; transformers >= 5.6)
scripts/fetch_vendor.sh MOSS-Transcribe-Diarize
uv venv --python 3.12 envs/moss_asr
VIRTUAL_ENV=$PWD/envs/moss_asr uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1 torchaudio==2.9.1
VIRTUAL_ENV=$PWD/envs/moss_asr uv pip install -r env-lock/moss_asr.txt
VIRTUAL_ENV=$PWD/envs/moss_asr uv pip install --no-deps -e ./vendor/MOSS-Transcribe-Diarize

# moss_audio (MOSS-Audio-4B; transformers == 4.57.1; vendor/MOSS-Audio is put on sys.path by the worker)
scripts/fetch_vendor.sh MOSS-Audio
uv venv --python 3.12 envs/moss_audio
VIRTUAL_ENV=$PWD/envs/moss_audio uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1 torchaudio==2.9.1
VIRTUAL_ENV=$PWD/envs/moss_audio uv pip install -r env-lock/moss_audio.txt

# weights (one repo at a time)
./install.sh model moss-ttsd-v1        # also pulls OpenMOSS-Team/MOSS-Audio-Tokenizer (shared with moss-tts-v1.5)
./install.sh model moss-tts-realtime
./install.sh model moss-tts-local-v1.5 # also pulls MOSS-Audio-Tokenizer-v2
./install.sh model moss-transcribe-diarize
./install.sh model moss-audio-4b-instruct
```

Vendor pins are in `vendor/vendor.lock` (`MOSS-Audio`, `MOSS-Transcribe-Diarize`); licences in `THIRD_PARTY_NOTICES.md`.

### Why three environments

| Env | transformers | Reason |
|---|---|---|
| `envs/moss` (existing, symlink) | 5.0.0 exactly | MOSS-TTS family remote code. Reused unchanged for TTSD / Realtime / Local v1.5. |
| `envs/moss_asr` | 5.22.1 | MOSS-Transcribe-Diarize needs >= 5.6. |
| `envs/moss_audio` | 4.57.1 exactly | MOSS-Audio's Whisper-style encoder calls `WhisperEncoderLayer` with the 4.x signature; on 5.22.1 it fails (`The size of tensor a (50) must match the size of tensor b (20)`). Its README pins 4.57.1. |

## Results

Each model was run once through its worker with the real weights (cold process, so `load` includes warm-up). "Peak mem" is the worker's
`torch.cuda` allocation after load (`gpu_gb`); the system-wide extra memory seen in the unified pool during the run is in brackets. Disk is the HF cache size of the
weights (the codec repos are shared where noted). Env sizes: `envs/moss_asr` 4.9 GB; `envs/moss_audio` is small on disk because uv hard-links torch from its cache.

| Model | Licence | Commercial | Disk | Load | Run | Peak mem (GPU) | Status | Check |
|---|---|---|---|---|---|---|---|---|
| moss-ttsd-v1 | Apache-2.0 | yes | 16 GB (+6.7 GB codec, shared) | 120 s | 20 s dialogue in 26 s; cloned 14 s in 20 s | 22.3 GB (+27 GB system) | ready | round trip through MOSS-Transcribe-Diarize: text exact, speakers S01/S02 correct, both modes |
| moss-tts-realtime | Apache-2.0 | yes | 4.4 GB | 60 s | ~1.25x real time | 11.1 GB | ready | round trip through ASR |
| moss-tts-local-v1.5 | Apache-2.0 | yes | 8.5 GB (+8.0 GB codec v2) | 48 s | 3.9 s Thai in 4.2 s | 11.8 GB (+24 GB system) | ready | Thai round trip; output 48 kHz stereo |
| moss-transcribe-diarize | Apache-2.0 | yes | 1.8 GB | 11 s | 20 s clip in 2.9 s | 1.7 GB (+3 GB system) | ready | 5-voice 52 s clip: speakers split, timestamps, srt + json |
| moss-audio-4b-instruct | Apache-2.0 | yes | 9.8 GB | 60 s | 52 s clip, 7 s | 9.7 GB (+16 GB system) | ready | described the 52 s clip correctly (speaker, quiet room, calm tone) |

Total new downloads: 48 GB (the six repos incl. codec v2; codec v1 was already cached for moss-tts-v1.5), within the 70 GB budget, plus about 5 GB for `envs/moss_asr`. Stale `.incomplete` blobs from interrupted downloads were removed.

### Notes per model

- **MOSS-TTSD v1.0.** 8 B Qwen3-based model (`Qwen3ForCausalLM`) on the 12.5 token/s MOSS-Audio-Tokenizer (24 kHz). The worker validates that the script uses `[S1]`..`[S5]` tags,
  merges tags the way the model card normalises them, and refuses a reference without a transcript (the model needs it). A 2-speaker script with no references
  gave natural turn taking; cloning from two 5-6 s excerpts produced distinct voices and correct text. A long script is generated in one pass (`max_seconds`
  caps the token budget at 12.5 tokens per second), so very long podcasts need a lot of time; the model card claims up to ~1 h.
- **MOSS-TTS-Realtime.** Uses the vendored `moss_tts_realtime` inferencer with `_load_audio` patched to `soundfile` (torchaudio has no torchcodec here). Not truly streamed in the studio: it returns a finished file.
- **MOSS-TTS-Local v1.5.** Auto-detects Thai; the duration control is the `tokens` parameter (exposed as `seconds`). 48 kHz stereo output is saved 24-bit.
- **MOSS-Transcribe-Diarize.** Output lines look like `[00:00:06.720 - 00:00:12.060] S02: ...`. See the model's `params` in `models.d/moss_more.yaml` (output style, prompt language, hotwords). Audio is decoded with soundfile (ffmpeg for video and odd formats) at 16 kHz.
- **MOSS-Audio-4B-Instruct.** Greedy decoding unless `temperature > 0`; the reply is saved as a text asset and also returned as `reply_text` meta.

## What failed or was skipped

| Item | Result | Reason |
|---|---|---|
| MOSS-Audio on transformers 5.22.1 | failed | Encoder layer signature mismatch; fixed by its own venv on transformers 4.57.1. |
| MOSS-TTS family on a newer transformers | not tried | `envs/moss` is pinned to 5.0.0 for the remote code and is read-only. |
| MOVA-360p / MOVA-720p (video + audio generation) | skipped | 77.7 GB of weights each, over the 70 GB budget; video belongs to the video group. |
| MOSS-Audio-8B-Instruct, MOSS-Audio-4B/8B-Thinking | not added | Same code and env as `moss-audio-4b-instruct` (18.1 GB, 10.5 GB, 18.1 GB); adding one is a yaml row plus a download. Left out to stay in budget; the 4B Instruct already answers the `audio_ask` use cases. |
| MOSS-Music-8B-Instruct / Thinking | not added | Music-understanding variant (audio to text, 18.1 GB). Overlaps `audio_ask`; not tried, so no claim about its quality or about which env it needs. |
| MOSS-Image Base / SFT | skipped | 27.2 GB image generators, not audio; outside this group. |
| MOSS-VL (Realtime, Instruct) | skipped | Vision-language models, not audio / speech. |
| MOSS-Speech, MOSS-Speech-Codec, SpeechGPT-2.0-preview-7B | not evaluated | Speech-to-speech research models. Not tried (time and budget went to the models above), so the `s2s` task still has no OpenMOSS entry. |
| MOSS-TTSD v0.x and the XY-Tokenizer | skipped | Superseded by v1.0, which uses the standard MOSS codec. |
| Text-only MOSS LLMs | skipped | Per brief. |

## Gotchas worth remembering

- The shared checkouts (`vendor/MOSS-TTS`) are read-only: workers set `sys.dont_write_bytecode` and the backend env sets `PYTHONDONTWRITEBYTECODE=1` so no `__pycache__` appears in them.
- `AutoTokenizer.from_pretrained` for remote-code tokenizers (Realtime, MOSS-Audio) asks for confirmation on stdin unless `trust_remote_code=True` is passed explicitly.
- cuDNN SDPA is disabled and SDPA attention is used (no flash-attn on aarch64).
- Registry "ready" requires every weight shard: the TTSD entry lists the safetensors globs so a half-finished download shows as `missing_weights`.
