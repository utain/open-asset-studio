# Music group: open-weight music / song models (studio, 2026-11)

Registry: `models.d/music.yaml` (13 entries, 8 backends). Workers: `workers/music_*_worker.py`. Venvs: `envs/music_*`
(pinned freeze in `env-lock/music_*.txt`; `music_riffusion` reuses `envs/music_audiocraft`). Tests: `tests/test_music.py`
(registry level, no GPU). Hardware: DGX Spark (GB10, aarch64, 121 GB unified memory, torch cu130, no nvcc).

Tasks used: `music` (instrumental), `song` (vocals + lyrics), `sfx` (AudioGen). No new task ids were needed.
UI inputs needed beyond the existing sound studio:

* `musicgen-melody` reads an audio input with `role: melody` (`features.melody: true`). The sound studio has no slot for it yet.
* `extend: true` models (MusicGen x3, AudioGen) use the existing "clip to extend" slot (`role: source`, `extend_seconds`).
* DiffRhythm and YuE accept an optional reference clip (`role: ref`, `features.ref_audio: true`); not wired in the UI.
* Song models use the existing `lyrics` param type.

All timings below were measured on the shared GPU (other agents were using it, so run times are upper bounds), one real
generation per model, output written to wav.

## Summary

| model id | status | HF repo | licence | peak GPU | load / run | sanity check |
|---|---|---|---|---|---|---|
| musicgen-medium | working | facebook/musicgen-medium | CC-BY-NC-4.0 (non-commercial) | 4.1 GB | 18 s / 8 s per 6 s clip | CLAP(prompt) 0.49 vs silence -0.03; text + continuation both run |
| musicgen-melody | working | facebook/musicgen-melody | CC-BY-NC-4.0 | 4.2 GB | 18 s / 9 s per 8 s clip | CLAP 0.34; pitch-class correlation with the sine melody 0.33 (weak but positive) |
| musicgen-stereo-medium | working | facebook/musicgen-stereo-medium | CC-BY-NC-4.0 | 4.5 GB | 9 s / 11 s per 8 s clip | 2 channels, CLAP 0.24 (ambient prompt), no NaN |
| audiogen-medium | working | facebook/audiogen-medium | CC-BY-NC-4.0 | 4.9 GB | 17 s / 13 s for 2x5 s | CLAP 0.40, 16 kHz mono (lo-fi) |
| magnet-medium-10secs | working | facebook/magnet-medium-10secs | CC-BY-NC-4.0 | 4.1 GB | 19 s / 12 s per 10 s clip | CLAP 0.04 and 0.19 (two takes): quality clearly below MusicGen |
| riffusion-v1 | working | riffusion/riffusion-model-v1 | CreativeML OpenRAIL-M (commercial with use restrictions) | 2 GB | 3 s / 2.8 s per 5.1 s clip | CLAP 0.39 / 0.32; spectrogram PNG attached to each take |
| acestep-1.5-turbo | working | ACE-Step/Ace-Step1.5 | MIT | 9.4 GB resident | 46-49 s / 11 s for 30 s, 14 s for 40 s (planner on); 2.4 s for 20 s (planner off) | Whisper word recall 0.88 (40 s, 8 lyric lines); instrumental (music task, 20 s, planner off): CLAP 0.40, 20 % quiet frames |
| diffrhythm-1.2-full | working | ASLP-lab/DiffRhythm-1_2-full | Apache-2.0 code, but MuQ-MuLan style encoder CC-BY-NC => non-commercial | 5.7 GB | 16 s / 11 s for 96 s | Whisper recall 0.96 with dense lyrics (0.0 with 4 sparse lines) |
| diffrhythm-1.2 | working | ASLP-lab/DiffRhythm-1_2 | same | 5.7 GB | 25 s / 22 s for 95 s | instrumental (music task) CLAP 0.64 |
| yue-s1-7b | working (slow) | m-a-p/YuE-s1-7B-anneal-en-cot + YuE-s2-1B-general + m-a-p/xcodec_mini_infer | Apache-2.0 | 17.7 GB | 93 s / 468 s for a 12 s segment | Whisper: first lyric line transcribed exactly ("morning light on the village road") |
| heartmula-3b (extra, Jan 2026) | working | HeartMuLa/HeartMuLa-oss-3B-happy-new-year + HeartCodec-oss-20260123 + HeartMuLaGen | Apache-2.0 | 15.8 GB resident | 11 s / 36 s for 30 s | Whisper recall 0.96 |
| magenta-rt2-small | working | google/magenta-realtime-2 | CC-BY-4.0 weights, Apache-2.0 code | not measured (JAX; <6 GB estimate) | 16 s / 4.5 s per 10 s clip | CLAP 0.55; two seeds give uncorrelated audio (corr 0.01) |
| songgeneration-levo | blocked | lglg666/SongGeneration-v2-large (third party) | unknown | n/a | n/a | upstream code gone |

## Notes per model

### MusicGen medium / melody / stereo-medium, AudioGen medium, MAGNeT medium (backend `music_audiocraft`, env `envs/music_audiocraft`)
* Code: `vendor/audiocraft` (Meta, MIT) installed `--no-deps -e`; weights CC-BY-NC-4.0 so `commercial: false`.
  We use the audiocraft-format `state_dict.bin` + `compression_state_dict.bin` files of the facebook repos (not the
  transformers port), loaded fully offline from the HF cache.
* Env: torch 2.9.1+cu130 plus a hand-picked dependency set (audiocraft's `requirements.txt` pins torch 2.1 and xformers,
  neither installs on aarch64). Three shims live in the worker: (1) `xformers` stub - its `memory_efficient_attention` runs
  on torch SDPA; MusicGen / AudioGen use audiocraft's own torch backend, MAGNeT checkpoints force the xformers code path
  (additive local-attention masks), which the stub serves; (2) `torch.load(weights_only=False)` default (OmegaConf configs
  inside the checkpoints); (3) legacy HF ids `t5-base` / `t5-large` mapped to `google-t5/...` for offline mode.
* Gotchas: AudioGen needs `google-t5/t5-large` (not t5-base); the stereo model needs `facebook/encodec_32khz` (its
  compression checkpoint just points at it); the melody model needs `demucs` (`uv pip install --no-deps demucs julius
  lameenc openunmix`) and downloads the `htdemucs` weights from dl.fbaipublicfiles.com on first use (torch hub, not HF);
  MAGNeT is always 10 s; continuation (`source` input) is supported for MusicGen/AudioGen only.
* Melody conditioning was checked with a synthetic sine scale: output pitch-class profile correlates 0.33 with the input
  (weak: a hummed or played tune with a real timbre is the intended input).

### Riffusion v1 (backend `music_riffusion`, same venv)
* SD 1.5 on spectrogram images (512x512 = 5.12 s). Spectrogram-to-audio is a re-implementation of the original
  `SpectrogramConverter` (44.1 kHz, 10 ms step, 100 ms window, 512 mels 0-10 kHz, power 0.25, Griffin-Lim 32 iterations).
* torchaudio 2.9: `InverseMelScale` has no SGD iterations any more (least squares) and its filter bank is not a buffer, so it
  runs on the CPU while Griffin-Lim runs on the GPU.
* Only the diffusers pieces (unet / vae / text_encoder / tokenizer / scheduler, `.bin` weights) are downloaded, not the
  14 GB single-file checkpoints.

### ACE-Step 1.5 turbo (backend `music_acestep`, env `envs/music_acestep`)
* v1.5 (not v1-3.5B) exists and is MIT. DiT turbo (8 steps) + optional 1.7B "5 Hz" language-model planner (pt backend,
  writes BPM / key / caption / structure metadata). Output 48 kHz stereo. `music` task = `[Instrumental]` lyrics.
* Needs torch 2.10.0+cu130 (the project pins it for aarch64), hence its own venv. The checkpoint dir is a symlink farm of the HF
  snapshot at `vendor/ace_project/checkpoints` because the code expects `checkpoints/<name>` next to the project root.
* There is no mid-run cancel hook in `generate_music`; cancelling takes effect between takes.
* Gotcha: with 4 lyric lines over 30 s Whisper heard only "oh oh oh" (recall 0.0); with 8 lines over 40 s recall was 0.88.

### DiffRhythm 1.2 / 1.2-full (backend `music_diffrhythm`)
* Latent diffusion, 44.1 kHz stereo, 95 s (base) or 96-285 s (full). Style prompt = text through MuQ-MuLan (needs
  `OpenMuQ/MuQ-MuLan-large`, `OpenMuQ/MuQ-large-msd-iter`, `FacebookAI/xlm-roberta-base`) or a reference clip.
* `transformers` MUST be 4.49.0 (5.x breaks the Llama rotary embedding call: `size of tensor a (32) must match b (64)`);
  `py3langid==0.2.2` (newer drops `from_pickled_model`); espeak-ng from the system (`libespeak-ng.so.1`), the worker sets
  `PHONEMIZER_ESPEAK_LIBRARY` / `ESPEAK_DATA_PATH` and chdirs into `vendor/DiffRhythm` (relative config paths).
* Lyrics: LRC with timestamps, or plain lines (the worker spreads them over the song and drops `[section]` lines). One line per
  6-8 s works (recall 0.96); a few lines in a long song make the vocals mumble.
* Licence: the pipeline is non-commercial because MuQ-MuLan is CC-BY-NC.

### YuE v1 (backend `music_yue`, env `envs/music_yue`)
* `vendor/YuE-v1` (branch YuE-v1) ported in-process: stage 1 7B bf16 SDPA, stage 2 1B, xcodec decoder + Vocos upsamplers,
  low-frequency band replaced from the 16 kHz xcodec mix as upstream. No `torch.compile` (no nvcc). Progress / cancel
  through a `StoppingCriteria` and a per-frame stage-2 loop.
* Cost: ~7 tokens/s in stage 1 and ~0.4 s per stage-2 frame: a 12 s segment takes ~8 min; a 30 s segment ~15 min.
  `kill_on_cancel: true`. `seconds` (per lyric section, 12-30) and `segments` keep test runs short.
* `inference/xcodec_mini_infer` in the repo is an (empty) submodule; the worker replaces it with a symlink to the HF snapshot
  and chdirs into `inference/` because the xcodec configs use `./xcodec_mini_infer/...` paths.
* YuE2 (newer, `vendor/YuE` main) was cloned for reference only; it uses MuQ again and is not included.

### HeartMuLa 3B (extra model, backend `music_heartmula`)
* Released Jan-Feb 2026 (Apache-2.0): 3B autoregressive LM over 12.5 Hz HeartCodec tokens, tags + lyrics, multilingual,
  48 kHz stereo. Not in the brief; chosen as the "one extra" because it is the best open song model that fits the task.
* `heartlib` wants one directory with `tokenizer.json`, `gen_config.json`, `HeartMuLa-oss-3B/`, `HeartCodec-oss/`; the worker
  builds it as a symlink farm in `vendor/heartmula_ckpt` from three HF repos. `torchao==0.9.0` pin (0.18 breaks heartlib's
  import of `ScalingType`). `_apply_compile` is disabled; the frame loop is re-implemented only to report progress and
  honour cancel. Run: 30 s of audio in 36 s.

### Magenta RealTime 2 small (backend `music_magenta`, env `envs/music_magenta`)
* JAX 0.12 + `jax[cuda13]` works on the GB10 (XLA compiles the streaming step at load, ~16 s). `mrt2_base` is registered
  in the worker but not in the registry (weights not downloaded).
* `vendor/magenta-realtime/magenta_rt/_vendor/sequence-layers` is a git submodule that `git clone` does not fetch: it was
  cloned manually from github.com/magenta/sequence-layers.
* The library fixes the sampler rng at compile time (PRNGKey(0)); the worker builds its own jitted initial state with the
  take seed, so different seeds give different audio. `XLA_PYTHON_CLIENT_PREALLOCATE=false` (unified memory).
* Output is a stream of style-driven loops, not a structured song; text is embedded with MusicCoCa (`use_mapper=True`).

### SongGeneration / LeVo (backend `music_levo`, blocked)
* github.com/tencent-ailab/SongGeneration returns 404, `tencent/SongGeneration` is not available on the Hub (401) and the
  licence cannot be verified. Third-party uploads of the weights exist (`lglg666/SongGeneration-v2-large`, 12.9 GB) but there is
  no official code to run them and no HF login/mirror hunting is allowed by the brief. The registry entry carries
  `features.blocked`; `workers/music_levo_worker.py` is a stub that reports the reason. Nothing was downloaded.

## Reproducible setup

All commands from `envs` with `export HF_HUB_DISABLE_XET=1` for downloads; `UV_HTTP_TIMEOUT=900` helps on a busy network.

```bash
# music_audiocraft (also used by music_riffusion)
uv venv --python 3.12 music_audiocraft && export VIRTUAL_ENV=$PWD/music_audiocraft
uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
uv pip install --no-deps -e ../vendor/audiocraft            # git clone https://github.com/facebookresearch/audiocraft first
uv pip install -r ../env-lock/music_audiocraft.txt --no-deps # exact versions used (includes diffusers, demucs, transformers, ...)
uv pip install --no-deps demucs julius lameenc openunmix     # melody conditioning

# music_acestep
uv venv --python 3.12 music_acestep && export VIRTUAL_ENV=$PWD/music_acestep
uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
uv pip install --no-deps -e ../vendor/ACE-Step-1.5 -e ../vendor/ACE-Step-1.5/acestep/third_parts/nano-vllm   # then the deps in env-lock/music_acestep.txt

# music_diffrhythm: git clone https://github.com/ASLP-lab/DiffRhythm vendor/DiffRhythm; requirements.txt with
#   transformers==4.49.0 tokenizers<0.22 huggingface_hub<1 py3langid==0.2.2 wandb tensorboard; apt: espeak-ng
# music_yue: git clone -b YuE-v1 https://github.com/multimodal-art-projection/YuE vendor/YuE-v1 (torch 2.9.1 cu130, transformers 4.51.x)
# music_heartmula: git clone https://github.com/HeartMuLa/heartlib vendor/heartlib; uv pip install --no-deps -e ../vendor/heartlib; torchao==0.9.0
# music_magenta: git clone https://github.com/magenta/magenta-realtime vendor/magenta-realtime; git clone https://github.com/magenta/sequence-layers
#   vendor/magenta-realtime/magenta_rt/_vendor/sequence-layers; uv pip install -e "../vendor/magenta-realtime[jax]" "jax[cuda13]"
```

Weights (default HF cache, ~100 GB total; the repos and file lists are in each registry entry):
`musicgen-medium`/`-melody`/`-stereo-medium`/`audiogen-medium`/`magnet-medium-10secs` (state_dict.bin + compression_state_dict.bin
only), `google-t5/t5-base`, `google-t5/t5-large` (AudioGen), `facebook/encodec_32khz` (stereo), `riffusion/riffusion-model-v1`
(diffusers subfolders), `ACE-Step/Ace-Step1.5`, `ASLP-lab/DiffRhythm-1_2`, `-1_2-full`, `-vae`, `OpenMuQ/*`, `FacebookAI/xlm-roberta-base`,
`m-a-p/YuE-s1-7B-anneal-en-cot`, `m-a-p/YuE-s2-1B-general`, `m-a-p/xcodec_mini_infer`, `HeartMuLa/*`, `google/magenta-realtime-2`
(resources + `checkpoints/mrt2_small.safetensors`).

Downloads: `hf download` ran at 0.1-1 MB/s per stream here; a parallel byte-range downloader writing the standard cache layout
(blobs/sha256 + snapshot symlinks) was used instead and the result is a normal HF cache.
