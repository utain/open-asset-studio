# sfx_more: AudioLDM 1/2, Tango 2, TangoFlux, Make-An-Audio 2, Stable Audio Open

Registry: `models.d/sfx_more.yaml` (backend `sfxmore`). Worker: `workers/sfxmore_worker.py`. Env: `envs/sfxmore`, lock `env-lock/sfxmore.txt`.
Registry test: `tests/test_sfx_more.py`. Vendored upstream code: `vendor/TangoFlux`, `vendor/tango`, `vendor/Make-An-Audio-2`, `vendor/BigVGAN-awei6`.

## Status

| model id | HF repo | status | license | peak GPU | load / run (warm, 100 steps unless noted) | output |
|---|---|---|---|---|---|---|
| `audioldm-l-full` | cvssp/audioldm-l-full | working | CC-BY-NC-SA-4.0 (non-commercial) | 2.0 GB | 3-5 s / 3.2 s (5 s clip), 3.7 s (10 s) | 16 kHz mono |
| `audioldm2` | cvssp/audioldm2 | working | CC-BY-NC-SA-4.0 | 2.5 GB | 5-6 s / 2.2-3.0 s | 16 kHz mono |
| `audioldm2-large` | cvssp/audioldm2-large | working | CC-BY-NC-SA-4.0 | 3.2 GB | 15 s / 8-12 s | 16 kHz mono |
| `audioldm2-music` | cvssp/audioldm2-music | working | CC-BY-NC-SA-4.0 | 2.5 GB | 8 s / 2.8-3.7 s | 16 kHz mono |
| `tango2` | declare-lab/tango2 | working | CC-BY-NC-SA-4.0 | 6.2 GB (fp32) | 15 s / 16.6 s per take | 16 kHz mono, fixed 10 s window |
| `tangoflux` | declare-lab/TangoFlux | working | Stability Community + WavCaps (academic only): treat as non-commercial | 7.0 GB (fp32) | 5-6 s / 3.4-4.0 s (25 steps) | 44.1 kHz stereo, <= 30 s |
| `make-an-audio-2` | ByteDance/Make-An-Audio-2 | working | MIT code and ckpt (`verify: true`: data terms, third-party vocoder copy) | 4.2 GB | 22 s / 1.7-3.0 s | 16 kHz mono, <= 10 s |
| `stable-audio-open-1.0` | stabilityai/stable-audio-open-1.0 | **blocked**: gated (HTTP 401, no HF login) | Stability Community | - | - | registry entry only |
| `stable-audio-open-small` | stabilityai/stable-audio-open-small | **blocked**: gated | Stability Community | - | - | registry entry only |

Checks on every working model: no NaN, expected duration, not silent. Whisper does not apply (no speech).

## CLAP scores (laion/clap-htsat-unfused, the `qa_score.py` method)

Prompts: door = "heavy wooden castle door creaking open slowly" (5 s), laser = "sci-fi laser blast" (3 s), forest = "forest ambience with birds singing and a gentle wind in the trees" (10 s), music = "orchestral battle music, epic drums and brass, fast tempo" (10 s). Seed 7, 100 steps (TangoFlux 25), one take each, so single-sample noise is large.
Reference from `audio_models.md`: SA3 medium door 0.42, music 0.76; SA3 small-sfx laser 0.18; MOSS-SFX door 0.37.

| model | door | laser | forest | music |
|---|---|---|---|---|
| audioldm-l-full | 0.63 | 0.45 | 0.15 | 0.40 |
| audioldm2 | 0.09 | 0.46 | 0.24 | 0.61 |
| audioldm2-large | 0.21 | 0.49 | 0.24 | 0.62 |
| audioldm2-music | - | - | 0.24 | 0.54 |
| tango2 | 0.33 | 0.14 | 0.46 | 0.65 |
| tangoflux | 0.41 | 0.23 | 0.40 | 0.54 |
| make-an-audio-2 | 0.55 | 0.62 | 0.22 (0.44 before level-up) | 0.37 |

Read these with care:
- **AudioLDM 1 and 2 are conditioned on CLAP embeddings**, so their CLAP-prompt score is biased upward. Do not rank them against SA3 with this metric. The door scores (0.63 vs 0.09 for the same family) show how noisy one sample is.
- The silence/speech anchors stayed low (< 0.2) except for the very quiet forest clips (silence anchor 0.44 on the raw AudioLDM 1 and MAA2 forest takes, before the level-up).
- By these proxies (I did not listen to the clips): TangoFlux and Tango 2 are the most consistent across the four prompts; MAA2 is best on one-shot events; AudioLDM 1 is weak on ambience; the AudioLDM 2 music model sounds lo-fi (16 kHz).
- None of them beats SA3 medium on music (0.76), and all but TangoFlux are 16 kHz mono. In practice they are second-opinion generators, with more variety than a quality upgrade.

## Gotchas (all handled in the worker)

1. **Quiet output.** AudioLDM 1/2 return takes as low as -60 dBFS (10 s forest peak 0.001). The AudioLDM reference code normalises every take to 0.5 peak, so the worker does too (the original level is stored as `normalized_from_db`). MAA2 ambience comes out at about -48 dBFS; the worker brings takes below 0.1 peak up to 0.3.
2. **TangoFlux silent clips.** Some seed/duration pairs give a fully silent clip (seed 7, 3 s and 5 s "sci-fi laser blast", peak 3e-5). The worker detects a peak below 0.003 and retries with seed + 7919 (up to 4 tries), reports the seed actually used, and logs the retry.
3. **AudioLDM 2 needs a GPT-2 patch.** With transformers 4.57 `GPT2Model` is no longer a `GenerationMixin`, and the diffusers pipeline calls `language_model._update_model_kwargs_for_generation`. The worker makes the language model's class inherit `GenerationMixin` after loading. (transformers 4.57.3 is pinned for this reason; do not go to 5.x without testing.)
4. **AudioLDM 1 pipeline is deprecated** in diffusers 0.39 (`pipelines/deprecated/audioldm`); still works, prints a deprecation notice. Weights are `.bin` only (pickle, from CVSSP), loaded with `use_safetensors=False`.
5. **Tango 2 vendor code vs modern libs.** `vendor/tango` imports `diffusers.utils.randn_tensor` (moved), and calls `librosa.util.pad_center` / `librosa.filters.mel` positionally (keyword-only since librosa 0.10). The worker shims all three before importing. Needs `progressbar2`, `torchlibrosa`, `h5py`, `omegaconf`. Scheduler config comes from `sd2-community/stable-diffusion-2-1` (`scheduler/scheduler_config.json` only; the original `stabilityai/stable-diffusion-2-1` repo was removed). Tango 2 always generates a 10.24 s window (about 16 s of compute, fp32), so short clips cost the same; the worker trims to `seconds`. fp16 was not tried.
6. **TangoFlux.** Needs `google/flan-t5-large` (`model.safetensors`, 3.1 GB) next to its own weights. `strict=False` load reports the tied `text_encoder.encoder.embed_tokens.weight` as missing: expected. It always denoises a 30 s latent (so time per take is flat) and trims to `duration`. `datasets` and `pyyaml` are imported at module level by the repo. fp32 peak is 7 GB.
7. **Make-An-Audio 2.**
   - The README names a BigVGAN vocoder on Google Drive and a CLAP checkpoint. I used a third-party HF copy of the BigVGAN files (`SC4949/make-an-audio-models`, `bigvgan/best_netG.pt` + `args.yml`, 449 MB; `torch.load(weights_only=True)` accepts it, so it is plain tensors, and MAA2 produced sensible audio with it). Provenance is not official; hence `license.verify: true`.
   - `maa2.ckpt` already contains the CLAP text encoder and T5 weights (`cond_stage_model.*`, 424 tensors), so the 2.3 GB CLAP file is not needed: the worker gives the constructor an empty stub (`data/cache/maa2_clap_stub.pth`). The T5 (`google/t5-v1_1-large`) and `bert-base-uncased` downloads are only used to build the module shapes before the ckpt overwrites them.
   - `maa2.ckpt` is a Lightning checkpoint with pickled callbacks and optimizer state, so it needs `torch.load(weights_only=False)`: trusted ByteDance repo only.
   - Old APIs are shimmed in the worker: `torch._six.string_classes`, `pytorch_lightning.utilities.distributed.rank_zero_only`. The BigVGAN package is `vendor/BigVGAN-awei6` (github awei-6/BigVGAN, put on `sys.path`, not pip installed).
   - The upstream flow asks ChatGPT to turn the prompt into a structured caption (`<event& start>@<event& end>`). The worker has an optional `struct_prompt` param; empty = `<prompt& all>`. Multi-event captions were not tested.
8. **Stable Audio Open 1.0 and Small.** Both repos answer 401 without a login (`gated: auto`). Not downloaded, no mirror sought. Registry entries exist with `features.blocked` and a `fix` text; the worker raises a clear error if one is run. Small needs stable-audio-tools format loading, 1.0 could use diffusers `StableAudioPipeline`: both left unwired because they are lower priority than SA3, which the studio has.
9. **Downloads on this box.** The shared link was crowded (many agents). Plain `hf download` gave 0.3 MB/s per file; `HF_HUB_ENABLE_HF_TRANSFER=1` (pip `hf_transfer`, in `envs/sfxmore`) gave several MB/s but sometimes stalls at the tail of a file: wrap in a retry loop. Repeating `--exclude` or `--include` flags overrides the earlier ones in huggingface_hub 0.36, give several patterns after one flag (`--include "a" "b"`). `huggingface_hub` is 0.36.2 in this env (transformers 4.57 needs < 1.0).
10. `audioldm2` also pulled `pytorch_model.bin` duplicates into the cache by mistake (about 1.7 GB); harmless, delete them if space matters.

## UI inputs

No new task ids. All models use the existing `sfx`, `ambience` and `music` tasks and take only the prompt (plus `negative` for AudioLDM 1/2 via the prompt box). Model-specific fields: `struct_prompt` (text, Make-An-Audio 2 only), `guidance`, `steps`, `seconds`, `batch`, `seed`. Prompt style: the natural-caption `moss_sfx` style for sfx (these models were trained on AudioCaps-style captions, not Stable Audio tags), `ambience` and `music` for the others.

## Where each fits (my read)

- Quick second-opinion SFX with different timbre than SA3: TangoFlux (44.1 kHz stereo) first, then Tango 2, AudioLDM 2.
- Event ordering ("A then B then C"): Make-An-Audio 2 with a structured caption.
- Music: SA3 medium stays the pick. `audioldm2-music` and TangoFlux are usable for short lo-fi stingers only.
- Licences: everything except Make-An-Audio 2 is non-commercial. Do not ship their output in a commercial game.

## Reproduce the environment

```bash
cd studio
uv venv --python 3.12 envs/sfxmore
export VIRTUAL_ENV=$PWD/envs/sfxmore
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1" "torchaudio==2.9.1"
uv pip install "diffusers>=0.35" "transformers==4.57.3" accelerate sentencepiece protobuf scipy soundfile librosa einops \
  datasets pyyaml huggingface_hub safetensors numpy hf_transfer progressbar2 torchlibrosa h5py omegaconf pandas matplotlib \
  pytorch-lightning importlib_resources ftfy kornia pillow
# the PyPI resolver pulls a CPU torch on aarch64: put the CUDA build back, and torchvision with it
uv pip install --reinstall-package torch --reinstall-package torchaudio --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1" "torchaudio==2.9.1"
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 "torchvision==0.24.1"
# upstream code (vendor/)
git clone --depth 1 https://github.com/declare-lab/TangoFlux vendor/TangoFlux
git clone --depth 1 https://github.com/declare-lab/tango vendor/tango
git clone --depth 1 https://github.com/bytedance/Make-An-Audio-2 vendor/Make-An-Audio-2
git clone --depth 1 https://github.com/awei-6/BigVGAN vendor/BigVGAN-awei6
# weights (HF cache); add a retry loop around each on a slow link
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1; H=envs/sfxmore/bin/hf
$H download cvssp/audioldm-l-full
for r in cvssp/audioldm2 cvssp/audioldm2-large cvssp/audioldm2-music; do
  $H download $r --include "*.json" "*.txt" "*.model" "*/*.safetensors"; done
$H download declare-lab/TangoFlux; $H download declare-lab/tango2
$H download google/flan-t5-large model.safetensors "*.json" spiece.model
$H download sd2-community/stable-diffusion-2-1 --include "scheduler/*"
$H download ByteDance/Make-An-Audio-2; $H download SC4949/make-an-audio-models --include "bigvgan/*"
$H download google/t5-v1_1-large --include "*.json" spiece.model pytorch_model.bin
$H download google-bert/bert-base-uncased --include "*.json" vocab.txt model.safetensors
```

Exact versions are in `env-lock/sfxmore.txt` (torch 2.9.1+cu130, diffusers 0.39.0, transformers 4.57.3, librosa 1.0.0, numpy 2.5.4, pytorch-lightning 2.6.6).

Test method: `workers/sfxmore_worker.py` driven over its JSON-lines stdin/stdout by a small script (one `run` per prompt, GPU lock held), outputs scored with `docs/research/audio_snippets/qa_score.py` (CLAP branch). Fixed settings: seed 7, 100 steps, batch 1.
