# Voice conversion and singing voice (group `vc_sing`)

Registry: `models.d/vc_sing.yaml`. Workers: `workers/vc_worker.py` (backend `vc`, venv `envs/vc`) and
`workers/vc_diffsinger_worker.py` (backend `vc_diffsinger`, venv `envs/vc_diffsinger`). Locks: `env-lock/vc.txt`,
`env-lock/vc_diffsinger.txt`. Upstream clones in `vendor/`: `seed-vc`, `rvc`, `diffsinger-legacy` (used) and
`diffsinger` (openvpi, vendored, not wired, see below). Test registry: `tests/test_vc_sing.py`.

New task ids (both `studio: sound`, `modality: audio`):

| task | label | UI inputs |
|---|---|---|
| `voice_convert` | Voice conversion | audio input role `source` (the recording to convert). Seed-VC also needs role `ref` (3-25 s clip of the target voice). RVC needs no reference: a `voice` text param names a model in `data/rvc_voices/`. The worker falls back to input order (first audio = source, second = ref) when roles are missing. |
| `sing` | Singing voice | no audio input. A `score` textarea (one `<word> <note> <seconds>` line per syllable). |

## Status

| model | status | HF repo | licence | peak GPU | load / run | check |
|---|---|---|---|---|---|---|
| `seed-vc` speech mode | working | `Plachta/Seed-VC` (+ campplus, bigvgan 22k, whisper-small, rmvpe) | GPL-3.0 code and weights | 3.0 GB (2.7 resident) | load 10 s, 7 s clip at 30 steps 2.3 s | Whisper recall 1.0; median F0 116 Hz (male source) -> 185 Hz (female ref 197 Hz) |
| `seed-vc` singing mode (F0 44.1 kHz) | working | same + bigvgan 44k | same | same | 2.3-2.8 s per 7 s clip | F0 follows the ref range (201 Hz); also run on DiffSinger output |
| `rvc` | working | tool: `lj1995/VoiceConversionWebUI` (hubert_base, rmvpe); voice: `maotianchen/maobailing_rvc` | MIT tool; test voice Apache-2.0 | 1.24 GB | load 3.4 s, 7 s clip 2.5-2.8 s | Whisper recall 1.0; F0 116 -> 163 Hz with +6 semitones |
| `diffsinger-opencpop` | working (Mandarin, one female voice) | `PillowTa1k/DiffSinger` | code MIT, **voice data CC BY-NC-ND (Opencpop): commercial false** | 0.28 GB | load 2.8 s, 5 s song 1.0 s (GPU) / 36 s (CPU) | Whisper (zh) transcribes the demo line "小酒窝长睫毛是你最美的记号" correctly (traditional characters, one word wrong: 酒窝 -> 舊我); F0 370 Hz = F#4 as scored |
| so-vits-svc 4.1 | not added (documented only) | none usable | AGPL-3.0, repo archived | - | - | see below |
| OpenVoice v2 tone-color converter | skipped on purpose | - | - | - | - | owned by tts_clone_a |

## Seed-VC

* Code is not on the Hub and the GitHub name is `Plachtaa/seed-vc` (not `Plachta`). `git clone --depth 1 https://github.com/Plachtaa/seed-vc vendor/seed-vc`.
* The upstream wrapper (`seed_vc_wrapper.SeedVCWrapper`) loads both the speech model (DiT small, 22.05 kHz) and the F0 model
  (DiT base F0, 44.1 kHz) plus two BigVGANs, CAMPPlus, rmvpe and whisper-small in one go (about 2.7 GB). Its loader uses
  `hf_hub_download(cache_dir="./checkpoints")`; the worker replaces `load_custom_model_from_hf` with a resolver for the normal HF
  cache (offline safe). It also maps the F0 checkpoint named in the wrapper (`..._ft_ema.pth`) to the `_ft_ema_v2.pth` that `inference.py` uses
  (only the v2 file is downloaded).
* `singing: true` switches to the F0 model. `auto_f0_adjust` (default on) moves the source melody into the reference voice's range, so
  do not add a big `pitch_shift` on top of it (a female singer into a male reference with -12 gave 65 Hz).
* Not wired: Seed-VC v2 (`v2/ar_base.pth`, `v2/cfm_small.pth`, `modules/v2`, `inference_v2.py`; AR + CFM, accent/emotion conversion).
  The v1 models above are the ones the project recommends for singing.
* Weights in the HF cache: DiT small 440 MB, DiT F0 v2 821 MB, whisper-small 967 MB (fp32 safetensors), BigVGAN 22k and 44k (489 MB), campplus, rmvpe 181 MB.
* Ethics: the reference clip is a voice you are cloning. Use voices you have the right to use.

## RVC: adding a voice

`rvc` loads the upstream pipeline from `vendor/rvc` (RVC-Project, MIT; the current master uses a Transformers HuBERT, so no fairseq
is needed). A voice is a trained model, not something the worker can create:

```
data/rvc_voices/<anything>/<name>.pth        # RVC v1 or v2 inference model (weights + config), with or without f0
data/rvc_voices/<anything>/<name>*.index     # optional faiss index; used when its name does not contain "trained"
```

Put the `.pth` (and `.index`) in `data/rvc_voices/` (flat, or one folder per voice). In the job set `voice` to the file name
without `.pth` or to the folder name. Errors list what was found. The worker keeps two voices resident. Index strength (`index_rate`),
`protect`, `rms_mix_rate` and `f0_method` (rmvpe, pm) are exposed.

Bundled test voice: `data/rvc_voices/maobailing/` from `maotianchen/maobailing_rvc` (Apache-2.0; RVC v2, 40 kHz, Mandarin female,
trained only on IndexTTS-2 synthetic speech, so no real person is imitated). To fetch it:
`hf download maotianchen/maobailing_rvc maobailing_rvc.pth maobailing_rvc.index LICENSE --local-dir data/rvc_voices/maobailing`.
The `rvc` registry entry needs only the folder `data/rvc_voices` to exist (it is a path weight), so it shows "ready" before any voice is there
and the job then fails with a clear message. RVC voices of real people are common on the Hub; their licences are mostly "other" or unstated and
the consent question is open, so none were used.

Gotchas: `RVC_CUDA_GRAPH=0` is set (upstream CUDA-graph speed-up, untested on sm_121). `torch.load(weights_only=False)` is required for `.pth` voices
(plain dict pickles): only put voices there that you trust. The output sample rate is the voice's (40 kHz here).

## DiffSinger

* **openvpi/DiffSinger** (`vendor/diffsinger`, Apache-2.0) is the maintained fork with `.ds` projects and NSF-HiFiGAN, but it needs a trained
  voicebank in its own checkpoint format; open voicebanks are shipped as OpenUtau ONNX bundles outside the Hub. The only Hub
  checkpoints found (`PillowTa1k/DiffSinger`, `fox7005/nyaru_diffsinger`) are for the **original** MoonInTheRiver codebase
  (24 kHz, `DiffSingerMIDITask`, own HiFi-GAN), so they do not load in the openvpi code. `LogiAI10/diffsinger-mobile-onnx` (MIT) has two ONNX files
  with no README or provenance and no documented I/O: not used.
* **Original DiffSinger** (`vendor/diffsinger-legacy`, MIT) with the Opencpop end-to-end model `0228_opencpop_ds100_rel` (lyrics + MIDI -> mel, 100 diffusion steps),
  `0109_hifigan_bigpopcs_hop128` and the pitch extractor `0102_xiaoma_pe` works as the "lyrics + notes -> singing" path. Mandarin only, one female voice,
  24 kHz. The worker converts the line format to upstream's phoneme-level input (pypinyin + the Opencpop pinyin->phoneme table) and symlinks the three
  cached checkpoint folders into `vendor/diffsinger-legacy/checkpoints/` at load time. A JSON object in `score` (upstream's input dicts) is passed through.
* Patches without editing vendor files: `scipy.signal.kaiser = scipy.signal.windows.kaiser` (removed in scipy 1.13). `torch.load` default `weights_only=True` is enough for these checkpoints.
* Licence: the code is MIT but the voice comes from Opencpop (CC BY-NC-ND 4.0), so the entry says `commercial: false` and `verify: true`. The Hub repo that hosts the files is tagged Apache-2.0, which cannot lift the data licence.
* Score line example (`score` param): `小 C#4 0.41` / `长 A#4,F#4 0.51,0.18` (a slur) / `SP rest 0.22`. Sample: `{"score": "你 C4 0.5\n好 E4 0.5\nSP rest 0.2\n歌 G4,E4 0.4,0.4"}` -> 2.04 s.
* Not done: tempo/duration prediction from plain lyrics (the user gives note lengths), English or Japanese lyrics, other voices.

## so-vits-svc 4.1 (not added)

Upstream `svc-develop-team/so-vits-svc` (AGPL-3.0, **archived**) needs a per-speaker `G_*.pth` + `config.json`, a ContentVec checkpoint,
NSF-HiFiGAN and an F0 tracker (rmvpe/fcpe/crepe). No usable Hub voice was found with a clearly open licence (the open ones are character voices of
copyrighted shows, or CC-BY-NC-SA). RVC covers the same use case with a cleaner codebase, so only this note was written. If needed later: same
folder pattern as RVC (`data/svc_voices/<name>/{G_*.pth,config.json}`) in a third venv.

## Reproducing the setup

```bash
cd studio
git clone --depth 1 https://github.com/Plachtaa/seed-vc vendor/seed-vc
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI vendor/rvc
git clone --depth 1 https://github.com/MoonInTheRiver/DiffSinger vendor/diffsinger-legacy
git clone --depth 1 https://github.com/openvpi/DiffSinger vendor/diffsinger        # optional, unused

# envs/vc  (Seed-VC + RVC)
uv venv --python 3.12 envs/vc
VIRTUAL_ENV=$PWD/envs/vc uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match torch==2.9.1+cu130 torchaudio==2.9.1 "transformers==4.57.3" "numpy<2.3" librosa scipy munch einops \
    soundfile onnxruntime matplotlib pydub tqdm "huggingface_hub<1" pyyaml descript-audio-codec faiss-cpu praat-parselmouth
# NOTE: without the explicit torch==...+cu130 the PyPI aarch64 wheel (CPU only) wins. Exact versions: env-lock/vc.txt

# envs/vc_diffsinger  (original DiffSinger)
uv venv --python 3.12 envs/vc_diffsinger
VIRTUAL_ENV=$PWD/envs/vc_diffsinger uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match torch==2.9.1+cu130 "numpy<2" "librosa==0.9.2" click h5py "lightning~=2.3.0" matplotlib MonkeyType onnx \
    praat-parselmouth pyworld PyYAML resampy "scipy>=1.10,<1.14" tqdm soundfile "setuptools<81" numba "huggingface_hub<1" \
    pypinyin pyloudnorm textgrid pandas scikit-learn einops tensorboardX pretty_midi g2pM jieba chardet pycwt scikit-image webrtcvad \
    resemblyzer jiwer PyWavelets g2p_en
# (the legacy code imports training helpers at import time, hence the long tail; versions in env-lock/vc_diffsinger.txt)

# weights (HF cache; export HF_HUB_DISABLE_XET=1)
hf download Plachta/Seed-VC DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
    DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
hf download funasr/campplus campplus_cn_common.bin
hf download nvidia/bigvgan_v2_22khz_80band_256x ; hf download nvidia/bigvgan_v2_44khz_128band_512x
hf download openai/whisper-small config.json preprocessor_config.json model.safetensors
hf download lj1995/VoiceConversionWebUI rmvpe.pt ; hf download lj1995/VoiceConversionWebUI --include "hubert_base/*"
hf download PillowTa1k/DiffSinger 0228_opencpop_ds100_rel/config.yaml 0228_opencpop_ds100_rel/model_ckpt_steps_160000.ckpt \
    0109_hifigan_bigpopcs_hop128/config.yaml 0109_hifigan_bigpopcs_hop128/model_ckpt_steps_1512000.ckpt \
    0102_xiaoma_pe/config.yaml 0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
hf download maotianchen/maobailing_rvc maobailing_rvc.pth maobailing_rvc.index LICENSE --local-dir data/rvc_voices/maobailing
```

## Machine notes

* Hub downloads were extremely slow while other agents shared the link (50-400 KB/s per connection, some stalled at 90 %). `hf download`
  does not resume across invocations (the `.incomplete` suffix changes). What worked: keep the partial blob, fetch the rest with 6-8 parallel
  `curl -r a-b` ranges, concatenate, verify the sha256 equals the blob name, move it to `blobs/<sha256>` and symlink it into `snapshots/<rev>/`.
* The source clips for the tests were made with Kokoro-82M on CPU (`envs/ttslight`, read-only use): a male source ("am_adam") and a female
  reference ("af_heart"), 7 s each. espeak and piper are not installed.
* torch warns that sm_121 is above its maximum (12.0); everything ran.
* Not done: Seed-VC v2, streaming, cancel inside a Seed-VC chunk (cancel is checked per chunk and per take), so-vits-svc, openvpi DiffSinger voicebanks.
