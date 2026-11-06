# tts_light: Kokoro, Piper, MeloTTS, StyleTTS 2, Parler-TTS

Registry: `models.d/tts_light.yaml` (7 entries, no new task ids). Backends: `ttslight` (`envs/ttslight`, `workers/ttslight_worker.py`)
and `ttslight_parler` (`envs/ttslight_parler`, `workers/ttslight_parler_worker.py`, which subclasses the first).
Helper: `workers/ttslight_styletts2.py`. Tests: `tests/test_tts_light.py`. Locks: `env-lock/ttslight.txt`, `env-lock/ttslight_parler.txt`.

Test line (EN): "Welcome, traveler. The dragon sleeps beneath the old mountain, so speak softly." Whisper large-v3-turbo (CPU),
word recall (character recall for Thai). All runs went through the worker's JSON-lines protocol (`run` op), GPU runs under the lock.
Peak = `torch.cuda.max_memory_allocated` reported by the worker in `meta.gpu_peak_gb`; RAM = VmHWM of the worker process.

| model id | status | HF repo | licence | VRAM peak / RAM | load / run | Whisper recall |
|---|---|---|---|---|---|---|
| kokoro-82m | working | hexgrad/Kokoro-82M | Apache-2.0 | 0.69 GB / 2.2 GB | 3.2 s / ~1 s per 5 s clip | 1.00 (bf_emma) |
| piper-en | working (CPU) | rhasspy/piper-voices | per voice, see below | none / 0.74 GB | 0.5 s / ~1.5 s | 0.92-1.00 over 5 voices |
| piper-thai | working (CPU) | rhasspy/piper-voices (th_TH-tsync2-medium) | CC-BY-NC-SA-3.0, non-commercial | none / 1.1 GB | 0.5 s / ~3 s | 1.00 (characters) |
| melo-english | working | myshell-ai/MeloTTS-English | MIT | 0.86 GB / 2.3 GB | 4.1 s / ~2 s | 1.00 (EN-US) |
| styletts2-libritts | working (tts + voice_clone) | yl4579/StyleTTS2-LibriTTS | MIT | 1.25 GB / 2.4 GB | 4.3 s / ~2 s per take | 1.00 on 4 takes (2 tts, 2 clone) |
| parler-mini (v1.1) | working (tts + voice_design) | parler-tts/parler-tts-mini-v1.1 | Apache-2.0 | 4.4 GB / 5.2 GB | 29 s / ~5 s per 5 s clip | tts 0.83 (traveller / mountains), voice_design 1.00 |
| parler-large (v1) | BLOCKED: output is babble | parler-tts/parler-tts-large-v1 | Apache-2.0 | 9.8 GB / 9.8 GB | 70-87 s cold / ~7 s per 5 s clip | 0.0 (Whisper hears "oh, oh, oh ...") |

## Notes per model

**Kokoro 82M.** `kokoro` 0.9.4 + `misaki[en]`. 28 English voices (af/am US, bf/bm UK) exposed as a `voice` choice. Deterministic (no seed);
a single take per job. Other Kokoro languages (es, fr, hi, it, ja, pt, zh) and Thai are not exposed. Gotcha: misaki loads spaCy
`en_core_web_sm` and tries to `pip install` it at runtime; it is installed in the venv from the GitHub wheel instead. Share one `KModel` between
the per-language `KPipeline`s (done in the worker).

**Piper.** `piper-tts` 1.8.0 (piper1-gpl, GPL-3.0 code, used only inside its own worker process), onnxruntime CPU, `device: cpu` in the registry.
Two entries because licences differ: `piper-en` has en_US-lessac-high, en_US-amy-medium, en_US-libritts_r-medium (904 speakers, `speaker` param),
en_GB-alba-medium, en_GB-northern_english_male-medium. Voice licences from the MODEL_CARDs: lessac = Blizzard 2013 licence, amy = "see mimic3-voices"
(both flagged `verify: true`), libritts_r and alba = CC-BY-4.0, northern_english_male = CC-BY-SA-4.0. en_US-ryan-high (CC-BY-NC-SA) is downloaded
but deliberately not registered. `piper-thai` = th_TH-tsync2-medium (CC-BY-NC-SA-3.0, NECTEC TSync2). Thai needs `tltk` (installed `--no-deps`
because it pins scikit-learn~=1.2 which has no py3.12 aarch64 wheel; its runtime imports pandas, sklearn-crfsuite, gensim, nltk, unicode-rbnf are
installed separately) . Piper has no seed: takes differ through onnxruntime noise; `noise_scale` 0 makes them repeatable.
`snapshot_download(local_files_only=True)` rejects partially cached repos (piper-voices), so the worker resolves the snapshot dir by hand.

**MeloTTS English.** Lives in the `ttslight_parler` env (transformers 4.46.1). `melotts` pins transformers 4.27.4 and imports every language
module at import time (each downloads a tokenizer or needs mecab / jieba / gruut). The worker installs empty stand-in modules for
chinese/japanese/korean/french/spanish before `import melo`, and puts `vendor/MeloTTS-ttslight` (a copy of upstream, no git) on `sys.path`. Needs
`bert-base-uncased` tokenizer + model in the HF cache and the NLTK data (`averaged_perceptron_tagger_eng`, `cmudict`) in `~/nltk_data`.
Accents: EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default. Only the English checkpoint is set up (other languages would need their tokenizers/packages).

**StyleTTS 2 (LibriTTS).** Upstream code in `vendor/StyleTTS2` (MIT; its Utils/ASR, JDC, PLBERT files are in the clone). Wrapper:
`workers/ttslight_styletts2.py` (port of Demo/Inference_LibriTTS.ipynb). Needs `munch einops einops-exts nltk librosa phonemizer(-fork)` and
espeak through `espeakng-loader` (no system espeak-ng on this box). Gotchas: torch >= 2.6 `torch.load` defaults to weights_only=True, patched to False inside
the loader; `utils.py` of the repo (imports `monotonic_align`, training only) is never imported. The LibriTTS model is multispeaker, so the diffusion
sampler needs a reference style: plain `tts` uses one of six demo speakers from `reference_audio.zip` of the HF repo (`voice` param);
`voice_clone` takes the style from the reference clip (no transcript needed; `alpha`/`beta` mix reference and sampled style). Seeded; text is
split per sentence. Upstream asks that cloned voices be disclosed as synthetic.

**Parler-TTS.** `parler-tts` installed `--no-deps` from `vendor/parler-tts` into `envs/ttslight_parler` (it pins transformers==4.46.1). `dac` /
`audiotools` need `flatten-dict julius ffmpy randomname argbind importlib_resources tensorboard ftfy imageio librosa pyloudnorm markdown2`.
v1.1 uses two tokenizers (description: google/flan-t5-large tokenizer, cached; prompt: the repo's), v1 uses one (`features.two_tokenizers`). `tts` takes a named
speaker (Jon, Lea, Gary, Jenna, Mike, Laura, Lauren, Eileen) plus optional description; `voice_design` takes the free description in `instruction`
(prompt style `moss_voicegen`, which also fills `text`). English only. One 7-word test with "Jon" lost a plural ("mountains") and was spelt with
British "traveller" by Whisper, so recall 0.83; the voice-design clip was word-perfect. No flash-attn: eager/SDPA attention.

**Parler Large v1 (blocked).** Weights are complete (two shards, 9.3 GB) and the model loads and generates (9.8 GB peak, ~7 s), but the audio is
unintelligible babble. Tried: my worker (named speaker and free description), the model card's exact call (no masks), with attention masks,
temperature 0.7, and a different prompt ("Hey, how are you doing today?"); all babble (Whisper: "oh, oh, oh ..." or "so"). The same code path gives
clean speech with Mini v1.1, so the suspect is a mismatch between the v1 checkpoint and `parler-tts` 0.2.2 / transformers 4.46.1 (not investigated further:
an older parler-tts commit or the pre-v1.1 transformers pin are the next things to try). Registry entry kept with `features.blocked`.

## Environment install (reproducible)

```bash
cd studio
# envs/ttslight
uv venv --python 3.12 envs/ttslight
VIRTUAL_ENV=$PWD/envs/ttslight uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1" "torchaudio==2.9.1"   # ONLY the cu130 index: with pypi as extra index uv picks the CPU wheel
VIRTUAL_ENV=$PWD/envs/ttslight uv pip install kokoro "misaki[en]" espeakng-loader librosa piper-tts soundfile munch einops einops-exts pyyaml scikit-learn matplotlib pandas sklearn-crfsuite
VIRTUAL_ENV=$PWD/envs/ttslight uv pip install --no-deps tltk unicode-rbnf nltk gensim
VIRTUAL_ENV=$PWD/envs/ttslight uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
git clone --depth 1 https://github.com/yl4579/StyleTTS2 vendor/StyleTTS2
# envs/ttslight_parler
git clone --depth 1 https://github.com/huggingface/parler-tts vendor/parler-tts
uv venv --python 3.12 envs/ttslight_parler
VIRTUAL_ENV=$PWD/envs/ttslight_parler uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1"
VIRTUAL_ENV=$PWD/envs/ttslight_parler uv pip install "transformers==4.46.1" "tokenizers<0.21" sentencepiece protobuf soundfile numpy "huggingface_hub<1" accelerate \
   flatten-dict julius ffmpy randomname argbind importlib_resources ftfy imageio librosa pyloudnorm markdown2 pygments rich matplotlib einops pyyaml scipy tensorboard \
   g2p_en cached_path inflect txtsplit num2words unidecode anyascii nltk
VIRTUAL_ENV=$PWD/envs/ttslight_parler uv pip install --no-deps ./vendor/parler-tts descript-audio-codec descript-audiotools
cp -r <upstream MeloTTS checkout without .git> vendor/MeloTTS-ttslight     # github.com/myshell-ai/MeloTTS
envs/ttslight_parler/bin/python -c "import nltk; [nltk.download(x) for x in ['averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"
# weights (export HF_HUB_DISABLE_XET=1)
hf download hexgrad/Kokoro-82M
hf download rhasspy/piper-voices --include "en/en_US/lessac/high/*" "en/en_US/amy/medium/*" "en/en_US/libritts_r/medium/*" "en/en_GB/alba/medium/*" "en/en_GB/northern_english_male/medium/*" "th/th_TH/tsync2/medium/*"
hf download myshell-ai/MeloTTS-English; hf download bert-base-uncased
hf download yl4579/StyleTTS2-LibriTTS Models/LibriTTS/config.yml Models/LibriTTS/epochs_2nd_00020.pth reference_audio.zip
hf download parler-tts/parler-tts-mini-v1.1 --exclude "*.bin"; hf download parler-tts/parler-tts-large-v1 --exclude "*.bin"
hf download google/flan-t5-large tokenizer.json spiece.model tokenizer_config.json special_tokens_map.json config.json
```

Download note: the plain `hf download` ran at ~200 KB/s per stream on this shared link. Parallel ranged downloads into the cache blob dir
(`scratchpad/tl/pdl.py` style, 16 streams of 8 MB, then a quick `hf download` to create the snapshot symlinks) gave 3-8 MB/s.
