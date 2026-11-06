# tts_clone_a: XTTS-v2, Chatterbox (+ multilingual), Zonos, OpenVoice v2, Spark-TTS, Llasa

Registry: `models.d/tts_clone_a.yaml` (8 entries, `tasks: {}`: no new task ids; all use `tts` and `voice_clone`, Spark-TTS also `voice_design`).
Backends: `clonea_xtts`, `clonea_chatterbox` (EN + multilingual), `clonea_zonos`, `clonea_openvoice`, `clonea_codec` (Spark-TTS + Llasa).
Workers: `workers/clonea_*_worker.py` + shared `workers/clonea_common.py`. Tests: `tests/test_tts_clone_a.py` (registry level). Locks: `env-lock/clonea_*.txt`.
Vendored repos (editable, `--no-deps`): `vendor/Zonos`, `vendor/OpenVoice`, `vendor/MeloTTS`, `vendor/Spark-TTS`.

Test: reference = a 4.6 s clip of a synthetic voice, not included in this repository (text: "Copy that command. Squad is in position and ready to move."),
line = "The bridge is secure. Hold your position until the signal, then fall back to the ridge." (seed 7). Driven through the worker JSON-lines
protocol under the GPU lock. Whisper large-v3-turbo on CPU gives word recall. Speaker similarity is a proxy: cosine of the Zonos ResNet293
speaker embeddings against the reference (same-speaker clones land about 0.5-0.8, unrelated voices about 0.0). GPU = `gpu_gb` the worker reports after load.

| model id | status | HF repo | licence | GPU | load / run | Whisper recall | speaker sim (clone) |
|---|---|---|---|---|---|---|---|
| xtts-v2 | working (tts, voice_clone) | coqui/XTTS-v2 | CPML, non-commercial | 1.75 GB | 15.6 s / ~1.3 s per 5 s | 1.00 (tts, clone, 16 s multi-chunk, es); ja 0.68 and zh 0.82 char recall | 0.66 (built-in voice -0.06) |
| chatterbox | working | ResembleAI/chatterbox | MIT (Perth watermark) | 3.0 GB | 9.4 s / ~2.4 s | 1.00 on 3 jobs (also exaggeration 1.1, cfg 0.3) | 0.74-0.78 (default voice 0.04) |
| chatterbox-multilingual | working (23 languages) | ResembleAI/chatterbox | MIT (Perth watermark) | 3.0 GB | 11.8 s / 2-4 s | 1.00 (es clone, en tts) | es clone 0.40 (cross-language), default voice 0.01 |
| zonos-v0.1-transformer | working | Zyphra/Zonos-v0.1-transformer | Apache-2.0 | 3.3 GB | 25.8 s / ~8.5 s per 5 s (eager) | 1.00 (tts, clone, angry) | 0.80-0.82 (random voice 0.04) |
| zonos-v0.1-hybrid | blocked | Zyphra/Zonos-v0.1-hybrid | Apache-2.0 | n/a | n/a | n/a | n/a |
| openvoice-v2 | working (English base) | myshell-ai/OpenVoiceV2 + MeloTTS-English | MIT | 0.32 GB | 5.4 s / 0.4 s clone, 3 s first tts | 1.00 (tts), 0.94 (clone) | 0.53 (base voice 0.27) |
| spark-tts | working (tts, voice_clone, voice_design) | SparkAudio/Spark-TTS-0.5B | CC-BY-NC-SA-4.0 per HF card, non-commercial | 2.7 GB | 35.8 s / ~9.5 s per take | 0.94 design ("ridge" heard as "rich"), 1.00 clone with and without ref text | 0.80-0.82 (design 0.19) |
| llasa-1b | working (tts, voice_clone) | HKUSTAudio/Llasa-1B + HKUST-Audio/xcodec2 | CC-BY-NC-4.0, non-commercial | 5.6 GB | 23-34 s / ~6.5 s for 6.8 s of speech | 1.00 (tts, clone with and without ref text) | 0.48 (weakest clone of the group) |

## Per-model notes

**XTTS-v2.** The maintained `coqui-tts` 0.28 fork, used through `TTS.tts.models.xtts` directly. 58 built-in speakers for `tts`, conditioning latents
for `voice_clone` (no transcript needed). Own sentence chunking by per-language character limit so long text is not truncated. 17 languages.
Japanese and Korean need `cutlet`, `fugashi`, `ko-speech-tools` (installed). coqui-tts insists on torchcodec being importable; there is no usable aarch64
wheel, so a broken 0.17 install only satisfies the availability check and the worker patches `torchaudio.load` to use soundfile (`patch_torchaudio_load`).

**Chatterbox / Chatterbox Multilingual.** `chatterbox-tts` 0.1.4 installed `--no-deps` with transformers 4.46.3 and numpy<2.3. Exposed: `exaggeration`, `cfg_weight`,
temperature, min_p, top_p, repetition_penalty. Multilingual needs `language`; for cross-language clones set `cfg_weight` 0 so the accent of the reference
is not copied. The Perth implicit watermark is part of the upstream code and stays on (`setuptools<81` is required for `pkg_resources`, otherwise the
watermarker silently becomes `None`). Worker gotcha fixed: a clone job replaces `model.conds`; the shipped conditionals are kept at load and restored for
`tts` jobs (before the fix a tts job after a clone job spoke with the cloned voice, sim 0.78 instead of 0.01). Do not `deepcopy` the conds (non-leaf tensors).

**Zonos v0.1 transformer.** `vendor/Zonos` editable, `backbone="torch"`. Needs espeak: `espeakng-loader` supplies the library, wired up in `setup_espeak()`. 8-way
emotion vector as `emo_*` params, plus speaking rate, pitch std, cfg. `disable_torch_compile=True`, so it runs eager at roughly RTF 1.6 (compile would be faster
but is not tried here). Use `torch.no_grad()`, not `inference_mode()`: the speaker-embedding model fails with "Inference tensors cannot be saved for backward".
**Hybrid is blocked**: it needs `mamba-ssm` and `causal-conv1d` CUDA kernels which cannot be built (no nvcc). The entry exists with `features.blocked`, the worker
refuses it and `probe` returns False; weights were not downloaded.

**OpenVoice v2.** MeloTTS English base (`vendor/MeloTTS`) then the tone-colour converter (`vendor/OpenVoice`). The converter is subclassed so the watermark
(not needed, and `enable_watermark=False` is broken upstream) is skipped. English only is exposed (the v2 checkpoint also covers other MeloTTS bases;
not wired). Base `tts` output is 44.1 kHz, clone output 22.05 kHz, and the clone peak is low (~0.28). Cloning is the weakest on voice match among the
non-codec models (0.53) but is the lightest and fastest (0.3 GB). Needed `gruut`, `g2pkk`, `unidic-lite` (the `unidic` package was removed, it broke MeCab).

**Spark-TTS 0.5B.** Qwen2.5 LM + BiCodec, `vendor/Spark-TTS` on `sys.path`, own generation loop (HF `generate` with a stopping criterion). `voice_design`:
gender, pitch, speed (levels) with global tokens pinned after the first chunk of a multi-piece text so the voice stays the same. `voice_clone` works with
or without the reference transcript. Licence: the code repo is Apache-2.0 but the HF model card says CC-BY-NC-SA-4.0; treated as non-commercial
(`verify: true` in the registry). Load pulls the 0.6 GB wav2vec2 feature extractor too.

**Llasa-1B + xcodec2.** Llama-3.2-1B speaking xcodec2 tokens, 16 kHz, en/zh. Clone is prompt continuation: the reference codes go in as the assistant prefix and
the prefix samples (`n * 320`) are trimmed from the output; the transcript is written by Whisper when none is given (adds ~10 s). Gotcha found and fixed in the
worker: transformers 4.57 renames SnakeBeta's `beta` to `bias` while loading, so xcodec2's 36 encoder activations silently stayed at init values. The tts
path does not use the encoder and was fine, but clones ran on without stopping (34.8 s). The worker copies the `*.act.beta` tensors from `model.safetensors`
into `.bias`; clones are now 6.8 s. It also builds `Wav2Vec2BertModel` from config only (the codec does not use its weights, saves a 2.3 GB download).
Clone similarity is the lowest of the group; the 3B variant was not tried.

## Environments

All are `uv venv --python 3.12 envs/<name>` and need the CUDA torch recipe (without the `+cu130` pin and `unsafe-best-match` PyPI serves the CPU wheel on aarch64):

    uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match "torch==2.9.1+cu130" "torchaudio==2.9.1"

- `clonea_xtts`: coqui-tts 0.28, transformers 5.22, six, cutlet, fugashi, ko-speech-tools, pypinyin, jieba, torchcodec (stub only).
- `clonea_chatterbox`: `chatterbox-tts==0.1.4 --no-deps`, transformers 4.46.3, numpy<2.3, librosa, s3tokenizer, diffusers, resemble-perth, conformer, setuptools<81.
- `clonea_zonos`: transformers<5, `-e vendor/Zonos --no-deps`, espeakng-loader, phonemizer, sudachi, inflect, kanjize.
- `clonea_openvoice`: numpy<2.3, transformers<5, `-e vendor/MeloTTS -e vendor/OpenVoice --no-deps`, gruut, g2pkk, nltk, unidic-lite.
- `clonea_codec`: transformers 4.57.6 with tokenizers 0.22.2 and huggingface_hub<1.0, `xcodec2==0.1.5 --no-deps`, vector-quantize-pytorch, torchtune, torchao 0.14.1, datasets<4.1, omegaconf, einx, einops, soxr.

Downloads: HF was heavily contended, so large files were fetched with a parallel ranged downloader (several connections per file, retry loop, sha check on xcodec2)
instead of `hf download`; `HF_HUB_DISABLE_XET=1`, no login. All repos used were ungated. Weight `files:` lists in the registry are tight (only what the worker
loads) so `status` is ready only when the worker can really start offline.

## Open items

- Zonos hybrid stays blocked until mamba-ssm kernels can be built (needs a CUDA toolkit with nvcc).
- Zonos could be faster with `torch.compile`; not tried (eager is acceptable for short game lines).
- Llasa-3B, OpenVoice non-English bases and Chatterbox-Turbo are not added.
- The speaker-similarity numbers are a proxy from one reference clip and one seed; they rank the models, they are not a benchmark.
