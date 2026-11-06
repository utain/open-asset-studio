# TTS / voice clone, group B (`tts_clone_b`)

Registry: `models.d/tts_clone_b.yaml` (12 entries). Tests: `tests/test_tts_clone_b.py` (registry only, no GPU).
Three backends (one venv each, because the dependency sets conflict):

| backend | venv | worker | lock | models |
|---|---|---|---|---|
| `cloneb_index` | `envs/cloneb_index` (torch 2.9.1+cu130, transformers 4.52.1) | `workers/cloneb_index_worker.py` | `env-lock/cloneb_index.txt` | IndexTTS 2, IndexTTS 2.5 |
| `cloneb_amphion` | `envs/cloneb_amphion` (torch 2.9.1+cu130, **transformers 4.44.2**) | `workers/cloneb_amphion_worker.py` | `env-lock/cloneb_amphion.txt` | MaskGCT, Vevo, Vevo1.5 |
| `cloneb_hf` | `envs/cloneb_hf` (torch 2.9.1+cu130, transformers 5.22.1, diffusers, `dia2`) | `workers/cloneb_hf_worker.py` | `env-lock/cloneb_hf.txt` | Dia 1.6B, Dia2 2B, Bark, Bark small, VibeVoice 1.5B / 7B, CSM (blocked) |

Upstream clones in `vendor/`: `index-tts` (d9e41aa), `Amphion` (26f6883), `dia2` (8687268).

New task id: **`voice_convert`** (`studio: sound`), declared in this yaml (the `vc_sing` group declares the same id with the same
meaning). UI inputs: audio role `source` (clip to convert) and audio role `ref` (reference voice); param `mode`
(Vevo: `timbre` / `style` / `voice`, Vevo1.5: `timbre` / `style`). `tts` and `voice_clone` of Vevo/Vevo1.5 also accept an optional
second reference (role `timbre`) and Vevo1.5 a `melody` clip (it sings the text on that pitch contour).
Input roles the workers read: `ref` (all cloners), `emo` (IndexTTS emotion clip), `ref2` / `ref3` / `ref4` (VibeVoice speakers 2-4,
Dia2 speaker 2), `source`, `timbre`, `melody` (Amphion).

## Status

All checks: one tiny generation through the worker's JSON-lines protocol under the GPU lock, then Whisper large-v3-turbo
(`qa_score.py`) word recall against the script. "GPU" is the worker-reported resident size after load; `memory_gb` in the registry
adds margin. Load times are cold (first load after a cache eviction) unless noted.

| model | status | HF repo | licence | GPU | load / run | check |
|---|---|---|---|---|---|---|
| `indextts-2` | working | `IndexTeam/IndexTTS-2` + `facebook/w2v-bert-2.0`, `amphion/MaskGCT` (semantic codec only), `funasr/campplus`, `nvidia/bigvgan_v2_22khz_80band_256x` | bilibili Model Use Licence (commercial ok below 100 M MAU / RMB 1 bn) | 6.4 GB | load 33 s, 4 s line in 4 s | recall 1.0 for plain voice, text-emotion ("furious, shouting") and emotion-clip modes |
| `indextts-2.5` | working | `IndexTeam/IndexTTS-2.5` + same helpers | same, repo tag "other" (verify) | 5.2 GB | load 33 s, 2-3 s per line | recall 1.0 EN x3 and ES; `duration_factor` 1.5 turned 3.47 s into 5.20 s (x1.50); emotion vector mode works |
| `maskgct` | working | `amphion/MaskGCT` + `facebook/w2v-bert-2.0` | CC-BY-NC-4.0 (**commercial: false**) | 8.3 GB | load 62 s, ~9 s per line | recall 0.86 and 1.0 on the fox sentence ("riverbank" vs "river bank" is the only miss), tts (bundled voice) and clone |
| `vevo` | working (tts/clone, voice conversion x3 modes) | `amphion/Vevo` + torchaudio HuBERT-large | CC-BY-NC-4.0 (**commercial: false**) | 5.6 GB | load 39 s; conversion 4-10 s per 8 s clip; tts ~50 s per line | tts recall 0.86; conversions keep the source text (Whisper: identical sentence) |
| `vevo15` | working (tts/clone, convert x2, melody-conditioned singing) | `amphion/Vevo1.5` (singnet7k variant only) + Whisper medium | CC-BY-NC-ND-4.0 (**commercial: false**) | 8.3 GB | load 55 s; tts ~6 s, conversion 3.5-7 s, singing 9 s | tts recall 0.86 ("quicked"/"dock"); zh singing line 0.8; conversions keep the source text |
| `dia-1.6b` | working | `nari-labs/Dia-1.6B-0626` | Apache-2.0 | 3.0 GB (bf16) | load 6 s; 8 s of dialogue in 19 s (CFG) | two-speaker script with `(coughs)` / `(laughs)`: recall 1.0; clone from 4 s ref + transcript: 1.0 but Whisper hears an extra leading "All right." |
| `dia2-2b` | working | `nari-labs/Dia2-2B` + `kyutai/mimi` | Apache-2.0 | 7.6 GB | load 5 s; 6.5 s of dialogue in 13 s | dialogue recall 1.0; clone with two spoken prefixes recall 1.0 |
| `bark` | working | `suno/bark` (transformers format, 4.5 GB `pytorch_model.bin` + `speaker_embeddings/`) | MIT | 2.1 GB (fp16) | load 4 s; 8-10 s of audio in 10-20 s | `[laughs]` line 0.94 (Whisper wrote "Sunil" for "Suno"), sung `♪ ... ♪` line 1.0, 30 s multi-sentence (chunked) 1.0 |
| `bark-small` | working | `suno/bark-small` | MIT | 0.8 GB | load 7 s | same checks: 0.94, 1.0 |
| `vibevoice-1.5b` | working | `vibevoice/VibeVoice-1.5B-hf` (transformers conversion of `microsoft/VibeVoice-1.5B`) | MIT; Microsoft: research use, no impersonation | 5.0 GB | load 32 s; 9 s of 2-speaker audio in 7 s | two speakers, with and without a reference per speaker: recall 1.0 |
| `vibevoice-7b` | working | `vibevoice/VibeVoice-7B-hf` (conversion of the withdrawn VibeVoice-Large) | MIT (community-hosted copy, `verify: true`) | 17.4 GB | load 111 s; 9 s of audio in 17-21 s | recall 1.0 for tts and for 2 cloned voices |
| `csm-1b` | **blocked** | `sesame/csm-1b` | Apache-2.0, gated | - | - | repo needs a logged-in account that accepted the terms: `hf download sesame/csm-1b config.json` -> "Access denied. This repository requires approval", HTTP 401 on the file URL. Nothing downloaded; registry entry has `features.blocked`; worker path (`CsmForConditionalGeneration`) is written but never run. |

Whisper recall for the voice-conversion outputs is not a recall number (no target text): the converted clip of the Vevo demo source
("I'm a versatile zero-shot voice cloner. ...") transcribes to the same sentence as the source.

## Which VibeVoice is available (checked 2026-11-06)

* `microsoft/VibeVoice-1.5B` (re-released, MIT, original checkpoint format; needs the `vibevoice` pip package, not used) and
  `microsoft/VibeVoice-Realtime-0.5B` (2 GB, streaming, not added) exist. `microsoft/VibeVoice-Large` / `-7B` return 404 (still withdrawn).
* The registry uses the **`vibevoice/VibeVoice-1.5B-hf`** and **`vibevoice/VibeVoice-7B-hf`** conversions, which load with plain
  transformers >= 5.17 (`AutoModelForTextToWaveform`, chat template). The 7B copy is a community-hosted mirror of weights Microsoft pulled:
  keep `verify: true`, treat the licence as MIT with Microsoft's "research use" caveat.
* No audible AI-disclaimer was heard in the outputs (Whisper transcribed exactly the script). The model card says outputs are watermarked.

## Worker notes (what each one does)

* **IndexTTS** (`cloneb_index`): task `voice_clone` only (always needs a reference, role `ref`, 3-15 s). `emotion_mode`
  voice / audio (role `emo`) / vector (8 sliders) / text (`emo_text`, uses the bundled Qwen emotion model). 2.5 adds `language`
  (EN ZH JA ES AR) and `duration_factor` (0.5-2.0 speed/length). The worker builds `vendor/index-tts/_run/<variant>` from symlinks into the HF cache, so
  nothing is downloaded at run time. 22.05 kHz output.
* **Amphion** (`cloneb_amphion`): MaskGCT (tts with a bundled prompt voice or voice_clone; optional `target_seconds`), Vevo (tts, clone, convert
  timbre/style/voice), Vevo1.5 (adds singing). Reference transcript empty = Whisper medium transcribes it. Source clips trimmed to 40 s, refs to 15 s.
* **Dia / Dia2 / Bark / VibeVoice / CSM** (`cloneb_hf`):
  * Dia 1.6B: `[S1]`/`[S2]` script in `text`; clone = reference clip + exact transcript in `ref_text` (the transcript is prepended to the script, the
    model continues after it). 44.1 kHz. `seconds` caps the length (86 tokens per second).
  * Dia2: same script format; clone = role `ref` (speaker 1) and optional `ref2` (speaker 2) as spoken prefixes; the upstream hard-codes
    `openai/whisper-large-v3` for word timestamps, the worker swaps in the cached large-v3-turbo. 24 kHz, up to 2 min.
  * Bark: `voice` choice (random + 130 `v2/*` presets across 13 languages), long text is split at sentence ends (about 190 characters) and joined with
    `pause_s` of silence; tags `[laughs] [sighs] [music]`, `...`, CAPS, `♪ lyrics ♪`.
  * VibeVoice: script of `Speaker N: ...` lines (N = 1-4); a reference clip per speaker with roles `ref`, `ref2`, `ref3`, `ref4`; `cfg_scale`, `steps`
    (diffusion), optional `max_tokens`. Cancel is checked every token.
  * CSM: written for `speaker` ids and clone context, untested because the repo is gated.

## Gotchas

* **Partial HF snapshots.** `huggingface_hub >= 1.0` refuses `snapshot_download(local_files_only=True)` on a snapshot that lacks files
  (we only download what is needed, e.g. Dia's safetensors without the 6.4 GB `.pth` / `.bin` duplicates). `cloneb_hf` resolves the snapshot dir from `refs/main`
  itself. The older-hub envs (index, amphion) do not have the problem.
* **Downloads.** HF throughput here is erratic with the default client (single stream, stalls at 0.3 MB/s). What worked: the old
  `hf` CLI with `HF_HUB_ENABLE_HF_TRANSFER=1` in a retry loop (the sfxmore env), and `snapshot_download(max_workers=32)` for repos with hundreds of tiny
  files (Bark's 784 speaker-embedding files). `hf download --include a b c` takes patterns after one `--include`; repeating `--include` keeps only the last.
* **torchaudio 2.9 needs torchcodec** (no aarch64 wheel) for `save`/`load`: the Index worker routes both through soundfile.
* **IndexTTS** needs `descript-audiotools` (install `--no-deps`, then `absl-py tensorboard rich markdown-it-py docstring-parser` with the torch/transformers pins),
  and `wetext` replaces `WeTextProcessing` (no `pynini` wheel).
* **Amphion** needs `transformers==4.44.2` (4.57.3 fails in `LlamaAttention.forward() missing position_embeddings`), `pyworld` (builds, gcc is present),
  `six`, `torchcrepe`, and the `LangSegment` line-1 patch (see the install commands). espeak comes from the `espeakng-loader` wheel (no apt).
  Vevo also needs the torchaudio HuBERT-large checkpoint: the file in `~/.cache/torch/hub/checkpoints` is **1.26 GB** (an earlier partial download of 168 MB
  gave `failed finding central directory`). Vevo's TTS path is slow (about 50 s per line); its conversions are fast. Japanese g2p needs the open_jtalk
  dictionary that pyopenjtalk fetches from github on first use (not verified here).
* **Dia2 on transformers 5**: `MimiModel.decode(return_dict=False)` returns a 1-tuple, upstream unpacks two. The worker monkey-patches `MimiCodec.decode`.
* **VibeVoice** loads through transformers only from the `-hf` conversions; it needs `diffusers` for the noise scheduler.
* Dia (0626) sometimes adds a short leading word in clone mode; Dia and Bark change the voice per seed unless a preset / reference is given.
* The GPU is shared: wait times on `gpu.lock` dominated wall-clock; none of the figures above include lock waits.

## Reproduce

```bash
cd <repo>
export HF_HUB_DISABLE_XET=1
IDX="--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match"

# --- cloneb_hf: Dia, Dia2, Bark, VibeVoice, CSM
uv venv --python 3.12 envs/cloneb_hf && export VIRTUAL_ENV=$PWD/envs/cloneb_hf
uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch==2.9.1 torchaudio==2.9.1
uv pip install $IDX "torch==2.9.1+cu130" torchaudio==2.9.1 transformers==5.22.1 accelerate soundfile librosa numpy scipy sentencepiece safetensors tqdm diffusers
git clone --depth 1 https://github.com/nari-labs/dia2 vendor/dia2 && uv pip install --no-deps -e vendor/dia2
uv pip install sphn whisper-timestamped        # Dia2 (whisper-timestamped only for voice cloning)

# --- cloneb_index: IndexTTS 2 / 2.5
uv venv --python 3.12 envs/cloneb_index && export VIRTUAL_ENV=$PWD/envs/cloneb_index
git clone https://github.com/index-tts/index-tts vendor/index-tts        # d9e41aa
uv pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch==2.9.1 torchaudio==2.9.1
uv pip install $IDX -r env-lock/cloneb_index.txt         # exact versions (transformers 4.52.1, numpy 2.2, wetext, tiktoken, ...)
uv pip install --no-deps descript-audiotools flatten-dict julius ffmpy randomname argbind

# --- cloneb_amphion: MaskGCT, Vevo, Vevo1.5   (full recipe and lock: env-lock/cloneb_amphion.txt)
uv venv --python 3.12 envs/cloneb_amphion && export VIRTUAL_ENV=$PWD/envs/cloneb_amphion
printf "torch==2.9.1+cu130\ntorchaudio==2.9.1\ntorchvision==0.24.1\ntransformers==4.44.2\nnumpy<2\n" > /tmp/amph_pin.txt
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1 torchaudio==2.9.1
uv pip install $IDX -c /tmp/amph_pin.txt torchvision==0.24.1 transformers==4.44.2 accelerate onnxruntime setuptools unidecode "numpy<2" scipy librosa encodec \
  phonemizer g2p_en jieba cn2an pypinyin LangSegment pyopenjtalk pykakasi praat-parselmouth json5 ruamel.yaml tqdm openai-whisper ipython soundfile \
  safetensors espeakng-loader six pyworld torchcrepe
sed -i '1s/.*/from .LangSegment import LangSegment,getTexts,classify,getCounts,printList,setfilters,getfilters/' $VIRTUAL_ENV/lib/python3.12/site-packages/LangSegment/__init__.py
git clone --depth 1 --filter=blob:none https://github.com/open-mmlab/Amphion.git vendor/Amphion      # 26f6883
python -c "import torchaudio; torchaudio.pipelines.HUBERT_LARGE.get_model()"                         # 1.26 GB, run in envs/cloneb_amphion
python -c "import whisper; whisper.load_model('medium')"                                             # Vevo1.5 + auto transcripts

# --- weights (HF cache)
for r in IndexTeam/IndexTTS-2 IndexTeam/IndexTTS-2.5 facebook/w2v-bert-2.0 funasr/campplus nvidia/bigvgan_v2_22khz_80band_256x \
         amphion/MaskGCT amphion/Vevo nari-labs/Dia2-2B kyutai/mimi suno/bark-small vibevoice/VibeVoice-1.5B-hf vibevoice/VibeVoice-7B-hf; do hf download $r; done
hf download nari-labs/Dia-1.6B-0626 --include "*.json" "model-0000*.safetensors"     # skip dia-v1.pth / pytorch_model.bin
hf download suno/bark --include "*.json" "*.txt" "pytorch_model.bin" "speaker_embeddings/*"
hf download amphion/Vevo1.5 --include config.json "tokenizer/*" "contentstyle_modeling/ar_emilia101k_singnet7k/*" "acoustic_modeling/fm_emilia101k_singnet7k/*" "acoustic_modeling/Vocoder/*"
hf download amphion/MaskGCT semantic_codec/model.safetensors       # (already part of the full MaskGCT download)
```

Try a worker by hand: `PYTHONPATH=workers envs/cloneb_hf/bin/python workers/cloneb_hf_worker.py` and feed it one `{"op":"run",...}` line (see
`docs/CONTRACTS.md` section 2); registry entries in `models.d/tts_clone_b.yaml` are the `model` objects.
