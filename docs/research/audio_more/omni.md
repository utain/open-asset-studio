# Omni / conversational speech models (group `omni`)

Models that listen and answer in speech, plus one pure TTS from the same family. All run in batch (one turn per job);
real-time full-duplex streaming is out of scope. Registry: `models.d/omni.yaml`. Workers: `workers/omni_*_worker.py`.
One venv per incompatible dependency set: `envs/omni_{moshi,qwen,glm,step,minicpm,kimi,llama2}` (freezes in `env-lock/`).

## New task `s2s` ("Voice reply (speech-to-speech)", studio `sound`)

Input: one audio clip (asset, role `audio`) and/or a typed message. Optional params: `instruction` (system prompt / persona;
not on Moshi), `text` (typed message), `voice` (preset speaker where the model has them), `seconds` (cap on the reply),
`batch`, `seed`, `temperature`. Optional second input with role `voice_ref` (a short voice sample): Step-Audio 2 and MiniCPM-o
answer in that voice.
Output per take: `take_N.wav` (type audio) plus `take_N.txt` (attached) with the reply text; the text is also in the asset meta
(`worker.reply_text`, `final_prompt`). MiniCPM-o also serves `tts` and `voice_clone` (reference clip with role `ref`).
`seconds` is a hard token cap on Step-Audio, Kimi and Qwen: when the model wants to say more, the audio stops mid-sentence
while the reply text (which runs ahead of the audio) is longer than what was spoken.

## Results

All timings on the DGX Spark (GB10, bf16, SDPA attention, no flash-attn) with the test question
"Give me a brief introduction to the Great Wall" (3.8 s, English). `gpu GB` = `torch.cuda.memory_allocated` right after load
(the registry `memory_gb` adds activation/vocoder margin). Word recall = Whisper large-v3-turbo transcript of the
reply wav vs the model's own reply text (on the spoken part when the audio was cut by the `seconds` cap). No NaN, no
silence in any output.

| Model (registry id) | Status | HF repo | License | gpu GB | load / run | Audio | Recall |
|---|---|---|---|---|---|---|---|
| Kyutai TTS 1.6B (`kyutai-tts`, task tts) | working | `kyutai/tts-1.6b-en_fr` + `kyutai/tts-voices` | CC-BY-4.0 | 3.9 | 23 s / 6.5 s | 5.8 s EN, 4.8 s FR | .94 EN, 1.0 FR |
| Kyutai Moshi (`moshi`) | working | `kyutai/moshiko-pytorch-bf16` | CC-BY-4.0 | 16.2 | 105 s cold / 26 s | 13.9 s (clip + 10 s window) | .97 |
| Qwen2.5-Omni 3B (`qwen25-omni-3b`) | working | `Qwen/Qwen2.5-Omni-3B` | Qwen Research (non-commercial) | 11.2 | 64 s / 100 s | 41.5 s | .99 |
| Qwen2.5-Omni 7B (`qwen25-omni-7b`) | working | `Qwen/Qwen2.5-Omni-7B` | Apache-2.0 | 20.8 | 132 s / 47 s | 18.8 s | .98 |
| Qwen3-Omni 30B-A3B (`qwen3-omni-30b-a3b`) | working | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Apache-2.0 | 65.7 | 445 s cold / 20 s | 16.4 s | .98 |
| GLM-4-Voice 9B (`glm-4-voice-9b`) | working | `zai-org/glm-4-voice-9b` + `-tokenizer` + `-decoder` | GLM-4-Voice license (commercial use needs a free registration) | 19.7 | 119 s / 56 s | 29.7 s | 1.0 |
| MiniCPM-o 4.5 (`minicpm-o-4_5`) | working (s2s, tts, voice_clone) | `openbmb/MiniCPM-o-4_5` | Apache-2.0 | 19.8 | 76-103 s / 15 s s2s, 7 s tts | 20.8 s s2s, 5.0 s tts, 4.3 s clone | 1.0 / 1.0 / 1.0 |
| Step-Audio 2 mini (`step-audio-2-mini`) | working | `stepfun-ai/Step-Audio-2-mini` | Apache-2.0 | 18.8 | 103 s / 94 s | 19.2 s (cut by cap) | 1.0 (spoken part) |
| Kimi-Audio 7B (`kimi-audio-7b`) | working (needs 5 runtime patches, see below) | `moonshotai/Kimi-Audio-7B-Instruct` | MIT (Qwen2.5 parts Apache-2.0) | 24.4 | 48 s / 27 s | 15.8 s (cut by cap) | .97 (spoken part) |
| LLaMA-Omni2 0.5B (`llama-omni2-0_5b`) | working | `ICTNLP/LLaMA-Omni2-0.5B` + `ICTNLP/cosy2_decoder` + openai whisper large-v3 | academic use only (non-commercial) | 4.2 | 32 s / 7 s | 16.7 s | .96 |

Peak system-memory deltas seen while a worker was alive (include other tenants' noise): Moshi 45 GB, Qwen2.5-Omni-7B 26 GB,
Step 33 GB, Kimi 51 GB, Qwen3-Omni 82 GB (the 66 GB checkpoint was read through the page cache). Qwen3-Omni was only run with
MemAvailable of 109 GB; skip it when MemAvailable is below ~85 GB. Workers unload by dropping the model handle; every
test process exited cleanly after `unload`/`shutdown`.

Voice-clone similarity (MiniCPM-o with `voice_ref`/`ref`) was not measured, only that the output is intelligible and the
right length. Moshi needs no instruction and has none; it answers like a chatty small-talk partner (it opened with
"Hi, how is it going?" before answering).

## Env install recipes

Common: `uv venv --python 3.12 envs/omni_X`, then torch first (PyPI torch is `+cpu` on aarch64):

```
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1+cu130" "torchaudio==2.9.1"
```
Every later `uv pip install` in the same venv must pass `-c torch_constraint.txt` (`torch==2.9.1+cu130`, `torchaudio==2.9.1`) or uv
upgrades torch to a CPU build. Exact versions are in `env-lock/omni_*.txt`.

- `omni_moshi`: `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install moshi soundfile librosa hf_transfer` (sphn builds audiopus_sys with cmake 4).
- `omni_qwen`: `transformers==4.57.3 accelerate qwen-omni-utils soundfile librosa numpy` plus `torchvision==0.24.1` (from the cu130 index) and `audioread`.
- `omni_glm`: `transformers==4.44.2 "tokenizers<0.20" accelerate hyperpyyaml conformer==0.3.2 omegaconf hydra-core einops soundfile librosa scipy sentencepiece tiktoken "numpy<2" diffusers==0.27.2 "huggingface_hub<0.26" rich gdown pyrootutils matplotlib wget`.
- `omni_step`: `transformers==4.49.0 accelerate librosa onnxruntime s3tokenizer diffusers hyperpyyaml soundfile numpy sentencepiece tiktoken "huggingface_hub<1" einops`.
- `omni_minicpm`: `transformers==4.51.0 accelerate numpy pillow librosa moviepy onnxruntime onnx hyperpyyaml einops scipy soundfile tqdm sentencepiece vector-quantize-pytorch vocos`, then `uv pip install --no-deps "minicpmo-utils==1.0.6"` (1.0.7 expects different token2wav files), `s3tokenizer diffusers`, then `"huggingface_hub>=0.30,<1"` again (s3tokenizer/diffusers pull hub 2.x which breaks transformers).
- `omni_kimi`: `transformers==4.44.1 "tokenizers<0.20" accelerate librosa loguru timm torchdyn soundfile safetensors pyyaml tqdm "huggingface_hub<0.26" "numpy<2.3" sentencepiece tiktoken omegaconf`.
- `omni_llama2`: `transformers==4.43.4 tokenizers==0.19.1 "huggingface_hub<0.26" accelerate openai-whisper hyperpyyaml conformer==0.3.2 diffusers==0.27.2 omegaconf onnxruntime lightning inflect einops librosa soundfile sentencepiece "numpy<2" peft shortuuid wetext hydra-core rich gdown pyrootutils matplotlib wget pyarrow`.

Vendor clones in `vendor/`: `delayed-streams-modeling`, `GLM-4-Voice`, `Kimi-Audio` (needs `git submodule update --init --depth 1 kimia_infer/models/tokenizer/glm4`), `Step-Audio2`, `LLaMA-Omni2`, `MiniCPM-o`, `Matcha-TTS` (shallow clone of `shivammehta25/Matcha-TTS`: the `third_party/Matcha-TTS` dirs of GLM-4-Voice and LLaMA-Omni2 are empty or incomplete and cosyvoice imports `matcha.models`).
Extra weight: `~/.cache/whisper/large-v3.pt` (3.09 GB, sha256 `e5b1a55b...dadb`, URL in `whisper._MODELS`) for LLaMA-Omni2.
Use `HF_HUB_DISABLE_XET=1`; weights are in the default HF cache.

## Per-model notes and gotchas

**Kyutai TTS** (`moshi` package, `TTSModel`). Voices are pre-computed embeddings from `kyutai/tts-voices`; only CC0/CC-BY sets are
offered (12 voices). The repo's Expresso and EARS voices are non-commercial and not listed; the cloning encoder is not released.
`torch.compile`/CUDA graphs deadlock on sm_121 (futex, 8 minutes lost): the worker sets `NO_TORCH_COMPILE=1`, `NO_CUDA_GRAPH=1`.
FR works (recall 1.0); EN recall .94 only because Whisper heard the brand "Kyutai" as "Qtite".

**Moshi** (Moshiko bf16, same worker). Full-duplex model used as one turn: the clip is fed frame by frame, followed by silence for the
reply window (`seconds`, default 10); output length = clip + window, and it may start talking over the clip. English only,
no system prompt. `moshi` warns that the repo has no config.json (assumes a 7B) - harmless.

**Qwen2.5-Omni 3B/7B, Qwen3-Omni** (`omni_qwen`). Speech output works only with the exact default system prompt, so the user's
instruction goes into the user turn (Qwen2.5). Voices: Chelsie/Ethan (Qwen3 adds Aiden). Needs `torchvision` (AutoVideoProcessor) and
`audioread` (qwen-omni-utils) even for audio-only. 3B is slow for its size (HF generate, talker without kernels): 2.4 s of compute per second of speech.
The 3B uses the Qwen Research License (non-commercial), the registry says so. Qwen3-Omni: 66 GB checkpoint, 445 s cold load, 65.7 GB on the GPU; this is the only entry that effectively evicts everything else.
Thinker token cap = `seconds` x 6.

**GLM-4-Voice** (`omni_glm`). Speech tokenizer (Whisper VQ, 12.5 Hz) -> 9B LM -> flow + HiFT decoder at 22.05 kHz. The `seconds` cap is a token budget (12.5 audio tokens/s x 3), so a request of 15 s produced 29.7 s of speech. License is its own text: free for academic use,
commercial use requires a free registration at open.bigmodel.cn/mla/form (registry: `verify: true`).

**MiniCPM-o 4.5** (`omni_minicpm`). `model.chat(..., use_tts_template=True, generate_audio=True, output_audio_path=...)`.
Needs `minicpmo-utils==1.0.6`. The bundled `assets/HT_ref_audio.wav` is the default voice; a `voice_ref` clip (s2s) or `ref` clip
(tts/voice_clone) replaces it. 4.5 supersedes the 2.6 model named in the brief.

**Step-Audio 2 mini** (`omni_step`). Vendor `stepaudio2` + `Token2wav`; default female/male prompt wavs or a `voice_ref`. Reply text runs ahead
of the speech tokens, so a token cap cuts the audio first.

**Kimi-Audio 7B** (`omni_kimi`). Needed five in-process patches (all in the worker; nothing in site-packages or vendor is edited):
(1) an SDPA stand-in for `flash_attn` (GQA and bottom-right causal mask, checked against incremental decoding) with `flash_attn.bert_padding`,
and `is_flash_attn_2_available()` forced to True because the remote modeling code refuses to import otherwise;
(2) the old `THUDM/glm-4-voice-tokenizer` id is resolved to the local `zai-org/` snapshot;
(3) the bundled whisper-large-v3 safetensors has no metadata, which transformers 4.44 crashes on;
(4) transformers 4.44's `apply_rotary_pos_emb` indexes cos/sin by position_ids a second time (shape mismatch / OOB gather), so the >=4.48 non-indexing helper is swapped in;
(5) BigVGAN is built without its custom CUDA kernel (needs nvcc). Output is one fixed voice. 43 GB of weights (the flow detokenizer checkpoint is 19 GB).

**LLaMA-Omni2 0.5B** (`omni_llama2`). Smallest and fastest (4.2 GB, 7 s run), plain answers. Needs openai-whisper large-v3 as the speech encoder,
the `cosy2_decoder`, a `tn` (WeTextProcessing) stub because pynini has no aarch64 wheel (only the speaker-embedding part of the cosyvoice frontend is used),
a `torchaudio.set_audio_backend` stub (removed in torchaudio 2.9) and the `Matcha-TTS` clone. License is academic only. English only; the bilingual variants were not tried.

## Environment gotchas shared by all of them

- `torchaudio.load/save` need torchcodec (no aarch64 wheel): workers monkeypatch them to soundfile (and accept the `sample_rate=` keyword).
- `transformers` offline: pass `snapshot_download(repo, local_files_only=True)` paths.
- First-time loads of the 20-66 GB checkpoints are disk-bound (page cache cold); the `est.load_s` values are cold loads.
- Network: when ~20 downloads from all agents were running, my streams got 1-2 MB/s and `hf_transfer` threw away whole files after a few chunk errors.
  Downloads were done with a resumable 32 MB-chunk fetcher (16-40 threads) writing the HF blob layout, then a plain `hf download` for the symlinks.
- `drive.py`-style test (JSON lines over stdin/stdout, `PYTHONPATH=workers`, `HF_HUB_OFFLINE=1`) was used for every model; no studio server was started.
