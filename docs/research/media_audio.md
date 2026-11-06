# Media audio: subtitles, stem separation, dubbing

Added to the Sound studio (tab **Media**: Subtitles / Stems / Dub). Registry: `models.d/media_audio.yaml`.
Worker: `workers/media_audio_worker.py` (backend `media_audio`, venv `envs/media_audio`), pure-Python helpers in
`workers/media_audio_common.py`, dubbing handler `app/plugins/media_audio.py` (`media.dub`).
Install recipe: `install/media_audio.yaml` (`./install.sh media_audio`), lock `env-lock/media_audio.txt`.
Tests: `tests/test_media_audio.py`, `tests/test_media_audio_dub.py` (no GPU, no weights).

Test machine: DGX Spark (GB10, aarch64, CUDA 13), 34 s test clip: one English speaker (TTS) over a music bed, as audio and as MP4.

## What was added

| id | task | what | licence | commercial | weights | load | run (34 s clip) | status |
|---|---|---|---|---|---|---|---|---|
| `whisper-subs-turbo` | subtitles | Whisper large-v3-turbo, word timings by cross-attention DTW, SRT / VTT / JSON | MIT | yes | 1.6 GB | 10 s | 2 s | ready |
| `whisper-subs-large-v3` | subtitles | same, plus Whisper's translate-to-English (turbo cannot translate) | MIT | yes | 2.9 GB | 17-20 s | 4-6 s | ready |
| `demucs-htdemucs` | separate | HTDemucs v4, 4 stems (default) | MIT | yes | ~0.3 GB | 1 s | 1.6 s | ready |
| `demucs-htdemucs-ft` | separate | fine-tuned HTDemucs (bag of 4), slower, a bit cleaner | MIT | yes | 0.4 GB | 1.7 s | 4.4 s | ready |
| `demucs-htdemucs-6s` | separate | 6 stems (adds guitar, piano; the piano stem is weak) | MIT | yes | 0.05 GB | 0.3 s | 1.4 s | ready |
| `melband-roformer-vocals` | separate | Mel-Band RoFormer vocal / instrumental (KimberleyJSN), `bs-roformer` 0.4.1 | MIT (model card) | yes, `verify: true` (training data undocumented) | 0.9 GB | 2 s | 13 s | ready |
| `dub-pipeline` | dub | separate -> Whisper -> translate -> voice clone -> time-fit -> mix -> MP4 | components' licences; the dub inherits the TTS model's licence | depends on the TTS model | (uses the above) | | 22 s with stems given, ~80 s end to end (incl. ~50 s LLM translation) | ready |

None of the weights are gated. Disk: venv 4.8 GB, `vendor/demucs` 80 MB, weights about 6 GB (Whisper 4.5 GB of it).

## Subtitles

* Input: audio, voice or video (ffmpeg decodes the first audio track). Output: one `text` asset (main `.srt` or `.vtt`, the other format and
  a `.json` with segments and word timings attached).
* Language: detected on the first 30 s, or chosen. `translate: true` (large-v3 only) gives English subtitles from any language.
* Cue building (`words_to_segments`): splits at sentence ends and pauses, balances the two lines of a cue, never leaves one-word
  orphans, handles scripts without spaces (Thai, Japanese, Chinese) by characters. Limits are params (max characters per line, lines, cue length, pause).
* Word timings: HF's `return_timestamps="word"` took about 1 s per second of audio here (attention caches copied to the CPU), so the worker aligns
  itself: eager attention for one forward pass per window with the decoded text, the model's `alignment_heads`, median filter, DTW. About 2 s per clip instead of 35-55 s.
  Windows start 1.5 s before the segment start (Whisper segment starts can be about 1 s early); if the window misses the speech (words squeezed into a few
  frames, seen on a vocal stem after a long pause) it is retried right after the previous words.
* Edge refinement: on clean audio (speech at least 24 dB above the quiet parts, e.g. a vocal stem) each sentence's start and end are snapped to the sound by
  short-time energy. With a steady noise or music bed it is skipped (energy cannot tell speech from background).
* Result on the test clip: cue starts within about 0.1 s of the true speech start on the mix and on the vocal stem; Thai and Japanese dubs are transcribed
  correctly by `whisper-subs-turbo` (language detected) and translated back to English by `whisper-subs-large-v3`.

## Stem separation

* Outputs (multi-output job): `vocals` and `instrumental` (default), or every stem of the model (`stems: all`); the same anti-clipping gain is applied to all.
* SDR on a synthetic mix (the clean TTS voice plus a music bed, so the reference is known), vocals / instrumental:
  HTDemucs 15.9 / 14.9 dB, HTDemucs-ft 16.5 / 14.7, HTDemucs-6s 16.1 / 13.2, Mel-Band RoFormer 18.7 / 18.0. The RoFormer is the best on this clip and 8x slower than HTDemucs; it is the right choice for dense music, the default stays HTDemucs.
* Demucs comes from the `adefossez/HTDemucs*` Hugging Face mirrors through `demucs.hf.get_hf_model` (needs demucs 4.1 from git, not PyPI 4.0.1, whose
  loader wants the training dependency `dora`). Mel-Band RoFormer is rebuilt with `bs_roformer.MelBandRoformer` from an embedded config; the checkpoint was
  trained with a different block layout, so the worker uses `linear_transformer_depth=0` and remaps the layer indices (strict load). Inference is fp32 (bf16
  autocast breaks `view_as_complex`), chunked with overlap-add, progress and cancel per chunk.

## Dubbing

Pipeline (`media.dub`, runs in the core server and drives workers through the normal queue):

1. Separate the speech from the background (default HTDemucs; `separate_model` selects another). Roles `vocals` / `bg` on the job inputs skip this step.
2. Transcribe the vocals with Whisper (segments from the word timings; lines over silence in the stem are dropped as hallucinations).
3. Translate the lines with the local prompt LLM (or any text backend id in `translate_backend`, e.g. `cli:claude@haiku`): JSON schema, batches of 12 lines, the source line and its slot length are
   sent so the translation stays about as long as the original. If no backend works the job ends as an editable **draft**.
4. Speak each line with a voice clone of the original speaker (reference = the cleanest 6-12 s of the vocals stem with its transcript), or the voice given in role `ref`.
   Any ready `voice_clone` model can be chosen (`tts_model`); the target languages offered are the ones that model supports (`GET /api/x/media_audio/dub/options`;
   models without a language list are treated as unrestricted).
5. Time-fit (`fit_plan`): a line longer than its slot first uses the silence that follows it, then is sped up with ffmpeg `atempo` by at most `max_speedup` (1.3 default); what still
   overflows is reported as a warning (the next line starts later instead of being cut).
6. Mix over the instrumental with ducking (`duck_db` under speech), loudness through `media.export_audio` (two-pass loudnorm to `lufs`, true peak below -1 dBFS; a private gain + soft-limiter fallback exists but is not needed when the core helper is present).
7. Outputs: dubbed audio (WAV), subtitles of the dub (SRT + VTT, text asset), for video input an MP4 with the video stream copied and the dub as AAC (`keep_original_track` adds the original as track 2), and the stems when they were made.

`stage: draft` stops after step 3: the outputs are the stems and a `dub plan` text asset (editable lines). The Sound studio opens it in **Edit lines & render**
(edit translations, untick lines, translate the empty ones, render), which submits a normal `dub` job with `segments` and the stems as inputs (22 s for the test clip).

### Results (test clip, 9 lines, one speaker, OmniVoice, translation by `cli:claude@haiku`)

| direction | result |
|---|---|
| EN -> TH | 34.0 s dub at -16.0 LUFS, true peak -1.1 dBFS; MP4 h264 copy + AAC. Thai lines intelligible; 2 lines overrun their slot by 0.8-1.0 s (reported). Back-translated with large-v3: all sentences recovered except small slips ("small teams" -> "Lick Lick" in one cue). |
| EN -> JA | same size and loudness; 3 overrun warnings (0.4, 1.1, 1.1 s). Back-translated by large-v3 almost word for word for 7 of 9 lines; the two lines that overrun are the garbled ones. |
| draft -> edited -> render | works through the UI and through the API; the stems from the draft are reused (no second separation). |

Honest quality notes:
* Overlong translations are the main weakness: Thai and Japanese lines were 10-40 % longer than the English slot, and `atempo` is capped at 1.3x to stay natural, so a few lines run into the next one. Shorter translations (hand-edit in the draft) or a higher `max_speedup` fix it.
* The cloned voice keeps the speaker's timbre but not the original's emotion or pacing; one speaker only (no diarization). Check names and numbers in the translation.
* Translation was tested with `cli:claude@haiku` only. The local llama-server (default backend) could not be tested on this machine (its port is reserved by another service); the code path is the same textgen call, and a failure there produces the editable draft.
* Only dub voices you have the right to clone.

## Loudness

The dub reuses core's `media.export_audio` (two-pass loudnorm). The subtitles and separation tasks do not alter loudness (the separation output only gets a shared anti-clip gain).

## Problems met

* `pkill -f` patterns match their own shell: a helper script uses `pgrep -f "[m]edia_audio_worker.py"`.
* demucs 4.0.1 needs `dora`; solved with demucs 4.1.0 from `vendor/demucs` installed `--no-deps`.
* bs-roformer >= 1.0 imports `torch_einops_utils` / `PoPE_pytorch` (not installable here); pinned to 0.4.1.
* Mel-Band RoFormer checkpoint layout differs from bs-roformer's current block order (see above).
* HF word-level timestamps too slow (see above); DTW written in `media_audio_common.dtw`.
* Whisper segment timestamps can be a second off on mixes and very wrong on stems with long exact silence; handled by the retry and the energy snap.
* `chatterbox-multilingual` (another group's registry entry) has a `no` language choice that YAML reads as `False`; `norm_lang` maps it back to `no`.

## Peak memory

Not measured per call (unified memory shared with other jobs); declared in the registry (`memory_gb`): Whisper turbo 4, large-v3 7, HTDemucs 5, HTDemucs-ft 6, HTDemucs-6s 5, Mel-Band RoFormer 6, dub pipeline 6 (plus the chosen TTS model). These are planning figures, not measurements.
