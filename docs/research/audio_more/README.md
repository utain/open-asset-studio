# 65 added audio models: index

All models below are registered in `models.d/<group>.yaml`, run in their own venv (`envs/<backend>`, see the
Environments table in `docs/CONTRACTS.md` section 7 and the README) and are usable from the Sound studio. One
report per group has the install recipe, measured timings, quality checks and gotchas.

* **Commercial** is the `license.commercial` flag of the registry entry (what the Models page shows). "No" means
  non-commercial or restricted weights or data; read the licence column and the report before shipping anything.
* **Status** is what the group report measured on the DGX Spark: *working* = at least one real generation checked;
  *blocked* = registered but not usable (missing weights, env or upstream code, so the Models page shows
  `missing_weights` / `missing_env` with a fix text) or known bad (`parler-large` shows *ready* but its output is babble).
* New Sound-studio tasks these models add: `voice_convert` (Voice > Convert), `sing` (Song > Sing a score),
  `s2s` (Voice > Reply), `v2a` (Video to sound). Extra inputs are declared per model with `input_slots`
  (`docs/CONTRACTS.md` section 1).


## Light TTS ([tts_light.md](tts_light.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `kokoro-82m` | Kokoro 82M | tts | Apache-2.0 | yes | working |
| `piper-en` | Piper (English voices) | tts | Per voice: Lessac (Blizzard 2013 licence), Amy (see mimic3-voices), libritts_r CC-BY-4.... | yes | working |
| `piper-thai` | Piper Thai (tsync2) | tts | CC-BY-NC-SA-3.0 (NECTEC TSync2 corpus via vaja-thai) | no | working |
| `melo-english` | MeloTTS English | tts | MIT | yes | working |
| `styletts2-libritts` | StyleTTS 2 (LibriTTS) | tts, voice_clone | MIT (code and weights) | yes | working |
| `parler-mini` | Parler-TTS Mini v1.1 | tts, voice_design | Apache-2.0 | yes | working |
| `parler-large` | Parler-TTS Large v1 | tts, voice_design | Apache-2.0 | yes | blocked: output is babble |

## Voice cloning A ([tts_clone_a.md](tts_clone_a.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `xtts-v2` | XTTS-v2 | tts, voice_clone | Coqui Public Model License 1.0 (CPML) | no | working |
| `chatterbox` | Chatterbox (0.5B) | tts, voice_clone | MIT | yes | working |
| `chatterbox-multilingual` | Chatterbox Multilingual (0.5B) | tts, voice_clone | MIT | yes | working |
| `zonos-v0.1-transformer` | Zonos v0.1 (transformer) | tts, voice_clone | Apache-2.0 | yes | working |
| `zonos-v0.1-hybrid` | Zonos v0.1 (hybrid) | tts, voice_clone | Apache-2.0 | yes | blocked: needs mamba-ssm / causal-conv1d kernels (no nvcc) |
| `openvoice-v2` | OpenVoice V2 (MeloTTS + tone-colour converter) | tts, voice_clone | MIT | yes | working |
| `spark-tts` | Spark-TTS (0.5B) | tts, voice_clone, voice_design | CC-BY-NC-SA-4.0 (HF model card; code Apache-2.0) | no | working |
| `llasa-1b` | Llasa 1B (+ xcodec2) | tts, voice_clone | CC-BY-NC-4.0 | no | working |

## Voice cloning B ([tts_clone_b.md](tts_clone_b.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `indextts-2` | IndexTTS 2 | voice_clone | bilibili Model Use License | yes | working |
| `indextts-2.5` | IndexTTS 2.5 | voice_clone | bilibili Model Use License | yes | working |
| `maskgct` | MaskGCT (Amphion) | tts, voice_clone | CC-BY-NC-4.0 | no | working |
| `vevo` | Vevo (Amphion) | tts, voice_clone, voice_convert | CC-BY-NC-4.0 | no | working (TTS path ~50 s per line) |
| `vevo15` | Vevo1.5 (Amphion, speech + singing) | tts, voice_clone, voice_convert | CC-BY-NC-ND-4.0 | no | working |
| `dia-1.6b` | Dia 1.6B (0626) | tts, voice_clone | Apache-2.0 | yes | working |
| `dia2-2b` | Dia2 2B | tts, voice_clone | Apache-2.0 | yes | working |
| `bark` | Bark (Suno) | tts | MIT | yes | working |
| `bark-small` | Bark Small (Suno) | tts | MIT | yes | working |
| `vibevoice-1.5b` | VibeVoice 1.5B | tts, voice_clone | MIT | yes | working |
| `vibevoice-7b` | VibeVoice 7B | tts, voice_clone | MIT | yes | working |
| `csm-1b` | Sesame CSM 1B | tts, voice_clone | Apache-2.0 | yes | blocked: gated repo, worker untested |

## Music and song ([music.md](music.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `musicgen-medium` | MusicGen Medium (1.5B) | music | CC-BY-NC-4.0 | no | working |
| `musicgen-melody` | MusicGen Melody (1.5B) | music | CC-BY-NC-4.0 | no | working |
| `musicgen-stereo-medium` | MusicGen Stereo Medium (1.5B) | music | CC-BY-NC-4.0 | no | working |
| `audiogen-medium` | AudioGen Medium (1.5B) | sfx | CC-BY-NC-4.0 | no | working |
| `magnet-medium-10secs` | MAGNeT Medium (10 s) | music | CC-BY-NC-4.0 | no | working |
| `riffusion-v1` | Riffusion v1 (spectrogram diffusion) | music | CreativeML OpenRAIL-M | yes | working |
| `acestep-1.5-turbo` | ACE-Step 1.5 Turbo (+ 1.7B planner) | song, music | MIT | yes | working |
| `diffrhythm-1.2-full` | DiffRhythm 1.2 Full (up to 4:45) | song, music | Apache-2.0 (code) / CC-BY-NC (MuQ-MuLan) | no | working |
| `diffrhythm-1.2` | DiffRhythm 1.2 (95 s) | song, music | Apache-2.0 (code) / CC-BY-NC (MuQ-MuLan) | no | working |
| `yue-s1-7b` | YuE (7B + 1B, English) | song | Apache-2.0 | yes | working (slow, ~8 min per 12 s) |
| `heartmula-3b` | HeartMuLa 3B (2026) | song, music | Apache-2.0 | yes | working |
| `magenta-rt2-small` | Magenta RealTime 2 Small (JAX) | music | CC-BY-4.0 (weights) / Apache-2.0 (code) | yes | working |
| `songgeneration-levo` | SongGeneration / LeVo (Tencent) | song | unknown (Tencent) | no | blocked: upstream code gone, not installed |

## More SFX / ambience ([sfx_more.md](sfx_more.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `audioldm-l-full` | AudioLDM L (full) | sfx, ambience | CC-BY-NC-SA-4.0 | no | working |
| `tangoflux` | TangoFlux | sfx, ambience, music | Stability AI Community + WavCaps (academic only) | no | working |
| `audioldm2` | AudioLDM 2 | sfx, ambience, music | CC-BY-NC-SA-4.0 | no | working |
| `audioldm2-large` | AudioLDM 2 Large | sfx, ambience, music | CC-BY-NC-SA-4.0 | no | working |
| `audioldm2-music` | AudioLDM 2 Music | music | CC-BY-NC-SA-4.0 | no | working |
| `tango2` | Tango 2 | sfx, ambience | CC-BY-NC-SA-4.0 | no | working |
| `make-an-audio-2` | Make-An-Audio 2 | sfx, ambience | MIT | yes | working |
| `stable-audio-open-1.0` | Stable Audio Open 1.0 | sfx, ambience, music | Stability AI Community | yes | blocked: gated repo (HTTP 401) |
| `stable-audio-open-small` | Stable Audio Open Small | sfx, ambience | Stability AI Community | yes | blocked: gated repo (HTTP 401) |

## Video to sound ([foley.md](foley.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `mmaudio-large-44k-v2` | MMAudio large 44k v2 | v2a, sfx | CC-BY-NC 4.0 (weights) | no | working |
| `hunyuan-foley` | HunyuanVideo-Foley (XXL) | v2a | Tencent HunyuanVideo-Foley Community | yes | working |
| `thinksound` | ThinkSound (light) | v2a | Apache-2.0 code | no | working |

## Speech-to-speech / omni ([omni.md](omni.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `kyutai-tts` | Kyutai TTS 1.6B | tts | CC-BY-4.0 (weights) | yes | working |
| `moshi` | Kyutai Moshi (Moshiko) | s2s | CC-BY-4.0 | yes | working |
| `qwen25-omni-3b` | Qwen2.5-Omni 3B | s2s | Qwen Research License | no | working |
| `qwen25-omni-7b` | Qwen2.5-Omni 7B | s2s | Apache-2.0 | yes | working |
| `qwen3-omni-30b-a3b` | Qwen3-Omni 30B-A3B Instruct | s2s | Apache-2.0 | yes | working (65 GB, 445 s cold load) |
| `glm-4-voice-9b` | GLM-4-Voice 9B | s2s | GLM-4-Voice License | yes | working |
| `minicpm-o-4_5` | MiniCPM-o 4.5 | s2s, tts, voice_clone | Apache-2.0 | yes | working (s2s, tts, voice_clone) |
| `step-audio-2-mini` | Step-Audio 2 mini | s2s | Apache-2.0 | yes | working |
| `kimi-audio-7b` | Kimi-Audio 7B Instruct | s2s | MIT (Qwen2.5 parts Apache-2.0) | yes | working |
| `llama-omni2-0_5b` | LLaMA-Omni2 0.5B | s2s | Academic research only (Apache-2.0 code) | no | working |

## Voice conversion and singing ([vc_sing.md](vc_sing.md))

| model id | name | task(s) | licence | commercial | status |
|---|---|---|---|---|---|
| `seed-vc` | Seed-VC | voice_convert | GPL-3.0 (code and weights) | yes | working |
| `rvc` | RVC (Retrieval-based Voice Conversion) | voice_convert | MIT (code) | yes | working |
| `diffsinger-opencpop` | DiffSinger (Opencpop, Mandarin) | sing | MIT code; Opencpop voice data CC BY-NC-ND 4.0 | no | working (Mandarin, one voice) |

65 models: 59 working, 6 blocked; 39 flagged commercial.
