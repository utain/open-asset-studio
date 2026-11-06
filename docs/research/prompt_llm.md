# Local LLM for "Improve prompt": feasibility (DGX Spark / GB10)

Date: 2026-11-05. Status: **feasible, proven.** One llama-server process on 127.0.0.1, started
lazily, put to sleep when idle, using JSON-schema-constrained output.

## TL;DR recommendation

| Role | Model (local file) | Why |
|---|---|---|
| **Default** (text + vision) | `Qwen3.6-35B-A3B-Q4_K_M.gguf` + `mmproj-Qwen3.6-35B-A3B-BF16.gguf` (LM Studio dir, see paths) | MoE, about 3B active params: **63-75 tok/s**, about 5-6 s per full JSON answer, best quality tested, good Thai, vision works. About 21-22 GB while loaded, 0.2 GB while asleep. |
| **Fast / low-memory fallback** | `gemma-4-E2B-it-UD-Q4_K_XL.gguf` (HF cache, unsloth) | 2.3 GB, about 100 tok/s, 10 s load. Usable quality with small formatting slips. No vision (mmproj not on disk). |
| Emergency fallback (no llama.cpp) | ollama `gemma3:4b` @ `127.0.0.1:11434` (default ollama address) | Supports a JSON schema via `format`. Quality is poor (Thai leaks into the English prompt, notes in the wrong language). Use only if llama-server cannot start. |
| Rejected | Qwen3.6-27B dense Q4_K_M (and the other dense 27B GGUFs) | **4.6-10 tok/s**, so 40-90 s per request. Too slow for a UI button. |
| Not tried (HF-format: Qwen3.8-27B FP8/NVFP4, Nemotron-3-Nano NVFP4, Qwen3.6-27B AutoRound, gemma-4-E4B, InternVL3.5-14B) | need vLLM/SGLang | vLLM preallocates `gpu_memory_utilization`, cold start takes minutes, and it cannot release memory while running. That works against "unload on idle so image/audio models get memory". A `vllm-node:latest` image and recipes exist (`<spark-vllm-docker>/recipes/nemotron-3-nano-nvfp4.yaml`, `gemma4-26b-a4b.yaml`) if a server LLM is ever wanted. |

## Environment findings

* **llama.cpp** `<llama.cpp>/build/bin/llama-server`: version 9040
  (`07eaf919e`, 2026-05-06), GNU 13.3 aarch64, `GGML_CUDA=ON`, FA on, CUDA graphs on, NCCL on.
  It sees `NVIDIA GB10, compute capability 12.1, VMM: yes, 124608 MiB`. It **works on GB10**.
  * Caveat: built with `CMAKE_CUDA_ARCHITECTURES=100` (`CUDA : ARCHS = 1000` in the log). It runs on
    sm_121 through PTX JIT. The cost shows on the **first request after a fresh model/kernel set**:
    TTFT was 23.8 s (Qwen MoE) and 7.9 s (gemma). Later starts hit the CUDA JIT cache and the first
    TTFT is normal. Optional fix: rebuild with `-DCMAKE_CUDA_ARCHITECTURES=121a-real` (or `121`) and
    raise `CUDA_CACHE_MAXSIZE` (for example 4 GB) so the JIT cache is not evicted.
  * Flags this build supports that matter here: `--sleep-idle-seconds`, router mode
    (`--models-dir/--models-preset/--models-max/--models-autoload`), `--reasoning on|off`,
    `--reasoning-budget`, `--json-schema(-file)`, `--grammar(-file)`, `--mmproj`, `--fit`,
    `--cache-ram`, `--kv-unified`.
  * **Gotcha:** `--no-mmproj-auto` is an alias of `--no-mmproj` and **disables an explicit
    `--mmproj`** (the server then answers "image input is not supported"). Leave it out.
* **GGUFs on disk.** The HF cache has: `unsloth/Qwen3.6-27B-GGUF` Q4_K_M (16 GB, dense),
  `unsloth/gemma-4-E2B-it-GGUF` UD-Q4_K_XL (3.0 GB), and `unsloth/MiniMax-H3-GGUF`. MiniMax-H3 holds
  two oddly named files (`minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf` 9 GB and
  `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` 17 GB). Provenance is unclear, so they were not tested.
  `scb10x/llama3.2-typhoon2-1b-instruct` is **safetensors only (no GGUF)** and is not needed:
  Qwen3.6 handles Thai well.
  **`~/.lmstudio/models/` has more, including vision projectors** (the inspiration for the default):
  * `lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf` (20 GB) + `mmproj-Qwen3.6-35B-A3B-BF16.gguf` (861 MB)
  * `lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/…Q4_0.gguf` (14 GB) + `mmproj-…BF16.gguf` (1.2 GB): MoE, a plausible second opinion, not benchmarked
  * `lmstudio-community/Qwen3.8-27B-GGUF` Q4_K_M + mmproj (dense, so slow, same class as 3.6-27B)
  * `unsloth/Qwen3.6-27B-GGUF` UD-Q4_K_XL + `mmproj-F32.gguf` (1.8 GB) (dense, slow)
  * plus uncensored/heretic variants and Ornith-35B (not considered)
  * **Advice:** copy or symlink the two chosen files into an app-owned dir (for example
    `<repo>/models/llm/`) so the app does not depend on LM Studio's folder layout.
* **mmproj for the fallback:** gemma-4-E2B has no local mmproj. If vision is wanted in the small
  model, get `mmproj-F16.gguf` / `mmproj-BF16.gguf` from the same repo `unsloth/gemma-4-E2B-it-GGUF`
  (small, under 1 GB; check the exact size on the repo page; not fetched). It is not needed, because
  the default model already does vision.
* **ollama** (tested in a local container, `127.0.0.1:11434`). Only `gemma3:4b` (Q4_K_M, vision-capable) is installed. Measured: 9.5 s load,
  52 tok/s, JSON valid with `format: <schema>`. Called with `keep_alive: 0`, it unloaded again
  (`/api/ps` returned empty). Nothing was modified.
* **Other folders (read-only):** `<gateway>` is LiteLLM pointing at LM Studio on `:1234`
  (`qwen3.6-35b-a3b`, `qwen3.6-27b`, …). That confirms the user already uses Qwen3.6-35B-A3B
  interactively. `localai/compose.yaml` is the generic LocalAI l4t-cuda13 image (phi-2). No
  llama.cpp recipes in sglang/vllm.

## Measurements

Setup: `-ngl 999 -fa on --jinja -c 16384 -np 2 --reasoning off` on 127.0.0.1:7881. The test set
was 5 ideas (3 Thai, 2 English) x 4 targets = 20 requests, each with N=3 variations,
`temperature 0.7`, and a `json_schema` response_format. Memory is per-process GPU memory from
`nvidia-smi` (unified memory, so it comes out of the same 121 GB pool). ComfyUI and others were
running at the same time, so the "available" drop is noisy.

| Model | Load (cold / warm) | Mem loaded | Gen tok/s (1 stream) | Agg tok/s (2 streams) | TTFT (500-tok prompt) | Avg wall / request | JSON valid | All checks pass |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M (+mmproj) | 47 s / 5-21 s | 21.0 GB (22.1 GB with mmproj) | **62-77** | **88-91** | 250-400 ms | **5-6.6 s** | 20/20 | 19/20 (v1 templates) then **20/20 (v2)** |
| gemma-4-E2B UD-Q4_K_XL | 10 s | **2.3 GB** | 88-106 (24 on the JIT-cold first calls) | n/m | 45-200 ms | 5.4 s (3 s warm) | 20/20 | 17/20 |
| Qwen3.6-27B dense Q4_K_M | 14 s | 17.4 GB | **4.6-10** | n/m | 0.8-1.0 s | 40-93 s | (run aborted after 8 req) | n/m |
| ollama gemma3:4b | 9.5 s | 3.7 GB | 52 | n/m | n/m | 14 s incl. load | 1/1 | 0/1 (Thai in prompt) |

**Sleep on idle** (`--sleep-idle-seconds 15`, Qwen MoE): after the idle timeout, GPU memory fell
from 22,162 MB to **195 MB** (about 19 GB "available" came back). `/health` still returns `ok` and
`/props` shows `is_sleeping: true`. The next request **wakes it automatically**: 14.6 s total for a
256-token request, about 10 s of reload from page cache. Prompt cache is lost on sleep, which is
harmless for this workload.

**Vision** (Qwen3.6-35B-A3B + mmproj, 576x768 JPEG of a gold Thai Buddha statue): 810 prompt
tokens, TTFT 4.4 s, 77 tok/s, 10.2 s total. It identified "seated Buddha statue in the traditional
Thai Sukhothai style … flame-shaped ushnisha … black wall with golden floral motifs and devas", and
the Thai note said "ปางมารวิชัย" (correct mudra). This is good enough to bootstrap prompts.

## JSON reliability

Same 20 cases on Qwen3.6-35B-A3B, varying only the constraint:

| Mode | Parseable JSON | Has all required keys / N items | Notes |
|---|---|---|---|
| `response_format: {type: json_schema, json_schema: {schema, strict}}` | 20/20 | **20/20** | The grammar is compiled from the schema. `minItems/maxItems = N` is enforced. |
| `response_format: {type: json_object}` | 8/8 | 5/8 | Valid JSON but drops `variations`/`notes` |
| no constraint (prompt only) | 19/20 | 12/20 | 1 nested object instead of string, 7 missing keys |
| raw GBNF `grammar` field | 1/1 | 1/1 | Works, but the schema route is simpler. The GBNF test (with a 1-line system prompt) also showed that the constraint does not supply content: variations were identical. |

Conclusion: **always send `json_schema`**, keep the full system prompt, and still validate in the
app (cheap content checks: no Thai in English fields, variations differ from `prompt`, lyrics
contain markers). Retry once at a higher temperature on a check failure.

### JSON schemas (what the harness sends)

```json
{"type":"object","additionalProperties":false,
 "required":["prompt","negative","variations","notes"],
 "properties":{
  "prompt":{"type":"string","minLength":10},
  "negative":{"type":"string"},
  "variations":{"type":"array","items":{"type":"string","minLength":10},"minItems":3,"maxItems":3},
  "notes":{"type":"string"}}}
```
music adds `"lyrics":{"type":"string"}`, and voice adds `"line":{"type":"string"}` (both required).

Request body:
```json
{"messages":[{"role":"system","content":"<preamble + target template>"},
             {"role":"user","content":"IDEA: ...\nUSER_LANG: th\nN_VARIATIONS: 3\nCONTEXT: ..."}],
 "temperature":0.7,"top_p":0.9,"max_tokens":1500,
 "chat_template_kwargs":{"enable_thinking":false},
 "response_format":{"type":"json_schema","json_schema":{"name":"sfx","strict":true,"schema":{...}}}}
```
For vision, `content` becomes `[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]`.
Downscale images to 768 px or less first, which keeps prompt tokens near 800.

## Quality: judged outputs

Verdicts: **Qwen3.6-35B-A3B** follows the target conventions closely:
* SFX: comma-separated descriptors in the requested order.
* Music: real BPM and instruments.
* Voice: age, timbre and pace, with the quoted Thai line kept verbatim.
* 3D: single subject, three-quarter view, plain grey background, the full negative list.
* Thai to English translation is faithful, and the notes are fluent Thai.

The v1 template had two flaws, both fixed in the current `prompt_templates.md`:
1. For "spooky forest + witch" -> music, it returned **visual tags** ("cinematic lighting, detailed
   textures") and empty lyrics. Fixed with "SOUND ONLY / CONTEXT only where it fits". The v2 rerun
   gave `dark ambient, horror game soundtrack, eerie atmosphere, minor key, slow tempo 60 BPM, low
   strings, dissonant woodwinds…`.
2. For English ideas it sometimes wrote the voice line and notes in Thai and added "speaks Thai".
   Mitigated by having the app pass `USER_LANG` explicitly. In the verification run (voice + music
   x 5 ideas, 10/10 valid JSON), English ideas now get English lines ("The trees are watching me…
   but I'm not afraid!") and English notes in 3 of 4 cases. One music `notes` was still Thai, and
   the app check flags it (`thai_for_en_user`), so retry once. This run averaged 44 tok/s because
   other GPU jobs were running at the same time; expect slowdowns like this when ComfyUI is busy.

Weaker but acceptable: variations are sometimes too similar (3D variations keep a long shared
suffix), and one variation duplicated `prompt` (v1). The rule "never identical to prompt" was added.

**gemma-4-E2B** gets the gist right with good SFX/music tags, but makes small slips:
* a "Prompt:" prefix inside variations
* lyric markers inline instead of on their own lines
* English notes for Thai ideas (3/20)
* the voice `line` just repeats the idea
* invents negatives

**Ollama gemma3:4b** put "Thai vocalizations (notes: ดาบฟันโดนเกราะเหล็ก)" into the SFX prompt, so it
is not usable for quality.

### Pasted outputs: Qwen3.6-35B-A3B (v1 templates unless marked)

SFX, `ดาบฟันโดนเกราะเหล็ก` ("sword strikes iron armor"):
```json
{"prompt":"steel sword striking iron plate armor, sharp metallic ring, close-mic perspective, short 1-second impact, high quality, foley, sound effect, 44.1kHz stereo",
 "negative":"music, melody, speech, vocals, distortion, clipping, low quality, noise, hum, echo",
 "variations":["heavy broadsword hitting thick steel plate, deep dull thud with low metallic resonance, close-mic proximity effect, short 1-second impact, high quality, foley, sound effect, 44.1kHz stereo",
  "sharp rapier piercing chainmail, crisp high-pitched clang, intimate close-mic, short 1-second impact, high quality, foley, sound effect, 44.1kHz stereo",
  "blunt sword clash on iron shield, loud ringing metallic boom, small stone room reverb, sustained loop fade out, high quality, foley, sound effect, 44.1kHz stereo"],
 "notes":"แปลงแนวคิดดาบฟันโดนเกราะเหล็กเป็นคำบรรยายเสียงเฉพาะเจาะจงสำหรับ Stable Audio โดยเน้นความหนักแน่นของโลหะและระยะใกล้ของไมโครโฟน..."}
```
SFX, `spooky forest at night with a little witch`:
```json
{"prompt":"creaky wooden broomstick flight, wind howling through dense pine branches, eerie night ambiance, distant owl hoot, short swoosh, high quality, foley, sound effect, 44.1kHz stereo",
 "negative":"music, melody, speech, vocals, distortion, clipping, low quality, noise, hum, city sounds, daytime birds",
 "variations":["soft leather cloak rustling, dry leaves crunching underfoot, intimate close-mic perspective, subtle footstep impacts, rhythmic walking, ...",
  "dark magical spell casting, low frequency hum building to sharp crackle, small enclosed forest clearing reverb, quick magical burst, ...",
  "heavy wooden staff tapping on mossy roots, hollow thud with damp wood resonance, close-mic detail, single impact with long decay, ..."]}
```
Music, `หุ่นยนต์ตัวเล็กขี้เล่น พูดว่า "ไปผจญภัยกันเถอะ!"` ("small playful robot says 'let's go adventure!'"):
```json
{"prompt":"chiptune, 8-bit, video game soundtrack, playful, energetic, bright, major key, 120 BPM, square wave lead, bouncy bass, retro, pixel art aesthetic, RPG, loopable, cheerful, lighthearted",
 "negative":"dark, minor key, slow, heavy, realistic, orchestral, vocal, speech, noise, distortion",
 "variations":["chiptune, 8-bit, ... 130 BPM, square wave lead, fast tempo, ...","synthwave, 80s retro, ... 110 BPM, analog synth lead, warm bass, ...","lo-fi chiptune, 16-bit, ... 100 BPM, soft square wave, gentle bass, ..."],
 "lyrics":"[inst]","notes":"สร้างเพลงแนวชิปทูน (Chiptune) ที่มีความสนุกและสดใส ..."}
```
Music (v2 templates), `บอสมังกรไฟในถ้ำลาวา` ("fire dragon boss in a lava cave"):
```json
{"prompt":"epic orchestral, dark fantasy, boss battle music, heavy brass, pounding timpani, aggressive choir, minor key, 140 bpm, cinematic, intense, dramatic, low end, powerful, fantasy action, loopable, wide stereo",
 "negative":"light melody, calm, ambient, electronic beats, pop, jazz, acoustic guitar, soft vocals",
 "lyrics":"[inst]\n[verse]\n[pre-chorus]\n[chorus]\n[bridge]\n[outro]"}
```
Voice, robot idea (quoted Thai line kept verbatim):
```json
{"prompt":"A playful, small robot with a bright, synthetic, and energetic voice. High pitch, fast pace, and cheerful emotion. Speaks Thai with a central Thai accent. Close studio mic, dry recording. Character archetype: whimsical sidekick in a stylized fantasy action-RPG.",
 "line":"ไปผจญภัยกันเถอะ!",
 "variations":["A curious, youthful robot voice. Bright timbre, slightly breathy, medium pace. ...","A mischievous, agile robot voice. Crisp, metallic timbre, fast pace, high energy. ...","A friendly, reliable robot voice. Warm synthetic tone, moderate pace, confident emotion. ..."]}
```
3D-ready image, `บอสมังกรไฟในถ้ำลาวา`:
```json
{"prompt":"A majestic fire dragon boss, stylized fantasy game asset, full body visible, centered in a three-quarter front view, standing on a flat base, plain light-grey seamless background, soft even studio lighting, no cast shadows, clear silhouette, clean readable forms, PBR-friendly materials, detailed scales, glowing orange and red accents, T-pose posture, high quality concept art",
 "negative":"multiple subjects, cropped, cut off, busy background, scenery, text, watermark, logo, motion blur, depth of field, harsh shadows, extreme perspective, fisheye, lowres, deformed",
 "variations":["A massive fire dragon boss, ... dark obsidian scales with molten cracks, fierce expression, ...","An agile fire dragon boss, ... vibrant crimson scales with golden horns, ...","A regal fire dragon boss, ... bronze and amber scales with intricate armor plating, ..."]}
```
3D-ready image, `cozy village market in the morning` (the scene was correctly reduced to one prop):
```json
{"prompt":"A single rustic wooden market stall with colorful awnings, filled with stacked crates of fresh produce and hanging herbs, isolated on a plain white seamless background, three-quarter front view, centered composition, soft even studio lighting, high-poly stylized game asset, hand-painted textures, clean silhouette, PBR-ready materials, ..."}
```
Vision describe (Buddha statue photo):
```json
{"prompt":"A hyper-realistic 3D game asset of a seated Buddha statue in the traditional Thai Sukhothai style, rendered in gleaming polished gold. The figure sits in a meditative lotus position with a serene expression, draped in a golden sash across the chest. The head is adorned with a tall, intricate golden flame-shaped ushnisha halo. The background is a deep matte black wall featuring a repeating pattern of embossed golden floral motifs and two celestial beings (devas) in flight. ...",
 "notes":"ภาพจำลองพระพุทธรูปทองคำปางมารวิชัยในสไตล์ไทยประยุกต์ สำหรับใช้ในเกมแฟนตาซี"}
```

### Pasted outputs: gemma-4-E2B (fallback), same ideas
```json
// sfx, ดาบฟันโดนเกราะเหล็ก
{"prompt":"steel sword striking thick iron plate armor, sharp metallic clang, close-mic, immediate impact, high quality, foley, sound effect, 44.1kHz stereo",
 "notes":"เสียงดาบฟันโดนเกราะเหล็ก เน้นความรู้สึกหนักแน่นและโลหะปะทะกัน"}
// music, spooky forest (good tags; lyrics written, markers inline on some runs)
{"prompt":"dark fantasy, spooky atmosphere, slow tempo, minor key, ethereal choir, haunting flute melody, deep synth pads, low strings, ambient sound design, ..., loopable, nocturnal forest setting",
 "lyrics":"[intro] [inst]\n[verse] The moon casts shadows long and thin\nThrough the ancient trees where darkness spins\n[chorus] Little witch walks beneath the night ..."}
// image3d, witch: variations start with a stray "Prompt: " label, notes "little witch" (terse, English)
```

## Recommended server configuration

```bash
LLM_DIR=$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF   # or the app-owned copy
HF_HUB_OFFLINE=1 CUDA_CACHE_MAXSIZE=4294967296 \
<llama.cpp>/build/bin/llama-server \
  -m  $LLM_DIR/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --mmproj $LLM_DIR/mmproj-Qwen3.6-35B-A3B-BF16.gguf \
  --host 127.0.0.1 --port 7880 \
  -ngl 999 -fa on --jinja \
  -c 16384 -np 2 \
  --reasoning off \
  --sleep-idle-seconds 300 \
  --no-webui --alias prompt-llm
```
* `-c 16384 -np 2` gives 8k tokens per slot. A request is about 500 prompt + about 500 generated
  tokens (about 1.3k with an image), so 2 slots let two fields improve at once (91 tok/s
  aggregate). The KV cache is only 320 MB. Do not raise ctx "just because".
* `--reasoning off` (plus `chat_template_kwargs.enable_thinking=false` per request) is required.
  Thinking adds 5-20 s and the grammar would otherwise have to allow a `<think>` block.
* Fallback config: same flags with `-m …/gemma-4-E2B-it-UD-Q4_K_XL.gguf`, no `--mmproj`,
  `--port 7881`.
* Ports: 7880 for the LLM, 7881 for the fallback (only one runs at a time). 7882-7889 are left
  for other studio workers. Avoid ports other services on the machine already use (ComfyUI's default is 8188).

## Lifecycle: lazy start, sleep or stop on idle

1. **Lazy start**, owned by the studio backend (a supervisor class, not systemd). On the first
   "Improve prompt" click, check `GET :7880/health`. If nothing is listening, spawn llama-server
   with the flags above (own process group, log to `logs/prompt_llm.log`). Poll `/health` every
   0.5 s, with a 120 s timeout. Show "loading prompt model… (~5-20 s, first ever ~50 s)" in the UI.
2. **Two-stage idle release:**
   * `--sleep-idle-seconds 300` makes llama-server free its weights after 5 min idle (measured
     22 GB to 0.2 GB). The process keeps the port, and the next request wakes it transparently
     (about 10 s).
   * The supervisor also kills the process after 30 min idle, which removes the last 0.7 GB RSS
     and the CUDA context.
3. **Yield to heavy jobs.** Before the app starts a big image, 3D or audio job that needs more than
   about 60 GB (Qwen-Image, Hunyuan3D, Stable Audio medium…), the job scheduler calls
   `llm.release()` = SIGTERM (graceful, under 2 s). Prompt improvement and heavy generation are
   never concurrent in the UI flow anyway.
4. **Fallback chain:**
   * llama-server Qwen MoE.
   * If that fails to start, or free memory is under 25 GB (check `MemAvailable` before spawning),
     use llama-server gemma-4-E2B (2.3 GB).
   * If both fail, use ollama `gemma3:4b` at `127.0.0.1:11434`, with `format=<schema>` and
     `keep_alive: 0`.
   * If everything fails, disable the button with a tooltip.
5. One process at a time. Use a file lock (`/tmp/studio-prompt-llm.lock`) so two app instances do
   not both spawn.
6. **Alternative:** llama-server router mode (`--models-preset presets.ini --models-max 1`) can
   switch default and fallback by the `model` field in one process. It is newer and less proven
   than the plain setup, so the plain single-model process plus the supervisor is recommended.

## Prompt templates

`docs/research/prompt_templates.md` holds the shared preamble and these targets: `sfx`, `music`
(tags + `[verse]/[chorus]` lyrics), `voice` (TTS voice design + line), `image3d` (3D-ready concept
for Hunyuan3D), `edit` (FLUX Kontext instruction), `translate`, `describe` (vision). The benchmark
harness parses that file directly. The `edit` / `translate` templates were written but not
benchmarked; the edit template should be tried against FLUX Kontext once the image tab exists.

## Reproduce

Scratch harness (not part of the app; not included in this repository):
* `serve.sh LABEL MODEL [args]`: start on :7881 and print load time and memory
* `bench.py --out X.jsonl [--mode schema|json_object|plain] [--image f.jpg] [--targets ...]`
* `par.py N`: concurrency throughput
* `sleeptest.sh`
* `stop.sh`

Raw results were written as `runs/*.jsonl` in the scratch directory and are not included.

## Open items

* Rebuild llama.cpp for sm_121 (removes the about 24 s JIT on the first request after a cold
  kernel cache). Not required.
* Optionally benchmark `gemma-4-26B-A4B-it-QAT` Q4_0 + mmproj (14 GB, MoE, strong Thai) as an
  alternative default.
* Optionally convert `google/gemma-4-E4B-it` (on disk, HF format) to GGUF locally
  (`convert_hf_to_gguf.py`, no download) for a mid-size fallback with mmproj.
