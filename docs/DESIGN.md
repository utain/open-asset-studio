# Open Asset Studio: design

An all-in-one local studio on the DGX Spark for teams that need **images, 3D models, sound and video** for games,
video, podcasts, social content and film. Open-weight models only, offline-first, one machine. This document is the shared spec for
everyone building it. Research on what actually runs here lives in `docs/research/`.

## 1. Goals and non-goals

Goals
- One web app with studios (Image, 3D, Sound, plus Video where installed) and a shared **asset library**, so an idea
  can go concept image -> cleaned RGBA -> textured 3D model -> sounds and voice for it, or thumbnail ->
  intro sting -> voice-over -> loudness-matched export, without downloading/uploading files between tools.
- **Use-case profiles** (game, video, podcast, social, film, general): one setting that reorders the style presets,
  sets the default aspect ratios and picks the default export (loudness, sample rate, image size). Game is a
  first-class profile and keeps its RGBA / seamless-loop / GLB behaviour; the others add presets, never remove any.
- **Export presets**: audio and images are converted on the way out into derived files (the library is never
  modified); see section 4.7.
- A **commercial-safe** switch: every model carries a licence badge, and one setting hides and refuses
  non-commercial models everywhere (section 4.8).
- **Switch models per task**, filtered by what each model can do (features) and whether it runs
  here (compatibility). Changing the model swaps the parameter form; shared fields stay.
- A built-in **prompt improver** (local LLM) next to every prompt: Thai or English in,
  model-specific prompt out, with variations.
- Production features for every medium: batches/takes, A/B compare, favourites/ratings, lineage, export in
  engine-friendly formats, sheets of many assets (voice line sheets, SFX event lists).

Non-goals (for now)
- Multi-user auth, training/fine-tuning, video. Cloud APIs are optional and off by default (Providers page,
  docs/CONTRACTS.md section 8): local models stay the default and never need a key.
- Exhaustive unit tests. Test only critical pieces: path safety of file routes, registry
  compatibility checks, the job queue / GPU lock / memory eviction, worker IPC framing.

## 2. Users and core flows

Primary user: a member of a small production team (game artist or sound designer, video editor, podcaster, social
content creator, animator) using a browser on the LAN. Typical flows, which the UI must make short:

1. **Concept to 3D prop**: prompt (Thai ok) -> improve -> 4 variations with a fast model -> pick ->
   refine with an edit model -> transparent cut-out + recenter -> 3D shape -> texture -> GLB export.
2. **Character kit**: concept image -> 3D model; voice designed from a text description of the
   character -> a sheet of barks/lines in that voice; footsteps / attack SFX.
3. **SFX pass**: paste a list of events (one per line or CSV) -> N takes each -> audition with
   waveform -> rate / pick -> normalize + trim + loop -> export a folder for Godot/Unity.
4. **Music**: describe mood/genre (+ optional lyrics) -> takes -> extend/loop -> export.
5. **Iterate**: open any asset, see exactly how it was made (model + params + parents), tweak one
   knob, regenerate, compare side by side.

## 3. Information architecture

Left sidebar (collapses to a bottom bar on phones):

| Section | Contents |
|---|---|
| Home | quick-start cards for the flows above, recent assets, queue, memory |
| Image | Generate / Edit / Tools (upscale, remove bg, extract subject, crop/recenter, tileable if a model supports it) |
| 3D | Image -> Shape / Texture / Retexture / Export (Hunyuan3D-2.1, existing engine) |
| Sound | SFX / Ambience & loops / Music / Voice (TTS, voice design, voice clone) / Sheets (batch) |
| Library | all assets of the current project: grid with filters (type, model, tag, rating, date), search, detail drawer |
| Pipelines | guided multi-step flows (Concept -> 3D, Character kit) built on the same jobs |
| Models | registry browser: capability matrix, status (ready / weights missing / env missing / broken), license, memory, speed; set default per task |
| Queue & System | jobs (running, queued, done, failed) with cancel/retry, loaded models, memory, logs |

Top bar: project switcher (projects are folders), global search, queue indicator, memory meter,
light/dark toggle.

### Studio page layout (same for every studio)
- **Left panel**: task tabs, model picker (with feature badges), prompt box with the ✨ Improve
  button, generated parameter form (basic fields first, "Advanced" collapsed), presets
  (Fast / Balanced / Quality + user-saved), Generate button with time estimate.
- **Center**: results of the current batch (grid of image/mesh thumbnails or audio cards with
  waveform + play), compare mode (2-up / A-B toggle).
- **Right drawer**: selected asset details: preview (image zoom with checkerboard for alpha,
  model-viewer for GLB, waveform player for audio), metadata, lineage, actions ("Use in ..."
  send-to buttons, regenerate with these params, variations, export, delete, rate, tag).

Send-to actions are the glue: image -> Edit / 3D / Extract subject / Upscale; mesh -> Retexture /
Export; audio -> Variations / Extend / Loop / Normalize / Use as voice reference.

## 4. Architecture

```
browser (SPA: vanilla JS modules, vendored libs, no CDN at runtime)
   |  REST + Server-Sent Events
FastAPI core (app)  -- one process, port 7860
   |-- registry   models.yaml -> capability + compatibility status
   |-- jobs       queue, one GPU job at a time, progress via SSE, cancel
   |-- memory     loaded-model table, MemAvailable checks, LRU eviction
   |-- assets     projects/<name>/assets/<id>/ (files + meta.json), thumbnails, waveforms
   |-- improver   local LLM (llama-server) started lazily, idle unload
   `-- workers    one subprocess per backend venv, JSON-lines over stdin/stdout
         |-- hy3d   (envs/hy3d)    Hunyuan3D shape/texture, Real-ESRGAN, rembg
         |-- qwen   (envs/qwen)    Qwen-Image-2.1 (+ other diffusers image models)
         |-- audio-* (per research)        Stable Audio 3, MOSS, TTS models ...
         `-- llm    llama-server on 127.0.0.1
```

- Frontend: FastAPI serves a single-page app from `app/web/` (ES modules, one file per studio:
  `web/studios/image.js`, `3d.js`, `sound.js`, shared `web/core/*.js`). Libraries are vendored in
  `app/web/vendor/` (model-viewer, wavesurfer.js, ...), pinned versions, so it works offline.
  Gradio is NOT used for the studio (its tab/layout quirks cost us before).
- Reuse, do not fork, the proven engine code in `engines/hy3d/app` (engine.py, postprocess.py, imgtools.py)
  and `workers/qwen_worker.py` by importing or moving it behind the worker protocol.
- Envs: every venv is referenced by path in `models.yaml` (`backends.<b>.python`) and lives in `envs/`. Every env has a lock file in
  `env-lock/`.

### 4.1 Model registry (`models.yaml`)
Each entry:
```yaml
- id: qwen-image-2.1
  name: Qwen-Image 2.1
  modality: image            # image | mesh | audio | text
  tasks: [t2i, edit]         # t2i edit multi_ref inpaint rgba extract upscale | shape texture |
                             # sfx ambience music song tts voice_design voice_clone | improve
  features: {rgba: true, max_refs: 10, mask: annotate, text_render: true, languages: [en, zh]}
  backend: qwen              # worker name -> venv + module
  weights: [{hf: Qwen/Qwen-Image-2.1}]
  license: {name: Qwen Research, commercial: false}
  memory_gb: 31              # resident estimate; peak per size in params hints
  speed: "1.5 s/step @1024"
  prompt_style: qwen_image   # key into the improver templates
  params:                    # drives the generated form; same names across models when meaning matches
    - {name: steps, type: int, min: 4, max: 60, default: 28, group: basic}
    - {name: seed, type: seed}
    - {name: size, type: choice, choices: [768, 1024, 1536, 2048], default: 1024}
    - {name: true_cfg, type: float, min: 1, max: 10, default: 4, group: advanced}
  presets: {fast: {steps: 16, size: 768}, balanced: {}, quality: {steps: 40, size: 1536}}
```
Status is computed at startup and on demand: weights present in HF cache, venv exists, worker
import probe ok -> `ready`; otherwise `missing_weights` / `missing_env` / `broken (reason)`.
The UI shows not-ready models greyed out with the reason and the fix command.
Defaults per task live in `settings.json` (editable on the Models page).

### 4.2 Jobs, GPU and memory
- One global queue; one GPU job runs at a time (CPU-only tools may run alongside).
- A job = {id, task, model, params, inputs (asset ids or uploads), project, status, progress,
  log, outputs (asset ids), timings}. Persisted to `data/jobs.jsonl` for the Queue page.
- Before a job starts, the memory manager ensures `MemAvailable >= model.memory_gb + margin`
  (unified memory: use /proc/meminfo, not only CUDA numbers), unloading other loaded models LRU
  first (the improver LLM is the cheapest to evict). Workers unload themselves after an idle
  timeout. Never kill ComfyUI or other users' processes.
- Cancel is cooperative (step callbacks); report clearly when a stage cannot be interrupted.

### 4.3 Worker protocol
JSON lines over stdin/stdout. Requests: `{"id", "op": "load"|"run"|"unload"|"cancel"|"probe",
"model", "params", "inputs", "out_dir"}`. Events: `progress {id, step, total, stage}`, `log`,
`result {id, files:[...], meta}`, `error {id, message, kind: oom|input|internal}`. Workers die
with the parent (PR_SET_PDEATHSIG + parent watch), never listen on the network. The existing
`workers/qwen_worker.py` is the reference implementation.

### 4.4 Assets and projects
`data/projects/<project>/assets/<asset_id>/` holds the files plus `meta.json`:
type, files, preview (thumb.webp / waveform.json), prompt, improved_prompt, model, params, seed,
parents (asset ids), job id, created, tags, rating, notes. The library reads only meta files
(fast). File routes are served only from inside `data/projects` with path-traversal protection;
uploads are validated; APIs never accept server-side paths.

### 4.5 Prompt improver
- Button next to every prompt; opens a small panel: improved prompt, 3-4 variations, notes,
  "translated from Thai" flag; insert / replace / keep original. Optional style chips per studio
  (e.g. image: 3D-ready, stylized, low-poly; sfx: impact, UI, foley; music: loopable, battle).
- Uses per-model templates (`prompt_style`) so it writes the right format (tags, lyric structure,
  instruction style, RGBA prompt format). JSON-constrained output.
- Runs on a local llama-server, started lazily, unloaded when idle or when memory is needed.

### 4.6 Sound studio specifics
- Audio cards: waveform (wavesurfer), play/stop, loop toggle, duration, loudness (LUFS), peak,
  quick rating; batch view per prompt with N takes.
- Post tools: trim silence, fade in/out, normalize to target LUFS, make seamless loop
  (crossfade), convert wav/ogg/mp3, mono/stereo, sample rate.
- QA: CLAP prompt-match score for SFX/music ranking, Whisper transcript + match % for voice lines
  (flags mismatches); both optional, run on demand or after batches.
- Voice: voice library (designed or cloned voices are assets of type `voice` with a reference
  clip), line sheet editor (character, line, emotion/direction, language) with CSV paste/import
  and per-line takes.
- Export: folder per category with consistent naming (`<event>_<take>.ogg`) + a manifest JSON.

### 4.7 Profiles and export presets
- `settings.json` `"profile"` (`game | video | podcast | social | film | general`, default `general`) is read by
  the UI through `GET /api/profiles` and the shared store (`getProfile()`, bus event `profile`). A profile owns:
  chip order (`profiles: [...]` tags on prompt chips; untagged chips belong to every profile), the default aspect
  ratio of image / video models (`aspect` param, applied on first use and on a profile change), the context line sent
  to the prompt improver and the default export preset. Presets are tagged, never duplicated.
- Export presets (`app/profiles.py`) are named bundles of audio options (format, sample rate, bit depth, target LUFS
  and true peak via two-pass ffmpeg `loudnorm`) and image options (format, aspect crop / pad, size). Plain "Export zip"
  applies the default preset; "Export / convert..." opens the dialog; the `asset.export` tool saves converted
  copies as new assets. Rules: originals are never modified, conversions run on CPU (ffmpeg, Pillow), a silent file
  is converted without a gain change, the report (input / output LUFS, true peak) goes into the manifest.
- Game and general default to `original`: exports are byte-identical to the stored files, as before.

### 4.8 Commercial-use licence filter
- Registry `license: {name, commercial, note?, verify?}` gives three states: *commercial*, *check* (`verify: true`),
  *non-commercial*. Badges on the Models page, in every model picker and on assets.
- `settings.license.commercial_only` (top-bar shield, Models page, Queue & System): pickers hide / disable
  non-commercial models, the assistant never lists them, and `POST /api/jobs` refuses them with HTTP 403 and
  `kind: "license"`. `license.strict` also refuses *check* models. The filter never deletes or hides existing assets.

## 5. Quality bar
- Every control reaches the model call (no dead knobs); every result is reproducible from its
  meta.json.
- Clear progress and time estimates; errors explain the cause and the fix.
- Works at 1440 px and on a phone (390 px): no horizontal page scroll, readable in light and dark.
- Offline: no CDN or Hub requests at runtime (HF_HUB_OFFLINE=1 by default).
- Licenses are visible on the Models page and on each asset (many models are non-commercial).
