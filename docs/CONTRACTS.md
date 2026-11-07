# Open Asset Studio: core contracts

This is the stable interface of the studio core. The Image + 3D studio and the Sound studio are built on it
**in parallel, without editing core files**. Every extension point is a drop-in file:

| You add | Where | Picked up by |
|---|---|---|
| models + backends | `models.d/<studio>.yaml` (or entries in `models.yaml`) | registry, at startup and on *Re-check* |
| a backend worker | `workers/<backend>_worker.py` | `WorkerPool`, on the first job of that backend |
| job handlers, CPU tools, extra API routes | `app/plugins/<studio>.py` | `Core.load_plugins()` at startup |
| prompt-improver styles | `prompts/<studio>.yaml` | improver, at startup and on *Re-check* |
| a studio page | `web/studios/<id>.js` (replace the stub) | the shell, which imports every file in `web/studios/` |

If you need a core change, prefer a small, backwards-compatible hook (a new optional field or callback) and note
it at the end of this file under "Changelog".

Core files (owned by the core): `app/{config,registry,workers,procs,memory,jobs,events,assets,media,improver,textgen,core,main}.py`,
`app/plugins/core_tools.py`, `workers/studio_worker.py`, `workers/echo_worker.py`, `web/index.html`, `web/main.js`,
`web/app.css`, `web/core/**`, `web/studios/{home,library,models,system,selftest}.js`, `prompts/core.yaml`.

Run the core: `./run_studio.sh` (port 7860) or `STUDIO_PORT=7861 ./run_studio.sh`. Tests: `envs/core/bin/python -m pytest`.

---

## 1. Model registry (`models.yaml`, `models.d/*.yaml`)

Three top-level keys. Files are merged in order (`models.yaml`, then `models.d/*.yaml` alphabetically); an entry
with an existing id replaces it.

```yaml
tasks:                    # task id -> label / modality / owning studio (used by pickers, defaults, "Regenerate")
  sfx: {label: Sound effect, modality: audio, studio: sound}

backends:                 # one worker process per backend (= per venv)
  audio:
    label: envs/audio (SA3, MOSS-SFX, ...)    # shown on the Models page
    python: envs/audio/bin/python      # relative to the repo root, or ~/..., or absolute
    script: workers/audio_worker.py    # convention: workers/<backend>_worker.py
    device: gpu                        # gpu -> jobs go through the single GPU queue; cpu -> run alongside
    lock: audio                        # env-lock/<lock>.txt (shown in the fix hint)
    max_loaded: 2                      # models resident at once in this worker (LRU unload inside the worker)
    model_idle_s: 600                  # optional override of settings.workers.model_idle_s
    kill_on_cancel: true               # kill the worker if a cancel is ignored for workers.cancel_grace_s (default true)
    env: {PYTHONPATH: "{studio}/envs/vllm_omni_overlay"}   # extra environment ({studio} = studio root)
  llm: {kind: service, binary: $LLAMA_SERVER}   # not a worker: status = binary + weights

models:
  - id: sa3-small-sfx                  # stable id: stored in every asset's meta.json
    name: Stable Audio 3 Small SFX
    modality: audio                    # image | mesh | audio | text | other
    tasks: [sfx]                       # task ids (see tasks:)
    default_for: [sfx]                 # registry default (the Models page ★ overrides it in settings.json)
    order: 10                          # sort order inside pickers
    tags: [fast, sfx]                  # 'fast' / 'fastest' / 'quality' become picker badges
    summary: "One-line description for pickers."       # quote YAML strings that contain ': '
    features: {max_seconds: 120, negative: true, languages: [en]}   # free-form; known keys below
    backend: audio
    device: cpu                        # optional per-model override of the backend device
    weights:                           # all must exist for status ready
      - {hf: stabilityai/stable-audio-3-small-sfx, files: [model.safetensors]}   # files = globs inside a snapshot
      - {path: ~/.u2net/u2net.onnx, fix: "how to get it"}
      - {lmstudio: lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf}
    license: {name: Stability AI Community, commercial: true, note: "conditions", verify: false}
    memory_gb: 4                       # free memory (MemAvailable) the job needs before it starts (peak, not resident)
    speed: "warm ~1-2 s per clip"      # human text
    est: {load_s: 5, run_s: 3}         # seconds, for the Generate button's estimate (run_s is per take)
    prompt_style: sfx                  # improver template id, or a map per task: {sfx: sfx, music: sa3_music}
    handler: worker                    # optional: job handler name (default "worker", see 3.3)
    kill_on_cancel: true               # optional, informational for UIs (the backend flag is what counts)
    test: false                        # true = self-test model, hidden from pickers unless ui.show_test_models
    params: [...]                      # the generated form, see below
    presets: {fast: {...}, balanced: {}, quality: {...}}   # partial param dicts; {} = the defaults
```

**Known `features` keys** (badges + studio logic): `rgba`, `max_refs`, `mask`, `text_render`, `negative`
(`true` | `false` | `cfg` = only with CFG > 1), `languages`, `thai` (`true` | `false` | `weak` | `untested`),
`needs_ref` (`true` | `clone` = only for voice_clone), `lyrics`, `instruct`, `tags`, `vision`, `pbr`, `inpaint`,
`native_size`, `remote`, `max_seconds`, `sample_rate`, `channels`, `speakers`, `voices`, `export`.

**Input slots** (`input_slots`, top level of a model entry; `features.input_slots` is accepted as an alias). A worker
that reads extra input roles declares them so the Sound studio renders one `AssetSlot` per slot instead of the
hard-coded reference / source slots:

```yaml
    input_slots:
      - {role: ref,   label: Reference voice, types: [audio, voice], required: true, hint: "5-15 s of clean speech"}
      - {role: emo,   label: Emotion clip,    types: [audio, voice], required: false}
      - {role: video, label: Video,           types: [video],        required: true, tasks: [v2a]}
```

| key | meaning |
|---|---|
| `role` | the `inputs[].role` sent to the worker (`[a-z][a-z0-9_]{0,31}`); must be the role the worker reads via `Job.input(role=...)` |
| `label`, `hint` | UI text |
| `types` | asset types the slot accepts: `audio`, `voice`, `video`, plus `image`, `mesh` for the 3D studio tasks (default `[audio, voice]`) |
| `required` | Generate is blocked with a toast while the slot is empty |
| `tasks` | optional: only for these tasks (models with several tasks, e.g. Vevo: `voice_convert` / `voice_clone` / `tts`) |
| `replaces_prompt` | optional: a filled slot makes the prompt text optional (DiffRhythm / Vevo 1.5 style prompt) |

The registry validates every entry (`Registry._normalize_slots`); a bad one is dropped and reported in `errors`.
`GET /api/models` exposes the list as `input_slots`. If a model declares no slot for the current task, the studio
falls back to its old behaviour (a reference slot for `voice_clone`, a source slot for the extend tasks), so
undeclared models keep working. Roles are free-form on the wire (`role` is any string), so the slot list is the
only place a UI learns them.

**Params** (one dict per field; the same `name` across models means the same thing, so values carry over when
the user switches model):

| key | meaning |
|---|---|
| `name` | parameter name sent to the worker in `params` |
| `type` | `int`, `float`, `seed`, `choice`, `bool`, `text`, `textarea`, `lyrics`, `asset`, `assets` (asset types are placeholders: studios render inputs with `AssetSlot`) |
| `label`, `help`, `unit`, `placeholder` | UI text |
| `default`, `min`, `max`, `step`, `choices` (values or `{value, label}`), `max_length` | validation; the core clips numbers and rejects bad choices (`registry.validate_params`) |
| `group` | `basic` (default) or `advanced` (collapsed) |
| `tasks` | only show/send this field for these tasks |
| `show_if` | `{other_param: value}` |

Shared names in use: `prompt`, `negative` (from the prompt box, not listed in params), `aspect` + `size` (long side
px; worker helper `size_from_aspect`), `steps`, `guidance`, `cfg_scale`, `true_cfg`, `seed` (`-1` = random),
`batch` (images / takes per job), `seconds`, `text` (the line to speak), `instruct`, `instruction`, `ref_text`,
`language`, `lyrics`, `speaker`, `voice`, `temperature`, `top_p`, `top_k`.

**Status** (`GET /api/models` -> `status: {status, code, reason, fix, checked}`), computed at startup, on
`POST /api/models/refresh`, and per backend on `POST /api/models/probe {backend}`:

| status | code | when |
|---|---|---|
| `ready` | `ready` | venv + weights + worker script present (and the last import probe, if any, passed) |
| `missing_env` | `missing_env` | `backends.<b>.python` (or `binary`) does not exist; `fix` = how to recreate |
| `missing_weights` | `missing_weights` | a `weights` entry is missing (no snapshot, glob not found, dangling blob link = unfinished download); `fix` = exact `hf download ...` command |
| `broken` | `no_worker` | `workers/<backend>_worker.py` does not exist yet (UI: "Coming soon") |
| `broken` | `unknown_backend`, `probe_failed` | backend not declared; `--probe` said the model cannot run |

The probe result is cached in `data/probe_cache.json` and is invalidated when the venv's `pyvenv.cfg` or the
worker script changes. Defaults: `GET /api/models` -> `defaults: {task: model}`; set with
`PUT /api/settings/defaults {task, model|null}` (stored in `settings.json`).

API: `GET /api/models?task=&modality=&include_test=` -> `{models, tasks, defaults, backends, errors}` ·
`GET /api/models/{id}` (+ `user_presets`) · `POST /api/models/refresh` · `POST /api/models/probe {backend}` ·
`GET|PUT /api/presets/{model}` `{name, params}` · `DELETE /api/presets/{model}/{name}`.

---

## 2. Worker protocol and the Backend base class

One subprocess per backend venv: `<python> -u workers/<backend>_worker.py --model-idle S --exit-idle S`, started
lazily by the first job, cwd = repo root, `PYTHONPATH` starts with `workers` (so `import studio_worker`
works in every venv), `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`, `PYTHONNOUSERSITE=1`. stderr goes to
`logs/worker-<backend>.log`. The worker never listens on a socket. It exits on `shutdown`, after `--exit-idle`
seconds with no model loaded, on stdin EOF, when its parent pid changes, and via `PR_SET_PDEATHSIG`.

### 2.1 Messages (JSON lines; protocol 1)

core -> worker (every request has an `id`; for `run` it is the job id):

```json
{"op":"run","id":"j2026...","model":{...registry entry...},"task":"sfx","params":{...},
 "inputs":[{"asset_id":"20261105-...","role":"ref","type":"audio","name":"door","path":"/abs/.../audio.wav","meta":{...}}],
 "out_dir":"/abs/<repo>/data/jobs/<job>/out"}
{"op":"load","id":"x","model":{...}}         {"op":"unload","id":"x","model":"<id>"|null}
{"op":"cancel","id":"<job id>"}               {"op":"status","id":"x"}     {"op":"shutdown"}
```

worker -> core:

| event | fields | notes |
|---|---|---|
| `hello` | `pid, backend, protocol` | once at start |
| `accepted` | `id` | the job left the worker's queue |
| `progress` | `id, step, total, stage, msg, interruptible` | `stage: loading` is shown as indeterminate; `interruptible:false` tells the UI that cancel waits for the stage |
| `log` | `id, msg` | appended to the job log (visible in the UI) |
| `loaded` / `unloaded` | `model, seconds, gpu_gb` / `model, reason` (`idle`, `request`, `make-room`, `shutdown`) | keep the core's loaded-model table right |
| `file` | `id, file` | one finished output (lets the core keep partial results on cancel) |
| `result` | `id, files:[{path, type, ...meta}], meta, timings` | terminal |
| `cancelled` | `id, files` | terminal; files finished before the cancel are kept as assets |
| `error` | `id, kind, message` | terminal; `kind`: `oom` \| `input` \| `internal` |
| `status` | `id, loaded, busy, gpu_gb, pid` | |
| `bye` | `reason` | `idle` \| `shutdown` |

Only the protocol is on fd 1: the framework re-points fd 1 to stderr at import, so library prints never corrupt it.

### 2.2 Writing a backend

```python
# workers/audio_worker.py   (runs in envs/audio)
import random
from studio_worker import Backend, InputError, serve, size_from_aspect

class AudioBackend(Backend):
    name = "audio"
    max_loaded = 2                    # resident models in this process (LRU); overridden by backends.audio.max_loaded

    def probe(self, model):           # used by --probe ("Deep check" on the Models page). Cheap: imports + files.
        import stable_audio_tools  # noqa: F401
        return True, ""

    def load(self, model, ctx):       # model = the registry entry (dict). Return the handle.
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if model["id"] == "sa3-small-sfx":
            return load_sa3("stabilityai/stable-audio-3-small-sfx")   # load straight to the GPU (device_map="cuda")
        raise InputError(f"unknown model {model['id']}")

    def unload(self, model_id):       # framework then runs gc.collect() + torch.cuda.empty_cache()
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        pipe = self.handle(job.model_id)
        p = job.params                                    # validated against the registry by the core
        seed0 = p["seed"] if p.get("seed", -1) >= 0 else random.randrange(2**31)
        for i in range(int(p.get("batch", 1))):
            ctx.check()                                   # raises Cancelled when the user pressed cancel
            wav = pipe(p["prompt"], seconds=p["seconds"], seed=seed0 + i,
                       callback=lambda s, n: (ctx.check(), ctx.progress(s, n, "sampling", f"take {i+1}: {s}/{n}")))
            f = job.out_dir / f"take_{i+1}.wav"
            write_wav(f, wav)
            ctx.output(f, "audio", seed=seed0 + i, index=i, name=f"{p['prompt'][:40]} #{i+1}")
        return {"meta": {"sampler": "pingpong"}}

if __name__ == "__main__":
    serve(AudioBackend())
```

Helpers: `ctx.progress(step, total, stage, msg, interruptible=True)`, `ctx.check()` / `ctx.cancelled`,
`ctx.log(msg)`, `ctx.output(path, type, **meta)` (types `image | audio | voice | mesh | text | other`;
known meta keys: `seed`, `index`, `name`, `final_prompt` (what was really sent, e.g. with the RGBA trigger),
`attach=[extra file names in out_dir]` for multi-file assets; anything else lands in the asset's `worker` dict),
`ctx.step_callback(total, stage, label)` (a diffusers `callback_on_step_end` that reports + cancels),
`ctx.gpu_gb()`, `job.input(role=, type=)`, `size_from_aspect(aspect, long_side, multiple=16)`.
Raise `InputError("message for the user")` for bad input; any exception text containing "out of memory" is
reported as `oom`. **Report the seed you actually used** (`seed=-1` means random) so every asset is reproducible.

Reuse the proven loaders: the fast direct-to-GPU recipe and cancel pattern of `workers/qwen_worker.py`,
`docs/research/image_snippets/loaders.py`, the audio snippets in `docs/research/audio_snippets/`. The bundled
Hunyuan3D engine code lives in `engines/hy3d/app`; import it by adding it to `sys.path` inside the worker rather than copying it.

Try a worker by hand: `echo '{"op":"run","id":"t","model":{"id":"test-tone","name":"T"},"task":"sfx","params":{"steps":3},"out_dir":"/tmp/x"}' | PYTHONPATH=workers envs/core/bin/python workers/echo_worker.py`.
Reference: `workers/echo_worker.py` (models `test-tone`, `test-swatch`), tests in `tests/test_worker_ipc.py`.

### 2.3 Memory

Before a job runs, the core calls `memory.ensure(model.memory_gb, key="<backend>:<model>")`: if
`MemAvailable < memory_gb + memory.margin_gb (6)`, it unloads OUR resident models, the prompt LLM first, then
least-recently-used worker models (busy ones and the job's own model never). Jobs with `memory_gb >= 40` always
stop the prompt LLM. If that is not enough, the job fails with kind `memory` and a message naming the other
tenants (for example ComfyUI or the Hunyuan3D engine). Other users' processes are never touched. Workers unload idle models themselves
(`model_idle_s`) and exit when empty (`exit_idle_s`). `GET /api/system` shows the table; `POST /api/system/unload
{backend, model} | {llm: true} | {all: true}` unloads.

---

## 3. Jobs

### 3.1 API

`POST /api/jobs` with a spec, returns the job:

```json
{"task": "sfx", "model": "sa3-small-sfx", "params": {"prompt": "...", "seconds": 3, "batch": 4, "seed": -1},
 "inputs": [{"asset": "20261105-171357-b20a5f", "role": "ref"}],
 "project": "default", "title": "Door creak",
 "meta": {"studio": "sound", "prompt": "<what the user typed>", "improved_prompt": "<improver output if used>",
          "tags": ["door"], "parents": [], "asset_extra": {"x_event": "door_open"}}}
```

* Input roles by task (the Sound studio sends them from `input_slots`): `ref` (voice / style / melody reference),
  `source` (clip to convert or extend), `melody`, `timbre`, `emo`, `ref2..ref4` (extra speakers), `audio` (spoken
  question, `s2s`), `voice_ref` (voice of the reply, `s2s`), `video` (`v2a`). A worker may attach extra files and
  return extra meta keys; `worker.reply_text` (or an attached `take_N.txt`) is the spoken reply of an `s2s` take, and
  `take_N_video.mp4` is the muxed video of a `v2a` take.
* `model` may be omitted: the task's default is used. A model that is not `ready` is refused (HTTP 409 with
  `{error, kind: "env", fix}`). Bad params -> 400 `{error, kind: "input"}`.
* Tools: `{"tool": "image.convert", "params": {...}, "inputs": [...]}` runs the handler `tool:image.convert`.
* `GET /api/jobs?status=running,queued&project=&limit=` -> `{jobs, counts}` · `GET /api/jobs/{id}` ·
  `POST /api/jobs/{id}/cancel` · `POST /api/jobs/{id}/retry` (new job, `retry_of`) · `DELETE /api/jobs` (clear history).

Job object: `{id, task, model, model_name, tool, handler, gpu, params, inputs, project, title, meta, status,
created, started, finished, progress: {frac, stage, msg, step, total, interruptible}, queue_position, outputs:
[asset ids], log: [last lines], error: {message, kind, fix} | null, timings: {queued_s, run_s, worker_load,
worker_run}, retry_of}`. Status: `queued -> running -> done | failed | cancelled`. Error kinds: `input`, `env`,
`memory`, `oom`, `internal`, `timeout`, `interrupted` (studio restarted). Terminal jobs are appended to
`data/jobs.jsonl` (also on submit); after a restart, unfinished ones show as failed/interrupted.

One GPU job runs at a time (FIFO); jobs whose model/handler is `device: cpu` run in a pool of
`jobs.cpu_concurrency` (3) next to it. Cancel is cooperative: queued jobs are removed at once; running ones get
`{"op":"cancel"}` and the worker raises at its next `ctx.check()`; after `workers.cancel_grace_s` (20 s) a worker
of a backend with `kill_on_cancel` is killed.

### 3.2 Server-Sent Events: `GET /api/events[?types=job,asset][&job=<id>]`

| event | data |
|---|---|
| `hello` | `{version, jobs}` on connect |
| `job` | the full job object (progress is throttled to 5/s per job; status changes always sent) |
| `asset` | `{action: created \| updated \| deleted, asset: meta}` |
| `models` | `{refreshed \| probed \| defaults}`: reload `/api/models` |
| `llm` | prompt-LLM status (`state: stopped \| starting \| ready \| sleeping \| failed \| external`) |
| `system` | every 3 s: `{memory: {available_gb, total_gb, ours_gb, loaded: [...]}, jobs: counts, llm}` |

The web client (`web/core/store.js`) keeps one EventSource and re-emits these on `bus` (`job`, `job:<id>`, `asset`,
`models`, `llm`, `system`). Never poll job status.

### 3.3 Handlers and plugins (server-side extension)

Any `app/plugins/<name>.py` is imported at startup. It may define `setup(core)` and a FastAPI `router`
(mounted at `/api/x/<name>`). Register handlers with the decorator:

```python
# app/plugins/sound.py
from fastapi import APIRouter
from ..jobs import JobContext, JobError, handler
from .. import media

router = APIRouter()               # -> /api/x/sound/...

@handler("tool:audio.normalize", gpu=False, label="Normalize loudness")   # CPU tool: runs beside GPU jobs
def normalize(ctx: JobContext):
    for i, inp in enumerate(ctx.input_assets()):          # [{asset_id, role, type, name, path, meta}]
        ctx.check_cancel()
        ctx.progress(i / len(ctx.input_assets()), f"Normalizing {inp['name']}")
        out = ctx.scratch / f"norm_{i}.wav"; ctx.scratch.mkdir(parents=True, exist_ok=True)
        ...                                                # write the file
        meta = ctx.asset_meta({"name": f"{inp['name']} (-14 LUFS)"}, i)
        meta["parents"] = [inp["asset_id"]]
        ctx.add_output(out, "audio", meta)                 # -> asset in the job's project, lineage kept

@handler("sound.sheet", gpu=True)                          # multi-step GPU handler: model jobs with handler: sound.sheet
def sheet(ctx: JobContext):
    lines = ctx.params["lines"]
    for k, line in enumerate(lines):
        res = ctx.run_worker(params={**ctx.params, "text": line}, frac_range=(k / len(lines), (k + 1) / len(lines)))
        ctx.ingest_worker_files(res["files"])

@router.get("/voices")
def voices(): ...

def setup(core):                                           # optional: keep a reference to the services
    global CORE; CORE = core
```

`JobContext`: `job`, `params`, `model`, `registry`, `assets`, `settings`, `scratch` (deleted after the job),
`cancel_event`, `check_cancel()`, `progress(frac, msg, stage=, step=, total=, interruptible=)`, `log(msg)`,
`input_assets()`, `run_worker(model_id=None, params=None, inputs=None, task=None, frac_range=(0, 1))`
(memory check + eviction + progress mapping + cancel; returns `{files, meta, timings}`),
`asset_meta(file_meta, index)`, `add_output(path, type, meta=None, attach=None)`, `ingest_worker_files(files)`.
Raise `JobError(message, kind, fix)` for clean failures. Reference: `app/plugins/core_tools.py`.

---

## 4. Projects and assets

```
data/projects/<project>/project.json                {id, title, description, created}
data/projects/<project>/assets/<asset_id>/           files + meta.json (written last)
data/projects/<project>/.trash/<asset_id>/           deleted assets
```

`project` matches `^[a-z0-9][a-z0-9_-]{0,47}$`; `asset_id` = `YYYYMMDD-HHMMSS-xxxxxx`; file names
`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. **meta.json (schema 1)**:

```json
{"schema": 1, "id": "20261105-171357-b20a5f", "project": "default", "type": "audio",
 "name": "sine 433 Hz", "main": "audio.wav",
 "files": [{"name": "audio.wav", "role": "main", "bytes": 132344, "mime": "audio/wav"}],
 "preview": {"waveform": "waveform.json"},            // images: {"thumb": "thumb.webp"}; videos: {"thumb": "thumb.webp"} (ffmpeg poster); meshes: thumb after the first view
 "info": {"duration": 1.5, "sample_rate": 44100, "channels": 1, "peak_db": -3.3, "rms_db": -14.0, "lufs": -15.2,
          "silence_fraction": 0.02, "clipped": false},           // image: width, height, has_alpha, transparent_fraction; video: duration, width, height, fps, codec, has_audio, audio_codec, bytes; mesh: vertices, faces, materials, textures
 "prompt": "<typed>", "improved_prompt": "<improver output or null>", "final_prompt": "<sent to the model>", "negative": "",
 "model": "test-tone", "model_name": "Test tone (self-test)", "task": "sfx", "params": {...}, "seed": 883581836,
 "parents": ["<asset ids used as inputs>"], "job": "j20261105171354...", "batch": {"id": "<job id>", "index": 0},
 "studio": "selftest", "license": {"name": "...", "commercial": true},
 "created": "2026-11-05T17:13:57", "updated": "...", "tags": [], "rating": 0, "favorite": false, "notes": "",
 "source": "generated | upload | tool", "worker": {...extra file meta from the worker...}, "x_<anything>": "studio fields"}
```

Studios may store their own fields as `x_*` keys (via `meta.asset_extra` on the job, or
`PATCH /api/assets/{id} {"x_event": "door"}`). Voice assets (`type: voice`) are audio files used as clone
references; keep the reference transcript in `x_voice: {ref_text, design_prompt, language}`.

API: `GET /api/assets?project=&type=image,audio&model=&tag=&q=&rating_min=&favorite=&job=&parent=&task=&studio=&ids=a,b&sort=new|old|rating|name&limit=&offset=`
-> `{total, items, tags, models}` · `GET /api/assets/{id}` · `PATCH /api/assets/{id}` (`name, tags, rating, notes,
favorite, x_*`) · `DELETE /api/assets/{id}` · `GET /api/assets/{id}/lineage` -> `{asset, ancestors, children,
siblings}` · `POST /api/assets/{id}/thumb` (multipart image, e.g. a model-viewer snapshot) ·
`POST /api/upload` (multipart `file`, `project`, `name`, `tags`, `type=voice`; validated by content: images by PIL,
audio by libsndfile, GLB by its header, video (mp4 / mov / webm / mkv) by magic bytes plus ffprobe; limits 60/300/500 MB, video 500 MB) · `POST /api/export {ids, naming: name|id,
include_meta, include_attachments}` -> `{url, name, bytes}` (zip: `images/`, `audio/`, `voices/`, `models/` +
`manifest.json`) · `GET /api/projects`, `POST /api/projects {id, title}`, `PATCH /api/projects/{id}`.

Files: `GET /files/<project>/<asset_id>/<name>[?download=1]`. Only names listed in the asset's `files` or `preview`
are served, from inside `data/projects`, after `resolve()` (symlinks out of the store are refused). No API takes a
server path; workers receive absolute paths only from the core (`inputs[].path`).

Python helpers for tools (`app/media.py`): `analyze_audio(path) -> (info, waveform)`, `integrated_lufs(x, sr)`,
`waveform_peaks`, `image_info`, `make_thumb`, `verify_image`, `glb_info`.
Video assets (`type: video`, `.mp4 .mov .webm .mkv`): the container is checked by its magic bytes first, then
`ffprobe` (run with a forced demuxer, `-protocol_whitelist file`, so a playlist / concat file cannot make it open
other files; without ffprobe the magic bytes alone decide) and must hold a video stream. The poster is an ffmpeg
webp (`preview.thumb`). Helpers: `video_container_ok`, `ffprobe_video`, `verify_video`, `video_info`,
`make_video_poster`. Videos are served by the same file route (range requests work, so seeking in `<video>` does).
A worker may attach a video to any asset (`attach=["take_1_video.mp4"]`); it is served and shown in the take card.

---

## 5. Prompt improver

`POST /api/improve`:

```json
{"text": "ดาบฟันโดนเกราะเหล็ก", "model_id": "sa3-small-sfx", "task": "sfx", "target_style": null,
 "lang": null, "n": 3, "chips": ["impact"], "context": "dark fantasy RPG", "image_asset": null,
 "backend": null}
```

`backend` (optional) picks the text backend for this call (section 5.1): `null` = settings `textgen.improve` (default
`local`, the llama-server below, unchanged behaviour), or `"cli:claude@haiku"`, `"api:openrouter/<model>"`, ... The
templates, schema and content checks are the same on every backend; CLI backends skip the content-check retry (each
call spends plan quota).

Style resolution: `target_style` > the model's `prompt_style` (per task if it is a map) > `describe` when only an
image is given. Response:

```json
{"prompt": "steel sword striking iron plate armor, ...", "negative": "music, melody, ...", "variations": ["...", "...", "..."],
 "notes": "<in the user's language>", "lyrics": "<music styles>", "line": "<voice styles>",
 "style": "sfx", "style_label": "Stable Audio SFX tags", "apply": {"prompt": "prompt", "negative": "negative"},
 "lang": "th", "translated": true, "llm": "Qwen3.6-35B-A3B ...", "seconds": 3.9, "warnings": [], "attempts": 1, "original": "...",
 "backend": {"id": "local", "kind": "local", "label": "Local LLM", "subscription": false}}
```

`backend` says which backend wrote the text (`kind`: local | api | cli; `subscription: true` for plan-quota CLIs);
`llm` is its human label. Backend errors keep the 503/4xx shape `{error, kind: "improve", fix}`.

`apply` says which form field each output goes to (e.g. voice styles: `{prompt: instruct, line: text}`); the
`PromptBox` does this automatically through `onApply`. `warnings` are failed content checks
(`thai_in_english_field`, `variations_not_distinct`, `label_prefix`, `lyrics_without_markers`, `notes_not_thai`,
`thai_for_en_user`, `empty_line`) that survived one retry. Errors: 503 `{error, fix}` (LLM missing / not enough
memory), 400 (empty text, unknown style).

Styles live in `prompts/*.yaml` (`core.yaml` has 25: `qwen_image, qwen_edit, kontext, zimage, hunyuan_image,
lumina, flux, image3d, describe, sfx, moss_sfx, ambience, sa3_music, music, minimax_music, voice, moss_voicegen,
omnivoice_design, qwen_tts, higgs, fish, orpheus, cosyvoice, tts_plain, translate`). Add or override one in
`prompts/<studio>.yaml`:

```yaml
styles:
  sfx_ui:
    label: UI sound (short)
    modality: audio              # image | audio | music | voice | text
    negative: true               # the target takes a negative prompt
    extra: []                    # extra required JSON keys: lyrics | line
    user_lang_variations: false  # variations are lines in the user's language (tag-based TTS)
    checks: []                   # e.g. [lyrics_markers]
    apply: {prompt: prompt, negative: negative}
    chips: [{id: click, label: Click, hint: "short dry click, no tail"}]
    system: |
      TARGET: ... (appended to the shared preamble; see docs/research/prompt_templates.md)
```

Other endpoints: `GET /api/improve/styles` · `GET /api/improve/status` · `POST /api/improve/warmup` ·
`POST /api/improve/release`. Lifecycle: llama-server on 127.0.0.1:7880 is started on the first request (primary
Qwen3.6-35B-A3B + mmproj when >= 25 GB free, else gemma-4-E2B), sleeps after 5 min idle (`--sleep-idle-seconds`),
is stopped after 30 min idle, and is the first thing evicted when a job needs memory. Settings: `settings.json` `llm`.

---

### 5.1 Text backends (`app/textgen.py`)

One interface (`core.textgen.generate(messages, schema, backend, purpose, ...)`) over:

| id | backend | JSON schema | vision | notes |
|---|---|---|---|---|
| `local` | the improver's llama-server (lazy start, eviction as above) | native | primary model | default for Improve |
| `api:<provider>/<model>` | OpenAI-compatible HTTP: `openrouter`, `openai`, `gemini` (OpenAI endpoint), `openai_compat:<name>` | `response_format` json_schema, prompt fallback | per model | keys via `core.providers.get_key()`; `core.providers.spend.check()` before / `.record()` after each call; refused when `cloud_allowed()` is false (loopback endpoints excepted) |
| `cli:claude[@model]` | `claude -p` (stream-json in/out) | `--json-schema` | stream-json image blocks | tested |
| `cli:codex`, `cli:gemini`, `cli:copilot` | `codex exec`, `gemini -p`, `copilot -p` | codex `--output-schema`; others prompt + validate + 1 retry | `-i` / `@file` / `--attachment` | untested (not installed here) |

CLI lockdown (unit-tested in `tests/test_textgen_assistant.py`): non-interactive mode, a fresh temp dir as cwd (never a
project), a scrubbed environment (no git identity / API keys / e-mail), no tools for text: claude `--tools ""
--strict-mcp-config --setting-sources "" --permission-mode dontAsk --disallowedTools Bash,Edit,Write,Read,...
--max-turns <=3 --no-session-persistence --safe-mode`; codex `--sandbox read-only -c approval_policy="never"
--ignore-user-config --ignore-rules --ephemeral`; gemini `--approval-mode default -e none`; copilot
`--deny-tool=shell,write,url,memory --no-ask-user`. Errors: `TextGenError {kind: not_installed | not_logged_in |
rate_limited | quota | timeout | cancelled | bad_output | http | budget | confirm | config | unavailable, fix}`.
Every call is logged (backend, seconds) in `logs/textgen.log`.

Settings (`settings.json` `textgen`, also `PUT /api/x/assistant/settings`): `improve`, `write`, `assistant` (backend ids),
`cli: {claude: {model, binary}}`, `api_models: [{provider, model, label, vision}]` (extra API models besides the
registry's cloud text models), `endpoints: {name: base_url}` (fallback for `openai_compat:<name>` when the providers
plugin has none), `timeout_s`, `assistant_timeout_s`.

### 5.2 Assistant + writing tools (`app/plugins/assistant.py`, `/api/x/assistant`)

* `GET /backends` -> `{backends: [{id, kind, label, available, reason, vision, tools, tested, cost, ...}], defaults, tos}` ·
  `POST /backends/test {backend}` (one tiny call) · `POST /write {kind: barks | sfx_events | asset_list | asset_meta,
  project, backend, input}` -> `{data, backend, seconds, usd}`.
* Chats: `GET|POST /chats`, `GET|PATCH|DELETE /chats/{id}?project=`, `POST /chats/{id}/message {project, text, backend}`
  (async; SSE `assistant` events `{chat, project, kind: message | status | run, ...}`), `POST /chats/{id}/cancel`.
  Files: `data/projects/<project>/assistant/<chat id>.json`.
* The agent may only use 4 tools: `list_models`, `search_assets`, `get_asset`, `propose_plan`. claude gets them from
  `workers/studio_mcp.py` (stdio MCP; `--mcp-config` + `--strict-mcp-config`, `--tools ""`, `--allowedTools
  mcp__studio__*`), which forwards to `POST /tools/{name}` with a per-turn token; local / API backends use OpenAI
  function calling in-process; other CLIs answer one JSON `{message, plan}`.
* Plan: `{title, summary, steps: [{id, title, task | tool, model, params, inputs: [{step, use: first|all|each} |
  {asset}]}]}`. `validate_plan()` rejects unknown tasks / models / tools / params, non-ready and (unless
  `ui.show_test_models`) self-test models, bad input references; it adds `est_s`, `est_usd`, `cloud`, licence. Shared
  params accepted for every model: `prompt, negative, batch, seed`.
* `POST /chats/{id}/validate {project, plan}` · `POST /chats/{id}/run {project, message, plan, steps}` runs the
  (edited, re-validated) plan: each step is submitted with `core.jobs.submit` (the POST /api/jobs path, so the cost guard
  applies) as soon as its input steps are done; `use: each` fans out one job per upstream output. Outputs carry
  `x_assistant: {chat, run, step, plan}` and tag `assistant`. `POST /runs/{run}/cancel`.

Frontend: `S.backendPicker({purpose, value, onChange})`, `S.writeHelper(kind, {onInsert})`, `S.generatedBy(result)`
(`web/core/components/writer.js`); the ✨ Improve panel has a per-click backend picker (`prefs improve.backend`). Sound
receives `{action: 'sheet_rows', params: {kind: lines|sfx, rows, name}}` (a new sheet from written rows).

## 6. Frontend studio module contract

No build step: ES modules served from `app/web/`. Libraries are vendored in `web/vendor/` (`VERSIONS.txt`);
never load anything from a CDN. One import gives you everything: `import * as S from '../core/sdk.js'`.

### 6.1 A studio module

```js
// web/studios/sound.js   (replaces the stub; the file name is free, the id must be unique)
import * as S from '../core/sdk.js';

export default S.defineStudio({
  id: 'sound', title: 'Sound', icon: 'audio-lines', color: 'var(--c-sound)', order: 30,
  section: 'studio',            // sidebar group: 'home' | 'studio' | 'workspace' | 'system'
  mobile: true, short: 'Sound', // show in the phone bottom bar (max 4 entries + More)
  receives: [                   // send-to actions offered in the asset drawer of every page
    { action: 'variations', label: 'Variations', icon: 'shuffle', types: ['audio'] },
    { action: 'voice_ref', label: 'Use as voice', icon: 'mic', types: ['audio', 'voice'] },
  ],
  mount(el, S) {                // once, on first visit; el is the page container (kept alive while hidden)
    const kit = new S.TaskStudio({ studio: 'sound', tasks: [/* ... */] });
    el.append(kit.el);
    return {                    // all optional
      route(r) { kit.route(r); },          // every navigation: r = {studio, sub, rest, query}
      receive(payload) { kit.receive(payload); },   // send-to: {action, assets: [meta], params, from}
      show() {}, hide() {},
    };
  },
});
```

Routes: `#/<studio>/<sub>?k=v` (`S.navigate(studio, sub, query)`). The shell imports every `web/studios/*.js`
listed by `/api/bootstrap`; a module that fails to load is skipped (console error), it does not break the app.

### 6.2 Send-to between studios

```js
S.sendTo('3d', assetId, { action: 'shape' });                    // from a button
S.sendTo('image', [a, b], { action: 'edit', params: { steps: 25 } });
S.sendTo('sound', [], { action: 'prompt', sub: 'sfx', params: { prompt: 'door creak', task: 'sfx' } });  // Home idea box
```

The shell navigates to the studio, fetches the asset metas and calls `receive({action, assets, params, from})`.
The asset drawer lists every studio's `receives` matching the asset type, plus **Regenerate** (sent to the asset's
`studio` with `action: 'regenerate'`; `TaskStudio.receive` restores task, model, params, seed, prompt and inputs).
Actions used today: `prompt`, `regenerate` (core); stubs declare `edit`, `extract`, `upscale` (image), `shape`,
`retexture` (3d), `variations`, `voice_ref` (sound). The Sound studio also receives `convert` (clip -> Voice conversion source), `reply` (clip -> Voice reply question) and `v2a` (video -> Video to sound). Keep these ids.

### 6.3 Shared components (all from `sdk.js`)

| export | use |
|---|---|
| `new TaskStudio({studio, tasks, includeTest, beforeSubmit, extraPanel, centerHeader, context})` | the full standard page (tabs, picker, inputs, prompt + Improve, params, presets, Generate + ETA, results, selection -> compare/export, receive/regenerate). `tasks[]`: `{id, label, result: image\|audio\|mesh, prompt: bool, promptRequired, placeholder, promptParam, exclude: [param names], inputs: [{key, label, types, role, multiple, max, required, hint}], tool, receives: [actions]}`. Members: `.picker .prompt .form .grid .slots .setTask() .generate() .spec()`. Example: `studios/selftest.js` |
| `new ModelPicker({task, value, onChange, includeTest})` | `.model`, `.setTask(task)`, `.set(id)`; greys out non-ready models with reason + fix |
| `new ParamForm({model, task, memoryKey, onChange, exclude, renderers})` | `.values()`, `.setModel(m, task)`, `.setValues(obj)`, `.applyPreset(name)`; `renderers: {mytype: (param, value, set, form) => Element}` for custom field types |
| `new PromptBox({label, placeholder, model, task, style, memoryKey, onApply, onSubmit, imageAsset: () => id, context: () => str})` | `.value`, `.negative`, `.set()`, `.setModel(m, task)`, `.setStyle(id)`, `.improved`; `S.promptMeta(box)` -> `{prompt, improved_prompt}` for job `meta` |
| `new ResultsGrid({onOpen, selectable, type, emptyState})` | `.set(list)`, `.add(a)`, `.update(a)`, `.remove(id)`, `.trackJob(job, {type, count, onDone})` (live placeholders replaced by outputs), `.selected()`, `.onSelectionChange(fn)`, `.onChange(fn)` |
| `S.assetCard(asset, opts)` | one card (image with checkerboard / mesh / audio waveform + play) |
| `new AssetSlot({label, types, multiple, max, role, onChange})` | input picker (drop, library, upload); `.inputs()` -> job `inputs` |
| `S.videoBox(asset, {name, poster, compact})`, `S.videoAttachments(asset)` | `<video controls>` + download link for a video asset or one of its attached `.mp4/.webm/.mov/.mkv` files; one player at a time |
| `S.pickAssets({types, multiple, title})` | modal library picker -> `Promise<ids \| null>` |
| `S.openAsset(idOrMeta, {list})`, `S.closeAsset()` | detail drawer (preview, prompts, lineage, rate / tag / notes, send-to, export, delete) |
| `S.compare(a, b)` | A/B modal (image wipe / side by side, audio A/B with synced position, 2-up GLB) |
| `S.jobCard(job)`, `new S.JobList({statuses, limit, project})` | live job UI (progress, ETA, cancel, retry, error + fix) |
| `S.player`, `S.waveCanvas(asset)`, `S.playButton(asset)`, `S.audioRow(asset)`, `S.waveSurfer(el, asset)` | shared audio player (one sound at a time, space toggles), peak-based waveforms, wavesurfer view |
| `S.loadModelViewer()` | lazy-load the vendored `<model-viewer>` element |
| `S.api.*`, `S.fileUrl(asset, name)`, `S.thumbUrl`, `S.waveform` | REST client (`ApiError {message, status, kind, fix}`) |
| `S.store`, `S.bus`, `S.prefs`, `S.modelsFor(task)`, `S.promptStyleOf(model, task)`, `S.waitJob(id)` | state (project, models, tasks, defaults, jobs, memory, llm), events, localStorage prefs |
| `S.h(sel, attrs, ...children)`, `S.ic(name)`, `S.btn(label, {icon, kind, size, onClick})`, `S.toast`, `S.toastError`, `S.modal`, `S.confirmDialog`, `S.promptDialog`, `S.menu`, `S.popover`, `S.seg`, `S.toggle`, `S.stars`, `S.progressBar`, `S.empty`, `S.statusBadge`, `S.licenseBadge`, `S.fmt*` | DOM + UI kit. `append/replaceChildren` ignore `null`, so `cond ? el : null` is fine. Icons: `web/core/icons.js` (`ICONS` lists names) |

Styling: use the classes and tokens in `web/app.css` (`.btn .btn-primary .btn-sm .chip .badge.ok|warn|err .card
.card-pad .field .field-label .input .select .textarea .seg .grid .studio .studio-panel .studio-center
.page-pad .section-head .kv .tbl`, colours `--c-image --c-3d --c-sound --accent --ok --warn --err`, surfaces
`--bg --panel --panel-2 --line`). Every page must work at 1440 px and 390 px (no horizontal scroll), in dark and
light. Add studio-specific CSS in your module (`document.head.append(S.h('style', css))`) with a `.<studio>-`
class prefix.

---

## 7. Environments

One venv per backend (`backends.<b>.python`, worker `workers/<b>_worker.py`); the rows from `ttslight` down were added with the
65 audio models of `models.d/{tts_light,tts_clone_a,tts_clone_b,music,sfx_more,foley,omni,vc_sing}.yaml` (research and
recreate recipes: `docs/research/audio_more/`, index in `docs/research/audio_more/README.md`). Vendored upstream
code lives in `vendor/` (git clones, not in the lock files).

| backend | venv | lock |
|---|---|---|
| core (FastAPI, echo worker, CPU tools) | `envs/core` (py3.12, no torch) | `env-lock/core.txt` |
| `qwen` (all diffusers image models) | `envs/qwen` | `env-lock/qwen.txt` |
| `hy3d`, `imgtools` (Hunyuan3D, Real-ESRGAN, rembg) | `envs/hy3d` | `env-lock/hy3d.txt` |
| `audio` (SA3, MiniMax, MOSS-SFX, OmniVoice, F5, Orpheus, CLAP, Whisper) | `envs/audio` | `env-lock/audio.txt` |
| `qwentts` (Qwen3-TTS) | `envs/qwentts` | `env-lock/qwentts.txt` |
| `moss` (MOSS-TTS, VoiceGenerator) | `envs/moss` | `env-lock/moss.txt` |
| `vllm_omni` (Higgs, Fish, Voxtral, CosyVoice) | `envs/vllm_omni` + `envs/vllm_omni_overlay` on PYTHONPATH | `env-lock/vllm_omni_overlay.txt` |
| `ttslight` (Kokoro, Piper, StyleTTS 2) | `envs/ttslight` | `env-lock/ttslight.txt` |
| `ttslight_parler` (Parler-TTS, MeloTTS) | `envs/ttslight_parler` | `env-lock/ttslight_parler.txt` |
| `clonea_xtts` (XTTS-v2 family) | `envs/clonea_xtts` | `env-lock/clonea_xtts.txt` |
| `clonea_chatterbox` (Chatterbox) | `envs/clonea_chatterbox` | `env-lock/clonea_chatterbox.txt` |
| `clonea_zonos` (Zonos) | `envs/clonea_zonos` | `env-lock/clonea_zonos.txt` |
| `clonea_openvoice` (OpenVoice V2, MeloTTS) | `envs/clonea_openvoice` | `env-lock/clonea_openvoice.txt` |
| `clonea_codec` (Spark-TTS, Llasa + xcodec2) | `envs/clonea_codec` | `env-lock/clonea_codec.txt` |
| `cloneb_index` (IndexTTS 2 / 2.5) | `envs/cloneb_index` | `env-lock/cloneb_index.txt` |
| `cloneb_amphion` (MaskGCT, Vevo, Vevo 1.5) | `envs/cloneb_amphion` | `env-lock/cloneb_amphion.txt` |
| `cloneb_hf` (Dia, Dia2, Bark, VibeVoice, CSM) | `envs/cloneb_hf` | `env-lock/cloneb_hf.txt` |
| `music_audiocraft`, `music_riffusion` (MusicGen, AudioGen, MAGNeT; Riffusion; shared venv) | `envs/music_audiocraft` | `env-lock/music_audiocraft.txt` |
| `music_acestep` (ACE-Step 1.5) | `envs/music_acestep` | `env-lock/music_acestep.txt` |
| `music_diffrhythm` (DiffRhythm 1.2) | `envs/music_diffrhythm` | `env-lock/music_diffrhythm.txt` |
| `music_yue` (YuE) | `envs/music_yue` | `env-lock/music_yue.txt` |
| `music_heartmula` (HeartMuLa) | `envs/music_heartmula` | `env-lock/music_heartmula.txt` |
| `music_magenta` (Magenta RealTime 2, JAX) | `envs/music_magenta` | `env-lock/music_magenta.txt` |
| `music_levo` (SongGeneration / LeVo; **not installed**, upstream code unavailable, models report `missing_env`) | `envs/music_levo` | - |
| `sfxmore` (AudioLDM 1/2, Tango 2, TangoFlux) | `envs/sfxmore` | `env-lock/sfxmore.txt` |
| `foley` (MMAudio, HunyuanVideo-Foley, ThinkSound: video to sound) | `envs/foley` | `env-lock/foley.txt` |
| `omni_moshi` (Moshi) | `envs/omni_moshi` | `env-lock/omni_moshi.txt` |
| `omni_qwen` (Qwen2.5-Omni, Qwen3-Omni) | `envs/omni_qwen` | `env-lock/omni_qwen.txt` |
| `omni_glm` (GLM-4-Voice) | `envs/omni_glm` | `env-lock/omni_glm.txt` |
| `omni_minicpm` (MiniCPM-o 4.5) | `envs/omni_minicpm` | `env-lock/omni_minicpm.txt` |
| `omni_step` (Step-Audio 2 mini) | `envs/omni_step` | `env-lock/omni_step.txt` |
| `omni_kimi` (Kimi-Audio) | `envs/omni_kimi` | `env-lock/omni_kimi.txt` |
| `omni_llama2` (LLaMA-Omni2) | `envs/omni_llama2` | `env-lock/omni_llama2.txt` |
| `vc` (Seed-VC, RVC) | `envs/vc` | `env-lock/vc.txt` |
| `vc_diffsinger` (DiffSinger) | `envs/vc_diffsinger` | `env-lock/vc_diffsinger.txt` |
| `video` (Wan 2.1/2.2, LTX-2.3, Kandinsky 6, HunyuanVideo 1.5, RIFE; Video studio) | `envs/video` | `env-lock/video.txt` |
| `comfyui` (optional remote FLUX.2) | `envs/core` (HTTP client of a ComfyUI server) | - |

Ports: studio 7860 (tests 7861-7869), prompt LLM 7880 (fallback may use 7881), workers have no ports. Make sure
none of these collide with other services on the machine (ComfyUI listens on 8188 by default).

## 8. Cloud providers (`app/providers.py`, `app/cloud/`, `app/plugins/providers.py`)

Optional, **off by default**. Cloud models live in `models.d/cloud.yaml`; the Providers page (`web/studios/providers.js`)
manages keys, the global switch, limits, prices, user endpoints and shows subscription CLIs.

### 8.1 `core.providers` (for other plugins; created first by `app/plugins/providers.py`, before every other plugin)

```python
P = core.providers                     # app.providers.Providers
P.get_key("openrouter")                # str | None: data/secrets.json first, then the env var (table below). Never send it to a browser.
P.cloud_allowed()                      # bool: the global "Allow cloud" switch (default False)
P.get_base_url("gemini")               # OpenAI-compatible base for chat: openai | openrouter | gemini (…/v1beta/openai) | openai_compat:<name>
P.is_local_provider("openai_compat:lab")   # True for loopback / LAN endpoints: no switch, no spend, no consent
P.label(pid); P.headers(pid)           # display name; auth header dict for raw calls
P.spend.check(est_usd, provider_id, *, confirmed=False, model_id=None)
    # raises ProviderError (a JobError) kind "cloud_disabled" | "budget" | "confirm"; e.data = {estimate_usd, month_usd,
    # pending_usd, budget_usd, confirm_usd, provider, provider_label}. est_usd None = unknown price -> confirm.
P.spend.record(usd, provider_id, model_id=None, project=None, job_id=None, *, estimated=False, note="")
P.estimate(model, params, n_inputs=0)  # USD from the registry `pricing` block (same formula as web/core/cost.js)
P.clis()                               # detected subscription CLIs (display data, cached 5 min)
```

| provider id | env var(s) | key test (free) |
|---|---|---|
| `fal` | `FAL_KEY`, `FAL_API_KEY` | `GET api.fal.ai/v1/models/pricing` (+ billing if the key has the scope) |
| `replicate` | `REPLICATE_API_TOKEN` | `GET /v1/account` |
| `openrouter` | `OPENROUTER_API_KEY` | `GET /api/v1/key` (usage, limit) |
| `openai` | `OPENAI_API_KEY` | `GET /v1/models` |
| `gemini` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `GET /v1beta/models` |
| `elevenlabs` | `ELEVENLABS_API_KEY`, `XI_API_KEY` | `GET /v1/user/subscription` (characters left) |
| `meshy` | `MESHY_API_KEY` | `GET /openapi/v1/balance` (credits) |
| `tripo` | `TRIPO_API_KEY` | `GET /v3/account/balance` (credits) |
| `openai_compat:<name>` | key optional (secrets file) | `GET {base_url}/models` |

Files (under `STUDIO_DATA`): `secrets.json` (mode 600, rewritten atomically, re-chmodded on load), `providers.json`
(switch, `budget_usd` 20, `confirm_usd` 0.50, `price_overrides`, `endpoints`, `upload_ack`), `spend.jsonl`
(`{ts, month, usd, provider, model, project, job, estimated, note}`). No API response ever contains a key: keys are
shown as `{set, source: file|env, env, last4}` (last 4 only for keys of 16+ characters); provider error texts are
scrubbed (`P.secrets_in`).

### 8.2 Registry entries (`backend: cloud`)

```yaml
backends:
  cloud: {kind: cloud, device: cpu, handler: cloud}       # handler / device apply to every cloud model
models:
  - id: fal-flux2-pro
    backend: cloud
    provider: fal                                          # see 8.1
    remote: {t2i: fal-ai/flux-2-pro, edit: fal-ai/flux-2-pro/edit}   # or remote_model: <id> (all tasks)
    adapter: openai                                        # optional; default = provider (fal, replicate, openai, gemini,
                                                           # elevenlabs, meshy, tripo, openrouter; openai_compat:* -> openai)
    api: el_sfx                                            # optional sub-API of the adapter (el_tts, gemini_tts, or_chat ...)
    request: {size: wh, images: image_urls, n: num_images, map: {seed: seed}}   # fal / Replicate input mapping (app/cloud/common.py)
    pricing: {first_megapixel: 0.03, per_megapixel: 0.015}
    pricing_checked: "2026-11-05"
    pricing_source: https://fal.ai/models/fal-ai/flux-2-pro
    usage_pricing: {text_in: 5, image_in: 8, image_out: 30}    # OpenAI images: actual cost from usage tokens (USD / 1M)
    unverified: "why"                                      # shown on the Providers page
    license: {name: ..., commercial: true, note: ...}      # per-model terms (shown on assets)
```

Status (hook `registry.status_hooks["cloud"]`): `cloud_disabled` (switch off; kind `cloud_disabled`) → `missing_key`
(kind `missing_key`) → `ready`. Local/LAN `openai_compat` endpoints are `ready` with the switch off. A submit of a
not-ready model returns 409 with `kind` = the status. Registry entries get `cloud: {provider, label, terms, terms_url,
remote_model, local}`, `device: cpu`, tag `cloud`, and the price override (`pricing`, `pricing_default`,
`pricing_overridden`). Text models (`modality: text`, tasks `improve` / `write`) use
`pricing: {input_per_mtok, output_per_mtok}` and `remote_model` = the API model id; they are not jobs (the assistant /
improver call them). User endpoints become entries `oc-<name>-<model>` (text) / `oc-<name>-<model>-img` (t2i).

`pricing` keys (USD; all optional, summed): `per_image` (x batch), `first_megapixel` + `per_megapixel` (per output image,
1 MP = 1024², rounded up), `per_input_image`, `per_input_megapixel`, `per_request` (x batch), `per_second` /
`per_minute` (x `seconds`, `default_seconds`), `per_1k_chars` (x `text`/`prompt` length), `credits` + `per_credit`,
`input_per_mtok` / `output_per_mtok`, `by: {param: {value: {<keys>}}}` (override per param value), `add: {param:
{value: usd}}`, `min_usd`, `per_predict_second` (Replicate: actual cost from `metrics.predict_time`).

### 8.3 Cloud jobs

Handler `cloud` (`gpu=False`, `pool="cloud"`, `jobs.cloud_concurrency` default 4): runs **in-process** in the core (only
HTTP waits; the key never leaves the process), never in the GPU queue. Its `precheck` (in `submit`, before the job
exists) refuses with 409 and these kinds; the web client handles `confirm` / `consent` automatically:

| kind | when | client action |
|---|---|---|
| `cloud_disabled` / `missing_key` | switch off / no key | error toast with a Providers link |
| `consent` | the job has inputs and the provider was never allowed to receive files (`data.terms`) | dialog, resubmit with `confirm_upload: true` (stored once per provider; revoke on the Providers page) |
| `confirm` | estimate ≥ `confirm_usd`, or unknown price | dialog, resubmit with `confirm_cost: true` |
| `budget` | month spend + running estimates + this job > `budget_usd` | error toast |

`POST /api/jobs` and `POST /api/jobs/{id}/retry` accept `confirm_cost` / `confirm_upload`. The job gets
`cost: {estimate_usd, provider, confirmed, usd, actual_usd, estimated, note, request_ids}`; each output asset gets
`cloud: {provider, provider_label, remote_model, request_ids, terms, terms_url, uploaded_inputs}` and
`cost: {usd, job_usd, estimated}`. Actual cost is used when the API reports it (OpenAI usage tokens, OpenRouter
`usage.cost`, Meshy / Tripo consumed credits, Replicate compute seconds), else the estimate (`estimated: true`).
Outputs: images → PNG, meshes → GLB (+ preview attachment), audio → wav / mp3, ElevenLabs voice design → `voice`
assets with `x_voice`. Inputs are sent as data: URIs (fal, Replicate, Meshy, OpenRouter, Gemini inline data),
multipart (OpenAI edits) or the provider's upload endpoint (Tripo `/v3/files`).

### 8.4 API (`/api/x/providers`)

`GET /` → `{allow_cloud, budget_usd, confirm_usd, providers[], endpoints[], price_overrides, spend, models[]}` ·
`PUT /keys/{pid} {key}` · `DELETE /keys/{pid}` · `POST /test/{pid}` → `{ok, message, details}` · `PUT /allow {on}` ·
`PUT /limits {budget_usd, confirm_usd}` · `GET /spend?month=YYYY-MM` · `PUT /endpoints {name, base_url, label, key,
models, modalities: [llm|image], price_per_image}` · `DELETE /endpoints/{name}` · `POST /endpoints/{name}/models`
(reads `GET /models`) · `PUT|DELETE /prices/{model}` `{pricing}` · `PUT /upload-ack/{pid} {on}` ·
`GET /clis?refresh=1` · `POST /precheck {task, model, params, inputs, confirm_*}` (dry run of 8.3, always 200) ·
`POST /estimate {model, params, inputs}` · `GET /catalog`. SSE: `providers` (settings changed), `spend` (a call was
recorded), `models` (status changed). Web: `web/core/cost.js` (`estimateCost`, `priceLabel`, `cloudBadge`,
`priceBadge`, `costSpan`, `jobCost`, exported from `sdk.js`), `api.setSubmitGuard(onError, before)`.

### 8.5 Subscription CLIs (display only)

`claude` (`claude auth status`, JSON, no quota), `codex` (`codex login status` / `~/.codex/auth.json`), `gemini` / `agy`
(`~/.gemini/oauth_creds.json`), `copilot` (env tokens / `~/.copilot`), `cursor-agent` (`cursor-agent status`): installed,
version, logged in (`true | false | null` = not checkable), install + login hints. No account names or e-mails are
returned.

### 8.6 Testing without keys

`tests/fixtures/cloud_mock.py` imitates every provider's submit / poll / result shapes. Run it
(`envs/core/bin/python tests/fixtures/cloud_mock.py --port 7899`) and start a core with
`STUDIO_CLOUD_MOCK=http://127.0.0.1:7899` (provider `p`, base kind `k` → `<mock>/<p>/<k>`); any key works except one
containing `bad`. Unit tests: `tests/test_cloud.py`.

## 9. Profiles, export presets, licence filter (`app/profiles.py`, `app/licensing.py`, `app/media.py`)

### 9.1 Profiles

`settings.json` key `profile`: `game | video | podcast | social | film | general` (default `general`; `PATCH /api/settings
{"profile": "video"}` validates it and publishes the SSE event `settings`). `GET /api/profiles` returns
`{current, default_export, license, profiles: [...], export_presets: [...]}`; a profile is
`{id, label, icon, summary, context?, aspect, aspects, export, export_presets, hints}`:

| field | meaning |
|---|---|
| `aspect` | default aspect ratio (`"16:9"`, `"9:16"`, `"1:1"`) or `null` = keep the model default (game, general) |
| `export` | id of the default export preset (`original` for game / general) |
| `export_presets` | presets listed first in the export dialog |
| `hints` | free-form: `rgba`, `loop`, `mesh_format` (game), `video_aspect` (for video studios) |
| `context` | one line prepended to the prompt improver's CONTEXT (absent for game / general: prompts unchanged) |

**Preset tags.** Prompt chips (`prompts/*.yaml`, in `chipsets:` or a style's `chips:`) may carry
`profiles: [game|video|podcast|social|film|general]` and `default_for: [profile, ...]`. Untagged chips belong to every
profile. The UI (`profileChips()` in `PromptBox`) lists chips of the active profile first, then untagged, then chips
tagged only for other profiles (dimmed); `game` keeps the file order for untagged chips; `general` never reorders.
`default_for` chips start selected the first time a prompt box is used under that profile. The improver ignores both
keys (it only reads `id` and `hint`). Add presets by tagging, never by copying a chip set per profile.

**Frontend API** (from `sdk.js`): `S.getProfile()` -> the active profile object; `S.store.profile` (id);
`S.profileAspect(choices)` -> the profile's aspect if it is one of the model's choices, else `null`;
`S.defaultExportPreset()`; `S.setProfile(id)`; bus events `'profile'` (data: the profile object), `'license'`
(`{commercial_only, strict}`), `'export'` (default preset id). `ParamForm` already applies `profileAspect` to any param
named `aspect` (type `choice`) on first use of a model and on every `'profile'` event, so a studio gets it for free by
naming its param `aspect` with choices like `"16:9"`. `TaskStudio` accepts `placeholder: profile => string`.

### 9.2 Export presets

`EXPORT_PRESETS` in `app/profiles.py` (`original`, `video_online`, `broadcast`, `film_master`, `podcast`,
`social_vertical`, `social_square`, `thumbnail`, `music_cd`, `web_images`). A preset is `{label, summary, audio, image}`;
`audio` / `image` are option dicts or `null` (leave that kind untouched):

```
audio: {format: wav|mp3|ogg|flac, sample_rate, bit_depth: 16|24|32, lufs, true_peak: -1.0, channels: 1|2,
        bitrate_kbps (mp3), music_sample_rate (used instead of sample_rate for assets whose task is music / song)}
image: {format: png|jpeg|webp, aspect: "16:9", width, height, max_side, fit: crop|pad, quality, background: "#rrggbb"}
```

Loudness is a two-pass ffmpeg `loudnorm` (`media.export_audio`): pass 1 measures, pass 2 applies the measured values
with `linear=true`; sample rate / bit depth / channels are applied after it. Silent input is converted without a gain
change (`report.note`). Images: `media.export_image` crops (centred) or pads to `aspect` (or `width:height`), resizes
(`width`+`height` exact, `max_side` never upscales), JPEG flattens alpha onto `background`. Both are CPU only and raise
`media.ExportError` (a `ValueError`, HTTP 400) for bad options.

**Originals are never modified.** Every route below writes a derived file elsewhere:

| route | body | result |
|---|---|---|
| `POST /api/export` | `{ids, naming, include_meta, include_attachments, export?}` | zip; with `export: {preset?, audio?, image?}` the main file of each audio / voice / image asset is replaced by its converted copy (`manifest.json` gets `export` and a per-asset `export` = report + `original` file name). Omitted = stored files as they are |
| `POST /api/assets/{id}/render` | `{export: {preset?, audio?, image?}}` | `{url, name, bytes, report}` for one converted file under `/api/export/render/<token>/<name>` (400 if the spec has nothing for the asset type) |
| tool `asset.export` (`POST /api/jobs {tool: "asset.export", params: {preset?, audio?, image?}, inputs}`) | | saves converted copies as NEW assets (`parents` = the originals, `info.export` = report, tags `export`, `<preset>`) |

Request resolution (`profiles.resolve_export`): `preset` gives the base options, explicit `audio` / `image` dicts are
merged over it, `null` / `{}` / `false` switch that kind off; with neither a preset nor option keys the settings default
applies (`settings.export.preset`, else the profile's `export`). `report` (audio): `{format, sample_rate, bit_depth,
target_lufs, true_peak, normalized, input_lufs, output_lufs, output_true_peak, result:{duration, sample_rate, channels,
peak_db, lufs}, note?}`; (image): `{format, width, height, source_width, source_height, alpha}`.

### 9.3 Licence filter

Every registry model has `license: {name, commercial: bool, note?, verify?}` (section 1). Three states
(`licensing.state()` / `S.licenseState()`):

| state | rule | badge |
|---|---|---|
| `commercial` | `commercial: true` | Commercial OK |
| `check` | `commercial: true, verify: true` | Check licence |
| `noncommercial` | `commercial: false` (or no licence) | Non-commercial |

Settings `license.commercial_only` (bool) and `license.strict` (bool), set with `PATCH /api/settings {"license": {...}}`
(also in `GET /api/bootstrap` -> `settings.license` and `GET /api/profiles`). When `commercial_only` is on:

* `JobManager.submit` (so `POST /api/jobs`, retries, pipelines, the assistant's plans) refuses a `noncommercial` model
  (and a `check` model when `strict`) **before** anything starts: HTTP **403**, body
  `{"error": "<model> is not licensed for commercial use (<licence>), and 'Commercial-safe models only' is on. ...",
  "kind": "license", "fix": "..."}` (`JobError(kind="license")`; `licensing.check`).
* the web pickers hide those models: `S.modelsFor(task)` omits them (`{includeBlocked: true}` keeps them); `ModelPicker`
  lists them greyed out at the bottom with the reason; `TaskStudio`'s Generate button is disabled for a blocked model;
  `S.isLicenseBlocked(model)` is the check for studios that build their own pickers. Studios should use `modelsFor`
  or `isLicenseBlocked` rather than testing `license.commercial` themselves, and re-read on bus `'license'`.
* the assistant's `list_models` tool and catalog leave them out.

Existing assets are never hidden or deleted by the filter; the licence of the model stays on every asset.

## Changelog
- 0.1.0: initial core.
- 0.1.1 (image/3D studio): `app/main.py` keeps the catch-all static mount last after `load_plugins()`, so plugin routers
  under `/api/x/<name>` are reachable (they were shadowed by the `/` mount).
- 0.2.0 (text backends + Assistant): new `app/textgen.py` (`core.textgen`), `Improver(textgen=...)` +
  `improve(..., backend=None)` (default unchanged: local), `POST /api/improve {backend}` + response `backend`,
  `TextGenError` handler in `app/main.py`, `textgen` allowed in `PATCH /api/settings`, SSE event `assistant` forwarded on
  the web bus (`web/core/store.js`), icons `bot`, `message-square`, `pen-line`, `web/core/components/writer.js` (exported
  from `sdk.js`), backend picker in the Improve panel. New plugin `app/plugins/assistant.py`, page
  `web/studios/assistant.js`, MCP server `workers/studio_mcp.py`. No new Python dependencies.
- 0.3.0 (cloud providers): `app/providers.py`, `app/cloud/*`, plugin `app/plugins/providers.py` (loaded first by
  `Core.load_plugins`), `models.d/cloud.yaml`, page `web/studios/providers.js`, `web/core/cost.js`. Core hooks (all
  backwards-compatible): `handler(..., precheck=, pool=)` + named non-GPU pools (`jobs.<pool>_concurrency`),
  `JobError(data=)` merged into the 409/400 body, a not-ready refusal uses the status' `kind` (default `env`), backend
  `handler:` default, `JobManager.retry(confirm_cost=, confirm_upload=)` + `POST /api/jobs/{id}/retry` body,
  `JobReq.confirm_cost` / `confirm_upload`; registry `model_sources`, `model_hooks`, `status_hooks[kind]`,
  `recompute()`, statuses `missing_key` / `cloud_disabled`, implicit task defaults prefer local models. Web: cost on
  Generate buttons (TaskStudio, Sound) and job cards, cloud + price badges in the ModelPicker / Models page, cloud fix
  button, asset drawer cost + terms, `api.submit` / `api.retry` confirm + consent dialogs, new task `text3d` (3D
  studio tab) and `write`, icons `cloud`, `circle-dollar-sign`, `key`, `plug`, `wallet`. No new Python dependencies
  (httpx was already in envs/core).
- 0.4.0 (65 more audio models, Sound studio UI): registry `input_slots` (see section 1; validated in
  `Registry._normalize_slots`, exposed on `GET /api/models`); new Sound tasks `voice_convert`, `sing`, `s2s`, `v2a`
  (`app/plugins/sound.py` categories); asset type `video` (upload of mp4 / mov / webm / mkv validated by content,
  `media.verify_video` / `video_info` / `make_video_poster`, ffmpeg poster, 500 MB limit, library filter, `<video>`
  in the drawer, `S.videoBox`, export folder `video/`); `x_inputs` stored on takes so Regenerate restores slots;
  prompt style `sing_score` (`prompts/sound.yaml`), Improve is hidden for models / tasks without a template;
  backends of the new models in section 7. Tests: `tests/test_files_safety.py` (video upload + serving, slot
  validation). No new Python dependencies in envs/core; ffmpeg / ffprobe are optional (magic bytes are the fallback).
- 0.5.0 (media profiles): section 9. `app/profiles.py`, `app/licensing.py`, `GET /api/profiles`, settings `profile` /
  `license` / `export` (allowed in `PATCH /api/settings`, SSE `settings`), `POST /api/assets/{id}/render`,
  `ExportReq.export`, `AssetStore.render/convert_main/export(convert=)`, `media.export_audio/export_image`, tool
  `asset.export`, `JobError(kind="license")` -> HTTP 403, prompt chips `profiles` / `default_for`, web `core/license.js`,
  `core/components/exportdialog.js`, `S.getProfile/profileAspect/setProfile/isLicenseBlocked/exportDialog`.
- video studio (Video tasks): new tasks `t2v`, `i2v`, `flf2v`, `interpolate` (`models.d/video.yaml`, backend `video`, worker
  `workers/video_worker.py` + `video_common.py` + `video_rife.py`); new input roles `last_image` (flf2v) and `video`
  (interpolate); output asset type `video` (h264 / yuv420p / faststart mp4, AAC when the model makes sound, poster by the core).
  Plugin `app/plugins/video.py`: `GET /api/x/video/formats` and tools `video.frame|trim|loop|speed|reframe|convert|concat|mux`.
  Web: `app/web/studios/video.js` (`receives` i2v / continue / interpolate / tools; "Add sound" = `S.sendTo('sound', id, {action:'v2a'})`).
  Prompt styles `video_wan|ltx|kandinsky|hunyuan` (`prompts/video.yaml`). Research: `docs/research/video.md`. Tests: `tests/test_video.py`.
- 3D rig preview (no core change): CPU tool `tool:mesh.flex_preview` (`app/plugins/mesh.py`; input one skinned GLB, output a new mesh asset with the animation clips `flex` / `wiggle`, `JobError(input)` when there is no skin), clip builder `workers/rig_anim.py` (numpy + stdlib; also used by the rig worker, which now returns the animated `rigged.glb` plus `rigged_plain.glb`, meta `worker.animations`). The shared mesh viewer `web/studios/i3d/common.js` autoplays a GLB's first clip and adds play / pause + clip picker (`autoplay`, `availableAnimations`); the 3D studio adds an "Animate preview" action for rigged meshes without clips. Tests: `tests/test_rig_anim.py`.
- 3D text to motion (no core change): task `motion` (3D studio; input role `mesh` = a rigged GLB, prompt, output a new `mesh` asset `animated.glb`
  with ONE clip `motion` plus attachments `motion.npz` and `retarget.json`), backend `motion_hy` (`workers/motion_hy_worker.py`, env `motion_hy`),
  model `hy-motion-lite` (`models.d/motion.yaml`). Pure helpers in `workers/motion_common.py` (SMPL-H FK, geometric humanoid detection that refuses
  non-humanoids with `NotHumanoidError`, retarget with foot lock); new public `rig_anim.append_clip(src, name, tracks, ...)`. Web: Motion tab and a
  "Motion" button for rigged meshes in `app/web/studios/3d.js` (`receives` action `motion`). Research: `docs/research/motion.md`. Tests: `tests/test_motion.py`.
