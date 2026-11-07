"""The settings schema: every user-visible setting, its type / range / scope, and the validator behind the Settings page.

`build(core)` is served at GET /api/settings/schema (the page is generated from it) and `validate_patch()` guards
PATCH /api/settings and PATCH /api/projects/<id>/settings. Field shape and rules: docs/CONTRACTS.md section 10.

Scope: "system" = settings.json (all users of this studio), "project" = a per-project override stored in the project's
`project.json` under `settings`. A setting that only exists in the system scope (memory, workers, the prompt LLM ...)
describes shared hardware, so a project cannot override it.

Apply: when a change takes effect. live = the next read; new_worker = workers started after the change;
llm_start = the next prompt-LLM start; restart = the next studio restart.
"""
from __future__ import annotations

import os
import re
from typing import Any

from . import config, profiles

TYPES = ("bool", "int", "float", "string", "text", "choice", "model", "backend", "list_str", "json")
SCOPES = ("system", "project")
APPLY = ("live", "new_worker", "llm_start", "restart")
PROMPT_LANGS = ("auto", "en", "th")
CLI_MODEL_RE = re.compile(r"[A-Za-z0-9._:\-\[\]]{1,80}")

GROUPS = [
    {"id": "general", "label": "General", "icon": "sliders-horizontal", "scopes": ["system", "project"],
     "help": "Use-case profile, commercial-use filter and export."},
    {"id": "models", "label": "Default models", "icon": "box", "scopes": ["system", "project"],
     "help": "The model each task uses when nothing else is picked."},
    {"id": "prompts", "label": "Prompts & text", "icon": "sparkles", "scopes": ["system", "project"],
     "help": "Which text backend improves prompts and writes, prompt language and project art direction."},
    {"id": "cloud", "label": "Cloud", "icon": "cloud", "scopes": ["system"],
     "help": "Optional paid cloud models. Off by default."},
    {"id": "memory", "label": "Memory", "icon": "memory-stick", "scopes": ["system"],
     "help": "How much memory is kept free, and when the studio unloads models."},
    {"id": "workers", "label": "Workers & queue", "icon": "cpu", "scopes": ["system"],
     "help": "Idle timeouts of model workers and the job queue."},
    {"id": "llm", "label": "Prompt LLM", "icon": "bot", "scopes": ["system"],
     "help": "The local llama.cpp model that improves prompts."},
    {"id": "env", "label": "Environment", "icon": "terminal", "scopes": ["system"],
     "help": "Read-only: what the studio was started with."},
]


def _f(key: str, type: str, label: str, group: str, *, help: str = "", scope=("system", "project"),  # noqa: A002
       apply: str = "live", default: Any = None, nullable: bool = False, editable: bool = True, **kw) -> dict:
    return {"key": key, "type": type, "label": label, "help": help, "group": group, "scope": list(scope), "apply": apply,
            "default": default, "nullable": nullable, "editable": editable, **kw}


SYS = ("system",)
D = config.DEFAULT_SETTINGS


def _static_fields() -> list[dict]:
    f: list[dict] = []
    add = f.append
    # ---- general
    add(_f("profile", "choice", "Use-case profile", "general", choices_from="profiles", default=profiles.DEFAULT_PROFILE,
           help="Changes the style chips, default aspect ratio and export preset to suit the kind of production."))
    add(_f("license.commercial_only", "bool", "Commercial-safe models only", "general", default=False,
           help="Hide and refuse models whose licence does not allow commercial use."))
    add(_f("license.strict", "bool", "Also refuse models to verify", "general", default=False,
           help="With the filter on, also refuse models flagged 'verify the licence'."))
    add(_f("export.preset", "choice", "Export preset", "general", choices_from="export_presets", nullable=True, default=None,
           help="Used when assets are downloaded or zipped. Empty = the profile's preset."))
    add(_f("ui.show_test_models", "bool", "Show self-test models", "general", scope=SYS, default=False,
           help="List the synthetic test models in the pickers."))
    # ---- prompts & text
    for p, lab in (("improve", "Prompt improver"), ("write", "Writer (barks, SFX lists ...)"), ("assistant", "Assistant")):
        add(_f(f"textgen.{p}", "backend", f"{lab} backend", "prompts", nullable=True, default=None,
               help="'local', 'cli:<name>[@model]' or 'api:<provider>/<model>'. Empty = automatic."))
    add(_f("prompt.lang", "choice", "Prompt language", "prompts", choices_from="prompt_lang", default="auto",
           help="Language of improved prompts when the request does not say. Auto = the language you typed."))
    for s, lab in (("image", "Image"), ("video", "Video"), ("sound", "Sound")):
        add(_f(f"prompt.context.{s}", "text", f"{lab} art direction", "prompts", maxlen=300, nullable=True, default=None,
               help=f"Added to every {s} prompt improvement, e.g. 'low-poly mobile game, warm palette'."))
    add(_f("textgen.timeout_s", "int", "Text request timeout", "prompts", scope=SYS, unit="s", min=10, max=3600, step=10,
           nullable=True, default=None, help="Per request. Empty = 180 s for API, 300 s for CLI backends."))
    add(_f("textgen.assistant_timeout_s", "int", "Assistant turn timeout", "prompts", scope=SYS, unit="s", min=30, max=7200,
           step=30, nullable=True, default=None, help="A whole Assistant turn with tools. Empty = 420 s."))
    from .textgen import CLI_INFO

    for name, info in CLI_INFO.items():
        add(_f(f"textgen.cli.{name}.model", "string", f"{info['label']} model", "prompts", scope=SYS, pattern=CLI_MODEL_RE.pattern,
               maxlen=80, nullable=True, default=None, subgroup="CLI models",
               help=f"Model flag passed to {info['binary']}. Empty = its default.", suggestions=info.get("models") or []))
        add(_f(f"textgen.cli.{name}.binary", "string", f"{info['label']} binary", "prompts", scope=SYS, nullable=True, default=None,
               editable=False, subgroup="CLI models", help="Path of the executable. It is run, so it is file-only."))
    add(_f("textgen.api_models", "json", "API models", "prompts", scope=SYS, default=[], gui=False,
           help="Managed in the Assistant > Text backends dialog."))
    add(_f("textgen.endpoints", "json", "Custom endpoints", "prompts", scope=SYS, default={}, gui=False,
           help="Managed in the Assistant > Text backends dialog."))
    # ---- memory
    add(_f("memory.margin_gb", "float", "Safety margin", "memory", scope=SYS, unit="GB", min=0, max=32, step=0.5,
           default=D["memory"]["margin_gb"], help="Kept free on top of a model's memory need before a job starts."))
    add(_f("memory.llm_yield_gb", "float", "Stop the prompt LLM above", "memory", scope=SYS, unit="GB", min=0, max=200, step=1,
           default=D["memory"]["llm_yield_gb"], help="Jobs whose model needs at least this much always stop the prompt LLM first."))
    add(_f("memory.evict_wait_s", "float", "Wait after unloading", "memory", scope=SYS, unit="s", min=0, max=120, step=1,
           default=D["memory"]["evict_wait_s"], help="How long to wait for memory to come back after a model was unloaded."))
    # ---- workers & queue
    add(_f("workers.model_idle_s", "int", "Unload an idle model after", "workers", scope=SYS, unit="s", min=30, max=86400, step=30,
           apply="new_worker", default=D["workers"]["model_idle_s"], help="A worker drops a model that was not used for this long."))
    add(_f("workers.exit_idle_s", "int", "Stop an idle worker after", "workers", scope=SYS, unit="s", min=30, max=86400, step=30,
           apply="new_worker", default=D["workers"]["exit_idle_s"], help="A worker exits when it holds no model for this long."))
    add(_f("workers.cancel_grace_s", "int", "Cancel grace", "workers", scope=SYS, unit="s", min=1, max=600, step=1,
           default=D["workers"]["cancel_grace_s"], help="A running job that ignores Cancel this long gets its worker killed."))
    add(_f("jobs.cpu_concurrency", "int", "CPU jobs at once", "workers", scope=SYS, min=1, max=16, step=1, apply="restart",
           default=D["jobs"]["cpu_concurrency"], help="CPU tools run alongside the single GPU job."))
    add(_f("jobs.cloud_concurrency", "int", "Cloud jobs at once", "workers", scope=SYS, min=1, max=16, step=1, apply="restart",
           default=D["jobs"]["cloud_concurrency"], help="Network-bound cloud jobs run alongside the GPU job."))
    add(_f("jobs.history_keep", "int", "Job history kept", "workers", scope=SYS, min=100, max=100000, step=100, apply="restart",
           default=D["jobs"]["history_keep"], help="How many finished jobs are loaded again at start."))
    # ---- prompt LLM
    L = D["llm"]
    add(_f("llm.temperature", "float", "Temperature", "llm", scope=SYS, min=0, max=2, step=0.05, default=L["temperature"],
           help="Sampling temperature of the prompt improver."))
    add(_f("llm.kill_idle_s", "int", "Stop after idle", "llm", scope=SYS, unit="s", min=60, max=604800, step=60,
           default=L["kill_idle_s"], help="The studio stops llama-server after this long without a request."))
    add(_f("llm.start_timeout_s", "int", "Start timeout", "llm", scope=SYS, unit="s", min=10, max=1800, step=10,
           default=L["start_timeout_s"], help="How long to wait for llama-server to become ready."))
    add(_f("llm.sleep_idle_s", "int", "Free weights after idle", "llm", scope=SYS, unit="s", min=0, max=86400, step=60,
           apply="llm_start", default=L["sleep_idle_s"], help="llama-server frees its weights itself after this long (0 = never)."))
    for slot, lab in (("primary", "Primary"), ("fallback", "Fallback")):
        sg = f"{lab} model"
        add(_f(f"llm.{slot}.name", "string", "Name", "llm", scope=SYS, maxlen=80, apply="llm_start", default=L[slot]["name"], subgroup=sg))
        add(_f(f"llm.{slot}.model", "string", "Model file", "llm", scope=SYS, maxlen=500, apply="llm_start", default=L[slot]["model"],
               subgroup=sg, file_check=True, help="lmstudio:<publisher>/<repo>/<file>, hf:<org>/<repo>/<file> or an absolute path."))
        add(_f(f"llm.{slot}.mmproj", "string", "Vision projector", "llm", scope=SYS, maxlen=500, apply="llm_start", nullable=True,
               default=L[slot]["mmproj"], subgroup=sg, file_check=True, help="Optional; same format as the model file."))
        add(_f(f"llm.{slot}.memory_gb", "float", "Memory need", "llm", scope=SYS, unit="GB", min=0.5, max=200, step=0.5,
               apply="llm_start", default=L[slot]["memory_gb"], subgroup=sg))
        add(_f(f"llm.{slot}.min_free_gb", "float", "Needs free", "llm", scope=SYS, unit="GB", min=0, max=200, step=0.5,
               apply="llm_start", default=L[slot]["min_free_gb"], subgroup=sg,
               help="The primary model is used only when at least this much memory is free."))
    for k, lab in (("server", "llama-server binary"), ("host", "Host"), ("port", "Port"), ("args", "Arguments")):
        add(_f(f"llm.{k}", "list_str" if k == "args" else "int" if k == "port" else "string", lab, "llm", scope=SYS, editable=False,
               default=L[k], apply="llm_start", subgroup="Server (file-only)",
               help="Executed by the studio, so it can only be changed by editing settings.json."))
    return f


def _task_fields(core) -> list[dict]:
    out = []
    for tid, t in (core.registry.tasks if core is not None else {}).items():
        out.append(_f(f"defaults.{tid}", "model", t.get("label") or tid, "models", task=tid, nullable=True, default=None,
                      subgroup=t.get("studio") or "other", help=f"Default model for '{tid}'."))
    return out


def _choices(field: dict) -> list[dict] | None:
    src = field.get("choices_from")
    if src == "profiles":
        return [{"value": k, "label": v["label"], "help": v.get("summary", "")} for k, v in profiles.PROFILES.items()]
    if src == "export_presets":
        return [{"value": k, "label": v["label"], "help": v.get("summary", "")} for k, v in profiles.EXPORT_PRESETS.items()]
    if src == "prompt_lang":
        return [{"value": "auto", "label": "Auto (follow what I type)"}, {"value": "en", "label": "English"}, {"value": "th", "label": "Thai"}]
    return None


def fields(core=None) -> list[dict]:
    return _static_fields() + _task_fields(core)


def by_key(core=None) -> dict[str, dict]:
    return {f["key"]: f for f in fields(core)}


def env_info() -> list[dict]:
    """Read-only environment panel. Never a secret: no API keys, tokens or provider variables."""
    def e(key, label, help="", value=None):  # noqa: A002
        v = os.environ.get(key) if value is None else value
        return {"key": key, "value": v if v not in (None, "") else None, "label": label, "help": help}

    return [
        e("STUDIO_HOST", "Bind address", "--host of run_studio.sh (0.0.0.0 = reachable on the LAN, there is no login)"),
        e("STUDIO_PORT", "Port"),
        e("STUDIO_DATA", "Data folder", "projects, uploads, exports", str(config.DATA)),
        e("STUDIO_LOGS", "Logs folder", value=str(config.LOGS)),
        e("STUDIO_SETTINGS", "Settings file", "your system-wide overrides", str(config.SETTINGS_JSON)),
        e("STUDIO_MODELS", "Model registry", value=str(config.MODELS_YAML)),
        e("HF_HOME", "Hugging Face home"),
        e("HF_HUB_CACHE", "Hugging Face cache", "where model weights live", str(config.HF_HUB_CACHE)),
        e("HF_HUB_OFFLINE", "HF offline mode", "1 = the runtime never goes online"),
        e("HF_HUB_DISABLE_XET", "HF Xet disabled"),
        e("LMSTUDIO_MODELS", "LM Studio models", value=str(config.LMSTUDIO_MODELS)),
        e("LLAMA_SERVER", "llama-server path"),
        e("STUDIO_ONLINE", "Online mode", "allows model downloads"),
        e("COMFYUI_URL", "ComfyUI URL"),
        e("STUDIO_NO_COMFYUI", "ComfyUI disabled"),
        e("STUDIO_CLOUD_MOCK", "Cloud mock", "on = cloud calls go to a local mock (tests)", "on" if os.environ.get("STUDIO_CLOUD_MOCK") else "off"),
        e("STUDIO_MCP_API", "MCP API base"),
    ]


def build(core) -> dict:
    flds = []
    for f in fields(core):
        f = dict(f)
        ch = _choices(f)
        if ch is not None:
            f["choices"] = ch
        if f.get("file_check"):                    # does the configured model file exist on this machine?
            val = core.settings.get(*f["key"].split(".")) if core is not None else None
            p = config.resolve_model_file(val) if val else None
            f["file"] = {"found": bool(p and p.exists()), "path": str(p) if p else None} if val else None
        flds.append(f)
    return {"version": 1, "groups": GROUPS, "fields": flds, "env": env_info(),
            "files": {"settings_json": str(config.SETTINGS_JSON), "data": str(config.DATA), "logs": str(config.LOGS),
                      "project_json_template": str(config.DATA / "projects" / "<project>" / "project.json")}}


# ------------------------------------------------------------------ validation
def _num(field: dict, v, key: str):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"'{key}' must be a number.")
    if field["type"] == "int":
        if isinstance(v, float):
            if v != int(v):
                raise ValueError(f"'{key}' must be a whole number.")
            v = int(v)
    else:
        v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"'{key}' must be a finite number.")
    lo, hi = field.get("min"), field.get("max")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"'{key}' must be between {lo} and {hi}{(' ' + field['unit']) if field.get('unit') else ''}.")
    return v


def _check_leaf(field: dict, v, core):
    key, t = field["key"], field["type"]
    if t == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"'{key}' must be true or false.")
        return v
    if t in ("int", "float"):
        return _num(field, v, key)
    if t in ("string", "text"):
        if not isinstance(v, str):
            raise ValueError(f"'{key}' must be text.")
        v = v.strip() if t == "text" else v
        if len(v) > int(field.get("maxlen") or 200):
            raise ValueError(f"'{key}' is longer than {field.get('maxlen') or 200} characters.")
        if field.get("pattern") and not re.fullmatch(field["pattern"], v):
            raise ValueError(f"'{key}' has characters that are not allowed.")
        if field.get("file_check") and not (v.startswith(("lmstudio:", "hf:", "/", "~"))):
            raise ValueError(f"'{key}' must start with lmstudio:, hf: or be an absolute path.")
        if t == "text" and not v and field.get("nullable"):
            return None
        return v
    if t == "choice":
        allowed = [c["value"] for c in _choices(field) or []]
        if v not in allowed:
            raise ValueError(f"Unknown value '{v}' for '{key}' (one of {', '.join(allowed)}).")
        return v
    if t == "model":
        m = core.registry.models.get(v) if core is not None and isinstance(v, str) else None
        if m is None:
            raise ValueError(f"Unknown model '{v}'.")
        if field.get("task") not in m["tasks"]:
            raise ValueError(f"{m['name']} does not support '{field.get('task')}'.")
        return v
    if t == "backend":
        from .textgen import TextGenError, parse_backend

        if not isinstance(v, str):
            raise ValueError(f"'{key}' must be a backend id such as 'local'.")
        try:
            parse_backend(v)
        except TextGenError as e:
            raise ValueError(str(e)) from None
        return v
    if t == "list_str":
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"'{key}' must be a list of text values.")
        return v
    return v        # json: passed through


def validate_patch(patch: dict, scope: str, core) -> dict:
    """Check a nested settings patch against the schema and return it normalised (numbers coerced). Raises ValueError
    (-> HTTP 400). `None` leaves always pass: they reset the setting (system: back to the default; project: inherit)."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope '{scope}'.")
    if not isinstance(patch, dict):
        raise ValueError("A settings patch must be an object.")
    table = by_key(core)
    prefixes: set[str] = set()                  # every proper dotted prefix of a field key ("llm", "llm.primary", ...)
    for k in table:
        parts = k.split(".")
        prefixes |= {".".join(parts[:i]) for i in range(1, len(parts))}
    out: dict = {}

    def put(key: str, value):
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value

    def walk(node: dict, prefix: str):
        for k, v in node.items():
            if not isinstance(k, str) or not k:
                raise ValueError("Setting names must be text.")
            key = f"{prefix}{k}"
            field = table.get(key)
            if field is None:
                if isinstance(v, dict) and key in prefixes:
                    walk(v, key + ".")
                    continue
                raise ValueError(f"Unknown setting '{key}'.")
            if scope not in field["scope"]:
                raise ValueError(f"'{key}' is system-wide and cannot be set for a single project.")
            if not field.get("editable", True):
                raise ValueError(f"'{key}' is file-only: edit settings.json ({config.SETTINGS_JSON}) and restart.")
            put(key, None if v is None else _check_leaf(field, v, core))

    walk(patch, "")
    return out
