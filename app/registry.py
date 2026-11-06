"""Model registry: models.yaml (+ models.d/*.yaml drop-ins) -> capabilities + compatibility status.

Status of a model (computed at startup, on demand, and cached):
  ready            weights present, venv present, worker script present, (optional) import probe ok
  missing_weights  one of the `weights` entries is not on disk (fix: the download command)
  missing_env      the backend's python interpreter does not exist (fix: recreate the venv from env-lock/)
  broken           anything else; `code` says what: no_worker | unknown_backend | probe_failed | bad_entry
See docs/CONTRACTS.md section 1 for the schema.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import yaml

from . import config

WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".pkl", ".npz")
STATUSES = ("ready", "missing_key", "cloud_disabled", "missing_weights", "missing_env", "broken")
PARAM_TYPES = ("int", "float", "seed", "choice", "bool", "text", "textarea", "size", "asset", "assets", "lyrics")


def _expand(p: str | None, base: Path = config.STUDIO) -> Path | None:
    if not p:
        return None
    p = os.path.expanduser(os.path.expandvars(str(p)))
    return Path(os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p)))  # no resolve(): keep venv symlinks


class Registry:
    def __init__(self, yaml_path: Path = config.MODELS_YAML, settings: config.Settings | None = None,
                 hf_cache: Path | None = None, lmstudio: Path | None = None, probe_cache: Path | None = None):
        self.yaml_path = Path(yaml_path)
        self.settings = settings
        self.hf_cache = Path(hf_cache or config.HF_HUB_CACHE)
        self.lmstudio = Path(lmstudio or config.LMSTUDIO_MODELS)
        self.probe_cache_path = Path(probe_cache or (config.DATA / "probe_cache.json"))
        self._lock = threading.RLock()
        self.backends: dict[str, dict] = {}
        self.models: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        self.status: dict[str, dict] = {}
        self.probes: dict[str, dict] = {}
        self.errors: list[str] = []
        # extension hooks (set by plugins, e.g. app/plugins/providers.py; all optional):
        self.model_sources: list = []   # fn() -> {tasks, backends, models} merged after the YAML files (generated entries)
        self.model_hooks: list = []     # fn(model) called on every entry after normalisation (may add fields)
        self.status_hooks: dict = {}    # backend kind -> fn(model, backend) -> {status, reason, fix, code, kind} | None
        self._load_probe_cache()
        self.reload()

    # ------------------------------------------------------------------ loading
    def _files(self) -> list[Path]:
        files = [self.yaml_path]
        d = self.yaml_path.parent / "models.d"
        if d.is_dir():
            files += sorted(d.glob("*.yaml"))
        return files

    def reload(self):
        backends, models, tasks, errors = {}, {}, {}, []
        for f in self._files():
            try:
                doc = yaml.safe_load(f.read_text()) or {}
            except (OSError, yaml.YAMLError) as e:
                errors.append(f"{f.name}: {e}")
                continue
            tasks.update(doc.get("tasks") or {})
            for name, b in (doc.get("backends") or {}).items():
                b = dict(b or {})
                b["name"] = name
                backends[name] = b
            for m in doc.get("models") or []:
                if not isinstance(m, dict) or not m.get("id"):
                    errors.append(f"{f.name}: model entry without id")
                    continue
                m = dict(m)
                m["_file"] = f.name
                models[m["id"]] = m
        for src in self.model_sources:
            try:
                doc = src() or {}
            except Exception as e:  # noqa: BLE001
                errors.append(f"model source {getattr(src, '__name__', src)}: {e}")
                continue
            tasks.update({k: v for k, v in (doc.get("tasks") or {}).items() if k not in tasks})
            for name, b in (doc.get("backends") or {}).items():
                backends.setdefault(name, {**b, "name": name})
            for m in doc.get("models") or []:
                if isinstance(m, dict) and m.get("id") and m["id"] not in models:
                    models[m["id"]] = {**m, "_file": "<generated>"}
        for m in models.values():
            self._normalize(m, errors)
            for hook in self.model_hooks:
                try:
                    hook(m)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{m['id']}: model hook failed: {e}")
        with self._lock:
            self.backends, self.models, self.tasks, self.errors = backends, models, tasks, errors
            self.status = {mid: self._compute_status(m) for mid, m in models.items()}

    @staticmethod
    def _normalize(m: dict, errors: list):
        m.setdefault("name", m["id"])
        m.setdefault("tasks", [])
        m.setdefault("features", {})
        m.setdefault("weights", [])
        m.setdefault("params", [])
        m.setdefault("presets", {})
        m.setdefault("license", {"name": "unknown", "commercial": False})
        m.setdefault("memory_gb", 0)
        m.setdefault("tags", [])
        if isinstance(m["tasks"], str):
            m["tasks"] = [m["tasks"]]
        seen = set()
        for p in m["params"]:
            if not isinstance(p, dict) or "name" not in p:
                errors.append(f"{m['id']}: param without name")
                continue
            p.setdefault("type", "text")
            p.setdefault("group", "basic")
            p.setdefault("label", p["name"].replace("_", " ").capitalize())
            if p["type"] not in PARAM_TYPES:
                errors.append(f"{m['id']}.{p['name']}: unknown type {p['type']}")
            if p["name"] in seen:
                errors.append(f"{m['id']}: duplicate param {p['name']}")
            seen.add(p["name"])
        Registry._normalize_slots(m, errors)

    SLOT_TYPES = ("audio", "voice", "video")

    @staticmethod
    def _normalize_slots(m: dict, errors: list):
        """`input_slots` (top level; `features.input_slots` is accepted too): the input pickers the UI shows, one per
        entry {role, label, types: [audio|voice|video], required, hint, tasks?, replaces_prompt?}. Bad entries are dropped."""
        raw = m.get("input_slots", (m.get("features") or {}).get("input_slots"))
        if raw is None:
            return
        out = []
        for i, s in enumerate(raw if isinstance(raw, list) else []):
            if not isinstance(s, dict) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", str(s.get("role", ""))):
                errors.append(f"{m['id']}: input_slots[{i}] needs a role (lowercase letters, digits, _)")
                continue
            types = s.get("types") or ["audio"]
            types = [types] if isinstance(types, str) else list(types)
            if not types or any(t not in Registry.SLOT_TYPES for t in types):
                errors.append(f"{m['id']}: input_slots[{i}] ({s['role']}): types must be from {list(Registry.SLOT_TYPES)}")
                continue
            tasks = s.get("tasks")
            tasks = None if tasks is None else ([tasks] if isinstance(tasks, str) else [str(t) for t in tasks])
            out.append({"role": s["role"], "label": str(s.get("label") or s["role"].replace("_", " ").capitalize()), "types": types,
                        "required": bool(s.get("required", False)), "hint": str(s.get("hint") or ""),
                        **({"tasks": tasks} if tasks else {}), **({"replaces_prompt": True} if s.get("replaces_prompt") else {})})
        m["input_slots"] = out

    # -------------------------------------------------------------- status
    def _check_weight(self, w: dict) -> tuple[bool, str, str]:
        """(present, human description, fix command)."""
        if "hf" in w:
            repo = w["hf"]
            files = w.get("files") or []
            fix = "HF_HUB_DISABLE_XET=1 hf download " + repo + "".join(f" --include '{f}'" for f in files)
            root = self.hf_cache / ("models--" + repo.replace("/", "--"))
            snaps = sorted((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
            if not snaps:
                return False, f"{repo} is not in the Hugging Face cache", fix
            if files:
                missing = [f for f in files if not any(self._glob_ok(s, f) for s in snaps)]
                if missing:
                    return False, f"{repo}: missing {', '.join(missing)}", fix
                return True, repo, ""
            for s in snaps:
                for dirpath, _dirs, names in os.walk(s, followlinks=True):
                    for n in names:
                        if n.endswith(WEIGHT_EXT) and os.path.exists(os.path.join(dirpath, n)):
                            return True, repo, ""
            return False, f"{repo}: no weight files in the cached snapshot", fix
        if "lmstudio" in w:
            p = self.lmstudio / w["lmstudio"]
            return p.exists(), str(p), "Download it in LM Studio (or copy the GGUF to that path)"
        if "path" in w:
            p = _expand(w["path"])
            what = f"{w['label']} not found ({p})" if w.get("label") else str(p)
            return bool(p and p.exists()), what, w.get("fix", f"Put the file at {p}")
        return False, f"unknown weight spec {w}", ""

    @staticmethod
    def _glob_ok(snap: Path, pattern: str) -> bool:
        for hit in glob.glob(str(snap / pattern), recursive=True):
            if os.path.exists(hit):  # resolves the blob symlink: a dangling link means an unfinished download
                return True
        return False

    def backend_paths(self, backend: dict) -> tuple[Path | None, Path | None]:
        return _expand(backend.get("python")), _expand(backend.get("script"))

    def _compute_status(self, m: dict) -> dict:
        def st(status, reason="", fix="", code=""):
            return {"status": status, "reason": reason, "fix": fix, "code": code or status,
                    "checked": time.strftime("%Y-%m-%d %H:%M:%S")}

        blocked = (m.get("features") or {}).get("blocked")
        if blocked:   # known not to work on this machine (gated, needs nvcc, bad output, ...): see the model's research notes
            return st("broken", f"blocked: {blocked}", "See docs/research/audio_more/ for what was tried", "blocked")
        b = self.backends.get(m.get("backend", ""))
        if b is None:
            return st("broken", f"unknown backend '{m.get('backend')}'", "Add it under `backends:` in models.yaml",
                      "unknown_backend")
        hook = self.status_hooks.get(b.get("kind") or "")
        if hook is not None:
            r = hook(m, b)
            if r:
                return {**st(r["status"], r.get("reason", ""), r.get("fix", ""), r.get("code", "")),
                        **{k: v for k, v in r.items() if k not in ("status", "reason", "fix", "code")}}
            return st("ready")
        if b.get("kind") == "cloud":
            return st("broken", "the cloud providers plugin is not loaded", "See plugin_errors in /api/system", "no_plugin")
        if b.get("kind") == "service":
            binary = _expand(b.get("binary"))
            if not binary or not binary.exists():
                return st("missing_env", f"service binary not found: {binary}", b.get("fix", "Build llama.cpp (see prompt_llm.md)"))
            for w in m["weights"]:
                ok, what, fix = self._check_weight(w)
                if not ok:
                    return st("missing_weights", what, fix)
            return st("ready")
        py, script = self.backend_paths(b)
        if not py or not py.exists():
            return st("missing_env", f"python interpreter not found: {py}",
                      b.get("fix") or f"Recreate the venv (see env-lock/{b.get('lock', b['name'])}.txt and README)")
        for w in m["weights"]:
            ok, what, fix = self._check_weight(w)
            if not ok:
                r = st("missing_weights", what, fix)
                r["downloadable"] = any(("hf" in x or x.get("download_cmd")) and not self._check_weight(x)[0] for x in m["weights"])
                return r
        if not script or not script.exists():
            rel = os.path.relpath(script, config.STUDIO) if script else "?"
            return st("broken", f"worker not implemented yet: {rel}",
                      "Add the backend worker (docs/CONTRACTS.md section 2)", "no_worker")
        pr = self.probes.get(m["id"])
        if pr and pr.get("stamp") == self._env_stamp(b) and not pr.get("ok"):
            return st("broken", pr.get("reason", "import probe failed"), pr.get("fix", "See logs/probe.log"), "probe_failed")
        return st("ready")

    # ---------------------------------------------------------------- probe
    def _env_stamp(self, b: dict) -> str:
        py, script = self.backend_paths(b)
        parts = []
        for p in (py.parent.parent / "pyvenv.cfg" if py else None, script):
            try:
                parts.append(str(int(p.stat().st_mtime)) if p else "-")
            except OSError:
                parts.append("-")
        return ":".join(parts)

    def _load_probe_cache(self):
        try:
            self.probes = json.loads(self.probe_cache_path.read_text())
        except (OSError, ValueError):
            self.probes = {}

    def probe_backend(self, name: str, timeout: float = 240.0) -> dict:
        """Run `<python> <script> --probe` (imports the backend in its venv, asks it about each of its models).
        Slow (imports torch); called from the Models page, never on the request path."""
        b = self.backends.get(name)
        if not b:
            raise KeyError(name)
        py, script = self.backend_paths(b)
        mods = [m for m in self.models.values() if m.get("backend") == name]
        stamp = self._env_stamp(b)
        results: dict = {}
        if not (py and py.exists() and script and script.exists()):
            return {}
        env = worker_env(b)
        try:
            out = subprocess.run([str(py), "-u", str(script), "--probe"], input=json.dumps(mods, default=str),
                                 capture_output=True, text=True, timeout=timeout, env=env, cwd=str(config.STUDIO))
            (config.LOGS / "probe.log").open("a").write(f"--- {time.ctime()} {name}\n{out.stderr[-4000:]}\n")
            line = next((ln for ln in reversed(out.stdout.splitlines()) if ln.startswith("{")), "{}")
            per = json.loads(line).get("probe", {})
            for m in mods:
                r = per.get(m["id"]) or {"ok": out.returncode == 0,
                                         "reason": (out.stderr.strip().splitlines() or ["probe produced no answer"])[-1][:300]}
                results[m["id"]] = {**r, "stamp": stamp, "at": time.time()}
        except subprocess.TimeoutExpired:
            for m in mods:
                results[m["id"]] = {"ok": False, "reason": f"probe timed out after {timeout:.0f}s", "stamp": stamp}
        with self._lock:
            self.probes.update(results)
            try:
                self.probe_cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.probe_cache_path.write_text(json.dumps(self.probes, indent=1))
            except OSError:
                pass
            for m in mods:
                self.status[m["id"]] = self._compute_status(m)
        return results

    def refresh(self):
        self.reload()

    def recompute(self, pred=None):
        """Recompute the status of the models matching pred(model) (all when None), e.g. after an API key change."""
        with self._lock:
            for mid, m in self.models.items():
                if pred is None or pred(m):
                    self.status[mid] = self._compute_status(m)

    # ------------------------------------------------------------------- query
    def get(self, model_id: str) -> dict:
        m = self.models.get(model_id)
        if m is None:
            raise KeyError(model_id)
        return m

    def backend_of(self, model_id: str) -> dict:
        return self.backends[self.get(model_id)["backend"]]

    def is_gpu(self, model_id: str) -> bool:
        m = self.get(model_id)
        dev = m.get("device") or self.backend_of(model_id).get("device", "gpu")
        return dev != "cpu"

    def backend_public(self, b: dict) -> dict:
        if b.get("kind") == "service":
            binary = _expand(b.get("binary"))
            return {"name": b["name"], "label": b.get("label", b["name"]), "kind": "service", "device": b.get("device", "gpu"),
                    "python": str(binary), "env_ok": bool(binary and binary.exists()), "worker_ok": True,
                    "models": [m["id"] for m in self.models.values() if m.get("backend") == b["name"]]}
        py, script = self.backend_paths(b)
        pr = [self.probes.get(m["id"]) for m in self.models.values() if m.get("backend") == b["name"]]
        pr = [p for p in pr if p]
        return {"name": b["name"], "label": b.get("label", b["name"]), "device": b.get("device", "gpu"),
                "python": str(py), "script": os.path.relpath(script, config.STUDIO) if script else None,
                "env_ok": bool(py and py.exists()), "worker_ok": bool(script and script.exists()),
                "probed": bool(pr), "probe_ok": all(p.get("ok") for p in pr) if pr else None, "lock": b.get("lock"),
                "models": [m["id"] for m in self.models.values() if m.get("backend") == b["name"]]}

    def prompt_style(self, model_id: str, task: str | None = None) -> str | None:
        ps = self.get(model_id).get("prompt_style")
        if isinstance(ps, dict):
            return ps.get(task) if task in ps else next(iter(ps.values()), None)
        return ps

    def public(self, m: dict) -> dict:
        out = {k: v for k, v in m.items() if not k.startswith("_")}
        out["status"] = self.status.get(m["id"], {"status": "broken", "reason": "not computed"})
        b = self.backends.get(m.get("backend", ""), {})
        out["device"] = m.get("device") or b.get("device", "gpu")
        out["env"] = b.get("label") or b.get("name")
        return out

    def list(self, task: str | None = None, modality: str | None = None, include_test: bool = True) -> list[dict]:
        out = []
        for m in self.models.values():
            if task and task not in m["tasks"]:
                continue
            if modality and m.get("modality") != modality:
                continue
            if not include_test and m.get("test"):
                continue
            out.append(self.public(m))
        order = {s: i for i, s in enumerate(STATUSES)}
        out.sort(key=lambda m: (m.get("test", False), order.get(m["status"]["status"], 9), m.get("order", 50), m["name"]))
        return out

    def default_for(self, task: str) -> str | None:
        if self.settings:
            d = (self.settings.get("defaults") or {}).get(task)
            if d in self.models and task in self.models[d]["tasks"]:
                return d
        cands = [m for m in self.models.values() if task in m["tasks"] and not m.get("test")]
        # local models first: a paid cloud model only becomes the implicit default when nothing local can do the task
        cands.sort(key=lambda m: self.backends.get(m.get("backend", ""), {}).get("kind") == "cloud")
        for m in cands:
            if task in (m.get("default_for") or []) and self.status.get(m["id"], {}).get("status") == "ready":
                return m["id"]
        for m in cands:
            if self.status.get(m["id"], {}).get("status") == "ready":
                return m["id"]
        for m in cands:
            if task in (m.get("default_for") or []):
                return m["id"]
        return cands[0]["id"] if cands else None

    def validate_params(self, model_id: str, params: dict) -> dict:
        """Coerce/clip user params against the model schema. Unknown keys are kept (workers may read extra keys such
        as `prompt`, `negative`, `batch`), declared ones are typed and range-checked."""
        m = self.get(model_id)
        out = dict(params or {})
        for p in m["params"]:
            n, t = p["name"], p["type"]
            if n not in out or out[n] is None or out[n] == "":
                if "default" in p:
                    out[n] = p["default"]
                continue
            v = out[n]
            try:
                if t == "int":
                    v = int(round(float(v)))
                elif t == "float":
                    v = float(v)
                elif t == "seed":
                    v = int(v)
                    if v < -1 or v > 2**32 - 1:
                        raise ValueError("seed out of range")
                elif t == "bool":
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif t == "choice":
                    choices = [c["value"] if isinstance(c, dict) else c for c in p.get("choices", [])]
                    if v not in choices:
                        alt = next((c for c in choices if str(c) == str(v)), None)
                        if alt is None:
                            raise ValueError(f"must be one of {choices}")
                        v = alt
                elif t in ("text", "textarea", "lyrics"):
                    v = str(v)
                    if p.get("max_length") and len(v) > int(p["max_length"]):
                        raise ValueError(f"longer than {p['max_length']} characters")
            except (TypeError, ValueError) as e:
                raise ValueError(f"{p['label']}: {e}") from None
            if t in ("int", "float"):
                if "min" in p and v < p["min"]:
                    v = type(v)(p["min"])
                if "max" in p and v > p["max"]:
                    v = type(v)(p["max"])
            out[n] = v
        return out


def worker_env(backend: dict) -> dict:
    """Environment of a backend subprocess: offline Hub, no user site, the backend's own extra variables."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_XET="1", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               TOKENIZERS_PARALLELISM="false", STUDIO_ROOT=str(config.STUDIO))
    if os.environ.get("STUDIO_ONLINE"):
        env.pop("HF_HUB_OFFLINE", None)
    if backend.get("max_loaded") is not None:
        env["STUDIO_MAX_LOADED"] = str(backend["max_loaded"])
    for k, v in (backend.get("env") or {}).items():
        v = os.path.expanduser(os.path.expandvars(str(v))).replace("{studio}", str(config.STUDIO))
        env[k] = v
    # the worker framework lives in workers; backend scripts import it from there
    pp = [str(config.WORKERS)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env
