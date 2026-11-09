"""The Core object: every service of the studio, wired together once. Plugins get it in `setup(core)`."""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import threading
import time
from pathlib import Path

from . import config
from .assets import AssetStore
from .events import EventBus
from .improver import Improver, LLMSupervisor, Templates
from .jobs import JobManager
from .memory import MemoryManager
from .registry import Registry
from .textgen import TextGen
from .workers import WorkerPool

log = logging.getLogger("studio")
VERSION = "0.1.0"


class Presets:
    """User-saved parameter presets per model: data/presets.json {model_id: {name: params}}."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.path)

    def get(self, model_id: str) -> dict:
        return self.data.get(model_id, {})

    def put(self, model_id: str, name: str, params: dict):
        name = str(name).strip()[:40]
        if not name:
            raise ValueError("Preset name is empty.")
        with self._lock:
            self.data.setdefault(model_id, {})[name] = params
            self._save()

    def delete(self, model_id: str, name: str):
        with self._lock:
            self.data.get(model_id, {}).pop(name, None)
            self._save()


class Core:
    def __init__(self):
        config.ensure_dirs()
        self.started = time.time()
        self.version = VERSION
        self.settings = config.Settings()
        self.bus = EventBus()
        self.registry = Registry(settings=self.settings)
        self.memory = MemoryManager(self.settings)
        self.assets = AssetStore(config.DATA / "projects", on_change=lambda a, m: self.bus.publish("asset", {"action": a, "asset": m}))
        self.pool = WorkerPool(self.registry, self.settings, self.memory)
        self.jobs = JobManager(self.registry, self.assets, self.pool, self.memory, self.bus, self.settings,
                               config.DATA / "jobs.jsonl", config.DATA / "jobs")
        self.llm = LLMSupervisor(self.settings, self.memory, self.bus)
        self.templates = Templates()
        self.textgen = TextGen(self.settings, self.llm, self.registry, core=self)   # local / API / CLI text backends
        self.improver = Improver(self.llm, self.templates, self.registry, textgen=self.textgen)
        self.presets = Presets(config.DATA / "presets.json")
        self.plugins: dict[str, object] = {}
        self.plugin_errors: dict[str, str] = {}
        # plugin hooks for POST /api/improve: fn(project, model_id, task, style_req) -> str (extra context for the improver, "" for none)
        self.improve_context_hooks: list = []

    def settings_for(self, project: str | None):
        """Effective settings for a project (system settings + the project's overrides); the system settings for
        no / an unknown project. Read-only snapshot: re-resolve after the project's settings change."""
        if not project:
            return self.settings
        try:
            return config.SettingsView(self.settings, self.assets.project_settings(project), project)
        except Exception:  # noqa: BLE001  unknown / invalid project id
            return self.settings

    def load_plugins(self, app=None):
        """Import app/plugins/*.py. A plugin may define `setup(core)` and/or `router` (mounted at /api/x/<name>)."""
        from . import plugins as pkg

        # "providers" first: it attaches core.providers (keys, cloud switch, spend) that other plugins may use in setup()
        for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: (i.name != "providers", i.name)):
            if info.name.startswith("_"):
                continue
            try:
                mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
                if hasattr(mod, "setup"):
                    mod.setup(self)
                if app is not None and getattr(mod, "router", None) is not None:
                    app.include_router(mod.router, prefix=f"/api/x/{info.name}", tags=[info.name])
                self.plugins[info.name] = mod
            except Exception as e:  # noqa: BLE001
                log.exception("plugin %s failed", info.name)
                self.plugin_errors[info.name] = f"{type(e).__name__}: {e}"

    def shutdown(self):
        self.jobs.shutdown()
        self.pool.stop_all()
        self.llm.shutdown()
