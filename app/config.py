"""Paths and user settings of the studio core.

Everything is relative to the studio root (the folder that holds `models.yaml`). Tests and second instances can
point the data / settings elsewhere with STUDIO_DATA, STUDIO_SETTINGS, STUDIO_MODELS and STUDIO_LOGS.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import threading
from pathlib import Path

STUDIO = Path(__file__).resolve().parent.parent
APP = STUDIO / "app"
WEB = APP / "web"
WORKERS = STUDIO / "workers"
PROMPTS = STUDIO / "prompts"
DATA = Path(os.environ.get("STUDIO_DATA", STUDIO / "data")).resolve()
LOGS = Path(os.environ.get("STUDIO_LOGS", STUDIO / "logs")).resolve()
MODELS_YAML = Path(os.environ.get("STUDIO_MODELS", STUDIO / "models.yaml")).resolve()
SETTINGS_JSON = Path(os.environ.get("STUDIO_SETTINGS", STUDIO / "settings.json")).resolve()
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
LMSTUDIO_MODELS = Path(os.environ.get("LMSTUDIO_MODELS", Path.home() / ".lmstudio/models"))

DEFAULT_SETTINGS: dict = {
    "defaults": {},                 # task -> model id (Models page "set default")
    "memory": {
        "margin_gb": 6.0,           # kept free on top of a model's memory_gb before a job starts
        "llm_yield_gb": 40.0,       # jobs whose model needs at least this always stop the prompt LLM first
        "evict_wait_s": 12.0,       # how long to wait for memory to come back after an unload
    },
    "workers": {
        "model_idle_s": 600,        # a worker unloads a model that was not used for this long
        "exit_idle_s": 900,         # ... and exits when it holds no model for this long
        "cancel_grace_s": 20,       # running job ignores cancel this long -> worker is killed (if the backend allows)
    },
    "jobs": {"cpu_concurrency": 3, "history_keep": 2000},
    "llm": {
        "server": os.environ.get("LLAMA_SERVER") or shutil.which("llama-server") or str(Path.home() / "llama.cpp/build/bin/llama-server"),
        "host": "127.0.0.1",
        "port": 7880,
        "primary": {
            "name": "Qwen3.6-35B-A3B (Q4_K_M, vision)",
            "model": "lmstudio:lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "mmproj": "lmstudio:lmstudio-community/Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf",
            "memory_gb": 22.0,
            "min_free_gb": 25.0,
        },
        "fallback": {
            "name": "gemma-4-E2B (UD-Q4_K_XL)",
            "model": "hf:unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-UD-Q4_K_XL.gguf",
            "mmproj": None,
            "memory_gb": 2.5,
            "min_free_gb": 4.0,
        },
        "args": ["-ngl", "999", "-fa", "on", "--jinja", "-c", "16384", "-np", "2", "--reasoning", "off",
                 "--no-webui", "--alias", "prompt-llm"],
        "sleep_idle_s": 300,        # llama-server frees its weights itself (process stays, wakes on next request)
        "kill_idle_s": 1800,        # the supervisor stops the process after this long without a request
        "start_timeout_s": 180,
        "temperature": 0.7,
    },
    "ui": {"show_test_models": False},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Settings:
    """settings.json merged over DEFAULT_SETTINGS. Writes are atomic; only the user's overrides are stored."""

    def __init__(self, path: Path = SETTINGS_JSON):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._user: dict = {}
        self.load()

    def load(self):
        try:
            self._user = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self._user = {}
        self.data = _merge(DEFAULT_SETTINGS, self._user)

    def get(self, *keys, default=None):
        d = self.data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def update(self, patch: dict) -> dict:
        """Deep-merge `patch` into the stored overrides (a None value removes a key)."""
        with self._lock:
            def apply(dst: dict, src: dict):
                for k, v in src.items():
                    if v is None:
                        dst.pop(k, None)
                    elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                        apply(dst[k], v)
                    else:
                        dst[k] = copy.deepcopy(v)
            apply(self._user, patch)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._user, indent=2, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
            self.data = _merge(DEFAULT_SETTINGS, self._user)
            return self.data


def resolve_model_file(spec: str | None) -> Path | None:
    """'lmstudio:<pub>/<repo>/<file>' | 'hf:<org>/<repo>/<file in snapshot>' | absolute path  ->  local Path (may not exist)."""
    if not spec:
        return None
    if spec.startswith("lmstudio:"):
        return LMSTUDIO_MODELS / spec.split(":", 1)[1]
    if spec.startswith("hf:"):
        org, repo, *rest = spec.split(":", 1)[1].split("/")
        base = HF_HUB_CACHE / f"models--{org}--{repo}" / "snapshots"
        if base.is_dir():
            for snap in sorted(base.iterdir(), reverse=True):
                p = snap.joinpath(*rest)
                if p.exists():
                    return p
        return base / "<missing>" / "/".join(rest)
    return Path(os.path.expanduser(spec))


def ensure_dirs():
    for d in (DATA, LOGS, DATA / "projects", DATA / "uploads", DATA / "exports"):
        d.mkdir(parents=True, exist_ok=True)
