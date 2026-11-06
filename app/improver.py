"""Prompt improver: a local llama-server (started lazily, sleeps / stops when idle) + per-model templates.

Lifecycle (docs/research/prompt_llm.md):
  * lazy start on the first request: spawn llama-server on 127.0.0.1:7880 (own session, PDEATHSIG, log file),
    poll /health; the primary model needs >= 25 GB free, otherwise the 2.3 GB fallback is used;
  * --sleep-idle-seconds makes llama-server drop its weights after 5 min idle (the process keeps the port and
    wakes on the next request); the supervisor stops the process after 30 min idle;
  * the memory manager can evict it at any time (it is registered as kind "llm", always evicted first);
  * a file lock keeps two studio instances from both spawning one; a server owned by another instance is used,
    never killed.
Answers are JSON-schema constrained; cheap content checks run on top, with one retry at a higher temperature.
Optional: `improve(..., backend="cli:claude" | "api:<provider>/<model>")` runs the same templates + JSON contract on
an external text backend through app/textgen.py (default: settings `textgen.improve`, else local).
"""
from __future__ import annotations

import base64
import fcntl
import io
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import httpx
import yaml

from . import config, procs
from .memory import MemoryManager

THAI = re.compile(r"[฀-๿]")
LABEL_PREFIX = re.compile(r"^\s*(prompt|variation\s*\d*|negative)\s*[:\-]\s*", re.I)
LYRIC_MARK = re.compile(r"^\s*\[(intro|verse|pre-chorus|chorus|post-chorus|bridge|outro|inst|instrumental|solo)\]\s*$", re.I | re.M)
LOCK_PATH = Path("/tmp/studio-prompt-llm.lock")
MEM_KEY = "llm:prompt"


class ImproveError(Exception):
    def __init__(self, msg, status=503, fix=""):
        super().__init__(msg)
        self.status, self.fix = status, fix


def detect_lang(text: str) -> str:
    return "th" if THAI.search(text or "") else "en"


# ===================================================================== templates
class Templates:
    def __init__(self, folder: Path = config.PROMPTS):
        self.folder = Path(folder)
        self.reload()

    def reload(self):
        self.preamble, self.styles, self.chipsets = "", {}, {}
        files = sorted(self.folder.glob("*.yaml"), key=lambda p: (p.name != "core.yaml", p.name))
        for f in files:
            try:
                doc = yaml.safe_load(f.read_text()) or {}
            except (OSError, yaml.YAMLError):
                continue
            self.preamble = doc.get("preamble") or self.preamble
            self.chipsets.update(doc.get("chipsets") or {})
            for k, v in (doc.get("styles") or {}).items():
                self.styles[k] = {**self.styles.get(k, {}), **v, "id": k}

    def get(self, style: str) -> dict:
        if style not in self.styles:
            raise ImproveError(f"Unknown prompt style '{style}'.", 400)
        return self.styles[style]

    def public(self) -> list[dict]:
        return [{"id": k, "label": v.get("label", k), "modality": v.get("modality"), "chips": v.get("chips") or [],
                 "apply": v.get("apply") or {"prompt": "prompt"}, "extra": v.get("extra") or [],
                 "negative": bool(v.get("negative"))} for k, v in self.styles.items()]


def build_schema(style: dict, n: int) -> dict:
    props = {"prompt": {"type": "string", "minLength": 3}, "negative": {"type": "string"},
             "variations": {"type": "array", "items": {"type": "string", "minLength": 3}, "minItems": n, "maxItems": n},
             "notes": {"type": "string"}}
    req = ["prompt", "negative", "variations", "notes"]
    for x in style.get("extra") or []:
        props[x] = {"type": "string"}
        req.append(x)
    return {"type": "object", "additionalProperties": False, "required": req, "properties": props}


def check_answer(ans: dict, style: dict, lang: str, n: int) -> list[str]:
    """Content checks on top of the schema (names are stable: the UI shows them as warnings)."""
    fails = []
    eng = [ans.get("prompt", ""), ans.get("negative", "")]
    if not style.get("user_lang_variations"):
        eng += list(ans.get("variations") or [])
    if any(THAI.search(s or "") for s in eng):
        fails.append("thai_in_english_field")
    vs = [v.strip().lower() for v in ans.get("variations") or []]
    if n and (len(vs) != n or len(set(vs)) != len(vs) or (ans.get("prompt", "").strip().lower() in vs)):
        fails.append("variations_not_distinct")
    if any(LABEL_PREFIX.match(s or "") for s in [ans.get("prompt", "")] + list(ans.get("variations") or [])):
        fails.append("label_prefix")
    if "lyrics" in (style.get("extra") or []) and "lyrics_markers" in (style.get("checks") or []):
        if not LYRIC_MARK.search(ans.get("lyrics") or ""):
            fails.append("lyrics_without_markers")
    notes = ans.get("notes") or ""
    if notes:
        if lang == "th" and not THAI.search(notes):
            fails.append("notes_not_thai")
        if lang == "en" and THAI.search(notes):
            fails.append("thai_for_en_user")
    if "line" in (style.get("extra") or []) and not (ans.get("line") or "").strip():
        fails.append("empty_line")
    return fails


def _clean(ans: dict) -> dict:
    def c(s):
        return LABEL_PREFIX.sub("", str(s or "")).strip().strip('"').strip()
    out = dict(ans)
    out["prompt"] = c(ans.get("prompt"))
    out["variations"] = [c(v) for v in ans.get("variations") or []]
    if "lyrics" in out:
        out["lyrics"] = str(out["lyrics"]).replace("\\n", "\n").strip()
    return out


# ===================================================================== supervisor
class LLMSupervisor:
    def __init__(self, settings: config.Settings, memory: MemoryManager | None = None, bus=None):
        self.settings = settings
        self.memory = memory
        self.bus = bus
        self.proc: subprocess.Popen | None = None
        self.which: str | None = None
        self.state = "stopped"       # stopped | starting | ready | sleeping | failed | external
        self.error = ""
        self.started_at: float | None = None
        self.last_request = 0.0
        self._lock = threading.Lock()
        self._lockfile = None
        self._stop = False
        threading.Thread(target=self._idle_loop, daemon=True, name="llm-idle").start()

    @property
    def cfg(self) -> dict:
        return self.settings.get("llm") or {}

    @property
    def url(self) -> str:
        return f"http://{self.cfg.get('host', '127.0.0.1')}:{int(self.cfg.get('port', 7880))}"

    def _publish(self):
        if self.bus:
            self.bus.publish("llm", self.status())

    def status(self) -> dict:
        c = self.cfg
        model = c.get(self.which) if self.which else None
        return {"state": self.state, "model": (model or {}).get("name"), "which": self.which, "url": self.url,
                "pid": self.proc.pid if self.proc and self.proc.poll() is None else None, "error": self.error,
                "vision": bool(model and model.get("mmproj")), "idle_s": round(time.time() - self.last_request, 0) if self.last_request else None,
                "primary": c.get("primary", {}).get("name"), "fallback": c.get("fallback", {}).get("name"),
                "available": self._files_ok("primary") or self._files_ok("fallback")}

    def _files_ok(self, which: str) -> bool:
        m = self.cfg.get(which) or {}
        mp = config.resolve_model_file(m.get("model"))
        server = Path(self.cfg.get("server", ""))
        if not (mp and mp.exists() and server.exists()):
            return False
        mm = m.get("mmproj")
        return not mm or bool(config.resolve_model_file(mm) and config.resolve_model_file(mm).exists())

    def _health(self, timeout=1.0) -> bool:
        try:
            r = httpx.get(self.url + "/health", timeout=timeout)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self):
        with self._lock:
            if self._alive() and self.state in ("ready", "sleeping"):
                return
            if self.state == "external" and self._health():
                return
            if not self._alive() and self._health():  # someone else's (another studio instance): use, never kill
                self.state, self.which, self.error = "external", None, ""
                self._publish()
                return
            self._spawn()

    def _choose(self) -> str:
        avail = self.memory.meminfo()[0] if self.memory else 999
        c = self.cfg
        if self._files_ok("primary") and avail >= float(c["primary"].get("min_free_gb", 25)):
            return "primary"
        if self._files_ok("fallback") and avail >= float(c["fallback"].get("min_free_gb", 4)):
            return "fallback"
        if not (self._files_ok("primary") or self._files_ok("fallback")):
            raise ImproveError("No prompt LLM is installed (llama-server or the GGUF files are missing).",
                               fix="See docs/research/prompt_llm.md for the model files.")
        raise ImproveError(f"Not enough free memory for the prompt LLM ({avail:.0f} GB free).",
                           fix="Unload a model on the Queue & System page and try again.")

    def _spawn(self):
        c = self.cfg
        try:
            LOCK_PATH.touch(exist_ok=True)
            self._lockfile = open(LOCK_PATH, "r+")  # noqa: SIM115
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ImproveError("Another studio instance is starting the prompt LLM; try again in a few seconds.") from None
        which = self._choose()
        m = c[which]
        cmd = [c["server"], "-m", str(config.resolve_model_file(m["model"]))]
        if m.get("mmproj"):
            cmd += ["--mmproj", str(config.resolve_model_file(m["mmproj"]))]
        cmd += ["--host", c.get("host", "127.0.0.1"), "--port", str(int(c.get("port", 7880))),
                "--sleep-idle-seconds", str(int(c.get("sleep_idle_s", 300)))] + list(c.get("args") or [])
        log = (config.LOGS / "prompt_llm.log").open("a")
        log.write(f"\n=== {time.ctime()} starting {m['name']}: {' '.join(cmd)}\n")
        log.flush()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
        env.update(HF_HUB_OFFLINE="1", CUDA_CACHE_MAXSIZE="4294967296")
        self.state, self.which, self.error = "starting", which, ""
        self.last_request = time.time()
        self._publish()
        self.proc = procs.spawn(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)
        self.started_at = time.time()
        deadline = time.time() + float(c.get("start_timeout_s", 180))
        proc = self.proc
        while time.time() < deadline:
            if proc.poll() is not None:
                self.state, self.error = "failed", f"llama-server exited with code {proc.returncode} (see logs/prompt_llm.log)"
                self._unlock()
                self._publish()
                raise ImproveError(self.error)
            if self._health(0.5):
                break
            time.sleep(0.5)
        else:
            self._kill()
            self.state, self.error = "failed", "llama-server did not become healthy in time"
            self._publish()
            raise ImproveError(self.error)
        self.state = "ready"
        self.last_request = time.time()
        if self.memory:
            self.memory.register(MEM_KEY, float(m.get("memory_gb", 20)), kind="llm", label=f"Prompt LLM ({m['name']})",
                                 evict=self.release)
        self._publish()

    def _unlock(self):
        if self._lockfile:
            try:
                fcntl.flock(self._lockfile, fcntl.LOCK_UN)
                self._lockfile.close()
            except OSError:
                pass
            self._lockfile = None

    def _kill(self):
        p = self.proc
        if p is not None and p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        self.proc = None
        self._unlock()

    def release(self):
        """Stop the server (memory manager eviction, idle timeout, app shutdown). Only ever our own process."""
        self._kill()
        if self.memory:
            self.memory.unregister(MEM_KEY)
        if self.state != "external":
            self.state = "stopped"
        self.which = None
        self._publish()

    def _idle_loop(self):
        while not self._stop:
            time.sleep(10)
            try:
                if not self._alive():
                    if self.state in ("ready", "sleeping"):
                        self.state, self.error = "failed", "llama-server exited (see logs/prompt_llm.log)"
                        if self.memory:
                            self.memory.unregister(MEM_KEY)
                        self._unlock()
                        self._publish()
                    continue
                if self.state in ("ready", "sleeping") and self.last_request and \
                        time.time() - self.last_request > float(self.cfg.get("kill_idle_s", 1800)):
                    self.release()
                    continue
                try:
                    props = httpx.get(self.url + "/props", timeout=2).json()
                    sleeping = bool(props.get("is_sleeping"))
                except (httpx.HTTPError, ValueError):
                    continue
                st = "sleeping" if sleeping else "ready"
                if st != self.state:
                    self.state = st
                    if self.memory and self.which:
                        gb = 0.3 if sleeping else float(self.cfg[self.which].get("memory_gb", 20))
                        self.memory.set_gb(MEM_KEY, gb)
                    self._publish()
            except Exception:  # noqa: BLE001
                pass

    def chat(self, messages: list, schema: dict, temperature: float, max_tokens: int = 1500) -> dict:
        self.ensure()
        self.last_request = time.time()
        if self.state == "sleeping" and self.memory and self.which:
            self.memory.set_gb(MEM_KEY, float(self.cfg[self.which].get("memory_gb", 20)))
            self.state = "ready"
            self._publish()
        body = {"messages": messages, "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "improve", "strict": True, "schema": schema}}}
        try:
            r = httpx.post(self.url + "/v1/chat/completions", json=body, timeout=240)
        except httpx.HTTPError as e:
            raise ImproveError(f"The prompt LLM did not answer: {e}") from None
        finally:
            self.last_request = time.time()
        if r.status_code != 200:
            raise ImproveError(f"The prompt LLM returned HTTP {r.status_code}: {r.text[:300]}")
        try:
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, ValueError, IndexError) as e:
            raise ImproveError(f"The prompt LLM returned an unreadable answer ({e}).") from None

    def has_vision(self) -> bool:
        return bool(self.which and self.cfg.get(self.which, {}).get("mmproj"))

    def shutdown(self):
        self._stop = True
        if self._alive():
            self.release()


# ===================================================================== service
class Improver:
    def __init__(self, llm: LLMSupervisor, templates: Templates, registry=None, textgen=None):
        self.llm = llm
        self.templates = templates
        self.registry = registry
        self.textgen = textgen   # app.textgen.TextGen: optional API / CLI backends (None -> local only)

    def _backend(self, backend: str | None) -> dict | None:
        """None for the classic local path; else a resolved textgen spec. `backend` None -> settings textgen.improve."""
        if self.textgen is None:
            return None
        from .textgen import TextGenError
        try:
            spec = self.textgen.resolve(backend, "improve") if backend else self.textgen.resolve(self.textgen.default_for("improve"), "improve")
        except TextGenError as e:
            raise ImproveError(str(e), 400, e.fix) from None
        return None if spec["kind"] == "local" else spec

    def _chat(self, spec: dict | None, messages: list, schema: dict, temperature: float) -> tuple[dict, dict]:
        if spec is None:
            return self.llm.chat(messages, schema, temperature), {}
        from .textgen import TextGenError
        try:
            r = self.textgen.generate(messages=messages, schema=schema, backend=spec, purpose="improve", temperature=temperature, max_tokens=1500)
        except TextGenError as e:
            raise ImproveError(str(e), e.status if e.status < 600 else 502, e.fix) from None
        if not isinstance(r.get("data"), dict):
            raise ImproveError("The text backend returned no usable answer.", 502, "Try again or pick another backend.")
        return r["data"], r

    def resolve_style(self, target_style: str | None, model_id: str | None, has_image: bool, text: str) -> tuple[str, dict | None]:
        model = None
        if model_id and self.registry and model_id in self.registry.models:
            model = self.registry.models[model_id]
        if target_style:
            return target_style, model
        if has_image and not text.strip():
            return "describe", model
        if model and model.get("prompt_style"):
            return model["prompt_style"], model
        raise ImproveError("Pick a target style or a model to improve the prompt for.", 400)

    @staticmethod
    def _image_data_url(path: Path) -> str:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
            im.thumbnail((768, 768))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def improve(self, text: str, *, target_style: str | None = None, model_id: str | None = None, lang: str | None = None,
                n: int = 3, image_path: Path | None = None, context: str = "", chips: list[str] | None = None,
                backend: str | None = None) -> dict:
        text = (text or "").strip()
        if not text and not image_path:
            raise ImproveError("Write an idea first (Thai or English).", 400)
        if len(text) > 4000:
            raise ImproveError("The idea is too long (4000 characters max).", 400)
        style_id, model = self.resolve_style(target_style, model_id, image_path is not None, text)
        style = self.templates.get(style_id)
        n = max(0, min(6, int(style.get("n", n) if style.get("n") is not None else n)))
        lang = lang if lang in ("th", "en") else detect_lang(text)
        hints = []
        chip_map = {c["id"]: c for c in style.get("chips") or []}
        for c in chips or []:
            if c in chip_map:
                hints.append(chip_map[c]["hint"])
        ctx_parts = [p for p in (context.strip(), "; ".join(hints)) if p]
        if model:
            ctx_parts.append(f"target model: {model['name']}")
        if not style.get("negative"):
            ctx_parts.append('this model takes no negative prompt: return "negative": ""')
        user = (f"IDEA: {text or '(describe the attached image)'}\nUSER_LANG: {lang}\nN_VARIATIONS: {n}\n"
                f"CONTEXT: {' | '.join(ctx_parts) or '-'}")
        system = (self.templates.preamble or "") + "\n" + style["system"]
        spec = self._backend(backend)
        if spec is None:
            self.llm.ensure()
        warnings = []
        content: list | str = user
        if image_path is not None:
            vision = (self.llm.has_vision() or self.llm.state == "external") if spec is None else \
                next((b.get("vision") for b in self.textgen.backends() if b["id"] == spec["id"].split("@")[0]), True)
            if vision:
                content = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": self._image_data_url(image_path)}}]
            else:
                warnings.append("The selected text backend has no vision; the image was ignored.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        schema = build_schema(style, n)
        t0 = time.time()
        temp = float(self.llm.cfg.get("temperature", 0.7))
        best, best_fails, attempts, meta = None, None, 0, {}
        for attempt in range(2):
            attempts += 1
            raw, meta = self._chat(spec, messages, schema, temp + 0.25 * attempt)
            ans = _clean(raw)
            fails = check_answer(ans, style, lang, n)
            if best is None or len(fails) < len(best_fails):
                best, best_fails = ans, fails
            if not fails or (spec is not None and spec["kind"] == "cli"):  # CLI calls cost plan quota: no content retry
                break
        out = {"prompt": best.get("prompt", ""), "negative": best.get("negative", "") if style.get("negative") else "",
               "variations": best.get("variations") or [], "notes": best.get("notes", ""), "style": style_id,
               "style_label": style.get("label", style_id), "apply": style.get("apply") or {"prompt": "prompt"},
               "lang": lang, "translated": lang == "th",
               "llm": (spec["label"] if spec else self.llm.status().get("model") or "external llama-server"),
               "backend": (meta.get("backend") if spec else {"id": "local", "kind": "local", "label": "Local LLM", "subscription": False}),
               "seconds": round(time.time() - t0, 2), "warnings": warnings + best_fails, "attempts": attempts,
               "original": text}
        for x in style.get("extra") or []:
            out[x] = best.get(x, "")
        return out
