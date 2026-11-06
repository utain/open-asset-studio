"""Cloud providers: API keys, the global "Allow cloud" switch, prices, spend control, subscription-CLI detection.

`core.providers` (an instance of Providers) is created by app/plugins/providers.py at startup. Interface used by
other parts of the studio (docs/CONTRACTS.md section 8):

    core.providers.get_key(provider_id) -> str | None        secrets file first, then the env var (FAL_KEY, ...)
    core.providers.cloud_allowed() -> bool                    the global switch (default OFF)
    core.providers.get_base_url(provider_id) -> str | None    OpenAI-compatible base URL (openai, openrouter, gemini,
                                                              openai_compat:<name>) for chat/completions
    core.providers.spend.check(est_usd, provider_id, *, confirmed=False)   raises ProviderError kind budget | confirm
                                                              (| cloud_disabled)
    core.providers.spend.record(usd, provider_id, model_id, project, job_id, *, estimated=False)

Files (all under STUDIO_DATA):
    secrets.json    {"keys": {provider_id: key}}   chmod 600, never sent to the browser
    providers.json  non-secret state: allow_cloud, budget_usd, confirm_usd, price_overrides, endpoints, upload_ack
    spend.jsonl     one line per paid call: {ts, month, usd, provider, model, project, job, estimated}
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from . import config
from .jobs import JobError

# ------------------------------------------------------------------ catalog
# base: named base URLs per provider ("api" is the main one). terms: shown on assets made with this provider.
PROVIDERS: dict[str, dict] = {
    "fal": {
        "label": "fal.ai", "env": ["FAL_KEY", "FAL_API_KEY"], "auth": ("Authorization", "Key {key}"),
        "base": {"queue": "https://queue.fal.run", "api": "https://api.fal.ai/v1"},
        "modalities": ["image", "mesh", "audio"],
        "keys_url": "https://fal.ai/dashboard/keys", "pricing_url": "https://fal.ai/pricing", "terms_url": "https://fal.ai/terms",
        "terms": "fal's terms let you use outputs commercially, but each hosted model keeps its own licence "
                 "(shown on the model): check it before shipping.",
        "test": "Prices one endpoint via the Platform API (free).",
    },
    "replicate": {
        "label": "Replicate", "env": ["REPLICATE_API_TOKEN", "REPLICATE_API_KEY"], "auth": ("Authorization", "Bearer {key}"),
        "base": {"api": "https://api.replicate.com/v1"}, "modalities": ["image", "mesh", "audio"],
        "keys_url": "https://replicate.com/account/api-tokens", "pricing_url": "https://replicate.com/pricing",
        "terms_url": "https://replicate.com/terms",
        "terms": "Replicate claims no rights in outputs; each model's own licence applies (some community models "
                 "are non-commercial).",
        "test": "Reads the account type (GET /account, free).",
    },
    "openrouter": {
        "label": "OpenRouter", "env": ["OPENROUTER_API_KEY"], "auth": ("Authorization", "Bearer {key}"),
        "base": {"api": "https://openrouter.ai/api/v1"}, "modalities": ["text", "image"],
        "keys_url": "https://openrouter.ai/settings/keys", "pricing_url": "https://openrouter.ai/models",
        "terms_url": "https://openrouter.ai/terms",
        "terms": "OpenRouter passes your request to the model's provider; that provider's terms decide output use "
                 "(generally yours to use commercially).",
        "test": "Reads the key's usage and limit (GET /key, free).",
    },
    "openai": {
        "label": "OpenAI", "env": ["OPENAI_API_KEY"], "auth": ("Authorization", "Bearer {key}"),
        "base": {"api": "https://api.openai.com/v1"}, "modalities": ["text", "image", "audio"],
        "keys_url": "https://platform.openai.com/api-keys", "pricing_url": "https://platform.openai.com/docs/pricing",
        "terms_url": "https://openai.com/policies/services-agreement",
        "terms": "You own the outputs (OpenAI Services Agreement). TTS voices are AI-generated: you must disclose "
                 "that to players.",
        "test": "Lists models (GET /models, free).",
    },
    "gemini": {
        "label": "Google Gemini API", "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "auth": ("x-goog-api-key", "{key}"),
        "base": {"api": "https://generativelanguage.googleapis.com/v1beta",
                 "openai": "https://generativelanguage.googleapis.com/v1beta/openai"},
        "modalities": ["text", "image", "audio"],
        "keys_url": "https://aistudio.google.com/apikey", "pricing_url": "https://ai.google.dev/gemini-api/docs/pricing",
        "terms_url": "https://ai.google.dev/gemini-api/terms",
        "terms": "Google claims no ownership of generated content; images carry a SynthID watermark. Free-tier "
                 "prompts may be used to improve Google products.",
        "test": "Lists one model (GET /models, free).",
    },
    "elevenlabs": {
        "label": "ElevenLabs", "env": ["ELEVENLABS_API_KEY", "XI_API_KEY"], "auth": ("xi-api-key", "{key}"),
        "base": {"api": "https://api.elevenlabs.io/v1"}, "modalities": ["audio"],
        "keys_url": "https://elevenlabs.io/app/settings/api-keys", "pricing_url": "https://elevenlabs.io/pricing/api",
        "terms_url": "https://elevenlabs.io/terms-of-use",
        "terms": "Commercial use needs a paid plan; free-plan audio is non-commercial and needs attribution. "
                 "Voices are AI-generated: disclose it.",
        "test": "Reads the subscription and remaining characters (GET /user/subscription, free).",
    },
    "meshy": {
        "label": "Meshy", "env": ["MESHY_API_KEY"], "auth": ("Authorization", "Bearer {key}"),
        "base": {"api": "https://api.meshy.ai/openapi"}, "modalities": ["mesh"],
        "keys_url": "https://www.meshy.ai/settings/api", "pricing_url": "https://docs.meshy.ai/en/api/pricing",
        "terms_url": "https://www.meshy.ai/terms-of-use",
        "terms": "Paid plans: assets are private and yours to use commercially. Free plan: public, CC BY 4.0.",
        "test": "Reads the credit balance (GET /v1/balance, free).",
    },
    "tripo": {
        "label": "Tripo", "env": ["TRIPO_API_KEY"], "auth": ("Authorization", "Bearer {key}"),
        "base": {"api": "https://openapi.tripo3d.ai/v3"}, "modalities": ["mesh"],
        "keys_url": "https://platform.tripo3d.ai/api-keys", "pricing_url": "https://platform.tripo3d.ai/pricing",
        "terms_url": "https://www.tripo3d.ai/terms",
        "terms": "Paid API credits: commercial use of generated models. Free credits: CC BY 4.0, public.",
        "test": "Reads the credit balance (GET /account/balance, free).",
    },
}
COMPAT_RE = re.compile(r"^openai_compat:([a-z0-9][a-z0-9_.-]{0,39})$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")
DEFAULTS = {"allow_cloud": False, "budget_usd": 20.0, "confirm_usd": 0.50, "price_overrides": {}, "endpoints": [],
            "upload_ack": {}, "base_urls": {}}


class ProviderError(JobError):
    """kind: budget | confirm | cloud_disabled | missing_key | consent | auth | billing | rate_limit | provider | input"""


# ------------------------------------------------------------------ pricing
def _megapixels(params: dict) -> float:
    """Output megapixels of one image (1 MP = 1024 x 1024, as the per-MP prices count it)."""
    MP = 1024 * 1024
    size = params.get("size")
    res = str(params.get("resolution") or "").upper().replace(" ", "")
    if isinstance(size, str) and "x" in size.lower():
        try:
            w, h = (int(x) for x in size.lower().split("x"))
            return w * h / MP
        except ValueError:
            pass
    if size not in (None, "", "auto"):
        try:
            long = float(size)
            a, b = (float(x) for x in str(params.get("aspect") or "1:1").split(":"))
            return long * long * min(a, b) / max(a, b) / MP
        except (TypeError, ValueError):
            pass
    m = re.fullmatch(r"(\d+(?:\.\d+)?)MP", res)
    if m:
        return float(m.group(1))
    return {"0.5K": 0.25, "512": 0.25, "1K": 1.0, "2K": 4.0, "4K": 16.0}.get(res, 1.0)


def _merge_by(pricing: dict, params: dict) -> dict:
    p = {k: v for k, v in pricing.items() if k not in ("by",)}
    for name, table in (pricing.get("by") or {}).items():
        v = params.get(name)
        key = str(v).lower() if isinstance(v, bool) else str(v)
        if isinstance(table, dict) and key in {str(k).lower() if isinstance(k, bool) else str(k) for k in table}:
            sub = next(t for k, t in table.items() if (str(k).lower() if isinstance(k, bool) else str(k)) == key)
            if isinstance(sub, dict):
                p.update(sub)
    return p


def estimate_cost(pricing: dict | None, params: dict | None, n_inputs: int = 0) -> float | None:
    """USD estimate of one job from a registry `pricing` block (schema: docs/CONTRACTS.md 8.2). None = unknown.
    The same formula runs in the browser (web/core/cost.js) for the Generate button."""
    if not pricing:
        return None
    params = params or {}
    p = _merge_by(pricing, params)
    n = max(1, int(params.get("batch") or params.get("num_images") or 1))
    usd = 0.0
    known = False
    if "per_image" in p:
        usd += float(p["per_image"]) * n
        known = True
    if "per_megapixel" in p or "first_megapixel" in p:
        mp = max(1, math.ceil(_megapixels(params) - 1e-6))
        first = float(p.get("first_megapixel", p.get("per_megapixel", 0)))
        usd += n * (first + (mp - 1) * float(p.get("per_megapixel", first)))
        known = True
    if "per_input_image" in p:
        usd += float(p["per_input_image"]) * n_inputs
        known = True
    if "per_input_megapixel" in p:
        usd += float(p["per_input_megapixel"]) * n_inputs
        known = True
    if "per_request" in p:
        usd += float(p["per_request"]) * n
        known = True
    secs = params.get("seconds", params.get("duration"))
    if "per_second" in p or "per_minute" in p:
        s = float(secs if secs not in (None, "") else p.get("default_seconds", 10))
        usd += n * s * (float(p.get("per_second", 0)) + float(p.get("per_minute", 0)) / 60)
        known = True
    if "per_1k_chars" in p:
        text = str(params.get("text") or params.get("prompt") or "")
        usd += n * max(len(text), 1) / 1000 * float(p["per_1k_chars"])
        known = True
    if "credits" in p:
        usd += n * float(p["credits"]) * float(p.get("per_credit", 0.01))
        known = True
    if "input_per_mtok" in p or "output_per_mtok" in p:
        tin = len(str(params.get("prompt") or params.get("text") or "")) / 3.5 + 200
        tout = float(params.get("max_tokens") or 800)
        usd += (tin * float(p.get("input_per_mtok", 0)) + tout * float(p.get("output_per_mtok", 0))) / 1e6
        known = True
    for name, table in (p.get("add") or {}).items():
        v = params.get(name)
        key = str(v).lower() if isinstance(v, bool) else str(v)
        for k, add in (table or {}).items():
            if (str(k).lower() if isinstance(k, bool) else str(k)) == key:
                usd += float(add) * n
                known = True
    if not known:
        return None
    usd = max(usd, float(p.get("min_usd", 0)))
    return round(usd, 4)


def fmt_usd(v: float | None) -> str:
    if v is None:
        return "price unknown"
    return f"${v:.3f}" if v < 0.1 else f"${v:.2f}"


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "::1") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


def _atomic_write(path: Path, text: str, mode: int = 0o644):
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(fd, mode)
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)
    os.chmod(path, mode)


# ------------------------------------------------------------------ spend
class Spend:
    """Monthly budget, per-job confirm threshold and the spend log (data/spend.jsonl)."""

    def __init__(self, providers: "Providers", path: Path):
        self.p = providers
        self.path = Path(path)
        self._lock = threading.Lock()
        self.entries: list[dict] = []
        self.reserved: dict[str, tuple[float, float]] = {}   # token -> (estimate, time) of queued/running cloud jobs
        self.is_active = None   # fn(token) -> bool: is the job holding this reservation still queued/running?
        try:
            for line in self.path.read_text().splitlines():
                try:
                    self.entries.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass

    @staticmethod
    def month_of(ts: float | None = None) -> str:
        return time.strftime("%Y-%m", time.localtime(ts or time.time()))

    def total(self, month: str | None = None) -> float:
        month = month or self.month_of()
        return round(sum(float(e.get("usd") or 0) for e in self.entries if e.get("month") == month), 6)

    def check(self, est_usd: float | None, provider_id: str, *, confirmed: bool = False, model_id: str | None = None):
        """Refuse a paid call: ProviderError kind cloud_disabled (switch off), budget (would exceed the monthly
        budget) or confirm (estimate >= the confirm threshold, or unknown, and the user did not confirm)."""
        p = self.p
        if not p.cloud_allowed() and not p.is_local_provider(provider_id):
            raise ProviderError("Cloud models are switched off.", "cloud_disabled",
                                "Turn on “Allow cloud” on the Providers page.")
        budget = float(p.state.get("budget_usd") or 0)
        thr = float(p.state.get("confirm_usd") or 0)
        spent = self.total()
        pending = self.pending()
        est = None if est_usd is None else float(est_usd)
        data = {"estimate_usd": est, "month_usd": round(spent, 4), "pending_usd": round(pending, 4), "budget_usd": budget,
                "confirm_usd": thr, "provider": provider_id, "provider_label": p.label(provider_id), "model": model_id}
        if budget > 0 and spent + pending + (est or 0) > budget + 1e-9:
            raise ProviderError(
                f"Monthly cloud budget reached: {fmt_usd(spent + pending)} of {fmt_usd(budget)} used this month"
                + (f", this job needs about {fmt_usd(est)}." if est else "."),
                "budget", "Raise the monthly budget on the Providers page, or use a local model.", data)
        if not confirmed and (est is None or (est > 0 and est >= thr)):
            msg = (f"This job costs about {fmt_usd(est)} on {p.label(provider_id)}." if est is not None
                   else f"The price of this {p.label(provider_id)} job is unknown.")
            raise ProviderError(msg, "confirm", "Confirm to run it (the threshold is set on the Providers page).", data)

    def pending(self) -> float:
        """Estimates of cloud jobs that are queued or running (they count against the budget until recorded)."""
        with self._lock:
            now = time.time()
            for t, (_usd, ts) in list(self.reserved.items()):
                stale = now - ts > 30 and self.is_active is not None and not self.is_active(t)
                if stale or now - ts > 6 * 3600:
                    self.reserved.pop(t, None)
            return round(sum(u for u, _ in self.reserved.values()), 6)

    def reserve(self, token: str, usd: float | None):
        with self._lock:
            self.reserved[token] = (float(usd or 0), time.time())

    def release(self, job_id: str):
        with self._lock:
            self.reserved.pop(job_id, None)

    def record(self, usd: float, provider_id: str, model_id: str | None = None, project: str | None = None,
               job_id: str | None = None, *, estimated: bool = False, note: str = "") -> dict:
        now = time.time()
        e = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "month": self.month_of(now),
             "usd": round(float(usd or 0), 6), "provider": provider_id, "model": model_id, "project": project,
             "job": job_id, "estimated": bool(estimated)}
        if note:
            e["note"] = note[:200]
        with self._lock:
            self.entries.append(e)
            if job_id:
                self.reserved.pop(job_id, None)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(e) + "\n")
            except OSError:
                pass
        if self.p.on_spend:
            self.p.on_spend(e)
        return e

    def summary(self, month: str | None = None) -> dict:
        month = month or self.month_of()
        es = [e for e in self.entries if e.get("month") == month]

        def group(key):
            out: dict = {}
            for e in es:
                k = e.get(key) or "–"
                g = out.setdefault(k, {"key": k, "usd": 0.0, "count": 0})
                g["usd"] += float(e.get("usd") or 0)
                g["count"] += 1
            return sorted(({**g, "usd": round(g["usd"], 4)} for g in out.values()), key=lambda g: -g["usd"])
        months: dict = {}
        for e in self.entries:
            months[e.get("month")] = months.get(e.get("month"), 0.0) + float(e.get("usd") or 0)
        total = self.total(month)
        budget = float(self.p.state.get("budget_usd") or 0)
        return {"month": month, "total_usd": round(total, 4), "budget_usd": budget,
                "confirm_usd": float(self.p.state.get("confirm_usd") or 0),
                "pending_usd": round(self.pending(), 4),
                "remaining_usd": round(max(0.0, budget - total), 4) if budget else None,
                "by_provider": [{**g, "label": self.p.label(g["key"])} for g in group("provider")],
                "by_model": group("model"), "by_project": group("project"),
                "recent": es[-25:][::-1], "estimated_count": sum(1 for e in es if e.get("estimated")),
                "months": [{"month": k, "usd": round(v, 4)} for k, v in sorted(months.items(), reverse=True)][:12]}


# ------------------------------------------------------------------ subscription CLIs (display only)
CLIS = [
    {"id": "claude", "label": "Claude Code", "bins": ["claude"], "version": ["--version"],
     "auth_cmd": ["auth", "status"], "auth_json": "loggedIn",
     "install": "curl -fsSL https://claude.ai/install.sh | bash   (or: npm i -g @anthropic-ai/claude-code)",
     "login": "claude   then /login", "docs": "https://code.claude.com/docs/en/cli-reference"},
    {"id": "codex", "label": "OpenAI Codex CLI", "bins": ["codex"], "version": ["--version"],
     "auth_cmd": ["login", "status"], "auth_files": ["~/.codex/auth.json"],
     "install": "npm i -g @openai/codex", "login": "codex login", "docs": "https://developers.openai.com/codex/cli"},
    {"id": "gemini", "label": "Gemini CLI / Antigravity CLI", "bins": ["gemini", "agy"], "version": ["--version"],
     "auth_files": ["~/.gemini/oauth_creds.json"], "auth_env": ["GEMINI_API_KEY"],
     "install": "npm i -g @google/gemini-cli   (personal accounts moved to Antigravity CLI `agy`)",
     "login": "gemini   (choose “Login with Google”)", "docs": "https://github.com/google-gemini/gemini-cli"},
    {"id": "copilot", "label": "GitHub Copilot CLI", "bins": ["copilot"], "version": ["--version"],
     "auth_env": ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"], "auth_dirs": ["~/.copilot"],
     "install": "npm i -g @github/copilot   (Node 22+)", "login": "copilot   then /login",
     "docs": "https://docs.github.com/copilot/how-tos/use-copilot-agents/use-copilot-cli"},
    {"id": "cursor-agent", "label": "Cursor Agent CLI", "bins": ["cursor-agent"], "version": ["--version"],
     "auth_cmd": ["status"], "auth_text": r"logged in|authenticated", "auth_env": ["CURSOR_API_KEY"],
     "install": "curl https://cursor.com/install -fsS | bash", "login": "cursor-agent login",
     "docs": "https://cursor.com/docs/cli"},
]
_EXTRA_BIN = ["~/.local/bin", "~/.npm-global/bin", "~/.bun/bin", "~/.cargo/bin", "~/bin", "/usr/local/bin"]


def _which(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    for d in _EXTRA_BIN:
        c = Path(os.path.expanduser(d)) / name
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _run(cmd: list[str], timeout: float = 8.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           env={**os.environ, "NO_COLOR": "1", "CI": "1"})
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def detect_clis() -> list[dict]:
    """Installed? version? logged in (only via quota-free checks)? Never returns account names or e-mails."""
    out = []
    for c in CLIS:
        path = next((p for p in (_which(b) for b in c["bins"]) if p), None)
        r = {"id": c["id"], "label": c["label"], "installed": bool(path), "bin": Path(path).name if path else None,
             "version": None, "logged_in": None, "auth_note": "", "install": c["install"], "login": c["login"],
             "docs": c["docs"]}
        if path:
            rc, txt = _run([path, *c["version"]], 6)
            m = re.search(r"\d+\.\d+(?:\.\d+)?[\w.-]*", txt or "")
            r["version"] = m.group(0) if m else (txt.strip().splitlines() or [""])[0][:40] or None
            if c.get("auth_cmd"):
                rc, txt = _run([path, *c["auth_cmd"]], 10)
                if c.get("auth_json"):
                    try:
                        d = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                        r["logged_in"] = bool(d.get(c["auth_json"]))
                        meth = d.get("authMethod")
                        r["auth_note"] = f"{meth}" + (f" · {d['subscriptionType']}" if d.get("subscriptionType") else "") if meth else ""
                    except ValueError:
                        r["logged_in"] = rc == 0
                elif c.get("auth_text"):
                    r["logged_in"] = bool(re.search(c["auth_text"], txt, re.I)) and not re.search(r"not logged in", txt, re.I)
                else:
                    r["logged_in"] = rc == 0
            if r["logged_in"] is not True:
                for f in c.get("auth_files", []):
                    if Path(os.path.expanduser(f)).exists():
                        r["logged_in"], r["auth_note"] = True, "credentials file found"
                for e in c.get("auth_env", []):
                    if os.environ.get(e):
                        r["logged_in"], r["auth_note"] = True, f"{e} set"
                for d in c.get("auth_dirs", []):
                    if r["logged_in"] is None and Path(os.path.expanduser(d)).is_dir():
                        r["auth_note"] = "config folder found (login state not checkable without a call)"
        out.append(r)
    return out


# ------------------------------------------------------------------ providers
class Providers:
    def __init__(self, data_dir: Path | None = None, registry=None, bus=None):
        self.dir = Path(data_dir or config.DATA)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.secrets_path = self.dir / "secrets.json"
        self.state_path = self.dir / "providers.json"
        self.registry = registry
        self.bus = bus
        self.on_spend = None
        self._lock = threading.RLock()
        self._secrets: dict = {"keys": {}}
        self.state: dict = json.loads(json.dumps(DEFAULTS))
        self._load()
        self.spend = Spend(self, self.dir / "spend.jsonl")
        self._cli_cache: tuple[float, list] | None = None

    # ---------------------------------------------------------- persistence
    def _load(self):
        try:
            if self.secrets_path.exists():
                if self.secrets_path.stat().st_mode & 0o077:
                    os.chmod(self.secrets_path, 0o600)
                d = json.loads(self.secrets_path.read_text())
                self._secrets = {"keys": {k: str(v) for k, v in (d.get("keys") or {}).items() if v}}
        except (OSError, ValueError):
            self._secrets = {"keys": {}}
        try:
            d = json.loads(self.state_path.read_text())
            for k, v in d.items():
                if k in DEFAULTS:
                    self.state[k] = v
        except (OSError, ValueError):
            pass

    def _save_secrets(self):
        _atomic_write(self.secrets_path, json.dumps(self._secrets, indent=1) + "\n", 0o600)

    def _save_state(self):
        _atomic_write(self.state_path, json.dumps(self.state, indent=1, ensure_ascii=False) + "\n", 0o644)

    def _changed(self, what: str):
        """Keys / switch / endpoints / prices changed: recompute cloud model status, tell every page."""
        if self.registry is not None:
            if what in ("endpoints", "prices"):
                self.registry.reload()
            else:
                self.registry.recompute(lambda m: self.is_cloud_model(m))
        if self.bus is not None:
            self.bus.publish("models", {"refreshed": True, "cloud": what})
            self.bus.publish("providers", {"changed": what})

    # ---------------------------------------------------------- ids
    def known(self, pid: str) -> bool:
        return pid in PROVIDERS or self.endpoint(pid) is not None

    def label(self, pid: str | None) -> str:
        if not pid:
            return "–"
        if pid in PROVIDERS:
            return PROVIDERS[pid]["label"]
        ep = self.endpoint(pid)
        return (ep.get("label") or ep["name"]) if ep else pid

    def info(self, pid: str) -> dict:
        if pid in PROVIDERS:
            return PROVIDERS[pid]
        ep = self.endpoint(pid)
        if ep:
            return {"label": ep.get("label") or ep["name"], "env": [], "auth": ("Authorization", "Bearer {key}"),
                    "base": {"api": ep["base_url"]}, "terms": ep.get("terms") or "User-added endpoint: its own terms apply.",
                    "modalities": ep.get("modalities") or ["text"]}
        raise KeyError(pid)

    def is_cloud_model(self, m: dict) -> bool:
        b = self.registry.backends.get(m.get("backend", ""), {}) if self.registry else {}
        return b.get("kind") == "cloud" or bool(m.get("provider"))

    # ---------------------------------------------------------- keys
    def get_key(self, provider_id: str) -> str | None:
        """Secrets file first, then the provider's environment variable(s). Never send the result to a browser."""
        k = self._secrets["keys"].get(provider_id)
        if k:
            return k
        for env in (PROVIDERS.get(provider_id) or {}).get("env", []):
            if os.environ.get(env):
                return os.environ[env]
        return None

    def key_info(self, pid: str) -> dict:
        """Public description of a key: never the key itself (only the last 4 characters of long keys)."""
        src, key, env = None, None, None
        if self._secrets["keys"].get(pid):
            src, key = "file", self._secrets["keys"][pid]
        else:
            for e in (PROVIDERS.get(pid) or {}).get("env", []):
                if os.environ.get(e):
                    src, key, env = "env", os.environ[e], e
                    break
        return {"set": bool(key), "source": src, "env": env, "last4": key[-4:] if key and len(key) >= 16 else None,
                "env_names": (PROVIDERS.get(pid) or {}).get("env", [])}

    def set_key(self, pid: str, key: str):
        key = (key or "").strip()
        if not self.known(pid):
            raise ProviderError(f"Unknown provider '{pid}'.", "input")
        if not key or len(key) > 400 or any(c.isspace() for c in key) or not key.isprintable():
            raise ProviderError("That does not look like an API key (empty, too long or contains spaces).", "input")
        with self._lock:
            self._secrets["keys"][pid] = key
            self._save_secrets()
        self._changed("keys")

    def delete_key(self, pid: str):
        with self._lock:
            self._secrets["keys"].pop(pid, None)
            self._save_secrets()
        self._changed("keys")

    def headers(self, pid: str) -> dict:
        key = self.get_key(pid)
        if not key:
            return {}
        name, fmt = self.info(pid).get("auth") or ("Authorization", "Bearer {key}")
        return {name: fmt.format(key=key)}

    def secrets_in(self, text: str) -> str:
        """Scrub every stored/env key from a text (provider error messages are shown to the user)."""
        keys = list(self._secrets["keys"].values())
        for p in PROVIDERS.values():
            keys += [os.environ[e] for e in p["env"] if os.environ.get(e)]
        for k in keys:
            if k and len(k) >= 6:
                text = text.replace(k, "***")
        return text

    # ---------------------------------------------------------- switch / settings
    def cloud_allowed(self) -> bool:
        return bool(self.state.get("allow_cloud"))

    def set_allowed(self, on: bool):
        with self._lock:
            self.state["allow_cloud"] = bool(on)
            self._save_state()
        self._changed("allow")

    def set_limits(self, budget_usd: float | None = None, confirm_usd: float | None = None):
        with self._lock:
            if budget_usd is not None:
                if not 0 <= float(budget_usd) <= 100000:
                    raise ProviderError("Budget must be between 0 and 100000 USD (0 = no budget).", "input")
                self.state["budget_usd"] = round(float(budget_usd), 2)
            if confirm_usd is not None:
                if not 0 <= float(confirm_usd) <= 10000:
                    raise ProviderError("Confirm threshold must be between 0 and 10000 USD.", "input")
                self.state["confirm_usd"] = round(float(confirm_usd), 3)
            self._save_state()
        if self.bus is not None:
            self.bus.publish("providers", {"changed": "limits"})

    # ---------------------------------------------------------- base URLs
    def base(self, pid: str, kind: str = "api") -> str:
        mock = os.environ.get("STUDIO_CLOUD_MOCK")
        if mock and pid in PROVIDERS:
            return f"{mock.rstrip('/')}/{pid}/{kind}"
        over = (self.state.get("base_urls") or {}).get(pid) or {}
        if isinstance(over, str):
            over = {"api": over}
        if over.get(kind):
            return over[kind].rstrip("/")
        return self.info(pid)["base"][kind].rstrip("/")

    def get_base_url(self, provider_id: str) -> str | None:
        """OpenAI-compatible base (POST {base}/chat/completions) for openai, openrouter, gemini, openai_compat:*."""
        if provider_id == "gemini":
            return self.base("gemini", "openai")
        if provider_id in ("openai", "openrouter"):
            return self.base(provider_id)
        ep = self.endpoint(provider_id)
        return ep["base_url"].rstrip("/") if ep else None

    base_url = get_base_url

    def is_local_provider(self, pid: str) -> bool:
        ep = self.endpoint(pid)
        return bool(ep and _is_local(ep["base_url"]))

    # ---------------------------------------------------------- user endpoints (openai_compat:<name>)
    def endpoints(self) -> list[dict]:
        return list(self.state.get("endpoints") or [])

    def endpoint(self, pid: str) -> dict | None:
        m = COMPAT_RE.match(pid or "")
        if not m:
            return None
        return next((e for e in self.endpoints() if e["name"] == m.group(1)), None)

    def put_endpoint(self, name: str, base_url: str, models: list[str], modalities: list[str], label: str = "",
                     key: str | None = None, price_per_image: float | None = None) -> dict:
        name = (name or "").strip().lower()
        if not NAME_RE.match(name):
            raise ProviderError("Name: lowercase letters, digits, '.', '_' or '-' (max 40).", "input")
        u = urlparse((base_url or "").strip())
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ProviderError("Base URL must be http(s)://host[:port]/path (e.g. http://127.0.0.1:7880/v1).", "input")
        if u.scheme == "http" and not _is_local(base_url):
            raise ProviderError("Plain http is only allowed for local / LAN endpoints; use https.", "input")
        mods = [x for x in (modalities or []) if x in ("llm", "image")] or ["llm"]
        ms = []
        for x in models or []:
            x = str(x).strip()
            if x and len(x) <= 120 and x not in ms and re.fullmatch(r"[\w./:@+-]+", x):
                ms.append(x)
        ep = {"name": name, "label": (label or name).strip()[:60], "base_url": base_url.strip().rstrip("/"),
              "models": ms[:40], "modalities": mods, "local": _is_local(base_url)}
        if price_per_image is not None:
            ep["price_per_image"] = max(0.0, float(price_per_image))
        with self._lock:
            eps = [e for e in self.endpoints() if e["name"] != name] + [ep]
            self.state["endpoints"] = eps
            self._save_state()
            if key:
                self._secrets["keys"][f"openai_compat:{name}"] = key.strip()
                self._save_secrets()
        self._changed("endpoints")
        return ep

    def delete_endpoint(self, name: str):
        with self._lock:
            self.state["endpoints"] = [e for e in self.endpoints() if e["name"] != name]
            self._secrets["keys"].pop(f"openai_compat:{name}", None)
            self._save_state()
            self._save_secrets()
        self._changed("endpoints")

    def endpoint_models(self) -> dict:
        """Registry model source: one entry per (endpoint, model, modality)."""
        models = []
        for ep in self.endpoints():
            pid = f"openai_compat:{ep['name']}"
            for i, rm in enumerate(ep.get("models") or []):
                slug = re.sub(r"[^a-z0-9]+", "-", rm.lower()).strip("-")[:40] or "model"
                base = {"provider": pid, "remote_model": rm, "backend": "cloud", "order": 90 + i,
                        "license": {"name": f"{ep.get('label') or ep['name']} (user endpoint)", "commercial": True,
                                    "verify": True, "note": "Terms of whoever runs the endpoint apply."},
                        "pricing_checked": None, "user_endpoint": True}
                if "llm" in ep["modalities"]:
                    models.append({**base, "id": f"oc-{ep['name']}-{slug}", "name": f"{rm} · {ep.get('label') or ep['name']}",
                                   "modality": "text", "tasks": ["improve", "write"], "api": "openai_chat",
                                   "summary": f"Text model on your endpoint {ep['base_url']}",
                                   "pricing": {"input_per_mtok": 0, "output_per_mtok": 0} if ep.get("local") else None})
                if "image" in ep["modalities"]:
                    models.append({**base, "id": f"oc-{ep['name']}-{slug}-img", "name": f"{rm} · {ep.get('label') or ep['name']}",
                                   "modality": "image", "tasks": ["t2i"], "api": "openai_images", "adapter": "openai",
                                   "summary": f"Images from your OpenAI-compatible endpoint {ep['base_url']} (/images/generations)",
                                   "pricing": {"per_image": float(ep.get("price_per_image") or 0)},
                                   "params": [
                                       {"name": "size", "label": "Size", "type": "choice", "default": "1024x1024",
                                        "choices": ["1024x1024", "1536x1024", "1024x1536", "512x512"]},
                                       {"name": "batch", "label": "Images", "type": "int", "min": 1, "max": 4, "default": 1},
                                   ]})
        return {"models": models}

    # ---------------------------------------------------------- prices
    def set_price_override(self, model_id: str, pricing: dict | None):
        if pricing is not None:
            if not isinstance(pricing, dict) or not pricing:
                raise ProviderError("Pricing must be an object like {\"per_image\": 0.04}.", "input")
            clean = {}
            for k, v in pricing.items():
                if k in ("by", "add") and isinstance(v, dict):
                    clean[k] = v
                elif k == "note":
                    clean[k] = str(v)[:200]
                else:
                    try:
                        clean[k] = float(v)
                    except (TypeError, ValueError):
                        raise ProviderError(f"{k}: not a number", "input") from None
                    if clean[k] < 0:
                        raise ProviderError(f"{k}: negative price", "input")
            pricing = clean
        with self._lock:
            ov = dict(self.state.get("price_overrides") or {})
            if pricing is None:
                ov.pop(model_id, None)
            else:
                ov[model_id] = {**pricing, "_set": time.strftime("%Y-%m-%d")}
            self.state["price_overrides"] = ov
            self._save_state()
        self._changed("prices")

    def estimate(self, model: dict, params: dict, n_inputs: int = 0) -> float | None:
        return estimate_cost(model.get("pricing"), params, n_inputs)

    # ---------------------------------------------------------- registry hooks
    def model_hook(self, m: dict):
        """Fill cloud fields of a registry entry: provider label + terms, price override, default licence."""
        if not self.is_cloud_model(m):
            return
        pid = m.get("provider") or ""
        try:
            info = self.info(pid)
        except KeyError:
            info = {"label": pid, "terms": ""}
        m["device"] = "cpu"
        m["cloud"] = {"provider": pid, "label": info.get("label", pid), "terms": m.get("terms") or info.get("terms", ""),
                      "terms_url": info.get("terms_url"), "remote_model": m.get("remote_model"),
                      "uploads": bool(m.get("uploads", True)), "local": self.is_local_provider(pid)}
        if not m.get("license") or m["license"].get("name") == "unknown":
            m["license"] = {"name": f"{info.get('label', pid)} terms", "commercial": True, "verify": True,
                            "note": info.get("terms", "")}
        if not m["license"].get("note"):
            m["license"]["note"] = info.get("terms", "")
        if "cloud" not in m["tags"]:
            m["tags"] = [*m["tags"], "cloud"]
        ov = (self.state.get("price_overrides") or {}).get(m["id"])
        if ov:
            m["pricing_default"] = m.get("pricing")
            m["pricing"] = {k: v for k, v in ov.items() if not k.startswith("_")}
            m["pricing_overridden"] = ov.get("_set") or True

    def model_status(self, m: dict, b: dict) -> dict | None:
        pid = m.get("provider")
        if not pid or not self.known(pid):
            return {"status": "broken", "code": "unknown_provider", "reason": f"unknown provider '{pid}'",
                    "fix": "Add the endpoint on the Providers page or fix `provider:` in models.d/cloud.yaml"}
        local = self.is_local_provider(pid)
        if not local and not self.cloud_allowed():
            return {"status": "cloud_disabled", "kind": "cloud_disabled", "reason": "Cloud models are switched off",
                    "fix": "Turn on “Allow cloud” on the Providers page"}
        if not self.get_key(pid) and pid in PROVIDERS:
            envs = PROVIDERS[pid]["env"]
            return {"status": "missing_key", "kind": "missing_key", "reason": f"No API key for {self.label(pid)}",
                    "fix": f"Add a key on the Providers page (or set {envs[0]})"}
        return None

    # ---------------------------------------------------------- uploads (consent, once per provider)
    def upload_acknowledged(self, pid: str) -> bool:
        return pid in (self.state.get("upload_ack") or {}) or self.is_local_provider(pid)

    def acknowledge_upload(self, pid: str, on: bool = True):
        with self._lock:
            ack = dict(self.state.get("upload_ack") or {})
            if on:
                ack[pid] = time.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                ack.pop(pid, None)
            self.state["upload_ack"] = ack
            self._save_state()

    # ---------------------------------------------------------- public views (never contain a key)
    def public(self) -> dict:
        provs = []
        models = list(self.registry.models.values()) if self.registry else []
        for pid, p in PROVIDERS.items():
            ms = [m for m in models if m.get("provider") == pid]
            provs.append({"id": pid, "label": p["label"], "key": self.key_info(pid), "modalities": p["modalities"],
                          "keys_url": p["keys_url"], "pricing_url": p["pricing_url"], "terms_url": p["terms_url"],
                          "terms": p["terms"], "test": p["test"], "models": len(ms),
                          "upload_ack": (self.state.get("upload_ack") or {}).get(pid),
                          "base_url": self.base(pid), "base_overridden": bool(os.environ.get("STUDIO_CLOUD_MOCK") or (self.state.get("base_urls") or {}).get(pid))})
        eps = []
        for e in self.endpoints():
            pid = f"openai_compat:{e['name']}"
            eps.append({**e, "id": pid, "key": self.key_info(pid),
                        "registry_models": [m["id"] for m in models if m.get("provider") == pid]})
        return {"allow_cloud": self.cloud_allowed(), "budget_usd": self.state.get("budget_usd"),
                "confirm_usd": self.state.get("confirm_usd"), "providers": provs, "endpoints": eps,
                "price_overrides": {k: v for k, v in (self.state.get("price_overrides") or {}).items()},
                "mock": bool(os.environ.get("STUDIO_CLOUD_MOCK"))}

    def clis(self, refresh: bool = False) -> list[dict]:
        if refresh or not self._cli_cache or time.time() - self._cli_cache[0] > 300:
            self._cli_cache = (time.time(), detect_clis())
        return self._cli_cache[1]
