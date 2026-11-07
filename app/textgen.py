"""Text generation: one interface over three kinds of backends.

  local   the studio's own llama-server (the prompt improver's LLM, app/improver.py LLMSupervisor)
  api     OpenAI-compatible HTTP: OpenRouter, OpenAI, Gemini's OpenAI endpoint, user-added "openai_compat:<name>"
          endpoints. Keys come from `core.providers.get_key()` (the cloud-provider plugin); spend is checked with
          `core.providers.spend.check()` before and recorded with `.record()` after every call.
  cli     subscription CLIs run as locked-down subprocesses: claude (tested), codex, gemini, copilot (untested).

Backend ids (strings, used by the UI and stored in settings.json `textgen`):
  "local" · "cli:claude" · "cli:claude@sonnet" (model override after '@') · "api:<provider>/<model>"
  e.g. "api:openrouter/anthropic/claude-haiku-4.5", "api:openai_compat:lab/prompt-llm". Registry entries with
  modality text that name a provider are offered as "api:<provider>/<api model>" too.

`TextGen.generate()` returns {"data", "text", "backend", "seconds", "usd", "attempts", ...}. With a JSON schema the
answer is schema-constrained where the backend supports it (llama-server, OpenAI-compatible json_schema, claude
--json-schema, codex --output-schema); elsewhere the schema is put in the prompt, the answer is validated and
retried once with the validation errors. Errors are `TextGenError(message, kind, fix)` with kind in
not_installed | not_logged_in | rate_limited | quota | timeout | cancelled | bad_output | http | budget | confirm |
config | unavailable.

CLI safety (tests/test_textgen.py checks the argv): non-interactive mode only, no tools for plain text (claude:
`--tools ""`, codex: read-only sandbox + approval never, gemini: default approval mode (tool calls that need
approval are refused headless) with no extensions, copilot: shell/write/url/memory denied), project/user settings
ignored, a fresh empty temp dir as cwd (never a user project), a scrubbed environment (no git identity, no API
keys, no e-mail). Nothing about the user's identity is ever put in a prompt.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

# ===================================================================== errors
FIXES = {
    "not_installed": "Install the CLI and log in once in a terminal, or pick another backend.",
    "not_logged_in": "Log in once in a terminal (e.g. `claude` then /login), or add the API key on the Providers page.",
    "rate_limited": "The provider is rate limiting: wait a minute, or switch to the local LLM.",
    "quota": "Your plan / credit limit is used up for now. Switch to the local LLM or another backend.",
    "timeout": "The backend took too long. Try again, shorten the request, or pick a faster model.",
    "budget": "The cloud budget would be exceeded. Raise it on the Providers page or use a local / CLI backend.",
    "confirm": "This call needs your confirmation of the estimated cost.",
    "config": "Check Settings → Text backends.",
    "unavailable": "Pick another backend.",
}


class TextGenError(Exception):
    def __init__(self, message: str, kind: str = "http", fix: str | None = None, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.fix = FIXES.get(kind, "") if fix is None else fix
        self.status = status or {"not_installed": 503, "not_logged_in": 401, "rate_limited": 429, "quota": 429,
                                 "timeout": 504, "cancelled": 499, "budget": 402, "confirm": 409,
                                 "config": 400, "unavailable": 503}.get(kind, 502)


# ===================================================================== backend catalogue
CLI_INFO: dict[str, dict] = {
    "claude": {"label": "Claude Code", "binary": "claude", "plan": "Claude Pro / Max subscription", "tested": True,
               "vision": True, "json_schema": True, "tools": "mcp", "default_model": "haiku",
               "models": ["haiku", "sonnet", "opus"], "login": "claude  (then /login)"},
    "codex": {"label": "Codex CLI", "binary": "codex", "plan": "ChatGPT Plus / Pro plan", "tested": False,
              "vision": True, "json_schema": True, "tools": None, "default_model": "",
              "models": [], "login": "codex login"},
    "gemini": {"label": "Gemini CLI", "binary": "gemini", "plan": "Google account / AI plan", "tested": False,
               "vision": True, "json_schema": False, "tools": None, "default_model": "",
               "models": ["gemini-2.5-flash", "gemini-2.5-pro"], "login": "gemini  (choose Login with Google)"},
    "copilot": {"label": "GitHub Copilot CLI", "binary": "copilot", "plan": "GitHub Copilot subscription", "tested": False,
                "vision": True, "json_schema": False, "tools": None, "default_model": "",
                "models": [], "login": "copilot  (then /login)"},
}
TOS_NOTE = ("Subscription CLIs use your personal plan: every call counts against its usage limits, and the plans are "
            "meant for personal use. Do not share this studio's CLI backends with other people.")

PROVIDER_BASE = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
PROVIDER_LABEL = {"openrouter": "OpenRouter", "openai": "OpenAI", "gemini": "Gemini API"}

# flags that would give a CLI tools / write access / skip permission checks: never built (tests check this)
FORBIDDEN_FLAGS = ("--dangerously-skip-permissions", "--allow-dangerously-skip-permissions", "bypassPermissions",
                   "acceptEdits", "--yolo", "yolo", "auto_edit", "--allow-all-tools", "--allow-all", "--allow-all-paths",
                   "workspace-write", "danger-full-access", "--full-auto", "--approve-for-me", "--add-dir", "--allow-tool",
                   "--dangerously-bypass-approvals-and-sandbox", "--include-directories", "auto")
CLAUDE_BUILTIN_DENY = ("Bash", "BashOutput", "KillShell", "Edit", "MultiEdit", "Write", "Read", "NotebookEdit", "NotebookRead",
                       "Glob", "Grep", "LS", "WebFetch", "WebSearch", "Task", "Agent", "TodoWrite", "PowerShell", "REPL",
                       "Skill", "SlashCommand", "ExitPlanMode", "Monitor")

ENV_ALLOW = ("HOME", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
             "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "CODEX_HOME", "GEMINI_CLI_HOME", "COPILOT_HOME", "CLAUDE_CONFIG_DIR",
             "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "SSL_CERT_FILE",
             "NODE_EXTRA_CA_CERTS")


def cli_env() -> dict:
    """Scrubbed environment for CLI subprocesses: no git identity, no API keys, no e-mail, no studio paths."""
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    home = env.get("HOME") or str(Path.home())
    extra = [str(Path(home) / ".local/bin"), str(Path(home) / ".npm-global/bin"), "/usr/local/bin", "/usr/bin", "/bin"]
    env["PATH"] = os.pathsep.join(dict.fromkeys([p for p in env.get("PATH", "").split(os.pathsep) if p] + extra))
    env.update(TERM="dumb", NO_COLOR="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1",
               DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1")
    return env


def parse_backend(bid: str | None) -> dict:
    """'local' | 'cli:<name>[@model]' | 'api:<provider>/<model>' -> {id, kind, name, provider, model}."""
    bid = (bid or "local").strip()
    if bid == "local":
        return {"id": "local", "kind": "local", "name": "local", "provider": None, "model": None}
    if bid.startswith("cli:"):
        rest = bid[4:]
        name, _, model = rest.partition("@")
        if name not in CLI_INFO:
            raise TextGenError(f"Unknown CLI backend '{name}'.", "config")
        if model and not re.fullmatch(r"[A-Za-z0-9._:\-\[\]]{1,80}", model):
            raise TextGenError("Invalid model name.", "config")
        return {"id": bid, "kind": "cli", "name": name, "provider": None, "model": model or None}
    if bid.startswith("api:"):
        rest = bid[4:]
        provider, sep, model = rest.partition("/")
        if not sep or not model or not provider:
            raise TextGenError("API backends look like api:<provider>/<model>.", "config")
        if not re.fullmatch(r"(openrouter|openai|gemini|openai_compat:[A-Za-z0-9_.\-]{1,40})", provider):
            raise TextGenError(f"Unknown provider '{provider}'.", "config")
        return {"id": bid, "kind": "api", "name": provider, "provider": provider, "model": model}
    raise TextGenError(f"Unknown backend '{bid}'.", "config")


# ===================================================================== JSON helpers
def validate_json(value, schema: dict, path: str = "$") -> list[str]:
    """Small JSON-schema subset validator (type, properties, required, items, enum, min/maxItems, minLength)."""
    errs: list[str] = []
    t = schema.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    pymap = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}

    def is_t(v, tt):
        if tt == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if tt == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        return isinstance(v, pymap.get(tt, object))
    if types and not any(is_t(value, tt) for tt in types):
        return [f"{path}: expected {'/'.join(types)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, dict):
        for k in schema.get("required") or []:
            if k not in value:
                errs.append(f"{path}.{k}: missing")
        for k, sub in (schema.get("properties") or {}).items():
            if k in value:
                errs += validate_json(value[k], sub, f"{path}.{k}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value[:200]):
                errs += validate_json(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        errs.append(f"{path}: too short")
    return errs


def extract_json(text: str):
    """The first JSON object in a model answer (tolerates ``` fences and prose around it)."""
    s = (text or "").strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except ValueError:
                        break
        start = s.find("{", start + 1)
    raise ValueError("no JSON object in the answer")


def schema_instruction(schema: dict) -> str:
    return ("\n\nOUTPUT FORMAT: reply with ONE JSON object only (no prose, no markdown fences) that matches this JSON "
            "Schema:\n" + json.dumps(schema, ensure_ascii=False))


def _split_messages(messages: list[dict]) -> tuple[str, list[dict], list[bytes]]:
    """OpenAI-style messages -> (system text, [{role, text}], images as JPEG/PNG bytes from data URLs)."""
    system, turns, images = [], [], []
    for m in messages:
        c = m.get("content")
        texts = []
        if isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        images.append(base64.b64decode(url.split(",", 1)[1]))
        else:
            texts.append(str(c or ""))
        txt = "\n".join(texts)
        if m.get("role") == "system":
            system.append(txt)
        else:
            turns.append({"role": m.get("role", "user"), "text": txt})
    return "\n\n".join(system), turns, images


def transcript(turns: list[dict]) -> str:
    """Conversation -> one prompt for CLIs that take a single message."""
    if len(turns) == 1:
        return turns[0]["text"]
    out = []
    for t in turns[:-1]:
        who = "USER" if t["role"] == "user" else "ASSISTANT" if t["role"] == "assistant" else t["role"].upper()
        out.append(f"[{who}]\n{t['text']}")
    out.append(f"[USER — answer this]\n{turns[-1]['text']}")
    return "Conversation so far:\n\n" + "\n\n".join(out)


def _img_ext(b: bytes) -> str:
    return ".png" if b[:8] == b"\x89PNG\r\n\x1a\n" else ".webp" if b[8:12] == b"WEBP" else ".jpg"


def _img_mime(b: bytes) -> str:
    return {".png": "image/png", ".webp": "image/webp"}.get(_img_ext(b), "image/jpeg")


def classify_cli_error(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ("not logged in", "please run /login", "invalid api key", "login required", "unauthorized",
                            "authentication", "oauth token", "not authenticated", "please log in", "auth login")):
        return "not_logged_in"
    if any(x in t for x in ("usage limit", "limit reached", "quota", "limit will reset", "out of credits", "insufficient",
                            "exceeded your", "plan limit")):
        return "quota"
    if any(x in t for x in ("rate limit", "rate_limit", "429", "too many requests", "overloaded")):
        return "rate_limited"
    return "http"


# ===================================================================== CLI command construction
def build_cli(name: str, *, prompt: str, system: str = "", model: str | None = None, schema: dict | None = None,
              images: list[Path] | None = None, workdir: Path, mcp_config: str | None = None,
              mcp_tools: list[str] | None = None, max_turns: int = 2, binary: str | None = None,
              effort: str | None = None) -> tuple[list[str], bytes | None]:
    """argv + stdin for one locked-down CLI call. Pure function (unit-tested): never adds tool-enabling flags."""
    info = CLI_INFO[name]
    exe = binary or info["binary"]
    images = list(images or [])
    if name == "claude":
        argv = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--no-session-persistence", "--permission-mode", "dontAsk", "--permission-prompts", "none",
                "--disable-slash-commands", "--setting-sources", "", "--strict-mcp-config",
                "--max-turns", str(int(max_turns))]
        if mcp_config:
            # the agent: only the studio's own MCP tools, every built-in tool removed and denied
            argv += ["--tools", "", "--mcp-config", mcp_config, "--allowedTools", ",".join(mcp_tools or []),
                     "--disallowedTools", ",".join(CLAUDE_BUILTIN_DENY)]
        else:
            argv += ["--tools", "", "--mcp-config", '{"mcpServers":{}}', "--safe-mode",
                     "--disallowedTools", ",".join(CLAUDE_BUILTIN_DENY)]
        if model:
            argv += ["--model", model]
        argv += ["--effort", effort or ("medium" if mcp_config else "low")]
        if system:
            argv += ["--system-prompt", system]
        if schema:
            argv += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in images:
            b = Path(p).read_bytes()
            content.append({"type": "image", "source": {"type": "base64", "media_type": _img_mime(b),
                                                         "data": base64.b64encode(b).decode()}})
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        return argv, (json.dumps(msg, ensure_ascii=False) + "\n").encode()
    full = (f"{system}\n\n" if system else "") + prompt
    if name == "codex":
        argv = [exe, "exec", "--sandbox", "read-only", "-c", 'approval_policy="never"', "--skip-git-repo-check",
                "--ephemeral", "--ignore-user-config", "--ignore-rules", "--json", "-C", str(workdir),
                "-o", str(workdir / "last_message.txt")]
        if model:
            argv += ["-m", model]
        if schema:
            sp = workdir / "schema.json"
            sp.write_text(json.dumps(schema))
            argv += ["--output-schema", str(sp)]
        for p in images:
            argv += ["-i", str(p)]
        argv.append("-")
        return argv, full.encode()
    if name == "gemini":
        refs = "".join(f" @{Path(p).name}" for p in images)   # files are copied into the cwd; @file is read by the CLI itself
        argv = [exe, "--output-format", "json", "--approval-mode", "default", "-e", "none"]
        if model:
            argv += ["-m", model]
        argv += ["-p", full + (f"\n\nAttached image(s):{refs}" if refs else "")]
        return argv, None
    if name == "copilot":
        argv = [exe, "-s", "--no-ask-user", "--deny-tool=shell", "--deny-tool=write", "--deny-tool=url", "--deny-tool=memory"]
        if model:
            argv.append(f"--model={model}")
        for p in images:
            argv.append(f"--attachment={p}")
        argv += ["-p", full]
        return argv, None
    raise TextGenError(f"Unknown CLI '{name}'.", "config")


def parse_claude_stream(lines: list[str]) -> dict:
    """claude -p --output-format stream-json -> {text, structured, is_error, error, tools, tool_uses, init, cost}."""
    out = {"text": "", "structured": None, "is_error": False, "error": "", "tool_uses": [], "tool_results": [],
           "init": None, "usd": 0.0, "turns": 0, "denials": [], "texts": []}
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            out["init"] = {"tools": ev.get("tools") or [], "mcp_servers": ev.get("mcp_servers") or [],
                           "model": ev.get("model"), "permissionMode": ev.get("permissionMode")}
        elif t == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text"):
                    out["texts"].append(c["text"])
                elif c.get("type") == "tool_use":
                    out["tool_uses"].append({"name": c.get("name"), "input": c.get("input"), "id": c.get("id")})
        elif t == "user":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    out["tool_results"].append({"id": c.get("tool_use_id"), "is_error": bool(c.get("is_error"))})
        elif t == "result":
            out["text"] = ev.get("result") or ""
            out["structured"] = ev.get("structured_output")
            out["is_error"] = bool(ev.get("is_error")) or (ev.get("subtype") not in (None, "success"))
            out["subtype"] = ev.get("subtype")
            out["usd"] = float(ev.get("total_cost_usd") or 0)
            out["turns"] = ev.get("num_turns") or 0
            out["denials"] = ev.get("permission_denials") or []
            if out["is_error"]:
                out["error"] = str(ev.get("result") or ev.get("subtype") or "error")
    if not out["text"] and out["texts"]:
        out["text"] = out["texts"][-1]
    return out


def parse_codex_jsonl(lines: list[str]) -> dict:
    out = {"text": "", "error": ""}
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        t = ev.get("type", "")
        item = ev.get("item") or {}
        if t == "item.completed" and item.get("type") in ("agent_message", "assistant_message"):
            out["text"] = item.get("text") or out["text"]
        elif t in ("error", "turn.failed"):
            out["error"] = (ev.get("error") or {}).get("message") if isinstance(ev.get("error"), dict) else str(ev.get("message") or ev.get("error") or t)
    return out


# ===================================================================== the service
def _providers(core):
    return getattr(core, "providers", None) if core is not None else None


class TextGen:
    def __init__(self, settings, llm=None, registry=None, core=None):
        self.settings = settings
        self.llm = llm
        self.registry = registry
        self.core = core
        self._auth_cache: dict[str, tuple[float, dict]] = {}

    # ------------------------------------------------------------ settings
    @property
    def cfg(self) -> dict:
        return self.settings.get("textgen") or {} if self.settings else {}

    def default_for(self, purpose: str, settings=None) -> str:
        """Settings `textgen.<purpose>` (improve | write | assistant), from `settings` (a project's SettingsView) when
        given. Improve defaults to local; the assistant to claude when it is installed."""
        cfg = (settings.get("textgen") or {}) if settings is not None else self.cfg
        v = cfg.get(purpose)
        if v:
            return v
        if purpose == "assistant" and self._which("claude"):
            return "cli:claude@sonnet"
        if purpose == "write" and self._which("claude"):
            return "cli:claude@haiku"
        return "local"

    # ------------------------------------------------------------ discovery
    def _which(self, name: str) -> str | None:
        b = ((self.cfg.get("cli") or {}).get(name) or {}).get("binary")
        if b:
            return b if os.path.isfile(os.path.expanduser(b)) else None
        return shutil.which(CLI_INFO[name]["binary"], path=cli_env()["PATH"])

    def cli_auth(self, name: str) -> dict:
        """{logged_in: bool|None, method, plan} without identity fields (cached 5 min). Only claude has a cheap check."""
        if name != "claude":
            return {"logged_in": None}
        hit = self._auth_cache.get(name)
        if hit and time.time() - hit[0] < 300:
            return hit[1]
        exe = self._which(name)
        res = {"logged_in": None}
        if exe:
            try:
                wd = tempfile.mkdtemp(prefix="studio-llm-")
                p = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, timeout=15, env=cli_env(), cwd=wd)
                shutil.rmtree(wd, ignore_errors=True)
                d = json.loads(p.stdout or "{}")
                res = {"logged_in": bool(d.get("loggedIn")), "method": d.get("authMethod"), "plan": d.get("subscriptionType")}
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        self._auth_cache[name] = (time.time(), res)
        return res

    def _provider_base(self, provider: str) -> str | None:
        prov = _providers(self.core)
        for fn in ("get_base_url", "base_url"):
            f = getattr(prov, fn, None)
            if callable(f):
                try:
                    u = f(provider)
                    if u:
                        return str(u).rstrip("/")
                except Exception:  # noqa: BLE001
                    pass
        getp = getattr(prov, "get", None)
        if callable(getp):
            try:
                d = getp(provider)
                if isinstance(d, dict) and d.get("base_url"):
                    return str(d["base_url"]).rstrip("/")
            except Exception:  # noqa: BLE001
                pass
        if provider in PROVIDER_BASE:
            return PROVIDER_BASE[provider]
        if provider.startswith("openai_compat:"):
            ep = (self.cfg.get("endpoints") or {}).get(provider.split(":", 1)[1])
            if isinstance(ep, dict):
                ep = ep.get("base_url")
            return str(ep).rstrip("/") if ep else None
        return None

    def _key(self, provider: str) -> str | None:
        prov = _providers(self.core)
        f = getattr(prov, "get_key", None)
        if callable(f):
            try:
                return f(provider)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _cloud_allowed(self) -> bool:
        prov = _providers(self.core)
        f = getattr(prov, "cloud_allowed", None)
        if callable(f):
            try:
                return bool(f())
            except Exception:  # noqa: BLE001
                return False
        return True

    @staticmethod
    def _is_local_url(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host in ("127.0.0.1", "localhost", "::1") or host.startswith(("10.", "192.168.")) or host.endswith(".local")

    def registry_api_models(self) -> list[dict]:
        """Cloud LLM entries of the registry (modality text, listed by the providers plugin): normalized."""
        out = []
        for m in (self.registry.models.values() if self.registry else []):
            if m.get("modality") != "text" or m.get("backend") == "llm" or m.get("test"):
                continue
            f = m.get("features") or {}
            prov = m.get("provider") or f.get("provider") or (m.get("cloud") or {}).get("provider")
            if not prov:
                continue
            api_model = (m.get("api_model") or m.get("remote_model") or f.get("api_model") or f.get("remote_model")
                         or (m.get("cloud") or {}).get("model") or m["id"])
            out.append({"registry_id": m["id"], "provider": prov, "model": api_model, "name": m.get("name"),
                        "vision": bool(f.get("vision")), "pricing": self._pricing(m), "license": m.get("license")})
        return out

    @staticmethod
    def _pricing(m: dict) -> dict | None:
        for key in ("pricing", "price", "cost"):
            p = m.get(key) or (m.get("features") or {}).get(key) or (m.get("cloud") or {}).get(key)
            if isinstance(p, dict):
                i = p.get("input_per_mtok", p.get("input", p.get("prompt")))
                o = p.get("output_per_mtok", p.get("output", p.get("completion")))
                if i is not None or o is not None:
                    return {"input_per_mtok": float(i or 0), "output_per_mtok": float(o or 0)}
        return None

    def backends(self, *, check_auth: bool = False) -> list[dict]:
        """Every backend option with availability + reason, for the pickers."""
        out = []
        st = self.llm.status() if self.llm else {}
        out.append({"id": "local", "kind": "local", "label": "Local LLM", "detail": st.get("model") or st.get("primary") or "llama-server",
                    "available": bool(st.get("available") or st.get("state") == "external"), "reason": "" if st.get("available") else "No local GGUF / llama-server found",
                    "vision": True, "tools": "openai", "json_schema": True, "cost": "free (local)", "tested": True})
        for name, info in CLI_INFO.items():
            exe = self._which(name)
            auth = self.cli_auth(name) if (exe and check_auth) else {"logged_in": None}
            model = ((self.cfg.get("cli") or {}).get(name) or {}).get("model") or info["default_model"]
            reason = "" if exe else f"{info['binary']} is not installed"
            if exe and auth.get("logged_in") is False:
                reason = f"not logged in: run `{info['login']}` in a terminal"
            out.append({"id": f"cli:{name}", "kind": "cli", "name": name, "label": info["label"], "detail": info["plan"],
                        "model": model or None, "models": info["models"], "available": bool(exe) and auth.get("logged_in") is not False,
                        "installed": bool(exe), "logged_in": auth.get("logged_in"), "reason": reason, "vision": info["vision"],
                        "tools": info["tools"] or "json", "json_schema": info["json_schema"], "tested": info["tested"],
                        "cost": "counts against your plan", "tos": TOS_NOTE})
        cloud = self._cloud_allowed()
        seen = set()
        for m in self.registry_api_models():
            bid = f"api:{m['provider']}/{m['model']}"
            seen.add(bid)
            out.append(self._api_option(bid, m["provider"], m["model"], m.get("name"), cloud, m.get("vision"), m.get("pricing")))
        for e in self.cfg.get("api_models") or []:
            if not isinstance(e, dict) or not e.get("provider") or not e.get("model"):
                continue
            bid = f"api:{e['provider']}/{e['model']}"
            if bid in seen:
                continue
            seen.add(bid)
            out.append(self._api_option(bid, e["provider"], e["model"], e.get("label"), cloud, e.get("vision"), None))
        return out

    def _api_option(self, bid, provider, model, name, cloud, vision, pricing) -> dict:
        base = self._provider_base(provider)
        local = bool(base and self._is_local_url(base))
        key = self._key(provider)
        reason = ""
        if not base:
            reason = "no endpoint URL configured"
        elif not local and not cloud:
            reason = "cloud calls are switched off (Providers page)"
        elif not local and not key:
            reason = f"no API key for {provider} (Providers page)"
        plabel = PROVIDER_LABEL.get(provider) or provider.split(":", 1)[-1]
        return {"id": bid, "kind": "api", "provider": provider, "label": f"{name or model}", "detail": f"{plabel} API",
                "model": model, "available": not reason, "reason": reason, "vision": bool(vision), "tools": "openai",
                "json_schema": True, "tested": True, "local": local, "pricing": pricing,
                "cost": "free (local endpoint)" if local else ("pay per token" if not pricing else
                                                                f"${pricing['input_per_mtok']:g} / ${pricing['output_per_mtok']:g} per M tokens")}

    def label(self, spec: dict) -> str:
        if spec["kind"] == "local":
            st = self.llm.status() if self.llm else {}
            return f"Local LLM ({(st.get('model') or 'llama-server').split(' (')[0]})"
        if spec["kind"] == "cli":
            m = spec.get("model") or ((self.cfg.get("cli") or {}).get(spec["name"]) or {}).get("model") or CLI_INFO[spec["name"]]["default_model"]
            return f"{CLI_INFO[spec['name']]['label']}{f' ({m})' if m else ''}"
        return f"{spec['model']} via {PROVIDER_LABEL.get(spec['provider']) or spec['provider'].split(':', 1)[-1]}"

    def resolve(self, backend: str | None, purpose: str = "write", settings=None) -> dict:
        spec = parse_backend(backend or self.default_for(purpose, settings))
        if spec["kind"] == "cli" and not spec.get("model"):
            spec["model"] = ((self.cfg.get("cli") or {}).get(spec["name"]) or {}).get("model") or CLI_INFO[spec["name"]]["default_model"] or None
        spec["label"] = self.label(spec)
        return spec

    def public_spec(self, spec: dict) -> dict:
        return {"id": spec["id"], "kind": spec["kind"], "label": spec.get("label") or self.label(spec), "model": spec.get("model"),
                "subscription": spec["kind"] == "cli", "tested": CLI_INFO.get(spec.get("name"), {}).get("tested", True) if spec["kind"] == "cli" else True}

    # ------------------------------------------------------------ generate
    def generate(self, *, messages: list[dict], schema: dict | None = None, backend: str | dict | None = None,
                 purpose: str = "write", temperature: float = 0.7, max_tokens: int = 2000, timeout: float | None = None,
                 cancel: threading.Event | None = None, project: str | None = None, confirm_cost: bool = False,
                 validate=None) -> dict:
        """messages: OpenAI-style (system + user/assistant turns; user content may hold image_url data URLs).
        validate(data) -> list[str] of extra content errors (optional; one retry on failure)."""
        view = self.core.settings_for(project) if (project and hasattr(self.core, "settings_for")) else None
        spec = backend if isinstance(backend, dict) else self.resolve(backend, purpose, view)   # project default when no backend
        timeout = float(timeout or self.cfg.get("timeout_s") or (300 if spec["kind"] == "cli" else 180))
        t0 = time.time()
        attempts, usd, last_err = 0, 0.0, ""
        msgs = list(messages)
        res = None
        for attempt in range(2):
            attempts += 1
            if spec["kind"] == "local":
                res = self._local(msgs, schema, temperature + 0.2 * attempt, max_tokens, timeout, cancel)
            elif spec["kind"] == "api":
                res = self._api(spec, msgs, schema, temperature + 0.2 * attempt, max_tokens, timeout, cancel, project, confirm_cost)
            else:
                res = self._cli(spec, msgs, schema, timeout, cancel)
            usd += res.get("usd") or 0
            if schema is None:
                break
            data = res.get("data")
            errs = []
            if data is None:
                try:
                    data = extract_json(res.get("text", ""))
                except ValueError as e:
                    errs = [str(e)]
            if data is not None and not errs:
                errs = validate_json(data, schema) + (list(validate(data)) if validate else [])
            res["data"] = data
            if not errs:
                break
            last_err = "; ".join(errs[:6])
            if attempt == 0:
                msgs = msgs + [{"role": "assistant", "content": res.get("text") or json.dumps(data, ensure_ascii=False)[:4000]},
                               {"role": "user", "content": f"That answer was not valid: {last_err}. Reply again with ONE corrected JSON object only."}]
        else:
            if schema is not None and res is not None:
                if res.get("data") is None:
                    raise TextGenError(f"The backend did not return valid JSON ({last_err}).", "bad_output",
                                       "Try again or pick another backend.")
                res.setdefault("warnings", []).append(f"schema: {last_err}")
        res.update(backend=self.public_spec(spec), seconds=round(time.time() - t0, 2), usd=round(usd, 6), attempts=attempts)
        _log(f"generate {spec['id']} purpose={purpose} attempts={attempts} {res['seconds']}s{f' retry: {last_err[:200]}' if attempts > 1 else ''}")
        return res

    # ------------------------------------------------------------ local llama-server
    def _local(self, messages, schema, temperature, max_tokens, timeout, cancel) -> dict:
        if not self.llm:
            raise TextGenError("The local LLM is not configured.", "unavailable")
        from .improver import ImproveError  # local import: improver imports this module lazily too
        try:
            self.llm.ensure()
        except ImproveError as e:
            raise TextGenError(str(e), "unavailable", e.fix) from None
        self.llm.last_request = time.time()
        r = self._openai_call(self.llm.url + "/v1", None, "prompt-llm", messages, schema, temperature, max_tokens,
                              timeout, cancel, extra={"chat_template_kwargs": {"enable_thinking": False}})
        self.llm.last_request = time.time()
        return r

    # ------------------------------------------------------------ OpenAI-compatible
    def _api(self, spec, messages, schema, temperature, max_tokens, timeout, cancel, project, confirm_cost) -> dict:
        provider, model = spec["provider"], spec["model"]
        base = self._provider_base(provider)
        if not base:
            raise TextGenError(f"No endpoint URL for {provider}.", "config", "Add it on the Providers page (or settings textgen.endpoints).")
        local = self._is_local_url(base)
        if not local and not self._cloud_allowed():
            raise TextGenError("Cloud calls are switched off.", "unavailable", "Enable cloud on the Providers page, or use a local / CLI backend.")
        key = self._key(provider)
        if not key and not local:
            raise TextGenError(f"No API key for {provider}.", "not_logged_in", "Add the key on the Providers page.")
        pricing = next((m["pricing"] for m in self.registry_api_models() if m["provider"] == provider and m["model"] == model), None)
        est = 0.0
        if pricing and not local:
            chars = sum(len(json.dumps(m.get("content"), ensure_ascii=False)) for m in messages)
            est = (chars / 3.5 * pricing["input_per_mtok"] + max_tokens * pricing["output_per_mtok"]) / 1e6
        if not local:
            self._spend_check(est, provider, confirm_cost)
        extra = {"usage": {"include": True}} if provider == "openrouter" else {}
        headers = {"X-Title": "Open Asset Studio"} if provider == "openrouter" else {}
        r = self._openai_call(base, key, model, messages, schema, temperature, max_tokens, timeout, cancel, extra=extra, headers=headers)
        usage = r.get("usage") or {}
        usd = usage.get("cost")
        if usd is None and pricing:
            usd = ((usage.get("prompt_tokens") or 0) * pricing["input_per_mtok"] + (usage.get("completion_tokens") or 0) * pricing["output_per_mtok"]) / 1e6
        r["usd"] = float(usd or 0)
        if not local:
            self._spend_record(r["usd"], provider, model, project)
        return r

    def _spend_check(self, est: float, provider: str, confirmed: bool):
        spend = getattr(_providers(self.core), "spend", None)
        f = getattr(spend, "check", None)
        if not callable(f):
            return
        try:
            try:
                f(est, provider, confirmed=confirmed) if confirmed else f(est, provider)
            except TypeError:
                f(est, provider)
        except Exception as e:  # noqa: BLE001
            kind = getattr(e, "kind", "")
            if kind in ("budget", "confirm"):
                raise TextGenError(str(e), kind, getattr(e, "fix", None)) from None
            raise

    def _spend_record(self, usd: float, provider: str, model: str, project: str | None):
        spend = getattr(_providers(self.core), "spend", None)
        f = getattr(spend, "record", None)
        if callable(f):
            try:
                f(usd, provider, model, project, None)
            except Exception:  # noqa: BLE001
                pass

    def _openai_call(self, base, key, model, messages, schema, temperature, max_tokens, timeout, cancel,
                     extra=None, headers=None, tools=None) -> dict:
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **(extra or {})}
        if tools:
            body["tools"] = tools
        hdr = {"Content-Type": "application/json", **(headers or {})}
        if key:
            hdr["Authorization"] = f"Bearer {key}"
        use_schema = bool(schema) and not tools
        if use_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": _strict_ok(schema), "schema": schema}}
        r = self._post(base + "/chat/completions", body, hdr, timeout, cancel)
        if use_schema and r.status_code == 400 and any(x in r.text.lower() for x in ("response_format", "json_schema", "schema")):
            body.pop("response_format")
            body["messages"] = _with_schema_prompt(messages, schema)
            r = self._post(base + "/chat/completions", body, hdr, timeout, cancel)
        if r.status_code != 200:
            raise _http_error(r)
        try:
            d = r.json()
            msg = d["choices"][0]["message"]
        except (ValueError, KeyError, IndexError):
            raise TextGenError(f"Unreadable answer from {base}: {r.text[:200]}", "bad_output") from None
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        out = {"text": text, "usage": d.get("usage") or {}, "message": msg, "data": None}
        if use_schema and "response_format" in body:
            try:
                out["data"] = json.loads(text)
            except ValueError:
                out["data"] = None
        return out

    def _post(self, url, body, headers, timeout, cancel) -> httpx.Response:
        box: dict = {}
        client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10))

        def go():
            try:
                box["r"] = client.post(url, json=body, headers=headers)
            except Exception as e:  # noqa: BLE001
                box["e"] = e
        th = threading.Thread(target=go, daemon=True)
        th.start()
        deadline = time.time() + timeout + 5
        while th.is_alive():
            th.join(0.25)
            if cancel is not None and cancel.is_set():
                client.close()
                raise TextGenError("Cancelled.", "cancelled", "")
            if time.time() > deadline:
                client.close()
                raise TextGenError("The backend did not answer in time.", "timeout")
        client.close()
        if "e" in box:
            e = box["e"]
            if isinstance(e, httpx.TimeoutException):
                raise TextGenError("The backend did not answer in time.", "timeout")
            raise TextGenError(f"Could not reach {urlparse(url).netloc}: {e}", "unavailable", "Check the endpoint / network.")
        return box["r"]

    def api_tool_chat(self, spec: dict, messages: list[dict], tools: list[dict], execute, *, max_rounds: int = 8,
                      temperature: float = 0.4, max_tokens: int = 2500, timeout: float = 180, cancel=None,
                      project=None, on_tool=None) -> dict:
        """OpenAI function-calling loop (local llama-server or an API backend). execute(name, args) -> result dict."""
        if spec["kind"] == "local":
            from .improver import ImproveError
            try:
                self.llm.ensure()
            except ImproveError as e:
                raise TextGenError(str(e), "unavailable", e.fix) from None
            base, key, model, extra, hdr = self.llm.url + "/v1", None, "prompt-llm", {"chat_template_kwargs": {"enable_thinking": False}}, {}
            local = True
        else:
            base = self._provider_base(spec["provider"])
            if not base:
                raise TextGenError(f"No endpoint URL for {spec['provider']}.", "config")
            local = self._is_local_url(base)
            if not local and not self._cloud_allowed():
                raise TextGenError("Cloud calls are switched off.", "unavailable")
            key, model = self._key(spec["provider"]), spec["model"]
            if not key and not local:
                raise TextGenError(f"No API key for {spec['provider']}.", "not_logged_in")
            extra = {"usage": {"include": True}} if spec["provider"] == "openrouter" else {}
            hdr = {"X-Title": "Open Asset Studio"} if spec["provider"] == "openrouter" else {}
        msgs = list(messages)
        usd, calls = 0.0, []
        t0 = time.time()
        for _ in range(max_rounds):
            if not local:
                self._spend_check(0.0, spec["provider"], False)
            if self.llm and spec["kind"] == "local":
                self.llm.last_request = time.time()
            r = self._openai_call(base, key, model, msgs, None, temperature, max_tokens, timeout, cancel, extra=extra, headers=hdr, tools=tools)
            u = (r.get("usage") or {}).get("cost")
            if u and not local:
                usd += float(u)
                self._spend_record(float(u), spec["provider"], model, project)
            m = r["message"]
            tcs = m.get("tool_calls") or []
            if not tcs:
                return {"text": r["text"], "tool_calls": calls, "usd": usd, "seconds": round(time.time() - t0, 2)}
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                name = fn.get("name", "")
                if on_tool:
                    on_tool(name, args)
                result = execute(name, args)
                calls.append({"name": name, "input": args})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(result, ensure_ascii=False, default=str)[:20000]})
        return {"text": "", "tool_calls": calls, "usd": usd, "seconds": round(time.time() - t0, 2), "truncated": True}

    # ------------------------------------------------------------ subscription CLIs
    def _cli(self, spec, messages, schema, timeout, cancel) -> dict:
        name = spec["name"]
        exe = self._which(name)
        if not exe:
            raise TextGenError(f"{CLI_INFO[name]['label']} is not installed (`{CLI_INFO[name]['binary']}` not found).", "not_installed")
        system, turns, images = _split_messages(messages)
        info = CLI_INFO[name]
        if not info["json_schema"] and schema:
            system += schema_instruction(schema)
        prompt = transcript(turns) if turns else ""
        wd = Path(tempfile.mkdtemp(prefix="studio-llm-"))
        try:
            img_paths = []
            for i, b in enumerate(images[:4]):
                p = wd / f"image{i + 1}{_img_ext(b)}"
                p.write_bytes(b)
                img_paths.append(p)
            argv, stdin = build_cli(name, prompt=prompt, system=system, model=spec.get("model"),
                                    schema=schema if info["json_schema"] else None, images=img_paths, workdir=wd,
                                    max_turns=3 if schema else 1, binary=exe)
            rc, out, err = run_cli(argv, stdin, wd, timeout, cancel)
            return self._parse_cli(name, rc, out, err, wd)
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def _parse_cli(self, name, rc, out, err, wd) -> dict:
        if name == "claude":
            p = parse_claude_stream(out.splitlines())
            if p["is_error"] or (rc != 0 and not p["text"]):
                msg = p["error"] or err.strip()[-400:] or f"claude exited with code {rc}"
                kind = classify_cli_error(msg + " " + err)
                if p.get("subtype") == "error_max_turns":
                    msg, kind = "Claude stopped at the turn limit before answering.", "bad_output"
                raise TextGenError(f"Claude Code: {msg[:400]}", kind)
            return {"text": p["text"], "data": p["structured"], "usd": 0.0, "plan_usd_equiv": p["usd"], "cli": p}
        if name == "codex":
            p = parse_codex_jsonl(out.splitlines())
            last = wd / "last_message.txt"
            text = last.read_text() if last.exists() else p["text"]
            if (rc != 0 and not text) or (p["error"] and not text):
                msg = p["error"] or err.strip()[-400:] or f"codex exited with code {rc}"
                raise TextGenError(f"Codex: {msg[:400]}", classify_cli_error(msg + " " + err))
            return {"text": text, "data": None, "usd": 0.0}
        if name == "gemini":
            try:
                d = extract_json(out)
            except ValueError:
                d = {}
            if d.get("error") or (rc != 0 and not d.get("response")):
                e = d.get("error") or {}
                msg = (e.get("message") if isinstance(e, dict) else str(e)) or err.strip()[-400:] or f"gemini exited with code {rc}"
                raise TextGenError(f"Gemini CLI: {msg[:400]}", classify_cli_error(msg + " " + err))
            return {"text": d.get("response") or "", "data": None, "usd": 0.0}
        if rc != 0 and not out.strip():
            msg = err.strip()[-400:] or f"copilot exited with code {rc}"
            raise TextGenError(f"Copilot CLI: {msg}", classify_cli_error(msg))
        return {"text": out.strip(), "data": None, "usd": 0.0}

    def claude_agent(self, *, system: str, prompt: str, mcp_config: dict, allowed_tools: list[str], model: str | None,
                     timeout: float, cancel=None, max_turns: int = 10, on_event=None) -> dict:
        """claude -p with ONLY the given MCP tools (no built-ins). Returns the parsed stream (text, tool_uses, init)."""
        exe = self._which("claude")
        if not exe:
            raise TextGenError("Claude Code is not installed.", "not_installed")
        wd = Path(tempfile.mkdtemp(prefix="studio-agent-"))
        try:
            cfgp = wd / "mcp.json"
            cfgp.write_text(json.dumps(mcp_config))
            argv, stdin = build_cli("claude", prompt=prompt, system=system, model=model, workdir=wd,
                                    mcp_config=str(cfgp), mcp_tools=allowed_tools, max_turns=max_turns, binary=exe)
            rc, out, err = run_cli(argv, stdin, wd, timeout, cancel, on_line=on_event)
            p = parse_claude_stream(out.splitlines())
            if p["is_error"] and p.get("subtype") != "error_max_turns":
                msg = p["error"] or err.strip()[-400:] or f"claude exited with code {rc}"
                raise TextGenError(f"Claude Code: {msg[:400]}", classify_cli_error(msg + " " + err))
            if rc != 0 and not p["text"] and not p["tool_uses"]:
                msg = err.strip()[-400:] or f"claude exited with code {rc}"
                raise TextGenError(f"Claude Code: {msg}", classify_cli_error(msg))
            return p
        finally:
            shutil.rmtree(wd, ignore_errors=True)


def run_cli(argv: list[str], stdin: bytes | None, cwd: Path, timeout: float, cancel: threading.Event | None = None,
            on_line=None) -> tuple[int, str, str]:
    """Run a CLI in its own process group (killed on timeout / cancel / studio exit). Returns (rc, stdout, stderr).
    claude stream-json: once the final `result` event is out, the process gets 3 s to exit, then it is stopped."""
    t_start = time.time()
    done_at: list[float] = []
    try:
        from .procs import spawn
    except ImportError:  # pragma: no cover
        spawn = subprocess.Popen
    try:
        p = spawn(argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE, cwd=str(cwd), env=cli_env(), start_new_session=True)
    except FileNotFoundError:
        raise TextGenError(f"`{argv[0]}` was not found.", "not_installed") from None
    out_lines: list[str] = []
    err_buf: list[bytes] = []

    def rd_out():
        for ln in iter(p.stdout.readline, b""):
            s = ln.decode("utf-8", "replace")
            out_lines.append(s)
            if '"type":"result"' in s and s.lstrip().startswith("{"):
                try:
                    if json.loads(s).get("type") == "result":
                        done_at.append(time.time())
                except ValueError:
                    pass
            if on_line:
                try:
                    on_line(s)
                except Exception:  # noqa: BLE001
                    pass

    def rd_err():
        err_buf.append(p.stderr.read())
    t1 = threading.Thread(target=rd_out, daemon=True)
    t2 = threading.Thread(target=rd_err, daemon=True)
    t1.start()
    t2.start()
    if stdin is not None:
        try:
            p.stdin.write(stdin)
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    deadline = time.time() + timeout
    killed = None
    while p.poll() is None:
        if cancel is not None and cancel.is_set():
            killed = "cancelled"
        elif time.time() > deadline:
            killed = "timeout"
        if killed:
            _kill_group(p)
            break
        if done_at and time.time() - done_at[0] > 3:
            _kill_group(p)
            break
        time.sleep(0.1)
    t1.join(3)
    t2.join(3)
    _log(f"{Path(argv[0]).name} rc={p.returncode} {time.time() - t_start:.1f}s"
         f"{f' (result after {done_at[0] - t_start:.1f}s)' if done_at else ''}{f' {killed}' if killed else ''}")
    if killed == "cancelled":
        raise TextGenError("Cancelled.", "cancelled", "")
    if killed == "timeout":
        raise TextGenError(f"`{Path(argv[0]).name}` did not finish within {timeout:.0f} s.", "timeout")
    return p.returncode, "".join(out_lines), b"".join(err_buf).decode("utf-8", "replace")


def _log(msg: str):
    try:
        from . import config
        with (config.LOGS / "textgen.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=4)
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired, OSError):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _strict_ok(schema: dict) -> bool:
    """OpenAI strict mode needs every property required and additionalProperties false, recursively."""
    if schema.get("type") == "object":
        props = schema.get("properties") or {}
        if schema.get("additionalProperties", True) is not False or set(schema.get("required") or []) != set(props):
            return False
        return all(_strict_ok(v) for v in props.values())
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        return _strict_ok(schema["items"])
    return True


def _with_schema_prompt(messages: list[dict], schema: dict) -> list[dict]:
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = str(msgs[0]["content"]) + schema_instruction(schema)
    else:
        msgs.insert(0, {"role": "system", "content": schema_instruction(schema).strip()})
    return msgs


def _http_error(r: httpx.Response) -> TextGenError:
    txt = r.text[:400]
    low = txt.lower()
    if r.status_code in (401, 403):
        return TextGenError(f"The API key was rejected (HTTP {r.status_code}).", "not_logged_in", "Check the key on the Providers page.")
    if r.status_code == 402 or "insufficient" in low or "credit" in low or "quota" in low:
        return TextGenError(f"Quota / credits exhausted (HTTP {r.status_code}): {txt[:200]}", "quota")
    if r.status_code == 429:
        return TextGenError("Rate limited (HTTP 429).", "rate_limited")
    if r.status_code == 404:
        return TextGenError(f"Model or endpoint not found (HTTP 404): {txt[:200]}", "config", "Check the model id.")
    return TextGenError(f"HTTP {r.status_code}: {txt}", "http", "")
