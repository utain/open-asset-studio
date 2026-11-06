#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""Open Asset Studio installer. Run it through ./install.sh (which checks git / ffmpeg / uv and runs this file with
uv-managed Python 3.12).

    ./install.sh                      core, then a menu of packs (default: starter)
    ./install.sh list [packs|backends|models]
    ./install.sh doctor [--check]
    ./install.sh core
    ./install.sh pack <name>...       (--weights: also download the weights of the pack's models)
    ./install.sh backend <id>...      (--weights: also the weights of all its models)
    ./install.sh model <model-id>...  backend + weights of that model
    options: --dry-run  --yes/-y  --weights  --force  --torch-index cu130|cu128

What it installs is described by install/<backend>.yaml (format: install/FORMAT.md) and install/packs.yaml. Weights
come from the model registry (models.yaml, models.d/*.yaml): the same `weights:` entries that decide ready /
missing_weights on the Models page.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
INSTALL_DIR = REPO / "install"
ENVS = REPO / "envs"
LOCKS = REPO / "env-lock"
VENDOR_LOCK = REPO / "vendor" / "vendor.lock"
MARKER = ".studio-install.json"

PYTORCH_INDEX = "https://download.pytorch.org/whl/{}"
SUPPORTED_INDEXES = ("cu130", "cu128")
KNOWN_PLATFORMS = ("linux-aarch64", "linux-x86_64")
# Backends of the registry that have no environment to install (a remote ComfyUI, the cloud providers plugin).
EXPECTED_MISSING = {"cloud": "cloud providers: no environment, add an API key in Settings",
                    "comfyui": "remote ComfyUI server (COMFYUI_URL): nothing to install here",
                    "music_levo": "blocked: the upstream SongGeneration code is gone, there is no environment to recreate"}
RECIPE_KEYS = {"backend", "env", "python", "torch", "lock", "lock_flags", "vendor", "steps", "check", "manual", "tested",
               "size_gb", "shares_env_with"}
TORCH_FAMILY = {"torch", "torchvision", "torchaudio", "torchcodec", "triton", "pytorch-triton"}
TORCH_NVIDIA = re.compile(r"^nvidia-(cublas|cuda-(cupti|nvrtc|runtime)|cudnn|cufft|cufile|curand|cusolver|cusparse|"
                          r"cusparselt|nccl|nvjitlink|nvshmem|nvtx)(-cu\d+)?$")
DL_PREFIX_ENV = "STUDIO_DL_PREFIX"      # e.g. "flock /path/dl.lock": prepended to every weights download
EXIT_OK, EXIT_FAIL, EXIT_STOPPED, EXIT_PREREQ = 0, 1, 2, 3


class InstallError(Exception):
    pass


class Stop(Exception):
    """A download hit 401/403/429: stop the whole run."""

    def __init__(self, message: str, code: int = EXIT_STOPPED):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------------------------------- output
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(msg: str) -> None:
    say()
    say(_c("1", f"==> {msg}"))


def warn(msg: str) -> None:
    say(_c("33", f"warning: {msg}"))


def err(msg: str) -> None:
    print(_c("31", f"error: {msg}"), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------- platform / GPU
def platform_tag() -> str:
    m = platform.machine().lower()
    m = {"arm64": "aarch64", "amd64": "x86_64"}.get(m, m)
    return f"{platform.system().lower()}-{m}"


def torch_index_for_cuda(cuda: str | None) -> str:
    """PyTorch wheel index for the CUDA version the driver supports (nvidia-smi 'CUDA Version'): CUDA >= 13 -> cu130,
    12.8 and 12.9 -> cu128, anything older is an error."""
    if not cuda:
        raise InstallError("could not read the CUDA version from nvidia-smi")
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?", str(cuda))
    if not m:
        raise InstallError(f"cannot parse CUDA version {cuda!r}")
    major, minor = int(m.group(1)), int(m.group(2) or 0)
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    raise InstallError(f"the NVIDIA driver supports CUDA {major}.{minor}, but the studio needs CUDA 12.8 or newer "
                       f"(PyTorch cu128 / cu130 wheels). Update the NVIDIA driver (R570+ for 12.8, R580+ for 13) and run again.")


def detect_gpu() -> dict:
    """{'ok': bool, 'name', 'driver', 'cuda', 'memory_gb', 'cc', 'error'} from nvidia-smi (no torch needed)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"ok": False, "error": "nvidia-smi not found: no NVIDIA driver installed (or not on PATH)"}
    try:
        head_out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
        q = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": f"nvidia-smi failed: {e}"}
    if q.returncode != 0 or not q.stdout.strip():
        return {"ok": False, "error": "nvidia-smi found no GPU: " + (q.stderr.strip() or head_out.strip().splitlines()[0] if head_out.strip() else "no output")}
    m = re.search(r"CUDA Version:\s*([\d.]+)", head_out)
    parts = [p.strip() for p in q.stdout.strip().splitlines()[0].split(",")]
    parts += [""] * (4 - len(parts))
    try:
        mem = round(float(parts[2]) / 1024, 1)
    except ValueError:
        mem = None   # unified-memory GPUs (DGX Spark) report [N/A]
    return {"ok": True, "name": parts[0], "driver": parts[1], "memory_gb": mem, "cc": parts[3] or "?",
            "cuda": m.group(1) if m else None}


def disk_free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def hf_cache_dir() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")


def hf_logged_in() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return any(p.is_file() and p.stat().st_size > 0 for p in (home / "token", Path.home() / ".huggingface/token"))


# --------------------------------------------------------------------------------------------------- recipes
def _is_list_of_str(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def vendor_entries() -> list[tuple[str, str, str]]:
    """[(path, url, commit)] of vendor/vendor.lock."""
    out = []
    if VENDOR_LOCK.is_file():
        for ln in VENDOR_LOCK.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                p = ln.split()
                if len(p) >= 3:
                    out.append((p[0], p[1], p[2]))
    return out


def vendor_matches(filters: list[str]) -> list[str]:
    """Paths of vendor.lock selected by scripts/fetch_vendor.sh filters (case-insensitive substring of the path)."""
    paths = [p for p, _, _ in vendor_entries()]
    if not filters:
        return []
    return [p for p in paths if any(f.lower() in p.lower() for f in filters)]


def validate_recipe(d, name: str, check_files: bool = True) -> list[str]:
    """Problems of one parsed recipe against install/FORMAT.md (empty list = valid)."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return [f"{name}: not a mapping"]
    steps_keys = {k for k in d if k.startswith("steps_")}
    for k in d:
        if k not in RECIPE_KEYS and k not in steps_keys:
            errs.append(f"{name}: unknown key '{k}'")
    for k in steps_keys:
        if k[len("steps_"):] not in KNOWN_PLATFORMS:
            errs.append(f"{name}: '{k}' is not one of steps_{', steps_'.join(KNOWN_PLATFORMS)}")
    be = d.get("backend")
    if not isinstance(be, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", be or ""):
        errs.append(f"{name}: 'backend' must be a lowercase id")
    elif be != Path(name).stem:
        errs.append(f"{name}: backend '{be}' does not match the file name")
    env = d.get("env", be)
    if not isinstance(env, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", env or ""):
        errs.append(f"{name}: 'env' must be a venv folder name")
    if str(d.get("python", "3.12")) != "3.12":
        errs.append(f"{name}: 'python' must be \"3.12\"")
    for k in ("torch", "lock_flags", "vendor", "steps", "manual", "tested") + tuple(sorted(steps_keys)):
        if k in d and d[k] is not None and not _is_list_of_str(d[k]):
            errs.append(f"{name}: '{k}' must be a list of non-empty strings")
    for t in d.get("torch") or []:
        if isinstance(t, str) and not re.fullmatch(r"[A-Za-z0-9_.-]+(==[A-Za-z0-9_.+]+)?", t):
            errs.append(f"{name}: torch entry '{t}' must be 'package' or 'package==version'")
    for f in d.get("lock_flags") or []:
        if isinstance(f, str) and not f.startswith("-"):
            errs.append(f"{name}: lock_flags entry '{f}' is not a flag")
    lock = d.get("lock")
    if lock is not None:
        if not isinstance(lock, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", lock):
            errs.append(f"{name}: 'lock' must be a name or null")
        elif check_files and not (LOCKS / f"{lock}.txt").is_file():
            errs.append(f"{name}: env-lock/{lock}.txt does not exist")
    if lock is None and d.get("lock_flags"):
        errs.append(f"{name}: lock_flags without a lock")
    if check_files and d.get("vendor"):
        for f in d["vendor"]:
            if isinstance(f, str) and not vendor_matches([f]):
                errs.append(f"{name}: vendor filter '{f}' matches nothing in vendor/vendor.lock")
    chk = d.get("check")
    if chk is not None:
        if not isinstance(chk, str):
            errs.append(f"{name}: 'check' must be a python one-liner")
        else:
            try:
                compile(chk, f"{name}:check", "exec")
            except SyntaxError as e:
                errs.append(f"{name}: 'check' is not valid python: {e.msg}")
    sz = d.get("size_gb")
    if sz is not None and (isinstance(sz, bool) or not isinstance(sz, (int, float)) or sz < 0):
        errs.append(f"{name}: 'size_gb' must be a number")
    for t in d.get("tested") or []:
        if isinstance(t, str) and t not in KNOWN_PLATFORMS:
            errs.append(f"{name}: tested platform '{t}' is not one of {list(KNOWN_PLATFORMS)}")
    sh = d.get("shares_env_with")
    if sh is not None and not isinstance(sh, str):
        errs.append(f"{name}: 'shares_env_with' must be a backend id or null")
    for key in ["steps"] + sorted(steps_keys):
        for s in d.get(key) or []:
            if not isinstance(s, str):
                continue
            if re.search(r"(^|[\s;&|(])sudo\s", s):
                errs.append(f"{name}: {key} uses sudo (recipes must not)")
            if re.search(r"hf\s+download|huggingface-cli|snapshot_download|hf_hub_download", s):
                errs.append(f"{name}: {key} downloads model weights (they come from the registry: ./install.sh model <id>)")
    own_env = bool(d.get("lock") or d.get("torch") or d.get("steps") or any(d.get(k) for k in steps_keys))
    if sh and (d.get("lock") or d.get("torch")):
        errs.append(f"{name}: shares_env_with {sh} but also has its own lock/torch")
    if not sh and not own_env and not d.get("manual"):
        errs.append(f"{name}: nothing to do: needs a lock, torch, steps, shares_env_with or manual entries")
    return errs


def normalize_recipe(d: dict) -> dict:
    r = dict(d)
    r["env"] = d.get("env") or d["backend"]
    r["python"] = str(d.get("python") or "3.12")
    for k in ("torch", "lock_flags", "vendor", "steps", "manual", "tested"):
        r[k] = list(d.get(k) or [])
    r["lock"] = d.get("lock")
    r["check"] = d.get("check")
    r["size_gb"] = float(d.get("size_gb") or 0)
    r["shares_env_with"] = d.get("shares_env_with")
    r["platform_steps"] = {k[len("steps_"):]: list(v or []) for k, v in d.items() if k.startswith("steps_")}
    return r


def recipe_kind(r: dict) -> str:
    """'shared' (uses another backend's venv), 'env' (own venv), 'manual' (nothing the installer can do)."""
    if r["shares_env_with"]:
        return "shared"
    if r["lock"] or r["torch"] or r["steps"] or any(r["platform_steps"].values()):
        return "env"
    return "manual"


def load_recipes(check_files: bool = True) -> tuple[dict[str, dict], list[str]]:
    recipes: dict[str, dict] = {}
    errors: list[str] = []
    for p in sorted(INSTALL_DIR.glob("*.yaml")):
        if p.name in ("packs.yaml", "sizes.yaml"):
            continue
        try:
            d = yaml.safe_load(p.read_text())
        except yaml.YAMLError as e:
            errors.append(f"{p.name}: YAML error: {e}")
            continue
        errs = validate_recipe(d, p.name, check_files)
        errors += errs
        if not errs:
            r = normalize_recipe(d)
            r["_hash"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            recipes[r["backend"]] = r
    for b, r in recipes.items():
        sh = r["shares_env_with"]
        if sh:
            if sh not in recipes:
                errors.append(f"{b}.yaml: shares_env_with '{sh}' has no recipe")
            elif r["env"] != recipes[sh]["env"]:
                errors.append(f"{b}.yaml: env '{r['env']}' differs from the env of {sh} ('{recipes[sh]['env']}')")
            elif recipes[sh]["shares_env_with"]:
                errors.append(f"{b}.yaml: shares_env_with chains are not supported ({sh} shares too)")
    return recipes, errors


def load_packs() -> dict[str, dict]:
    p = INSTALL_DIR / "packs.yaml"
    d = yaml.safe_load(p.read_text()) if p.is_file() else {}
    packs = (d or {}).get("packs") or {}
    for name, pk in packs.items():
        pk.setdefault("title", name)
        pk.setdefault("description", "")
        pk["models"] = list(pk.get("models") or [])
        pk["backends"] = list(pk.get("backends") or [])
        pk["include"] = list(pk.get("include") or [])
    return packs


def load_sizes() -> dict:
    p = INSTALL_DIR / "sizes.yaml"
    d = (yaml.safe_load(p.read_text()) if p.is_file() else {}) or {}
    return {"weights_gb": d.get("weights_gb") or {}, "gated": set(d.get("gated") or [])}


def load_registry():
    sys.path.insert(0, str(REPO))
    from app.registry import Registry   # noqa: PLC0415
    return Registry()


# --------------------------------------------------------------------------------------------------- lock handling
def lock_has_torch(text: str) -> bool:
    return any(re.match(r"^torch(==|\s|$)", ln) for ln in text.splitlines())


def filter_lock(text: str, torch_index: str, has_torch: bool) -> str:
    """The env-lock files were frozen on cu130. On cu130 (or for envs without torch, e.g. JAX) they are used as they are.
    On cu128 the torch family, its nvidia-* libraries and triton are dropped (the torch step brought matching ones) and
    '+cuNNN' local version suffixes are removed."""
    if torch_index == "cu130" or not has_torch:
        return text
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith(("#", "-", ".")):
            name = re.split(r"[=<>!~ @\[;]", s, maxsplit=1)[0].lower().replace("_", "-")
            if name in TORCH_FAMILY or TORCH_NVIDIA.match(name):
                continue
            ln = re.sub(r"\+cu\d+\b", "", ln)
        out.append(ln)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, dry_run=False, yes=False, weights=False, force=False, torch_index=None, check=False):
        self.dry_run, self.yes, self.weights, self.force, self.check = dry_run, yes, weights, force, check
        self.torch_index_opt = torch_index or os.environ.get("STUDIO_TORCH_INDEX")
        self._index: str | None = None
        self.gpu: dict | None = None
        self.recipes: dict[str, dict] = {}
        self.reg = None
        self.packs: dict[str, dict] = {}
        self.sizes: dict = {"weights_gb": {}, "gated": set()}
        self.installed: list[str] = []
        self.skipped: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.manual: list[tuple[str, str]] = []
        self.weights_done: list[str] = []
        self.weights_skipped: list[str] = []
        self.weights_failed: list[tuple[str, str]] = []
        self.weights_manual: list[tuple[str, str]] = []

    def load(self, need_registry=True):
        self.recipes, errs = load_recipes()
        for e in errs:
            warn(f"recipe problem: {e}")
        self.packs = load_packs()
        self.sizes = load_sizes()
        if need_registry:
            self.reg = load_registry()
        return self

    def torch_index(self) -> str:
        """The PyTorch index to use: --torch-index / STUDIO_TORCH_INDEX, else from nvidia-smi."""
        if self._index:
            return self._index
        if self.torch_index_opt:
            if self.torch_index_opt not in SUPPORTED_INDEXES:
                raise InstallError(f"--torch-index must be one of {', '.join(SUPPORTED_INDEXES)}")
            self._index = self.torch_index_opt
            return self._index
        self.gpu = self.gpu or detect_gpu()
        if not self.gpu["ok"]:
            if self.dry_run:
                warn(self.gpu["error"] + "; the dry run assumes cu130")
                self._index = "cu130"
                return self._index
            raise InstallError(self.gpu["error"] + ". The studio needs an NVIDIA GPU with a driver that supports CUDA 12.8+.")
        try:
            self._index = torch_index_for_cuda(self.gpu.get("cuda"))
        except InstallError:
            if self.dry_run:
                warn("CUDA version unknown/too old; the dry run assumes cu130")
                self._index = "cu130"
                return self._index
            raise
        return self._index


# --------------------------------------------------------------------------------------------------- running things
def base_env() -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("UV_HTTP_TIMEOUT", "900")
    env["HF_HUB_DISABLE_XET"] = "1"
    env.setdefault("CMAKE_POLICY_VERSION_MINIMUM", "3.5")  # old sdists (sphn, pyworld) vs cmake 4
    return env


def run(ctx: Ctx, cmd: list[str] | str, env: dict | None = None, cwd: Path = REPO, shell: bool = False,
        capture: bool = False) -> subprocess.CompletedProcess | None:
    shown = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    if "\n" in shown:
        shown = shown.strip().replace("\n", "\n    ")
    say(f"  $ {shown}")
    if ctx.dry_run:
        return None
    if shell:
        return subprocess.run(["bash", "-euo", "pipefail", "-c", cmd], env=env or base_env(), cwd=cwd,
                              capture_output=capture, text=True)
    return subprocess.run(cmd, env=env or base_env(), cwd=cwd, capture_output=capture, text=True)


def _rel(p: Path) -> str:
    """Path relative to the repo when it is inside it (shorter output), else absolute."""
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def env_dir(recipe: dict) -> Path:
    return ENVS / recipe["env"]


def env_python(recipe: dict) -> Path:
    return env_dir(recipe) / "bin" / "python"


def read_marker(recipe: dict) -> dict | None:
    try:
        return json.loads((env_dir(recipe) / MARKER).read_text())
    except (OSError, ValueError):
        return None


def backend_state(recipe: dict) -> str:
    """installed | linked | present | broken-link | missing | manual   (no commands are run)."""
    kind = recipe_kind(recipe)
    if kind == "manual":
        return "manual"
    d = env_dir(recipe)
    if d.is_symlink():
        return "linked" if env_python(recipe).exists() else "broken-link"
    if not env_python(recipe).exists():
        return "missing"
    mk = read_marker(recipe)
    if mk and mk.get("hash") == recipe["_hash"]:
        return "installed"
    return "present"


# --------------------------------------------------------------------------------------------------- backend install
def order_with_deps(ctx: Ctx, ids: list[str]) -> list[str]:
    out: list[str] = []

    def add(b: str):
        r = ctx.recipes.get(b)
        if r is None:
            raise InstallError(f"no install recipe for backend '{b}' (install/{b}.yaml)")
        if r["shares_env_with"]:
            add(r["shares_env_with"])
        if b not in out:
            out.append(b)
    for b in ids:
        add(b)
    return out


def fetch_vendor(ctx: Ctx, recipe: dict) -> None:
    filters = recipe["vendor"]
    if not filters:
        return
    todo = [p for p in vendor_matches(filters) if not (REPO / p / ".git").exists()]
    if not todo and not ctx.force:
        say(f"  = vendor present: {', '.join(vendor_matches(filters))}")
        return
    say(f"  vendor clones: {', '.join(todo) or '(already present, re-checking commits)'}")
    r = run(ctx, ["bash", "scripts/fetch_vendor.sh", *filters])
    if r is not None and r.returncode != 0:
        raise InstallError("scripts/fetch_vendor.sh failed (see the output above)")


def step_env(ctx: Ctx, recipe: dict) -> dict:
    env = base_env()
    venv = env_dir(recipe)
    env.update(VIRTUAL_ENV=str(venv), PY=str(venv / "bin" / "python"), REPO=str(REPO),
               PATH=f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}")
    try:
        env["TORCH_INDEX"] = ctx.torch_index()
    except InstallError:
        pass
    return env


def install_backend(ctx: Ctx, bid: str) -> str:
    """Install one backend (its shared-env dependency must already be installed). Returns the final state."""
    r = ctx.recipes[bid]
    kind = recipe_kind(r)
    head(f"backend {bid}" + (f" (venv of {r['shares_env_with']})" if kind == "shared" else f" (envs/{r['env']})"))
    if kind == "manual":
        say("  This backend cannot be installed automatically. Manual steps:")
        for m in r["manual"]:
            say(f"    - {m}")
            ctx.manual.append((bid, m))
        ctx.skipped.append(f"{bid} (manual install)")
        return "manual"
    state = backend_state(r)
    if state == "linked":
        say(f"  = envs/{r['env']} is a symlink to another install: left untouched")
        ctx.skipped.append(f"{bid} (linked env)")
        return state
    if state == "broken-link":
        raise InstallError(f"envs/{r['env']} is a symlink to a missing folder; remove it and run again")
    if state == "installed" and not ctx.force:
        say(f"  = already installed (envs/{r['env']})")
        ctx.skipped.append(f"{bid} (already installed)")
        _collect_manual(ctx, bid, r)
        return state
    if state == "present" and not ctx.force and r["check"] and not ctx.dry_run:
        probe = subprocess.run([str(env_python(r)), "-c", r["check"]], capture_output=True, text=True, cwd=REPO,
                               env={**base_env(), "HF_HUB_OFFLINE": "1"})
        if probe.returncode == 0:
            say(f"  = envs/{r['env']} exists and passes its check: adopted")
            _write_marker(ctx, r)
            ctx.skipped.append(f"{bid} (existing env adopted)")
            _collect_manual(ctx, bid, r)
            return "installed"
    if state == "installed" and ctx.force:
        say("  --force: running the recipe again")
    elif state == "present":
        say("  envs/%s exists but is not complete: running the recipe (uv skips what is installed)" % r["env"])

    index = ctx.torch_index() if (r["torch"] or r["lock"]) else None
    py = r["python"]
    venv = env_dir(r)
    if kind == "env":
        fetch_vendor(ctx, r)
        if not env_python(r).exists():
            if not ctx.dry_run:
                ENVS.mkdir(parents=True, exist_ok=True)
            res = run(ctx, ["uv", "venv", "--python", py, _rel(venv)])
            if res is not None and res.returncode != 0:
                raise InstallError("uv venv failed")
        env = step_env(ctx, r)
        if r["torch"]:
            cmd = ["uv", "pip", "install", "--index-url", PYTORCH_INDEX.format(index), *r["torch"]]
            res = run(ctx, cmd, env=env)
            if res is not None and res.returncode != 0:
                raise InstallError(f"installing {' '.join(r['torch'])} from the PyTorch {index} index failed")
        if r["lock"]:
            if "--no-build-isolation" in r["lock_flags"]:
                res = run(ctx, ["uv", "pip", "install", "setuptools", "wheel"], env=env)   # build tools for the lock's sdists
                if res is not None and res.returncode != 0:
                    raise InstallError("installing setuptools / wheel failed")
            text = (LOCKS / f"{r['lock']}.txt").read_text()
            flt = filter_lock(text, index or "cu130", lock_has_torch(text))
            lock_path = REPO / "env-lock" / f"{r['lock']}.txt"
            if flt != text:
                lock_path = venv / ".lock.filtered.txt"
                say(f"  (lock filtered for {index}: torch family and nvidia-* libraries come from the torch step)")
                if not ctx.dry_run:
                    lock_path.write_text(flt)
            res = run(ctx, ["uv", "pip", "install", *r["lock_flags"], "-r", _rel(lock_path)], env=env)
            if res is not None and res.returncode != 0:
                raise InstallError(f"installing env-lock/{r['lock']}.txt failed")
    else:   # shares another backend's venv: only its own vendor clones, steps and check
        fetch_vendor(ctx, r)
        env = step_env(ctx, r)
    steps = list(r["steps"]) + list(r["platform_steps"].get(platform_tag(), []))
    for i, st in enumerate(steps, 1):
        say(f"  step {i}/{len(steps)}")
        res = run(ctx, st, env=env, shell=True)
        if res is not None and res.returncode != 0:
            raise InstallError(f"step {i} failed (exit {res.returncode})")
    if r["check"]:
        if ctx.dry_run:
            say(f"  check: {r['check']}")
        else:
            res = subprocess.run([str(env_python(r)), "-c", r["check"]], capture_output=True, text=True, cwd=REPO,
                                 env={**base_env(), "HF_HUB_OFFLINE": "1"})
            if res.returncode != 0:
                tail = "\n".join((res.stderr or res.stdout).strip().splitlines()[-6:])
                raise InstallError(f"installed, but the import check failed:\n{tail}")
            say("  check ok")
    _write_marker(ctx, r)
    ctx.installed.append(bid)
    _collect_manual(ctx, bid, r)
    return "installed"


def _collect_manual(ctx: Ctx, bid: str, r: dict) -> None:
    for m in r["manual"]:
        if (bid, m) not in ctx.manual:
            ctx.manual.append((bid, m))


def _write_marker(ctx: Ctx, r: dict) -> None:
    if ctx.dry_run or env_dir(r).is_symlink():
        return
    try:
        (env_dir(r) / MARKER).write_text(json.dumps({"backend": r["backend"], "hash": r["_hash"], "torch_index": ctx._index,
                                                    "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    except OSError:
        pass


def install_backends(ctx: Ctx, ids: list[str]) -> bool:
    """Install in dependency order; one failure does not stop the others. True if all went well."""
    ok = True
    done: set[str] = set()
    for bid in order_with_deps(ctx, ids):
        dep = ctx.recipes[bid]["shares_env_with"]
        if dep and dep in {b for b, _ in ctx.failed}:
            ctx.failed.append((bid, f"needs {dep}, which failed"))
            ok = False
            continue
        try:
            install_backend(ctx, bid)
            done.add(bid)
        except InstallError as e:
            err(f"{bid}: {e}")
            ctx.failed.append((bid, str(e).splitlines()[0]))
            ok = False
    return ok


# --------------------------------------------------------------------------------------------------- weights
class Download:
    def __init__(self, repo: str):
        self.repo = repo
        self.patterns: list[str] | None = []   # [] = nothing yet, None = whole repo
        self.models: list[str] = []

    def add(self, files, model_id: str):
        if model_id not in self.models:
            self.models.append(model_id)
        if self.patterns is None:
            return
        if not files:
            self.patterns = None
        else:
            for f in files:
                if f not in self.patterns:
                    self.patterns.append(f)

    def fix(self) -> str:
        inc = "" if not self.patterns else "".join(f" --include '{p}'" for p in self.patterns)
        return f"hf download {self.repo}{inc}"


def plan_weights(ctx: Ctx, model_ids: list[str]) -> tuple[list[Download], list[tuple[str, dict, str]], list[tuple[str, str]]]:
    """(hf downloads still needed, path downloads with a download_cmd still needed, manual notes), merged per repo."""
    reg = ctx.reg
    hf: dict[str, Download] = {}
    cmds: list[tuple[str, dict, str]] = []
    notes: list[tuple[str, str]] = []
    for mid in model_ids:
        m = reg.models[mid]
        for w in m["weights"]:
            present, what, fix = reg._check_weight(w)
            if present:
                continue
            if "hf" in w:
                hf.setdefault(w["hf"], Download(w["hf"])).add(w.get("files"), mid)
            elif w.get("download_cmd"):
                if not any(c[1] is w or c[1].get("download_cmd") == w["download_cmd"] for c in cmds):
                    cmds.append((mid, w, what))
            else:
                notes.append((mid, f"{what}: {fix}" if fix else what))
    return list(hf.values()), cmds, notes


def fetch_repo(ctx: Ctx, d: Download) -> None:
    """Download one repo. Raises Stop on 401/403/429."""
    py = ENVS / "core" / "bin" / "python"
    cmd = [str(py), "scripts/hf_fetch.py", d.repo]
    if d.patterns:
        cmd += ["--include", *d.patterns]
    prefix = shlex.split(os.environ.get(DL_PREFIX_ENV, ""))
    env = base_env()
    env.update(HF_HUB_OFFLINE="0", HF_HUB_DISABLE_XET="1", HF_HUB_ENABLE_HF_TRANSFER="0")
    head(f"weights {d.repo}" + (f"  ({len(d.patterns)} file pattern{'s' if len(d.patterns) != 1 else ''})" if d.patterns else "  (whole repo)")
         + f"  for {', '.join(d.models)}")
    res = run(ctx, [*prefix, *cmd], env=env)
    if res is None:
        return
    code = res.returncode
    if code == 0:
        ctx.weights_done.append(d.repo)
        return
    url = f"https://huggingface.co/{d.repo}"
    if code == 10:
        msg = (f"Hugging Face refused access to {d.repo} (HTTP 401/403): the repository is gated or private.\n"
               f"  1. Open {url} and accept its terms (you need a free Hugging Face account).\n"
               f"  2. Log in once:  envs/core/bin/hf auth login    (or export HF_TOKEN=...)\n"
               f"  3. Run the same command again: finished downloads are skipped.")
        ctx.weights_failed.append((d.repo, "gated (HTTP 401/403)"))
        raise Stop(msg)
    if code == 11:
        ctx.weights_failed.append((d.repo, "rate limited (HTTP 429)"))
        raise Stop(f"Hugging Face is rate limiting this machine (HTTP 429) while downloading {d.repo}.\n"
                   f"  Stop here and retry in a few minutes (or later): downloads already finished are kept and skipped.")
    if code == 12:
        ctx.weights_failed.append((d.repo, "not found (HTTP 404)"))
        err(f"{d.repo}: not found on Hugging Face (HTTP 404). The registry entry may be outdated: {url}")
        return
    ctx.weights_failed.append((d.repo, f"download failed (exit {code})"))
    err(f"{d.repo}: download failed (exit {code}); see the output above. Run again to retry.")


def fetch_weights(ctx: Ctx, model_ids: list[str]) -> None:
    """Download what the registry says is missing for these models, one repo at a time, skipping what is cached."""
    hf, cmds, notes = plan_weights(ctx, model_ids)
    already = [m for m in model_ids if not any(m in d.models for d in hf) and not any(c[0] == m for c in cmds)
               and not any(n[0] == m for n in notes)]
    for m in already:
        ctx.weights_skipped.append(m)
    if already:
        say(f"  weights already present for: {', '.join(already)}")
    if hf or cmds:
        core_py = ENVS / "core" / "bin" / "python"
        if hf and not core_py.exists() and not ctx.dry_run:
            raise InstallError("envs/core is missing: run ./install.sh core first")
    for d in hf:
        fetch_repo(ctx, d)
    for mid, w, what in cmds:
        head(f"weights {what}  for {mid}")
        cmd = [str(c) for c in w["download_cmd"]]
        env = base_env()
        env.update(HF_HUB_OFFLINE="0", HF_HUB_DISABLE_XET="1", HF_HUB_ENABLE_HF_TRANSFER="0")
        prefix = shlex.split(os.environ.get(DL_PREFIX_ENV, ""))
        res = run(ctx, [*prefix, *cmd], env=env)
        if res is not None:
            if res.returncode == 0:
                ctx.weights_done.append(what)
            else:
                ctx.weights_failed.append((what, f"download command failed (exit {res.returncode})"))
                err(f"{what}: the download command failed (exit {res.returncode})")
    for mid, note in notes:
        ctx.weights_manual.append((mid, note))
        say(f"  manual: {mid}: {note}")


# --------------------------------------------------------------------------------------------------- planning / sizes
def pack_models(ctx: Ctx, name: str, _seen: tuple = ()) -> tuple[list[str], list[str]]:
    """(model ids, extra backend ids) of a pack, includes and `everything` expanded."""
    pk = ctx.packs.get(name)
    if pk is None:
        raise InstallError(f"unknown pack '{name}'. Packs: {', '.join(ctx.packs)}")
    if name in _seen:
        raise InstallError(f"pack include loop at '{name}'")
    models = list(pk["models"])
    backends = list(pk["backends"])
    for inc in pk["include"]:
        m, b = pack_models(ctx, inc, _seen + (name,))
        models += m
        backends += b
    if pk.get("everything"):
        for mid in installable_models(ctx):
            models.append(mid)
    seen: set[str] = set()
    models = [m for m in models if not (m in seen or seen.add(m))]
    backends = [b for b in dict.fromkeys(backends)]
    return models, backends


def installable_models(ctx: Ctx) -> list[str]:
    """Registry models the installer can set up: not blocked, not cloud / service, backend has an env recipe."""
    out = []
    for mid, m in ctx.reg.models.items():
        if model_problem(ctx, mid) is None and recipe_kind(ctx.recipes[m["backend"]]) != "manual":
            out.append(mid)
    return out


def model_problem(ctx: Ctx, mid: str) -> str | None:
    """Why this model cannot be installed (or None)."""
    m = ctx.reg.models.get(mid)
    if m is None:
        return f"unknown model '{mid}'"
    blocked = (m.get("features") or {}).get("blocked")
    if blocked:
        return f"blocked: {blocked}"
    b = ctx.reg.backends.get(m.get("backend", ""), {})
    if b.get("kind") == "cloud":
        return "cloud model: add the provider's API key in Settings, nothing to install"
    if m["backend"] in EXPECTED_MISSING:
        return EXPECTED_MISSING[m["backend"]]
    if m["backend"] not in ctx.recipes:
        return f"backend '{m['backend']}' has no install recipe"
    return None


def backends_for_models(ctx: Ctx, model_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ctx.reg.models[m]["backend"] for m in model_ids))


def models_of_backend(ctx: Ctx, bid: str) -> list[str]:
    return [mid for mid, m in ctx.reg.models.items() if m.get("backend") == bid and model_problem(ctx, mid) is None]


def weights_gb_missing(ctx: Ctx, model_ids: list[str]) -> tuple[float, int]:
    """(GB still to download according to install/sizes.yaml, number of models without a known size)."""
    total, unknown = 0.0, 0
    seen_repos: set[str] = set()
    for mid in model_ids:
        m = ctx.reg.models[mid]
        missing = [w for w in m["weights"] if not ctx.reg._check_weight(w)[0]]
        if not missing:
            continue
        gb = ctx.sizes["weights_gb"].get(mid)
        if gb is None:
            unknown += 1
            continue
        key = tuple(sorted(w.get("hf", "") for w in missing))
        if key in seen_repos:
            continue   # two models sharing the same missing repos (e.g. the two Hunyuan3D-2mini entries)
        seen_repos.add(key)
        total += float(gb)
    return total, unknown


def envs_gb_missing(ctx: Ctx, bids: list[str]) -> float:
    total = 0.0
    seen_env: set[str] = set()
    for b in bids:
        r = ctx.recipes[b]
        if recipe_kind(r) != "env" or r["env"] in seen_env:
            continue
        seen_env.add(r["env"])
        if backend_state(r) in ("missing", "present"):
            total += r["size_gb"]
    return total


def pack_summary(ctx: Ctx, name: str) -> dict:
    models, extra = pack_models(ctx, name)
    ok_models = [m for m in models if m in ctx.reg.models and model_problem(ctx, m) is None]
    bids = order_with_deps(ctx, backends_for_models(ctx, ok_models) + extra)
    env_gb = envs_gb_missing(ctx, bids)
    wgb, _unknown = weights_gb_missing(ctx, ok_models)
    gated = [m for m in ok_models if m in ctx.sizes["gated"]]
    states = [backend_state(ctx.recipes[b]) for b in bids]
    done = sum(s in ("installed", "linked", "present") for s in states)
    return {"models": ok_models, "backends": bids, "env_gb": env_gb, "weights_gb": wgb, "gated": gated,
            "installed": done, "total": len(bids), "manual": [b for b in bids if recipe_kind(ctx.recipes[b]) == "manual"]}


def check_disk(ctx: Ctx, need_env_gb: float, need_weights_gb: float) -> None:
    free_repo = disk_free_gb(REPO)
    cache = hf_cache_dir()
    same = False
    try:
        same = os.stat(REPO).st_dev == os.stat(cache if cache.exists() else cache.parent if cache.parent.exists() else Path.home()).st_dev
    except OSError:
        pass
    say(f"  disk: need about {need_env_gb:.0f} GB for environments + {need_weights_gb:.0f} GB for weights; "
        f"free {free_repo:.0f} GB in {REPO}" + ("" if same else f", {disk_free_gb(cache):.0f} GB in {cache}"))
    if same:
        if need_env_gb + need_weights_gb > free_repo:
            raise InstallError(f"not enough disk space ({free_repo:.0f} GB free, about {need_env_gb + need_weights_gb:.0f} GB needed)")
    else:
        if need_env_gb > free_repo:
            raise InstallError(f"not enough disk space in {REPO} ({free_repo:.0f} GB free, about {need_env_gb:.0f} GB needed)")
        if need_weights_gb > disk_free_gb(cache):
            raise InstallError(f"not enough disk space in {cache} ({disk_free_gb(cache):.0f} GB free, about {need_weights_gb:.0f} GB needed)")


def confirm(ctx: Ctx, question: str, default: bool = True) -> bool:
    if ctx.yes or ctx.dry_run:
        return True
    if not sys.stdin.isatty():
        raise InstallError("not an interactive terminal: pass --yes to run without questions")
    ans = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not ans else ans.startswith("y")


# --------------------------------------------------------------------------------------------------- commands
# espeak-ng (used by Kokoro, StyleTTS2, Zonos, ...) cannot open its data files when the path to them is longer than
# about 160 characters; the data sit in <repo>/envs/<env>/lib/python3.12/site-packages/espeakng_loader/espeak-ng-data
# (about 85 characters below the repo root).
MAX_REPO_PATH = 70


def long_path_warning() -> str | None:
    if len(str(REPO)) <= MAX_REPO_PATH:
        return None
    return (f"the install path is {len(str(REPO))} characters long ({REPO}). espeak-ng, used by Kokoro / StyleTTS2 / Zonos, "
            f"fails to load its data when the full path exceeds about 160 characters: clone to a shorter path "
            f"(at most {MAX_REPO_PATH} characters) if those models report 'Error processing file ... phontab'")


def prereqs(ctx: Ctx, need_gpu: bool = True) -> None:
    """Hard prerequisites for installing (install.sh checked git / ffmpeg / uv already; this also covers a direct run)."""
    missing = [t for t in ("git", "uv", "ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing and not ctx.dry_run:
        raise InstallError(f"missing tools: {', '.join(missing)} (see ./install.sh doctor)")
    if platform_tag() not in KNOWN_PLATFORMS:
        raise InstallError(f"unsupported platform {platform_tag()}: the studio needs Linux (x86_64 or aarch64) with an NVIDIA GPU")
    lp = long_path_warning()
    if lp:
        warn(lp)
    if need_gpu:
        idx = ctx.torch_index()
        g = ctx.gpu
        if g and g.get("ok"):
            say(f"  GPU: {g['name']}, driver {g['driver']}, CUDA {g['cuda']} -> PyTorch wheels {idx}")
        else:
            say(f"  PyTorch wheels: {idx}")
        if idx == "cu128":
            warn("cu128 wheels are supported but the pinned environments were frozen and tested on cu130; "
                 "the JAX music backend (music_magenta) needs CUDA 13")


def ensure_core(ctx: Ctx) -> bool:
    return install_backends(ctx, ["core"])


def run_plan(ctx: Ctx, backends: list[str], model_ids: list[str], with_weights: bool, label: str) -> int:
    """Shared tail of core / pack / backend / model: plan, disk check, confirm, install, weights, summary."""
    bids = order_with_deps(ctx, ["core"] + backends)
    env_gb = envs_gb_missing(ctx, bids)
    w_gb, w_unknown = (weights_gb_missing(ctx, model_ids) if with_weights else (0.0, 0))
    head(f"plan: {label}")
    say(f"  backends: {', '.join(bids)}")
    if with_weights:
        say(f"  weights: {len(model_ids)} model(s), about {w_gb:.0f} GB still to download"
            + (f" (+ {w_unknown} model(s) of unknown size)" if w_unknown else ""))
    else:
        say("  weights: not downloaded (add --weights, or run ./install.sh model <id> later)")
    if not ctx.dry_run:
        check_disk(ctx, env_gb, w_gb)
    else:
        say(f"  (dry run) would need about {env_gb:.0f} GB for environments and {w_gb:.0f} GB for weights; free {disk_free_gb(REPO):.0f} GB")
    if not confirm(ctx, "Continue?"):
        say("Aborted.")
        return EXIT_FAIL
    ok = install_backends(ctx, bids)
    if with_weights and model_ids:
        bad = {b for b, _ in ctx.failed}
        todo = [m for m in model_ids if ctx.reg.models[m]["backend"] not in bad]
        for m in model_ids:
            if m not in todo:
                ctx.weights_failed.append((m, "its backend failed to install"))
        try:
            head("model weights" + (" (dry run)" if ctx.dry_run else ""))
            fetch_weights(ctx, todo)
        except Stop as s:
            summary(ctx)
            err(str(s))
            return s.code
        ok = ok and not ctx.weights_failed
    summary(ctx)
    return EXIT_OK if ok else EXIT_FAIL


def summary(ctx: Ctx) -> None:
    head("summary" + (" (dry run: nothing was changed)" if ctx.dry_run else ""))
    if ctx.installed:
        say(f"  installed: {', '.join(ctx.installed)}")
    if ctx.skipped:
        say(f"  skipped:   {', '.join(ctx.skipped)}")
    if ctx.failed:
        say(_c("31", "  FAILED backends:"))
        for b, why in ctx.failed:
            say(f"    - {b}: {why}")
    if ctx.weights_done:
        say(f"  weights downloaded: {', '.join(ctx.weights_done)}")
    if ctx.weights_skipped:
        say(f"  weights already in the cache for: {', '.join(ctx.weights_skipped)}")
    if ctx.weights_failed:
        say(_c("31", "  FAILED weights:"))
        for w, why in ctx.weights_failed:
            say(f"    - {w}: {why}")
    if ctx.manual or ctx.weights_manual:
        say()
        say("  Manual steps (the installer cannot do these):")
        for b, m in ctx.manual:
            say(f"    [{b}] {m}")
        for mid, m in ctx.weights_manual:
            say(f"    [{mid}] weights: {m}")
    port = os.environ.get("STUDIO_PORT", "7860")
    say()
    say(f"  Start the studio:  ./run_studio.sh      then open http://127.0.0.1:{port}")
    say("  See what is installed: ./install.sh list     Add more: ./install.sh pack <name>   ./install.sh model <id>")


def cmd_core(ctx: Ctx) -> int:
    prereqs(ctx)
    return run_plan(ctx, ["core"], [], False, "core")


def cmd_pack(ctx: Ctx, names: list[str]) -> int:
    prereqs(ctx)
    models: list[str] = []
    extra: list[str] = []
    for n in names:
        m, b = pack_models(ctx, n)
        models += m
        extra += b
    models = list(dict.fromkeys(models))
    bad = [(m, model_problem(ctx, m) if m in ctx.reg.models else "not in the registry") for m in models]
    bad = [(m, why) for m, why in bad if why]
    for m, why in bad:
        warn(f"skipping {m}: {why}")
    models = [m for m in models if m not in {b[0] for b in bad}]
    backends = backends_for_models(ctx, models) + extra
    return run_plan(ctx, backends, models, ctx.weights, "pack " + " ".join(names))


def cmd_backend(ctx: Ctx, ids: list[str]) -> int:
    for b in ids:
        if b not in ctx.recipes:
            hint = EXPECTED_MISSING.get(b) or f"known: {', '.join(sorted(ctx.recipes))}"
            raise InstallError(f"no recipe for backend '{b}' ({hint})")
    prereqs(ctx)
    models = list(dict.fromkeys(m for b in ids for m in models_of_backend(ctx, b)))
    return run_plan(ctx, ids, models, ctx.weights, "backend " + " ".join(ids))


def cmd_model(ctx: Ctx, ids: list[str]) -> int:
    for mid in ids:
        why = model_problem(ctx, mid)
        if why:
            raise InstallError(f"{mid}: {why}")
    prereqs(ctx)
    return run_plan(ctx, backends_for_models(ctx, ids), ids, True, "model " + " ".join(ids))


def render_pack_table(ctx: Ctx) -> list[str]:
    rows = []
    for name, pk in ctx.packs.items():
        s = pack_summary(ctx, name)
        total = s["env_gb"] + s["weights_gb"]
        note = []
        if s["gated"]:
            note.append("HF login + licence for " + ", ".join(s["gated"]))
        if s["manual"]:
            note.append("manual install: " + ", ".join(s["manual"]))
        rows.append((name, pk, s, total, note))
    return rows


def cmd_menu(ctx: Ctx) -> int:
    prereqs(ctx)
    default = next((n for n, p in ctx.packs.items() if p.get("default")), "starter")
    head("core")
    ok = ensure_core(ctx)
    if not ok:
        summary(ctx)
        return EXIT_FAIL
    head("packs")
    names = list(ctx.packs)
    rows = render_pack_table(ctx)
    say(f"  {'#':>2}  {'pack':9} {'models':>6} {'to install':>10} {'to download':>11}  description")
    for i, (name, pk, s, total, note) in enumerate(rows, 1):
        star = " (default)" if name == default else ""
        say(f"  {i:>2}  {name:9} {len(s['models']):>6} {s['env_gb']:>8.0f}GB {s['weights_gb']:>9.0f}GB  {pk['title']}{star}: {pk['description']}")
        for n in note:
            say(f"{'':>47}! {n}")
    say("  Sizes are rough and count only what is still missing: environments (one venv per backend) + model weights. Weights of models whose")
    say("  licence you must accept (marked !) need `hf auth login` first. Some models are non-commercial: see the Models page.")
    chosen: list[str]
    if ctx.yes:
        chosen = [default]
        say(f"  --yes: installing the default pack '{default}'")
    elif not sys.stdin.isatty():
        raise InstallError("not an interactive terminal: pass --yes (installs the default pack) or use ./install.sh pack <name>")
    else:
        ans = input(f"Packs to install (numbers or names, comma separated; Enter = {default}; 0 = none): ").strip().lower()
        if ans in ("0", "none", "n"):
            chosen = []
        elif not ans:
            chosen = [default]
        else:
            chosen = []
            for tok in re.split(r"[,\s]+", ans):
                if tok.isdigit() and 1 <= int(tok) <= len(names):
                    chosen.append(names[int(tok) - 1])
                elif tok in ctx.packs:
                    chosen.append(tok)
                elif tok:
                    raise InstallError(f"unknown pack '{tok}'")
    if not chosen:
        summary(ctx)
        say("  Core installed. Add packs later with ./install.sh pack <name>.")
        return EXIT_OK
    if not ctx.weights and not ctx.yes:
        ctx.weights = confirm(ctx, "Download the model weights now (needs the disk space shown above)?", True)
    return cmd_pack(ctx, chosen)


def cmd_list(ctx: Ctx, what: str | None) -> int:
    what = what or "all"
    if what in ("all", "packs"):
        head("packs  (./install.sh pack <name> [--weights])")
        say(f"  {'pack':9} {'status':12} {'install':>8} {'download':>9}  contents (sizes: only what is still missing)")
        for name, pk, s, total, note in render_pack_table(ctx):
            st = "installed" if s["total"] and s["installed"] == s["total"] else f"{s['installed']}/{s['total']} envs"
            say(f"  {name:9} {st:12} {s['env_gb']:>6.0f}GB {s['weights_gb']:>7.0f}GB  {pk['title']}: {len(s['models'])} models"
                + (f"  [login: {len(s['gated'])}]" if s["gated"] else ""))
    if what in ("all", "backends"):
        head("backends  (./install.sh backend <id>)")
        say(f"  {'backend':18} {'status':12} {'size':>6}  models / notes")
        for bid in sorted(set(ctx.reg.backends) | set(ctx.recipes)):
            b = ctx.reg.backends.get(bid, {})
            if bid in EXPECTED_MISSING:
                say(f"  {bid:18} {'n/a':12} {'':>6}  {EXPECTED_MISSING[bid]}")
                continue
            r = ctx.recipes.get(bid)
            n = len([m for m in ctx.reg.models.values() if m.get("backend") == bid])
            if r is None:
                if b.get("kind") == "service":
                    say(f"  {bid:18} {'service':12} {'':>6}  {n} model(s); {b.get('label', '')}")
                else:
                    say(f"  {bid:18} {'no recipe':12} {'':>6}  {n} model(s)")
                continue
            st = backend_state(r)
            size = f"{r['size_gb']:.0f}GB" if recipe_kind(r) == "env" else ""
            extra = f"shares envs/{r['env']} with {r['shares_env_with']}" if r["shares_env_with"] else ""
            if st == "manual":
                extra = "manual install, see ./install.sh backend " + bid
            say(f"  {bid:18} {st:12} {size:>6}  {n} model(s) {extra}")
    if what in ("all", "models"):
        head("models  (./install.sh model <id>)")
        say(f"  {'model':28} {'backend':17} {'weights':>8} {'status':16} licence")
        for mid, m in sorted(ctx.reg.models.items(), key=lambda kv: (kv[1].get("backend", ""), kv[0])):
            if ctx.reg.backends.get(m.get("backend"), {}).get("kind") == "cloud":
                continue
            st = ctx.reg.status.get(mid, {}).get("status", "?")
            gb = ctx.sizes["weights_gb"].get(mid)
            lic = ("commercial" if m["license"].get("commercial") else "non-commercial") + (", login" if mid in ctx.sizes["gated"] else "")
            say(f"  {mid:28} {m.get('backend', ''):17} {(f'{gb:.1f}GB' if gb is not None else '?'):>8} {st:16} {lic}")
    if what not in ("all", "packs", "backends", "models"):
        raise InstallError("list takes: packs | backends | models")
    return EXIT_OK


def cmd_doctor(ctx: Ctx) -> int:
    problems = 0
    head("system")
    say(f"  platform: {platform_tag()}  python (installer): {platform.python_version()}")
    if platform_tag() not in KNOWN_PLATFORMS:
        say(_c("31", "  unsupported platform: the studio needs Linux with an NVIDIA GPU"))
        problems += 1
    for tool, hint in (("git", "install git"), ("uv", "curl -LsSf https://astral.sh/uv/install.sh | sh"),
                       ("ffmpeg", "install ffmpeg (apt install ffmpeg)"), ("ffprobe", "comes with ffmpeg")):
        p = shutil.which(tool)
        if p:
            try:
                ver = subprocess.run([p, "--version" if tool == "uv" else "-version" if tool.startswith("ff") else "--version"],
                                     capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
            except (OSError, subprocess.TimeoutExpired, IndexError):
                ver = p
            say(f"  {tool:8} ok   {ver[:70]}")
        else:
            say(_c("31", f"  {tool:8} MISSING ({hint})"))
            problems += 1
    g = detect_gpu()
    ctx.gpu = g
    if g["ok"]:
        mem = f", {g['memory_gb']} GB" if g.get("memory_gb") else " (unified memory)"
        say(f"  GPU      {g['name']}{mem}, compute capability {g['cc']}")
        say(f"  driver   {g['driver']}   CUDA (driver) {g['cuda']}")
        try:
            say(f"  PyTorch  wheels {torch_index_for_cuda(g['cuda'])}")
        except InstallError as e:
            say(_c("31", f"  PyTorch  {e}"))
            problems += 1
    else:
        say(_c("31", f"  GPU      {g['error']}"))
        problems += 1
    say(f"  disk     {disk_free_gb(REPO):.0f} GB free in {REPO}; {disk_free_gb(hf_cache_dir()):.0f} GB free for the HF cache ({hf_cache_dir()})")
    lp = long_path_warning()
    if lp:
        say(_c("33", f"  path     {lp}"))
    say(f"  HF login {'yes' if hf_logged_in() else 'no (only needed for gated models: envs/core/bin/hf auth login)'}")
    say(f"  HF env   HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '(unset)')} HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '(unset)')}"
        "  (the installer sets the right values for downloads; run_studio.sh runs offline)")
    head("backends")
    counts = {"installed": 0, "linked": 0, "present": 0, "missing": 0, "broken": 0, "manual": 0}
    for bid in sorted(ctx.recipes):
        r = ctx.recipes[bid]
        st = backend_state(r)
        detail = ""
        if st in ("installed", "linked", "present"):
            py = env_python(r)
            try:
                res = subprocess.run([str(py), "-c", r["check"] if (ctx.check and r["check"]) else "import sys"],
                                     capture_output=True, text=True, timeout=300, cwd=REPO, env={**base_env(), "HF_HUB_OFFLINE": "1"})
                if res.returncode != 0:
                    st = "broken"
                    detail = (res.stderr.strip().splitlines() or ["check failed"])[-1][:140]
            except (OSError, subprocess.TimeoutExpired) as e:
                st, detail = "broken", str(e)[:140]
        if st == "broken-link":
            st = "broken"
            detail = "symlink to a missing folder"
        miss = [p for p in vendor_matches(r["vendor"]) if not (REPO / p).exists()]
        if miss and st != "manual":
            detail += f" vendor missing: {', '.join(miss)}"
        counts[st] = counts.get(st, 0) + 1
        if st in ("broken",):
            problems += 1
        if st != "missing" or ctx.check:
            say(f"  {bid:18} {st:10} {detail}")
    say(f"  -> {counts['installed']} installed, {counts['linked']} linked, {counts['present']} present (not made by the installer), "
        f"{counts['missing']} not installed, {counts['broken']} broken, {counts['manual']} manual"
        + ("   (--check imports every installed env)" if not ctx.check else ""))
    head("models (registry status)")
    by: dict[str, int] = {}
    for mid, st in ctx.reg.status.items():
        if ctx.reg.backends.get(ctx.reg.models[mid].get("backend"), {}).get("kind") == "cloud":
            continue
        by[st["status"]] = by.get(st["status"], 0) + 1
    say("  " + ", ".join(f"{k}: {v}" for k, v in sorted(by.items())))
    say()
    say(_c("32", "Everything looks fine.") if not problems else _c("31", f"{problems} problem(s) found (see above)."))
    return EXIT_OK if not problems else EXIT_PREREQ


def cmd_validate(ctx: Ctx) -> int:
    recipes, errs = load_recipes()
    for e in errs:
        err(e)
    reg = load_registry()
    for bid, b in reg.backends.items():
        if b.get("kind") in ("cloud", "service") and bid not in recipes:
            continue
        if bid in EXPECTED_MISSING:
            continue
        if bid not in recipes:
            err(f"registry backend '{bid}' has no install/{bid}.yaml")
            errs.append(bid)
    say(f"{len(recipes)} recipes, {len(errs)} problem(s)")
    return EXIT_OK if not errs else EXIT_FAIL


USAGE = __doc__.split("What it installs")[0].strip()


def main(argv: list[str]) -> int:
    flags = {"--dry-run", "--yes", "-y", "--weights", "--force", "--check", "--help", "-h"}
    opts: set[str] = set()
    pos: list[str] = []
    torch_index = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--torch-index":
            i += 1
            torch_index = argv[i] if i < len(argv) else None
        elif a.startswith("--torch-index="):
            torch_index = a.split("=", 1)[1]
        elif a in flags:
            opts.add(a)
        elif a.startswith("-") and a != "-":
            err(f"unknown option {a}")
            say(USAGE)
            return EXIT_PREREQ
        else:
            pos.append(a)
        i += 1
    if "--help" in opts or "-h" in opts or (pos and pos[0] == "help"):
        say(USAGE)
        return EXIT_OK
    ctx = Ctx(dry_run="--dry-run" in opts, yes=bool(opts & {"--yes", "-y"}), weights="--weights" in opts,
              force="--force" in opts, torch_index=torch_index, check="--check" in opts)
    cmd = pos[0] if pos else None
    args = pos[1:]
    try:
        ctx.load()
        if cmd is None:
            if ctx.dry_run:
                prereqs(ctx)
                head("dry run: core, then the menu would offer these packs")
                ensure_core(ctx)
                return cmd_list(ctx, "packs")
            return cmd_menu(ctx)
        if cmd == "core":
            return cmd_core(ctx)
        if cmd == "pack":
            if not args:
                raise InstallError(f"usage: ./install.sh pack <name>...   packs: {', '.join(ctx.packs)}")
            return cmd_pack(ctx, args)
        if cmd == "backend":
            if not args:
                raise InstallError("usage: ./install.sh backend <id>...   (./install.sh list backends)")
            return cmd_backend(ctx, args)
        if cmd == "model":
            if not args:
                raise InstallError("usage: ./install.sh model <model-id>...   (./install.sh list models)")
            return cmd_model(ctx, args)
        if cmd == "list":
            return cmd_list(ctx, args[0] if args else None)
        if cmd == "doctor":
            return cmd_doctor(ctx)
        if cmd == "validate":
            return cmd_validate(ctx)
        err(f"unknown command '{cmd}'")
        say(USAGE)
        return EXIT_PREREQ
    except InstallError as e:
        err(str(e))
        return EXIT_PREREQ
    except Stop as s:
        err(str(s))
        return s.code
    except KeyboardInterrupt:
        err("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
