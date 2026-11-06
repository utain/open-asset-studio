"""Shared plumbing of the cloud adapters: one CloudRun per job (HTTP client, auth, polling with cancel, downloads,
output files, cost). Adapters are plain functions `run(r: CloudRun)` in app/cloud/<provider>.py.

Cloud jobs run in-process in the core (handler "cloud", pool "cloud", no GPU lock): they only wait on HTTP, the API
keys never leave the core process, and cancel is just a flag checked between polls.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image

from ..jobs import JobCancelled, JobContext
from ..providers import ProviderError, Providers

UA = "open-asset-studio/0.1 (+local)"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
MESH_EXT = (".glb", ".gltf", ".fbx", ".obj", ".usdz", ".ply", ".stl")
MAX_DOWNLOAD = 600 * 2**20


def size_wh(aspect: str | None, long_side, multiple: int = 16) -> tuple[int, int]:
    try:
        a, b = (float(x) for x in str(aspect or "1:1").split(":"))
    except ValueError:
        a, b = 1.0, 1.0
    long_side = int(long_side or 1024)
    if a >= b:
        w, h = long_side, long_side * b / a
    else:
        w, h = long_side * a / b, long_side
    r = lambda v: max(multiple, int(round(v / multiple)) * multiple)  # noqa: E731
    return r(w), r(h)


def nearest_aspect(aspect: str | None, allowed: list[str]) -> str:
    try:
        a, b = (float(x) for x in str(aspect or "1:1").split(":"))
        t = a / b
    except ValueError:
        return allowed[0]
    def ratio(s):
        x, y = (float(v) for v in s.split(":"))
        return x / y
    return min(allowed, key=lambda s: abs(ratio(s) - t))


def find_files(obj, out=None, key="") -> list[dict]:
    """Every file reference in a provider result: {url, content_type, file_name, key} (url may be a data: URI)."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        u = obj.get("url") if isinstance(obj.get("url"), str) else None
        if u and (u.startswith(("http://", "https://", "data:"))):
            out.append({"url": u, "content_type": obj.get("content_type") or obj.get("mime_type") or "",
                        "file_name": obj.get("file_name") or "", "key": key})
        for k, v in obj.items():
            if k != "url":
                find_files(v, out, k)
    elif isinstance(obj, list):
        for v in obj:
            find_files(v, out, key)
    elif isinstance(obj, str) and obj.startswith(("http://", "https://")) and len(obj) < 4000:
        out.append({"url": obj, "content_type": "", "file_name": "", "key": key})
    return out


def kind_of(f: dict) -> str:
    ct = (f.get("content_type") or "").lower()
    name = (f.get("file_name") or urlparse(f["url"]).path if not f["url"].startswith("data:") else f.get("file_name") or "").lower()
    if f["url"].startswith("data:"):
        ct = ct or f["url"][5:f["url"].find(";")]
    if ct.startswith("image/") or name.endswith(IMG_EXT):
        return "image"
    if ct.startswith("audio/") or name.endswith(AUDIO_EXT):
        return "audio"
    if "gltf" in ct or name.endswith(MESH_EXT):
        return "mesh"
    k = (f.get("key") or "").lower()
    if any(x in k for x in ("glb", "mesh", "model")):
        return "mesh"
    if "audio" in k:
        return "audio"
    if "image" in k:
        return "image"
    return "other"


class CloudRun:
    def __init__(self, ctx: JobContext, providers: Providers, model: dict):
        self.ctx, self.p, self.m = ctx, providers, model
        self.pid = model["provider"]
        self.label = providers.label(self.pid)
        self.params = dict(ctx.params)
        self.task = ctx.job["task"]
        self.inputs = ctx.input_assets()
        self.client = httpx.Client(timeout=httpx.Timeout(180.0, connect=20.0), follow_redirects=True,
                                   headers={"User-Agent": UA})
        self.files: list[dict] = []      # {path, type, meta}
        self.cost_usd: float | None = None   # actual cost reported by the API (None = use the estimate)
        self.cost_note = ""
        self.request_ids: list[str] = []
        self.submitted = False
        self.cancel_remote = None        # callable: best-effort cancel at the provider
        self.extra_meta: dict = {}
        self.out = ctx.scratch / "cloud"
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    @property
    def prompt(self) -> str:
        return str(self.params.get("prompt") or "")

    def remote(self, task: str | None = None) -> str:
        r = self.m.get("remote") or {}
        return (r.get(task or self.task) if isinstance(r, dict) else None) or self.m.get("remote_model") or self.m["id"]

    def url(self, path: str, kind: str = "api") -> str:
        return self.p.base(self.pid, kind) + path

    def progress(self, frac: float | None, msg: str, stage: str = "cloud", interruptible: bool = True):
        self.ctx.progress(frac, msg, stage=stage, interruptible=interruptible)

    def log(self, msg: str):
        self.ctx.log(self.p.secrets_in(msg))

    def check(self):
        if self.ctx.cancel_event.is_set():
            if self.cancel_remote and self.submitted:
                try:
                    self.cancel_remote()
                    self.log("cancel sent to the provider")
                except Exception as e:  # noqa: BLE001
                    self.log(f"remote cancel failed: {e}")
            raise JobCancelled()

    # ------------------------------------------------------------ HTTP
    def _err_text(self, resp: httpx.Response) -> str:
        try:
            d = resp.json()
        except ValueError:
            return resp.text[:300]
        for path in (("error", "message"), ("detail",), ("error",), ("message",), ("detail", 0, "msg"), ("title",)):
            cur = d
            try:
                for k in path:
                    cur = cur[k]
                if cur:
                    return (cur if isinstance(cur, str) else json.dumps(cur))[:400]
            except (KeyError, IndexError, TypeError):
                continue
        return json.dumps(d)[:300]

    def request(self, method: str, url: str, *, auth: bool = True, headers: dict | None = None, timeout: float | None = None,
                **kw) -> httpx.Response:
        h = {**(self.p.headers(self.pid) if auth else {}), **(headers or {})}
        try:
            resp = self.client.request(method, url, headers=h, timeout=timeout or httpx.USE_CLIENT_DEFAULT, **kw)
        except httpx.TimeoutException:
            raise ProviderError(f"{self.label} did not answer in time.", "provider", "Retry later.") from None
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach {self.label}: {type(e).__name__}.", "network",
                                "Check the internet connection (or the provider's status page).") from None
        if resp.status_code < 400:
            return resp
        msg = self.p.secrets_in(self._err_text(resp))
        sc = resp.status_code
        if sc in (401, 403):
            raise ProviderError(f"{self.label} refused the API key ({sc}): {msg}", "auth",
                                "Check or replace the key on the Providers page.")
        if sc == 402:
            raise ProviderError(f"{self.label}: out of credits or billing not set up ({msg}).", "billing",
                                f"Top up your {self.label} account.")
        if sc == 429:
            raise ProviderError(f"{self.label} rate limit / quota: {msg}", "rate_limit", "Wait a minute and retry.")
        if 400 <= sc < 500:
            raise ProviderError(f"{self.label} rejected the request ({sc}): {msg}", "input",
                                "Change the settings or prompt (some providers block certain content).")
        raise ProviderError(f"{self.label} server error ({sc}): {msg}", "provider", "Retry later.")

    def get_json(self, url: str, **kw) -> dict:
        return self.request("GET", url, **kw).json()

    def post_json(self, url: str, body: dict, **kw) -> dict:
        r = self.request("POST", url, json=body, **kw)
        try:
            return r.json()
        except ValueError:
            raise ProviderError(f"{self.label} answered with something that is not JSON.", "provider") from None

    def poll(self, fn, *, interval: float = 2.0, timeout: float | None = None, what: str = "Generating"):
        """Call fn() until it returns non-None; cancel-aware; backs off to 6 s."""
        timeout = timeout or float(self.m.get("timeout_s") or 1200)
        t0 = time.time()
        i = interval
        while True:
            self.check()
            r = fn()
            if r is not None:
                return r
            if time.time() - t0 > timeout:
                raise ProviderError(f"{self.label} did not finish within {int(timeout)} s.", "timeout",
                                    "Retry; the result may still appear in the provider's dashboard.")
            end = time.time() + i
            while time.time() < end:
                self.check()
                time.sleep(0.25)
            i = min(6.0, i * 1.3)

    # ------------------------------------------------------------ inputs
    def input_list(self, type_: str | None = None, role: str | None = None) -> list[dict]:
        return [x for x in self.inputs if (type_ is None or x["type"] == type_) and (role is None or x["role"] == role)]

    def images_in(self) -> list[dict]:
        main = [x for x in self.inputs if x["type"] == "image" and x["role"] in ("image", "input")]
        refs = [x for x in self.inputs if x["type"] == "image" and x["role"] not in ("image", "input")]
        return main + refs

    def read_input(self, inp: dict, max_side: int | None = None, fmt: str | None = None) -> tuple[bytes, str]:
        p = Path(inp["path"])
        if inp["type"] == "image" and (max_side or fmt):
            with Image.open(p) as im:
                im.load()
                if max_side and max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.LANCZOS)
                buf = io.BytesIO()
                f = (fmt or "png").upper()
                if f == "JPEG" and im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGB")
                im.save(buf, f)
                return buf.getvalue(), f"image/{f.lower()}"
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return p.read_bytes(), mime

    def data_url(self, inp: dict, max_side: int | None = 2048) -> str:
        data, mime = self.read_input(inp, max_side=max_side if inp["type"] == "image" else None,
                                     fmt="png" if inp["type"] == "image" else None)
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    # ------------------------------------------------------------ outputs
    def download(self, url: str, name: str, *, auth: bool = False) -> Path:
        dest = self.out / name
        if url.startswith("data:"):
            head, _, b64 = url.partition(",")
            dest.write_bytes(base64.b64decode(b64) if ";base64" in head else b64.encode())
            return dest
        u = urlparse(url)
        mock = os.environ.get("STUDIO_CLOUD_MOCK")
        if u.scheme != "https" and not (mock and url.startswith(mock)) and not self.p.is_local_provider(self.pid):
            raise ProviderError(f"{self.label} returned a non-https file URL; refused.", "provider")
        n = 0
        h = self.p.headers(self.pid) if auth else {}
        try:
            with self.client.stream("GET", url, headers=h, timeout=httpx.Timeout(300.0, connect=20.0)) as resp:
                if resp.status_code >= 400:
                    raise ProviderError(f"Downloading the result from {self.label} failed ({resp.status_code}).", "provider",
                                        "Retry; result links expire after a while.")
                with dest.open("wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        self.check()
                        n += len(chunk)
                        if n > MAX_DOWNLOAD:
                            raise ProviderError("Result file larger than 600 MB; refused.", "provider")
                        f.write(chunk)
        except httpx.HTTPError as e:
            raise ProviderError(f"Downloading the result from {self.label} failed: {type(e).__name__}.", "network") from None
        return dest

    def ext_of(self, f: dict, default: str) -> str:
        if f["url"].startswith("data:"):
            ct = f["url"][5:f["url"].find(";")]
            return mimetypes.guess_extension(ct) or default
        e = os.path.splitext(f.get("file_name") or urlparse(f["url"]).path)[1].lower()
        return e if re.fullmatch(r"\.[a-z0-9]{2,5}", e or "") else default

    def add_image(self, src: bytes | Path, name: str, **meta):
        """Normalise to PNG (keeps alpha)."""
        data = src.read_bytes() if isinstance(src, Path) else src
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGBA" if "A" in im.mode or im.mode == "P" else "RGB")
                dest = self.out / f"{name}.png"
                im.save(dest, "PNG")
        except Exception:  # noqa: BLE001
            raise ProviderError(f"{self.label} returned an image that cannot be read.", "provider") from None
        if isinstance(src, Path) and src != dest:
            src.unlink(missing_ok=True)
        self.files.append({"path": dest, "type": "image", "meta": meta})

    def add_file(self, path: Path, type_: str, attach: list[Path] | None = None, **meta):
        if type_ == "mesh":
            with open(path, "rb") as f:
                if f.read(4) != b"glTF":
                    raise ProviderError(f"{self.label} returned a mesh that is not a GLB file.", "provider")
        self.files.append({"path": path, "type": type_, "meta": meta, "attach": attach or []})

    def add_audio_bytes(self, data: bytes, name: str, ext: str = ".mp3", **meta):
        if data[:4] == b"RIFF":
            ext = ".wav"
        elif data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            ext = ".mp3"
        dest = self.out / f"{name}{ext}"
        dest.write_bytes(data)
        self.files.append({"path": dest, "type": meta.pop("type", "audio"), "meta": meta})

    def fetch_outputs(self, result, want: str, prefix: str = "out", limit: int = 8, auth: bool = False) -> int:
        """Download every file of kind `want` (image | mesh | audio) found in a result JSON; returns the count."""
        fs = [f for f in find_files(result) if kind_of(f) == want]
        if want == "mesh":
            glb = [f for f in fs if self.ext_of(f, ".glb") == ".glb"]
            fs = glb[:1] or fs[:1]
        seen, k = set(), 0
        for f in fs:
            if f["url"] in seen or k >= limit:
                continue
            seen.add(f["url"])
            k += 1
            self.progress(0.9, f"Downloading result {k}", stage="download")
            if want == "image":
                p = self.download(f["url"], f"{prefix}_{k}{self.ext_of(f, '.png')}", auth=auth)
                self.add_image(p, f"{prefix}_{k}", index=k - 1)
            elif want == "mesh":
                p = self.download(f["url"], f"{prefix}_{k}.glb", auth=auth)
                self.add_file(p, "mesh", index=k - 1)
            else:
                p = self.download(f["url"], f"dl_{prefix}_{k}{self.ext_of(f, '.mp3')}", auth=auth)
                self.add_audio_bytes(p.read_bytes(), f"{prefix}_{k}", self.ext_of(f, ".mp3"), index=k - 1)
                p.unlink(missing_ok=True)
        return k

    def close(self):
        self.client.close()


def pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
