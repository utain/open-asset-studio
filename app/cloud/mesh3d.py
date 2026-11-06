"""Meshy and Tripo: text to 3D and image to 3D with textures (async tasks: create -> poll -> download GLB).

Meshy (Bearer, https://api.meshy.ai/openapi):
  text:  POST /v2/text-to-3d {mode: preview, prompt, ai_model, topology, target_polycount}  -> {result: id}
         then (textured) POST /v2/text-to-3d {mode: refine, preview_task_id, enable_pbr}    -> {result: id}
  image: POST /v1/image-to-3d {image_url: data URI, ai_model, should_texture, enable_pbr, ...} -> {result: id}
  poll:  GET  .../{id} -> {status: PENDING|IN_PROGRESS|SUCCEEDED|FAILED|CANCELED, progress, model_urls.glb,
         texture_urls, thumbnail_url, task_error, consumed_credits}
Tripo v3 (Bearer, https://openapi.tripo3d.ai/v3):
  text:  POST /generation/text-to-model {prompt, model, texture, pbr, face_limit} -> {code: 0, data: {task_id}}
  image: POST /files (multipart `file`) -> data.file_token; POST /generation/image-to-model {input: token, ...}
  poll:  GET /tasks/{id} -> data {status: queued|running|success|failed|cancelled, progress, output.model_url}
Actual cost: consumed credits x pricing.per_credit when the API reports them.
"""
from __future__ import annotations

from .base import CloudRun
from ..providers import ProviderError


def _credits_cost(r: CloudRun, credits):
    try:
        c = float(credits)
    except (TypeError, ValueError):
        return
    per = float((r.m.get("pricing") or {}).get("per_credit") or 0)
    if c > 0 and per:
        r.cost_usd = round((r.cost_usd or 0) + c * per, 4)
        r.cost_note = f"{(r.cost_usd or 0) / per:.0f} credits"


# ------------------------------------------------------------------ Meshy
def _meshy_wait(r: CloudRun, path: str, tid: str, lo: float, hi: float, what: str) -> dict:
    def step():
        d = r.get_json(r.url(f"{path}/{tid}"))
        s = d.get("status")
        if s == "SUCCEEDED":
            return d
        if s in ("FAILED", "CANCELED", "EXPIRED"):
            err = (d.get("task_error") or {}).get("message") if isinstance(d.get("task_error"), dict) else d.get("task_error")
            raise ProviderError(f"Meshy {what} {s.lower()}: {err or 'no reason given'}", "input" if s == "FAILED" else "provider")
        prog = float(d.get("progress") or 0)
        r.progress(lo + (hi - lo) * prog / 100, f"Meshy {what}: {s.lower().replace('_', ' ') if s else 'waiting'} {int(prog)}%",
                   stage="generating")
        return None
    return r.poll(step, interval=3.0)


def _meshy_create(r: CloudRun, path: str, body: dict) -> str:
    d = r.post_json(r.url(path), body)
    tid = d.get("result") or d.get("id")
    if not tid:
        raise ProviderError("Meshy did not return a task id.", "provider")
    r.request_ids.append(str(tid))
    r.submitted = True
    r.log(f"meshy task {tid} ({path})")
    return str(tid)


def meshy(r: CloudRun):
    p = r.params
    common = {"ai_model": p.get("ai_model") or r.remote(), "topology": p.get("topology") or "triangle"}
    if p.get("target_polycount"):
        common["target_polycount"] = int(p["target_polycount"])
        common["should_remesh"] = True
    textured = bool(p.get("textured", True))
    credits = 0.0
    if r.task == "text3d":
        if not r.prompt.strip():
            raise ProviderError("Describe the object to model.", "input")
        r.progress(0.03, "Meshy: creating the preview mesh", stage="submit")
        tid = _meshy_create(r, "/v2/text-to-3d", {"mode": "preview", "prompt": r.prompt[:800], **common,
                                                   **({"art_style": p["art_style"]} if p.get("art_style") else {})})
        d = _meshy_wait(r, "/v2/text-to-3d", tid, 0.05, 0.5 if textured else 0.9, "preview")
        credits += float(d.get("consumed_credits") or 0)
        if textured:
            r.progress(0.5, "Meshy: painting textures (refine)", stage="submit")
            tid2 = _meshy_create(r, "/v2/text-to-3d", {"mode": "refine", "preview_task_id": tid,
                                                       "enable_pbr": bool(p.get("pbr", True))})
            d = _meshy_wait(r, "/v2/text-to-3d", tid2, 0.5, 0.9, "texture")
            credits += float(d.get("consumed_credits") or 0)
    else:
        imgs = r.images_in()
        if not imgs:
            raise ProviderError("Pick the concept image.", "input")
        r.progress(0.03, "Meshy: uploading the image", stage="submit")
        tid = _meshy_create(r, "/v1/image-to-3d", {"image_url": r.data_url(imgs[0]), **common, "should_texture": textured,
                                                    "enable_pbr": bool(p.get("pbr", False)) and textured})
        d = _meshy_wait(r, "/v1/image-to-3d", tid, 0.05, 0.9, "image to 3D")
        credits += float(d.get("consumed_credits") or 0)
    _credits_cost(r, credits)
    glb = (d.get("model_urls") or {}).get("glb")
    if not glb:
        raise ProviderError("Meshy finished without a GLB link.", "provider")
    r.progress(0.92, "Downloading the GLB", stage="download")
    path = r.download(glb, "meshy.glb")
    attach = []
    if d.get("thumbnail_url"):
        try:
            attach.append(r.download(d["thumbnail_url"], "preview.png"))
        except ProviderError:
            pass
    r.add_file(path, "mesh", attach=attach, textured=textured)


# ------------------------------------------------------------------ Tripo
def _tripo_data(r: CloudRun, d: dict) -> dict:
    if d.get("code", 0) != 0:
        raise ProviderError(f"Tripo error {d.get('code')}: {d.get('message') or d.get('suggestion') or ''}", "input")
    return d.get("data") or {}


def tripo(r: CloudRun):
    p = r.params
    textured = bool(p.get("textured", True))
    common = {"model": p.get("model_version") or r.remote(), "texture": textured, "pbr": bool(p.get("pbr", True)) and textured}
    if p.get("face_limit"):
        common["face_limit"] = int(p["face_limit"])
    if p.get("texture_quality") and textured:
        common["texture_quality"] = p["texture_quality"]
    if r.task == "text3d":
        if not r.prompt.strip():
            raise ProviderError("Describe the object to model.", "input")
        body = {"prompt": r.prompt[:1024], **common}
        if p.get("negative"):
            body["negative_prompt"] = str(p["negative"])[:255]
        r.progress(0.03, "Tripo: creating the task", stage="submit")
        d = _tripo_data(r, r.post_json(r.url("/generation/text-to-model"), body))
    else:
        imgs = r.images_in()
        if not imgs:
            raise ProviderError("Pick the concept image.", "input")
        data, mime = r.read_input(imgs[0], max_side=2048, fmt="png")
        r.progress(0.02, "Tripo: uploading the image", stage="submit")
        up = _tripo_data(r, r.request("POST", r.url("/files"), files={"file": ("image.png", data, mime)}).json())
        token = up.get("file_token") or up.get("image_token")
        if not token:
            raise ProviderError("Tripo upload returned no file token.", "provider")
        d = _tripo_data(r, r.post_json(r.url("/generation/image-to-model"), {"input": token, **common}))
    tid = d.get("task_id")
    if not tid:
        raise ProviderError("Tripo did not return a task id.", "provider")
    r.request_ids.append(tid)
    r.submitted = True
    r.log(f"tripo task {tid}")

    def step():
        t = _tripo_data(r, r.get_json(r.url(f"/tasks/{tid}")))
        s = t.get("status")
        if s == "success":
            return t
        if s in ("failed", "cancelled", "banned", "expired", "unknown"):
            raise ProviderError(f"Tripo task {s}.", "input" if s in ("failed", "banned") else "provider")
        prog = float(t.get("progress") or 0)
        r.progress(0.05 + 0.85 * prog / 100, f"Tripo: {s or 'queued'} {int(prog)}%", stage="generating")
        return None
    t = r.poll(step, interval=2.5)
    out = t.get("output") or {}
    _credits_cost(r, t.get("consumed_credit") or t.get("consumed_credits"))
    url = out.get("pbr_model_url") or out.get("model_url") or out.get("pbr_model") or out.get("model") or out.get("base_model")
    if isinstance(url, dict):
        url = url.get("url")
    if not url:
        raise ProviderError("Tripo finished without a model link.", "provider")
    r.progress(0.92, "Downloading the GLB (Tripo links expire in minutes)", stage="download")
    path = r.download(url, "tripo.glb")
    attach = []
    img = out.get("rendered_image_url") or out.get("rendered_image")
    if isinstance(img, dict):
        img = img.get("url")
    if img:
        try:
            attach.append(r.download(img, "preview.webp"))
        except ProviderError:
            pass
    r.add_file(path, "mesh", attach=attach, textured=textured)
