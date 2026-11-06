"""Local mock of every cloud provider the studio talks to (submit / poll / result shapes), for end-to-end tests
without API keys. Point the core at it with STUDIO_CLOUD_MOCK=http://127.0.0.1:<port>: provider <p> base URL
"<kind>" becomes <mock>/<p>/<kind> (app/providers.py Providers.base).

    envs/core/bin/python tests/fixtures/cloud_mock.py --port 7899

Any key works except one containing "bad" (-> 401). Each async job needs 2 polls before it completes.
GET /_log returns the recorded requests (path, auth header kind, body keys, whether inputs were data: URIs).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import uuid

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

app = FastAPI()
LOG: list[dict] = []
JOBS: dict[str, dict] = {}
BASE = "http://127.0.0.1:7899"


# ------------------------------------------------------------------ fake media
def png(color=(200, 60, 40), size=(64, 48), alpha=False) -> bytes:
    im = Image.new("RGBA" if alpha else "RGB", size, (*color, 128) if alpha else color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def webp() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (40, 40), (20, 120, 220)).save(b, "WEBP")
    return b.getvalue()


def glb() -> bytes:
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32).tobytes()
    idx = np.array([0, 1, 2], dtype=np.uint16).tobytes() + b"\0\0"
    binc = pos + idx
    doc = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
           "buffers": [{"byteLength": len(binc)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
                         {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
    js = json.dumps(doc).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binc)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binc), b"BIN\0") + binc)


def audio(fmt="mp3", secs=0.6, sr=44100) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-3 * t)).astype(np.float32)
    b = io.BytesIO()
    sf.write(b, x, sr, format="MP3" if fmt == "mp3" else "WAV")
    return b.getvalue()


def pcm(secs=0.5, sr=24000) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


FILES = {"img.png": (png, "image/png"), "img.webp": (webp, "image/webp"), "mesh.glb": (glb, "model/gltf-binary"),
         "clip.mp3": (lambda: audio("mp3"), "audio/mpeg"), "clip.wav": (lambda: audio("wav"), "audio/wav"),
         "thumb.png": (lambda: png((90, 90, 90)), "image/png")}


@app.get("/files/{name}")
def files(name: str):
    if name not in FILES:
        raise HTTPException(404)
    fn, ct = FILES[name]
    return Response(fn(), media_type=ct)


def furl(name: str) -> str:
    return f"{BASE}/files/{name}"


# ------------------------------------------------------------------ auth + log
AUTH = {"fal": ("authorization", "Key "), "replicate": ("authorization", "Bearer "), "openai": ("authorization", "Bearer "),
        "openrouter": ("authorization", "Bearer "), "meshy": ("authorization", "Bearer "), "tripo": ("authorization", "Bearer "),
        "gemini": ("x-goog-api-key", ""), "elevenlabs": ("xi-api-key", "")}


async def record(req: Request, prov: str) -> dict:
    h, prefix = AUTH[prov]
    val = req.headers.get(h, "")
    if not val.startswith(prefix) or len(val) <= len(prefix):
        raise HTTPException(401, f"missing {h}")
    if "bad" in val:
        return None  # type: ignore[return-value]
    body: dict = {}
    ct = req.headers.get("content-type", "")
    if "json" in ct:
        body = await req.json()
    elif "multipart" in ct:
        form = await req.form()
        body = {k: (f"<file {v.filename} {len(await v.read())} bytes>" if hasattr(v, "filename") else v) for k, v in form.multi_items()}
    flat = json.dumps(body)
    LOG.append({"prov": prov, "method": req.method, "path": req.url.path, "query": str(req.url.query), "keys": sorted(body)[:30],
                "data_uri": "data:image" in flat or "data:audio" in flat, "multipart_files": "<file" in flat, "body": flat[:400]})
    return body


def bad():
    return JSONResponse({"error": {"message": "invalid api key"}, "detail": "invalid api key"}, status_code=401)


@app.get("/_log")
def get_log():
    return LOG


@app.delete("/_log")
def clear_log():
    LOG.clear()
    return {"ok": True}


def new_job(kind: str, **kw) -> str:
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"polls": 0, "kind": kind, **kw}
    return jid


def tick(jid: str) -> int:
    JOBS[jid]["polls"] += 1
    return JOBS[jid]["polls"]


# ------------------------------------------------------------------ fal (queue API)
def fal_result(endpoint: str, body: dict) -> dict:
    n = int(body.get("num_images") or 1)
    if "trellis" in endpoint:
        return {"model_glb": {"url": furl("mesh.glb"), "content_type": "model/gltf-binary", "file_name": "model.glb"}}
    if "hunyuan3d" in endpoint:
        return {"model_glb": {"url": furl("mesh.glb")}, "thumbnail": {"url": furl("thumb.png")}, "model_urls": {"glb": {"url": furl("mesh.glb")}}}
    if "tripo" in endpoint:
        return {"model_mesh": {"url": furl("mesh.glb"), "content_type": "model/gltf-binary"}, "rendered_image": {"url": furl("thumb.png")}}
    if "stable-audio" in endpoint:
        return {"audio": furl("clip.wav")}           # bare string, as in fal's example
    if "minimax-music" in endpoint or "speech" in endpoint or "elevenlabs" in endpoint:
        return {"audio": {"url": furl("clip.mp3"), "content_type": "audio/mpeg"}, "duration_ms": 600}
    return {"images": [{"url": furl("img.png"), "content_type": "image/png", "width": 64, "height": 48} for _ in range(n)], "seed": 1234}


@app.post("/fal/queue/{endpoint:path}")
async def fal_submit(endpoint: str, req: Request):
    body = await record(req, "fal")
    if body is None:
        return bad()
    if "prompt" not in body and "image_url" not in body and "input_image_url" not in body and "text" not in body:
        return JSONResponse({"detail": [{"msg": "field required: prompt"}]}, status_code=422)
    jid = new_job("fal", endpoint=endpoint, body=body)
    app_id = "/".join(endpoint.split("/")[:2])
    base = f"{BASE}/fal/queue/{app_id}/requests/{jid}"
    return {"request_id": jid, "status_url": base + "/status", "response_url": base, "cancel_url": base + "/cancel", "queue_position": 0}


@app.get("/fal/queue/{owner}/{app_}/requests/{jid}/status")
async def fal_status(owner: str, app_: str, jid: str, req: Request):
    if await record(req, "fal") is None:
        return bad()
    n = tick(jid)
    if n == 1:
        return {"status": "IN_QUEUE", "queue_position": 2}
    if n == 2:
        return {"status": "IN_PROGRESS", "logs": [{"message": "step 10/28"}]}
    return {"status": "COMPLETED", "logs": [], "metrics": {"inference_time": 1.2}}


@app.get("/fal/queue/{owner}/{app_}/requests/{jid}")
async def fal_response(owner: str, app_: str, jid: str, req: Request):
    if await record(req, "fal") is None:
        return bad()
    j = JOBS[jid]
    return fal_result(j["endpoint"], j["body"])


@app.put("/fal/queue/{owner}/{app_}/requests/{jid}/cancel")
async def fal_cancel(owner: str, app_: str, jid: str, req: Request):
    await record(req, "fal")
    return {"status": "CANCELLATION_REQUESTED"}


@app.get("/fal/api/models/pricing")
async def fal_pricing(req: Request):
    if await record(req, "fal") is None:
        return bad()
    return {"prices": [{"endpoint_id": "fal-ai/flux-2-pro", "unit_price": 0.03, "unit": "megapixels", "currency": "USD"}], "has_more": False}


@app.get("/fal/api/account/billing")
async def fal_billing(req: Request):
    await record(req, "fal")
    return JSONResponse({"detail": "missing scope billing:usage:read"}, status_code=403)


# ------------------------------------------------------------------ Replicate
def rep_output(slug: str, inp: dict):
    if "trellis" in slug:
        return {"model_file": furl("mesh.glb"), "color_video": furl("clip.mp3")}
    if "audio" in slug or "music" in slug or "speech" in slug:
        return furl("clip.mp3")
    if "recraft" in slug:
        return furl("img.webp")
    return furl("img.png")


def rep_pred(jid: str) -> dict:
    j = JOBS[jid]
    n = j["polls"]
    st = "starting" if n == 0 else "processing" if n == 1 else "succeeded"
    p = {"id": jid, "status": st, "model": j["slug"], "input": j["input"], "logs": "step 3/8\n" if n else "",
         "urls": {"get": f"{BASE}/replicate/api/predictions/{jid}", "cancel": f"{BASE}/replicate/api/predictions/{jid}/cancel"},
         "output": rep_output(j["slug"], j["input"]) if st == "succeeded" else None, "error": None}
    if st == "succeeded":
        p["metrics"] = {"predict_time": 4.2}
    return p


@app.post("/replicate/api/models/{owner}/{name}/predictions")
async def rep_create(owner: str, name: str, req: Request):
    body = await record(req, "replicate")
    if body is None:
        return bad()
    jid = new_job("rep", slug=f"{owner}/{name}", input=body.get("input") or {})
    return JSONResponse(rep_pred(jid), status_code=201)


@app.post("/replicate/api/predictions")
async def rep_create_version(req: Request):
    body = await record(req, "replicate")
    if body is None:
        return bad()
    jid = new_job("rep", slug="firtoz/trellis", input=body.get("input") or {})
    return JSONResponse(rep_pred(jid), status_code=201)


@app.get("/replicate/api/models/{owner}/{name}")
async def rep_model(owner: str, name: str, req: Request):
    await record(req, "replicate")
    return {"owner": owner, "name": name, "latest_version": {"id": "v" + "0" * 63}}


@app.get("/replicate/api/predictions/{jid}")
async def rep_get(jid: str, req: Request):
    if await record(req, "replicate") is None:
        return bad()
    tick(jid)
    return rep_pred(jid)


@app.post("/replicate/api/predictions/{jid}/cancel")
async def rep_cancel(jid: str, req: Request):
    await record(req, "replicate")
    return {"id": jid, "status": "canceled"}


@app.get("/replicate/api/account")
async def rep_account(req: Request):
    if await record(req, "replicate") is None:
        return bad()
    return {"type": "user", "username": "mock", "name": "Mock"}


# ------------------------------------------------------------------ OpenAI
@app.get("/openai/api/models")
async def oa_models(req: Request):
    if await record(req, "openai") is None:
        return bad()
    return {"data": [{"id": "gpt-image-2.5-sunburst"}, {"id": "gpt-4o-mini-tts"}, {"id": "gpt-6.1-terra"}]}


def oa_images(n: int, transparent: bool):
    return {"created": 1, "data": [{"b64_json": base64.b64encode(png(alpha=transparent)).decode()} for _ in range(n)],
            "usage": {"input_tokens": 60, "input_tokens_details": {"text_tokens": 40, "image_tokens": 20}, "output_tokens": 1056, "total_tokens": 1116}}


@app.post("/openai/api/images/generations")
async def oa_gen(req: Request):
    body = await record(req, "openai")
    if body is None:
        return bad()
    return JSONResponse(oa_images(int(body.get("n") or 1), body.get("background") == "transparent"), headers={"x-request-id": "req_mock_gen"})


@app.post("/openai/api/images/edits")
async def oa_edit(req: Request):
    body = await record(req, "openai")
    if body is None:
        return bad()
    if not any(k.startswith("image") for k in body):
        return JSONResponse({"error": {"message": "image is required"}}, status_code=400)
    return JSONResponse(oa_images(int(body.get("n") or 1), body.get("background") == "transparent"), headers={"x-request-id": "req_mock_edit"})


@app.post("/openai/api/audio/speech")
async def oa_tts(req: Request):
    body = await record(req, "openai")
    if body is None:
        return bad()
    return Response(audio("wav"), media_type="audio/wav", headers={"x-request-id": "req_mock_tts"})


# ------------------------------------------------------------------ Gemini
@app.get("/gemini/api/models")
async def gm_models(req: Request):
    if await record(req, "gemini") is None:
        return bad()
    return {"models": [{"name": "models/gemini-nano-banana-2.1"}, {"name": "models/gemini-3.8-flash"}]}


@app.post("/gemini/api/models/{rest:path}")
async def gm_generate(rest: str, req: Request):
    body = await record(req, "gemini")
    if body is None:
        return bad()
    mods = (body.get("generationConfig") or {}).get("responseModalities") or []
    if "AUDIO" in mods:
        part = {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm()).decode()}}
    else:
        part = {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png()).decode()}}
    return {"candidates": [{"content": {"parts": [{"text": "here you go"}, part]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1290}, "responseId": "gm_" + uuid.uuid4().hex[:8]}


@app.post("/gemini/api/interactions")
async def gm_interactions(req: Request):
    body = await record(req, "gemini")
    if body is None:
        return bad()
    return {"id": "int_" + uuid.uuid4().hex[:8], "steps": [{"type": "model_output", "content": [
        {"type": "audio", "mime_type": "audio/mpeg", "data": base64.b64encode(audio("mp3")).decode()}]}]}


# OpenAI-compatible chat of Gemini (for the assistant agent's text backend)
@app.post("/gemini/openai/chat/completions")
async def gm_chat(req: Request):
    if await record(req, "gemini") is None:
        return bad()
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


# ------------------------------------------------------------------ ElevenLabs
@app.post("/elevenlabs/api/sound-generation")
async def el_sfx(req: Request):
    if await record(req, "elevenlabs") is None:
        return bad()
    return Response(audio("mp3"), media_type="audio/mpeg", headers={"character-cost": "120", "request-id": "el_sfx_1"})


@app.post("/elevenlabs/api/text-to-speech/{voice}")
async def el_tts(voice: str, req: Request):
    if await record(req, "elevenlabs") is None:
        return bad()
    return Response(audio("mp3"), media_type="audio/mpeg", headers={"character-cost": "30", "request-id": "el_tts_1"})


@app.post("/elevenlabs/api/text-to-voice/design")
async def el_design(req: Request):
    body = await record(req, "elevenlabs")
    if body is None:
        return bad()
    b64 = base64.b64encode(audio("mp3")).decode()
    return {"previews": [{"audio_base_64": b64, "generated_voice_id": f"gv{i}", "media_type": "audio/mpeg", "duration_secs": 0.6} for i in range(3)],
            "text": body.get("text") or "Well met, traveller. The forge is hot and the steel is ready."}


@app.post("/elevenlabs/api/music")
async def el_music(req: Request):
    if await record(req, "elevenlabs") is None:
        return bad()
    return Response(audio("mp3"), media_type="audio/mpeg", headers={"song-id": "song1"})


@app.get("/elevenlabs/api/user/subscription")
async def el_sub(req: Request):
    if await record(req, "elevenlabs") is None:
        return bad()
    return {"tier": "creator", "status": "active", "character_count": 1200, "character_limit": 100000}


# ------------------------------------------------------------------ Meshy
def meshy_task(jid: str) -> dict:
    j = JOBS[jid]
    n = tick(jid)
    if n < 3:
        return {"id": jid, "status": "IN_PROGRESS" if n > 1 else "PENDING", "progress": 40 * n}
    return {"id": jid, "status": "SUCCEEDED", "progress": 100, "model_urls": {"glb": furl("mesh.glb"), "fbx": furl("mesh.glb")},
            "thumbnail_url": furl("thumb.png"), "texture_urls": [{"base_color": furl("img.png")}],
            "consumed_credits": j.get("credits", 20)}


@app.post("/meshy/api/v2/text-to-3d")
async def meshy_t3d(req: Request):
    body = await record(req, "meshy")
    if body is None:
        return bad()
    if body.get("mode") == "refine" and body.get("preview_task_id") not in JOBS:
        return JSONResponse({"message": "preview task not found"}, status_code=400)
    return {"result": new_job("meshy", credits=10 if body.get("mode") == "refine" else 20)}


@app.get("/meshy/api/v2/text-to-3d/{jid}")
async def meshy_t3d_get(jid: str, req: Request):
    if await record(req, "meshy") is None:
        return bad()
    return meshy_task(jid)


@app.post("/meshy/api/v1/image-to-3d")
async def meshy_i3d(req: Request):
    body = await record(req, "meshy")
    if body is None:
        return bad()
    if not str(body.get("image_url", "")).startswith(("data:image", "http")):
        return JSONResponse({"message": "image_url required"}, status_code=400)
    return {"result": new_job("meshy", credits=30 if body.get("should_texture") else 20)}


@app.get("/meshy/api/v1/image-to-3d/{jid}")
async def meshy_i3d_get(jid: str, req: Request):
    if await record(req, "meshy") is None:
        return bad()
    return meshy_task(jid)


@app.get("/meshy/api/v1/balance")
async def meshy_bal(req: Request):
    if await record(req, "meshy") is None:
        return bad()
    return {"balance": 970}


# ------------------------------------------------------------------ Tripo v3
@app.post("/tripo/api/files")
async def tripo_upload(req: Request):
    if await record(req, "tripo") is None:
        return bad()
    return {"code": 0, "data": {"file_token": "tok_" + uuid.uuid4().hex[:8]}}


@app.post("/tripo/api/generation/{kind}")
async def tripo_gen(kind: str, req: Request):
    body = await record(req, "tripo")
    if body is None:
        return bad()
    if kind == "image-to-model" and not str(body.get("input", "")).startswith("tok_"):
        return {"code": 2002, "message": "invalid input"}
    return {"code": 0, "data": {"task_id": new_job("tripo", credits=30 if body.get("texture") else 20)}}


@app.get("/tripo/api/tasks/{jid}")
async def tripo_task(jid: str, req: Request):
    if await record(req, "tripo") is None:
        return bad()
    n = tick(jid)
    if n < 3:
        return {"code": 0, "data": {"task_id": jid, "status": "queued" if n == 1 else "running", "progress": 30 * n}}
    return {"code": 0, "data": {"task_id": jid, "status": "success", "progress": 100,
                                "output": {"model_url": furl("mesh.glb"), "rendered_image_url": furl("img.webp")}}}


@app.get("/tripo/api/account/balance")
async def tripo_bal(req: Request):
    if await record(req, "tripo") is None:
        return bad()
    return {"code": 0, "data": {"balance": 480, "frozen": 0}}


# ------------------------------------------------------------------ OpenRouter
@app.post("/openrouter/api/images")
async def or_images(req: Request):
    body = await record(req, "openrouter")
    if body is None:
        return bad()
    n = int(body.get("n") or 1)
    return {"data": [{"b64_json": base64.b64encode(png()).decode()} for _ in range(n)], "usage": {"cost": 0.0312 * n}, "id": "gen-img-1"}


@app.post("/openrouter/api/chat/completions")
async def or_chat(req: Request):
    body = await record(req, "openrouter")
    if body is None:
        return bad()
    if "image" in (body.get("modalities") or []):
        url = "data:image/png;base64," + base64.b64encode(png()).decode()
        return {"id": "gen-1", "choices": [{"message": {"role": "assistant", "content": "", "images": [{"type": "image_url", "image_url": {"url": url}}]}}],
                "usage": {"cost": 0.039, "prompt_tokens": 10, "completion_tokens": 1290}}
    return {"id": "gen-2", "choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"cost": 0.0001, "prompt_tokens": 5, "completion_tokens": 1}}


@app.get("/openrouter/api/key")
async def or_key(req: Request):
    if await record(req, "openrouter") is None:
        return bad()
    return {"data": {"label": "mock", "usage": 1.25, "limit": 10, "limit_remaining": 8.75, "is_free_tier": False}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7899)
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
