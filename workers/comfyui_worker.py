"""Optional remote backend: the ComfyUI container (COMFYUI_URL, default http://127.0.0.1:8188). Runs in envs/core.

FLUX.2 klein-base 9B (fp8) text-to-image and reference edits through chained ReferenceLatent nodes. Graph and API use
from docs/research/image_snippets/comfyui_remote.py (node names checked against /object_info). Rules:
  * never queue behind somebody else: /queue must be empty, otherwise the job fails with a clear message;
  * one client_id per job, poll /history, POST /interrupt on cancel;
  * POST /free {"unload_models": true, "free_memory": true} afterwards: ComfyUI keeps models resident in the unified
    memory our local workers need.
The studio core marks this model ready only while ComfyUI answers (app/plugins/image.py).
"""
from __future__ import annotations

import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from studio_worker import Backend, InputError, serve, size_from_aspect

BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
UNET = os.environ.get("COMFYUI_KLEIN_UNET", "flux-2-klein-base-9b-fp8.safetensors")
CLIP = os.environ.get("COMFYUI_KLEIN_TE", "qwen_3_8b_fp8mixed.safetensors")
VAE = os.environ.get("COMFYUI_FLUX2_VAE", "flux2-vae.safetensors")


def _req(path, data=None, timeout=30):
    r = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        body = f.read()
        return json.loads(body) if body else {}


def upload_image(path: str) -> str:
    b = uuid.uuid4().hex
    name = f"studio_{uuid.uuid4().hex[:8]}{Path(path).suffix or '.png'}"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {mimetypes.guess_type(path)[0] or 'image/png'}\r\n\r\n").encode() + Path(path).read_bytes() + \
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
    r = urllib.request.Request(BASE + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())["name"]


def klein_graph(prompt, negative, w, h, steps, cfg, seed, refs=()):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative or ""}},
        "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "sig": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": w, "height": h}},
        "smp": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }
    pos = ["pos", 0]
    for i, name in enumerate(refs):
        g[f"li{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[f"le{i}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"li{i}", 0], "vae": ["vae", 0]}}
        g[f"rl{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [f"le{i}", 0]}}
        pos = [f"rl{i}", 0]
    g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": ["unet", 0], "positive": pos, "negative": ["neg", 0], "cfg": cfg}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                   "sampler": ["smp", 0], "sigmas": ["sig", 0], "latent_image": ["lat", 0]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "studio"}}
    return g


class ComfyBackend(Backend):
    name = "comfyui"
    max_loaded = 1

    def probe(self, model):
        try:
            _req("/system_stats", timeout=3)
        except (urllib.error.URLError, OSError, ValueError) as e:
            return False, f"ComfyUI is not reachable at {BASE} ({e})"
        return True, ""

    def load(self, model, ctx):
        try:
            _req("/system_stats", timeout=5)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise InputError(f"ComfyUI is not reachable at {BASE}: start the container first ({e}).") from None
        return True

    def run(self, job, ctx):
        p = job.params
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a prompt first.")
        q = _req("/queue")
        if q.get("queue_running") or q.get("queue_pending"):
            raise InputError("ComfyUI is busy with somebody else's queue; try again when it is idle.")
        refs = [upload_image(i["path"]) for i in job.inputs if i.get("type") == "image"][:4]
        if job.task in ("edit", "multi_ref") and not refs:
            raise InputError("Add the image to edit first.")
        w, h = size_from_aspect(p.get("aspect", "1:1"), int(p.get("size", 1024)), 16)
        steps, cfg = int(p.get("steps", 28)), float(p.get("guidance", 4.5))
        seed0 = int(p.get("seed", -1))
        if seed0 < 0:
            seed0 = random.randrange(2**31)
        n = max(1, int(p.get("batch", 1)))
        try:
            for i in range(n):
                ctx.check()
                seed = seed0 + i
                cid = uuid.uuid4().hex
                pid = _req("/prompt", {"prompt": klein_graph(prompt, p.get("negative", ""), w, h, steps, cfg, seed, refs),
                                       "client_id": cid})["prompt_id"]
                t = time.time()
                while True:
                    if ctx.cancelled:
                        _req("/interrupt", {})
                        ctx.check()
                    hist = _req(f"/history/{pid}")
                    if pid in hist:
                        break
                    ctx.progress(min(0.95, (time.time() - t) / max(10, steps * 2.5)) * 100, 100, "remote",
                                 f"image {i + 1}/{n}: ComfyUI working ({time.time() - t:.0f} s)")
                    time.sleep(1.0)
                outs = hist[pid].get("outputs") or {}
                imgs = [im for o in outs.values() for im in o.get("images", [])]
                if not imgs:
                    raise RuntimeError(f"ComfyUI returned no image: {json.dumps(hist[pid].get('status'))[:400]}")
                im = imgs[0]
                qs = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                data = urllib.request.urlopen(f"{BASE}/view?{qs}", timeout=60).read()
                f = job.out_dir / f"image_{i + 1}.png"
                f.write_bytes(data)
                ctx.output(f, "image", seed=seed, index=i, final_prompt=prompt, seconds=round(time.time() - t, 2))
        finally:
            try:
                _req("/free", {"unload_models": True, "free_memory": True})
            except (urllib.error.URLError, OSError, ValueError):
                pass
        return {"meta": {"remote": BASE, "width": w, "height": h}}


if __name__ == "__main__":
    serve(ComfyBackend())
