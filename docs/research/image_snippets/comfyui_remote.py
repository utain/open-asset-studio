"""Optional remote backend: a ComfyUI server (default http://127.0.0.1:8188, ComfyUI 0.20.1, torch 2.9.1).
Stdlib only. Builds an API-format graph for FLUX.2 klein-base 9B (fp8) t2i (+ optional reference images) and polls
/history. NOT run during the survey (heavy jobs on the shared GPU were off-limits); node names / inputs were checked
against GET /object_info. Models visible there (2026-11-05): flux-2-klein-base-9b-fp8, flux2_dev_fp8mixed (UNETLoader);
text encoders qwen_3_8b_fp8mixed (klein 9B), mistral_3_small_flux2_{bf16,fp4_mixed} (FLUX.2 dev), qwen_3_4b (Z-Image);
VAE flux2-vae; LoRA Flux2TurboComfyv2 / Flux_2-Turbo-LoRA_comfyui (FLUX.2 dev turbo); checkpoint SwarmUI_Z-Image-Turbo-FP8Mix.
Rules for the studio: check GET /queue is empty first, use a client_id, never hold the GPU while our own worker is loaded
(ComfyUI keeps models resident: POST /free {"unload_models": true, "free_memory": true} after a job).
"""
import json
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8188"


def _req(path, data=None):
    r = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as f:
        return json.loads(f.read() or b"{}")


def available_models():
    info = _req("/object_info/UNETLoader")["UNETLoader"]["input"]["required"]["unet_name"][0]
    return info


def upload_image(path):
    """POST /upload/image (multipart) -> name usable by LoadImage."""
    import mimetypes
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.split('/')[-1]}\"\r\n"
            f"Content-Type: {mimetypes.guess_type(path)[0] or 'image/png'}\r\n\r\n").encode() + open(path, "rb").read() + \
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
    r = urllib.request.Request(BASE + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())["name"]


def klein_graph(prompt, w=1024, h=1024, steps=20, cfg=5.0, seed=0, refs=()):
    """klein *base* (non-distilled): ~20-50 steps, real CFG ~4-5 with a negative prompt. refs = uploaded image names."""
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-base-9b-fp8.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}},
        "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "sig": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": w, "height": h}},
        "smp": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }
    pos = ["pos", 0]
    for i, name in enumerate(refs):  # FLUX.2 multi-reference: VAE-encode each ref and chain ReferenceLatent
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


def run(graph, timeout=900):
    cid = uuid.uuid4().hex
    pid = _req("/prompt", {"prompt": graph, "client_id": cid})["prompt_id"]
    t = time.time()
    while time.time() - t < timeout:
        h = _req(f"/history/{pid}")
        if pid in h:
            outs = h[pid]["outputs"]
            imgs = [im for o in outs.values() for im in o.get("images", [])]
            return [urllib.request.urlopen(f"{BASE}/view?filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}").read()
                    for im in imgs]
        time.sleep(1)
    _req("/interrupt", {})
    raise TimeoutError(pid)


if __name__ == "__main__":
    print(available_models())
    print(json.dumps(klein_graph("a treasure chest, game asset"))[:300])  # build only; call run() to queue
