# Image models on the DGX Spark: what runs, how, and what each is good for

Survey date 2026-11-05. Machine: DGX Spark (aarch64, GB10 sm_121, 121 GB unified memory, CUDA 13), offline.
Every number below comes from a real run on this box (seed 42, 1024x1024 unless stated).

**Important caveat: most runs shared the GPU with other agents** (llama-server, the audio studio, later vLLM). A
re-run of Qwen-Image-2.1 bf16 on an idle GPU gave **1.93 s/step and a 28.5 s load**. Under contention the same run
had given 4.3 s/step and 50 s. So:
* The contended s/step figures below are up to about **2.3x too slow**.
* Use them for *relative* comparison only (they were measured under similar load), except for the rows marked
  *idle*.
Test images, the raw timings (`results.jsonl`) and the bench harness lived in a throwaway scratch directory and are not included.
Working snippets: `docs/research/image_snippets/` (`loaders.py` is shared by all of them).

## TL;DR

* **Everything runs in the `envs/qwen` venv** (torch 2.9.0 cu130, diffusers 0.42.1,
  transformers 5.22.1). No new venv is needed. A separate test venv (torch 2.9.1 + torchao 0.16) was created only to test
  native INT8, which turned out **slower** here.
* Runnable today: **Qwen-Image-2.1**, **FLUX.1-Kontext-dev**, **Z-Image-Turbo**, **Lumina-Image-2.0** and
  **HunyuanImage-2.1**. The last three exist in the cache only as the unsloth INT8 transformer plus a base repo
  that has no transformer weights. We dequantise the INT8 file to bf16 at load time (`loaders.py`).
* **Not runnable**:
  * **FLUX.1-Krea-dev**: the repo has no transformer weights and no INT8 file is cached. It needs 24 GB bf16 or
    about 12 GB of unsloth INT8.
  * **Qwen-Image (v1) INT8**: it needs the Qwen/Qwen-Image VAE and configs (about 250 MB). Its text encoder,
    Qwen2.5-VL-7B, is already cached inside HunyuanImage-2.1-Diffusers. v1 is superseded by 2.1, so it is not
    worth adding.
* **The fast direct-to-GPU loader matters for every model.** Stock `from_pretrained().to("cuda")` reads at about
  170 MB/s here:
  * FLUX.1-Kontext: **186 s stock vs 36 s fast**.
  * Qwen-Image-2.1: about 180 s stock vs 28.5 s fast (idle GPU; 43-50 s contended).
  * The recipe has two parts:
    * diffusers models: `device_map="cuda", disable_mmap=True`
    * transformers models: build the lazy mmap model, then swap in tensors read with `safetensors.load_file(device="cuda")`
* **unsloth INT8/FP8 single files are not ComfyUI-only.** All of them load into diffusers:
  * v1 safetensors: `w = (qdata - zp) * scale`
  * v2 ConvRot: additionally multiply by the 256x256 Hadamard (`W = Wdeq @ H`), verified to within 1.3 % of the plain INT8 weight
  * legacy torchao `.pt`: stub the classes, then `int_data * scale`
  * FP8 text encoder: plain upcast
  * The INT8-ConvRot text encoder (comfy_quant format) also un-rotates correctly (3.9 % from the FP8 TE)

  On GB10, keeping them as int8 (torchao W8A8) is **1.7x slower** than bf16 (Z-Image: 3.26 vs 1.88 s/step) and saves
  about 6 GB. Unified memory is plentiful, so dequantise.
* Background removal (rembg u2net, CPU) and Real-ESRGAN x4 are local and work. Qwen-Image-2.1 generates **real
  RGBA** by itself.

## Capability matrix

Legend: ✅ works and was tested here · ☑️ supported by the pipeline, not tested here · ❌ not supported · — n/a

| Model (local form) | t2i | edit | multi-ref | inpaint (mask) | native RGBA | text rendering | neg. prompt | licence (commercial?) |
|---|---|---|---|---|---|---|---|---|
| **Qwen-Image-2.1** (bf16 31 GB, or unsloth INT8 + FP8 TE) | ✅ | ✅ instruction edit | ☑️ 1-10 images | ❌ (no mask input; describe the region in the prompt) | ✅ trained trigger sentence | ✅ excellent (EN + ZH) | only with `true_cfg_scale>1` (2x cost) | Apache-2.0 like Qwen-Image (check the 2.1 card) ✅ |
| **FLUX.1-Kontext-dev** (bf16 32 GB) | ✅ | ✅ instruction edit, 1 image | ❌ (one image; stitch for more) | ☑️ `FluxKontextInpaintPipeline` | ❌ (use rembg) | ✅ good | ❌ (true_cfg optional) | FLUX.1 [dev] Non-Commercial ⚠️ |
| **Z-Image-Turbo** (unsloth INT8 -> bf16, 12 GB TE + 12 GB DiT) | ✅ | ☑️ img2img only (`ZImageImg2ImgPipeline`) | ❌ (the Omni pipeline exists, but no Omni weights) | ☑️ `ZImageInpaintPipeline` | ❌ | ✅ very good EN, plus ZH | ❌ (guidance 0) | Apache-2.0 ✅ |
| **HunyuanImage-2.1** (unsloth INT8 -> bf16, 34 GB DiT + 15 GB TE) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ (ByT5 glyph encoder, OCR guider) | ✅ (real CFG via APG guider) | Tencent Hunyuan Community: commercial OK, but not in EU/UK/KR and not above 100 M MAU ⚠️ |
| **Lumina-Image-2.0** (unsloth INT8 -> bf16, 5 GB DiT + 5 GB Gemma2) | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ poor ("LOOOT") | ✅ | Apache-2.0 ✅ |
| FLUX.1-Krea-dev | ❌ weights missing | | | | | | | FLUX.1 [dev] NC ⚠️ |
| Qwen-Image (v1) INT8 .pt | ❌ needs VAE + configs | | | | | | | Apache-2.0 |
| ComfyUI: FLUX.2 klein-base 9B fp8, FLUX.2 dev fp8 | ☑️ | ☑️ reference edit | ☑️ (ReferenceLatent chain) | ☑️ | ❌ | ✅ | klein-base ✅ / dev ❌ | both FLUX Non-Commercial ⚠️ |
| ComfyUI: Z-Image-Turbo FP8Mix, SDXL base | ☑️ | | | ☑️ | | | | Apache / OpenRAIL++ ✅ |

## Measured performance (1024x1024, `envs/qwen`, fast loader)

| Model | load (fast) | load (stock) | s/step | steps used | total / image | torch peak alloc | system memory drop* |
|---|---|---|---|---|---|---|---|
| Qwen-Image-2.1 bf16, t2i (*idle*) | **28.5 s** | ~180 s | **1.93** | 20 | **41 s** | 36.8 GB | 57 GB |
| Qwen-Image-2.1 bf16, t2i (contended) | 43-50 s | — | 4.3-4.6 | 20 (12 for RGBA) | 90 s @20 steps | 36.8 GB | 49-52 GB |
| Qwen-Image-2.1 bf16, edit (1 ref) | — | — | 4.9 | 12 | 97 s (38 s to the first step: vision encode + ref latents) | 38.8 GB | 57 GB |
| Qwen-Image-2.1 unsloth INT8-ConvRot + FP8 TE, dequant (*idle*) | 13.5 s | — | **1.85** | 20 | 39 s | 36.8 GB | — |
| FLUX.1-Kontext-dev, edit | 36 s | **186 s** | 7.8 | 20 | 156 s | 33.8 GB | 55 GB |
| FLUX.1-Kontext-dev, t2i | — | — | 2.3 | 20 | 50 s | 33.8 GB | 58 GB |
| **Z-Image-Turbo** (INT8 -> bf16) | 12 s | — | **1.9** | 9 (= 8 NFE) | **18.6 s** | 21.6 GB | 30 GB |
| Z-Image-Turbo, torchao W8A8 (separate test venv) | 5 s | — | 3.3 | 9 | 31 s | 16.0 GB | 27 GB |
| Lumina-Image-2.0 (INT8 -> bf16) | 21 s | — | 1.6 (CFG on for the first 25 % of steps) | 30 | 50 s | 12.3 GB | 24 GB |
| HunyuanImage-2.1 (INT8 -> bf16), 1024 | 235 s (cold: 22 GB torch.load of the legacy .pt, then the 15 GB TE) | — | 2.8 (CFG on; DC-AE 32x means only 1024 tokens) | 20 | 59 s | 50.5 GB | — |
| HunyuanImage-2.1, 2048 (native) | — | — | 8.3 | 20 | 169 s | 54.2 GB (61 GB system) | — |
| Real-ESRGAN x4plus, 512 -> 2048 RGBA | 1 s | | | | 4.3 s | 2.2 GB | |
| rembg u2net (CPU onnxruntime), 1024 | 0.4 s | | | | 0.37 s | 0 | |

\* *System memory drop* is the fall in MemAvailable, which on unified memory is what counts. It includes the CUDA
context, the allocator cache and page cache that the mmap reads touched.

The torch 2.9.0 wheel warns that GB10 is "cuda capability 12.1, max supported 12.0". Every model ran fine anyway
(PTX/sm_120 kernels), so the warning can be ignored.

## Recommended default per task

| Task | Default | Why / settings | Fallback |
|---|---|---|---|
| **Fast draft / ideation** | **Z-Image-Turbo** | 18 s per 1024 image, very clean product-render look, good text. 9 steps, guidance 0. | Kontext t2i (50 s) |
| **Quality t2i** | **Qwen-Image-2.1** | Best prompt adherence and text, bilingual. 25-30 steps, true_cfg 1.0. | HunyuanImage-2.1 @2048 (8.3 s/step at 20 steps; rich stylised game-art look, best detail of all) |
| **Edit an image** (instruction) | **Qwen-Image-2.1 edit** | Keeps the geometry, the text and the gem exactly; multi-ref up to 10; output can be RGBA. 25 steps. | FLUX.1-Kontext (good, but smooths materials toward "plastic"; NC licence) |
| **Transparent asset (RGBA)** | **Qwen-Image-2.1** with the prompt prefix `This is an RGBA image with transparency.` | True alpha from the VAE (the fox test: 75 % fully transparent pixels, clean edges). No matting step. | Any model on a plain white background, then rembg u2net (birefnet-general for hair or fur) |
| **3D-ready concept** (Hunyuan3D input) | Qwen-Image-2.1 RGBA, single object, 3/4 view, centred, no ground shadow | Alpha removes the rembg step that hy3d does anyway; or Z-Image for speed then rembg | Z-Image-Turbo + rembg |
| **Texture / tileable** | none of these is trained for seamless tiling | Use Z-Image or Qwen with "seamless tileable texture, flat lighting, top-down orthographic", then make it tileable in post: offset by half and inpaint the seam with Qwen edit / Kontext, or circular padding (patch the VAE decoder and the DiT convs to `padding_mode="circular"`; not tested). ComfyUI has `SwarmTileableVAE` / `easy kSamplerTiled`. | — |
| **Upscale** | Real-ESRGAN x4plus (local, 4 s per 512->2048, handles alpha) | then optionally a low-denoise img2img pass with Z-Image | — |

## Per-model notes

### Qwen-Image-2.1 (baseline): `Qwen/Qwen-Image-2.1`
* **venv / recipe**: `envs/qwen`. See `image_snippets/qwen_image_21.py`; the production worker is
  `workers/qwen_worker.py`.
  * Components: `QwenImage21Pipeline`, `QwenImage21Transformer2DModel` (about 7 B, 14 GB bf16), text encoder
    `Qwen3VLForConditionalGeneration` (17 GB) and a 4-channel RGBA VAE `AutoencoderKLQwenImage21`.
  * `unsloth/Qwen-Image-2.1` is a configs-only mirror (1.3 GB VAE + configs). Use `Qwen/Qwen-Image-2.1`.
* **Tasks**:
  * t2i
  * edit and multi-reference: `image=[1..10 PIL]`, `output_resolution` 512/1024/2048 for the reference encoding
  * native RGBA: the output PIL image is RGBA
  * bilingual text rendering
* **Parameters**:

  | Parameter | Values |
  |---|---|
  | `num_inference_steps` | default 40; sane range 20-50; 25 is a good default. 12 is usable for previews, but slightly over-sharpened. |
  | `true_cfg_scale` | 1.0 = no CFG (the model is trained to work without it). 3-5 enables `negative_prompt` and costs 2x. |
  | `width` / `height` | multiples of 32; max side 2752; max area about 2400x1792 |
  | Good ratios | 1024², 1328x1024, 1024x1328, 1536x1024. 2K: 2048², 2400x1792, 2528x1696, 2752x1536. |
  | `use_kv_cache` | True (the worker default) |
  | Scheduler | dynamic exponential shift with `shift_terminal` 0.02. Leave it as shipped. |
* **Quality (seen)**:
  * The chest t2i is excellent: crisp "LOOT" plaque, correct materials, clean white background.
  * The RGBA fox is clean, with real alpha.
  * The edit changed only the colour and materials and kept the text and layout pixel-aligned.
* **Failure modes**:
  * Edits at few steps look over-sharpened.
  * The edit adds about 30 s before the first step (vision tower plus reference latents).
  * Peak memory is about 39 GB plus a 17 GB text encoder resident. Budget **≥ 56 GB** free, as `image_client.py`
    already does.
  * Negative prompts are silently ignored when `true_cfg_scale` is 1.
* **Prompt advice (for the improver LLM)**:
  * Write long, descriptive natural language (English or Chinese both work).
  * Put exact text in double quotes.
  * For a transparent asset, start the prompt with `This is an RGBA image with transparency.` and say "isolated,
    transparent background".
  * For edits, give an imperative instruction plus "keep everything else identical". Refer to references as
    "image 1", "image 2".
* **unsloth/Qwen-Image-2.1-FP8 files**: all four are usable from diffusers with `loaders.load_unsloth_safetensors`.

  | File | Format | How to load |
  |---|---|---|
  | INT8 | torchao v1 | dequant |
  | INT8-ConvRot | v2 | dequant, then `@ H` |
  | text_encoder-FP8 | plain e4m3 | `.to(bf16)` |
  | text_encoder-INT8-ConvRot | comfy_quant layout | `q*scale @ H` |

  Measured: INT8-ConvRot DiT + FP8 TE dequantised gives an image almost identical to bf16 (same composition, plaque and gem), 13.5 s load, 1.85 s/step (bf16 on the same idle GPU: 1.93 s/step, 28.5 s load). Because the weights end up as bf16, there is no speed or memory gain at runtime.
  The gain is the **faster load**: 16 GB is read instead of 31 GB, so 13.5 s instead of 28.5 s. Quality is
  equivalent by eye.
  * Use bf16 by default.
  * The INT8+FP8 pair is a valid "fast cold start" option.

### FLUX.1-Kontext-dev: `unsloth/FLUX.1-Kontext-dev` (full diffusers repo, bf16)
* **Recipe**: `FluxKontextPipeline` with a fast-loaded `FluxTransformer2DModel` (12 B), T5-XXL and CLIP-L. See
  `image_snippets/flux_kontext.py`.
* **Tasks**:
  * instruction edit of one image
  * t2i when `image=None`, which works but is not its strength
  * inpaint via `FluxKontextInpaintPipeline` (untested)
* **Parameters**:

  | Parameter | Values |
  |---|---|
  | `guidance_scale` | distilled guidance. 2.5 for edit, 3.5 for t2i. Range 1.5-4. |
  | `num_inference_steps` | 20-28 (default 28) |
  | `max_area` | 1024² |
  | `_auto_resize=True` | snaps the input to a Kontext-preferred resolution of about 1 MP |
  | `negative_prompt` | needs `true_cfg_scale>1` (2x cost; rarely useful) |
  | `max_sequence_length` | 512 |
* **Quality (seen)**:
  * The edit kept the shape, the plaque text and the background.
  * It flattened the wood grain into a smooth, plastic, toy-like finish and recoloured the gem.
  * Its t2i is a nice stylised, Pixar-ish chest with correct text.
* **Speed**: the edit runs at 7.8 s/step, because image tokens double the sequence. That is the slowest per step of
  all the models.
* **Failure modes**:
  * The non-commercial licence.
  * Only one input image.
  * It drifts the identity over chained edits.
  * It tends to restyle when the instruction is vague.
* **Prompt advice**:
  * Use Kontext instruction style: one concrete change per instruction ("Change X to Y while keeping the same pose,
    framing, text and background").
  * Name what must be preserved.
  * Do not re-describe the whole scene. Avoid pronouns ("it"); name the object.
  * Put text in quotes: `replace "LOOT" with "GOLD"`.
  * No negative prompt.

### Z-Image-Turbo: `unsloth/Z-Image-Turbo` (no DiT weights) + `unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-INT8.safetensors`
* **Recipe**: `ZImagePipeline`.
  * The text encoder (`Qwen3Model` 4B) and VAE come from `unsloth/Z-Image-Turbo`.
  * The DiT (6 B) comes from the INT8 file, via `load_unsloth_safetensors` (which restores the
    `unsloth_root::cap_pad_token` / `x_pad_token` root tensors).
  * See `image_snippets/prequant_models.py zimage`.
* **Tasks**:
  * t2i
  * img2img and inpaint pipelines exist (same weights, untested)
  * no edit
* **Parameters**:

  | Parameter | Values |
  |---|---|
  | `num_inference_steps` | 9, which is 8 NFE (range 6-12) |
  | `guidance_scale` | **0.0**. It is distilled, so there is no CFG and no negative prompt. |
  | Resolution | 512-2048, multiples of 16. 1024² and 1280x720-type ratios are good. |
  | `max_sequence_length` | 512 |
* **Quality (seen)**:
  * A photoreal, clean 3D-render chest.
  * The engraved "LOOT" is perfect.
  * It added a padlock and handles (it invents plausible detail).
  * Best value per second in the cache.
* **Failure modes**:
  * Low seed diversity (Turbo models collapse toward similar compositions).
  * It does not understand negative phrasing ("no shadow" can add a shadow).
  * fp16 is unsafe, so use bf16.
* **Prompt advice**:
  * Use short to medium natural sentences (1-3). Subject first, then style, then background.
  * Avoid long tag lists and negatives.
  * Text in quotes renders well.
  * Chinese is also supported.
* **torchao native INT8**: works in that separate test venv with `unflatten_tensor_state_dict`, but runs at 3.26 s/step vs
  1.88. Not worth it on GB10.

### HunyuanImage-2.1: `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` (no DiT weights) + `unsloth/HunyuanImage-2.1-FP8/HunyuanImage-2.1-INT8.pt`
* **Recipe**: `HunyuanImagePipeline`.
  * The DiT (17 B) is the base, non-distilled model (`guidance_embeds=false`), loaded from the legacy `.pt` with
    `load_unsloth_legacy_pt`.
  * Text encoders: Qwen2.5-VL-7B (15 GB) and ByT5 glyph T5 (0.9 GB).
  * VAE: 32x-compression DC-AE (0.8 GB).
* **Tasks**:
  * t2i only, at **2K native**: diffusers warns that 1K produces artefacts
  * text rendering through ByT5, with the OCR guider applied when the prompt contains quoted text
* **Parameters**:

  | Parameter | Values |
  |---|---|
  | `num_inference_steps` | 50 (default); 20-30 is usable |
  | Guidance | the base model uses the `guider` component (AdaptiveProjectedMixGuidance, scale 3.5), i.e. real CFG at 2 batches per step. `distilled_guidance_scale` is ignored for the base model. |
  | `negative_prompt` | supported |
  | Resolutions | 2048², 2560x1536, 1536x2560, 2304x1792 |
* **Quality (seen)**:
  * At 2048 (20 steps) it gave the most detailed, polished image of the survey: a "mobile game" stylised chest,
    perfect "LOOT", rope handle and clean white background.
  * At 1024 it is still decent, only slightly soft. The artefacts diffusers warns about did not show on this
    subject.
* **Failure modes**:
  * The heaviest model: 61 GB system memory while running.
  * Slow cold load (235 s). Convert the legacy `.pt` once into an int8 safetensors file (qdata + scale) and read it
    with `device="cuda"` for a load like Z-Image's.
  * No edit and no RGBA.
  * The licence's territory exclusions.
* **Prompt advice**:
  * Use long, structured descriptive prose (the model is trained on dense captions; Chinese or English).
  * Put text in quotes.
  * Upstream ships a "reprompt" LLM: our prompt-improver should expand short ideas into about 80-150 words.

### Lumina-Image-2.0: `unsloth/Lumina-Image-2.0` (no DiT weights) + `unsloth/Lumina-Image-2.0-FP8/Lumina-Image-2.0-INT8.pt`
* **Recipe**: `Lumina2Pipeline` with a `Gemma2Model` (2B) text encoder and a 2.6 B DiT from the legacy `.pt`. See
  `prequant_models.py lumina`.
* **Parameters**:

  | Parameter | Values |
  |---|---|
  | `num_inference_steps` | 30-50 |
  | `guidance_scale` | 4.0 |
  | `cfg_trunc_ratio` | 0.25 (CFG only for the first 25 % of steps, which is why it is fast) |
  | `cfg_normalization` | True |
  | `negative_prompt` | supported |
  | `system_prompt` | default "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts." |
  | Resolution | up to 1024-ish; trained at about 1 MP |
* **Quality (seen)**:
  * A painterly, cel-shaded stylised chest, nice for 2D game art.
  * The text was misspelt ("LOOOT").
  * The weakest model in the cache for prompt adherence.
* **Use**: optional stylised alternative only. Low priority.

### Not runnable from the cache
* **FLUX.1-Krea-dev** (`unsloth/FLUX.1-Krea-dev`): configs, VAE, CLIP and T5 are present, but the transformer
  shards and `unsloth/FLUX.1-Krea-dev-FP8` are missing. It would reuse the Kontext loader (`FluxPipeline`,
  guidance 3.5-4.5, 28 steps). It is non-commercial. Skip unless the user downloads it.
* **unsloth/Qwen-Image-FP8 / Qwen-Image-INT8.pt** (20 B Qwen-Image v1 DiT):
  * The `.pt` loads with `load_unsloth_legacy_pt`.
  * It still needs `Qwen/Qwen-Image` `vae/` (about 250 MB) plus `scheduler`, `tokenizer` and `transformer/config.json`.
  * The text encoder (Qwen2.5-VL-7B) is the same as HunyuanImage-2.1's `text_encoder`.
  * v1 is superseded by 2.1. Not worth adding.

## Utility models for game assets

| Utility | Local? | Notes |
|---|---|---|
| rembg u2net | ✅ `~/.u2net/u2net.onnx` (176 MB), rembg 2.0.65 in `envs/hy3d` | 0.37 s per 1024 image on CPU. Clean cut of the chest on white (tested). |
| BiRefNet (general / lite / HR) | ❌ (rembg session classes exist; it auto-downloads to ~/.u2net) | `birefnet-general` onnx is about 973 MB, `-lite` about 224 MB, MIT. Best for hair and fur. Tiny to fetch. |
| BRIA RMBG-1.4 / 2.0 | ❌ | Non-commercial (2.0 is also gated). Avoid. ComfyUI's `easy imageRemBg` would download them. |
| Real-ESRGAN x4plus | ✅ `engines/hy3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth` | realesrgan 0.3.0 + basicsr 1.4.2 in `envs/hy3d`; needs the `torchvision.transforms.functional_tensor` shim (in `postprocess.py`; the Hunyuan3D engine in `engines/hy3d/app` has `torchvision_fix`). RGBA handled. 4.3 s for 512->2048, 2.2 GB. `engines/hy3d/app/imgtools.Upscaler` already wraps it. |
| 4x-UltraSharp / anime ESRGAN variants | ❌ | spandrel-loadable; ~67 MB each if wanted |

## ComfyUI as an optional remote backend

A ComfyUI server (ComfyUI 0.20.1, torch 2.9.1, launched with `--fp8_e4m3fn-unet --use-sage-attention --reserve-vram 2`).
It was probed read-only:
* `/system_stats` and `/object_info` (1785 nodes) were read.
* `/queue` was empty.
* No jobs were queued.

**What it adds that diffusers here doesn't have locally**:
* **FLUX.2 klein-base 9B fp8** (`flux-2-klein-base-9b-fp8.safetensors` with TE `qwen_3_8b_fp8mixed` and
  `flux2-vae`). It does t2i and multi-reference edit through chained `ReferenceLatent`.
  * This is the *base* model, not the 4-step distilled klein: about 20-50 steps, CFG about 4-5.
* **FLUX.2 dev fp8mixed** (35 GB) with the Mistral-Small-3.2 TE (bf16 / fp4 mixed) and Flux2 Turbo LoRAs.
* SwarmUI's Z-Image-Turbo FP8Mix checkpoint and SDXL base.
* Nodes for alpha, layers, tiling and rembg:
  * `SaveImageWithAlpha`, `JoinImageWithAlpha`, `EmptyQwenImageLayeredLatentImage`
  * `SwarmTileableVAE`, `easy kSamplerTiled`
  * `RemBGSession+` (u2net, isnet-anime), `easy imageRemBg` (RMBG / BEN2 / Inspyrenet, which would auto-download)

**Licences**: FLUX.2 klein 9B and FLUX.2 dev are both under the FLUX Non-Commercial licence. Klein **4B** is
Apache, but it is not installed.

**Integration**: `image_snippets/comfyui_remote.py` (stdlib only).
* `POST /prompt` with an API-format graph and a client_id
* poll `GET /history/{id}`
* `GET /view` to fetch outputs
* `POST /upload/image` for references
* `POST /interrupt` to cancel
* `POST /free {"unload_models":true}` afterwards, because ComfyUI keeps models resident in the unified memory we share
* Graph: UNETLoader → CLIPLoader(type flux2) → CLIPTextEncode → [ReferenceLatent per ref] → CFGGuider →
  SamplerCustomAdvanced(Flux2Scheduler, euler) → VAEDecode → SaveImage

**Verdict**: worth having as an optional "FLUX.2 (ComfyUI)" backend when the container is up.
* The app must check `/system_stats` reachability and `/queue` first.
* The app must account for ComfyUI's resident memory before starting a local worker. The two should not run models
  at the same time.

## Integration notes for the studio app

* **Worker model**: one model per worker process; exit the process to free memory (the existing qwen_worker
  pattern). Generalise `qwen_worker.py` with a `--family` switch whose load functions come from
  `image_snippets/loaders.py` / `prequant_models.py`.
* **Memory budgets** (free MemAvailable needed): Qwen-Image-2.1 ≥ 56 GB · Kontext ≥ 55 GB · HunyuanImage-2.1
  ≥ 65 GB · Z-Image ≥ 30 GB · Lumina ≥ 25 GB.
* **Unified memory**: drop the INT8 state dict right after `load_state_dict(assign=True)`. A live reference doubles
  the peak.
* **Parameter UI per family**:
  * Qwen: steps, true_cfg (+negative when >1), RGBA toggle (prepends the trigger), refs 0-10, output_resolution.
  * Kontext: steps, guidance 1.5-4, 1 ref.
  * Z-Image: steps 6-12 only (no guidance or negative).
  * Hunyuan: steps, guidance, negative, 2K presets only.
  * Lumina: steps, guidance, negative, cfg_trunc.
