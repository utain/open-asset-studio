# Seamless PBR materials (Materials studio, `Image > Generate > Seamless`, Make tileable)

Task `material` (model `material-zimage`, handler `material.make`), task `tileable` (model `tileable-blend`, CPU, and Z-Image img2img), tool `material.sample_glb` (CPU).
Backend `material` shares `envs/qwen` (no new venv). Contract: `docs/CONTRACTS.md` section 13. Measured on the DGX Spark (aarch64, GB10, 121 GB unified memory), 2026-11-10, with the
GPU shared with other users. Every number below was measured in a live core on :7861, project `material-test`, unless it says otherwise.

## 1. What it does

| mode | input | base colour comes from | seams |
|---|---|---|---|
| text | a description of the surface | Z-Image Turbo, 9 steps, 1024 px, latent rolling + circular VAE | tileable by construction |
| photo | a photo (`image` slot) | centre-cropped 1024 square | `photo_tiling`: `blend` (offset blend, CPU, keeps the pixels in the middle), `regenerate` (Z-Image img2img, strength 0.35, redraws), `none` |
| variation | an existing material or image (`base` slot) | Z-Image img2img, `variation_strength` 0.2 to 0.8 (default 0.5), same prompt | tileable by construction |

Then the same chain for every mode: project palette on the **base colour only** (section 11 of the contracts, `STYLE_TASKS["material"] = "texture"`) -> PBRify SPAN networks estimate normal, height and
roughness (separate worker process, `workers/material_worker.py`) -> CPU finishing (`build_maps`: de-lighting, normal strength, height contrast, AO from height, roughness bias, ORM) -> files.
Output: `basecolor`, `normal` (OpenGL), `height` (16-bit), `roughness`, `metallic`, `ao`, `orm`, `preview.png`, `preview_{sphere,cube,plane}.gltf`, `material.json`; 1k, 2k or 4k.
Metallic is a constant (`metallic` param, default 0): the estimators do not predict it, and a guessed metal mask is worse than none.

## 2. The models and their licences

| part | model | licence | where |
|---|---|---|---|
| base colour | Z-Image Turbo (already in the studio) | Apache-2.0 | HF cache (`unsloth/Z-Image-Turbo`) |
| normal | `PBRify_NormalV3` (SPAN, 1x) | CC0-1.0 | PBRify_Remix v1.7.2 |
| height | `PBRify_Height` (SPAN, 1x) | CC0-1.0 | same |
| roughness | `PBRify_RoughnessV2` (SPAN, 1x) | CC0-1.0 | same |
| 4k up-scaling | `PBRify_UpscalerSPANV4` (SPAN, 4x) | CC0-1.0 | same |

PBRify_Remix is by Kim2091 ([release v1.7.2_ComfyOnly](https://github.com/Kim2091/PBRify_Remix/releases/tag/v1.7.2_ComfyOnly), the ComfyUI zip, 72.5 MB, sha256 verified by
`scripts/fetch_pbrify.py` together with each member). The four files are extracted to `~/.cache/pbrify/1.7.2` and are the `path:` weights of the registry entry, so a missing file shows as
`missing_weights` with a Download button that runs the script. The networks are tiny (a few MB each); a material is commercially usable (`commercial: true`, licence `CC0-1.0` on the estimators, the
asset takes the most restrictive licence of everything it used, including a photo or base material). The SPAN architecture is re-implemented in `workers/span_lite.py` (about 100 lines, no
spandrel dependency; credits in `THIRD_PARTY_NOTICES.md`).

### The stale `eval_conv` finding

Every `Conv3XC` block of a SPAN checkpoint stores its training branches (`conv.0/1/2` and the skip `sk`) **and** a re-parametrised 3x3 `eval_conv` that is stale: it is not the fusion of those
branches. Loading the checkpoint as it is (what a naive loader and the ComfyUI node do with `eval_conv`) gives wrong maps: the 4x upscaler returns black, the height map is a flat grey. We rebuild the
kernel from the branches in numpy (`material_common.reparam_conv3xc`, `span_lite.load_span`). Largest per-layer difference between the stored `eval_conv` and the rebuilt kernel: **1.1 in NormalV3, 24 in the
4x upscaler**. `tests/test_material.py::test_reparam_conv3xc_*` checks the fusion numerically against the un-fused three-conv stack.

## 3. Tileable generation (Z-Image Turbo)

`TileMode` (`workers/qwen_worker.py`) is a context manager around one pipeline call:

* **latent rolling**: every transformer call rolls the latent by a random even offset (whole 2x2 patches) and un-rolls the prediction, so the model never sees a fixed border;
* **circular VAE**: `padding_mode="circular"` on every padded `Conv2d` of the VAE decoder (and encoder for img2img).

They must be used together. Seam score (p95-normalised, section 4) on the prototype prompt set: plain generation 1.32 to 2.63, **rolling alone 1.96 to 3.16 (worse)**, rolling + circular VAE **0.83 to 1.17**.
The pipeline is shared with plain `t2i`, so `__exit__` restores `forward` and zero padding also on errors; `Seamless` off is byte-identical to before.
(These three ranges are from the first prototype run; the prototype files did not survive a machine reboot, the live numbers below are new.)

Live text results (1024 px): cobblestone seam 0.945 (base colour), 0.891 (normal), 0.786 (height); the 3x3 grid shows no visible join. A red brick wall scored 1.376 on the 1k base colour (1.125 at 4k)
although the 3x3 grid shows no seam: regular courses and mortar lines score high, see section 4.

## 4. The seam score

`material_common.seam_score(img)`: per axis, the mean absolute difference across the wrap edge divided by the 95th percentile of the same statistic over all interior boundaries; the worst axis wins.
`<= 1.05` seamless, `<= 1.25` acceptable, above that a seam may show (a yellow or red badge on the card and in the viewer, and a line in the asset `warnings`). It is a **hint**: grout, brick courses and plank
lines can score 1.2 to 1.5 without a visible join, and a faint one-pixel line in a noisy texture can score 1.7 (see the regenerate seed 1 case below). Always confirm with the 3x3 view.

## 5. Make tileable: blend, regenerate, (inpaint not used)

| method | where | time (1400 x 1100 photo, then 1024 square) | seam before -> after | what changes |
|---|---|---|---|---|
| offset blend (`tileable-blend`, `photo_tiling: blend`) | CPU | 0.3 s (tool), 0.1 s in the job | 3.67 -> 0.87 (tool, 1400 x 1100), 2.79 -> 0.90 (job, 1024) | keeps the pixels in the middle, blends a 25 % band at the borders, keeps contrast |
| regenerate (`photo_tiling: regenerate`, img2img 0.35) | GPU | 7.7 s (warm) | 2.79 -> 0.96 (seed 2) | redraws the whole picture, also removes the gradient lighting of the photo |
| regenerate, a different seed | GPU | same | 2.79 -> 1.77 (seed 1, **a faint centre line is visible**) | the score is right: use another seed or the blend method |

An inpaint of the seam band was not built: blend is enough for flat photos, regenerate for the rest, and both are already fast.
The test photo (cobbles with a strong brightness ramp and no wrap match) after blend keeps its stones; after regenerate it is a similar but different pattern (`material_photo_modes_tiled.png`).

### Profiling the slow `regenerate` mode (fixed)

`regenerate` took 25.7 to 26 s where about 8 s were expected. Phase timing (hooks with `cuda.synchronize` around the VAE encoder, the transformer and the VAE decoder, now logged on every run as
`tileable phases: ...`) showed the VAE at 0.4 and 0.6 s and the transformer at 6.0 s per call against 1.3 s in text mode, with a peak of 43 GB against 22 GB. Cause: diffusers'
`Pipeline.from_pipe(pipe)` defaults to `torch_dtype=float32` and calls `.to(float32)` on the **shared** modules, so the bf16 transformer, text encoder and VAE became fp32 in place. It also slowed every later
`t2i` job in the same worker. Fix (`workers/qwen_worker.py`): `from_pipe(pipe, torch_dtype=torch.bfloat16)`; a unit test pins it.

| | transformer per call | `regenerate` total | peak GPU |
|---|---|---|---|
| before | 6.0 s | 25.9 s | 43.1 GB |
| after | 1.55 s (4 calls = 6.2 s) | 7.7 s | 22.1 GB |

Variation at strength 0.5 (5 calls): transformer 8.4 s.

## 6. The map estimators and up-scaling

Base colour: 1k is used as generated; 2k and 4k run the 4x SPAN up-scaler (1024 -> 4096, circular padding, fp16) and then the exact periodic box filter down to 2048 for 2k. The estimators run on the base colour at
`map_res = min(size, 2048)` (a 4k material is estimated at 2048 and the normal, height, roughness and AO are up-scaled 2x with a wrap-safe resample, `resize_wrap`: exact periodic filters, never PIL's clamping
resize on a tileable image).
All the nets use circular padding, so the maps stay tileable, which the normal and height seam scores confirm (0.89 and 0.79 on the cobble).

Measured in the live worker (warm, 1024 base colour):

| size | estimation (normal / height / rough) | worker total | peak GPU (material worker) | total job (text, warm Z-Image) |
|---|---|---|---|---|
| 1k | 4.1 / 2.0 / 2.0 s cold, 0.27 / 0.27 / 0.27 s warm | 8.4 s cold, 1.2 s warm | 2.08 GB | about 25 s (15 s base + 1 to 9 s maps + 0.2 s post) |
| 4k (estimated at 2048) | 0.67 / 0.63 / 1.05 s + 2.3 s up-scale + 3.6 s PNG writes | 8.4 s | 4.18 GB | 41.6 s (15.7 s base, 8.4 s maps, 2.7 s post, the rest PNG encoding of 33 + 25 + 26 MB maps) |

The very first 1k text job on a cold core: 39.9 s (Z-Image load 13 s, generation 15 s, maps 8.9 s cold). Registered `memory_gb` of `material-zimage` is 30 (Z-Image dominates);
the maps worker itself needs 2 to 4 GB, the registry value only drives the eviction check.

## 7. Normal map convention and packing

Stored normals are **OpenGL (Y+)**; DirectX only at export. The PBRify normal net outputs DirectX, so the handler flips green and renormalises. `render_sphere` (the thumbnail) lights the sphere from
the upper left, in front with the stored normal: a wrong convention makes the stones look dented, which is also a unit test (`test_normal_convention_opengl_bump`). Layouts reuse `engine_common.pack_textures`:

| layout | files | normal |
|---|---|---|
| `gltf` | basecolor, normal, orm | OpenGL |
| `unreal` | `T_<name>_BC`, `_N`, `_ORM` | DirectX (green flipped; verified pixel-exact) |
| `unity_urp` | `_BaseMap`, `_Normal`, `_MetallicSmoothness` (alpha = 1 - roughness), `_Occlusion` | OpenGL |
| `unity_hdrp` | `_BaseMap`, `_Normal`, `_MaskMap` (R metallic, G AO, A = 1 - roughness) | OpenGL |
| `godot` | `_albedo`, `_normal`, `_orm` | OpenGL |
| `separate` | basecolor, normal, height, roughness, metallic, AO | OpenGL |

Every layout also gets the 16-bit height, single roughness / metallic / AO images, `<stem>.material.json` (layout, size, seams, tiling, licence) and an optional `<stem>_sample.glb`
(1024 px sphere with the textures embedded, about 6.4 MB). A 1k cobble zip is 14.4 to 16 MB. `size` never up-scales (asking 2048 from a 1024 material keeps 1024).

## 8. Interface limits

* The 3D viewer changes the texture repeat of the loaded model-viewer scene (`mvScene`), so repeat 1 to 8 needs no reload; the 3x3 view tiles any map as a CSS background.
* Maps of a 4k material are 25 to 33 MB PNGs each; the library card and the sphere preview are made from the 512 px maps.
* `physical_size_m` is stored as `null`: the generators do not know real-world size.

## 9. Known limits

* The seam score gives false alarms on regular patterns (brick, tile) and can miss nothing worse than 1.05 on a clean result; it cannot tell a 1-pixel line from a pattern. Use the 3x3 view.
* A regenerate or variation can still leave a faint line for some seeds (seed 1 above); the warning says so. A cheap automatic repair was not added: a blend would ghost a regular pattern that only scores high.
* Metallic is constant; metals (`metallic` slider 1) work for bare metal only. Height is an estimate from one image, not a scan: do not use it for displacement beyond a few millimetres.
* Z-Image sometimes adds perspective or objects despite the suffix `top-down orthographic view, flat even diffuse lighting, no shadows, no perspective, no objects`; describe the surface only.
* The estimators are small image-to-image networks (PBRify_Remix is meant for remastering game textures): the height and roughness are plausible estimates, not measurements, and smooth or organic surfaces are the weakest case.
* x86_64 and Windows are untested.
