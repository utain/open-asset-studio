# Project style guide: findings and follow-ups

Feature: `docs/CONTRACTS.md` section 11 (`app/style_common.py`, `app/plugins/style.py`, `web/studios/style.js`).
Machine: NVIDIA DGX Spark (GB10, CUDA 13), 2026-11-09, project `style-test`, 6 jobs, seed 1234 (and 1235).

## Setup

One style string ("hand-painted stylized fantasy game item icon, thick dark brown outlines, flat cel shading with two tones, limited warm
palette of teal, ochre and deep crimson, soft rim light, centred on a plain cream parchment background"), three subjects (potion, sword,
boot). The reference is the Z-Image potion. Metric: RGB histogram intersection with the reference (8 bins per channel), mean corner
(background) colour, mean saturation, share of dark pixels, edge energy.

| # | run | worker time | hist intersection with ref | background RGB (delta to ref) | look |
|---|---|---|---|---|---|
| 0-1 | Z-Image, potion, text only (x2) | 30.4 s / 2 images | 1.00 / 0.84 | 245,227,193 (0) | the reference |
| 2-3 | Z-Image, sword, text only | 30.5 s / 2 | 0.76 / 0.77 | 246,234,206 (15) | same outlines, background, shading |
| 4-5 | Z-Image, boot, text only | 30.7 s / 2 | 0.76 / 0.75 | 242,225,191 (4) / 234,216,182 (19) | same; teal panels on the boot |
| 6 | Qwen-Image 2.1, sword, text only | 33.8 s | 0.07 | 255,250,231 (45) | different look: near-white background, painterly, teal blade |
| 7 | Qwen-Image 2.1, sword, text + 1 reference | 53.9 s | 0.78 | 253,246,217 (32) | background, framing and ground shadow follow the reference; no potion features leaked |
| 8 | Qwen-Image 2.1, boot, text + 1 reference | 53.7 s | 0.79 | 253,245,216 (31) | background follows the reference; weaker outline and cel shading than the Z-Image boots |

## What the design took from it

1. On one model a concrete style text already gives strong consistency. It must name the medium, line, shading, light and background; "fantasy
   style" alone is not enough. So `positive` + `background` are the core of the guide and apply everywhere.
2. The model matters more than any style mechanism: the same text on Qwen looked clearly different from Z-Image. The Style page shows which
   model made each reference and the preview/chip note says "References were made with X; Y may look different". It records this and does not enforce it.
3. A reference image fixes cross-model drift in colour, background and composition (histogram intersection 0.07 -> 0.78, background delta 45 -> 32) but transfers
   less of the rendering technique (outline weight). Cost: +20 s per image (+59 %) at 1024 px / 20 steps with one reference
   (`style_common.SEC_PER_REF_IMAGE`). So references are attached at `normal` (max 2) and `strong` (max 4) only. The phrasing "only as a style reference,
   do not draw its subject" gave no subject leakage in 2 of 2 samples (small sample).
4. Colour names in the prompt leak into the subject (a teal blade, teal panels from "teal, ochre and crimson"). Palette words are off by default
   (`palette.in_prompt`) and forced on only at `strong`; the explicit tool is the palette post-process.
5. Palette post-process on the CPU, at most 0.2 s per 1024 px image: the Lab mean/std transfer at 0.6 pulled the background toward the reference but
   over-tinted (reddish cast), so the default is `tint_strength` 0.35 and the implementation clips the std ratio and keeps the lightness contrast. A hard 6-colour
   lock turned the brass pommel red and the blade grey, so `lock` is for designed palettes (sprites, flat icons). The untouched output is kept as `<stem>_raw`.
6. The Qwen-Image avoid list does nothing at the default `true_cfg = 1` (the worker logs "negative prompt ignored"), and Wan's built-in default negative
   contains "style, artwork, painting" (the Chinese terms for style / work / painting), which works against a stylised guide.
   So: the avoid list goes only into a real negative prompt (`features.negative` true, or `cfg` with the CFG value above 1), the preview says why it was
   skipped, and for Wan / LTX / Lumina a quality-only base negative (`style_common.BASE_NEGATIVE`) is kept instead of the model's default.

## Wan negative note

A worker uses its own built-in negative prompt ONLY when `params.negative` is empty. Adding our avoid terms to an empty negative would therefore silently drop that
default, so `compose` starts from `BASE_NEGATIVE[prompt_style]` (recorded in `x_style.added.base_negative`). The Wan one is deliberately not the worker default:
it leaves out the style / artwork / painting terms.

## 3D scope

Hunyuan3D-2.1 texture is image-conditioned and takes no prompt: it projects the appearance of its input image. A style reference showing a different object would paint that
object's colours onto the mesh, so the style is never applied to `shape`, `texture` or `multiview`. The guide acts upstream: the concept image (`t2i` / `rgba`, background
dropped automatically) is styled, and the shape and texture are made from it. Text-to-3D (cloud) uses the `texture` variant as text only.

## Not measured (follow-ups)

* FLUX.2 klein t2i with ReferenceLatent references. Until verified, `flux2-klein-comfy` does not declare `features.style_ref` (so it gets text and avoid only).
* More than one reference at once, and `rgba` with references.
* The prompt improver with the style context line (needs the 35B prompt LLM loaded); `POST /api/improve` is covered by a unit test with a stubbed improver only.
* A numeric style-consistency metric (CLIP or DINO similarity to the references): no such model is in envs/core; automatic drift alerts depend on it.
* Describe-from-reference quality (`prompts/style.yaml`, `style_describe`) with a vision-capable local LLM.
* ETA: the extra reference time is shown in the chip and preview, but the Generate-button estimate does not include it yet.
* Several named guides per project, a global style library, LoRA / IP-Adapter, a "Restyle concept" button for 3D: out of scope for v1.

## Integration points for later work

* Seamless PBR materials: a module-level `STYLE_TASKS = {"material": "texture"}` in the plugin file (or `core.style.register_task`); mark texture maps with
  `x_material: {map: "basecolor" | "normal" | "roughness" | "orm" | "height"}` so the palette post-process touches only base-colour maps.
* LOD / collision / engine export: copy `x_style.{name, version, hash}` per asset into the manifest; ship `data/projects/<p>/style/versions/vNNNN.json` as
  `style_guide.json` for audit. Post-output processing uses `jobs.OUTPUT_HOOKS` (a no-op unless the plugin's own meta flag is set); "Export to engine" can be a
  `registerSelectionAction` button in the Library.
