# 3D: more open-weight models (image to mesh, multi-view, auto-rigging)

Added on top of the local Hunyuan3D 2.1 (backend `hy3d`, shape + texture): twelve models in four new backends, two new
tasks (`multiview`, `rig`) and a reworked 3D studio (`app/web/studios/3d.js`). Everything runs on the DGX Spark (aarch64, GB10 /
sm_121, torch 2.9.0+cu130, **no nvcc**). Registry: `models.d/3d_more.yaml`. Test data: TripoSR's `examples/robot.png`
(a normal object photo), UniRig's `examples/tira.glb` and a TripoSG bust of `police_woman.png` (rigging).

## What was added

| backend (env) | models | tasks | worker |
|---|---|---|---|
| `stab3d` | TripoSR, Stable Fast 3D (SF3D), Stable Point-Aware 3D (SPAR3D), Zero123++ v1.2, SV3D-u, SV3D-p | shape, multiview | `workers/stab3d_worker.py`, `stab3d_multiview.py` |
| `shape3d` | TripoSG, Hunyuan3D-2mini (+ turbo), Step1X-3D geometry | shape | `workers/shape3d_worker.py`, `shape3d_step1x.py` |
| `trellis` | TRELLIS image-large | shape | `workers/trellis_worker.py`, `trellis_shims.py` |
| `rig` | UniRig (skeleton + skin) | rig | `workers/rig_worker.py`, `rig_runner.py`, `rig_shims.py` |

Output meshes are GLB, +Y up, front +Z, with `input.png` attached and the metadata flags `textured` / `vertex_colors`
(same convention as the hy3d worker). `max_faces` (default 300 000) decimates dense results (TripoSG / Hunyuan3D / TRELLIS
produce 1-2 M faces) and removes floating fragments. `multiview` writes `sheet.png` + `view_NN.png` (SV3D also an orbit
video); `rig` writes `rigged.glb` (original mesh + glTF skeleton + skin, JOINTS_0 / WEIGHTS_0) and `skeleton.json`.

Shared code edits (small, additive): `app/registry.py` (`Registry.SLOT_TYPES` now also accepts `image` and `mesh` input slots),
`docs/CONTRACTS.md` (same), `app/web/core/shell.js` (nothing new needed), `README.md`, `docs/SETUP.md` section 3,
`THIRD_PARTY_NOTICES.md`, `vendor/vendor.lock`.

## Results (GB10, one job at a time)

Time is the generation step after loading (weights already in page cache); peak is `torch.cuda.max_memory_allocated`.

| model | task | licence | commercial | weights | time | peak GPU | status |
|---|---|---|---|---|---|---|---|
| TripoSR | shape | MIT | yes | 1.6 GB | 2-4 s (marching cubes 256) | 2.3 GB | ready; vertex colours |
| SF3D | shape | Stability AI Community | yes, under USD 1M annual revenue | 3.8 GB (gated) | 2.3 s | 6.0 GB | ready; baked PBR texture, UV unwrapped, ~32k faces |
| SPAR3D | shape | Stability AI Community | yes, under USD 1M | 6.9 GB (gated) + OpenAI CLIP ViT-L/14-336 (0.9 GB) | 4.3 s | 9.95 GB | ready; baked PBR texture |
| TRELLIS image-large | shape | MIT | yes | 3.1 GB + DINOv2 reg4 1.2 GB | 16.7 s (12 + 12 steps) | 11.1 GB | ready; vertex colours from the Gaussian decoder |
| TripoSG | shape | MIT | yes | 7.5 GB | 58 s (50 steps, octree 512) | 5.5 GB | ready; sharp, geometry only |
| Hunyuan3D-2mini turbo | shape | Tencent Hunyuan Community | yes (not EU / UK / South Korea) | 7.9 GB (turbo + standard share the repo) | 3.6 s (5 steps, octree 384) | 7.0 GB | ready; geometry only |
| Hunyuan3D-2mini | shape | Tencent Hunyuan Community | same | same repo | 46 s (30 steps) | 4.6 GB | ready; geometry only |
| Step1X-3D geometry | shape | Apache-2.0 | yes | 6.8 GB | 17.9 s (30 steps, octree 384) | 6.7 GB | ready; geometry only |
| Zero123++ v1.2 | multiview | Apache-2.0 (verify) | yes | 5.2 GB | 16.3 s (75 steps, 6 views of 320 px) | 5.0 GB | ready |
| SV3D-u / SV3D-p | multiview | Stability AI Community | yes, under USD 1M | 9.4 GB each (gated) | 203 s (50 steps, 21 views of 576 px + orbit.mp4) | 20.7 GB | ready (sv3d-p tested with elevation 20, same time and memory) |
| UniRig | rig | MIT | yes | 5.5 GB + OPT-350m config | 75 s on a 115k-face character (skeleton 27 s, skin 22 s, export 24 s); 47 s on a 30k-face bust | n/m (subprocess) | ready |

Geometry-only models give a white mesh: use the Texture tab (Hunyuan3D 2.1, `hy3d`) on them. Texture baking by nvdiffrast /
custom_rasterizer needs nvcc, so none of the models here call it.

## Models not (fully) added, and why

| model | outcome | reason |
|---|---|---|
| Stable Zero123 | not implemented | licence is non-commercial (Stability non-commercial research terms) and Zero123++ / SV3D cover the use case; the entry would only be a research toy |
| TRELLIS.2-4B | blocked | needs `flexgemm`, `o-voxel`, `cumesh` (custom CUDA kernels built with nvcc); no aarch64 wheels, no pure-PyTorch fallback is practical |
| TRELLIS texture (GLB bake) | skipped | needs nvdiffrast + diffoctreerast + a CUDA Gaussian rasterizer; replaced by vertex colours sampled from the Gaussian decoder's DC colours |
| Hunyuan3D-2 texture, Step1X-3D texture | skipped | `custom_rasterizer` / differentiable renderer need nvcc; paint with Hunyuan3D 2.1 on the Texture tab |
| SF3D / SPAR3D remesh options | off | `gpytoolbox`, `pynanoinstantmeshes` have no aarch64 wheel and need CMake + CUDA; stubbed, options are not offered |
| xformers (SV3D / sgm) | not installed | replaced by a PyTorch SDPA stand-in for `xformers.ops.memory_efficient_attention` injected into `sgm` |
| UniRig Blender path (bpy, FBX) | replaced | no `bpy` for aarch64; GLB in, npz exchange and glTF skin out (see below). Mixamo-style retargeting / FBX export is therefore not offered |

## How each hard dependency was replaced

No upstream file in `vendor/` is edited: everything is a stand-in installed before the upstream import.

- **spconv** (TRELLIS, UniRig): `workers/trellis_shims.py` has a pure-PyTorch `SubMConv3d` / `SparseConv3d` / `SparseConvTensor` with a
  neighbour hash; the test compares it against a dense `conv3d`.
- **flash_attn** (TRELLIS, UniRig): varlen qkv / kv-packed functions on `scaled_dot_product_attention` with padding; a `MHA` module with
  flash-attn's parameter layout (`Wq` / `Wkv` / `out_proj`) for UniRig. UniRig's OPT decoder runs on `attn_implementation: sdpa`.
- **torchmcubes** (SF3D / SPAR3D / TripoSR): CPU `skimage` marching cubes behind the same function.
- **texture_baker / uv_unwrapper** (SF3D, SPAR3D): built CPU-only (`USE_CUDA=0`, `--no-build-isolation`, from a copy); the baker's
  `rasterize` / `interpolate` are wrapped to move tensors to the CPU and back (`_patch_baker`).
- **diso** (TripoSG, Step1X-3D): stubbed; TripoSG uses its hierarchical decoder, Step1X-3D skimage marching cubes.
- **FlexiCubes** (TRELLIS submodule, empty in the clone): cloned as `vendor/FlexiCubes` (Apache-2.0) and registered under the
  package name the import expects; `kaolin` (debug shape checks only) is stubbed.
- **torch_scatter / torch_cluster** (UniRig): `segment_csr` on `torch.segment_reduce`, farthest point sampling in plain torch.
- **UniRig voxelisation** (pyrender / OpenGL in the original): CPU re-implementation (surface sampling plus an axis-column fill using
  the same two-of-three rule as the open3d branch).
- **Stubs for importable-only packages**: bpy, open3d, pyrender, fast_simplification, wandb, utils3d, transparent_background.
- **Upstream bugs worked around in the worker**: Step1X-3D `check_inputs` rejects PIL / tensor inputs (inverted `isinstance`), so the
  worker passes a temp PNG path; its vendored DINOv2-with-registers `_init_weights` breaks on transformers 4.57 (patched to a
  no-op, weights come from the checkpoint).

## UniRig design (no Blender)

1. The input GLB is converted from +Y up to Blender's +Z up (`(x, -z, y)`), decimated to `max_faces` (default 50 000) and written as
   UniRig's `raw_data.npz`.
2. UniRig's own `run.py` runs twice in a subprocess (`workers/rig_runner.py`, with the shims), in a temp directory with a copy of its
   `configs/`; the task YAMLs are rewritten so the writers emit npz only (no FBX / OBJ export):
   `predict_skeleton.npz` (autoregressive skeleton, beam search), then `predict_skin.npz`.
3. UniRig's `reskin` transfers the sampled skin weights to every original vertex; joints are mapped back from the normalised frame
   (`joints * scale + center`) and from Blender to glTF (`(x, z, -y)`).
4. A glTF skin is written with `pygltflib`: joint nodes, inverse bind matrices (checked equal to the inverse of the joint world
   matrices), `JOINTS_0` / `WEIGHTS_0` (top 4, weights sum to 1).

Checked on `tira.glb` (52 joints) and on a generated bust (16 joints): the skeleton overlay sits inside the mesh in front and side
view, and vertices are mostly weighted to nearby joints (median distance to the dominant joint 0.10 for a 1.28 m character).
Limitations: the skeleton predictor does not rig tails or loose accessories; it works best on a character in an A- or T-pose.

## Install

`./install.sh` builds the four envs from `install/<backend>.yaml` (torch cu130, `env-lock/<backend>.txt`, steps, vendor clones):

```
./install.sh backend stab3d shape3d trellis rig    # envs + vendor clones (no weights)
./install.sh model sf3d spar3d sv3d-u       # weights, one repo per call (the gated ones need `hf auth login` + accepted terms)
```

By hand, from the repo root (versions in `env-lock/*.txt`):

```
uv venv --python 3.12 envs/<env>
VIRTUAL_ENV=envs/<env> uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.0 torchvision==0.24.0
VIRTUAL_ENV=envs/<env> uv pip install -r env-lock/<env>.txt
scripts/fetch_vendor.sh TripoSR stable-fast-3d stable-point-aware-3d generative-models TripoSG TRELLIS FlexiCubes dinov2 Hunyuan3D-2 Step1X-3D UniRig
```

`stab3d` needs the CPU builds of `texture_baker` / `uv_unwrapper` and the git packages `clip` / `alpha_clip` (steps in
`install/stab3d.yaml`; header of `env-lock/stab3d.txt`). Weights that are not on the Hugging Face cache path are listed under
`weights:` in the registry with a `download_cmd`: SPAR3D needs `~/.cache/clip/ViT-L-14-336px.pt`, TRELLIS needs
`~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`.

**Gated repos** (accept the licence on the model page, then `hf auth login`): `stabilityai/stable-fast-3d`,
`stabilityai/stable-point-aware-3d`, `stabilityai/sv3d`. All others are open. The Models page shows `missing_weights` with the fix
command until the files are present.

## Licence notes

- SF3D, SPAR3D, SV3D: Stability AI Community License. Free for research and for commercial use below USD 1M annual revenue, above
  that an enterprise licence is needed. Marked `verify: true` in the registry; read the licence at the pinned commit.
- Hunyuan3D-2mini: Tencent Hunyuan community licence, **not valid in the EU, UK and South Korea**; also `verify: true`.
- TripoSR, TripoSG, TRELLIS, UniRig: MIT. Step1X-3D, Zero123++: Apache-2.0 (Zero123++ weights flagged `verify`).
- The licence of generated assets follows the model's licence; the licence name and the commercial flag are stored on every asset.

## Disk

New envs `envs/stab3d`, `envs/shape3d`, `envs/trellis`, `envs/rig`: about 7 GB on disk after de-duplication (uv hardlinks the shared torch
wheels; `du` of one env alone reports 5 GB for `stab3d`). Weights in `~/.cache/huggingface` about 67 GB (SV3D 18, TripoSG 7.5, Hunyuan3D-2mini 7.9,
SPAR3D 6.9, Step1X-3D 6.8, UniRig 5.5, zero123plus 5.2, SF3D 3.8, TRELLIS 3.1, TripoSR, DINOv2-large 1.2, OPT-350m 0.6, OpenCLIP B/32 0.6) plus
2.2 GB under `~/.cache/clip` and `~/.cache/torch/hub`. Vendor clones about 1 GB.
