# 3D: text to motion on UniRig-rigged characters (design)

Status: **HY-Motion 1.0-Lite is implemented and measured (section 0 below); Kimodo is still design only.** Every time / memory
number marked *(unverified)* comes from upstream READMEs or model cards, not from this machine (aarch64, GB10 / sm_121, 121 GB unified memory,
CUDA 13, torch 2.9.0+cu130, no nvcc, no Blender / bpy). Research date: 2026-11-07.

Goal: the user types "walk forward and wave" in the 3D studio, picks a character that the studio generated and rigged
with UniRig (`rig` task, `workers/rig_worker.py`), and gets the same GLB back with an embedded animation clip that
`<model-viewer>` plays, plus BVH / npz exports.

## 0. Implemented: HY-Motion 1.0-Lite (measured on the DGX Spark)

Files: `workers/motion_common.py` (pure numpy helpers), `workers/motion_hy_worker.py`, `models.d/motion.yaml` (task `motion`,
backend `motion_hy`, model `hy-motion-lite`), `install/motion_hy.yaml`, `env-lock/motion_hy.txt`, `rig_anim.append_clip`,
the Motion tab in `app/web/studios/3d.js`, `tests/test_motion.py` (43 tests, no GPU, no weights).

Environment: python 3.12, torch 2.9.0+cu130 (warns that sm_121 is above the tested 12.0, works), transformers 4.57.1 (pinned by
the recipe check; upstream asks for a different pin), numpy 2.5.4, scipy 1.18.1, torchdiffeq. Vendor repo
`vendor/HY-Motion-1.0` at 4e426f5 plus `git lfs pull` (body-model `.bin` and `stats/*.npy`). Weights: `tencent/HY-Motion-1.0`
(`HY-Motion-1.0-Lite/` only), `Qwen/Qwen3-8B`, `openai/clip-vit-large-patch14`. `Tencent/Text2MotionPrompter` (prompt rewriter) is not used.

Pipeline: the upstream `MotionFlowMatching` is built with `load_object` from `config.yml` (cwd is switched to the vendor dir while
loading because the wooden-mesh bins are read relative to it), flow matching with Euler steps (`steps`, default 50) and
classifier-free guidance (`guidance`, default 5). Output: rot6d `[L,52,6]`, root translation, keypoints at 30 fps, up to 360 frames.

Measured (prompt "a person walks forward and waves", then "a person walks in place, then turns around", 4 to 5 s clips):

| | value |
|---|---|
| network build | 1.9 s |
| checkpoint + text encoders from the HF cache | 3.1 s |
| move to the GPU (`pipe.to(cuda)`, dominated by the 16 GB Qwen3-8B) | about 110 s |
| total first load | 110 to 130 s |
| generate, 4 s of motion, 50 steps | 4.1 s |
| retarget + GLB write | 0.04 s |
| GPU resident after load | 17.5 GB |
| GPU peak during a run | 19.0 to 21.4 GB |
| system memory delta (unified) | 22 to 24 GB, registered `memory_gb: 24` |

The slow GPU move is not disk (the safetensors read at 24 GB/s from the page cache); it is the per-parameter host to device copy
on the unified-memory device. Possible later optimisation: build the text encoder directly on `cuda` (`device_map`).

Retarget on the real UniRig 22-joint rig: limb direction error 0 degrees with foot lock off, bone lengths preserved exactly,
root travel scaled by the leg-length ratio, feet planted by contact detection plus two-bone IK (foot world rotation kept).
Quirks and limits:
- Humanoid detection is geometric (names only cross-check). It refuses quadrupeds, busts, snakes, one-armed rigs, fewer than 10
  joints and lying skeletons with `NotHumanoidError` (reported as an input error with a hint) before any GPU work.
- HY-Motion has no fingers, no face and a fixed SMPL-H 22-joint body: hands, jaw and extra bones (tails, hair, cloth) stay rigid
  with their parent.
- Rigs whose rest pose is bent (A-pose, bent elbows) are aligned per bone with a bend-axis basis; very unusual rests may still twist
  a forearm by a few degrees. `arm_offset` (0 to 30 degrees) lifts the arms away from the body if the mesh clips through the torso.
- Foot lock is an approximation: contacts are "toe low and slow", IK only touches hip and knee, and fast turns or jumps can show a
  slight slide. Root translation is written on the skeleton root joint, so rigs rooted at the chest work.
- The clip is written as the single first animation `motion` (30 fps keys, LINEAR); older clips are dropped unless `keep_clips`.
- Not checked in a browser: the 390 px and 1440 px layouts of the Motion tab and the in-viewer playback were only syntax checked.
- Licence: Tencent HY-Motion 1.0 Community License. Not licensed in the EU, UK and South Korea; more than 1 million monthly
  active users needs a separate licence. The registry entry has `commercial: true, verify: true`.

## Decision in one table

| | choice | why |
|---|---|---|
| primary model | **Kimodo SOMA-RP v1.2** (`nvidia/Kimodo-SOMA-RP-v1.2`, NVIDIA, 2026-10-15) | the only strong text-to-motion model whose training data is *not* AMASS / HumanML3D and whose weights are commercially usable; 10 s @ 30 fps per prompt, multi-prompt chaining, outputs local rotations + root + foot contacts; pure PyTorch apart from one optional x86 C++ post-processor |
| fallback | **HY-Motion 1.0-Lite / 1.0** (`tencent/HY-Motion-1.0`, Tencent, 2025-12-30), phase 2, own env | also not AMASS-licensed, commercial under the Tencent community terms (not EU / UK / South Korea), 12 s @ 30 fps, SMPL-H-style 52 joints without SMPL registration; conflicting pins force a separate venv |
| retargeting | **geometric humanoid mapper on the UniRig skeleton + global-rotation transfer with per-bone rest alignment** (option a), name-table shortcut when bones are named | UniRig bones are `bone_0 …` with identity rest rotations; a template re-rig is not available in the released checkpoint, and SMPL fitting would bring the SMPL licence and a lot of code |
| new task / backend | task `motion` (3D studio), backend `motion` (env `envs/motion`), worker `workers/motion_worker.py`, pure helpers `workers/motion_common.py` | follows the `rig` layout; reuses `workers/rig_anim.py` for the GLB clip writer |
| not added | MDM, T2M-GPT, MoMask, MotionGPT, MotionLCM (all HumanML3D / AMASS) | AMASS licence forbids commercial use and training for commercial use, so every one of them would be `commercial: false`; quality is also below Kimodo / HY-Motion |

## 1. Model comparison

"Data" is the training set; for HumanML3D models the AMASS licence decides the weight licence, not the code licence
(see Licence notes). Sizes are on-disk weights from the HF / Drive listings.

| model | released | code licence | weights / data licence | commercial (registry) | output | max length | weights | text encoder | speed / VRAM (upstream) | aarch64 / torch 2.9 feasibility | HF | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kimodo SOMA-RP v1.2** | 2026-10-15 | Apache-2.0 | NVIDIA Open Model License; data Bones Rigplay (700 h studio mocap, not AMASS) | `true, verify: true` (Llama 3 terms for the encoder) | SOMA skeleton (30 joints in the model, 77 in the npz / BVH export): local + global rot mats, posed joints, root path, foot contacts [T,4]; +Y up, +Z forward, T-pose rest | 10 s / 300 frames @ 30 fps per prompt; multi-prompt chaining with 5 transition frames | 1.13 GB (282 M params) | LLM2Vec on Meta-Llama-3-8B-Instruct (16.1 GB, **gated**) + 2 adapters (0.35 GB) | ~17 GB VRAM with the encoder on GPU, < 3 GB with `TEXT_ENCODER_DEVICE=cpu`; DDIM 100 steps *(unverified)* | pure PyTorch; transformers 5.1.0 + peft; **MotionCorrection C++ extension is x86-only** (SSE / AVX), skip or port with sse2neon; scenepic (viz) has no aarch64 wheel, not needed | yes, open | **primary** |
| Kimodo SOMA-SEED v1.1 | 2026-10 | Apache-2.0 | NVIDIA Open Model License; data BONES-SEED (open subset) | `true, verify: true` | same as above | same | ~1.1 GB | same Llama encoder | same | same | yes, open | optional second entry, same backend |
| Kimodo SMPLX-RP v1 | 2026-10 | Apache-2.0 | NVIDIA R&D licence (non-commercial), needs `SMPLX_NEUTRAL.npz` (registration) | `false` | SMPL-X 22 joints | 10 s | ~1.1 GB | same | same | same | **gated ("auto")** | not added |
| ARDY (nv-tlabs/ardy) | 2026-10-15 | Apache-2.0 | NVIDIA Open Model License (SOMA-RP-30FPS checkpoints) | `true, verify: true` | SOMA, autoregressive, real time | streaming | ~1 GB | same Llama encoder | real time *(unverified)* | same family, same deps | yes | later (interactive / streaming use) |
| **HY-Motion 1.0** | 2025-12-30 | Tencent HY-Motion 1.0 licence ("Other") | same Tencent licence; Tencent claims no rights in outputs; not EU / UK / South Korea; > 1 M MAU needs a licence | `true, verify: true` | rot6d [L,52,6] SMPL-H topology (22 body + 30 fingers) + transl + keypoints; 30 fps | 12 s (train_frames 360) | 4.17 GB (1.0 B params) | Qwen3-8B (16.4 GB, open) + CLIP ViT-L/14 (1.71 GB) | min 26 GB VRAM; flow matching, Euler 50 steps *(unverified)* | pure PyTorch, SDPA attention ("torch" mode), no flash-attn; pins torch 2.5.1 / transformers 4.53.3 / diffusers 0.26.3 (only `DDIMScheduler` used); `fbxsdkpy` optional (FBX export only, skip); body-model bins in **git LFS** | yes, open | **fallback** (phase 2, env `motion_hy`) |
| HY-Motion 1.0-Lite | 2025-12-30 | same | same | same | same | 12 s | 1.84 GB (0.46 B) | same | min 24 GB VRAM *(unverified)* | same | yes (subfolder) | fallback, first one to test |
| MoMask | 2024 (CVPR) | MIT | HumanML3D / KIT → AMASS licence | `false` | 263-d HumanML3D features → 22 joint positions; BVH via IK | 196 frames @ 20 fps (9.8 s) | ~1 GB (Google Drive) | CLIP ViT-B/32 (openai `clip` package) | fast, runs on CPU | pure PyTorch; Drive downloads | Space only (MeYourHint/MoMask) | not added (licence) |
| MDM | 2023 (ICLR) | MIT | AMASS licence | `false` | 263-d → joints (rotations unreliable); SMPL mesh needs SMPL files | 9.8 s @ 20 fps | ~0.3 GB (Drive) | CLIP ViT-B/32 | 1000 / 50 diffusion steps, slow | pure PyTorch | no | not added |
| T2M-GPT | 2023 (CVPR) | Apache-2.0 | AMASS licence | `false` | 263-d → joints | 9.8 s | ~0.5 GB (Drive) | CLIP ViT-B/32 | fast | pure PyTorch | no | not added |
| MotionGPT | 2023 (NeurIPS) | MIT | AMASS licence | `false` | 263-d → joints | 9.8 s | ~1 GB | flan-t5-base | moderate | pure PyTorch | `OpenMotionLab/MotionGPT-base` | not added |
| MotionLCM v2 | 2024 (ECCV) | **non-commercial research licence (code)** | AMASS licence | `false` | 263-d → joints | 9.8 s | small (Drive) | sentence-T5 | ~30 ms per motion | pure PyTorch | no | not added |

Notes:
- HumanML3D-era models share three problems besides the licence: 20 fps, 263-d features whose rotations are not usable
  for retargeting (positions only, so IK would be needed), and visualisation through SMPL meshes that need registration.
  Kimodo and HY-Motion output real joint rotations, which makes retargeting a rotation transfer instead of an IK solve.
- No other 2026 release found with open weights, a commercial-friendly licence and non-AMASS data (searched HF for
  text-to-motion / motion-generation, Oct-Nov 2026). Recheck before implementing.

### Why Kimodo over HY-Motion as primary

| | Kimodo SOMA-RP v1.2 | HY-Motion 1.0 (Lite) |
|---|---|---|
| licence territory | worldwide (NVIDIA OML) | excludes EU, UK, South Korea |
| weights + encoders | 1.13 + 16.5 GB | 4.17 (1.84) + 18.1 GB |
| VRAM | ~17 GB (< 3 GB with encoder on CPU) | 24-26 GB minimum |
| deps vs studio torch 2.9 | transformers 5.1.0, compatible with torch 2.9 | pins torch 2.5.1 / transformers 4.53, needs relaxing |
| output | local rot mats + foot contacts + T-pose rest | rot6d SMPL-H, no contacts |
| constraints | root path, keyframes, multi-prompt | single prompt |
| blocker | Llama 3 8B is gated (manual approval) | LFS assets (git-lfs is installed) |

The gated Llama repo is the one real cost of Kimodo; HY-Motion is the fallback precisely because it has no gated
download.

## 2. Retargeting to UniRig skeletons

What UniRig gives us (from `_write_rigged_glb` in `workers/rig_worker.py`): joint nodes `bone_0 …` with translations only
(local = joint − parent joint, **every rest rotation is identity**), inverse bind matrices = translate(−joint_world), +Y up,
front +Z, the character's left on +X, a predicted hierarchy of any shape, and `rigged_plain.glb` attached without clips.
Example (UniRig's MIT `examples/tira.glb`, 52 joints, T-pose): 0 pelvis root at y = 0.914; spine 1→2→3 (chest, arms branch
here)→4→5 (head leaf); left arm 6 (collar)→7→8→9 (hand), fingers 10-24; right arm 25-43; legs from the root 44→45→46→47
(toe, z = +0.13) and 48→49→50→51. A 16-joint bust (no legs, odd branches) must be refused.

### Options

| option | how | handles stylised / A-T pose | missing limbs / non-humanoid | code | verdict |
|---|---|---|---|---|---|
| (a) heuristic mapping | detect pelvis / spine / legs / arms / head geometrically on the UniRig skeleton, then transfer global rotations with per-bone rest alignment | yes: alignment uses the target's own rest directions, scale from leg length | detector refuses with a reason | ~600 lines numpy + ~400 lines tests | **chosen** |
| (b) template re-rig | UniRig with `cls = vroid` (names bones `J_Bip_C_Hips …` via `Order.make_names`) or `mixamo` | yes | template forces a humanoid even on a dog | config change | released checkpoint is trained on Articulation-XL2.0 only; the tokenizer comments mixamo as "currently untrained" and the VRoid / Rig-XL checkpoint is still marked pending in the README → **experiment only** (`assign_cls: vroid` in a copy of `ar_inference_articulationxl.yaml`); if it works, the name-table shortcut of (a) picks the names up for free |
| (b') other auto-rigger with named joints | Make-It-Animatable (MIT code, `jasongzy/Make-It-Animatable`, Mixamo template) | yes | humanoid only | new backend | needs bpy + FBX2glTF (x86), data Mixamo-derived; revisit only if (a) fails on many generated humanoids |
| (c) fit an SMPL-like template | optimise SMPL shape / pose to the mesh, transfer skin | good on realistic bodies, poor on chibi | humanoid only | large | SMPL licence (registration, non-commercial), lots of code → **rejected** |
| (d) position IK | drive target joints toward scaled source joint positions with FABRIK / CCD | yes | same detector | medium | loses twist; kept only as a fallback idea for extreme proportions |

### Humanoid detection (pure numpy, `motion_common.detect_humanoid(joints, parents, front=+Z)`)

Inputs: rest world positions `X[J,3]` (from the GLB: compose node translations; UniRig rest rotations are identity, but
use the general `Rig.world(i)` from `rig_anim` so named / rotated rigs also work), `parents[J]` (−1 for roots), optional
names. Let `H = ymax − ymin` of the joints (or of the mesh bbox when available), `lat(x) = x · L` with `L = +X`
(left = up × front).

1. **Name shortcut.** If the names match a table (VRoid `J_Bip_*`, Mixamo `mixamorig:*`, SMPL / SMPL-H, SOMA, Unreal
   `pelvis/thigh_l/...`, common `Hips/LeftUpLeg/...`) for at least hips, spine, head, both legs and both arms, use it and
   skip to the checks in step 9.
2. `J ≥ 10`, else refuse ("too few joints").
3. **Feet.** Candidate leaves with `y < ymin + 0.15 H`. Group them by the top-level branch they belong to (the child of
   the lowest common ancestor). Need exactly two floor-reaching groups, one with `lat > 0` (left) and one `< 0` (right).
   More than two floor-reaching groups, or floor groups whose chains start from a joint that is forward of and far
   from the pelvis candidate (front legs) → refuse as "four-legged / not a biped".
4. **Pelvis** = lowest common ancestor of the two foot chains. **Leg chain** = pelvis-child → … → foot leaf; need ≥ 3
   joints below the pelvis (hip, knee, ankle). If the last bone is roughly horizontal (`|dy| < 0.5 |d|`) and points
   forward (`d · F > 0`), it is a toe and the joint before it is the ankle. Knee = the joint at ~50 % of chain arc length
   from the hip (closest joint); hip = first joint.
5. **Spine.** From the pelvis follow, among the non-leg children, the one whose subtree reaches the highest `y`, then
   repeat to the top. Accept if the top is `> 0.25 H` above the pelvis and the pelvis → top vector is within 35° of `+Y`.
   If the pelvis is not the skeleton root and the upward chain goes through the pelvis' parent (rig rooted at the
   chest), walk up through the parent instead; if no upright chain exists → refuse ("no upright spine").
6. **Arms.** For every spine joint, consider side branches (children not on the spine) whose subtree has lateral extent
   `> 0.15 H`. Need one on each side, mirror-symmetric: extents within 35 % of each other and tips within `0.15 H` of
   each other after mirroring `x → −x`. Missing one → refuse ("one arm missing"); none → refuse ("no arms").
7. **Arm chain.** From the branch joint follow the child with the largest lateral subtree extent until a joint with
   ≥ 3 children (hand with fingers) or a leaf. If the chain has ≥ 4 joints and its first bone is shorter than 0.35 × chain
   length, the first joint is the collar (clavicle). Shoulder = next joint, hand = last, elbow = joint closest to the
   middle of the shoulder → hand arc length.
8. **Chest** = the spine joint where the arms branch (lower one if they branch at different joints). Joints between chest
   and top are neck(s); top leaf = head (if the top joint has children that are face / hair leaves, head = the joint
   with the most children above the chest). Remaining spine joints between pelvis and chest are spine1..n.
9. **Checks.** Both feet below the pelvis, hands above the knees, left side has `lat > 0`, bone count per chain ≥ 2.
   Any failure → `InputError` with the reason and a hint: *"Text to motion needs a two-legged, two-armed character
   standing upright (T- or A-pose). Re-rig with another seed or use a humanoid mesh."*
10. **Facing.** Default `+Z` (studio convention). `facing: auto` takes the mean of the toe directions (or the knee-bend
    direction when there are no toes) projected on the ground plane and snaps it to ±X / ±Z; the param can force it.

Output: a `HumanoidMap` dict `{role: joint_index}` with roles `hips, spine[], chest, neck[], head, {l,r}_collar?,
{l,r}_upperarm, {l,r}_forearm, {l,r}_hand, {l,r}_thigh, {l,r}_shin, {l,r}_foot, {l,r}_toe?`, plus `facing`, `leg_length`,
warnings. Everything else (fingers, tails, hair, accessories) is "unmapped".

Source role tables (fixed in code):

| role | Kimodo SOMA (npz / BVH) | HY-Motion (SMPL-H order) |
|---|---|---|
| hips | Hips | Pelvis (0) |
| spine (in order) | Spine1, Spine2 | Spine1 (3), Spine2 (6) |
| chest | Chest | Spine3 (9) |
| neck | Neck1, Neck2 | Neck (12) |
| head | Head | Head (15) |
| collar L / R | LeftShoulder / RightShoulder | L_Collar (13) / R_Collar (14) |
| upper arm | LeftArm / RightArm | L_Shoulder (16) / R_Shoulder (17) |
| forearm | LeftForeArm / RightForeArm | L_Elbow (18) / R_Elbow (19) |
| hand | LeftHand / RightHand | L_Wrist (20) / R_Wrist (21) |
| thigh | LeftLeg / RightLeg | L_Hip (1) / R_Hip (2) |
| shin | LeftShin / RightShin | L_Knee (4) / R_Knee (5) |
| foot | LeftFoot / RightFoot | L_Ankle (7) / R_Ankle (8) |
| toe | LeftToeBase / RightToeBase | L_Foot (10) / R_Foot (11) |

(Kimodo's `LeftLeg` is the thigh and `LeftShin` the knee bone; read the names from `model.skeleton.bone_order_names`
at run time and assert the table rather than trusting indices.)

### Rotation transfer (the maths)

Notation: rotations are 3×3 matrices, `W` = world (global) rotation, `L` = local, column vectors, `p(j)` = parent.
Source: per frame `t` global rotations `G_s(t)` and world positions `x_s(t)` (Kimodo: `global_rot_mats`, `posed_joints`;
HY-Motion: FK of rot6d on its template). Target: rest world positions `X_j`, rest world rotations `R0_j` (identity for UniRig).

1. **Facing.** If the target faces `f ≠ +Z`, `Q_face` = rotation about `+Y` taking `+Z` to `f`; apply
   `G'(t) = Q_face G(t) Q_faceᵀ` and `x'(t) = Q_face x(t)` to the source (both models face `+Z` already).
2. **Source rest directions, convention-free.** For every mapped source bone `s` with chain child `c`:
   `d_S(s) = normalize( mean_t  G_s(t)ᵀ (x_c(t) − x_s(t)) )` — the bone direction expressed in the bone's own frame,
   which is constant over time for a rigid bone. Spread (max angle of the per-frame vectors to the mean) must be < 2°;
   otherwise the role table is wrong → error, not a silent bad animation. For the hips use the lateral axis
   `a_S = normalize(mean_t G_hipsᵀ (x_lthigh − x_rthigh))` and the up axis `u_S` from hips → spine1.
   (This avoids depending on whether SOMA's rest offsets are identity or the T-pose has baked offsets.)
3. **Target rest directions.** For a mapped target joint `j` with chain child `k`:
   `d_T(j) = R0_jᵀ normalize(X_k − X_j)` (with `R0 = I`, just the world bone direction). Hips: `a_T = normalize(X_lthigh − X_rthigh)`, `u_T = normalize(X_spine1 − X_hips)`.
4. **Alignment per bone.** `Q_j = R_arc(d_T(j) → d_S(σ(j)))`, the minimal rotation taking `d_T` to `d_S` (Rodrigues;
   for antiparallel vectors rotate π about any axis perpendicular to `d_T`, preferring the bone's bend axis).
   For the hips use the two-axis alignment: `Q_hips = B_S B_Tᵀ` where `B = [a, u', a × u']`, `u'` = `u` orthogonalised
   against `a` (exact for both axes, keeps twist). End effectors without a chain child (head, hand without fingers,
   foot without toe) reuse the parent's `Q` (their rest is assumed to continue the parent bone). If the arc angle is
   > 100° (badly predicted bone), warn and use the parent's `Q`.
   Twist: `R_arc` is the minimal-twist choice; for upper arm / forearm / thigh / shin additionally align the bend axis
   (target: normal of the rest chain plane, defaulting to `±X` for legs and `±Y`-ish for arms; source: estimated the same
   way as in step 2 from the elbow / knee bend over the clip) with a second rotation about `d_T`. This keeps palms and
   knees from flipping. Implement it as a 2-axis basis alignment like the hips.
5. **Driver assignment.** `σ(j)` maps target to source roles. When the target chain has more joints than the source
   (e.g. UniRig spine of 4 vs SOMA's 3), assign drivers by normalised arc length along the chain (target joint at
   fraction `u` gets the source joint whose fraction is closest below `u`); fewer target joints simply skip source
   joints — global rotations absorb the skipped ones.
6. **Target world rotations.** `W_j(t) = G'_σ(j)(t) · Q_j · R0_j` — i.e. the target bone in its rest frame is first rotated
   onto the source rest direction, then moved like the source bone.
   Check: at a frame where the source is in its rest pose (`G = G_rest`), `W_j` maps `d_T` onto the source rest bone
   direction in world, so an A-pose target driven by a T-pose source becomes T-posed, as it should.
7. **Unmapped joints** keep their rest local (`L = R0_p(j)ᵀ R0_j`, identity for UniRig) and follow the parent rigidly.
8. **Locals for glTF.** `L_j(t) = W_p(j)(t)ᵀ W_j(t)` (general form `(W_p R0_p … )` already folded in step 6). Root:
   `L_root = W_root`. Convert to quaternions (`rig_anim.mat_to_q`) and run `_canon` for hemisphere continuity.
9. **Root translation.** `k = leg_T / leg_S`, leg = |hip→knee| + |knee→ankle| averaged over both sides (source from the
   rest estimate). With source hips position `r(t)`:
   `P(t) = X_hips_rest + k · (r(t) − [r_x(0), h_S, r_z(0)])` where `h_S` = source hips height in its rest pose, so frame 0
   starts on the spot and vertical motion (jumps, crouches) is scaled. `root_motion: in_place` zeroes the x / z terms.
   If the target hips joint is not the node that carries the root, write translation on the hips node itself (its
   parent stays at rest). Ground: after FK, compute the median of the lowest foot / toe joint `y` over frames that
   Kimodo marks as contact (or the lowest 10 % of frames for HY-Motion) and shift `P_y` so it equals the rest foot height.
10. **Foot lock (optional, default on).** Per foot, runs of contact frames (Kimodo `foot_contacts`, heel or toe;
    HY-Motion: foot speed < 2 cm/frame and height < 3 cm above the ground) define a locked world position (mean of the run,
    blended in / out over 3 frames). Analytic two-bone IK per frame: thigh length `a`, shin length `b`, target distance
    `c = clamp(|ankle* − hip|, |a − b| + ε, a + b − ε)`; knee angle by the law of cosines; the knee pole is the knee
    position from the unlocked FK, so the knee direction is kept. `R1` rotates the thigh from its FK direction to the IK
    direction, `R2` the shin; the foot keeps its pre-IK world rotation (`L_foot = W_shinᵀ W_foot_fk`). Bone lengths are
    preserved exactly.
11. **Arm offset (advanced, default 0).** `arm_offset_deg` adds an abduction about the forward axis to both upper arms,
    for wide characters whose arms intersect the body.
12. **Sampling.** Source fps 30. Output `fps` 30 by default; 24 / 60 resample by slerp per joint (and lerp for the root).

Stylised proportions: alignment uses the target's own rest bone directions and the root scale uses leg length only, so
long arms or a big head do not distort the motion; contacts are kept by the foot lock. Extreme chibi (legs < 25 % of
height) mostly works but hands may reach the head differently — expected and documented, not corrected.

### Write-out

Start from the attachment `rigged_plain.glb` of a rig result if present, else from the input GLB with its
`animations` removed. Use `workers/rig_anim.py` (`read_glb`, `Rig`, `_Blob`, `write_glb`, `_canon`) and add a small
additive function there (or in `motion_common`), e.g.
`add_tracks(js, binary, name, tracks=[(node, path, times, values)])` with `path ∈ {rotation, translation}`, so the root
can carry translation (today's clips are rotation-only). One clip named `motion` (or the first 32 characters of the
prompt, slugged), LINEAR interpolation, float32, written as the **first** animation so `<model-viewer>` plays it by
default; keep `flex` / `wiggle` only if the user ticks "keep test clips" (default off). Only mapped joints and joints whose
local changes get channels.

Code size estimate: `motion_common.py` ~600 lines (detection 250, transfer 200, foot lock 80, BVH writer 70),
`motion_worker.py` ~250 lines, `rig_anim.add_tracks` ~40 lines, tests ~400 lines.

## 3. Studio integration

### Registry (`models.d/motion.yaml`, sketch)

```yaml
tasks:
  motion: {label: Text to motion, modality: mesh, studio: 3d}

backends:
  motion:
    python: envs/motion/bin/python
    script: workers/motion_worker.py
    device: gpu
    max_loaded: 1
    lock: motion
    env:
      PYTHONPATH: "{studio}/vendor/kimodo"      # or a sys.path insert in the worker
      TEXT_ENCODER_MODE: local                   # default "auto" first tries an HTTP API on 127.0.0.1:9550
      LOCAL_CACHE: "true"                        # Kimodo: snapshot_download(local_files_only=True)
      HF_HUB_OFFLINE: "1"

models:
  kimodo-soma-rp:
    label: Kimodo SOMA-RP v1.2 (NVIDIA)
    task: motion
    backend: motion
    tags: [gated]                                # because of the Llama 3 encoder
    weights:
      - {hf: nvidia/Kimodo-SOMA-RP-v1.2}
      - {hf: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp}
      - {hf: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised}
      - {hf: meta-llama/Meta-Llama-3-8B-Instruct, files: ["*.json", "model-*.safetensors"]}
    license:
      name: "NVIDIA Open Model License (weights) + Apache-2.0 (code); text encoder: Llama 3 Community License"
      commercial: true
      verify: true
      note: >-
        Distributions must carry "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"; rights end if
        guardrails are bypassed. Llama 3: "Built with Meta Llama 3" attribution, 700 M MAU cap, Meta AUP. Trained on
        Bones Rigplay mocap, not AMASS.
    memory_gb: 24
    est: {load_s: 40, run_s: 20}                 # unverified on GB10
    input_slots:
      - {role: mesh, label: Rigged character, types: [mesh], required: true}
    params:
      prompt:       {type: text, label: Motion, default: "walk forward and wave", multiline: true}   # one line = one segment
      seconds:      {type: float, min: 1, max: 10, step: 0.5, default: 5}                          # per segment
      seed:         {type: seed}
      root_motion:  {type: choice, choices: [root, in_place], default: root}
      foot_lock:    {type: bool, default: true}
      fps:          {type: choice, choices: [30, 24, 60], default: 30}
      steps:        {type: int, min: 50, max: 200, default: 100, advanced: true}
      guidance:     {type: float, min: 1, max: 5, step: 0.5, default: 2.0, advanced: true}
      facing:       {type: choice, choices: [auto, "+Z", "-Z", "+X", "-X"], default: auto, advanced: true}
      arm_offset_deg: {type: float, min: 0, max: 30, default: 0, advanced: true}

  # phase 2
  hy-motion-1-lite:
    task: motion
    backend: motion_hy
    weights:
      - {hf: tencent/HY-Motion-1.0, files: ["HY-Motion-1.0-Lite/*"]}
      - {hf: Qwen/Qwen3-8B}
      - {hf: openai/clip-vit-large-patch14, files: ["*.json", "*.txt", "model.safetensors"]}
    license:
      name: Tencent HY-Motion 1.0 Community License
      commercial: true
      verify: true
      note: Not licensed in the EU, UK or South Korea; > 1 M MAU needs a licence from Tencent; outputs may not be used to improve other AI models.
    memory_gb: 32
```

(Param keys follow `docs/CONTRACTS.md` §2; adjust names like `multiline` / `advanced` to whatever the contract uses.)

Prompts: prefix "A person " when the line does not start with "a person" / "the person" (Kimodo is trained on that
phrasing). Several non-empty lines → Kimodo `multi_prompt=True`, one segment per line, `seconds` each,
`num_transition_frames=5`; total capped at 30 s.

### Worker (`workers/motion_worker.py`)

- `probe`: import torch, kimodo; check the 4 HF repos are in the cache (no network).
- `load`: `model, name = load_model("Kimodo-SOMA-RP-v1.2", device="cuda", default_family="Kimodo", return_resolved_name=True)`
  with `CHECKPOINT_DIR` / `TEXT_ENCODERS_DIR` / `HUGGINGFACE_CACHE_DIR` pointing at the HF cache; text encoder on GPU
  (memory is plentiful). Try `import motion_correction` once; if missing, set `post_processing=False` and log it.
- `run`: read the GLB, `detect_humanoid` **first** (refuse in < 1 s, before any GPU work), generate
  `model(texts, num_frames, num_denoising_steps=steps, num_samples=1, multi_prompt=…, post_processing=…, return_numpy=True, cfg_weight=[guidance, guidance])`
  with `torch.manual_seed(seed)`, retarget, write outputs. Progress: encode 10 %, denoise 10-80 % (callback per step if
  available, else coarse), retarget 90 %.
- Errors: non-humanoid / no skin → `InputError` (kind `input`), reuse `rig_anim.NoSkinError` message for meshes without a
  skin ("Rig the mesh first (Auto-rig tab)").

### Outputs

| file | type | content |
|---|---|---|
| `animated.glb` (primary) | mesh | input mesh + skin + clip `motion` first; meta `rigged: true`, `animations: ["motion", …]`, `motion: {model, prompt, seconds, fps, frames, root_motion}` |
| `motion.npz` | attach | raw source: posed_joints, local / global rot mats, root, foot_contacts, fps, bone names, parents |
| `motion_source.bvh` | attach | source skeleton BVH (Kimodo `--bvh` writer, or our own 70-line writer from local rotations) |
| `motion_target.bvh` | attach | the retargeted motion on the UniRig skeleton (for Blender / other DCC users) |
| `retarget.json` | attach | role map, scale `k`, facing, warnings, per-bone alignment angles |

### UI (3D studio)

A "Motion" tab with `receives: ['motion']` and `inputs: [{key: 'mesh', role: 'mesh'}]`, plus an "Animate with text" action
on rig results (`sendTo` action `motion`). The viewer already lists clips via `clipsOf(a)` (worker / info `animations`).

### Memory, time, disk

| item | estimate | basis |
|---|---|---|
| Kimodo + Llama encoder in GPU memory | ~17 GB (bf16 Llama 16 GB + 0.6 GB model + activations) | Kimodo README *(unverified)* |
| registry `memory_gb` | 24 (headroom for 30 s multi-prompt) | |
| load | ~40 s (16 GB from page cache) | *(unverified)* |
| run, 5 s clip, 100 DDIM steps | ~10-20 s | *(unverified)* |
| retarget + GLB write | < 1 s for 52 joints × 300 frames | numpy, measured later |
| weights on disk | 17.6 GB (1.13 + 0.35 + 16.1) | HF listings |
| venv | ~5-6 GB | torch 2.9 + transformers 5.1 |
| HY-Motion phase 2 | 1.84 / 4.17 + 16.4 + 1.71 GB, `memory_gb` 32 | model card |

### Offline

Everything from the HF cache; `HF_HUB_OFFLINE=1`, `LOCAL_CACHE=true`, `TEXT_ENCODER_MODE=local`. Kimodo's adapter
`base_model_name_or_path` is the hub id `meta-llama/Meta-Llama-3-8B-Instruct`, which resolves through the cache when
offline. gradio / boto3 imports in Kimodo are demo / S3 helpers; the worker imports only `kimodo.model` /
`kimodo.load_model` paths (verify none of them import gradio at module level; stub if they do).

### Installer recipe (`install/motion.yaml`, sketch)

```yaml
backend: motion
env: motion
python: "3.12"
torch:
  - torch==2.9.0
lock: motion                      # uv pip freeze of: transformers==5.1.0, huggingface-hub>=1.3, peft>=0.18, hydra-core,
lock_flags: [--no-deps]           #   omegaconf, einops, numpy, scipy, bvhio, trimesh, safetensors, accelerate
vendor: [kimodo, sse2neon]        # vendor.lock: nv-tlabs/kimodo @ e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954
                                  #              DLTcollab/sse2neon @ 2272251cc3ad9cba8b8c47c4065811cfb396b802 (MIT)
steps:
  # NOT `pip install -e vendor/kimodo`: its setup.py builds the MotionCorrection C++ extension (x86 SSE/AVX).
  # put vendor/kimodo on sys.path with a .pth file (idempotent)
  - "echo \"$REPO/vendor/kimodo\" > \"$($PY -c 'import site; print(site.getsitepackages()[0])')/kimodo.pth\""
steps_linux-aarch64:
  # optional: build MotionCorrection from a copy with <immintrin.h> -> sse2neon.h and -msse4.1 -mavx dropped.
  # Failure only disables post-processing (the worker falls back to its own foot lock).
  # (new helper script, to be written with the recipe)
  - "bash scripts/build_motion_correction.sh || echo 'MotionCorrection not built: post-processing off'"
steps_linux-x86_64:
  - "uv pip install --no-deps --no-build-isolation ./vendor/kimodo/MotionCorrection || true"
check: "import torch, transformers, peft, hydra, bvhio, kimodo"
manual:
  - "Kimodo's text encoder is meta-llama/Meta-Llama-3-8B-Instruct (gated). Accept the licence at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct, run `hf auth login`, then `./install.sh model kimodo-soma-rp`."
tested: []
size_gb: 6
shares_env_with: null
```

The build script copies `vendor/kimodo/MotionCorrection` into `$VIRTUAL_ENV/src/`, so the vendor tree stays clean, and
sse2neon's header goes on the include path. Kimodo uses only SSE intrinsics (`_mm_*`, no `_mm256_*`), all covered by
sse2neon. The exact path of the C++ sources and the CMake target name must be checked when the recipe is written.

Phase 2 `install/motion_hy.yaml`: separate env, `vendor: [HY-Motion-1.0]` (default branch `master`, pin
4e426f5a1021cbcf7f375458c37b840ee7225229) **plus `git lfs pull --include scripts/gradio/static/assets/dump_wooden/*`**
in `steps` (the WoodenMesh body-model bins are LFS pointers; `scripts/fetch_vendor.sh` does a `blob:none` clone and
does not fetch LFS). Relax the pins to torch 2.9.0 / a transformers 4.5x that loads Qwen3, keep diffusers only for
`DDIMScheduler` (or stub it), skip `fbxsdkpy` and `bitsandbytes` (verify bitsandbytes is not imported unconditionally).

## 4. Licence notes

- **AMASS** (https://amass.is.tue.mpg.de/license.html): licensed "for the sole purpose of performing non-commercial
  scientific research"; "Any other use, in particular any use for commercial purposes, is prohibited", including "use in a
  commercial service, or production of other artefacts for commercial purposes", and it "prohibits the use of the Dataset
  to train methods/algorithms/neural networks/etc. for commercial use of any kind". HumanML3D (MIT code) is a set of
  annotations on AMASS, so MDM, T2M-GPT, MoMask, MotionGPT, MotionLCM and similar weights are `commercial: false`
  regardless of their MIT / Apache code.
- **NVIDIA Open Model License** (version of 2025-10-24; Kimodo SOMA-RP / SOMA-SEED, ARDY): "Models are commercially
  usable"; NVIDIA claims no ownership of outputs; distribution requires the notice "Licensed by NVIDIA Corporation under
  the NVIDIA Open Model License"; rights terminate if built-in guardrails are bypassed. Kimodo-SMPLX uses a separate NVIDIA
  R&D (non-commercial) licence and SMPL-X → not added.
- **Llama 3 Community License** (Kimodo text encoder): commercial use allowed; > 700 M MAU needs a Meta licence;
  "Built with Meta Llama 3" attribution when distributing; Meta Acceptable Use Policy applies. Because the studio
  only uses Llama as an embedding model, the encoder terms are a condition, not a blocker → `verify: true`.
- **Tencent HY-Motion 1.0 licence**: territory excludes the EU, UK and South Korea; > 1 M MAU needs a separate licence;
  outputs may not be used to improve other AI models; "Tencent claims no rights in Outputs". Same treatment as the
  Hunyuan3D-2mini entry (`commercial: true, verify: true` with a note).
- **Code**: Kimodo Apache-2.0, ARDY Apache-2.0, sse2neon MIT, UniRig MIT, MoMask MIT, MDM MIT, T2M-GPT Apache-2.0,
  MotionGPT MIT, HumanML3D MIT, **MotionLCM non-commercial**, HY-Motion Tencent licence (GitHub reports "Other").
- `THIRD_PARTY_NOTICES.md` gets Kimodo, sse2neon, the NVIDIA OML notice and the Llama attribution when this ships.

## 5. Downloads for testing (in fetch order, one repo at a time)

**Step 0, do first: request Llama 3 access** — approval is manual and can take hours to days.

| # | what | HF repo / files | size | gated | needed for |
|---|---|---|---|---|---|
| 0 | accept terms | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct ("Agree and access repository"), then `hf auth login` with a token of the same account (a token is already on the machine; check it belongs to the approving account) | – | **yes, manual** | Kimodo |
| 1 | Kimodo model | `nvidia/Kimodo-SOMA-RP-v1.2` (`model.safetensors`, `config.yaml`, README) | 1.13 GB | no | Kimodo |
| 2 | LLM2Vec adapter 1 | `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` (all files) | 0.18 GB | no (MIT) | Kimodo |
| 3 | LLM2Vec adapter 2 | `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` (all files) | 0.17 GB | no (MIT) | Kimodo |
| 4 | Llama 3 8B base | `meta-llama/Meta-Llama-3-8B-Instruct`: `config.json`, `generation_config.json`, `model-00001…00004-of-00004.safetensors`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`. **Exclude `original/*`** (another 16 GB of the same weights in Meta format) | 16.1 GB | **yes** (step 0) | Kimodo |
| 5 | fallback model | `tencent/HY-Motion-1.0`, subfolder `HY-Motion-1.0-Lite/` (`latest.ckpt`, `config.yml`); optionally `HY-Motion-1.0/` (4.17 GB) | 1.84 GB | no | HY-Motion |
| 6 | fallback LLM encoder | `Qwen/Qwen3-8B` (safetensors + tokenizer + configs) | 16.4 GB | no (Apache-2.0) | HY-Motion |
| 7 | fallback CLIP encoder | `openai/clip-vit-large-patch14`: `model.safetensors` (config / tokenizer are already cached) | 1.71 GB | no | HY-Motion |
| – | do **not** download | `tencent/Text2MotionPrompter` (prompt rewriter) | 61 GB | no | optional, skipped |

Example: `hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"` (or the registry's
`./install.sh model kimodo-soma-rp`, which passes `--include` per file). All go to `~/.cache/huggingface/hub`.

Not needed by the chosen design (listed so nobody registers by mistake):

| item | register / agree page | files | where it would go | why not needed |
|---|---|---|---|---|
| SMPL | https://smpl.is.tue.mpg.de/ | – | – | no model here outputs SMPL meshes we use |
| SMPL-H / MANO | https://mano.is.tue.mpg.de/ | – | – | HY-Motion ships its own WoodenMesh template (LFS) |
| SMPL-X | https://smpl-x.is.tue.mpg.de/ | `SMPLX_NEUTRAL.npz` | `vendor/kimodo/kimodo/assets/skeletons/smplx22/` | only for Kimodo-SMPLX, which is non-commercial and not added |
| Kimodo-SMPLX-RP-v1 | https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1 (gated, NVIDIA R&D licence) | – | – | non-commercial |
| AMASS / HumanML3D | https://amass.is.tue.mpg.de/ | – | – | only for training / HumanML3D models, which are not added |

## 6. Risks

| risk | impact | mitigation |
|---|---|---|
| Llama 3 access not granted / slow | Kimodo unusable | test HY-Motion-Lite first (no gating); the installer prints the agree URL in `manual:` |
| transformers 5.1 / peft / LLM2Vec on aarch64 + torch 2.9 | load failure | all pure Python; lock with `uv pip compile` on the DGX; fallback env for HY-Motion is independent |
| MotionCorrection x86-only | more foot sliding | sse2neon build (optional step); own contact-driven foot lock in `motion_common` |
| Kimodo imports gradio / scenepic / boto3 at module level | import error offline | import only `kimodo.model`; stub modules in the worker if needed (pattern: `rig_shims.py`) |
| UniRig predicts odd hierarchies on generated characters | detection refuses or maps wrongly | clear refusal text; `retarget.json` exposes the map; tests on synthetic variants; VRoid-cls experiment |
| twist (palms, forearms) wrong | ugly hands | bend-axis alignment (step 4); per-bone angle report; `arm_offset_deg` |
| skin weights from UniRig tear on large motions (kicks, crouch) | visible artefacts | out of scope for retargeting; record in the DGX test; consider limiting `seconds`/prompts in docs |
| HY-Motion pins (torch 2.5.1, transformers 4.53) | phase-2 install work | separate env; relax pins; only `DDIMScheduler` from diffusers |
| licence misreading | legal | `verify: true` on both, notes quote the conditions |

## 7. Test plan

### Unit tests, no GPU (`tests/test_motion.py`, run in `envs/core`: numpy + scipy, no torch, no pygltflib)

`workers/motion_common.py` must import with numpy only; GLB I/O goes through `rig_anim` (also numpy-only).
Synthetic skinned GLBs are built by hand as in `tests/test_rig_anim.py`.

| test | checks |
|---|---|
| registry | `models.d/motion.yaml` loads; param defaults validate; task `motion` in studio `3d` |
| `R_arc` | maps `a` to `b` for random pairs, parallel (identity) and antiparallel (π, orthonormal, det 1) |
| identity invariant | target = the source skeleton itself (same rest) → retargeted FK positions equal source positions (< 1e-5) |
| A-pose target | source frame in T-pose drives an A-pose target → arms horizontal (< 1°) |
| local / FK consistency | `FK(locals)` reproduces `W_j` (< 1e-6); quaternions unit-norm; consecutive dot ≥ 0 after `_canon` |
| root scale | target legs 2× source → horizontal displacement 2×; `in_place` keeps root x / z constant |
| ground | contact frames: lowest foot joint at rest foot height (< 1 mm) |
| foot lock | contact run: ankle world position constant (< 1 mm); thigh / shin lengths preserved (< 1e-6) |
| facing | target facing −Z → walk moves along −Z |
| GLB round trip | `add_tracks` output parses; channel count = mapped joints + 1 translation; times strictly increasing; clip `motion` is `animations[0]`; `rigged_plain` input untouched |
| BVH writer | header joint count, frame count, frame time 1/30 |
| detection: tira fixture | `skeleton.json` derived from UniRig's MIT `examples/tira.glb` output (52 joints) → hips 0, chest 3, head 5, L arm 6-9 with collar 6, R arm 25-28, legs 44-47 / 48-51 with toes; left = +X |
| detection: variants | A-pose, no collar, 5-joint spine, chibi (legs 20 % of height), tail + hair + skirt bones, rig rooted at the chest, facing −Z (with `facing: auto`), mirrored joint order |
| rejections | quadruped (4 floor limbs), bust (16-joint fixture, no legs), snake chain, one arm missing, < 10 joints → `InputError` with the expected reason |
| name shortcut | VRoid / Mixamo / SOMA-named skeletons map without geometry |

### DGX measurements (when weights are downloaded)

| measurement | how | target / note |
|---|---|---|
| install | `./install.sh motion`; does the sse2neon MotionCorrection build work | record `tested: [linux-aarch64]` |
| load time | Llama 8B + adapters + Kimodo from page cache and cold | ~40 s expected |
| encode / generate | 5 s and 10 s clips at 100 steps; 3 × 10 s multi-prompt | record s, `torch.cuda.max_memory_allocated` |
| encoder on CPU | `TEXT_ENCODER_DEVICE=cpu` time vs memory | decide the default |
| retarget | time for 52 joints × 300 frames | < 1 s |
| quality set | tira + 3 generated characters (T-pose, A-pose, chibi), prompts: walk forward, run, jump, wave right hand, kick, dance, turn around | screenshots / videos in model-viewer |
| metrics | foot skate (cm/s during contact), ground penetration (cm), left / right correctness (right hand is on −X), forward = +Z, palm orientation during wave, arm-body intersection | with and without foot lock / MotionCorrection |
| fallback | same set on HY-Motion-Lite | time / memory / quality vs Kimodo |
| UniRig VRoid cls | rerun UniRig on tira with `assign_cls: vroid` | do bones come out named and sensible? |

## 8. Open questions only a real run settles

1. Does Kimodo (LLM2Vec + transformers 5.1 + peft) run on aarch64 / torch 2.9 / GB10, at what speed and memory, and
   does the Llama access arrive? (If not, HY-Motion-Lite becomes the primary.)
2. Does the convention-free rest estimate plus bend-axis twist alignment look right on UniRig rigs of generated
   characters: detection success rate, A-pose and chibi characters, palms, arm-body intersection?
3. How bad is foot skating without MotionCorrection (and does the sse2neon build work), how good is our own foot lock,
   and how well do UniRig's predicted skin weights deform under large motions?
