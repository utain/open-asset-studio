# Retopo + bake (Instant Meshes, xatlas, Open3D)

Task `retopo`, model `retopo-instant-meshes`, backend `retopo` (CPU, no weights, no Blender). Measured on the DGX Spark
(aarch64, 20 CPU cores, 121 GB unified memory). x86_64 is untested.

## 1. Goal and constraints

A dense generated mesh (Hunyuan3D / TRELLIS: 60k to 550k triangles, no UV or a messy one) becomes a clean, mostly-quad low poly with a
face budget, a UV atlas and baked PBR textures, so that Rig and Motion get a game-ready mesh. Constraints: no Blender, CPU only, offline, only
permissive licences (BSD / MIT), builds on aarch64 and x86_64 with no sudo.

## 2. Candidates

| Candidate | Decision |
|---|---|
| Instant Meshes (PyNanoInstantMeshes, BSD-3) | chosen: quads, 0.9 to 2 s on 550k triangles; needs a small patch for `pure_quad` |
| QuadriFlow | rejected: Boost dependency, x86 wheel only |
| pymeshlab | rejected: GPL |
| Blender quad remesher / QRemeshify | rejected: needs Blender |
| nvdiffrast | rejected: licence and nvcc |
| embreex | rejected: no aarch64 wheel (Open3D's bundled Embree is used instead) |
| warp | rejected: size |
| AI artist-mesh models | phase 2 |

## 3. Remesh quality (548k-triangle character, Instant Meshes)

| mode / target | faces | quad % | mean dist x diag | p99 x diag |
|---|---|---|---|---|
| quad-dominant 5k | 5,643 | 82.7 | 0.42e-3 | 2.8e-3 |
| quad-dominant 15k | 16,565 | 87.1 | 0.16 to 0.18e-3 | 1.2e-3 |
| quad-dominant 30k | 32,397 | 89.2 | 0.09e-3 | 0.7e-3 |
| pure 5k (asks 1/4, subdivides) | 4,867 | 100 | 1.1 to 1.4e-3 | - |
| pure 15k | - | 100 | 0.52e-3 | - |

(Prototype numbers.) Quad-dominant has 10 to 18 % irregular vertices. Without the correction pass the face count overshoots by about 10 %; the
worker re-runs once with a corrected target when the result is more than 8 % off. Raw output defects: 31 to 120 boundary edges (holes of 7 to
19 edges), 0 to 2 non-manifold edges, 2 to 17 components (fragments of 2 to 6 faces). That is why the repair stage exists.

## 4. End to end: timings and memory

Shipped code, run through `workers/retopo_worker.py` (driver keeps stdin open, peak RSS from the process tree). Input: `white_mesh.glb`
548k triangles for the shape and detail, colour from the 60k-triangle textured `mesh.glb` (4096 albedo + MR), AO 16 rays.

| preset | result polys (quad %) | wall s | peak RSS | notes |
|---|---|---|---|---|
| mobile 2,500 @ 1024 | 2,539 (78.2 %) | 6.3 | 0.97 GB | ray misses 0.15 % |
| pc 12,000 @ 2048 | 12,111 (85.8 %) | 14.0 | 1.24 GB | ray misses 0.14 %, no warnings |
| hero 30,000 @ 4096 | 32,381 (89.5 %) | 41.8 | 2.05 GB | ray misses 0.03 %, 5 fragment faces dropped |
| hero, input subdivided to 2.19M triangles | - | 40.3 | 2.48 GB | |

The unchunked prototype peaked at 7.73 GB for the hero preset. The shipped bake rasterises, casts rays and samples colour in row chunks, so the
peak is 2.05 GB (2.48 GB for a 2.2M-triangle input). `memory_gb` in the registry is 4 (measured peak plus headroom for the Open3D scene of a
larger input).

Through the core (`retopo.bake` handler, project `retopo-test`, copy of the same character, preset pc): job 13.9 s, worker run 13.45 s
(load 0.39 s). Stage split (s): load 1.03, remesh 2.35, repair 0.31, uv 1.21, tangents 0.60, raster 0.15, cage 0.08, normal 1.62, colour 1.29,
AO 4.46, write 0.35 (PNG encoding is threaded and inside "write" / the stages). Output GLB 8.8 MB: 12,111 polys (10,395 quads, 1,716 triangles)
= 22,506 triangles, 13,995 vertices, 0 boundary edges, 1 component, 6 holes filled, mikktspace tangents, cage 0.0035 x diagonal, mean
surface distance 0.022 % and p99 0.15 % of the diagonal, 3 textures at 2048 (normal, base colour, ORM), `extras.oas_quad_pairs` = 10,395.
The 2.19M-triangle input spends 1.7 s loading and about 8.5 s remeshing (linear in the input size).

## 5. Repair statistics

- pc: raw 12,043 faces with 68 boundary edges, 0 non-manifold edges, 1 component; 6 holes filled, 0 faces dropped, 0 boundary edges left.
- Prototype 15k: 7 non-manifold and 5 fragment faces dropped, 8 holes filled; 16 boundary edges remained because of pinched (figure-8) loops.
  The shipped walker (`boundary_loops_ex`) follows directed half-edges, prefers the edge that returns to a vertex already on the path and cuts
  off sub-loops, so a figure-8 hole is filled as two loops. Regression: `tests/test_retopo.py::test_boundary_loops_figure_8`
  (old walker finds none, the new one finds 4 + 4). Loops that cannot be closed (open chains) count as `open_holes` and give a warning.
- Components smaller than `max(min_component_faces, min_part x faces)` are dropped; the largest is always kept.

## 6. Bake quality

- Cage 0.002 to 0.012 x diagonal, ray misses 0.03 to 0.17 %, UV utilisation 0.68 to 0.73, texel coverage 0.52 to 0.54 at 2048.
- Prototype checks (not re-measured on the shipped code): normal error 0.19 degrees against 3.1 to 3.7 degrees without the map, green-flipped
  control 3.1 to 3.5 degrees; model-viewer render check against the dense mesh 0.48 / 0.49 / 2.33 / 1.97 (the conventions of section 7); albedo
  MAE 0.77 to 0.97 / 255; vertex-colour transfer 0.35 / 255 (p99 1.4).
- Shipped code on the character: see section 4 (distance and ray statistics from `retopo_report.json`). Screenshots of before / after in the
  studio: the compare modal at 1440 and 390 px, dark and light (taken in headless Chrome with a software GL; not committed).

## 7. Conventions

- UV: glTF (v down); xatlas charts, 1 texel padding, EDT dilation of the unused texels.
- Normal map: tangent space, OpenGL (Y+, green up), MikkTSpace tangents (fallback: Lengyel) written as `TANGENT`; the same tangents are used for
  the bake and by the viewer, so shading is identical in model-viewer, Blender, Godot and Unity.
- ORM: one texture, R = ambient occlusion, G = roughness, B = metallic (glTF `occlusionTexture` + `metallicRoughnessTexture` point at it).
- Base colour: sRGB, sampled from the colour source (texture or vertex colours) at the point the bake ray hits.
- GLB triangles in quad pairs first (`extras.oas_quad_pairs`: the number of leading triangle pairs that form quads, used by the viewer's
  quad-aware wireframe); `retopo_quads.obj` (+ `.mtl`) has the real quads.

## 8. Rig chain

- The trimesh round trip inside `rig_worker` dropped `NORMAL` / `TANGENT` / extras of a retopo GLB (seam normals: median 0.83 degrees, p99 80
  degrees off). Fixed by `m3d_common.carry_vertex_attrs`, called from `_write_rigged_glb`: it matches vertices (count, then position within
  1e-5 x diagonal) and re-attaches normals, tangents and extras; with no match it leaves the file untouched.
- Checked on the real retopo GLB: trimesh keeps 13,995 vertices, the round trip exports `POSITION / TEXCOORD_0 / NORMAL`, and `carry_vertex_attrs`
  returns True and restores `TANGENT` and `oas_quad_pairs`.
- `rig_anim` and motion preserve bytes. The 30k hero has 57k triangles, which is above UniRig's 50k `max_faces`: the rig worker decimates for the
  prediction only, and the weights go to every vertex.
- A rigged or animated input is refused (`kind: input`, text E3 / E4); checked live with a skinned GLB whose asset meta carried no flag (the
  plugin reads the GLB).

## 9. Install notes

- `./install.sh backend retopo`: 9 s from the lock file, venv about 0.49 GB (54 pinned packages in `env-lock/retopo.txt`, plus two built
  from source).
- PyNanoInstantMeshes is built from the pinned checkout (`vendor/vendor.lock`) after `install/files/pynim_pure_quad.patch`, which adds the
  `pure_quad` switch; GCC 13 needs `CXXFLAGS="-fpermissive -Wno-changes-meaning"`. The licence of the binding is marked `verify: true`.
- `pymikktspace` needs `cython<3` to build; the step is optional, the worker falls back to Lengyel tangents.
- Open3D 0.20 is installed with its dependencies (no `--no-deps`); its Embree backs `RaycastingScene`.
- aarch64 tested, x86_64 untested.

## 10. Deviations from the plan, limits

- `install/retopo.yaml` removes the `.git` of the copied checkout before patching and building (the pinned checkout is only a source tree).
- Chunked bake: the prototype's whole-image arrays are processed in row chunks (rasterisation, rays, colour), not in one pass.
- Limits: a mesh below about 45k triangles is warned about (retopo needs about 4 x the face target to bake detail from); self-intersecting or
  open sources give ray misses that fall back to the closest point; the result is one atlas, no per-part layout control; no x86_64 run.
- Out of scope in phase 1: artist-grade edge flow, symmetric remeshing, multiple materials, UDIMs.
