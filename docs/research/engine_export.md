# Game-engine export (LOD chain, collision, Unreal / Unity / Godot / glTF packages, delivery)

Task `engine_export`, model `engine-export`, handler `engine.export` (CPU, `gpu=False`), backend `engine` which shares the
`envs/retopo` venv (`install/engine.yaml` adds `coacd` and `libmeshoptimizer.so`). No weights. Measured on the DGX Spark
(aarch64, 20 CPU cores, 121 GB unified memory). x86_64 is untested.

## 1. What it does

A mesh asset (best the retopo result with its baked textures) becomes a package for one engine:

| target | files | LODs | collision | textures |
|---|---|---|---|---|
| `unreal` | binary FBX 7.4 (cm, Y up, `SM_` names, `LOD0..n` meshes under one node) | meshes `SM_<name>_LODn` | `UCX_SM_<name>_LOD0_NN` meshes | `T_<name>_BC`, `_N` (DirectX, green flipped), `_ORM` |
| `unity` | FBX (m) + optional editor script `Editor/OpenAssetStudioPostprocessor.cs` | `<name>_LOD0..n` children (Unity builds a LOD Group by name) | `UCX_<name>_LOD0_NN` hull meshes (the editor script makes them convex MeshColliders) | `_BaseMap`, `_Normal`, HDRP `_MaskMap` and / or URP `_MetallicSmoothness` + `_Occlusion` |
| `godot` | GLB | none (Godot builds LODs on import) | `<name>_hull_NN-convcolonly` nodes | `_albedo`, `_normal`, `_orm` |
| `gltf` | GLB with `MSFT_lod` + `<name>_collision.glb` | `MSFT_lod` | separate collision GLB | embedded, also plain PNG |

Rigged or animated meshes go out as GLB for every engine (skin, skeleton and clips are kept); for Unreal and Unity each LOD is a separate
`_LODn.glb`, glTF uses `MSFT_lod`. They get a capsule collision (reported in the manifest) because skeletal meshes carry no `UCX_`.

The result asset is a **mesh asset**: `mesh.glb` is the preview (LOD0 plus the lower LODs as `MSFT_lod` nodes and hidden hull nodes, with
`extras.oas_engine = {lod0, lods, hulls, root}` for the viewer). Attachments: `engine_<target>.zip` (the package), `engine_report.json`,
`engine_collision.glb`. `meta.worker.engine` holds the summary the UI draws (target, name, rigged, lods, hulls, collision, volume_ratio,
package, package_bytes, files, textures, images_resized, style, deliveries). The licence is that of the source (the tools are MIT).

### Pipeline

1. Read the GLB (pure Python, `engine_common.py`): primitives merged per material, normals rebuilt where zero (reported), UVs kept.
2. LOD chain with meshoptimizer `meshopt_simplify` (ctypes, `meshopt_ctypes.py`): ratios of LOD0 (`lod_preset` halves 50/25/12.5 %, aggressive, gentle or custom `lod_ratios`),
   UVs, normals and material borders are kept intact by the simplifier options. A level stops when the error limit (`lod_max_error`, 0.03 of the model size) or the triangle floor is reached;
   the report says why ("LOD3 and below not made ...").
3. Collision: CoACD 1.0.14 on a proxy of at most 5k triangles (the mesh is decimated first), 120 s watchdog; every hull is reduced to at most 64 vertices / 124 triangles
   (`max_verts_cap`, well inside the engines' convex limits). Simple shapes (`box`, `obb`, `capsule`, `convex_hull`) take well under a second.
   `auto` is decomposition for props and a capsule for rigged characters.
4. Placement: scale (`size_mode` keep | height with `height_m`), pivot (bottom_center, center, origin), yaw, units per engine.
5. Textures: `engine_common.pack_textures(maps, target)` converts the canonical maps (OpenGL normal; ORM with R = AO, G = roughness, B = metallic) to the engine's
   naming and packing. It is generic and documented so that a later PBR-material agent can reuse it (`engine_material(target)` gives the material block).
   `tex_max` resizes (Lanczos, never up), reported as `images_resized`.
6. Package: FBX written by `workers/fbx_write.py` (a pure-Python binary FBX 7.4 writer, no Autodesk code, no SDK) or GLB by `glb_bytes`; zip with `manifest.json`
   (units, axes, per LOD error, collision, textures with import hints, `x_style` reference, licence), `engine_report.json` and a `README.txt` with the import steps.

## 2. Measured

Soldier (retopo result, 4,420 triangles, 3,228 vertices, three 1024² maps) through the running core (job wall time includes the worker start):

| target | job | package | LOD triangles | hulls |
|---|---|---|---|---|
| unreal | 7.5 s | 2.6 MB (FBX 353 KB) | 4,420 / 2,210 / 1,104 / 552 | 16 (63 to 64 vertices, 124 triangles) |
| unity | 7.0 s | 3.6 MB | same | 16 |
| godot | 7.0 s | 4.7 MB | 4,420 (no LODs) | 16 |
| gltf | 7.0 s | 2.7 MB | same as unreal | 16 |
| godot, `collision: box` | 0.5 s | 4.7 MB | 4,420 | 1 |
| unity, `capsule` / `obb` / `convex_hull` | 0.5 s each | 3.6 MB | same as unreal | 1 |

LOD shape error (fraction of the model size): LOD1 0.23 %, LOD2 0.62 %, LOD3 1.31 %. CoACD takes 6.9 s of the 7 s (proxy 4,420 triangles).

Other inputs (worker run by hand, `envs/retopo/bin/python`, `/usr/bin/time`):

| input | wall | peak RSS | LODs | collision |
|---|---|---|---|---|
| tank (3,000 triangles) | 4.0 s | 0.24 GB | 3,000 / 1,500 / 780 (LOD3 refused: error limit) | 16 hulls, 3.4 s, hull volume 1.05x |
| dragon (6,000 triangles) | 6.6 s | 0.61 GB | 6,000 / 3,000 / 1,500 / 1,258 | 16 hulls, 6.2 s, hull volume 2.5x (loose: concave) |
| soldier, unreal | 7.4 s | about 0.7 GB | 4,420 / 2,210 / 1,104 / 552 | 16 hulls |
| rigged soldier (24.8 MB GLB, 4k maps), unreal | 2.0 s | about 0.7 GB | 4,448 / 2,224 / 1,112 / 556 | capsule |
| same, `tex_max: 1024` | 3.5 s | | | 4 maps resized, package 8.3 MB instead of 51 MB |

The core process stays near 100 MB; the worker is a separate process that is unloaded like the others (CPU job, so it never waits for the GPU queue).

## 3. Validation

* `mesh.glb` (preview), the glTF and Godot packages, the collision GLB and the rigged outputs: Khronos glTF-Validator 2.0 (npm `gltf-validator`): 0 errors, 0 warnings
  (info only: unused UVs on hull nodes, MSFT_lod is an unknown but harmless extension).
* Unreal FBX: loaded with ufbx (`fbxdump`, the loader behind many tools): 0 warnings, 4 LOD meshes + 16 `UCX_` meshes, material `M_Soldier` with `T_Soldier_BC` / `_N`.
* Bugs found by the validator and fixed: empty top-level arrays in the collision GLB (`EMPTY_ENTITY`), and a skinned mesh below a non-root node (`NODE_SKINNED_MESH_NON_ROOT`:
  skinned meshes stay scene roots).

**Not yet verified in the real applications.** Nobody has imported these files into Unreal Editor, Unity or Godot yet. What to check on the first real import:
Unreal: FBX scale (should be 1:1 in cm), LOD0..3 detected as LODs of one static mesh, `UCX_` pieces become the collision (Collision complexity "Use simple as complex" off),
ORM masked as "Masks" with sRGB off, normal map compression "Normalmap". Unity: LODGroup built from the `_LODn` names (the editor script does it; without it the LODs
are just child meshes), the colliders as MeshColliders (convex), URP / HDRP material remap. Godot: `-convcolonly` nodes become `StaticBody3D` collision on import.

## 4. Delivery into a game project (security)

A **Deliver** button copies the package into a game project folder. Everything about it is designed so that a bad request can write nowhere but a folder the user chose by hand:

* Destinations live **only in `settings.json`** (`engine.destinations`: `id`, `label`, `path`, `engine`, `subdir`, `max_mb`, `allow_lan`); the web UI and the API cannot create or edit them
  (the schema field is `editable: false`), and a request names a destination by **id** (`[a-z0-9_-]{1,32}`), never by path (`extra=forbid` on the body).
* A destination is refused at load time when its path is `/`, the home directory, has fewer than 3 components, is inside the studio, `data/` or `logs/`, or is a system folder.
* **Local only** unless `allow_lan` is set for that destination: the caller must be on the studio's machine, judged from the socket peer and `X-Forwarded-*` / `Forwarded` headers
  (`app/netguard.py`); the HTTP MCP server calls back from 127.0.0.1, so the MCP tool `deliver_engine_export` is gated on the MCP caller's own locality instead.
* A browser request must carry an allowed `Origin` (or none) - a web page on another site cannot post to the local studio.
* Only the whitelisted members of the package are written (names `[A-Za-z0-9_.-]`, optionally under `Textures/` or `Editor/`; extensions fbx, glb, png, json, txt, cs; at most 64 files;
  total size under `max_mb`; zip-bomb guard on the declared and the real size).
* The write walks the folder with file descriptors and `O_NOFOLLOW` (no symlink is followed at any level), creates a temp file with `O_EXCL` and `os.replace`s it.
* A ledger `.oas-delivery.json` (sha256 per file) lets a second delivery replace **only** what the studio wrote and the user did not change. A file edited since is never overwritten
  (409 `edited`, nothing written), a folder that exists without a ledger is not touched (409 `not_ours`), and nothing is ever deleted.
* One delivery at a time (409 `busy`). Every refusal is a plain 4xx with a `code`, logged with the asset, destination and client.

Live check on a second core (:7861, scratch data and a destination under the scratchpad): first delivery wrote 7 files; a repeat was "unchanged"; a path or unknown name as id gave 404
`no_destination`; `/home/<user>` as a destination was refused as `bad destination`; an extra `path` field 422; a Unity package into an Unreal destination 400 `engine_mismatch`;
`Origin: http://evil.example` 403; `X-Forwarded-For: 10.1.2.3` 403 `not_local`; an edited `README.txt` 409 `edited` (also with `overwrite: true`); a symlink in place of the
`OpenAssetStudio` folder 409 `unsafe_path` (nothing written outside); a folder without ledger 409 `not_ours`. GET `/api/settings` and `/api/x/engine/destinations` hide the paths from a non-local caller.

## 5. Settings

Per project (`engine.*`, Settings page, group "Game engine"): `target`, `height_m`, `prefix_static`, `prefix_texture`. They pre-fill the form (target, height); the handler itself only
takes the prefixes, so an explicit choice in the form is never overridden. `engine.destinations` is system-wide and **file-only** (listed, redacted for remote callers, never editable in the UI).

## 6. UI

3D studio, step "4 Export" (grouped tabs) > "Game engine": the form (generated from the registry, presets unreal_prop, unreal_character, unity_prop, unity_mobile, godot, gltf_web),
a "What you get" card that follows the form (and whether the first mesh is rigged), then after a job the report (LOD table, collision, textures, warnings, licence, Download package,
Deliver). The viewer gets a LOD switcher (client-side swap of the mesh in the GLB; the stats line shows the triangles of the shown LOD) and a Hulls toggle (translucent hull overlay).
The Library multi-select bar has "Export for game engine...", the asset drawer / viewer toolbar a "Game engine" button. Checked at 1440 px and 390 px, dark and light, no horizontal overflow.
(i) balloons: `uiinfo.js` (`engine_*`) and `app/param_info.py` for each parameter.

## 7. Limits and open points

* Collision for rigged meshes is a single capsule only; no per-bone physics bodies.
* The FBX writer covers static meshes (nodes, meshes, materials, textures, LOD names, UCX meshes), not skins or animation: rigged output is GLB by design.
* CoACD is the slow part (about 7 s for 5k triangles, `collision_quality: fast` is cheaper); concave props get loose hulls (dragon 2.5x volume), raise `max_hulls` / `detailed`.
* Not tested: x86_64, Windows paths in destinations, Unreal / Unity / Godot import (see section 3).
