"""Game-engine export backend (envs/retopo, CPU only, no weights): one mesh asset -> an engine-ready package.

  task engine_export   input role `mesh` (a GLB)  ->  mesh asset `mesh.glb` (the preview: LOD0, the lower LODs through MSFT_lod, the collision
                       hulls as hidden nodes) + attachments `engine_<target>.zip` (the package), `engine_report.json`, `engine_collision.glb`

Pipeline (docs/research/engine_export.md): read the GLB -> sanitize (normals, weights, unused vertices) -> LOD chain (meshoptimizer through
ctypes, workers/meshopt_ctypes.py) -> collision (CoACD on a 5k-triangle proxy, or box / capsule / one hull) -> engine texture packing
(engine_common.pack_textures) -> the target's files (Unreal / Unity static: binary FBX from workers/fbx_write.py; Godot / glTF and every rigged or
animated mesh: GLB with the skeleton and clips kept) -> manifest, report, README -> zip.

The pure maths lives in workers/engine_common.py (tested in envs/core). The job handler is app/plugins/engine.py, which passes the source asset's
name, licence, style record and the project's naming prefixes in the private param `_ctx`.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
import io
import json
import resource
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import engine_common as C
import fbx_write as FW
import meshopt_ctypes as MO
from studio_worker import Backend, InputError, serve

HERE = Path(__file__).resolve().parent
COACD_TIMEOUT_S = 120
MAX_TRIS = 4_000_000


# ------------------------------------------------------------------ params
def _params(p: dict) -> dict:
    g = lambda k, d: d if p.get(k) is None else p[k]  # noqa: E731
    target = str(g("target", "unreal"))
    if target not in C.TARGETS:
        raise InputError(f"Unknown engine '{target}' (one of {', '.join(C.TARGETS)}).")
    tm = g("tex_max", "keep")
    return {"target": target, "name": str(g("name", "") or ""), "lods": int(g("lods", 3)), "lod_preset": str(g("lod_preset", "halves")),
            "lod_ratios": str(g("lod_ratios", "0.5,0.25,0.1")), "lod_max_error": float(g("lod_max_error", 0.03)),
            "lod_min_tris": int(g("lod_min_tris", 300)), "collision": str(g("collision", "auto")),
            "collision_quality": str(g("collision_quality", "balanced")), "max_hulls": int(g("max_hulls", 16)),
            "hull_max_verts": int(g("hull_max_verts", 64)), "size_mode": str(g("size_mode", "keep")), "height_m": float(g("height_m", 1.8)),
            "pivot": str(g("pivot", "bottom_center")), "tex_max": 0 if str(tm) == "keep" else int(tm),
            "unity_pipeline": str(g("unity_pipeline", "both")), "yaw": int(g("yaw", 0)),
            "preview_collision": bool(g("include_collision_preview", True))}


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


# ------------------------------------------------------------------ vertex order
def _reorder(prim: dict) -> dict:
    """Vertex cache order for the triangles, then vertex fetch order (first use) for the attributes; same triangles."""
    F = MO.optimize_vertex_cache(prim["F"], len(prim["attrs"]["POSITION"]))
    remap, used = MO.optimize_vertex_fetch_remap(F, len(prim["attrs"]["POSITION"]))
    attrs = {}
    for k, v in prim["attrs"].items():
        out = np.empty((used,) + v.shape[1:], v.dtype)
        sel = remap != 0xFFFFFFFF
        out[remap[sel]] = v[sel]
        attrs[k] = out
    return {**prim, "attrs": attrs, "F": remap[F].astype(np.int64)}


# ------------------------------------------------------------------ collision
def _proxy(V, F):
    """Welded positions, simplified to at most PROXY_TRIS triangles (CoACD is slow on dense meshes)."""
    Vw, Fw = C.weld_positions(V, F)
    if len(Fw) > C.PROXY_TRIS:
        F2, _ = MO.simplify(Fw, Vw, C.PROXY_TRIS, 1.0, 0)
        if len(F2):
            Fw, a = C.compact(F2.astype(np.int64), {"POSITION": Vw})
            Vw = a["POSITION"]
    return Vw, Fw


def _run_coacd(Vp, Fp, cap, preset, hull_max_verts, ctx):
    import coacd

    coacd.set_log_level("error")
    kw = {"threshold": preset["threshold"], "max_convex_hull": cap, "preprocess_resolution": preset["preprocess_resolution"],
          "mcts_iterations": preset["mcts_iterations"], "decimate": True, "max_ch_vertex": hull_max_verts, "seed": 0}
    if preset.get("mcts_nodes"):
        kw["mcts_nodes"] = preset["mcts_nodes"]
    box: dict = {}

    def work():
        try:
            parts = coacd.run_coacd(coacd.Mesh(np.asarray(Vp, np.float64), np.asarray(Fp, np.int32)), **kw)
            box["parts"] = [(np.asarray(v), np.asarray(f)) for v, f in parts]
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=work, daemon=True, name="coacd")
    t.start()
    t0 = time.time()
    while t.is_alive():
        t.join(0.5)
        ctx.check()
        if time.time() - t0 > COACD_TIMEOUT_S:
            return None, f"CoACD took longer than {COACD_TIMEOUT_S} s"
    if "err" in box:
        return None, f"CoACD failed: {box['err']}"
    return box["parts"], None


def make_collision(P, prims0, rigged, ctx, warnings) -> dict:
    """-> {hulls: [(V, F, kind)], report, capsule}. All geometry in the model's own units (not placed)."""
    t0 = time.time()
    kind = P["collision"]
    if kind == "auto":
        kind = "capsule" if rigged else "decomposition"
    rep = {"method": kind, "requested": P["collision"], "quality": P["collision_quality"], "hulls": 0, "max_verts": 0, "max_tris": 0,
           "sum_tris": 0, "volume_ratio": None, "seconds": 0.0, "dropped": 0}
    if kind == "none":
        return {"hulls": [], "report": rep, "capsule": None}
    allV = np.concatenate([p["attrs"]["POSITION"].astype(np.float64) for p in prims0])
    allF = np.concatenate([p["F"] + o for p, o in zip(prims0, np.cumsum([0] + [len(p["attrs"]["POSITION"]) for p in prims0[:-1]]))])
    Vw, Fw = C.weld_positions(allV, allF)
    bbox_vol = float(np.prod(Vw.max(0) - Vw.min(0))) or 1.0
    hulls, capsule = [], None
    if kind in ("box", "obb"):
        b = C.simple_box(Vw) if kind == "box" else C.simple_obb(Vw)
        h = C.box_mesh(b)
        hulls = [(h[0], h[1], "box")] if h else []
    elif kind == "capsule":
        capsule = C.simple_capsule(Vw, tight=rigged)
        h = C.capsule_mesh(capsule)
        hulls = [(h[0], h[1], "convex")] if h else []
        capsule = {k: (np.asarray(v).tolist() if hasattr(v, "tolist") else v) for k, v in capsule.items()}
    elif kind in ("convex_hull", "decomposition"):
        Vp, Fp = _proxy(allV, allF) if len(Fw) > C.PROXY_TRIS else (Vw, Fw)
        rep["proxy_tris"] = int(len(Fp))
        if kind == "decomposition":
            preset = C.COLLISION_PRESETS.get(P["collision_quality"], C.COLLISION_PRESETS["balanced"])
            cap = max(1, min(P["max_hulls"], preset["cap"]))
            parts, err = _run_coacd(Vp, Fp, cap, preset, P["hull_max_verts"], ctx)
            if parts is None:
                warnings.append(f"{err}: one convex hull is used instead")
                rep["method"] = "convex_hull"
                parts = [(Vp, Fp)]
            good, dropped = C.validate_hulls(parts, P["hull_max_verts"], bbox_vol)
            rep["dropped"] = dropped
            if not good:
                warnings.append("CoACD gave no usable hull: one convex hull is used instead")
                rep["method"] = "convex_hull"
                good, _ = C.validate_hulls([(Vp, Fp)], P["hull_max_verts"], bbox_vol)
            hulls = [(v, f, "convex") for v, f in good]
        else:
            good, _ = C.validate_hulls([(Vp, Fp)], P["hull_max_verts"], bbox_vol)
            hulls = [(v, f, "convex") for v, f in good]
    if rigged and kind == "decomposition":
        pass
    rep["hulls"] = len(hulls)
    if hulls:
        rep["max_verts"] = int(max(len(h[0]) for h in hulls))
        rep["max_tris"] = int(max(len(h[1]) for h in hulls))
        rep["sum_tris"] = int(sum(len(h[1]) for h in hulls))
        mv = C.mesh_volume(Vw, Fw)
        if mv and mv > 0:
            rep["volume_ratio"] = round(sum(C.hull_volume(v, f) for v, f, _ in hulls) / mv, 3)
    rep["seconds"] = round(time.time() - t0, 2)
    return {"hulls": hulls, "report": rep, "capsule": capsule}


# ------------------------------------------------------------------ GLB assembly
class Assembler:
    """Builds the GLB variants of one model from the (resized) source GLB and the finished LOD levels / hulls."""

    def __init__(self, j, blob, glb_mode, mesh_node, levels, names, pl, hulls, coverage):
        self.j, self.blob, self.glb_mode, self.mesh_node = j, blob, glb_mode, mesh_node
        self.levels, self.names, self.pl, self.hulls, self.coverage = levels, names, pl, hulls, coverage

    def _trs(self) -> dict:
        pl = self.pl
        if pl["identity"]:
            return {}
        kw = {}
        if pl["scale"] != 1.0:
            kw["scale"] = [pl["scale"]] * 3
        if pl["yaw_deg"]:
            kw["rotation"] = [round(x, 9) for x in pl["rotation"]]
        if any(abs(x) > 1e-12 for x in pl["translation"]):
            kw["translation"] = [float(x) for x in pl["translation"]]
        return kw

    def build(self, variant: str, only_level: int | None = None) -> bytes:
        """variant: preview (LOD0 + MSFT_lod + hidden hull nodes) | gltf (LOD0 + MSFT_lod) | godot (LOD0 + -convcolonly hulls) |
        main (LOD0 only) | lod (level `only_level` alone, no images, for the extra LOD files of a rigged model)."""
        j = copy.deepcopy(self.j)
        B = C.GlbBuilder(j, self.blob)
        names = self.names
        lvl = 0 if only_level is None else only_level
        prims = [B.primitive(p["attrs"], p["F"], p["material"], p.get("extras")) for p in self.levels[lvl]]
        mname = names["lods"][lvl] if lvl < len(names["lods"]) else f"{names['asset']}_LOD{lvl}"
        if self.glb_mode:
            node = j["nodes"][self.mesh_node]
            j["meshes"][node["mesh"]]["primitives"] = prims
            j["meshes"][node["mesh"]]["name"] = mname
            node["name"] = mname
            lod0 = self.mesh_node
            base = {k: node[k] for k in ("translation", "rotation", "scale", "matrix", "skin") if k in node}
            scene = j["scenes"][j.get("scene", 0)]
            children = list(scene["nodes"])
        else:
            for k in ("skins", "animations"):
                j.pop(k, None)
            j["nodes"], j["meshes"] = [], []
            mi = B.mesh(mname, prims)
            lod0 = B.node(mname, mesh=mi)
            base = {}
            j["scenes"] = [{"nodes": []}]
            j["scene"] = 0
            scene = j["scenes"][0]
            children = [lod0]
        lod_nodes: list[int] = []
        if variant in ("preview", "gltf") and len(self.levels) > 1 and only_level is None:
            for k in range(1, len(self.levels)):
                mk = B.mesh(names["lods"][k], [B.primitive(p["attrs"], p["F"], p["material"]) for p in self.levels[k]])
                lod_nodes.append(B.node(names["lods"][k], mesh=mk, **base))
            C.add_msft_lod(j, lod0, lod_nodes, self.coverage)
        if variant == "godot" and self.hulls:
            hull_ids = C.add_godot_hulls(B, [(v, f) for v, f, _ in self.hulls], names["stem"])
            children = children + hull_ids
        root_name = names["asset"] if names["asset"] != mname else f"{names['asset']}_Root"
        # a skinned mesh node must stay a scene root (glTF ignores its parent transforms); the placement root wraps the skeleton instead
        out_of_root = [self.mesh_node] if self.glb_mode and "skin" in j["nodes"][self.mesh_node] else []
        children = [c for c in children if c not in out_of_root]
        trs = self._trs()
        if not self.glb_mode or trs or variant == "godot" and self.hulls:
            root = B.node(root_name, children=children, **trs)
            scene["nodes"] = [root] + out_of_root
        else:
            root = None
            scene["nodes"] = children + out_of_root
        if variant == "preview" and self.hulls:
            hid = C.add_hull_meshes(B, [(v, f) for v, f, _ in self.hulls], [f"{names['stem']}_hull_{k:02d}" for k in range(len(self.hulls))])
            j["extras"] = {"oas_engine": {"lod0": lod0, "lods": lod_nodes, "hulls": hid, "root": root}}
        elif variant == "preview":
            j["extras"] = {"oas_engine": {"lod0": lod0, "lods": lod_nodes, "hulls": [], "root": root}}
        return B.finish(drop_images=(variant == "lod"))

    def collision_glb(self) -> bytes:
        """The hulls alone: one node per hull `<stem>_hull_NN`, no material, under the same placement root."""
        j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [],
             "bufferViews": [], "buffers": [{"byteLength": 0}]}
        B = C.GlbBuilder(j, b"")
        ids = C.add_hull_meshes(B, [(v, f) for v, f, _ in self.hulls], [f"{self.names['stem']}_hull_{k:02d}" for k in range(len(self.hulls))])
        root = B.node(f"{self.names['stem']}_collision", children=ids, **self._trs())
        j["scenes"][0]["nodes"] = [root]
        return B.finish()


# ------------------------------------------------------------------ the backend
class EngineBackend(Backend):
    name = "engine"
    max_loaded = 1

    def probe(self, model):
        import coacd  # noqa: F401
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import scipy  # noqa: F401

        MO.lib()
        return True, ""

    def load(self, model, ctx):
        MO.lib()
        return {"ready": True}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # -------------------------------------------------------------- run
    def run(self, job, ctx):
        if job.task != "engine_export":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        T0 = time.time()
        tim: dict = {}
        stamp = [T0]

        def lap(key):
            now = time.time()
            tim[key] = round(now - stamp[0], 2)
            stamp[0] = now
            ctx.check()

        P = _params(job.params)
        X = job.params.get("_ctx") or {}
        meshes = [i for i in job.inputs if i.get("type") == "mesh"]
        if not meshes:
            raise InputError("Pick a mesh asset (GLB) to export.")
        inp = job.input(role="mesh") or meshes[0]
        src = Path(inp["path"])
        if src.suffix.lower() != ".glb":
            raise InputError("The engine export needs a GLB (binary glTF). Export this model as GLB first (3D > Export).")
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        target = P["target"]
        source = dict(X.get("source") or {})
        display = source.get("name") or inp.get("name") or "model"

        # ------------------------------------------------ 1. read
        ctx.progress(0, 100, "load", "Reading the model")
        try:
            j, blob = C.glb_split(src.read_bytes())
        except C.EngineInputError as e:
            raise InputError(str(e)) from None
        except Exception as e:  # noqa: BLE001
            raise InputError("Could not read this file as a GLB model. Re-export it as GLB and try again.") from e
        rigged, animated = bool(j.get("skins")), bool(j.get("animations"))
        glb_mode = rigged or animated
        mnodes = C.mesh_nodes(j)
        if not mnodes:
            raise InputError("This GLB has no mesh in its scene.")
        try:
            if glb_mode:
                if len(mnodes) > 1:
                    raise InputError("This rigged / animated model has several mesh nodes. Merge them into one mesh before the engine export "
                                     "(rig it again from a single mesh), or export the static pieces separately.")
                mesh_node = mnodes[0][0]
                prims = C.read_prims(j, blob, j["nodes"][mesh_node]["mesh"])
            else:
                mesh_node = None
                raw = []
                for ni, M in mnodes:
                    raw += [C.bake_node_transform(p, M) for p in C.read_prims(j, blob, j["nodes"][ni]["mesh"])]
                prims = C.merge_same_material(raw)
        except C.EngineInputError as e:
            raise InputError(str(e)) from None
        tris_in = sum(len(p["F"]) for p in prims)
        if tris_in > MAX_TRIS:
            raise InputError(f"This model has {tris_in / 1e6:.1f} million triangles; the engine export takes up to {MAX_TRIS / 1e6:.0f} million. "
                             "Reduce it first (3D > Export > decimate, or Retopo).")
        lap("read")

        # ------------------------------------------------ 2. sanitize
        ctx.progress(8, 100, "sanitize", "Cleaning normals, weights and unused vertices")
        srep = {k: 0 for k in ("zero_normals", "non_unit_normals", "weight_rows", "joint_slots_cleared", "degenerate_tris", "unused_vertices")}
        clean = []
        for p in prims:
            a, F, r = C.sanitize(p["attrs"], p["F"])
            if len(F) == 0:
                continue
            for k in srep:
                srep[k] += r[k]
            clean.append(_reorder({**p, "attrs": a, "F": F}))
            warnings += [w for w in r["warnings"] if w not in warnings]
        if not clean:
            raise InputError("The model has no usable triangles.")
        prims = clean
        if any("COLOR_0" in p["attrs"] for p in prims) and target in ("unreal", "unity") and not glb_mode:
            warnings.append("Vertex colours are not written to FBX (the GLB preview keeps them).")
        if any("TEXCOORD_0" not in p["attrs"] for p in prims):
            warnings.append("Some of the mesh has no UV map: textures cannot be applied to it.")
        allP = np.concatenate([p["attrs"]["POSITION"] for p in prims]).astype(np.float64)
        pl = C.placement(allP.min(0), allP.max(0), P["size_mode"], P["height_m"], P["pivot"], P["yaw"])
        lap("sanitize")

        # ------------------------------------------------ 3. LODs
        n_lods = 0 if target == "godot" else max(0, P["lods"])
        ctx.progress(12, 100, "lods", "Building the LOD chain")
        chain = C.build_lod_chain(prims, n_lods, P["lod_preset"], P["lod_ratios"], P["lod_min_tris"], P["lod_max_error"], glb_mode and rigged,
                                  MO.simplify, scale_of=MO.simplify_scale)
        levels = [chain["levels"][0]] + [[_reorder(p) for p in lv] for lv in chain["levels"][1:]]
        warnings += chain["skipped"]
        if target == "godot" and P["lods"] > 0:
            warnings.append("Godot builds its own LODs when it imports the model, so no LOD levels were made.")
        n_made = len(levels) - 1
        names = C.engine_names(target, P["name"] or display, rigged=glb_mode, n_lods=n_made,
                               prefix_static=X.get("prefix_static"), prefix_texture=X.get("prefix_texture"))
        sizes = C.screen_sizes(target, len(levels))
        lod_rows = [{"name": names["lods"][i], "tris": r["tris"], "verts": r["verts"], "ratio": r["ratio"], "error": r["error"],
                     "error_abs": r["error_abs"], "screen_size": sizes[i]} for i, r in enumerate(chain["info"])]
        lap("lods")

        # ------------------------------------------------ 4. collision
        ctx.progress(25, 100, "collision", "Building the collision shapes")
        col = make_collision(P, levels[0], glb_mode and rigged, ctx, warnings)
        hulls = col["hulls"]
        if rigged and P["collision"] in ("auto", "capsule") and target in ("unreal", "unity"):
            warnings.append("Rigged model: no UCX_ hull is written (skeletal meshes use a physics asset); the capsule for the character "
                            "controller is in engine_report.json.")
        lap("collision")

        # ------------------------------------------------ 5. textures
        ctx.progress(55, 100, "textures", "Packing the textures for the engine")
        used_mats = []
        for p in levels[0]:
            if p["material"] is not None and p["material"] not in used_mats:
                used_mats.append(p["material"])
        tex_files: dict[str, bytes] = {}
        tex_rows: list[dict] = []
        tex_by_mat: dict = {}
        mat_info: dict = {}
        for k, mi in enumerate(used_mats):
            maps = C.material_maps(j, blob, mi)
            fac = maps.pop("factors")
            mat_info[mi] = fac
            stem = names["stem"] if len(used_mats) == 1 else f"{names['stem']}_{C.safe_stem(fac['name'], 24)}"
            packed = C.pack_textures(target, maps, stem, {"tex_max": P["tex_max"], "unity_pipeline": P["unity_pipeline"],
                                                           "prefix": names["texture_prefix"]})
            tex_by_mat[mi] = {}
            for fn, img, meta in packed:
                tex_by_mat[mi][meta["slot"]] = fn
                tex_rows.append({"file": f"Textures/{fn}", "material": fac["name"], **meta})
                tex_files[fn] = img
            if "orm" not in maps and "orm" not in {m["slot"] for _, _, m in packed} and maps.get("basecolor") is not None:
                warnings.append(f"Material '{fac['name']}' has no metallic / roughness map: only the factors (metallic {fac['metallic']}, "
                                f"roughness {fac['roughness']}) are in the GLB.")
        if tex_files:
            def enc(img):
                b = io.BytesIO()
                img.save(b, "PNG", compress_level=4)
                return b.getvalue()

            with ThreadPoolExecutor(max_workers=4) as ex:
                keys = list(tex_files)
                for kname, data in zip(keys, ex.map(enc, [tex_files[x] for x in keys])):
                    tex_files[kname] = data
        lap("textures")

        # ------------------------------------------------ 6. files of the target
        ctx.progress(70, 100, "write", "Writing the engine files")
        rj, rblob, n_resized = C.resize_glb_images(j, blob, P["tex_max"]) if P["tex_max"] else (j, blob, 0)
        coverage = C.switch_coverage(len(levels))
        asm = Assembler(rj, rblob, glb_mode, mesh_node, levels, names, pl, hulls, coverage)
        files: dict[str, bytes] = {}                         # zip member -> bytes
        if target in ("unreal", "unity", "godot"):             # glTF keeps its textures inside the GLB
            for fn, data in tex_files.items():
                files[f"Textures/{fn}"] = data
        if target in ("unreal", "unity") and not glb_mode:
            fbx = out / f"{names['file']}.fbx"
            self._write_fbx(fbx, target, names, levels, hulls, pl, tex_by_mat, mat_info)
            files[fbx.name] = fbx.read_bytes()
            fbx.unlink()
        elif target in ("unreal", "unity"):
            files[f"{names['file']}.glb"] = asm.build("main")
            for k in range(1, len(levels)):
                files[f"{names['lods'][k]}.glb"] = asm.build("lod", only_level=k)
        elif target == "godot":
            files[f"{names['file']}.glb"] = asm.build("godot")
        else:
            files[f"{names['file']}.glb"] = asm.build("gltf")
            if hulls:
                files[f"{names['stem']}_collision.glb"] = asm.collision_glb()
        if target == "unity":
            files["Editor/OpenAssetStudioPostprocessor.cs"] = (HERE / "engine_files" / "OpenAssetStudioPostprocessor.cs").read_bytes()
        preview = asm.build("preview")
        (out / "mesh.glb").write_bytes(preview)
        coll_glb = None
        if hulls and P["preview_collision"]:
            coll_glb = asm.collision_glb()
            (out / "engine_collision.glb").write_bytes(coll_glb)
        lap("write")

        # ------------------------------------------------ 7. manifest, report, zip
        ctx.progress(90, 100, "package", "Zipping the package")
        style = X.get("style")
        col_rep = {**col["report"], "max_verts_cap": P["hull_max_verts"]}
        if col["capsule"]:
            col_rep["capsule"] = col["capsule"]
        units = {"file": "cm" if (target == "unreal" and not glb_mode) or (target == "unity" and not glb_mode) else "m", "engine": "cm" if target == "unreal" else "m",
                 "note": "FBX vertices are centimetres (UnitScaleFactor 1)" if target in ("unreal", "unity") and not glb_mode else "glTF metres"}
        tex_manifest = [{k: v for k, v in r.items() if k in ("file", "slot", "colorspace", "packing", "size", "material", "use")} for r in tex_rows]
        src_ref = {"id": source.get("id"), "name": display, "x_style": source.get("x_style")}
        man = C.manifest(target, names["stem"], src_ref, source.get("license") or (job.model or {}).get("license"), units, lod_rows, col_rep,
                         tex_manifest, sorted(files), style=style, rigged=glb_mode, names={"asset": names["asset"], "group": names["group"],
                                                                                          "lods": names["lods"], "material": names["material"]})
        rep = C.report(target, {"name": display, "asset_id": source.get("id"), "triangles": tris_in, "rigged": rigged, "animated": animated,
                                "animations": [a.get("name") for a in j.get("animations", [])], "materials": len(used_mats),
                                "primitives": len(prims), "placement": {"scale": round(pl["scale"], 6), "yaw": pl["yaw_deg"],
                                                                        "pivot": P["pivot"], "size_mode": P["size_mode"]}},
                       srep, lod_rows, col_rep, tex_manifest, man["license"], warnings,
                       {**tim, "total": round(time.time() - T0, 2), "peak_rss_mb": _rss_mb()})
        rep["lods_skipped"] = chain["skipped"]
        rep["style"] = style
        rep["package"] = f"engine_{target}.zip"
        files["manifest.json"] = json.dumps(man, indent=1, default=str).encode()
        files["engine_report.json"] = json.dumps(rep, indent=1, default=str).encode()
        files["README.txt"] = C.readme(target, names["stem"], glb_mode).encode()
        zpath = out / f"engine_{target}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for arc, data in files.items():
                zi = zipfile.ZipInfo(arc, time.localtime()[:6])
                zi.compress_type = zipfile.ZIP_STORED if arc.endswith((".png", ".glb")) else zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                z.writestr(zi, data)
        (out / "engine_report.json").write_bytes(files["engine_report.json"])
        tim["total"] = round(time.time() - T0, 2)
        attach = [zpath.name, "engine_report.json"] + (["engine_collision.glb"] if coll_glb else [])
        summary = {"target": target, "name": names["stem"], "rigged": glb_mode, "lods": [{k: r[k] for k in ("name", "tris", "ratio", "error", "screen_size")} for r in lod_rows],
                   "hulls": col_rep["hulls"], "collision": col_rep["method"], "volume_ratio": col_rep.get("volume_ratio"),
                   "package": zpath.name, "package_bytes": zpath.stat().st_size, "files": sorted(files), "textures": len(tex_rows),
                   "images_resized": n_resized, "style": style}
        ctx.progress(100, 100, "done", f"{len(levels)} LOD level(s), {col_rep['hulls']} hull(s), {len(files)} files")
        ctx.output(out / "mesh.glb", "mesh", name=f"{display} · {target} package", attach=attach, textured=bool(tex_rows),
                   engine=summary, stats={"time": tim, "total": tim["total"]}, warnings=warnings)
        return {"meta": {"timings": tim, "warnings": warnings}}

    # -------------------------------------------------------------- static FBX (Unreal, Unity)
    def _write_fbx(self, path, target, names, levels, hulls, pl, tex_by_mat, mat_info):
        sc = FW.Scene(up="Y", unit_cm_per_unit=1.0)
        fmat = {}
        for k, (mi, tx) in enumerate(tex_by_mat.items()):
            slots = {}
            if "basecolor" in tx:
                slots["DiffuseColor"] = f"Textures/{tx['basecolor']}"
            if "normal" in tx:
                slots["NormalMap"] = f"Textures/{tx['normal']}"
            nm = names["material"] if len(tex_by_mat) == 1 else f"{names['material']}_{k}"
            fac = mat_info.get(mi, {})
            fmat[mi] = sc.material(nm, slots, tuple(fac.get("base_color", [0.8, 0.8, 0.8])[:3]))
        if None in {p["material"] for lv in levels for p in lv} and None not in fmat:
            fmat[None] = sc.material(f"{names['material']}_default", {})
        grp = sc.group(names["group"], 0, lod_levels=len(levels)) if names["group"] else 0
        for i, level in enumerate(levels):
            m = C.merge_prims(level)
            V = C.apply_placement(m["V"], pl)
            Nn = C.apply_placement(m["N"], pl, normals=True)
            mats = [fmat[x] for x in m["materials"] if x in fmat]
            sc.mesh(names["lods"][i], V, m["F"], Nn, m["UV"], materials=mats, mat_per_face=m["mat"] if len(mats) > 1 else None,
                    parent=grp, scale=C.FBX_UNIT_SCALE)
        for k, (V, F, kind) in enumerate(hulls):
            sc.mesh(names["hull"](k, kind), C.apply_placement(V, pl), F, None, None, None, parent=0, scale=C.FBX_UNIT_SCALE)
        sc.save(str(path))


if __name__ == "__main__":
    serve(EngineBackend())
