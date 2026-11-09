"""Retopo + bake backend (envs/retopo, CPU only, no weights): dense mesh -> quad-dominant low poly with UVs and baked PBR maps.

  task retopo   input role `mesh` (a GLB, not rigged)  ->  mesh asset `mesh.glb` (POSITION NORMAL TANGENT TEXCOORD_0, quad pairs first,
                primitive extras {"oas_quad_pairs": n}, material with baseColor / normal / ORM textures) + attachments
                `retopo_quads.obj` + `.mtl` (real quads), `retopo_basecolor.png`, `retopo_normal.png`, `retopo_orm.png`, `retopo_report.json`

Pipeline (docs/research/retopo.md, spec in the commit message): load + clean -> Instant Meshes (PyNanoInstantMeshes, patched for
quad-dominant output) -> repair (non-manifold faces, fragments, small holes) -> xatlas UV unwrap -> MikkTSpace tangents -> UV raster
through Embree -> cage distance -> rays along +-N to the high mesh -> tangent-space normal map, base colour + metallic / roughness from
the colour source, AO -> GLB. The pure maths lives in workers/retopo_common.py (imported by the tests in envs/core).

The high detail source is the `white_mesh.glb` attachment of the input asset when it is denser than the main GLB and its bounding box
matches (Hunyuan3D keeps the 548k-triangle shape there and a 60k textured copy as mesh.glb); the colour always comes from the main file.
The bake runs on texel chunks of 2^20, so the peak memory does not grow with the texture size.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import m3d_common as C3
import retopo_common as C
from studio_worker import Backend, InputError, serve

CHUNK = 1 << 20                  # texels per bake chunk
MAX_FACES = 20_000_000
PREDECIMATE_ABOVE = 2_500_000
PREDECIMATE_TO = 2_000_000
HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ open3d helpers
def _scene(v, f):
    import open3d as o3d
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.ascontiguousarray(v, np.float32)), o3d.core.Tensor(np.ascontiguousarray(f, np.uint32)))
    return s


def _tensor(a):
    import open3d as o3d
    return o3d.core.Tensor(np.ascontiguousarray(a, np.float32))


def _cast(scene, o, d):
    r = scene.cast_rays(_tensor(np.c_[o, d]))
    return r["t_hit"].numpy(), r["primitive_ids"].numpy().astype(np.int64), r["primitive_uvs"].numpy().astype(np.float64)


def _closest(scene, p):
    r = scene.compute_closest_points(_tensor(p))
    return r["points"].numpy(), r["primitive_ids"].numpy().astype(np.int64), r["primitive_uvs"].numpy().astype(np.float64)


def _occluded(scene, o, d, tfar):
    return scene.test_occlusions(_tensor(np.c_[o, d]), tnear=0.0, tfar=float(tfar)).numpy()


# ------------------------------------------------------------------ mesh loading
class Geom:
    """A GLB flattened to one triangle soup (node transforms applied): V (n,3), F (m,3), prim (m,) primitive index of every face,
    uv (n,2) glTF UVs (v down) or None, vcol (n,4) linear COLOR_0 in 0..1 or None, mats[prim] material info."""

    def __init__(self, path: Path):
        import trimesh
        try:
            sc = trimesh.load(str(path), force="scene", process=False)
            parts = [g for g in sc.dump() if len(getattr(g, "faces", [])) > 0]
        except Exception as e:  # noqa: BLE001
            raise InputError("Could not read this file as a 3D model. Re-export it as GLB (glTF binary) and try again.") from e
        if not parts:
            raise InputError("Could not read this file as a 3D model. Re-export it as GLB (glTF binary) and try again.")
        self.mats, vs, fs, prims, uvs, vcols = [], [], [], [], [], []
        off = 0
        cache = {}
        for i, g in enumerate(parts):
            vs.append(np.asarray(g.vertices, np.float64))
            fs.append(np.asarray(g.faces, np.int64) + off)
            prims.append(np.full(len(g.faces), i, np.int64))
            vis = g.visual
            mat = {"base": None, "mr": None, "factor": np.ones(4), "rough": None, "metal": None, "alpha_mode": None}
            uv = vc = None
            kind = getattr(vis, "kind", None)
            if kind == "texture" and getattr(vis, "uv", None) is not None:
                uv = np.asarray(vis.uv, np.float64).copy()
                uv[:, 1] = 1.0 - uv[:, 1]                     # trimesh flips glTF v on load: back to glTF (v down)
                pm = vis.material
                for key, attr, mode in (("base", "baseColorTexture", "RGBA"), ("mr", "metallicRoughnessTexture", "RGB")):
                    img = getattr(pm, attr, None)
                    if img is not None:
                        k = (id(img), mode)
                        if k not in cache:
                            cache[k] = np.asarray(img.convert(mode))
                        mat[key] = cache[k]
                if getattr(pm, "baseColorFactor", None) is not None:
                    mat["factor"] = np.asarray(pm.baseColorFactor, np.float64)[:4] / 255.0
                mat["rough"] = getattr(pm, "roughnessFactor", None)
                mat["metal"] = getattr(pm, "metallicFactor", None)
                am = getattr(pm, "alphaMode", None)
                mat["alpha_mode"] = am if am in ("MASK", "BLEND") else None
            elif kind == "vertex":
                vc = np.asarray(vis.vertex_colors, np.float64)[:, :4] / 255.0
            uvs.append(uv if uv is not None else np.full((len(g.vertices), 2), np.nan))
            vcols.append(vc if vc is not None else np.full((len(g.vertices), 4), np.nan))
            self.mats.append(mat)
            off += len(g.vertices)
        self.V, self.F, self.prim = np.concatenate(vs), np.concatenate(fs), np.concatenate(prims)
        self.uv, self.vcol = np.concatenate(uvs), np.concatenate(vcols)
        self.has_tex = any(m["base"] is not None for m in self.mats)
        self.has_vcol = bool(np.isfinite(self.vcol).any()) and not self.has_tex

    def bounds(self):
        return self.V[np.unique(self.F)].min(0), self.V[np.unique(self.F)].max(0)

    def cleaned(self):
        """Welded copy for the shape: (V, F, face_map). Vertices are merged by position (so vertex normals are smooth across UV
        seams), degenerate and duplicate faces dropped; face_map[k] = the index of the raw face behind cleaned face k."""
        _, first, inv = np.unique(np.round(self.V / max(1e-12, np.ptp(self.V, axis=0).max()) * 1e7).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
        Vm = self.V[first]
        Fm = inv.reshape(-1)[self.F]
        a, b, c = Fm.T
        good = (a != b) & (b != c) & (a != c)
        area = np.linalg.norm(np.cross(Vm[Fm[:, 1]] - Vm[Fm[:, 0]], Vm[Fm[:, 2]] - Vm[Fm[:, 0]]), axis=1)
        good &= area > 1e-14 * max(1e-12, float(np.ptp(Vm, axis=0).max())) ** 2
        idx = np.nonzero(good)[0]
        key = np.sort(Fm[idx], axis=1)
        _, uniq = np.unique(key, axis=0, return_index=True)
        idx = idx[np.sort(uniq)]
        used = np.zeros(len(Vm), bool)
        used[Fm[idx].ravel()] = True
        remap = -np.ones(len(Vm), np.int64)
        remap[used] = np.arange(used.sum())
        return Vm[used], remap[Fm[idx]], idx


class RetopoBackend(Backend):
    name = "retopo"
    max_loaded = 1

    def probe(self, model):
        import fast_simplification  # noqa: F401
        import numpy  # noqa: F401
        import open3d  # noqa: F401
        import PIL  # noqa: F401
        import pynanoinstantmeshes as nim
        import scipy  # noqa: F401
        import trimesh  # noqa: F401
        import xatlas  # noqa: F401
        if "pure_quad" not in (nim.remesh.__doc__ or ""):
            return False, ("PyNanoInstantMeshes was installed without the pure_quad switch (the PyPI wheel always makes pure quads). "
                           "Run: ./install.sh backend retopo --force")
        return True, ""

    def load(self, model, ctx):
        import open3d  # noqa: F401
        import pynanoinstantmeshes  # noqa: F401
        import trimesh  # noqa: F401
        import xatlas  # noqa: F401
        return {"ready": True}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        if job.task != "retopo":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        T0 = time.time()
        tim: dict[str, float] = {}
        stamp = [T0]

        def lap(key):
            now = time.time()
            tim[key] = round(now - stamp[0], 2)
            stamp[0] = now
            ctx.check()

        P = _params(job.params)
        meshes = [i for i in job.inputs if i.get("type") == "mesh"]
        if not meshes:
            raise InputError(C.E1)
        if len(meshes) > 1:
            ctx.log(f"warning: {len(meshes)} meshes given, only the first one is used")
        inp = job.input(role="mesh") or meshes[0]
        src = Path(inp["path"])
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------- 1. inputs
        ctx.progress(0, 100, "load", "Reading the model")
        try:
            flags = C3.glb_flags(src)
        except Exception as e:  # noqa: BLE001
            raise InputError("Could not read this file as a 3D model. Re-export it as GLB (glTF binary) and try again.") from e
        if flags["skins"]:
            raise InputError(C.E3)
        if flags["animations"] or flags["morph_targets"]:
            raise InputError(C.E4)
        if flags["triangles"] <= 0:
            raise InputError("Could not read this file as a 3D model. Re-export it as GLB (glTF binary) and try again.")
        colour_geom = Geom(src)
        high_geom, high_file, white_rejected = colour_geom, src.name, False
        white = src.parent / "white_mesh.glb"
        if P["use_shape_mesh"] and white.is_file() and white != src:
            try:
                wf = C3.glb_flags(white)
                if wf["triangles"] >= 1.5 * flags["triangles"] and not wf["skins"]:
                    wg = Geom(white)
                    if C.bbox_match(*colour_geom.bounds(), *wg.bounds()):
                        high_geom, high_file = wg, white.name
                    else:
                        white_rejected = True
                        del wg
                else:
                    white_rejected = False       # a stored mesh that is not denser is simply not useful, not a mismatch
            except Exception as e:  # noqa: BLE001
                ctx.log(f"white_mesh.glb ignored: {e}")
        same = high_geom is colour_geom
        HV, HF, fmap = high_geom.cleaned()
        if len(HF) == 0:
            raise InputError("Could not read this file as a 3D model. Re-export it as GLB (glTF binary) and try again.")
        HN = C.vertex_normals(HV, HF)
        ext = HV.max(0) - HV.min(0)
        diag = float(np.linalg.norm(ext))
        F_hi = len(HF)

        # ---------------------------------------------------------------- 2. size guards
        target = P["target_faces"]
        if F_hi > MAX_FACES:
            raise InputError(f"This model has {F_hi / 1e6:.1f} million faces; the limit is 20 million. Reduce it first in Export "
                             "(Decimate, target 2,000,000) and retopologise that.")
        if F_hi < 2 * 1.9 * target:
            raise InputError(f"The model has only {F_hi:,} triangles, fewer than twice the target ({target:,} faces ≈ {int(1.9 * target):,} "
                             "triangles). Lower the target face count; this model may already be light enough to use as it is.")
        remesh_v, remesh_f, predec = HV, HF, None
        if F_hi > PREDECIMATE_ABOVE:
            import fast_simplification
            remesh_v, remesh_f = fast_simplification.simplify(HV.astype(np.float32), HF.astype(np.int32),
                                                              target_reduction=1.0 - PREDECIMATE_TO / F_hi)
            predec = F_hi
            ctx.log(f"input reduced {F_hi:,} -> {len(remesh_f):,} faces for the remesh only")
        hs = _scene(HV, HF)
        lap("load")

        # ---------------------------------------------------------------- 3. remesh
        ctx.progress(8, 100, "remesh", "Instant Meshes (native step, can't be cancelled)", interruptible=False)
        nv, nf = self._remesh(remesh_v, remesh_f, target, P)
        ctx.progress(30, 100, "remesh", "Remeshed", interruptible=True)
        lap("remesh")

        # ---------------------------------------------------------------- 4. repair
        ctx.progress(35, 100, "repair", "Repairing the new mesh")
        quads, tris = C.split_polys(nf)
        raw = C.poly_stats(quads, tris)
        nv, quads, tris, rep = C.clean_polys(nv, quads, tris, min_component_faces=24, keep_components_frac=P["min_part"],
                                             max_hole_edges=P["fill_holes"])
        if len(quads) + len(tris) == 0:
            raise InputError("Nothing usable is left after removing broken faces: the model has no solid surface to retopologise.")
        if rep["hole_vertices"]:
            p, _, _ = _closest(hs, nv[rep["hole_vertices"]])
            nv[rep["hole_vertices"]] = p
        stats = C.poly_stats(quads, tris)
        lap("repair")

        # ---------------------------------------------------------------- 5. triangulate
        LT, nq = C.triangulate(quads, tris, nv)
        LN = C.vertex_normals(nv, LT)

        # ---------------------------------------------------------------- 6. UV unwrap
        ctx.progress(40, 100, "uv", "Unwrapping UVs (xatlas)")
        import xatlas
        try:
            atlas = xatlas.Atlas()
            atlas.add_mesh(nv.astype(np.float32), LT.astype(np.uint32), LN.astype(np.float32))
            po = xatlas.PackOptions()
            po.resolution = P["tex_size"]
            po.padding = max(2, P["tex_size"] // 256)
            po.bilinear = True
            po.blockAlign = True
            atlas.generate(xatlas.ChartOptions(), po)
            vmap, UF, UV = atlas.get_mesh(0)
            vmap, UF, UV = vmap.astype(np.int64), UF.astype(np.int64), UV.astype(np.float64)
            if not (len(UF) == len(LT) and np.all(vmap[UF] == LT)):
                raise RuntimeError("xatlas changed the face order")
            util = float(atlas.utilization)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"UV unwrap failed (xatlas): {e}") from e
        V, N = nv[vmap], LN[vmap]
        lap("uv")

        # ---------------------------------------------------------------- 7. tangents
        ctx.progress(50, 100, "tangents", "Computing tangents")
        V, N, UV, TAN, UF, orig, tan_kind = _tangents(V, N, UV, UF, vmap)
        lap("tangents")

        # ---------------------------------------------------------------- 8. texel raster
        want_hits = P["bake_normal"] or P["bake_color"] or P["bake_ao"]
        res = P["tex_size"]
        report_extra: dict = {}
        maps: dict[str, bytes] = {}
        files: dict[str, bytes] = {}
        mask = np.zeros((res, res), bool)
        cage_frac = miss_pct = dist_mean = dist_p99 = alpha_mode = None
        colour_info = {"file": src.name, "kind": "none", "size": 0}
        if want_hits:
            ctx.progress(55, 100, "raster", "Finding the texels of the UV layout")
            idx, tpid, tpuv = _raster(UV, UF, res, ctx)
            mask.reshape(-1)[idx] = True
            lap("raster")

            # ------------------------------------------------------------ 9. cage
            ctx.progress(58, 100, "cage", "Measuring the ray distance")
            import trimesh
            ls = _scene(V, UF)
            samp = trimesh.sample.sample_surface(trimesh.Trimesh(HV, HF, process=False), 60000, seed=1)[0]
            d = ls.compute_distance(_tensor(samp)).numpy()
            dist_mean, dist_p99 = float(d.mean() / diag), float(np.percentile(d, 99) / diag)
            cage = P["ray_distance"] * diag if P["ray_distance"] > 0 else C.auto_cage(d, diag)
            cage_frac = cage / diag
            del ls, samp, d
            lap("cage")

            # ------------------------------------------------------------ 10. bake rays (high triangle + barycentrics per texel)
            ctx.progress(60, 100, "normal", "Casting bake rays")
            hp = np.zeros(len(idx), np.int32)
            hu = np.zeros((len(idx), 2), np.float32)
            misses = 0
            for sl in _chunks(len(idx)):
                _, _, _, _, Nn = _texel_frame(V, N, TAN, UF, tpid[sl], tpuv[sl])
                Pw = C.bary_interp(V, UF, tpid[sl], tpuv[sl])
                p_, u_, m_ = _rays(hs, Pw, Nn, cage)
                hp[sl], hu[sl] = p_, u_
                misses += int(m_.sum())
                ctx.check()
            miss_pct = round(100.0 * misses / max(1, len(idx)), 2)
            ctx.log(f"{len(idx):,} texels, cage {cage_frac:.4f} x diagonal, {miss_pct}% ray misses")

            # ------------------------------------------------------------ 11. normal map
            if P["bake_normal"]:
                img = np.empty((res * res, 3), np.uint8)
                img[:] = (128, 128, 255)
                for sl in _chunks(len(idx)):
                    Pw, Ni, Ti, wi, _ = _texel_frame(V, N, TAN, UF, tpid[sl], tpuv[sl])
                    hn = C.bary_interp(HN, HF, hp[sl].astype(np.int64), hu[sl].astype(np.float64))
                    hn /= np.maximum(np.linalg.norm(hn, axis=1, keepdims=True), 1e-20)
                    img[idx[sl]] = C.to_rgb8(C.encode_ts_normal(hn, Ni, Ti, wi))
                    ctx.check()
                maps["normal"] = C.dilate(img.reshape(res, res, 3), mask)
                del img
            lap("normal")

            # ------------------------------------------------------------ 12. base colour + metallic / roughness
            ctx.progress(70, 100, "colour", "Baking colour")
            rough = np.full(len(idx), 0.8, np.float32)
            metal = np.zeros(len(idx), np.float32)
            if P["bake_color"]:
                colour_info = _colour_info(colour_geom, src)
                cs = None if same else _scene(colour_geom.V, colour_geom.F)
                base_img = np.zeros((res * res, 4), np.uint8) if colour_info["kind"] != "none" else None
                alpha_mode = next((m["alpha_mode"] for m in colour_geom.mats if m["alpha_mode"]), None)
                for sl in (_chunks(len(idx)) if colour_info["kind"] != "none" else []):
                    if same:
                        cpid, cuv = fmap[hp[sl].astype(np.int64)], hu[sl].astype(np.float64)
                    else:
                        Pw, _, _, _, Nn = _texel_frame(V, N, TAN, UF, tpid[sl], tpuv[sl])
                        cpid, cuv, _ = _rays(cs, Pw, Nn, cage)
                    col, r_, m_ = _sample_colour(colour_geom, cpid, cuv)
                    if base_img is not None:
                        base_img[idx[sl]] = np.clip(np.round(col * 255.0), 0, 255).astype(np.uint8)
                    rough[sl], metal[sl] = r_, m_
                    ctx.check()
                if base_img is not None:
                    bi = C.dilate(base_img.reshape(res, res, 4), mask)
                    if alpha_mode is None and bi[..., 3].min() == 255:
                        bi = np.ascontiguousarray(bi[..., :3])
                    maps["basecolor"] = bi
                del base_img, cs
            lap("colour")

            # ------------------------------------------------------------ 14. AO (+ ORM)
            ao = np.ones(len(idx), np.float32)
            if P["bake_ao"]:
                ctx.progress(80, 100, "ao", "Baking ambient occlusion")
                rng = np.random.default_rng(0)
                maxd = P["ao_distance"] * diag
                for sl in _chunks(len(idx)):
                    hn = C.bary_interp(HN, HF, hp[sl].astype(np.int64), hu[sl].astype(np.float64))
                    hn /= np.maximum(np.linalg.norm(hn, axis=1, keepdims=True), 1e-20)
                    org = C.bary_interp(HV, HF, hp[sl].astype(np.int64), hu[sl].astype(np.float64)) + hn * (1e-4 * diag)
                    occ = np.zeros(len(hn), np.float32)
                    for _ in range(P["ao_rays"]):
                        dirs = C.ao_directions(hn, 1, rng)[0]
                        occ += _occluded(hs, org, dirs, maxd)
                        ctx.check()
                    ao[sl] = 1.0 - occ / P["ao_rays"]
            if P["bake_ao"] or P["bake_color"]:
                orm = np.zeros((res * res, 3), np.uint8)
                orm[idx, 0] = np.clip(np.round(ao * 255), 0, 255).astype(np.uint8)
                orm[idx, 1] = np.clip(np.round(rough * 255), 0, 255).astype(np.uint8)
                orm[idx, 2] = np.clip(np.round(metal * 255), 0, 255).astype(np.uint8)
                maps["orm"] = C.dilate(orm.reshape(res, res, 3), mask)
                del orm
            lap("ao")
            del hp, hu, tpid, tpuv

        # ---------------------------------------------------------------- 15. write
        ctx.progress(94, 100, "write", "Writing the model")
        png = _encode_pngs(maps)
        for k, b in png.items():
            (out / f"retopo_{k}.png").write_bytes(b)
        mat = {"name": "retopo", "double_sided": False,
               "base_color_image": "basecolor" if "basecolor" in png else None,
               "base_color_factor": [1, 1, 1, 1] if "basecolor" in png else [0.8, 0.8, 0.8, 1.0],
               "orm_image": "orm" if "orm" in png else None, "normal_image": "normal" if "normal" in png else None,
               "metallic_factor": 1.0 if "orm" in png else 0.0, "roughness_factor": 1.0 if "orm" in png else 0.8,
               "alpha_mode": alpha_mode if want_hits and P["bake_color"] else None}
        C.pack_glb(out / "mesh.glb", V, N, TAN, UV, UF, png, mat, {"oas_quad_pairs": int(nq)})
        # real quads for DCC tools: the repaired positions, the xatlas UVs per corner
        nqd = int(nq)
        qcorner = np.c_[UF[0:2 * nqd:2, 0], UF[0:2 * nqd:2, 1], UF[0:2 * nqd:2, 2], UF[1:2 * nqd:2, 2]]
        polys = [orig[r].tolist() for r in qcorner] + [orig[r].tolist() for r in UF[2 * nqd:]]
        poly_uv = [r.tolist() for r in qcorner] + [r.tolist() for r in UF[2 * nqd:]]
        C.write_obj(out / "retopo_quads.obj", nv, polys, UV, poly_uv, mtl_name="retopo_quads.mtl")
        C.write_mtl(out / "retopo_quads.mtl", "retopo", "retopo_basecolor.png" if "basecolor" in png else None,
                    "retopo_normal.png" if "normal" in png else None)

        report = {"target_faces": target, "mode": P["mode"], "faces": stats["faces"], "quads": stats["quads"], "tris": stats["tris"],
                  "quad_pct": stats["quad_pct"], "triangles": int(len(UF)), "vertices": int(len(V)),
                  "boundary_edges": stats["boundary_edges"], "open_holes": rep["open_holes"], "components": stats["components"],
                  "irregular_vert_pct": stats["irregular_vert_pct"],
                  "repair": {k: v for k, v in rep.items() if k not in ("hole_vertices", "open_holes")},
                  "raw": {k: raw[k] for k in ("faces", "boundary_edges", "nonmanifold_edges", "components")},
                  "high": {"file": high_file, "faces": int(F_hi)}, "colour_source": colour_info,
                  "tex_size": res, "maps": list(png), "ao_rays": P["ao_rays"] if P["bake_ao"] else 0,
                  "cage_frac_diag": None if cage_frac is None else round(cage_frac, 5), "ray_miss_pct": miss_pct,
                  "dist_mean_frac": None if dist_mean is None else round(dist_mean, 6),
                  "dist_p99_frac": None if dist_p99 is None else round(dist_p99, 5),
                  "uv_utilization": round(util, 3), "texel_coverage": round(float(mask.mean()), 3), "tangents": tan_kind,
                  "quad_pairs": nqd, "white_rejected": white_rejected, "predecimated_from": predec}
        if miss_pct is None:
            report["ray_miss_pct"] = 0.0
        warnings = C.report_warnings(report, P)
        report["warnings"] = warnings
        lap("write")
        tim["total"] = round(time.time() - T0, 2)
        report["timings"] = tim
        (out / "retopo_report.json").write_text(json.dumps(report, indent=1))
        for w in warnings:
            ctx.log("warning: " + w)
        ctx.progress(100, 100, "done", f"{stats['faces']:,} faces, {stats['quad_pct']}% quads")
        attach = ["retopo_quads.obj", "retopo_quads.mtl"] + [f"retopo_{k}.png" for k in png] + ["retopo_report.json"]
        name = f"{(inp.get('name') or 'mesh')[:50]} · retopo {stats['faces'] / 1000:.1f}k"
        ctx.output(out / "mesh.glb", "mesh", name=name, attach=attach, textured=True,
                   retopo={k: v for k, v in report.items() if k not in ("warnings", "timings", "raw")},
                   stats={"time": tim, "total": tim["total"]}, warnings=warnings)
        return {"meta": {"timings": tim, "warnings": warnings}}

    # ------------------------------------------------------------------ Instant Meshes with one face-count correction pass
    def _remesh(self, V, F, target, P):
        import pynanoinstantmeshes as nim
        pure = P["mode"] == "pure"
        req = max(16, target // 4 if pure else target)
        V32, F32 = np.ascontiguousarray(V, np.float32), np.ascontiguousarray(F, np.uint32)
        nv = nf = None
        try:
            for attempt in range(2):
                nv, nf = nim.remesh(V32, F32, -1, face_count=int(req), rosy=4, posy=4, creaseAngle=float(P["crease_angle"]),
                                    align_to_boundaries=True, extrinsic=True, smooth_iter=int(P["smooth_iter"]), deterministic=True,
                                    pure_quad=pure)
                nv, nf = np.asarray(nv, np.float64), np.asarray(nf)
                if len(nf) == 0:
                    break
                got = len(nf)                              # pure mode: the extraction already subdivides, `req` was target / 4
                nxt = C.next_request(req, target, got)
                if nxt is None or attempt == 1:
                    break
                req = nxt
        except Exception as e:  # noqa: BLE001
            raise InputError(E7) from e
        if nv is None or len(nf) == 0 or len(nv) == 0:
            raise InputError(E7)
        return nv, nf


E7 = ("Instant Meshes could not remesh this shape. This usually means it has no connected surface (a point cloud, or many loose pieces). "
      "Try Export › Remove floaters first, or a higher face target.")


# ------------------------------------------------------------------ helpers
def _params(p: dict) -> dict:
    g = lambda k, d: d if p.get(k) is None else p[k]  # noqa: E731
    return {"target_faces": int(g("target_faces", 12000)), "tex_size": int(g("tex_size", 2048)),
            "bake_normal": bool(g("bake_normal", True)), "bake_color": bool(g("bake_color", True)), "bake_ao": bool(g("bake_ao", True)),
            "mode": "pure" if g("quad_mode", "quad_dominant") == "pure_quads" else "quad",
            "crease_angle": float(g("crease_angle", 0)), "smooth_iter": int(g("smooth_iter", 2)),
            "use_shape_mesh": bool(g("use_shape_mesh", True)), "fill_holes": int(g("fill_holes", 64)),
            "min_part": float(g("min_part", 0.01)), "ray_distance": float(g("ray_distance", 0)),
            "ao_rays": int(g("ao_rays", 16)), "ao_distance": float(g("ao_distance", 0.08))}


def _chunks(n: int):
    return [slice(i, min(n, i + CHUNK)) for i in range(0, n, CHUNK)]


def _tangents(V, N, UV, UF, vmap):
    """MikkTSpace on unindexed corners fed (u, 1 - v), re-welded by (vertex, tangent). Fallback: Lengyel. Returns
    V, N, UV, TAN (n,4), UF, orig (final vertex -> index in the repaired position list), kind."""
    try:
        import mikktspace
        P = V[UF].reshape(-1).astype(np.float32)
        NN = N[UF].reshape(-1).astype(np.float32)
        UU = np.c_[UV[:, 0], 1.0 - UV[:, 1]][UF].reshape(-1).astype(np.float32)
        Tc = np.frombuffer(mikktspace.gen_default(list(range(3 * len(UF))), P, NN, UU), np.float32).reshape(-1, 4)
        key = np.c_[UF.reshape(-1), np.round(Tc * 1e4).astype(np.int64)]
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        corner_v = UF.reshape(-1)[first]
        return (V[corner_v], N[corner_v], UV[corner_v], Tc[first].astype(np.float64), inv.reshape(-1, 3).astype(np.int64),
                vmap[corner_v], "mikktspace")
    except ImportError:
        return V, N, UV, C.tangents_lengyel(V, N, UV, UF), UF, vmap, "lengyel"


def _raster(UV, UF, res, ctx):
    """UV layout as a flat mesh at z=0, one ray per texel centre from z=1: -> covered texel indices, low triangle, barycentrics."""
    uvs = _scene(np.c_[UV, np.zeros(len(UV))], UF)
    rows = max(1, CHUNK // res)
    c = (np.arange(res) + 0.5) / res
    idx, pid, puv = [], [], []
    for r0 in range(0, res, rows):
        r1 = min(res, r0 + rows)
        u, v = np.meshgrid(c, (np.arange(r0, r1) + 0.5) / res)
        o = np.c_[u.ravel(), v.ravel(), np.ones(u.size)]
        th, p, uv_ = _cast(uvs, o, np.tile([0.0, 0.0, -1.0], (len(o), 1)))
        cov = np.isfinite(th)
        idx.append((np.nonzero(cov)[0] + r0 * res).astype(np.int64))
        pid.append(p[cov].astype(np.int32))
        puv.append(uv_[cov].astype(np.float32))
        ctx.check()
    return np.concatenate(idx), np.concatenate(pid).astype(np.int64), np.concatenate(puv).astype(np.float64)


def _texel_frame(V, N, TAN, UF, pid, puv):
    """Per covered texel: world position, the interpolated (unnormalised) normal and tangent, the handedness w, unit normal."""
    Pw = C.bary_interp(V, UF, pid, puv)
    Ni = C.bary_interp(N, UF, pid, puv)
    Ti = C.bary_interp(TAN[:, :3], UF, pid, puv)
    wi = TAN[UF[pid, 0], 3]
    Nn = Ni / np.maximum(np.linalg.norm(Ni, axis=1, keepdims=True), 1e-20)
    return Pw, Ni, Ti, wi, Nn


def _rays(scene, Pw, Nn, cage):
    """Nearest hit along +-n within the cage, misses -> closest point. Returns (triangle, barycentrics, miss mask)."""
    tf, pf, uf = _cast(scene, Pw, Nn)
    tb, pb, ub = _cast(scene, Pw, -Nn)
    use_f, hit = C.choose_hit(tf, tb, cage)
    pid = np.where(use_f, pf, pb)
    uv = np.where(use_f[:, None], uf, ub)
    miss = ~hit
    if miss.any():
        _, cp, cu = _closest(scene, Pw[miss])
        pid[miss], uv[miss] = cp, cu
    return pid, uv, miss


def _colour_info(g: Geom, src: Path) -> dict:
    if g.has_tex:
        size = max(max(m["base"].shape[:2]) for m in g.mats if m["base"] is not None)
        return {"file": src.name, "kind": "texture", "size": int(size)}
    if g.has_vcol:
        return {"file": src.name, "kind": "vertex", "size": 0}
    return {"file": src.name, "kind": "none", "size": 0}


def _sample_colour(g: Geom, cpid, cuv):
    """Base colour (n,4) sRGB 0..1, roughness (n,), metallic (n,) of the colour source at the hits (face, barycentrics)."""
    n = len(cpid)
    col = np.ones((n, 4), np.float64)
    rough = np.full(n, 0.8, np.float32)
    metal = np.zeros(n, np.float32)
    prim = g.prim[cpid]
    for p in np.unique(prim):
        sel = np.nonzero(prim == p)[0]
        m = g.mats[p]
        f, u = cpid[sel], cuv[sel]
        if m["base"] is not None:
            suv = C.bary_interp(g.uv, g.F, f, u)
            col[sel] = C.bilinear(m["base"], suv) / 255.0 * m["factor"]
            if m["mr"] is not None:
                mr = C.bilinear(m["mr"], suv) / 255.0
                rough[sel] = mr[:, 1] * (1.0 if m["rough"] is None else m["rough"])
                metal[sel] = mr[:, 2] * (1.0 if m["metal"] is None else m["metal"])
            else:
                rough[sel] = 0.8 if m["rough"] is None else m["rough"]
                metal[sel] = 0.0 if m["metal"] is None else m["metal"]
        elif g.has_vcol and np.isfinite(g.vcol[g.F[f[0], 0]]).all():
            lin = C.bary_interp(g.vcol, g.F, f, u)
            col[sel, :3] = C.linear_to_srgb(np.clip(lin[:, :3], 0, 1))        # COLOR_0 is linear in glTF
            col[sel, 3] = lin[:, 3]
        else:
            col[sel] = m["factor"] if np.any(m["factor"] != 1.0) else (0.8, 0.8, 0.8, 1.0)
    return np.clip(col, 0.0, 1.0), rough, metal


def _encode_pngs(maps: dict) -> dict:
    from PIL import Image

    def enc(a):
        b = io.BytesIO()
        Image.fromarray(a).save(b, "PNG", compress_level=4)
        return b.getvalue()

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {k: ex.submit(enc, a) for k, a in maps.items()}
        return {k: f.result() for k, f in futs.items()}


if __name__ == "__main__":
    serve(RetopoBackend())
