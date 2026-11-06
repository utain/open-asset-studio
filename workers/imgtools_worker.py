"""Image + mesh tools worker (runs in envs/hy3d, read-only): one process for the light tools.

  rembg-u2net / rembg-birefnet   task rmbg       background removal (CPU onnxruntime)
  realesrgan-x4                  task upscale    Real-ESRGAN x4plus, x2 / x4, tiled, alpha kept      (GPU, ~2 GB)
  recenter-3d                    task recenter   crop to the object, centre on a square with a margin, resize
  mesh-export                    task mesh_export  floaters / degenerate / decimate / smooth / normals; GLB OBJ PLY STL

Reused, not forked (imported from engines/hy3d/app; nothing under engines/hy3d is edited):
  imgtools.py     Upscaler (sets RealESRGANer.tile_size, the attribute it really reads), alpha bleed under transparent
                  pixels, clean_alpha (Qwen haze), crop_pad_square, object_bbox, flatten, open_checked
  postprocess.py  run_pipeline (pymeshlab quadric decimation, Taubin smoothing ...), export_mesh, mesh_stats
The "RGB flattening" fix: an RGBA input is composited on white before rembg (the colours under transparent pixels are
garbage, e.g. Qwen's purple), exactly like engines/hy3d/app/studio.py does.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the venv is read-only for the studio: no __pycache__ writes into engines/hy3d

import json
import os
import shutil
import struct
import time
import zipfile
from pathlib import Path

from studio_worker import Backend, InputError, serve

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
HY3D_ENGINE = Path(os.environ.get("STUDIO_HY3D_ENGINE") or STUDIO / "engines" / "hy3d")
APP = HY3D_ENGINE / "app"
if str(APP) not in sys.path:
    sys.path.append(str(APP))
ESRGAN = HY3D_ENGINE / "vendor" / "Hunyuan3D-2.1" / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
REMBG_SESSIONS = {"rembg-u2net": "u2net", "rembg-birefnet": "birefnet-general"}


def _tv_shim():
    """basicsr 1.4.2 imports torchvision.transforms.functional_tensor (removed in torchvision 0.17+)."""
    import torchvision.transforms.functional as F
    sys.modules.setdefault("torchvision.transforms.functional_tensor", F)


def glb_has_texture(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(20)
            if head[:4] != b"glTF":
                return False
            clen = struct.unpack("<I", head[12:16])[0]
            doc = json.loads(f.read(clen))
        return bool(doc.get("textures") or doc.get("images"))
    except (OSError, ValueError, struct.error):
        return False


class ToolsBackend(Backend):
    name = "imgtools"
    max_loaded = 4   # tiny models: rembg session(s), Real-ESRGAN, the CPU tools

    def probe(self, model):
        mid = model["id"]
        if mid.startswith("rembg"):
            import rembg  # noqa: F401
        elif mid == "realesrgan-x4":
            _tv_shim()
            import basicsr  # noqa: F401
            import realesrgan  # noqa: F401
            if not ESRGAN.exists():
                return False, f"missing {ESRGAN}"
        elif mid == "mesh-export":
            import pymeshlab  # noqa: F401
            import trimesh  # noqa: F401
        import imgtools  # noqa: F401
        return True, ""

    def load(self, model, ctx):
        mid = model["id"]
        if mid in REMBG_SESSIONS:
            from rembg import new_session
            return {"session": new_session(REMBG_SESSIONS[mid])}
        if mid == "realesrgan-x4":
            _tv_shim()
            import imgtools
            return {"up": imgtools.Upscaler(ESRGAN)}
        if mid == "recenter-3d":
            return {"session": None}   # rembg u2net is created on first need
        if mid == "mesh-export":
            import postprocess  # noqa: F401
            return {}
        raise InputError(f"unknown tool model {mid}")

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict) and h.get("up") is not None:
            h["up"].unload()

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _images(job):
        ims = [i for i in job.inputs if i.get("type") == "image"]
        if not ims:
            raise InputError("Add at least one image.")
        return ims

    @staticmethod
    def _rembg(session, img, post_process=False, alpha_matting=False):
        import imgtools
        from rembg import remove
        src = imgtools.flatten(img) if img.mode != "RGB" else img
        out = remove(src, session=session, post_process_mask=bool(post_process), alpha_matting=bool(alpha_matting))
        return out.convert("RGBA")

    @staticmethod
    def _name(inp, suffix):
        return f"{(inp.get('name') or 'image')[:60]} ({suffix})"

    # --------------------------------------------------------------------- run
    def run(self, job, ctx):
        mid = job.model_id
        if mid in REMBG_SESSIONS:
            return self._run_rmbg(job, ctx)
        if mid == "realesrgan-x4":
            return self._run_upscale(job, ctx)
        if mid == "recenter-3d":
            return self._run_recenter(job, ctx)
        if mid == "mesh-export":
            return self._run_export(job, ctx)
        raise InputError(f"unknown tool model {mid}")

    def _run_rmbg(self, job, ctx):
        import imgtools
        s = self.handle(job.model_id)["session"]
        p = job.params
        ims = self._images(job)
        for k, inp in enumerate(ims):
            ctx.check()
            ctx.progress(k, len(ims), "rembg", f"Removing the background of {inp.get('name')}")
            t = time.time()
            img = imgtools.open_checked(inp["path"])
            out = imgtools.clean_alpha(self._rembg(s, img, p.get("post_process"), p.get("alpha_matting")), int(p.get("alpha_clean", 8)))
            if imgtools.object_bbox(out) is None:
                raise InputError(f"Nothing was left of {inp.get('name')}: the image has no clear foreground object.")
            f = job.out_dir / f"nobg_{k + 1}.png"
            out.save(f)
            frac = imgtools.alpha_stats(out)[1]
            ctx.output(f, "image", index=k, name=self._name(inp, "no bg"), seconds=round(time.time() - t, 2),
                       transparent_fraction=round(frac, 4))
        return {"meta": {"session": REMBG_SESSIONS[job.model_id]}}

    def _run_upscale(self, job, ctx):
        import imgtools
        up = self.handle(job.model_id)["up"]
        p = job.params
        scale, tile = int(p.get("scale", 4)), int(p.get("tile", 0))
        ims = self._images(job)
        for k, inp in enumerate(ims):
            ctx.check()
            ctx.progress(k, len(ims), "upscaling", f"Real-ESRGAN x{scale} {inp.get('name')}", interruptible=False)
            t = time.time()
            img = imgtools.open_checked(inp["path"])
            try:
                out = up(img, scale, tile)
            except ValueError as e:
                raise InputError(str(e)) from None
            f = job.out_dir / f"upscaled_{k + 1}.png"
            out.save(f)
            ctx.output(f, "image", index=k, name=self._name(inp, f"x{scale}"), seconds=round(time.time() - t, 2),
                       size=[out.width, out.height])
        return {"meta": {"scale": scale, "tile": tile}}

    def _run_recenter(self, job, ctx):
        import imgtools
        h = self.handle(job.model_id)
        p = job.params
        ims = self._images(job)
        for k, inp in enumerate(ims):
            ctx.check()
            ctx.progress(k, len(ims), "recenter", f"Recentering {inp.get('name')}")
            img = imgtools.clean_alpha(imgtools.open_checked(inp["path"]), int(p.get("alpha_clean", 8)))
            notes = []
            if not imgtools.alpha_stats(img)[0]:
                if p.get("remove_bg_first", True):
                    if h.get("session") is None:
                        from rembg import new_session
                        h["session"] = new_session("u2net")
                    img = imgtools.clean_alpha(self._rembg(h["session"], img), int(p.get("alpha_clean", 8)))
                    notes.append("background removed first (rembg u2net)")
                else:
                    notes.append("no transparency: the whole frame is treated as the object (padded to a square only)")
            try:
                out = imgtools.crop_pad_square(img, float(p.get("margin", 0.15)), int(p.get("out_size", 1024) or 0),
                                               p.get("background", "transparent"), int(p.get("alpha_threshold", 16)))
            except ValueError as e:
                raise InputError(str(e)) from None
            f = job.out_dir / f"recentered_{k + 1}.png"
            out.save(f)
            for n in notes:
                ctx.log(f"{inp.get('name')}: {n}")
            ctx.output(f, "image", index=k, name=self._name(inp, "3D-ready"), notes=notes, size=[out.width, out.height])
        return {"meta": {}}

    def _run_export(self, job, ctx):
        """Mesh clean-up + export (engines/hy3d/app/postprocess.py export_run_mesh, adapted to asset folders)."""
        import postprocess as pp
        p = job.params
        meshes = [i for i in job.inputs if i.get("type") == "mesh"]
        if not meshes:
            raise InputError("Add a mesh (GLB) to export.")
        fmt = str(p.get("export_type", "glb")).lower()
        if fmt not in pp.SUPPORTED_FORMATS:
            raise InputError(f"Unsupported format {fmt}")
        for k, inp in enumerate(meshes):
            ctx.check()
            src = Path(inp["path"])
            adir = src.parent
            textured = glb_has_texture(src)
            stem = f"export_{k + 1}"
            report, warnings = {}, []
            name = self._name(inp, fmt.upper())
            t = time.time()
            ctx.progress(k, len(meshes), "export", f"Exporting {inp.get('name')} as {fmt.upper()}")
            if textured and p.get("include_texture", True):
                if any(p.get(x) for x in ("post_simplify",)) or int(p.get("post_smooth", 0) or 0):
                    warnings.append("Decimate / smooth were skipped: they would destroy the UV layout of a textured mesh. "
                                    "Untick 'Keep texture' to process the shape, then retexture it.")
                if fmt == "glb":
                    dst = job.out_dir / f"{stem}.glb"
                    shutil.copy2(src, dst)
                    otype = "mesh"
                elif fmt == "obj":
                    dst = job.out_dir / f"{stem}_obj.zip"
                    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
                        objs = sorted(adir.glob("textured_mesh*"))
                        if any(f.suffix == ".obj" for f in objs):
                            for f in objs:
                                if f.suffix in (".obj", ".mtl", ".jpg", ".png"):
                                    z.write(f, f.name)
                        else:   # an uploaded textured GLB: let trimesh write obj + mtl + textures
                            import trimesh
                            scene = trimesh.load(str(src), force="mesh", process=False)
                            text, tex = trimesh.exchange.obj.export_obj(scene, include_texture=True, return_texture=True,
                                                                        mtl_name="mesh.mtl")
                            z.writestr("mesh.obj", text)
                            for fn, data in (tex or {}).items():
                                z.writestr(fn, data)
                    otype = "other"
                else:
                    mesh = pp.load_mesh(src)
                    if fmt == "ply":
                        try:
                            mesh.visual = mesh.visual.to_color()
                        except Exception as e:  # noqa: BLE001
                            warnings.append(f"Could not bake the texture to vertex colours ({e}); geometry only.")
                    else:
                        warnings.append("STL cannot carry colour or texture; geometry only.")
                    dst = job.out_dir / f"{stem}.{fmt}"
                    mesh.export(str(dst), file_type=fmt)
                    otype = "other"
                report["textured"] = True
            else:
                base = adir / "white_mesh.glb" if (textured and (adir / "white_mesh.glb").exists()) else src
                if textured and base == src:
                    warnings.append("No untextured source was stored with this mesh: its texture is dropped.")
                mesh = pp.load_mesh(base)
                mesh, report = pp.run_pipeline(
                    mesh, floaters=bool(p.get("post_floaters", True)), floater_ratio=float(p.get("floater_ratio", 0.005)),
                    degenerate=bool(p.get("post_degenerate", True)), simplify=bool(p.get("post_simplify", False)),
                    target_faces=int(p.get("post_target_faces", 10000)), smooth=int(p.get("post_smooth", 0) or 0),
                    normals=bool(p.get("post_fix_normals", False)))
                dst = pp.export_mesh(mesh, job.out_dir / f"{stem}.{fmt}", fmt)
                report["result"] = pp.mesh_stats(mesh)
                otype = "mesh" if fmt == "glb" else "other"
            attach = []
            if otype == "other" and fmt != "glb":  # a GLB preview next to the file so the library can show it
                try:
                    prev = job.out_dir / f"{stem}_preview.glb"
                    if report.get("textured"):
                        shutil.copy2(src, prev)
                    else:
                        pp.export_mesh(pp.load_mesh(dst) if fmt != "obj" else mesh, prev, "glb")
                    attach.append(prev.name)
                except Exception as e:  # noqa: BLE001
                    ctx.log(f"preview GLB skipped: {e}")
            for w in warnings:
                ctx.log(w)
            ctx.output(dst, otype, index=k, name=name, report=report, warnings=warnings, format=fmt,
                       seconds=round(time.time() - t, 2), attach=attach)
        return {"meta": {"format": fmt}}


if __name__ == "__main__":
    serve(ToolsBackend())
