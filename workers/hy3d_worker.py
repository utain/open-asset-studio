"""Hunyuan3D-2.1 worker (runs in envs/hy3d, read-only): image -> shape, texture / retexture.

The proven Hunyuan3D engine (engines/hy3d) is imported, not forked: engines/hy3d/app/engine.py does every vendor override through
wrappers and monkeypatches (multiview steps / CFG / seed via _MVProxy, Real-ESRGAN `tile_size` + render-size fix via
_SuperRes, xatlas padding, UV inpaint / full-res save hooks, PBR GLB builder, border ratio, scheduler shift, FlashVDM,
mesh clean-up), so engines/hy3d/vendor/Hunyuan3D-2.1 is never edited. engines/hy3d/app/params.py normalises every knob.

  task shape    inputs: image (role image)              -> mesh asset (white GLB, or textured PBR GLB when `textured`)
  task texture  inputs: mesh (+ optional image)          -> textured mesh asset (the white shape is kept as attachment)

Asset layout produced (main file + attachments in the same folder):
  mesh.glb                 textured PBR GLB (or the white mesh for shape-only jobs)
  white_mesh.glb           untextured shape (textured assets only; used by retexture and mesh export)
  input.png                the cleaned RGBA conditioning image; input_original.png = the image as given
  textured_mesh.{obj,mtl,jpg} + _metallic.jpg / _roughness.jpg   texture maps + OBJ for the OBJ export
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the venv is read-only for the studio: no __pycache__ writes into engines/hy3d

import json
import os
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path

from studio_worker import Backend, Cancelled, InputError, log, serve

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
HY3D_ENGINE = Path(os.environ.get("STUDIO_HY3D_ENGINE") or STUDIO / "engines" / "hy3d")
APP = HY3D_ENGINE / "app"
DATA = Path(os.environ.get("STUDIO_DATA") or STUDIO / "data")
MAX_SEED = 2**32 - 1

ATTACH = ("white_mesh.glb", "input.png", "input_original.png", "textured_mesh.jpg", "textured_mesh_metallic.jpg",
          "textured_mesh_roughness.jpg", "textured_mesh.obj", "textured_mesh.mtl")


def _engine():
    tmp = DATA / "tmp" / "hy3d"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    if "engine" not in sys.modules:
        # torch.distributed's RPC template module writes a temp dir at import time: import it while tempfile points at
        # the studio, before engine.py's bootstrap points tempfile at engines/hy3d/tmp (nothing may be written under engines/hy3d)
        import torch  # noqa: F401
        try:
            import torch.distributed.nn.jit.instantiator  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import engine  # noqa: F401  (bootstraps cwd / sys.path / torchvision shim for the vendor code)
    tempfile.tempdir = str(tmp)   # vendor temp files stay inside the studio data folder
    return engine


def engine_params(p: dict) -> dict:
    """Studio params (registry names) -> engines/hy3d params dict (seed -1 = random -> randomize flags)."""
    import params as P
    raw = {k: p[k] for k in P.SPECS if k in p}
    seed = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    raw["randomize_seed"] = seed < 0
    raw["seed"] = max(0, seed)
    ts = int(p.get("tex_seed", 0) if p.get("tex_seed") is not None else 0)
    raw["tex_randomize_seed"] = ts < 0
    raw["tex_seed"] = max(0, ts)
    raw["preset"] = "Custom"
    return P.normalize(raw)


class _Progress:
    """Polls the engine Tracker and turns it into protocol progress events; forwards cancel to the engine."""

    def __init__(self, ctx, eng, tracker):
        self.ctx, self.eng, self.tr = ctx, eng, tracker
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join(timeout=2)

    def _loop(self):
        last = None
        while not self.stop.wait(0.5):
            if self.ctx.cancelled and self.eng.busy:
                self.eng.cancel_event.set()
            desc = self.tr.desc or ""
            cur = (round(self.tr.frac, 3), desc)
            if cur != last:
                last = cur
                loading = desc.startswith("Loading")
                self.ctx.progress(0 if loading else int(self.tr.frac * 1000), 0 if loading else 1000,
                                  "loading" if loading else "3d", desc, interruptible=not loading)


class Hy3DBackend(Backend):
    name = "hy3d"
    max_loaded = 1

    def probe(self, model):
        vendor = HY3D_ENGINE / "vendor" / "Hunyuan3D-2.1"
        if not (APP / "engine.py").exists() or not vendor.is_dir():
            return False, f"engines/hy3d engine or vendor checkout missing ({APP})"
        _engine()
        return True, ""

    def load(self, model, ctx):
        ctx.progress(0, 0, "loading", "Preparing Hunyuan3D-2.1 (the shape / paint models load on first use)", interruptible=False)
        eng = _engine()
        root = DATA / "tmp" / "hy3d_runs"
        return {"engine": eng, "eng": eng.Engine(root)}

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict) and h.get("eng") is not None:
            h["eng"].unload()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _open_image(path: str, alpha_clean: int):
        import imgtools
        im = imgtools.open_checked(path)
        return imgtools.clean_alpha(im, int(alpha_clean)) if im.mode == "RGBA" else im

    def _emit(self, job, ctx, run_dir: Path, res: dict, task: str, extra_name: str):
        """Hand the run folder over as one mesh asset (main GLB + attachments)."""
        rd = Path(run_dir)
        main = rd / ("textured_mesh.glb" if res.get("has_textured") else "white_mesh.glb")
        if not main.exists():
            raise RuntimeError("The engine finished without a mesh file.")
        dst_dir = job.out_dir / "asset"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(main, dst_dir / "mesh.glb")
        attach = []
        for n in ATTACH:
            if n == "white_mesh.glb" and not res.get("has_textured"):
                continue
            if (rd / n).exists():
                shutil.copy2(rd / n, dst_dir / n)
                attach.append(n)
        st = res.get("stats") or {}
        c = res.get("controls") or {}
        seed = c.get("seed") if task == "shape" else c.get("tex_seed")
        times = st.get("time") or {}
        ctx.log("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in times.items() if isinstance(v, (int, float))))
        for w in res.get("warnings") or []:
            ctx.log(f"warning: {w}")
        ctx.output(dst_dir / "mesh.glb", "mesh", seed=seed, name=extra_name, attach=attach, textured=bool(res.get("has_textured")),
                   tex_seed=c.get("tex_seed"), shape_seed=c.get("seed"), stats=st, warnings=res.get("warnings") or [],
                   engine_controls=c)
        return {"meta": {"timings": times, "peak_gpu_gb": st.get("peak_gpu_gb"), "warnings": res.get("warnings")}}

    # ---------------------------------------------------------------------- run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        E, eng = h["engine"], h["eng"]
        from runs import RunStore
        eng.store = RunStore(job.out_dir / "runs")       # one run folder per job, inside the job's scratch
        p = engine_params(job.params)
        tracker = E.Tracker()
        try:
            with _Progress(ctx, eng, tracker):
                if job.task == "shape":
                    return self._shape(job, ctx, E, eng, p, tracker)
                if job.task == "texture":
                    return self._texture(job, ctx, E, eng, p, tracker)
                raise InputError(f"Hunyuan3D cannot do '{job.task}'.")
        except E.JobCancelled:
            raise Cancelled() from None
        except E.UserError as e:
            if "out of memory" in str(e).lower() or "ran out of memory" in str(e).lower():
                raise MemoryError(f"out of memory: {e}") from None
            raise InputError(str(e)) from None

    def _shape(self, job, ctx, E, eng, p, tracker):
        inp = job.input(role="image") or job.input(type="image")
        if inp is None:
            raise InputError("Add the image to turn into 3D.")
        img = self._open_image(inp["path"], job.params.get("alpha_clean", 8))
        textured = bool(job.params.get("textured", True))
        res = eng.generate(img, p, textured, tracker)
        return self._emit(job, ctx, Path(res["run_dir"]), res, "shape",
                          f"{(inp.get('name') or 'mesh')[:50]} ({'textured' if textured else 'shape'})")

    def _texture(self, job, ctx, E, eng, p, tracker):
        mesh = job.input(type="mesh")
        if mesh is None:
            raise InputError("Add the mesh to texture.")
        ref_inp = job.input(type="image")
        src = Path(mesh["path"])
        adir = src.parent
        white = adir / "white_mesh.glb" if (adir / "white_mesh.glb").exists() else src
        rid, d = eng.store.new_run("shape")
        shutil.copy2(white, d / "white_mesh.glb")
        for n in ("input.png", "input_original.png"):
            if (adir / n).exists():
                shutil.copy2(adir / n, d / n)
        current = None
        if ref_inp is not None:
            current = self._open_image(ref_inp["path"], job.params.get("alpha_clean", 8))
            if not (d / "input.png").exists():
                current.convert("RGBA").save(d / "input.png")
        if not (d / "input.png").exists():
            raise InputError("This mesh has no source image stored with it: add a reference image for the texture.")
        if p["tex_reference"] == "current" and current is None:
            ctx.log("Reference 'current' needs an image in the slot: using the image the shape was made from.")
            p["tex_reference"] = "cleaned"
        elif current is not None and p["tex_reference"] != "current" and not (adir / "input.png").exists():
            p["tex_reference"] = "current"
        shape_seed = (mesh.get("meta") or {}).get("seed")
        (d / "params.json").write_text(json.dumps({"controls": {"seed": int(shape_seed) if isinstance(shape_seed, int) and shape_seed >= 0 else 0}}))
        (d / "stats.json").write_text(json.dumps({}))
        res = eng.retexture(rid, p, current, tracker)
        return self._emit(job, ctx, Path(res["run_dir"]), res, "texture", f"{(mesh.get('name') or 'mesh')[:50]} (textured)")


if __name__ == "__main__":
    serve(Hy3DBackend())
