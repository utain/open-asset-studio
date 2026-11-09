"""Retopo + bake: the job handler of the `retopo-instant-meshes` model (CPU, no weights; the work is workers/retopo_worker.py).

  handler retopo.bake   inputs: one mesh asset (GLB) that is NOT rigged
                        runs the retopo backend (envs/retopo) beside the GPU queue and writes ONE new mesh asset `mesh.glb`
                        (quad-dominant low poly, UVs, normal / base colour / ORM maps, TANGENT) with attachments retopo_quads.obj / .mtl,
                        the three PNG maps and retopo_report.json. Meta: worker.retopo (faces, quad %, fidelity, warnings ...).

Why a handler and not the default `worker` one:
  * a rigged or animated input is refused here, before the worker starts (retopology rebuilds every vertex, so skin weights and clips
    would silently vanish); the worker repeats the check for direct API / MCP calls,
  * the result licence is the most restrictive of the source asset and the tool (otherwise a non-commercial shape would become
    "commercial" after retopo); same rule as app/plugins/video.py::_lic.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import config
from ..jobs import JobContext, JobError, handler


def _common():
    if str(config.WORKERS) not in sys.path:
        sys.path.append(str(config.WORKERS))
    import m3d_common
    import retopo_common
    return retopo_common, m3d_common


def _lic(parents: list) -> dict | None:
    """The most restrictive licence (non-commercial wins); in a tie the first one, so the source asset comes first."""
    lics = [x for x in parents if x]
    nc = [x for x in lics if not x.get("commercial", True)]
    return (nc or lics or [None])[0]


def check_input(meta: dict, path, rc, m3) -> None:
    """Refuse a rigged or animated mesh asset (JobError kind input)."""
    w, info = meta.get("worker") or {}, meta.get("info") or {}
    if w.get("rigged") or info.get("rigged"):
        raise JobError(rc.E3, "input")
    if w.get("animations") or info.get("animations"):
        raise JobError(rc.E4, "input")
    if str(path).lower().endswith(".glb"):
        try:
            f = m3.glb_flags(path)
        except Exception:  # noqa: BLE001 - the worker reports unreadable files
            return
        if f["skins"]:
            raise JobError(rc.E3, "input")
        if f["animations"] or f["morph_targets"]:
            raise JobError(rc.E4, "input")


@handler("retopo.bake", gpu=False, label="Retopo + bake")
def bake(ctx: JobContext):
    rc, m3 = _common()
    ins = [i for i in ctx.input_assets() if i["type"] == "mesh"]
    if not ins:
        raise JobError(rc.E1, "input")
    src = ctx.assets.get(ins[0]["asset_id"])
    check_input(src, ins[0]["path"], rc, m3)
    res = ctx.run_worker(frac_range=(0.0, 0.95))
    ctx.progress(0.97, "Saving outputs", stage="saving")
    lic = _lic([src.get("license"), (ctx.model or {}).get("license")])
    for i, f in enumerate(res.get("files") or []):
        meta = ctx.asset_meta(f, i)
        if lic:
            meta["license"] = lic
        meta["tags"] = sorted(set(meta.get("tags") or []) | {"retopo"})
        att = [str(Path(f["path"]).parent / a) if not str(a).startswith("/") else a for a in f.get("attach") or []]
        ctx.add_output(f["path"], f.get("type") or "mesh", meta, attach=att)
    if res.get("meta"):
        ctx.job["result_meta"] = res["meta"]
