"""3D mesh CPU tools (no model, they run beside GPU jobs).

  tool:mesh.flex_preview   inputs: one mesh asset (GLB) that has a glTF skin (e.g. a Rig result)
                           params: {clips: ["flex", "wiggle"], angle: 30}
                           writes a NEW mesh asset (parents = the input) whose GLB carries the animation clips `flex`
                           (every joint swings in turn, a skinning test) and `wiggle` (a gentle all-joint loop). The mesh,
                           skin and materials are byte-identical; the studio previews the first clip automatically.

The clip builder is workers/rig_anim.py (numpy + stdlib), shared with the rig worker.
"""
from __future__ import annotations

import sys

from .. import config
from ..jobs import JobContext, JobError, handler


def _rig_anim():
    if str(config.WORKERS) not in sys.path:
        sys.path.append(str(config.WORKERS))
    import rig_anim
    return rig_anim


@handler("tool:mesh.flex_preview", gpu=False, label="Animate preview (flex test)")
def flex_preview(ctx: JobContext):
    ra = _rig_anim()
    p = ctx.params
    ins = [i for i in ctx.input_assets() if i["type"] == "mesh"]
    if not ins:
        raise JobError("Pick a rigged mesh asset (GLB with a skeleton) first.", "input")
    clips = p.get("clips") or list(ra.CLIPS)
    if isinstance(clips, str):
        clips = [c.strip() for c in clips.split(",") if c.strip()]
    try:
        angle = float(p.get("angle", 30) or 30)
    except (TypeError, ValueError):
        raise JobError("angle must be a number of degrees.", "input") from None
    if not 5 <= angle <= 80:
        raise JobError("angle must be between 5 and 80 degrees.", "input")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, inp in enumerate(ins):
        ctx.check_cancel()
        ctx.progress(i / len(ins), f"Animating {inp['name']}", stage="animating")
        if not str(inp["path"]).lower().endswith(".glb"):
            raise JobError(f"{inp['name']}: only GLB meshes can be animated.", "input")
        out = ctx.scratch / f"animated_{i + 1}.glb"
        try:
            info = ra.animate_file(inp["path"], out, clips, angle_deg=angle)
        except ra.NoSkinError:
            raise JobError(f"{inp['name']} has no skeleton / skin, so there is nothing to animate.", "input",
                           "Rig it first (3D studio, Rig tab), then animate the rigged result.") from None
        except ra.RigAnimError as e:
            raise JobError(f"{inp['name']}: {e}", "input") from None
        ctx.log(f"{inp['name']}: {info['joints']} joints, " + ", ".join(f"{c['name']} {c['duration']} s" for c in info["clips"])
                + f", {info['bytes_before']} -> {info['bytes_after']} bytes")
        parent = ctx.assets.get(inp["asset_id"])
        meta = ctx.asset_meta({"name": f"{inp['name']} (animated)"}, i)
        meta["parents"] = [inp["asset_id"]]
        meta["source"] = "tool"
        for k in ("task", "studio", "prompt", "improved_prompt", "final_prompt", "model", "model_name", "seed"):
            if parent.get(k) is not None:
                meta[k] = parent[k]
        if parent.get("license"):
            meta["license"] = parent["license"]
        meta["studio"] = parent.get("studio") or "3d"
        meta["tags"] = sorted(set(parent.get("tags") or []) | {"animated"})
        meta["info"] = {"rigged": True, "joints": info["joints"], "animations": [c["name"] for c in info["clips"]]}
        meta["x_post"] = {"tool": ctx.job.get("tool"), "op": {"clips": [c["name"] for c in info["clips"]], "angle": angle},
                          "from": [inp["asset_id"]]}
        ctx.add_output(out, "mesh", meta)
        made += 1
    ctx.progress(1.0, f"{made} animated mesh(es)", stage="done")
