"""Rig editing: a CPU tool and a read endpoint (the maths is workers/rig_edit.py, numpy + stdlib).

  tool:rig.edit   inputs: one mesh asset (GLB) that has a glTF skin (a Rig / Motion / Animate result)
                  params: {ops: [{op: rename | rename_standard | move_joint | delete_bone | add_bone | add_chain | reparent, ...}],
                           clips: true}   (clips: rebuild the flex / wiggle preview clips when the input had them)
                  writes a NEW mesh asset (parents = the input) with the edited skeleton; the mesh, textures and the
                  rest pose are unchanged. Meta: info.rigged / joints / needs_reskin / rig_edit (ops + summary),
                  attachment skeleton.json. New bones have zero skin weight: run task `reskin` on the result.
  GET  /api/x/rig/<asset>/skeleton        the bones of a rigged mesh (name, index, parent, position, role, weight share)
  POST /api/x/rig/<asset>/dry_run {ops}   what an edit would do (summary only, nothing is written)

Ops, bone references (name, index or role word like "left elbow") and units are described in workers/rig_edit.py.
"""
from __future__ import annotations

import json
import sys

from fastapi import APIRouter, HTTPException

from .. import config
from ..assets import AssetError
from ..jobs import JobContext, JobError, handler

router = APIRouter()
CORE = None


def setup(core):
    global CORE
    CORE = core


def _edit():
    if str(config.WORKERS) not in sys.path:
        sys.path.append(str(config.WORKERS))
    import rig_edit
    return rig_edit


def _glb_of(meta: dict, assets) -> str:
    if meta.get("type") != "mesh":
        raise JobError(f"{meta.get('name') or meta.get('id')} is not a mesh.", "input")
    p = assets.file_path(meta["project"], meta["id"], meta["main"])
    if not str(p).lower().endswith(".glb"):
        raise JobError(f"{meta.get('name')}: only GLB meshes can be edited.", "input")
    return str(p)


def _skeleton_json(info: dict) -> dict:
    return {"joints": [{"name": r["name"], "parent": r["parent"], "position": r["position"]} for r in info["joints"]]}


@handler("tool:rig.edit", gpu=False, label="Edit rig")
def rig_edit_tool(ctx: JobContext):
    re_ = _edit()
    ins = [i for i in ctx.input_assets() if i["type"] == "mesh"]
    if len(ins) != 1:
        raise JobError("Pick exactly one rigged mesh asset (GLB with a skeleton) to edit.", "input")
    inp = ins[0]
    p = ctx.params
    try:
        ops = re_.parse_ops(p.get("ops"))
    except re_.RigEditError as e:
        raise JobError(str(e), "input", "Example: ops=[{\"op\": \"rename_standard\", \"scheme\": \"mixamo\"}]") from None
    clips = p.get("clips", True)
    if isinstance(clips, str):
        clips = clips.strip().lower() not in ("0", "false", "no", "off")
    parent = ctx.assets.get(inp["asset_id"])
    path = _glb_of(parent, ctx.assets)
    ctx.progress(0.1, f"Editing the rig of {inp['name']}", stage="editing")
    try:
        out, rep = re_.edit(path, ops, regen_clips=bool(clips))
        info = re_.describe(out)
    except re_.NoSkinError:
        raise JobError(f"{inp['name']} has no skeleton / skin, so there is nothing to edit.", "input",
                       "Rig it first (3D studio, Rig tab), then edit the rigged result.") from None
    except re_.RigAnimError as e:
        raise JobError(f"{inp['name']}: {e}", "input") from None
    ctx.check_cancel()
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    f = ctx.scratch / "edited.glb"
    f.write_bytes(out)
    sk = ctx.scratch / "skeleton.json"
    sk.write_text(json.dumps(_skeleton_json(info), indent=1))
    for s in rep["ops"]:
        ctx.log(f"{s['op']}: {s['summary']}")
    for n in rep["notes"]:
        ctx.log(n)
    meta = ctx.asset_meta({"name": f"{inp['name']} (rig edited)"[:80]}, 0)
    meta["parents"] = [inp["asset_id"]]
    meta["source"] = "tool"
    for k in ("task", "studio", "prompt", "improved_prompt", "final_prompt", "model", "model_name", "seed"):
        if parent.get(k) is not None:
            meta[k] = parent[k]
    if parent.get("license"):
        meta["license"] = parent["license"]
    meta["studio"] = parent.get("studio") or "3d"
    meta["tags"] = sorted(set(parent.get("tags") or []) | {"rig-edited"})
    meta["info"] = {"rigged": True, "joints": info["count"], "animations": info["animations"], "needs_reskin": rep["needs_reskin"],
                    "rig_edit": {"ops": rep["ops"], "needs_reskin": rep["needs_reskin"], "reskin_suggested": rep["reskin_suggested"],
                                 "zero_weight_bones": rep["zero_weight_bones"], "notes": rep["notes"],
                                 "animations_dropped": rep["animations_dropped"]}}
    meta["x_post"] = {"tool": ctx.job.get("tool"), "op": {"ops": ops}, "from": [inp["asset_id"]]}
    ctx.add_output(f, "mesh", meta, attach=[sk])
    ctx.progress(1.0, "Rig edited" + (" - re-skin needed for the new bones" if rep["needs_reskin"] else ""), stage="done")


def _asset_glb(asset_id: str) -> tuple[dict, str]:
    if CORE is None:
        raise HTTPException(503, "studio core not ready")
    try:
        meta = CORE.assets.get(asset_id)
        return meta, _glb_of(meta, CORE.assets)
    except AssetError as e:
        raise HTTPException(e.status, str(e)) from None
    except JobError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/{asset_id}/skeleton")
def skeleton(asset_id: str):
    meta, path = _asset_glb(asset_id)
    re_ = _edit()
    try:
        info = re_.describe(path)
    except re_.NoSkinError:
        raise HTTPException(400, f"{meta.get('name') or asset_id} has no skeleton / skin. Rig it first (task rig).") from None
    except re_.RigAnimError as e:
        raise HTTPException(400, str(e)) from None
    return {"asset": asset_id, "project": meta.get("project"), "name": meta.get("name"),
            "needs_reskin": bool((meta.get("info") or {}).get("needs_reskin")), **info}


@router.post("/{asset_id}/dry_run")
def dry_run(asset_id: str, body: dict):
    meta, path = _asset_glb(asset_id)
    re_ = _edit()
    try:
        _, rep = re_.edit(path, body.get("ops"))
    except re_.NoSkinError:
        raise HTTPException(400, f"{meta.get('name') or asset_id} has no skeleton / skin. Rig it first (task rig).") from None
    except re_.RigAnimError as e:
        raise HTTPException(400, str(e)) from None
    rep.pop("bytes_before", None)
    rep.pop("bytes_after", None)
    return rep
