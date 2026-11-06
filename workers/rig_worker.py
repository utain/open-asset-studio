"""Auto-rigging backend (envs/rig): UniRig (VAST-AI-Research/UniRig, MIT), mesh -> skeleton + skin weights -> rigged GLB.

  task rig   input role `mesh` (a mesh asset, GLB)  ->  mesh asset `rigged.glb` (the original geometry / materials plus a glTF
                                                         skeleton, `skin` with JOINTS_0 / WEIGHTS_0) + `skeleton.json`

UniRig's own pipeline needs Blender (bpy) to read meshes and write FBX, and CUDA extensions (spconv, flash-attn, torch-scatter,
torch-cluster) plus an OpenGL context for voxelisation. None exists here, so:
  * the mesh is read with trimesh and written as UniRig's raw_data.npz (Blender coordinates, +Z up),
  * both neural stages (skeleton: autoregressive OPT-350m + shape encoder; skin: PTv3 point transformer + bone attention) run
    unchanged from vendor/UniRig through its run.py in a subprocess, on pure-PyTorch stand-ins (workers/rig_shims.py),
  * the predicted skeleton and per-point skin weights are read back from the npz files, the weights are transferred to every
    vertex of the original mesh with UniRig's own `reskin`, and the glTF skin is written here (no Blender).
Skeleton quality follows the UniRig 'articulation-xl' checkpoints: humanoids, quadrupeds and many creatures rig well, rigid
props get a plausible chain; there is no retargeting to Mixamo / VRM names.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

import m3d_common as C
from studio_worker import Backend, Cancelled, InputError, serve

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
UNIRIG = STUDIO / "vendor" / "UniRig"
HERE = Path(__file__).resolve().parent

SKELETON_CKPT = ("VAST-AI/UniRig", "skeleton/articulation-xl_quantization_256/model.ckpt")
SKIN_CKPT = ("VAST-AI/UniRig", "skin/articulation-xl/model.ckpt")


class RigBackend(Backend):
    name = "rig"
    max_loaded = 1

    def probe(self, model):
        import torch  # noqa: F401
        import trellis_shims  # noqa: F401
        import pygltflib  # noqa: F401
        if not (UNIRIG / "run.py").is_file():
            return False, "vendor/UniRig is missing (scripts/fetch_vendor.sh UniRig)"
        return True, ""

    def load(self, model, ctx):
        return {"ready": True}      # the two models are loaded by the UniRig subprocess for each run

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        if job.task != "rig":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        inp = job.input(role="mesh") or job.input(type="mesh")
        if inp is None:
            raise InputError("Pick the mesh to rig.")
        p = job.params
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)
        work = job.out_dir / "unirig"
        if work.exists():
            shutil.rmtree(work)
        t = {}
        t0 = time.time()
        ctx.progress(0, 4, "prep", "Reading the mesh")
        scene_meshes = _load_meshes(inp["path"])
        full_v = np.concatenate([m.vertices for m in scene_meshes]).astype(np.float64)
        full_f = np.concatenate([m.faces + off for m, off in zip(scene_meshes, _offsets(scene_meshes))])
        if len(full_f) == 0:
            raise InputError("The mesh has no faces.")
        # 1. UniRig input: Blender frame (+Z up, -Y front), at most max_faces triangles
        v_b = _to_blender(full_v)
        dv, df = _decimate(v_b, full_f, int(p.get("max_faces", 50000) or 50000), ctx)
        center, scale = _norm(dv)
        _write_raw(work, dv, df)
        t["prep_s"] = round(time.time() - t0, 2)
        ctx.check()
        seed = C.seed_of(p)

        # 2. skeleton
        ctx.progress(1, 4, "skeleton", "Predicting the skeleton")
        t1 = time.time()
        _make_configs(work, p)
        self._stage(work, "skeleton_task", seed, ctx)
        t["skeleton_s"] = round(time.time() - t1, 2)
        sk_path = work / "dataset_inference_clean" / "job" / "predict_skeleton.npz"
        if not sk_path.is_file():
            raise RuntimeError("UniRig produced no skeleton (see the worker log); try another seed or a cleaner mesh.")
        sk = np.load(sk_path, allow_pickle=True)
        joints_in = np.asarray(sk["joints"], dtype=np.float64)
        parents = [None if q is None else int(q) for q in list(sk["parents"])]
        names = [str(n) for n in list(sk["names"])]
        sk_vertices = np.asarray(sk["vertices"], dtype=np.float64)
        ctx.log(f"skeleton: {len(names)} joints; stored vertices span {np.abs(sk_vertices).max():.3f}")
        ctx.check()

        # 3. skin
        ctx.progress(2, 4, "skin", "Predicting skin weights")
        t2 = time.time()
        self._stage(work, "skin_task", seed, ctx)
        t["skin_s"] = round(time.time() - t2, 2)
        skin_path = work / "dataset_inference_clean" / "job" / "predict_skin.npz"
        if not skin_path.is_file():                                  # writer variants that honour output_dir=results
            skin_path = work / "results" / "dataset_inference_clean" / "job" / "predict_skin.npz"
        if not skin_path.is_file():
            raise RuntimeError("UniRig produced no skin weights (see the worker log).")
        sg = np.load(skin_path, allow_pickle=True)
        sampled_skin = np.asarray(sg["skin"], dtype=np.float32)
        sampled_v = np.asarray(sg["vertices"], dtype=np.float64)
        joints_norm = np.asarray(sg["joints"], dtype=np.float64)
        ctx.check()

        # 4. transfer weights to every original vertex, write the glTF skin
        ctx.progress(3, 4, "export", "Writing the rigged GLB")
        t3 = time.time()
        sys.path.insert(0, str(UNIRIG))
        import rig_shims
        rig_shims.install()
        from src.system.skin import reskin
        nv = (v_b - center) / scale                                  # all original vertices in UniRig's normalised frame
        skin = reskin(sampled_vertices=sampled_v, vertices=nv, parents=parents, faces=full_f, sampled_skin=sampled_skin,
                      sample_method="median", alpha=2.0, threshold=0.03)
        skin = np.nan_to_num(skin, nan=0.0)
        joints_b = joints_norm * scale + center                      # back to the Blender frame of the input
        joints_g = _from_blender(joints_b)                          # and to glTF (+Y up)
        glb = out / "rigged.glb"
        _write_rigged_glb(scene_meshes, skin, joints_g, parents, names, glb)
        (out / "skeleton.json").write_text(json.dumps({
            "joints": [{"name": n, "parent": q, "position": [round(float(x), 5) for x in j]}
                       for n, q, j in zip(names, parents, joints_g)]}, indent=1))
        t["export_s"] = round(time.time() - t3, 2)
        info = C.mesh_info(glb)
        ctx.log(f"timings {t}, {len(names)} joints, {info}")
        shutil.rmtree(work, ignore_errors=True)
        ctx.output(glb, "mesh", seed=seed, name=f"{(inp.get('name') or 'mesh')[:50]} (rigged)", attach=["skeleton.json"],
                   rigged=True, joints=len(names), stats=info)
        return {"meta": {"timings": t, "joints": len(names)}}

    # ------------------------------------------------------------------ UniRig subprocess
    def _stage(self, work: Path, key: str, seed: int, ctx):
        env = dict(os.environ, STUDIO_ROOT=str(STUDIO), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false")
        task = f"{key}.yaml"
        log = open(work / f"{key}.log", "w")
        proc = subprocess.Popen([sys.executable, str(HERE / "rig_runner.py"), str(work), task, str(seed)],
                                cwd=str(work), env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                try:
                    ctx.check()
                except Cancelled:
                    proc.kill()
                    raise
                time.sleep(0.5)
        finally:
            log.close()
        tail = (work / f"{key}.log").read_text(errors="replace")[-1500:]
        if proc.returncode != 0:
            ctx.log(tail)
            raise RuntimeError(f"UniRig {key.split('_')[0]} stage failed: {tail.strip().splitlines()[-1][:300] if tail.strip() else proc.returncode}")


# ------------------------------------------------------------------------------------------------------ geometry helpers
def _load_meshes(path):
    """GLB -> list of trimesh.Trimesh with the scene transforms applied (materials and vertex attributes kept)."""
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    ms = [g for g in sc.dump() if len(getattr(g, "faces", [])) > 0]
    if not ms:
        raise InputError("The file has no triangle meshes.")
    return ms


def _offsets(ms):
    off, o = [], 0
    for m in ms:
        off.append(o)
        o += len(m.vertices)
    return off


def _to_blender(v):
    """glTF (+Y up, +Z front) -> Blender (+Z up, -Y front), the frame UniRig is trained in."""
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)


def _from_blender(v):
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def _norm(v):
    lo, hi = v.min(0), v.max(0)
    return (lo + hi) / 2, float(np.max(hi - lo) / 2) or 1.0


def _decimate(v, f, max_faces, ctx):
    if len(f) <= max_faces:
        return v, f
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=np.asarray(f, dtype="int32")))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(max_faces), preservenormal=True)
        m = ms.current_mesh()
        ctx.log(f"decimated for rigging {len(f)} -> {m.face_number()} faces")
        return m.vertex_matrix(), m.face_matrix()
    except Exception as e:  # noqa: BLE001
        ctx.log(f"decimation skipped: {e}")
        return v, f


def _write_raw(work: Path, v, f):
    """UniRig raw_data.npz for one mesh (no skeleton)."""
    import trimesh
    d = work / "dataset_inference_clean" / "job"
    d.mkdir(parents=True, exist_ok=True)
    m = trimesh.Trimesh(v, f, process=False)
    np.savez(d / "raw_data.npz", vertices=v.astype(np.float32), vertex_normals=np.asarray(m.vertex_normals, dtype=np.float32),
             faces=np.asarray(f, dtype=np.int64), face_normals=np.asarray(m.face_normals, dtype=np.float32), joints=None,
             tails=None, skin=None, no_skin=None, parents=None, names=None, matrix_local=None, path=None, cls=None)
    (work / "dataset_inference_clean" / "inference_datalist.txt").write_text("job\n")


def _make_configs(work: Path, p: dict):
    """Copy UniRig's configs/ next to the job and write the two task files (npz-only writers, no FBX / OBJ export)."""
    import yaml
    shutil.copytree(UNIRIG / "configs", work / "configs", dirs_exist_ok=True)
    # flash-attn is not available: the OPT decoder runs on PyTorch SDPA instead
    for f in (work / "configs" / "model").glob("*.yaml"):
        t = f.read_text()
        if "flash_attention_2" in t:
            f.write_text(t.replace("flash_attention_2", "sdpa"))
    # data: point at this job
    (work / "configs" / "data" / "quick_inference.yaml").write_text(
        (UNIRIG / "configs" / "data" / "quick_inference.yaml").read_text())
    sysf = work / "configs" / "system" / "ar_inference_articulationxl.yaml"
    sy = yaml.safe_load(sysf.read_text())
    gk = sy["generate_kwargs"]
    gk["num_beams"] = int(p.get("num_beams", 15))
    gk["temperature"] = float(p.get("temperature", 1.5))
    sysf.write_text(yaml.safe_dump(sy))
    tf = work / "configs" / "transform" / "inference_skin_transform.yaml"
    txt = tf.read_text().replace("grid: 196", f"grid: {int(p.get('skin_grid', 196))}")
    tf.write_text(txt)
    sk = yaml.safe_load((UNIRIG / "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml").read_text())
    sk["writer"].update({"export_obj": None, "export_fbx": None, "export_npz": "predict_skeleton"})
    (work / "skeleton_task.yaml").write_text(yaml.safe_dump(sk))
    sn = yaml.safe_load((UNIRIG / "configs/task/quick_inference_unirig_skin.yaml").read_text())
    sn["writer"].update({"export_fbx": None, "export_npz": "predict_skin", "output_dir": None})
    (work / "skin_task.yaml").write_text(yaml.safe_dump(sn))


# ------------------------------------------------------------------------------------------------------------ glTF skin
def _write_rigged_glb(meshes, skin, joints, parents, names, dst: Path):
    """Export `meshes` (same vertex order as `skin`) as GLB, then add a glTF skeleton + skin with JOINTS_0 / WEIGHTS_0."""
    import pygltflib as G
    import trimesh

    sc = trimesh.Scene()
    for i, m in enumerate(meshes):
        sc.add_geometry(m, node_name=f"part{i}", geom_name=f"part{i}")
    tmp = dst.with_suffix(".base.glb")
    sc.export(str(tmp))
    g = G.GLTF2().load_binary(str(tmp))
    blob = bytearray(g.binary_blob() or b"")

    J = len(names)
    # top-4 weights per vertex
    k = min(4, J)
    idx = np.argsort(-skin, axis=1)[:, :k]
    w = np.take_along_axis(skin, idx, axis=1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-8)
    if k < 4:
        idx = np.pad(idx, ((0, 0), (0, 4 - k)))
        w = np.pad(w, ((0, 0), (0, 4 - k)))
    idx = idx.astype(np.uint16)
    w = w.astype(np.float32)

    def add_view(data: bytes, target=None) -> int:
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(data)
        g.bufferViews.append(G.BufferView(buffer=0, byteOffset=off, byteLength=len(data), target=target))
        return len(g.bufferViews) - 1

    def add_acc(view, ctype, count, atype, **kw) -> int:
        g.accessors.append(G.Accessor(bufferView=view, componentType=ctype, count=count, type=atype, **kw))
        return len(g.accessors) - 1

    # skeleton nodes (local translation = joint - parent joint)
    first_joint = len(g.nodes)
    for j in range(J):
        t = joints[j] - (joints[parents[j]] if parents[j] is not None else 0.0)
        g.nodes.append(G.Node(name=names[j] or f"bone_{j}", translation=[float(x) for x in t]))
    roots = []
    for j in range(J):
        if parents[j] is None:
            roots.append(first_joint + j)
        else:
            n = g.nodes[first_joint + parents[j]]
            n.children = (n.children or []) + [first_joint + j]
    ibm = np.zeros((J, 4, 4), dtype=np.float32)
    for j in range(J):
        ibm[j] = np.eye(4, dtype=np.float32)
        ibm[j][:3, 3] = -joints[j]
    ibm_acc = add_acc(add_view(np.ascontiguousarray(ibm.transpose(0, 2, 1)).astype("<f4").tobytes()), G.FLOAT, J, G.MAT4)
    g.skins.append(G.Skin(inverseBindMatrices=ibm_acc, joints=list(range(first_joint, first_joint + J)),
                          skeleton=roots[0] if roots else None, name="UniRig"))
    skin_index = len(g.skins) - 1

    # per mesh primitive: vertex ranges in the same order as the input meshes
    offs = [0]
    for m in meshes:
        offs.append(offs[-1] + len(m.vertices))
    for ni, node in enumerate(list(g.nodes[:first_joint])):
        if node.mesh is None:
            continue
        mesh = g.meshes[node.mesh]
        part = int(str(node.name or mesh.name or "part0").replace("part", "").split("_")[0].split(".")[0] or 0)
        start = offs[part]
        for prim in mesh.primitives:
            n_verts = g.accessors[prim.attributes.POSITION].count
            sl = slice(start, start + n_verts)
            jv = add_view(np.ascontiguousarray(idx[sl]).astype("<u2").tobytes(), G.ARRAY_BUFFER)
            wv = add_view(np.ascontiguousarray(w[sl]).astype("<f4").tobytes(), G.ARRAY_BUFFER)
            prim.attributes.JOINTS_0 = add_acc(jv, G.UNSIGNED_SHORT, n_verts, G.VEC4)
            prim.attributes.WEIGHTS_0 = add_acc(wv, G.FLOAT, n_verts, G.VEC4)
            start += n_verts
        node.skin = skin_index
    for scn in g.scenes:
        scn.nodes = list(scn.nodes) + roots
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save_binary(str(dst))
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    serve(RigBackend())
