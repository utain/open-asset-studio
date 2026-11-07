"""Rig editing (workers/rig_edit.py), tool:rig.edit + the skeleton endpoint (app/plugins/rig.py), the re-skin bridge to UniRig.
No GPU, no weights: skinned GLBs are built here by hand, with consistent inverse bind matrices and several influences per vertex,
and every edit is checked by skinning the output again with an independent evaluator (rest pose must not move)."""
import json
import sys
import textwrap
import time

import numpy as np
import pytest

from app import config
from app.assets import AssetStore
from app.config import Settings
from app.jobs import TERMINAL, JobManager
from app.memory import MemoryManager
from app.registry import Registry
from app.workers import WorkerPool

sys.path.insert(0, str(config.STUDIO / "workers"))
import rig_anim as ra  # noqa: E402
import rig_edit as rE  # noqa: E402
from test_motion import humanoid, unirig_skeleton  # noqa: E402


# ------------------------------------------------------------------ synthetic skinned GLB
def skinned_glb(P, par, names=None, *, quant=False, bind=None, armature=(0.5, 0.0, 0.25), anim=False, extra_mesh_node=False):
    """-> glb bytes. Joint i is node i+2 under an 'armature' node (node 1, not a joint, translated); node 0 is the skinned mesh.
    IBM_i = inv(world_i) @ bind (bind = a rigid transform, default identity). 4 vertices per joint, 3 influences each."""
    P = np.asarray(P, dtype=float)
    J = len(P)
    names = names or [f"bone_{i}" for i in range(J)]
    arm = np.asarray(armature, dtype=float)
    nodes = [{"name": "mesh", "mesh": 0, "skin": 0}, {"name": "Armature", "translation": [float(x) for x in arm]}]
    for i in range(J):
        t = P[i] - (P[par[i]] if par[i] is not None else 0)
        nodes.append({"name": names[i], "translation": [float(x) for x in t]})
    for i, p in enumerate(par):
        nodes[(p + 2) if p is not None else 1].setdefault("children", []).append(i + 2)
    root = par.index(None)
    rng = np.random.default_rng(5)
    verts, jj, ww = [], [], []
    kids = {i: [c for c in range(J) if par[c] == i] for i in range(J)}
    for i in range(J):
        for _ in range(4):
            verts.append(P[i] + arm + rng.normal(0, 0.03, 3))
            infl = [i, par[i] if par[i] is not None else (kids[i][0] if kids[i] else i), (kids[i][0] if kids[i] else 0)]
            infl = list(dict.fromkeys(infl))                       # distinct joints only
            w = np.array([0.6, 0.3, 0.1][:len(infl)])
            w = w / w.sum()
            jj.append(infl + [0] * (4 - len(infl)))
            ww.append(list(w) + [0.0] * (4 - len(infl)))
    verts = np.array(verts, dtype="<f4")
    jj = np.array(jj, dtype="<u1" if quant else "<u2")
    ww = np.array(ww, dtype="<f4")
    if quant:
        ww = np.rint(ww * 255).astype("<u1")
        ww[:, 0] += (255 - ww.sum(1)).astype("<u1")                # make each row sum to exactly 255
    B = np.eye(4) if bind is None else np.asarray(bind, dtype=float)
    ibm = np.zeros((J, 4, 4), dtype="<f4")
    for i in range(J):
        G = np.eye(4)
        G[:3, 3] = P[i] + arm
        ibm[i] = np.linalg.inv(G) @ B
    bin_, views, accs = bytearray(), [], []

    def add(arr, ctype, atype, target=None, **kw):
        while len(bin_) % 4:
            bin_.append(0)
        views.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": arr.nbytes, **({"target": target} if target else {})})
        bin_.extend(arr.tobytes())
        accs.append({"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": atype, **kw})
        return len(accs) - 1

    pa = add(verts, 5126, "VEC3", 34962, min=[float(x) for x in verts.min(0)], max=[float(x) for x in verts.max(0)])
    n = len(verts)
    ia = add(np.arange(n - n % 3, dtype="<u2"), 5123, "SCALAR", 34963)
    ja = add(jj, 5121 if quant else 5123, "VEC4", 34962)
    wa = add(ww, 5121 if quant else 5126, "VEC4", 34962, **({"normalized": True} if quant else {}))
    ib = add(np.ascontiguousarray(ibm.transpose(0, 2, 1)), 5126, "MAT4")
    prims = [{"attributes": {"POSITION": pa, "JOINTS_0": ja, "WEIGHTS_0": wa}, "indices": ia}]
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": nodes, "meshes": [{"primitives": prims}],
          "skins": [{"inverseBindMatrices": ib, "joints": list(range(2, J + 2)), "skeleton": root + 2}]}
    js["bufferViews"], js["accessors"], js["buffers"] = views, accs, [{"byteLength": len(bin_)}]
    glb = ra.write_glb(js, bytes(bin_))
    if anim:
        glb, _ = ra.add_animations(glb)
    return glb


def world_of(js, n, memo=None):
    parent = {c: p for p, nd in enumerate(js["nodes"]) for c in nd.get("children") or []}
    chain, p = [], n
    while p is not None:
        chain.append(p)
        p = parent.get(p)
    m = np.eye(4)
    for c in reversed(chain):
        t, q, s = ra._trs(js["nodes"][c])
        loc = np.eye(4)
        loc[:3, :3] = ra.q_to_mat(q) * s[None, :]
        loc[:3, 3] = t
        m = m @ loc
    return m


def rest_skin(glb):
    """Independent skinning of the rest pose: sum_k w_k * (world(joint_k) @ IBM_k) @ v, from the file alone."""
    js, binary = ra.read_glb(glb)

    def acc(i):
        a = js["accessors"][i]
        v = js["bufferViews"][a["bufferView"]]
        dt = {5121: "u1", 5123: "<u2", 5126: "<f4"}[a["componentType"]]
        n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        arr = np.frombuffer(binary, dtype=dt, count=a["count"] * n, offset=v.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(a["count"], n)
        return arr.astype(np.float64), a
    sk = js["skins"][0]
    ibm = acc(sk["inverseBindMatrices"])[0].reshape(-1, 4, 4).transpose(0, 2, 1)
    jm = np.array([world_of(js, j) @ ibm[k] for k, j in enumerate(sk["joints"])])
    out = []
    for prim in js["meshes"][0]["primitives"]:
        at = prim["attributes"]
        v = acc(at["POSITION"])[0]
        j, _ = acc(at["JOINTS_0"])
        w, wa = acc(at["WEIGHTS_0"])
        if wa.get("normalized"):
            w = w / 255.0
        vh = np.concatenate([v, np.ones((len(v), 1))], 1)
        res = np.zeros((len(v), 3))
        for k in range(4):
            M = jm[j[:, k].astype(int)]
            res += w[:, k:k + 1] * np.einsum("nij,nj->ni", M, vh)[:, :3]
        out.append(res)
    return np.concatenate(out)


def weights_of(glb):
    js, binary = ra.read_glb(glb)
    at = js["meshes"][0]["primitives"][0]["attributes"]

    def acc(i):
        a = js["accessors"][i]
        v = js["bufferViews"][a["bufferView"]]
        dt = {5121: "u1", 5123: "<u2", 5126: "<f4"}[a["componentType"]]
        return np.frombuffer(binary, dtype=dt, count=a["count"] * 4, offset=v.get("byteOffset", 0)).reshape(a["count"], 4), a
    j, _ = acc(at["JOINTS_0"])
    w, wa = acc(at["WEIGHTS_0"])
    w = w.astype(np.float64) / (255.0 if wa.get("normalized") else 1.0)
    return j.astype(int), w


def joint_names(glb):
    js, _ = ra.read_glb(glb)
    return [js["nodes"][j]["name"] for j in js["skins"][0]["joints"]]


def joint_pos(glb):
    js, _ = ra.read_glb(glb)
    return {js["nodes"][j]["name"]: world_of(js, j)[:3, 3] for j in js["skins"][0]["joints"]}


def parent_names(glb):
    js, _ = ra.read_glb(glb)
    node_parent = {c: p for p, nd in enumerate(js["nodes"]) for c in nd.get("children") or []}
    joints = set(js["skins"][0]["joints"])
    out = {}
    for j in js["skins"][0]["joints"]:
        p = node_parent.get(j)
        while p is not None and p not in joints:
            p = node_parent.get(p)
        out[js["nodes"][j]["name"]] = js["nodes"][p]["name"] if p is not None else None
    return out


def check_structure(glb):
    """Every accessor / bufferView in range, every node reference valid, skin joints are nodes, JOINTS_0 < joint count."""
    js, binary = ra.read_glb(glb)
    assert js["buffers"][0]["byteLength"] == len(binary)
    for v in js["bufferViews"]:
        assert v["byteOffset"] % 4 == 0 and v["byteOffset"] + v["byteLength"] <= len(binary)
    n = len(js["nodes"])
    for nd in js["nodes"]:
        assert all(0 <= c < n for c in nd.get("children") or [])
    for sc in js["scenes"]:
        assert all(0 <= c < n for c in sc["nodes"])
    sk = js["skins"][0]
    assert all(0 <= j < n for j in sk["joints"]) and len(set(sk["joints"])) == len(sk["joints"])
    ibm = js["accessors"][sk["inverseBindMatrices"]]
    assert ibm["count"] == len(sk["joints"])
    j, w = weights_of(glb)
    assert j.max() < len(sk["joints"])
    parents = {c: p for p, nd in enumerate(js["nodes"]) for c in nd.get("children") or []}
    assert len(parents) == sum(len(nd.get("children") or []) for nd in js["nodes"])          # nobody has two parents
    for a in js.get("animations") or []:
        assert all(c["target"]["node"] < n for c in a["channels"])
    return js


HUM, HPAR, HNAMES = humanoid(extras=True)
SIMPLE = skinned_glb(HUM, HPAR, HNAMES)


def assert_rest_unchanged(before, after, tol=2e-5):
    assert np.abs(rest_skin(before) - rest_skin(after)).max() < tol


# ------------------------------------------------------------------ describe
def test_describe_lists_bones_roles_and_weight_share():
    d = rE.describe(SIMPLE)
    assert d["count"] == len(HUM) and d["humanoid"]["ok"] and d["weighted"]
    by = {r["name"]: r for r in d["joints"]}
    assert by["pelvis"]["role"] == "hips" and by["L_elbow"]["role"] == "left_forearm" and by["head"]["role"] == "head"
    assert by["pelvis"]["parent"] is None and by["head"]["parent"] == HNAMES.index("neck")
    assert np.allclose(by["L_elbow"]["position"], HUM[HNAMES.index("L_elbow")] + [0.5, 0, 0.25], atol=1e-4)     # world, armature included
    assert by["pelvis"]["children"] >= 3 and abs(sum(r["weight_share"] for r in d["joints"]) - 1) < 1e-3
    assert d["humanoid"]["roles"]["left_forearm"] == "L_elbow" and d["humanoid"]["facing"] == "+Z"
    assert d["zero_weight"] == [] and "left side is +X" in d["units"]


def test_describe_non_humanoid_and_no_skin():
    glb = skinned_glb(*(np.array([[0, 1, 0], [0, 1.3, 0], [0.2, 1.3, 0], [0, 1.6, 0], [-0.2, 1.3, 0]]), [None, 0, 1, 1, 1]))
    d = rE.describe(glb)
    assert d["humanoid"]["ok"] is False and "joints" in d["humanoid"]["reason"] and all(r["role"] is None for r in d["joints"])
    js, b = ra.read_glb(glb)
    js.pop("skins")
    with pytest.raises(ra.NoSkinError):
        rE.describe(ra.write_glb(js, b))
    with pytest.raises(ra.RigAnimError):
        rE.describe(b"not a glb at all, really not a glb")


# ------------------------------------------------------------------ bone references
def test_resolve_names_indices_and_role_words():
    doc = rE.RigDoc(SIMPLE)
    n = lambda ref: doc.names[doc.resolve(ref)]      # noqa: E731
    assert n("L_elbow") == "L_elbow" and n("l_elbow") == "L_elbow" and n(HNAMES.index("L_wrist")) == "L_wrist" and n("#3") == HNAMES[3]
    for ref in ("left elbow", "left forearm", "left lower arm", "L forearm", "forearm left", "leftForeArm", "Left_Elbow"):
        assert n(ref) == "L_elbow", ref
    assert n("right knee") == "R_knee" and n("left wrist") == "L_handtip" and n("left hand") == "L_handtip" and n("left ankle") == "L_ankle" and n("left toe") == "L_toe"
    assert n("hips") == "pelvis" and n("pelvis") == "pelvis" and n("hip") == "pelvis" and n("left hip") == "L_hip" and n("head") == "head"
    assert n("chest") == "chest" and n("neck") == "neck" and n("root") == "pelvis" and n("spine 2") == "spine2" and n("spine1") == "spine1"
    assert n("left shoulder") == "L_collar" and n("left upper arm") == "L_shoulder" and n("clavicle right") == "R_collar"


def test_resolve_errors_list_candidates():
    doc = rE.RigDoc(SIMPLE)
    with pytest.raises(rE.RigEditError, match=r"needs a side.*L_elbow.*R_elbow"):
        doc.resolve("elbow")
    with pytest.raises(rE.RigEditError, match=r"ambiguous.*spine_1.*spine_2|ambiguous.*spine1.*spine2"):
        doc.resolve("spine")
    with pytest.raises(rE.RigEditError, match=r"No bone matches.*Bones: pelvis"):
        doc.resolve("tentacle")
    with pytest.raises(rE.RigEditError, match="does not exist"):
        doc.resolve(999)
    with pytest.raises(rE.RigEditError):
        doc.resolve("")
    with pytest.raises(rE.RigEditError):
        doc.resolve(["x"])
    with pytest.raises(rE.RigEditError, match="left toe|no left"):
        rE.RigDoc(skinned_glb(*humanoid(toes=False))).resolve("left toe")
    # substring match: unique ok, several -> ambiguous with candidates
    assert doc.names[doc.resolve("skirt2")] == "skirt2"
    with pytest.raises(rE.RigEditError, match=r"ambiguous.*tail1.*tail2|ambiguous.*tail"):
        doc.resolve("tail")


def test_role_words_need_a_humanoid_and_duplicate_names_are_ambiguous():
    P = np.array([[0, 1, 0], [0, 1.3, 0], [0.2, 1.3, 0], [0, 1.6, 0], [-0.2, 1.3, 0]])
    doc = rE.RigDoc(skinned_glb(P, [None, 0, 1, 1, 1], ["a", "b", "arm", "arm", "z"]))
    with pytest.raises(rE.RigEditError, match="not recognised as a humanoid|No bone matches"):
        doc.resolve("left forearm")
    with pytest.raises(rE.RigEditError, match=r"ambiguous.*arm \(#2.*arm \(#3"):
        doc.resolve("arm")
    assert doc.names[doc.resolve(3)] == "arm"
    with pytest.raises(rE.RigEditError, match="unique"):
        rE.skeleton_arrays(skinned_glb(P, [None, 0, 1, 1, 1], ["a", "b", "arm", "arm", "z"]))


# ------------------------------------------------------------------ rename
def test_rename_keeps_rest_pose_clips_and_refuses_duplicates():
    glb = skinned_glb(HUM, HPAR, HNAMES, anim=True)
    out, rep = rE.edit(glb, [{"op": "rename", "bone": "left elbow", "to": "Elbow.L"}, {"op": "rename", "bone": "Elbow.L", "to": "Elbow_Left"}])
    assert joint_names(out)[HNAMES.index("L_elbow")] == "Elbow_Left" and not rep["needs_reskin"] and not rep["reskin_suggested"]
    assert_rest_unchanged(glb, out)
    assert ra.animation_names(out) == ["flex", "wiggle"] and rep["animations_dropped"] == []        # a rename keeps the clips
    for bad in ({"op": "rename", "bone": "head", "to": "neck"}, {"op": "rename", "bone": "head", "to": "  "}, {"op": "rename", "bone": "head", "to": "x" * 80}):
        with pytest.raises(rE.RigEditError):
            rE.edit(glb, [bad])


@pytest.mark.parametrize("scheme,expect", [
    ("mixamo", {"pelvis": "mixamorig:Hips", "L_elbow": "mixamorig:LeftForeArm", "R_hip": "mixamorig:RightUpLeg", "head": "mixamorig:Head",
                "L_collar": "mixamorig:LeftShoulder", "L_shoulder": "mixamorig:LeftArm", "L_toe": "mixamorig:LeftToeBase",
                "spine1": "mixamorig:Spine", "spine2": "mixamorig:Spine1", "chest": "mixamorig:Spine2", "neck": "mixamorig:Neck",
                "L_knee": "mixamorig:LeftLeg", "R_ankle": "mixamorig:RightFoot", "R_handtip": "mixamorig:RightHand", "R_wrist": "R_wrist", "tail1": "tail1"}),
    ("vrm", {"pelvis": "hips", "L_elbow": "leftLowerArm", "R_hip": "rightUpperLeg", "head": "head", "L_collar": "leftShoulder",
             "L_shoulder": "leftUpperArm", "L_toe": "leftToes", "spine1": "spine", "spine2": "chest", "chest": "upperChest", "neck": "neck"}),
    ("unreal", {"pelvis": "pelvis", "L_elbow": "lowerarm_l", "R_hip": "thigh_r", "L_knee": "calf_l", "L_toe": "ball_l", "L_collar": "clavicle_l",
                "spine1": "spine_01", "spine2": "spine_02", "chest": "spine_03", "neck": "neck_01", "head": "head"}),
])
def test_rename_standard(scheme, expect):
    out, rep = rE.edit(SIMPLE, [{"op": "rename_standard", "scheme": scheme}])
    names = joint_names(out)
    for old, new in expect.items():
        assert names[HNAMES.index(old)] == new, (old, names[HNAMES.index(old)])
    assert len(set(names)) == len(names) and not rep["needs_reskin"]
    assert_rest_unchanged(SIMPLE, out)
    # the standard names are role words again: a second pass is stable and left/right stay on the right bones
    d = rE.describe(out)
    assert d["humanoid"]["ok"] and d["humanoid"]["roles"]["left_forearm"] == expect["L_elbow"]


def test_rename_standard_on_the_real_unirig_skeleton_and_name_collisions():
    P, par, _ = unirig_skeleton()
    glb = skinned_glb(P, par)
    out, _ = rE.edit(glb, [{"op": "rename_standard", "scheme": "mixamo"}])
    names = joint_names(out)
    assert names[0] == "mixamorig:Hips" and names[5] == "mixamorig:Head" and names[8] == "mixamorig:LeftForeArm"
    # a bone that already uses a target name moves out of the way
    nm = [f"bone_{i}" for i in range(len(P))]
    nm[1] = "mixamorig:Hips"
    out2, _ = rE.edit(skinned_glb(P, par, nm), [{"op": "rename_standard", "scheme": "mixamo"}])
    n2 = joint_names(out2)
    assert n2[0] == "mixamorig:Hips" and n2[1] != "mixamorig:Hips" and len(set(n2)) == len(n2)


def test_rename_standard_refuses_non_humanoids_clearly():
    P = np.array([[0, 1, 0], [0, 1.3, 0], [0.2, 1.3, 0], [0, 1.6, 0], [-0.2, 1.3, 0]])
    glb = skinned_glb(P, [None, 0, 1, 1, 1])
    with pytest.raises(rE.RigEditError, match="humanoid") as e:
        rE.edit(glb, [{"op": "rename_standard", "scheme": "mixamo"}])
    assert e.value.kind == "input"
    with pytest.raises(rE.RigEditError, match="scheme"):
        rE.edit(SIMPLE, [{"op": "rename_standard", "scheme": "blender"}])


# ------------------------------------------------------------------ move_joint
def test_move_joint_stay_moves_only_the_pivot_and_keeps_the_rest_pose():
    glb = skinned_glb(HUM, HPAR, HNAMES, anim=True, bind=_rigid())
    before = joint_pos(glb)
    out, rep = rE.edit(glb, [{"op": "move_joint", "bone": "left elbow", "offset": [0.0, -0.05, 0.02]}])
    after = joint_pos(out)
    for k, v in before.items():
        if k == "L_elbow":
            assert np.allclose(after[k], v + [0, -0.05, 0.02], atol=1e-6)
        else:
            assert np.allclose(after[k], v, atol=1e-6), k          # children stay where they were
    assert_rest_unchanged(glb, out)
    check_structure(out)
    assert rep["reskin_suggested"] and not rep["needs_reskin"]
    assert rep["animations_dropped"] == ["flex", "wiggle"] and rep["animations_rebuilt"] == ["flex", "wiggle"]
    assert ra.animation_names(out) == ["flex", "wiggle"]


def _rigid():
    th = 0.4
    M = np.eye(4)
    M[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    M[:3, 3] = [0.1, -0.2, 0.3]
    return M


def test_move_joint_follow_moves_the_subtree_and_modes():
    before = joint_pos(SIMPLE)
    out, _ = rE.edit(SIMPLE, [{"op": "move_joint", "bone": "L_shoulder", "offset": [0, 0.1, 0], "children": "follow"}])
    after = joint_pos(out)
    sub = {"L_shoulder", "L_elbow", "L_wrist", "L_handtip"}
    for k, v in before.items():
        assert np.allclose(after[k], v + ([0, 0.1, 0] if k in sub else 0), atol=1e-6), k
    assert_rest_unchanged(SIMPLE, out)
    # delta is a fraction of the bone length (distance to the parent), to is absolute
    o2, _ = rE.edit(SIMPLE, [{"op": "move_joint", "bone": 7, "delta": [0, -0.5, 0]}, {"op": "move_joint", "bone": "head", "to": [1, 2, 3]}])
    a2 = joint_pos(o2)
    h = 7                                                      # joint #7 by index; its parent is HPAR[7]
    assert np.allclose(a2[HNAMES[h]] - before[HNAMES[h]], [0, -0.5 * np.linalg.norm(before[HNAMES[h]] - before[HNAMES[HPAR[h]]]), 0], atol=1e-6)
    assert np.allclose(a2["head"], [1, 2, 3], atol=1e-6)
    for bad in ({"op": "move_joint", "bone": "head"}, {"op": "move_joint", "bone": "head", "offset": [0, 1, 0], "to": [0, 0, 0]},
                {"op": "move_joint", "bone": "head", "offset": [0, 1]}, {"op": "move_joint", "bone": "head", "offset": [0, 1, 0], "children": "x"},
                {"op": "move_joint", "bone": "head", "offset": [0, float("nan"), 0]}):
        with pytest.raises(rE.RigEditError):
            rE.edit(SIMPLE, [bad])


def test_move_joint_mirror_moves_the_opposite_side():
    before = joint_pos(SIMPLE)
    out, rep = rE.edit(SIMPLE, [{"op": "move_joint", "bone": "left elbow", "offset": [0.03, -0.04, 0.01], "mirror": True}])
    a = joint_pos(out)
    assert np.allclose(a["L_elbow"] - before["L_elbow"], [0.03, -0.04, 0.01], atol=1e-6)
    assert np.allclose(a["R_elbow"] - before["R_elbow"], [-0.03, -0.04, 0.01], atol=1e-6) and "mirrored" in rep["ops"][0]["summary"]
    out2, rep2 = rE.edit(SIMPLE, [{"op": "move_joint", "bone": "head", "offset": [0, 0.01, 0], "mirror": True}])      # on the centre line
    assert "no mirrored joint" in rep2["ops"][0]["summary"]
    assert_rest_unchanged(SIMPLE, out)


# ------------------------------------------------------------------ delete
def test_delete_bone_merges_weights_into_the_parent():
    j0, w0 = weights_of(SIMPLE)
    out, rep = rE.edit(SIMPLE, [{"op": "delete_bone", "bone": "left elbow"}])
    names = joint_names(out)
    assert "L_elbow" not in names and len(names) == len(HUM) - 1 and rep["joints_after"] == len(HUM) - 1
    assert parent_names(out)["L_wrist"] == "L_shoulder"                       # the child is re-attached, same world position
    assert np.allclose(joint_pos(out)["L_wrist"], joint_pos(SIMPLE)["L_wrist"], atol=1e-6)
    j1, w1 = weights_of(out)
    assert np.allclose(w1.sum(1), w0.sum(1), atol=1e-6) and np.allclose(w1.sum(1), 1.0, atol=1e-6)
    assert j1.max() < len(names)
    assert_rest_unchanged(SIMPLE, out)
    check_structure(out)
    # the weights that were on the elbow are now on the shoulder (by name)
    old_el, old_sh = HNAMES.index("L_elbow"), HNAMES.index("L_shoulder")
    new_sh = names.index("L_shoulder")
    mask = (j0 == old_el) & (w0 > 0)
    assert mask.any()
    rows = np.where(mask.any(1))[0]
    for r in rows:
        share_sh = w1[r][j1[r] == new_sh].sum()
        assert share_sh >= w0[r][j0[r] == old_el].sum() - 1e-6
    # no vertex lists a joint twice
    for r in range(len(j1)):
        used = j1[r][w1[r] > 0]
        assert len(set(used)) == len(used)
    assert rep["reskin_suggested"] and not rep["needs_reskin"]


def test_delete_with_children_glob_and_merge_target():
    out, rep = rE.edit(SIMPLE, [{"op": "delete_bone", "match": "tail*", "include_children": True}, {"op": "delete_bone", "bone": "skirt1", "include_children": True}])
    names = joint_names(out)
    assert not any(n.startswith(("tail", "skirt")) for n in names) and len(names) == len(HUM) - 5
    j, w = weights_of(out)
    assert np.allclose(w.sum(1), 1.0, atol=1e-6) and j.max() < len(names)
    assert_rest_unchanged(SIMPLE, out)
    check_structure(out)
    out2, _ = rE.edit(SIMPLE, [{"op": "delete_bone", "bone": "L_wrist", "merge_weights_to": "chest"}])
    j2, w2 = weights_of(out2)
    n2 = joint_names(out2)
    assert w2[j2 == n2.index("chest")].sum() > weights_of(SIMPLE)[1][weights_of(SIMPLE)[0] == HNAMES.index("chest")].sum()
    assert_rest_unchanged(SIMPLE, out2)
    for bad in ({"op": "delete_bone"}, {"op": "delete_bone", "match": "nothing*"}, {"op": "delete_bone", "bone": "head", "merge_weights_to": "head", "include_children": True},
                {"op": "delete_bone", "bone": "pelvis"}):
        with pytest.raises(rE.RigEditError):
            rE.edit(SIMPLE, [bad])
    # deleting everything is refused
    with pytest.raises(rE.RigEditError, match="every bone"):
        rE.edit(SIMPLE, [{"op": "delete_bone", "bone": "pelvis", "include_children": True, "merge_weights_to": "pelvis"}])


def test_delete_the_root_with_one_child_and_quantized_weights():
    P = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0.5, 2, 0]])
    par = [None, 0, 1, 2]
    glb = skinned_glb(P, par)
    out, _ = rE.edit(glb, [{"op": "delete_bone", "bone": "bone_0"}])
    assert parent_names(out)["bone_1"] is None
    assert_rest_unchanged(glb, out)
    check_structure(out)
    P2 = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0.5, 2, 0], [-0.5, 2, 0]])
    with pytest.raises(rE.RigEditError, match="root"):
        rE.edit(skinned_glb(P2, [None, 0, 0, 0, 0]), [{"op": "delete_bone", "bone": "bone_0"}])
    q = skinned_glb(HUM, HPAR, HNAMES, quant=True)
    out_q, _ = rE.edit(q, [{"op": "delete_bone", "bone": "L_elbow"}, {"op": "delete_bone", "bone": "R_elbow"}])
    j, w = weights_of(out_q)
    assert np.abs(w.sum(1) - 1).max() < 1e-6 and j.max() < len(HUM) - 2
    assert_rest_unchanged(q, out_q, tol=1e-4)


# ------------------------------------------------------------------ add bones
def test_add_bone_and_chain():
    glb = skinned_glb(HUM, HPAR, HNAMES, anim=True, bind=_rigid())
    out, rep = rE.edit(glb, [{"op": "add_chain", "parent": "hips", "count": 5, "direction": "back", "length": 0.1, "name_prefix": "tail"},
                             {"op": "add_bone", "parent": "head", "direction": [0, 1, 0], "length": 0.2, "name": "antenna"},
                             {"op": "add_bone", "parent": "tail_05", "to": [0.5, 0.3, -1.5]}])
    names = joint_names(out)
    assert names[-7:-2] == [f"tail_0{i}" for i in range(1, 6)] and names[-2] == "antenna" and len(names) == len(HUM) + 7
    pos, base = joint_pos(out), joint_pos(glb)
    for k in range(1, 6):
        assert np.allclose(pos[f"tail_0{k}"], base["pelvis"] + [0, 0, -0.1 * k], atol=1e-6)
    assert np.allclose(pos["antenna"], base["head"] + [0, 0.2, 0], atol=1e-6) and np.allclose(pos[names[-1]], [0.5, 0.3, -1.5], atol=1e-6)
    pn = parent_names(out)
    assert pn["tail_01"] == "pelvis" and pn["tail_03"] == "tail_02" and pn["antenna"] == "head" and pn[names[-1]] == "tail_05"
    assert rep["needs_reskin"] and rep["zero_weight_bones"][:2] == ["tail_01", "tail_02"] and len(rep["zero_weight_bones"]) == 7
    assert any("re-skin" in n for n in rep["notes"])
    assert_rest_unchanged(glb, out)
    js = check_structure(out)
    # new bones have a consistent inverse bind matrix and zero weight
    d = rE.describe(out)
    assert set(d["zero_weight"]) == set(rep["zero_weight_bones"]) and d["joints"][-1]["parent"] == names.index("tail_05")
    ibm = js["accessors"][js["skins"][0]["inverseBindMatrices"]]
    assert ibm["count"] == len(names)
    assert ra.animation_names(out) == ["flex", "wiggle"]
    # defaults: direction only -> half the parent's bone length; chain without direction continues the parent's direction
    o2, _ = rE.edit(SIMPLE, [{"op": "add_bone", "parent": "head", "direction": "up"}, {"op": "add_chain", "parent": "R_wrist", "count": 2, "length": 0.05}])
    p2, b2 = joint_pos(o2), joint_pos(SIMPLE)
    assert p2["bone_new_1"][1] > b2["head"][1]
    assert p2["chain_01"][0] < b2["R_wrist"][0]                     # continues outwards along the right arm
    for bad in ({"op": "add_bone", "parent": "head"}, {"op": "add_bone", "parent": "head", "to": [0, 0, 0], "direction": "up"},
                {"op": "add_bone", "parent": "head", "to": [float(b2["head"][0]), float(b2["head"][1]), float(b2["head"][2])]},
                {"op": "add_bone", "parent": "head", "direction": "sideways"}, {"op": "add_bone", "parent": "head", "direction": "up", "name": "neck"},
                {"op": "add_chain", "parent": "head", "count": 0, "direction": "up"}, {"op": "add_chain", "parent": "head", "count": 99, "direction": "up"},
                {"op": "add_chain", "parent": "head", "count": 2, "direction": [0, 0, 0]}):
        with pytest.raises(rE.RigEditError):
            rE.edit(SIMPLE, [bad])


def test_ops_see_the_result_of_earlier_ops():
    out, _ = rE.edit(SIMPLE, [{"op": "add_chain", "parent": "head", "count": 2, "direction": "up", "length": 0.05, "name_prefix": "ear"},
                              {"op": "rename", "bone": "ear_02", "to": "ear_tip"}, {"op": "move_joint", "bone": "ear_tip", "offset": [0, 0.01, 0]},
                              {"op": "delete_bone", "bone": "ear_01", "include_children": True}])
    assert "ear_tip" not in joint_names(out) and len(joint_names(out)) == len(HUM)
    with pytest.raises(rE.RigEditError, match="Operation 2"):
        rE.edit(SIMPLE, [{"op": "delete_bone", "bone": "tail1"}, {"op": "rename", "bone": 1, "to": "pelvis"}])
    with pytest.raises(rE.RigEditError, match="deleted by an earlier"):
        rE.edit(SIMPLE, [{"op": "delete_bone", "bone": HNAMES.index("tail3")}, {"op": "rename", "bone": HNAMES.index("tail3"), "to": "x"}])


def test_many_joints_upgrade_nothing_and_stay_valid():
    out, _ = rE.edit(skinned_glb(HUM, HPAR, HNAMES, quant=True),
                     [{"op": "add_chain", "parent": "pelvis", "count": 64, "direction": "back", "length": 0.01}] * 4 + [{"op": "delete_bone", "bone": "tail1"}])
    js = check_structure(out)
    assert len(js["skins"][0]["joints"]) == len(HUM) - 1 + 256
    assert_rest_unchanged(skinned_glb(HUM, HPAR, HNAMES, quant=True), out, tol=1e-4)


# ------------------------------------------------------------------ reparent
def test_reparent_keeps_world_positions_and_refuses_loops():
    out, rep = rE.edit(SIMPLE, [{"op": "reparent", "bone": "L_wrist", "parent": "chest"}])
    assert parent_names(out)["L_wrist"] == "chest" and np.allclose(joint_pos(out)["L_wrist"], joint_pos(SIMPLE)["L_wrist"], atol=1e-6)
    assert_rest_unchanged(SIMPLE, out)
    check_structure(out)
    assert rep["reskin_suggested"] and not rep["needs_reskin"]
    for bad in ({"op": "reparent", "bone": "chest", "parent": "head"}, {"op": "reparent", "bone": "head", "parent": "head"},
                {"op": "reparent", "bone": "pelvis", "parent": "head"}):
        with pytest.raises(rE.RigEditError):
            rE.edit(SIMPLE, [bad])


# ------------------------------------------------------------------ op validation
def test_bad_ops_are_explained():
    for ops, match in [([], "non-empty"), ("not json", "JSON"), ([{"op": "explode"}], "Known ops"), ([{"bone": "head"}], "\"op\""),
                       ([{"op": "rename", "bone": "head"}], "missing to"), ([{"op": "rename", "bone": "head", "to": "x", "colour": 1}], "unknown field"),
                       (["rename"], "object"), ([{"op": "move_joint", "bone": "head", "offest": [0, 1, 0]}], "offest")]:
        with pytest.raises(rE.RigEditError, match=match):
            rE.edit(SIMPLE, ops)
    with pytest.raises(rE.RigEditError, match="At most"):
        rE.edit(SIMPLE, [{"op": "rename", "bone": "head", "to": "h"}] * (rE.MAX_OPS + 1))
    out, _ = rE.edit(SIMPLE, '[{"op": "rename", "bone": "head", "to": "Kopf"}]')                # a JSON string works too
    assert "Kopf" in joint_names(out)
    out, _ = rE.edit(SIMPLE, {"op": "rename", "bone": "head", "to": "Kopf2"})                  # one op without the list is fine
    assert "Kopf2" in joint_names(out)


def test_external_buffer_and_multiple_skins_are_refused():
    js, b = ra.read_glb(SIMPLE)
    js["buffers"][0]["uri"] = "x.bin"
    with pytest.raises(ra.RigAnimError):
        rE.edit(ra.write_glb(js, b), [{"op": "rename", "bone": "head", "to": "h"}])
    js, b = ra.read_glb(SIMPLE)
    js["skins"].append({"joints": js["skins"][0]["joints"][:5]})
    with pytest.raises(rE.RigEditError, match="several different skins"):
        rE.edit(ra.write_glb(js, b), [{"op": "rename", "bone": "head", "to": "h"}])


def test_original_bytes_are_untouched():
    js0, b0 = ra.read_glb(SIMPLE)
    out, _ = rE.edit(SIMPLE, [{"op": "add_bone", "parent": "head", "direction": "up"}, {"op": "delete_bone", "bone": "tail3"}])
    _, b1 = ra.read_glb(out)
    # the meshes' position / index data are identical; joints / weights / ibm are rewritten or appended
    pa = js0["accessors"][js0["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    v = js0["bufferViews"][pa["bufferView"]]
    assert b1[v["byteOffset"]:v["byteOffset"] + v["byteLength"]] == b0[v["byteOffset"]:v["byteOffset"] + v["byteLength"]]
    assert len(out) < len(SIMPLE) + 4096


# ------------------------------------------------------------------ the UniRig bridge
def test_skeleton_arrays_and_unirig_helpers():
    sk = rE.skeleton_arrays(SIMPLE)
    assert sk["names"] == HNAMES and sk["parents"] == HPAR and np.allclose(sk["positions"], HUM + [0.5, 0, 0.25], atol=1e-5)
    assert rE.unirig_order(HPAR) == list(range(len(HPAR)))
    par = [2, None, 1, 0]                                     # root is not first, a parent comes after its child
    o = rE.unirig_order(par)
    assert o[0] == 1 and sorted(o) == [0, 1, 2, 3]
    inv = {x: i for i, x in enumerate(o)}
    assert all(par[x] is None or inv[par[x]] < inv[x] for x in o)
    with pytest.raises(rE.RigEditError, match="one root"):
        rE.unirig_order([None, None, 0])
    # tails: single child -> the child, leaf -> half a bone further, branch -> half its own bone along the parent direction
    X = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [-1, 2, 0]], dtype=float)
    pr = [None, 0, 1, 2, 2]
    t = rE.unirig_tails(X, pr)
    assert np.allclose(t[0], X[1]) and np.allclose(t[1], X[2]) and np.allclose(t[3], [1.5, 2, 0]) and np.allclose(t[2], [0, 2.5, 0])
    multi = rE.unirig_tails(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), [None, 0, 0])
    assert np.allclose(multi[0], [0, 0, 0.5 * 1.0])            # the root with several children: +Z by half the mean child distance
    # after an edit the arrays follow the new skeleton
    out, _ = rE.edit(SIMPLE, [{"op": "add_chain", "parent": "pelvis", "count": 3, "direction": "back", "length": 0.1, "name_prefix": "tail"}])
    sk2 = rE.skeleton_arrays(out)
    assert sk2["names"][-3:] == ["tail_01", "tail_02", "tail_03"] and sk2["parents"][-3:] == [0, len(HNAMES), len(HNAMES) + 1]


def test_rig_worker_reskin_writes_unirig_skeleton_npz(tmp_path):
    import rig_worker
    glb = tmp_path / "in.glb"
    out, _ = rE.edit(SIMPLE, [{"op": "add_chain", "parent": "pelvis", "count": 2, "direction": "back", "length": 0.1, "name_prefix": "tail"}])
    glb.write_bytes(out)
    pytest.importorskip("trimesh")
    v_g = np.array([[0.4, 1.0, 0.2], [0.6, 1.9, 0.3], [0.5, 0.1, 0.25], [0.9, 1.5, 0.25]])
    f = np.array([[0, 1, 2], [1, 2, 3]])
    v_b = rig_worker._to_blender(v_g)
    center, scale = rig_worker._norm(v_b)

    class Ctx:
        def log(self, m):
            pass
    work = tmp_path / "work"
    parents, nm = rig_worker.RigBackend()._write_given_skeleton(work, str(glb), v_b, f, center, scale, Ctx())
    sk = np.load(work / "dataset_inference_clean" / "job" / "predict_skeleton.npz", allow_pickle=True)
    assert [str(x) for x in sk["names"]] == nm == joint_names(out)
    assert [None if q is None else int(q) for q in sk["parents"]] == parents
    assert parents[0] is None and all(parents[i] < i for i in range(1, len(parents)))
    assert sk["joints"].shape == (len(nm), 3) and sk["tails"].shape == (len(nm), 3) and sk["vertices"].dtype == np.float32
    assert np.abs(sk["vertices"]).max() == pytest.approx(1.0, abs=1e-5)
    # the joints come back to the glTF frame with the worker's own inverse conversion
    back = rig_worker._from_blender(sk["joints"].astype(np.float64) * scale + center)
    assert np.allclose(back, rE.skeleton_arrays(out)["positions"], atol=1e-4)
    for key in ("skin", "no_skin", "matrix_local", "path", "cls"):
        assert sk[key][()] is None
    assert sk["faces"].shape == f.shape and sk["vertex_normals"].shape == sk["vertices"].shape and sk["face_normals"].shape == f.shape


def test_reskin_registry_entry_and_worker_task_check():
    reg = Registry(config.STUDIO / "models.yaml", Settings(config.STUDIO / "data" / "_none_settings.json"), probe_cache=config.STUDIO / "data" / "_none_probe.json")
    m = reg.models["unirig"]
    assert m["tasks"] == ["rig", "reskin"] and "reskin" in reg.tasks and reg.tasks["reskin"]["studio"] == "3d"
    assert reg.default_for("reskin") == "unirig" and reg.default_for("rig") == "unirig"
    assert {s["role"] for s in m["input_slots"]} == {"mesh"} and [s["tasks"] for s in m["input_slots"]] == [["rig"], ["reskin"]]
    p = reg.validate_params("unirig", {})
    assert p["skin_grid"] in (96, 128, 196) and "max_faces" in p
    assert [x["tasks"] for x in m["params"] if x["name"] in ("num_beams", "temperature")] == [["rig"], ["rig"]]
    assert not reg.errors


# ------------------------------------------------------------------ tool:rig.edit through the job queue
@pytest.fixture
def mgr(tmp_path):
    root = config.STUDIO
    (tmp_path / "models.yaml").write_text(textwrap.dedent(f"""
    backends:
      echo_cpu: {{python: {root}/envs/core/bin/python, script: {root}/workers/echo_worker.py, device: cpu}}
    models:
      - {{id: test-swatch, backend: echo_cpu, tasks: [t2i], memory_gb: 0, weights: []}}
    """))
    s = Settings(tmp_path / "settings.json")
    reg = Registry(tmp_path / "models.yaml", s, probe_cache=tmp_path / "p.json")
    assets = AssetStore(tmp_path / "projects")
    mem = MemoryManager(s)
    pool = WorkerPool(reg, s, mem)

    class Bus:
        def publish(self, t, d):
            pass
    import app.plugins.rig as rigplugin
    m = JobManager(reg, assets, pool, mem, Bus(), s, tmp_path / "jobs.jsonl", tmp_path / "scratch")
    rigplugin.CORE = type("C", (), {"assets": assets})()
    yield m
    rigplugin.CORE = None
    m.shutdown()
    pool.stop_all()


def wait(m, jid, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        if m.get(jid)["status"] in TERMINAL:
            return m.get(jid)
        time.sleep(0.05)
    raise AssertionError(f"job {jid} still {m.get(jid)['status']}")


def ingest(mgr, tmp_path, data, name, **meta):
    f = tmp_path / f"{name}.glb"
    f.write_bytes(data)
    return mgr.assets.ingest("default", f, "mesh", {"name": name, "license": {"name": "MIT", "commercial": True}, "studio": "3d", "tags": ["rig"], **meta})


def run_edit(mgr, src, ops, **params):
    return wait(mgr, mgr.submit({"tool": "rig.edit", "params": {"ops": ops, **params}, "inputs": [{"asset": src["id"], "role": "mesh"}], "project": "default"})["id"])


def test_tool_makes_a_derived_asset(mgr, tmp_path):
    src = ingest(mgr, tmp_path, skinned_glb(HUM, HPAR, HNAMES, anim=True), "hero", model="unirig", task="rig")
    j = run_edit(mgr, src, [{"op": "add_chain", "parent": "hips", "count": 5, "direction": "back", "length": 0.1, "name_prefix": "tail"},
                            {"op": "rename", "bone": "left elbow", "to": "Elbow.L"}])
    assert j["status"] == "done", j.get("error")
    assert not j["gpu"] and len(j["outputs"]) == 1
    out = mgr.assets.get(j["outputs"][0])
    assert out["type"] == "mesh" and out["parents"] == [src["id"]] and out["name"] == "hero (rig edited)" and "rig-edited" in out["tags"]
    assert out["license"]["name"] == "MIT" and out["model"] == "unirig"
    info = out["info"]
    assert info["rigged"] is True and info["joints"] == len(HUM) + 5 and info["needs_reskin"] is True
    assert info["rig_edit"]["zero_weight_bones"][0] == "tail_01" and [o["op"] for o in info["rig_edit"]["ops"]] == ["add_chain", "rename"]
    assert info["animations"] == ["flex", "wiggle"]
    path = mgr.assets.asset_dir("default", out["id"]) / out["main"]
    names = joint_names(path.read_bytes())
    assert "Elbow.L" in names and names[-1] == "tail_05"
    assert {f["name"] for f in out["files"]} == {"edited.glb", "skeleton.json"} or out["main"].endswith(".glb")
    sk = json.loads((mgr.assets.asset_dir("default", out["id"]) / "skeleton.json").read_text())
    assert sk["joints"][-1]["name"] == "tail_05" and sk["joints"][-1]["parent"] == len(HUM) + 3
    # the original is untouched
    assert joint_names((mgr.assets.asset_dir("default", src["id"]) / src["main"]).read_bytes()) == HNAMES


def test_tool_errors_are_input_errors_with_help(mgr, tmp_path):
    src = ingest(mgr, tmp_path, SIMPLE, "hero")
    j = run_edit(mgr, src, [{"op": "rename", "bone": "elbow", "to": "x"}])
    assert j["status"] == "failed" and j["error"]["kind"] == "input" and "needs a side" in j["error"]["message"] and "L_elbow" in j["error"]["message"]
    assert not j["outputs"]
    j = run_edit(mgr, src, "")
    assert j["status"] == "failed" and j["error"]["kind"] == "input" and "ops" in j["error"]["message"] and "rename_standard" in (j["error"].get("fix") or "")
    j = run_edit(mgr, src, [{"op": "rename_standard", "scheme": "mixamo"}], clips="false")
    assert j["status"] == "done"
    # no skin / no input / two inputs / not a mesh
    js, b = ra.read_glb(SIMPLE)
    js.pop("skins")
    for n in js["nodes"]:
        n.pop("skin", None)
    static = ingest(mgr, tmp_path, ra.write_glb(js, b), "static")
    j = run_edit(mgr, static, [{"op": "rename", "bone": 1, "to": "x"}])
    assert j["status"] == "failed" and j["error"]["kind"] == "input" and "no skeleton" in j["error"]["message"] and "Rig" in (j["error"].get("fix") or "")
    j = wait(mgr, mgr.submit({"tool": "rig.edit", "params": {"ops": [{"op": "rename", "bone": 1, "to": "x"}]}, "inputs": [], "project": "default"})["id"])
    assert j["status"] == "failed" and j["error"]["kind"] == "input"
    j = wait(mgr, mgr.submit({"tool": "rig.edit", "params": {"ops": [{"op": "rename", "bone": 1, "to": "x"}]},
                              "inputs": [{"asset": src["id"]}, {"asset": static["id"]}], "project": "default"})["id"])
    assert j["status"] == "failed" and "exactly one" in j["error"]["message"]


def test_skeleton_endpoint(mgr, tmp_path):
    from fastapi import HTTPException

    import app.plugins.rig as rigplugin
    src = ingest(mgr, tmp_path, SIMPLE, "hero")
    d = rigplugin.skeleton(src["id"])
    assert d["asset"] == src["id"] and d["count"] == len(HUM) and d["humanoid"]["ok"] and d["needs_reskin"] is False
    assert d["joints"][HNAMES.index("L_elbow")]["role"] == "left_forearm"
    j = run_edit(mgr, src, [{"op": "add_bone", "parent": "head", "direction": "up", "name": "hat"}])
    out = mgr.assets.get(j["outputs"][0])
    d2 = rigplugin.skeleton(out["id"])
    assert d2["needs_reskin"] is True and d2["zero_weight"] == ["hat"]
    rep = rigplugin.dry_run(src["id"], {"ops": [{"op": "rename_standard", "scheme": "vrm"}]})
    assert rep["ops"][0]["op"] == "rename_standard" and "bytes_before" not in rep
    for call, code in [(lambda: rigplugin.skeleton("20260101-000000-zzzzzz"), 404), (lambda: rigplugin.dry_run(src["id"], {"ops": [{"op": "nope"}]}), 400)]:
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == code
    img = mgr.assets.ingest("default", _png(tmp_path), "image", {"name": "pic"})
    with pytest.raises(HTTPException) as e:
        rigplugin.skeleton(img["id"])
    assert e.value.status_code == 400


def _png(tmp_path):
    import struct
    import zlib
    raw = b"\x00\xff\x00\x00"
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    f = tmp_path / "pic.png"
    f.write_bytes(data)
    return f


def test_real_asset_if_present():
    """The measured UniRig soldier of this machine (skipped elsewhere): every op keeps the rest pose."""
    import glob
    hits = glob.glob(str(config.STUDIO / "data" / "projects" / "*" / "assets" / "20261107-123527-d7b8c6" / "mesh.glb"))
    if not hits:
        pytest.skip("asset not present")
    data = open(hits[0], "rb").read()
    out, rep = rE.edit(data, [{"op": "move_joint", "bone": "left elbow", "delta": [0, -0.2, 0]}, {"op": "delete_bone", "bone": "right toe"},
                              {"op": "add_chain", "parent": "hips", "count": 5, "direction": "back", "length": 0.08, "name_prefix": "tail"},
                              {"op": "rename_standard", "scheme": "mixamo"}])
    assert rep["joints_after"] == 22 - 1 + 5 and rep["animations_rebuilt"] == ["flex", "wiggle"]
    assert_rest_unchanged(data, out, tol=1e-4)
    check_structure(out)
