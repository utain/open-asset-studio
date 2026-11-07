"""Flex / wiggle animation clips for rigged GLBs (workers/rig_anim.py), the rig worker's hook and tool:mesh.flex_preview.
No GPU, no weights: a tiny skinned GLB is built here by hand (independent of the helper's own GLB writer)."""
import json
import math
import struct
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


# ------------------------------------------------------------------ a synthetic skinned GLB
def build_glb(parents, rest=None, matrix_joints=(), skin=True, pad_junk=False, translations=None):
    """parents[i] = parent joint index or None. Joint i is node i+1 (node 0 is the mesh). Returns (glb bytes, raw bin, json)."""
    J = len(parents)
    rest = rest or {}
    rng = np.random.default_rng(3)
    nodes = [{"name": "mesh", "mesh": 0, **({"skin": 0} if skin else {})}]
    pos = {}
    for i, p in enumerate(parents):
        if translations and i in translations:
            t = translations[i]
        else:
            t = [0.0, 0.3, 0.0] if p is not None else [0.0, 1.0, 0.0]
            if p is not None and i % 3 == 1:
                t = [0.25, 0.0, 0.0]
            elif p is not None and i % 3 == 2:
                t = [-0.25, 0.0, 0.0]
        node = {"name": f"bone_{i}", "translation": t}
        if i in rest:
            node["rotation"] = rest[i]
        if i in matrix_joints:        # column-major translation matrix
            node.pop("translation")
            node["matrix"] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, *t, 1]
        nodes.append(node)
    for i, p in enumerate(parents):
        if p is not None:
            nodes[p + 1].setdefault("children", []).append(i + 1)
    root = parents.index(None) + 1
    bin_ = bytearray()
    views, accs = [], []

    def add(arr, ctype, atype, target=None, **kw):
        while len(bin_) % 4:
            bin_.append(0)
        views.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": arr.nbytes, **({"target": target} if target else {})})
        bin_.extend(arr.tobytes())
        accs.append({"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": atype, **kw})
        return len(accs) - 1

    v = rng.random((6, 3)).astype("<f4")
    pa = add(v, 5126, "VEC3", 34962, min=[float(x) for x in v.min(0)], max=[float(x) for x in v.max(0)])
    ia = add(np.array([0, 1, 2, 3, 4, 5], dtype="<u2"), 5123, "SCALAR", 34963)
    attrs = {"POSITION": pa}
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, root]}], "nodes": nodes,
          "meshes": [{"primitives": [{"attributes": attrs, "indices": ia}]}]}
    if skin:
        jo = np.zeros((6, 4), dtype="<u2")
        jo[:, 0] = np.arange(6) % J
        wt = np.zeros((6, 4), dtype="<f4")
        wt[:, 0] = 1.0
        attrs["JOINTS_0"] = add(jo, 5123, "VEC4", 34962)
        attrs["WEIGHTS_0"] = add(wt, 5126, "VEC4", 34962)
        ibm = np.tile(np.eye(4, dtype="<f4"), (J, 1, 1))
        ib = add(ibm, 5126, "MAT4")
        js["skins"] = [{"inverseBindMatrices": ib, "joints": list(range(1, J + 1)), "skeleton": root}]
    if pad_junk:
        bin_.extend(b"\x01\x02\x03\x04")            # trailing bytes in the chunk that are not in any bufferView
    js["bufferViews"], js["accessors"] = views, accs
    js["buffers"] = [{"byteLength": len(bin_)}]
    return ra.write_glb(js, bytes(bin_)), bytes(bin_), js


HUMAN = [None, 0, 1, 2, 1, 4, 5, 1, 7, 8, 0, 10, 11, 0, 13, 14]            # 16 joints, branching
FIVE = [None, 0, 1, 0, 3]                                                  # root, spine, head, arm, hand


def parse(glb):
    """Independent re-parse: -> (json, bin, chunk lengths)."""
    magic, ver, total = struct.unpack_from("<III", glb, 0)
    assert magic == 0x46546C67 and ver == 2 and total == len(glb) and total % 4 == 0
    pos, chunks = 12, []
    while pos < total:
        n, t = struct.unpack_from("<II", glb, pos)
        assert n % 4 == 0
        chunks.append((t, glb[pos + 8:pos + 8 + n]))
        pos += 8 + n
    assert pos == total
    assert chunks[0][0] == 0x4E4F534A and chunks[1][0] == 0x004E4942 and len(chunks) == 2
    return json.loads(chunks[0][1]), chunks[1][1]


def accessor_data(js, bin_, idx):
    a = js["accessors"][idx]
    v = js["bufferViews"][a["bufferView"]]
    assert a["componentType"] == 5126
    n = {"SCALAR": 1, "VEC4": 4}[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    assert off % 4 == 0
    return np.frombuffer(bin_, dtype="<f4", count=a["count"] * n, offset=off).reshape(a["count"], n).astype(np.float64)


def qangle(a, b):
    return 2 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


# ------------------------------------------------------------------ structure of the output
@pytest.mark.parametrize("parents,rest", [(FIVE, {}), (HUMAN, {3: [0.0, 0.3826834, 0.0, 0.9238795], 6: [0.5, 0.5, 0.5, 0.5]})])
def test_output_is_valid_gltf_with_both_clips(parents, rest):
    src, src_bin, src_js = build_glb(parents, rest)
    out, info = ra.add_animations(src)
    js, bin_ = parse(out)
    assert [a["name"] for a in js["animations"]] == ["flex", "wiggle"]
    assert info["joints"] == len(parents) and [c["name"] for c in info["clips"]] == ["flex", "wiggle"]
    joints = set(js["skins"][0]["joints"])
    assert js["buffers"][0]["byteLength"] == len(bin_)
    # every accessor / bufferView lies inside the buffer, 4-byte aligned
    for v in js["bufferViews"]:
        assert v["byteOffset"] % 4 == 0 and v["byteOffset"] + v["byteLength"] <= len(bin_)
    for a in js["accessors"]:
        v = js["bufferViews"][a["bufferView"]]
        size = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]] * {5126: 4, 5123: 2}[a["componentType"]]
        assert a.get("byteOffset", 0) + a["count"] * size <= v["byteLength"]
    for anim in js["animations"]:
        assert anim["channels"] and len(anim["channels"]) == len(joints)
        assert {c["target"]["node"] for c in anim["channels"]} == joints               # every joint, nothing else
        for c in anim["channels"]:
            assert c["target"]["path"] == "rotation"                                    # root translation untouched
            s = anim["samplers"][c["sampler"]]
            assert s["interpolation"] == "LINEAR"
            t = accessor_data(js, bin_, s["input"])[:, 0]
            q = accessor_data(js, bin_, s["output"])
            assert np.all(np.diff(t) > 0) and t[0] == 0.0
            ta = js["accessors"][s["input"]]
            assert ta["min"] == [float(t.min())] and ta["max"] == [float(t.max())]     # required for animation inputs
            assert q.shape == (len(t), 4)
            assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
            rest_q = np.array(js["nodes"][c["target"]["node"]].get("rotation", [0, 0, 0, 1]))
            rest_q = rest_q / np.linalg.norm(rest_q)
            if anim["name"] == "flex":                                                  # starts and ends at the rest pose
                assert qangle(q[0], rest_q) < 1e-5 and qangle(q[-1], rest_q) < 1e-5
            else:                                                                       # wiggle: a seamless loop around it
                assert np.allclose(q[0], q[-1]) and qangle(q[0], rest_q) < math.radians(6)
    flex = js["animations"][0]
    ends = [accessor_data(js, bin_, s["input"])[-1, 0] for s in flex["samplers"]]
    assert len(set(ends)) == 1                                                          # all tracks span the same duration
    assert math.isclose(ends[0], info["clips"][0]["duration"], rel_tol=1e-4)


def test_original_content_is_byte_identical():
    src, src_bin, src_js = build_glb(HUMAN, pad_junk=True)
    out, _ = ra.add_animations(src)
    js, bin_ = parse(out)
    assert bin_[:len(src_bin)] == src_bin                                                # appended only, nothing moved
    n_views, n_accs = len(src_js["bufferViews"]), len(src_js["accessors"])
    assert js["bufferViews"][:n_views] == src_js["bufferViews"]
    assert js["accessors"][:n_accs] == src_js["accessors"]
    for k in ("meshes", "skins", "scenes", "asset"):
        assert js[k] == src_js[k]
    assert js["nodes"] == src_js["nodes"]                                                # no matrices here: nodes untouched
    assert js["animations"][0]["channels"][0]["target"]["node"] >= 1
    # mesh data read back through the new file still equals the originals
    for idx in (src_js["meshes"][0]["primitives"][0]["attributes"]["POSITION"],):
        v = js["bufferViews"][js["accessors"][idx]["bufferView"]]
        o = v["byteOffset"]
        assert bin_[o:o + v["byteLength"]] == src_bin[o:o + v["byteLength"]]


def test_glb_without_bin_chunk_and_alignment():
    js = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"name": "a", "children": [1]}, {"name": "b", "translation": [0, 1, 0]}],
          "skins": [{"joints": [0, 1]}]}
    out, _ = ra.add_animations(ra.write_glb(js, b""))
    j2, b2 = parse(out)
    assert len(b2) % 4 == 0 and j2["buffers"][0]["byteLength"] == len(b2) and len(j2["animations"]) == 2


def test_matrix_nodes_become_trs():
    src, _, _ = build_glb(FIVE, matrix_joints=(3,))
    out, _ = ra.add_animations(src)
    js, _ = parse(out)
    n = js["nodes"][4]
    assert "matrix" not in n and np.allclose(n["translation"], [0.0, 0.3, 0.0]) and n["rotation"] == [0, 0, 0, 1]


def test_clip_selection_and_existing_names():
    src, _, _ = build_glb(FIVE)
    out, info = ra.add_animations(src, ["wiggle"])
    assert [a["name"] for a in parse(out)[0]["animations"]] == ["wiggle"]
    out2, _ = ra.add_animations(out, ["flex", "wiggle"])                                 # wiggle exists: only flex is added
    assert [a["name"] for a in parse(out2)[0]["animations"]] == ["wiggle", "flex"]
    with pytest.raises(ra.RigAnimError, match="already"):
        ra.add_animations(out2)
    with pytest.raises(ra.RigAnimError, match="Unknown"):
        ra.add_animations(src, ["dance"])
    assert ra.animation_names(out2) == ["wiggle", "flex"]


# ------------------------------------------------------------------ what the clips do
def humanoid():
    """~50 joints: spine chain, head, two 4-deep arms with 5 fingers of 3 joints, two 4-deep legs."""
    p = [None]
    sp = 0
    for _ in range(3):
        p.append(sp)
        sp = len(p) - 1
    p.append(sp)                                    # neck
    p.append(len(p) - 1)                            # head
    for _ in range(2):
        a = sp
        for _ in range(4):
            p.append(a)
            a = len(p) - 1
        for _ in range(5):
            f = a
            for _ in range(3):
                p.append(f)
                f = len(p) - 1
    for _ in range(2):
        a = 0
        for _ in range(4):
            p.append(a)
            a = len(p) - 1
    return p


def test_timing_stays_between_8_and_20_seconds_even_for_many_joints():
    h = humanoid()
    assert len(h) >= 50
    rig = ra.Rig(build_glb(h)[2])
    plan = ra.flex_plan(rig)
    assert 8 <= plan["duration"] <= 20 and plan["levels_per_slot"] == 1
    for chain in (30, 60, 120):                      # one long tail / chain: several levels share a slot
        rig = ra.Rig(build_glb([None] + list(range(chain - 1)))[2])
        d = ra.flex_plan(rig)["duration"]
        assert d <= 20, (chain, d)
        assert d >= 3
    rig = ra.Rig(build_glb(FIVE)[2])
    assert ra.flex_plan(rig)["duration"] >= 3        # tiny rigs are not instantaneous


def test_flex_swings_each_joint_about_a_perpendicular_axis_within_the_angle():
    src, _, js = build_glb(HUMAN)
    rig = ra.Rig(js)
    clip = ra.build_flex(rig, angle_deg=30)
    lvl_slot = clip["plan"]["level_slot"]
    for j, t, q in clip["tracks"]:
        rest = np.array([0.0, 0.0, 0.0, 1.0])
        ang = [qangle(x, rest) for x in q.astype(np.float64)]
        amp = math.radians(15 if rig.parent[j] is None else 30)
        assert max(ang) == pytest.approx(amp, abs=math.radians(1.5))                    # +-30 deg (root half)
        assert ang[0] < 1e-6 and ang[-1] < 1e-6                                         # rests at both ends
        # axis of the largest delta is the flex axis; it is perpendicular to the bone direction (checked in the next test)
        k = int(np.argmax(ang))
        axis = q[k, :3].astype(np.float64)
        axis /= np.linalg.norm(axis)
        assert np.allclose(np.abs(axis), np.abs(rig.flex[j]), atol=1e-4)
        assert abs(float(np.dot(rig.flex[j], rig.bend[j]))) > 0.7                         # stays close to the bend axis
        assert abs(float(np.dot(rig.bend[j], rig.side[j]))) < 1e-9
        # every joint is moving only during its own level's slot
        k0 = lvl_slot[rig.depth[j]]
        moving = t[np.abs(np.array(ang)) > 1e-4]
        lo = clip["plan"]["lead"] + k0 * clip["plan"]["slot"]
        assert moving.min() >= lo - 1e-5 and moving.max() <= lo + clip["plan"]["slot"] + 1e-5


def test_bend_axis_is_perpendicular_to_the_bone_in_world_space():
    # a bone pointing along +X from a parent at the origin: bend axis must be perpendicular to X
    js = build_glb([None, 0], translations={0: [0, 0, 0], 1: [1.0, 0, 0]})[2]
    rig = ra.Rig(js)
    for j in rig.joints:
        assert abs(float(rig.bend[j][0])) < 1e-9 and abs(float(rig.flex[j][0])) < 1e-9 and abs(float(rig.side[j][0])) < 1e-9
    # a joint with a rest rotation: the axis is expressed in the joint's own frame (rest rotation of 90 deg about Z swaps x / y)
    s = math.sqrt(0.5)
    js = build_glb([None, 0], rest={0: [0, 0, s, s]}, translations={0: [0, 0, 0], 1: [0, 1.0, 0]})[2]
    rig = ra.Rig(js)
    # world bone direction is R(+90 z) * (0,1,0) = (-1,0,0); world bend = d x z_ref... must be perpendicular to it
    rw = ra.q_to_mat([0, 0, s, s])
    assert abs(float(np.dot(rw @ rig.bend[1], [-1.0, 0, 0]))) < 1e-9 and abs(float(np.dot(rw @ rig.flex[1], [-1.0, 0, 0]))) < 1e-9


def test_wiggle_is_small_and_loops():
    js = build_glb(HUMAN)[2]
    rig = ra.Rig(js)
    clip = ra.build_wiggle(rig, angle_deg=3)
    assert clip["duration"] == 4.0
    lags = set()
    for j, t, q in clip["tracks"]:
        assert np.allclose(q[0], q[-1])                                                 # seamless loop
        assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
        ang = max(qangle(x, np.array([0, 0, 0, 1.0])) for x in q.astype(np.float64))
        assert 0.005 < ang < math.radians(6)
        lags.add(round(float(np.argmax(q[:, 0] + q[:, 1] + q[:, 2])), 1))
    assert len(lags) > 3                                                                # phase-shifted, not in lockstep


def test_slerp_hemisphere_is_continuous():
    q = np.array([[0, 0, 0, 1.0], [0, 0, 0, -1.0], [0.1, 0, 0, 0.99]])
    c = ra._canon(q)
    assert all(np.dot(c[i], c[i + 1]) >= 0 for i in range(2))


def test_quaternion_helpers_match_matrices():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert qangle(ra.mat_to_q(ra.q_to_mat(q)), q) < 1e-9
        r = rng.normal(size=4)
        r /= np.linalg.norm(r)
        assert np.allclose(ra.q_to_mat(ra.qmul(q, r)), ra.q_to_mat(q) @ ra.q_to_mat(r), atol=1e-9)


# ------------------------------------------------------------------ refusals
def test_no_skin_and_bad_files_are_refused():
    glb, _, _ = build_glb(FIVE, skin=False)
    js, bin_ = parse(glb)
    js["skins"] = []
    plain = ra.write_glb(js, bin_)
    assert not ra.has_skin(plain)
    with pytest.raises(ra.NoSkinError):
        ra.add_animations(plain)
    with pytest.raises(ra.RigAnimError, match="Not a binary glTF"):
        ra.add_animations(b"glTF" + b"\x00" * 64)
    with pytest.raises(ra.RigAnimError):
        ra.add_animations(b"abc")
    assert ra.has_skin(build_glb(FIVE)[0])


def test_external_buffer_is_refused():
    _, b, js = build_glb(FIVE)
    js["buffers"][0]["uri"] = "x.bin"
    with pytest.raises(ra.RigAnimError, match="external"):
        ra.add_animations(ra.write_glb(js, b))


# ------------------------------------------------------------------ the rig worker's hook never fails the job
class Ctx:
    def __init__(self):
        self.lines = []

    def log(self, m):
        self.lines.append(m)


def test_rig_worker_hook_swaps_files_and_survives_failures(tmp_path):
    import rig_worker
    glb = tmp_path / "rigged.glb"
    glb.write_bytes(build_glb(HUMAN)[0])
    plain_bytes = glb.read_bytes()
    attach, ctx = ["skeleton.json"], Ctx()
    out, clips = rig_worker._with_clips(glb, ctx, attach)
    assert out == glb and clips == ["flex", "wiggle"]
    assert attach == ["skeleton.json", "rigged_plain.glb"]
    assert (tmp_path / "rigged_plain.glb").read_bytes() == plain_bytes
    assert not (tmp_path / "rigged_anim.glb").exists()
    assert [a["name"] for a in parse(glb.read_bytes())[0]["animations"]] == ["flex", "wiggle"]
    # a broken animation step is logged and the plain file stays the output
    bad = tmp_path / "bad" / "rigged.glb"
    bad.parent.mkdir()
    bad.write_bytes(b"not a glb at all, but long enough to try")
    attach, ctx = ["skeleton.json"], Ctx()
    out, clips = rig_worker._with_clips(bad, ctx, attach)
    assert out == bad and clips == [] and attach == ["skeleton.json"]
    assert bad.read_bytes().startswith(b"not a glb") and any("skipped" in m for m in ctx.lines)
    assert not list(bad.parent.glob("rigged_*"))
    # no skin at all (a rig that produced nothing usable)
    glb2 = tmp_path / "noskin" / "rigged.glb"
    glb2.parent.mkdir()
    glb2.write_bytes(build_glb(FIVE, skin=False)[0])
    out, clips = rig_worker._with_clips(glb2, Ctx(), [])
    assert clips == [] and out == glb2


# ------------------------------------------------------------------ tool:mesh.flex_preview through the job queue
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
    import app.plugins.mesh  # noqa: F401  (registers tool:mesh.flex_preview)
    m = JobManager(reg, assets, pool, mem, Bus(), s, tmp_path / "jobs.jsonl", tmp_path / "scratch")
    yield m
    m.shutdown()
    pool.stop_all()


def wait(m, jid, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        if m.get(jid)["status"] in TERMINAL:
            return m.get(jid)
        time.sleep(0.05)
    raise AssertionError(f"job {jid} still {m.get(jid)['status']}")


def ingest(mgr, tmp_path, data, name):
    f = tmp_path / f"{name}.glb"
    f.write_bytes(data)
    return mgr.assets.ingest("default", f, "mesh", {"name": name, "license": {"name": "MIT", "commercial": True}, "studio": "3d"})


def test_tool_makes_a_derived_asset_with_clips(mgr, tmp_path):
    src = ingest(mgr, tmp_path, build_glb(HUMAN)[0], "rigged guy")
    j = wait(mgr, mgr.submit({"tool": "mesh.flex_preview", "inputs": [{"asset": src["id"], "role": "mesh"}], "project": "default"})["id"])
    assert j["status"] == "done", j.get("error")
    assert not j["gpu"] and len(j["outputs"]) == 1
    out = mgr.assets.get(j["outputs"][0])
    assert out["type"] == "mesh" and out["parents"] == [src["id"]] and out["name"] == "rigged guy (animated)"
    assert out["license"]["name"] == "MIT" and out["info"]["animations"] == ["flex", "wiggle"] and out["info"]["joints"] == len(HUMAN)
    path = mgr.assets.asset_dir("default", out["id"]) / out["main"]
    assert ra.animation_names(path) == ["flex", "wiggle"]
    # the original is untouched
    assert ra.animation_names(mgr.assets.asset_dir("default", src["id"]) / src["main"]) == []
    # again on the result: nothing new to add -> a clean input error
    j2 = wait(mgr, mgr.submit({"tool": "mesh.flex_preview", "inputs": [{"asset": out["id"], "role": "mesh"}], "project": "default"})["id"])
    assert j2["status"] == "failed" and j2["error"]["kind"] == "input" and "already" in j2["error"]["message"]


def test_tool_refuses_a_mesh_without_skin(mgr, tmp_path):
    glb, _, _ = build_glb(FIVE, skin=False)
    js, b = parse(glb)
    js.pop("skins", None)
    src = ingest(mgr, tmp_path, ra.write_glb(js, b), "static")
    j = wait(mgr, mgr.submit({"tool": "mesh.flex_preview", "inputs": [{"asset": src["id"], "role": "mesh"}], "project": "default"})["id"])
    assert j["status"] == "failed" and j["error"]["kind"] == "input"
    assert "no skeleton" in j["error"]["message"] and "Rig" in (j["error"].get("fix") or "")
    assert not j["outputs"]


def test_tool_rejects_missing_input_and_bad_angle(mgr, tmp_path):
    j = wait(mgr, mgr.submit({"tool": "mesh.flex_preview", "inputs": [], "project": "default"})["id"])
    assert j["status"] == "failed" and j["error"]["kind"] == "input"
    src = ingest(mgr, tmp_path, build_glb(FIVE)[0], "five")
    j = wait(mgr, mgr.submit({"tool": "mesh.flex_preview", "params": {"angle": 500}, "inputs": [{"asset": src["id"], "role": "mesh"}], "project": "default"})["id"])
    assert j["status"] == "failed" and j["error"]["kind"] == "input"
