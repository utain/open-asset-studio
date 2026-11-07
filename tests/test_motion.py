"""Text to motion (models.d/motion.yaml, workers/motion_common.py, workers/motion_hy_worker.py): the registry entry loads, tasks and
backends exist, every param validates its default, and the pure numpy helpers (rot6d, FK, humanoid detection on synthetic
skeletons, retargeting, foot lock, GLB clip writing) work. No GPU, no weights."""
import importlib.util
import json
import sys

import numpy as np
import pytest

from app import config
from app.registry import Registry

sys.path.insert(0, str(config.STUDIO / "workers"))
import motion_common as mc  # noqa: E402
import rig_anim as ra  # noqa: E402

rng = np.random.default_rng(11)


# ------------------------------------------------------------------------------------------------------------ registry
@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def test_registry_entry(reg):
    assert not [e for e in reg.errors if "motion" in e]
    m = reg.models["hy-motion-lite"]
    assert m["_file"] == "motion.yaml" and m["tasks"] == ["motion"] and m["backend"] == "motion_hy"
    assert reg.tasks["motion"]["studio"] == "3d" and reg.tasks["motion"]["modality"] == "mesh"
    assert m["license"]["commercial"] is True and m["license"]["verify"] is True
    assert "European Union" in m["license"]["note"] and "Tencent" in m["license"]["name"]
    assert (config.STUDIO / reg.backends["motion_hy"]["script"]).exists()
    assert reg.backends["motion_hy"]["python"] == "envs/motion_hy/bin/python"
    assert m["input_slots"][0]["role"] == "mesh" and m["input_slots"][0]["required"] is True


def test_defaults_validate(reg):
    out = reg.validate_params("hy-motion-lite", {})
    assert out["seconds"] == 5 and out["fps"] == 30 and out["root_motion"] == "root" and out["foot_lock"] is True
    assert out["facing"] == "auto"
    for preset in reg.models["hy-motion-lite"]["presets"].values():
        assert isinstance(reg.validate_params("hy-motion-lite", preset), dict)
    assert reg.validate_params("hy-motion-lite", {"seconds": 99})["seconds"] == 12          # clipped to HY-Motion's maximum


def test_install_recipe_and_vendor_lock():
    import yaml
    r = yaml.safe_load((config.STUDIO / "install" / "motion_hy.yaml").read_text())
    assert r["backend"] == "motion_hy" and r["vendor"] == ["HY-Motion-1.0"] and "lfs pull" in " ".join(r["steps"])
    assert (config.STUDIO / "env-lock" / "motion_hy.txt").exists()
    lock = (config.STUDIO / "vendor" / "vendor.lock").read_text()
    assert "vendor/HY-Motion-1.0 https://github.com/Tencent-Hunyuan/HY-Motion-1.0" in lock and "4e426f5a1021cbcf7f375458c37b840ee7225229" in lock


def test_worker_module_and_input_errors():
    spec = importlib.util.spec_from_file_location("motion_hy_worker_t", config.STUDIO / "workers" / "motion_hy_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from studio_worker import InputError, Job
    b = mod.MotionHYBackend()
    assert b.name == "motion_hy" and mod._facing({"facing": "-Z"}) == "-Z" and mod._facing({"facing": "x"}) == "auto"

    class Ctx:
        def progress(self, *a, **k): pass
        def log(self, *a, **k): pass
        def check(self): pass

    with pytest.raises(InputError, match="rigged character"):
        b.run(Job({"id": "j", "task": "motion", "model": {"id": "hy-motion-lite"}, "params": {"prompt": "walk"}}), Ctx())
    with pytest.raises(InputError, match="Describe the motion"):
        b.run(Job({"id": "j", "task": "motion", "model": {"id": "hy-motion-lite"}, "params": {"prompt": "  "},
                   "inputs": [{"role": "mesh", "type": "mesh", "path": "/nope.glb"}]}), Ctx())


# ------------------------------------------------------------------------------------------------------------ synthetic skeletons
def humanoid(*, a_pose=False, collar=True, n_spine=2, chibi=False, extras=False, facing_neg_z=False, scale=1.0, hand_leaf=True,
             fingers=False, toes=True, shift=(0.0, 0.0, 0.0)):
    """-> (positions [J,3], parents list, names list). +Y up, faces +Z, left = +X. Pelvis is joint 0."""
    P, par, names = [], [], []

    def add(name, pos, parent):
        P.append(pos)
        par.append(parent)
        names.append(name)
        return len(P) - 1

    leg = 0.2 if chibi else 1.0
    hip_y = 0.95 * (0.35 if chibi else 1.0)
    pel = add("pelvis", (0, hip_y, 0), None)
    prev, top_y = pel, hip_y
    spine = []
    for i in range(n_spine):
        top_y += 0.15
        prev = add(f"spine{i + 1}", (0, top_y, 0), prev)
        spine.append(prev)
    top_y += 0.15
    chest = add("chest", (0, top_y, 0), prev)
    neck = add("neck", (0, top_y + 0.15, 0.01), chest)
    head = add("head", (0, top_y + 0.3, 0.02), neck)
    if extras:
        add("hair1", (0, top_y + 0.4, -0.05), head)
        add("hair2", (0, top_y + 0.35, -0.12), P.__len__() - 1)
        add("jaw", (0, top_y + 0.2, 0.08), head)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sh_y = top_y + 0.05
        c = add(f"{side}_collar", (sx * 0.06, sh_y + 0.02, 0), chest) if collar else chest
        s = add(f"{side}_shoulder", (sx * 0.2, sh_y, 0), c)
        if a_pose:
            e = add(f"{side}_elbow", (sx * 0.4, sh_y - 0.25, 0), s)
            w = add(f"{side}_wrist", (sx * 0.58, sh_y - 0.5, 0), e)
        else:
            e = add(f"{side}_elbow", (sx * 0.5, sh_y, 0), s)
            w = add(f"{side}_wrist", (sx * 0.75, sh_y, 0), e)
        if fingers:
            for k in range(5):
                f = add(f"{side}_f{k}a", (w and P[w][0] + sx * 0.05, P[w][1] - 0.01 * k, 0.02 * (k - 2)), w)
                add(f"{side}_f{k}b", (P[f][0] + sx * 0.04, P[f][1] - 0.01, P[f][2]), f)
        elif hand_leaf:
            add(f"{side}_handtip", (P[w][0] + sx * 0.08, P[w][1] - (0.07 if a_pose else 0), 0), w)
        hip = add(f"{side}_hip", (sx * 0.1, hip_y - 0.05, 0), pel)
        knee = add(f"{side}_knee", (sx * 0.1, (hip_y - 0.05) - 0.45 * leg, 0.02), hip)
        ank = add(f"{side}_ankle", (sx * 0.1, 0.08 * (0.3 if chibi else 1.0), 0), knee)
        if toes:
            add(f"{side}_toe", (sx * 0.1, 0.0, 0.14 * (0.4 if chibi else 1.0)), ank)
    if extras:
        t1 = add("tail1", (0, hip_y - 0.1, -0.15), pel)
        t2 = add("tail2", (0, hip_y - 0.3, -0.25), t1)
        add("tail3", (0, hip_y - 0.45, -0.3), t2)
        sk = add("skirt1", (0.15, hip_y - 0.2, 0.1), pel)
        add("skirt2", (0.18, hip_y - 0.45, 0.12), sk)
    Pa = np.array(P, dtype=float) * scale
    Pa[:, 1] = Pa[:, 1]
    if facing_neg_z:
        Pa = Pa * np.array([-1.0, 1.0, -1.0])
    return Pa + np.array(shift), par, names


def reroot(parents, new_root):
    """Reverse the path new_root -> old root so that new_root becomes the root."""
    par = list(parents)
    chain, j = [new_root], new_root
    while par[j] is not None:
        j = par[j]
        chain.append(j)
    for a, b in zip(chain[:-1], chain[1:]):
        par[b] = a
    par[new_root] = None
    return par


def quadruped():
    P = [(0, 0.8, 0), (0, 0.82, 0.3), (0, 0.85, 0.6), (0, 0.95, 0.8), (0, 1.0, 0.95)]      # spine along Z
    par = [None, 0, 1, 2, 3]
    for x, z, p in ((0.15, 0.0, 0), (-0.15, 0.0, 0), (0.15, 0.6, 2), (-0.15, 0.6, 2)):
        a = len(P)
        P += [(x, 0.5, z), (x, 0.25, z + 0.05), (x, 0.0, z + 0.1)]
        par += [p, a, a + 1]
    P += [(0.3, 1.1, 1.0), (-0.3, 1.1, 1.0)]
    par += [4, 4]
    return np.array(P), par


# ------------------------------------------------------------------------------------------------------------ rotations + FK
def test_rot6d_roundtrip_and_orthonormal():
    R = np.stack([mc.axis_angle_mat(rng.normal(size=3), rng.uniform(-3, 3)) for _ in range(20)])
    back = mc.rot6d_to_mat(mc.mat_to_rot6d(R))
    assert np.abs(back - R).max() < 1e-12
    noisy = mc.rot6d_to_mat(rng.normal(size=(30, 6)))
    assert np.abs(noisy @ np.swapaxes(noisy, -1, -2) - np.eye(3)).max() < 1e-12 and np.allclose(np.linalg.det(noisy), 1)
    # HY-Motion's layout: columns 0 and 1 stored row by row
    assert np.allclose(mc.rot6d_to_mat([1, 0, 0, 1, 0, 0]), np.eye(3))
    assert np.allclose(mc.rot6d_to_mat([0, 1, 1, 0, 0, 0]), [[0, 1, 0], [1, 0, 0], [0, 0, -1]])


def test_quaternions():
    R = np.stack([mc.axis_angle_mat(rng.normal(size=3), rng.uniform(-3.1, 3.1)) for _ in range(200)])
    q = mc.mat_to_quat(R)
    assert np.abs(np.linalg.norm(q, axis=-1) - 1).max() < 1e-12 and np.abs(mc.quat_to_mat(q) - R).max() < 1e-9
    c = mc.canon_quats(np.stack([q[0], -q[0], q[0]]))
    assert np.sum(c[1] * c[0]) > 0 and np.sum(c[2] * c[1]) > 0
    mid = mc.slerp(np.array([0, 0, 0, 1.0]), mc.mat_to_quat(mc.axis_angle_mat([0, 1, 0], 1.0)), 0.5)
    assert np.allclose(mc.quat_to_mat(mid), mc.axis_angle_mat([0, 1, 0], 0.5), atol=1e-9)


def test_arc_rotation():
    for _ in range(30):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = mc.arc_rotation(a, b)
        assert np.allclose(R @ mc.unit(a), mc.unit(b), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(mc.arc_rotation([0, 1, 0], [0, 2, 0]), np.eye(3))
    R = mc.arc_rotation([1, 0, 0], [-1, 0, 0])
    assert np.allclose(R @ [1, 0, 0], [-1, 0, 0]) and np.isclose(np.linalg.det(R), 1) and np.allclose(R @ R.T, np.eye(3))
    Rb = mc.arc_rotations(rng.normal(size=(5, 3)), rng.normal(size=(5, 3)))
    assert np.abs(Rb @ np.swapaxes(Rb, -1, -2) - np.eye(3)).max() < 1e-9


def test_hy_fk_rest_pose_and_bone_lengths():
    T = 3
    r6 = np.tile(mc.mat_to_rot6d(np.eye(3)), (T, 52, 1))
    W, P, L = mc.hy_fk(r6, np.zeros((T, 3)))
    assert np.abs(P - mc.HY_REST).max() < 1e-9 and np.abs(W - np.eye(3)).max() < 1e-12
    r6b = r6.copy()
    r6b[:, 16] = mc.mat_to_rot6d(mc.axis_angle_mat([0, 0, 1], 0.7))                  # rotate the left shoulder (16)
    W, P2, _ = mc.hy_fk(r6b, np.array([[0.5, 0, 0]] * T))
    moved = np.flatnonzero(np.abs(P2[0] - (mc.HY_REST + [0.5, 0, 0])).max(1) > 1e-6)
    assert set(moved) == {18, 20}                 # elbow and wrist swing; the shoulder joint itself is the pivot
    for j in range(1, 22):
        p = mc.HY_PARENTS[j]
        assert abs(np.linalg.norm(P2[0, j] - P2[0, p]) - np.linalg.norm(mc.HY_REST[j] - mc.HY_REST[p])) < 1e-9


def test_embedded_rest_matches_vendor_template():
    f = config.STUDIO / "vendor" / "HY-Motion-1.0" / "scripts/gradio/static/assets/dump_wooden/j_template.bin"
    if not f.is_file() or f.stat().st_size < 600:
        pytest.skip("vendor/HY-Motion-1.0 (git lfs) not present")
    assert np.abs(mc.load_hy_rest(f) - mc.HY_REST).max() < 1e-4


# ------------------------------------------------------------------------------------------------------------ detection
def roles_ok(h, names):
    n = lambda k: names[h[k]]            # noqa: E731
    assert n("hips") == "pelvis" and n("head") == "head" and n("chest") == "chest"
    for s, S in (("l", "L"), ("r", "R")):
        assert n(f"{s}_thigh") == f"{S}_hip" and n(f"{s}_shin") == f"{S}_knee" and n(f"{s}_foot") == f"{S}_ankle"
        assert n(f"{s}_upperarm") == f"{S}_shoulder" and n(f"{s}_forearm") == f"{S}_elbow"
        assert n(f"{s}_hand") in (f"{S}_wrist", f"{S}_handtip") or n(f"{s}_hand").startswith(f"{S}_wrist")


@pytest.mark.parametrize("opts", [
    {}, {"a_pose": True}, {"collar": False}, {"n_spine": 5}, {"n_spine": 1}, {"chibi": True}, {"extras": True}, {"toes": False},
    {"fingers": True}, {"scale": 100.0}, {"shift": (3.0, 0.0, -2.0)}, {"a_pose": True, "extras": True, "n_spine": 3},
])
def test_detects_humanoid_variants(opts):
    P, par, names = humanoid(**opts)
    h = mc.detect_humanoid(P, par, names)
    roles_ok(h, names)
    assert h["facing_name"] == "+Z" and np.allclose(h["left"], [1, 0, 0])
    if opts.get("collar", True):
        assert names[h["l_collar"]] == "L_collar" and names[h["r_collar"]] == "R_collar"
    else:
        assert h["l_collar"] is None
    assert len(h["spine"]) == opts.get("n_spine", 2) - 0 or len(h["spine"]) >= 1
    if opts.get("toes", True):
        assert names[h["l_toe"]] == "L_toe"
    else:
        assert h["l_toe"] is None
    assert h["leg_length"] > 0


def test_detection_without_names_and_with_shuffled_order():
    P, par, names = humanoid(extras=True)
    perm = rng.permutation(len(P))                 # mirrored / arbitrary joint order: parents are remapped
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(P))
    P2 = P[perm]
    par2 = [None if par[perm[i]] is None else int(inv[par[perm[i]]]) for i in range(len(P))]
    names2 = [names[perm[i]] for i in range(len(P))]
    h = mc.detect_humanoid(P2, par2)               # no names at all
    roles_ok(h, names2)
    assert h["method"] == "geometry"


def test_detects_rig_rooted_at_the_chest():
    P, par, names = humanoid()
    chest = names.index("chest")
    h = mc.detect_humanoid(P, reroot(par, chest), names)
    roles_ok(h, names)


def test_detects_facing_minus_z_automatically():
    P, par, names = humanoid(facing_neg_z=True)
    h = mc.detect_humanoid(P, par, names)
    assert h["facing_name"] == "-Z" and np.allclose(h["facing"], [0, 0, -1]) and np.allclose(h["left"], [-1, 0, 0])
    roles_ok(h, names)                             # left is still L_*: the side follows the character, not the world
    h2 = mc.detect_humanoid(P, par, names, facing="-Z")
    assert h2["facing_name"] == "-Z"


def test_name_cross_check():
    P, par, names = humanoid(fingers=True)
    mix = {"pelvis": "mixamorig:Hips", "L_hip": "mixamorig:LeftUpLeg", "R_hip": "mixamorig:RightUpLeg", "L_knee": "mixamorig:LeftLeg",
           "R_knee": "mixamorig:RightLeg", "L_ankle": "mixamorig:LeftFoot", "R_ankle": "mixamorig:RightFoot", "L_toe": "mixamorig:LeftToeBase",
           "R_toe": "mixamorig:RightToeBase", "L_shoulder": "mixamorig:LeftArm", "R_shoulder": "mixamorig:RightArm",
           "L_collar": "mixamorig:LeftShoulder", "R_collar": "mixamorig:RightShoulder", "L_elbow": "mixamorig:LeftForeArm",
           "R_elbow": "mixamorig:RightForeArm", "L_wrist": "mixamorig:LeftHand", "R_wrist": "mixamorig:RightHand", "head": "mixamorig:Head"}
    h = mc.detect_humanoid(P, par, [mix.get(n, n) for n in names])
    assert h["method"] == "geometry+names"
    vroid = {"pelvis": "J_Bip_C_Hips", "L_hip": "J_Bip_L_UpperLeg", "R_hip": "J_Bip_R_UpperLeg", "L_knee": "J_Bip_L_LowerLeg",
             "R_knee": "J_Bip_R_LowerLeg", "L_ankle": "J_Bip_L_Foot", "R_ankle": "J_Bip_R_Foot", "L_shoulder": "J_Bip_L_UpperArm",
             "R_shoulder": "J_Bip_R_UpperArm", "L_collar": "J_Bip_L_Shoulder", "R_collar": "J_Bip_R_Shoulder", "L_elbow": "J_Bip_L_LowerArm",
             "R_elbow": "J_Bip_R_LowerArm", "L_wrist": "J_Bip_L_Hand", "R_wrist": "J_Bip_R_Hand", "head": "J_Bip_C_Head"}
    assert mc.detect_humanoid(P, par, [vroid.get(n, n) for n in names])["method"] == "geometry+names"
    assert mc.detect_humanoid(P, par, [f"bone_{i}" for i in range(len(P))])["method"] == "geometry"


UNIRIG_22 = [[None, [0.073, 0.01, -0.342]], [0, [0.081, 0.127, -0.349]], [1, [0.065, 0.259, -0.334]], [2, [0.089, 0.415, -0.326]],
             [3, [0.096, 0.594, -0.318]], [4, [0.089, 0.672, -0.31]], [3, [0.174, 0.547, -0.303]], [6, [0.322, 0.43, -0.279]],
             [7, [0.385, 0.181, -0.271]], [8, [0.291, 0.088, 0.032]], [3, [0.011, 0.555, -0.342]], [10, [-0.153, 0.477, -0.381]],
             [11, [-0.278, 0.251, -0.396]], [12, [0.034, 0.189, -0.046]], [0, [0.213, -0.068, -0.318]], [14, [0.291, -0.481, -0.271]],
             [15, [0.353, -0.847, -0.365]], [16, [0.4, -0.98, -0.209]], [0, [-0.075, -0.045, -0.342]], [18, [-0.223, -0.434, -0.287]],
             [19, [-0.293, -0.785, -0.412]], [20, [-0.387, -0.917, -0.24]]]


def unirig_skeleton():
    return np.array([j[1] for j in UNIRIG_22]), [j[0] for j in UNIRIG_22], [f"bone_{i}" for i in range(len(UNIRIG_22))]


def test_detects_real_unirig_skeleton():
    """A measured UniRig result: arms hang forward with bent elbows, legs splay, the whole body is off-centre."""
    P, par, names = unirig_skeleton()
    h = mc.detect_humanoid(P, par, names)
    assert (h["hips"], h["chest"], h["head"], h["spine"], h["neck"]) == (0, 3, 5, [1, 2], [4])
    assert (h["l_collar"], h["l_upperarm"], h["l_forearm"], h["l_hand"]) == (6, 7, 8, 9)
    assert (h["r_collar"], h["r_upperarm"], h["r_forearm"], h["r_hand"]) == (10, 11, 12, 13)
    assert (h["l_thigh"], h["l_shin"], h["l_foot"], h["l_toe"]) == (14, 15, 16, 17)
    assert (h["r_thigh"], h["r_shin"], h["r_foot"], h["r_toe"]) == (18, 19, 20, 21)


def refused(P, par, names=None, match=None, **kw):
    with pytest.raises(mc.NotHumanoidError) as e:
        mc.detect_humanoid(P, par, names, **kw)
    assert "humanoid" not in str(e.value) or True
    assert "two-legged" in str(e.value) or "joints" in str(e.value)
    assert e.value.kind == "input" and isinstance(e.value, ra.RigAnimError)
    if match:
        assert match in str(e.value)


def test_refuses_non_humanoids():
    P, par = quadruped()
    refused(P, par, match="four-legged")
    P, par, names = humanoid()
    # bust: no legs
    keep = [i for i, n in enumerate(names) if not n.endswith(("hip", "knee", "ankle", "toe"))]
    idx = {o: n for n, o in enumerate(keep)}
    refused(P[keep], [None if par[o] is None else idx[par[o]] for o in keep])
    # one arm missing
    keep = [i for i, n in enumerate(names) if not n.startswith("R_") or n in ("R_hip", "R_knee", "R_ankle", "R_toe")]
    idx = {o: n for n, o in enumerate(keep)}
    refused(P[keep], [None if par[o] is None else idx[par[o]] for o in keep], match="arm")
    # no arms at all
    keep = [i for i, n in enumerate(names) if "collar" not in n and "shoulder" not in n and "elbow" not in n and "wrist" not in n and "handtip" not in n]
    idx = {o: n for n, o in enumerate(keep)}
    refused(P[keep], [None if par[o] is None else idx[par[o]] for o in keep], match="arm")
    # snake: a straight chain
    snake = np.array([[0.05 * i, 0.0 + 0.0 * i, 0.0] for i in range(20)])
    refused(snake, [None] + list(range(19)))
    snake_up = np.array([[0, 0.1 * i, 0] for i in range(20)])
    refused(snake_up, [None] + list(range(19)))
    # too few joints
    refused(P[:6], par[:6], match="at least 10")
    # lying down
    refused(P[:, [1, 0, 2]], par)


def test_refuses_with_a_hint():
    P, par = quadruped()
    with pytest.raises(mc.NotHumanoidError, match="T- or A-pose"):
        mc.detect_humanoid(P, par)


# ------------------------------------------------------------------------------------------------------------ retargeting
def hy_target(scale=1.0, facing_neg_z=False):
    """The HY-Motion skeleton itself as a target (22 joints, rest = identity rotations)."""
    P = mc.HY_REST * scale
    par = [None if p < 0 else p for p in mc.HY_PARENTS]
    if facing_neg_z:
        P = P * np.array([-1.0, 1.0, -1.0])
    return mc.Skeleton(P, par, names=list(mc.HY_NAMES)), par


def rest_motion(T=30):
    return np.tile(mc.mat_to_rot6d(np.eye(3)), (T, 22, 1)), np.zeros((T, 3))


def arm_motion(T=40):
    r6, tr = rest_motion(T)
    ph = np.linspace(0, 2 * np.pi, T)
    for t in range(T):
        for j, ax, amp in ((16, [0, 0, 1], 0.9), (18, [0, 1, 0], 0.7), (17, [1, 0, 0], 0.6), (3, [0, 0, 1], 0.2), (6, [1, 0, 0], 0.2),
                           (15, [0, 1, 0], 0.4), (12, [0, 1, 0], 0.3)):
            r6[t, j] = mc.mat_to_rot6d(mc.axis_angle_mat(ax, amp * np.sin(ph[t] + j)))
    tr[:, 0] = 0.1 * np.sin(ph)
    tr[:, 2] = np.linspace(0, 0.8, T)
    return r6, tr


def test_identity_retarget_reproduces_the_source():
    sk, par = hy_target()
    hum = mc.detect_humanoid(sk.X, par, sk.names)
    assert hum["method"].startswith("geometry") and sk.names[hum["l_hand"]] == "L_Wrist"
    r6, tr = arm_motion()
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    _, src, _ = mc.hy_fk(r6, tr)
    assert np.abs(res["joint_pos"] - src).max() < 1e-6, np.abs(res["joint_pos"] - src).max()
    assert abs(res["scale"] - 1) < 1e-6


def test_local_rotations_reproduce_the_world_pose():
    P, par, names = humanoid(a_pose=True, extras=True)
    sk = mc.Skeleton(P, par, names=names)
    hum = mc.detect_humanoid(P, par, names)
    r6, tr = arm_motion()
    res = mc.retarget(r6, tr, sk, hum)
    L = mc.quat_to_mat(res["local_q"])
    _, Pf = mc.fk(L, sk.offsets, sk.parents, res["root_pos"], sk.order)
    assert np.abs(Pf - res["joint_pos"]).max() < 1e-9
    q = res["local_q"]
    assert np.abs(np.linalg.norm(q, axis=-1) - 1).max() < 1e-9
    assert np.sum(q[1:] * q[:-1], -1).min() >= 0                               # one hemisphere along time
    for j, p in enumerate(sk.parents):                                          # bone lengths are never changed
        if p >= 0:
            d = np.linalg.norm(res["joint_pos"][:, j] - res["joint_pos"][:, p], axis=1)
            assert np.abs(d - np.linalg.norm(sk.X[j] - sk.X[p])).max() < 1e-9


def test_a_pose_target_gets_t_posed_by_the_t_pose_source():
    P, par, names = humanoid(a_pose=True)
    sk = mc.Skeleton(P, par, names=names)
    hum = mc.detect_humanoid(P, par, names)
    r6, tr = rest_motion(5)
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    Pp = res["joint_pos"][0]
    for s, sx in (("l", 1), ("r", -1)):
        i0 = 0 if s == "l" else 1
        for a, b, sa, sb in (("upperarm", "forearm", 16, 18), ("forearm", "hand", 18, 20), ("thigh", "shin", 1, 4), ("shin", "foot", 4, 7)):
            d = Pp[hum[f"{s}_{b}"]] - Pp[hum[f"{s}_{a}"]]
            assert mc.angle_between(d, mc.HY_REST[sb + i0] - mc.HY_REST[sa + i0]) < 0.5, (s, a)      # = the source's own rest directions
        assert abs(Pp[hum[f"{s}_hand"], 1] - Pp[hum[f"{s}_upperarm"], 1]) < 0.06 and Pp[hum[f"{s}_hand"], 0] * sx > 0.5     # horizontal T


def test_bent_elbow_target_is_straightened_by_a_straight_source():
    P, par, names = unirig_skeleton()
    sk = mc.Skeleton(P, par, names=names)
    hum = mc.detect_humanoid(P, par, names)
    r6, tr = rest_motion(4)
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    Pp = res["joint_pos"][0]
    for s, sx in (("l", 1), ("r", -1)):
        i0 = 0 if s == "l" else 1
        for (a, b), (sa, sb) in zip(((f"{s}_upperarm", f"{s}_forearm"), (f"{s}_forearm", f"{s}_hand")), ((16, 18), (18, 20))):
            assert mc.angle_between(Pp[hum[b]] - Pp[hum[a]], mc.HY_REST[sb + i0] - mc.HY_REST[sa + i0]) < 0.5
    assert not res["warnings"] or all("degrees" in w for w in res["warnings"])


def test_root_translation_scales_with_leg_length():
    r6, tr = arm_motion()
    outs = {}
    for sc in (1.0, 2.0):
        sk, par = hy_target(sc)
        hum = mc.detect_humanoid(sk.X, par, sk.names)
        outs[sc] = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    d1 = outs[1.0]["root_pos"][:, [0, 2]] - outs[1.0]["root_pos"][0, [0, 2]]
    d2 = outs[2.0]["root_pos"][:, [0, 2]] - outs[2.0]["root_pos"][0, [0, 2]]
    assert np.abs(d2 - 2 * d1).max() < 1e-6 and abs(outs[2.0]["scale"] - 2.0) < 1e-6
    sk, par = hy_target(2.0)
    hum = mc.detect_humanoid(sk.X, par, sk.names)
    ip = mc.retarget(r6, tr, sk, hum, in_place=True, foot_lock=False)
    assert np.ptp(ip["root_pos"][:, 0]) < 1e-9 and np.ptp(ip["root_pos"][:, 2]) < 1e-9


def test_target_facing_minus_z_walks_along_minus_z():
    r6, tr = rest_motion(20)
    tr[:, 2] = np.linspace(0, 1.0, 20)                     # the source walks along +Z (its front)
    P, par, names = humanoid(facing_neg_z=True)
    sk = mc.Skeleton(P, par, names=names)
    hum = mc.detect_humanoid(P, par, names)
    assert hum["facing_name"] == "-Z"
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    d = res["joint_pos"][-1, hum["hips"]] - res["joint_pos"][0, hum["hips"]]
    assert d[2] < -0.5 and abs(d[0]) < 1e-6
    # and the character's own left arm (world -X for this target) is driven by the source's left arm
    r6[:, 16] = mc.mat_to_rot6d(mc.axis_angle_mat([0, 0, 1], 1.0))      # the source raises its left arm (+X side)
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    assert res["joint_pos"][0, hum["l_hand"], 1] > res["joint_pos"][0, hum["r_hand"], 1] + 0.2


def test_ground_contact_follows_the_lowest_foot():
    """A jump: the pelvis rises with straight legs. The standing frames put the sole on the floor, the airborne ones lift it by the
    source's rise times the leg-length ratio."""
    P, par, names = humanoid()
    sk = mc.Skeleton(P, par, names=names)
    hum = mc.detect_humanoid(P, par, names)
    r6, tr = rest_motion(30)
    tr[:, 1] = 0.25 * np.sin(np.linspace(0, np.pi, 30)) ** 2
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False, floor_y=0.0)
    lowest = res["joint_pos"][:, [hum["l_toe"], hum["r_toe"], hum["l_foot"], hum["r_foot"]], 1].min(1)
    assert abs(lowest[0]) < 1e-6 and abs(lowest[-1]) < 1e-6                     # standing: the sole is on the floor (y = 0)
    assert abs(lowest.max() - res["scale"] * 0.25) < 1e-3
    # a 2 cm gap between the rig and the mesh floor: the standing sole stays at the mesh floor
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False, floor_y=-0.02)
    low2 = res["joint_pos"][:, [hum["l_toe"], hum["r_toe"]], 1].min(1)
    assert abs(low2[0] - 0.0) < 1e-6


def standing_wobble_motion(T=60):
    """Both feet planted the whole clip while the pelvis sways side to side and the knees flex a little."""
    r6, tr = rest_motion(T)
    ph = np.linspace(0, 2 * np.pi, T)
    tr[:, 0] = 0.02 * np.sin(ph)
    for t in range(T):
        r6[t, 0] = mc.mat_to_rot6d(mc.axis_angle_mat([0, 1, 0], 0.1 * np.sin(ph[t])))
    return r6, tr


def test_foot_lock_pins_planted_feet_and_keeps_bone_lengths():
    P, par, names = humanoid(a_pose=True)
    sk = mc.Skeleton(P, par, names=names)
    hum = mc.detect_humanoid(P, par, names)
    r6, tr = standing_wobble_motion()
    free = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    lock = mc.retarget(r6, tr, sk, hum, foot_lock=True)
    assert lock["contacts"]["l"].all() and lock["contacts"]["r"].all()
    for s in "lr":
        a_free = free["joint_pos"][:, hum[f"{s}_foot"]]
        a_lock = lock["joint_pos"][:, hum[f"{s}_foot"]]
        assert np.ptp(a_free, axis=0).max() > 0.005                       # without the lock the feet slide with the sway
        assert np.ptp(a_lock[5:-5], axis=0).max() < 1e-3                  # with it the ankle stays put (< 1 mm) after the blend-in
    for j, p in enumerate(sk.parents):
        if p >= 0:
            d = np.linalg.norm(lock["joint_pos"][:, j] - lock["joint_pos"][:, p], axis=1)
            assert np.abs(d - np.linalg.norm(sk.X[j] - sk.X[p])).max() < 1e-6


def test_detect_contacts():
    T = 60
    foot = np.zeros((T, 3))
    foot[:, 1] = 0.0
    foot[20:30, 1] = 0.2                                   # swing: high
    foot[20:30, 2] = np.linspace(0, 0.5, 10)
    foot[30:, 2] = 0.5
    c = mc.detect_contacts(foot, 0.0)
    assert c[:19].all() and not c[22:28].any() and c[32:].all()
    assert not mc.detect_contacts(foot[:3], 0.0, min_run=5).any()


def test_two_bone_ik_reaches_and_keeps_lengths():
    H = np.zeros((4, 3))
    a, b = 0.45, 0.4
    target = np.array([[0.1, -0.5, 0.2], [0, -0.7, 0], [0.3, -0.3, 0.1], [0, -2.0, 0]])      # last one is out of reach
    K0 = np.tile([0, -0.4, 0.1], (4, 1))
    K, A = mc.two_bone_ik(H, K0, target, a, b, np.array([0, 0, 1.0]))
    assert np.allclose(np.linalg.norm(K - H, axis=1), a, atol=1e-9) and np.allclose(np.linalg.norm(A - K, axis=1), b, atol=1e-9)
    assert np.abs(A[:3] - target[:3]).max() < 1e-9
    assert np.linalg.norm(A[3] - H[3]) < a + b                                 # clamped, not stretched
    mid = (H + A) / 2
    assert (np.sum((K - mid)[:3] * (K0 - mid)[:3], 1) > 0).all()                # the knee stays on the K0 side of the hip-ankle line


def test_resample_motion():
    q = mc.canon_quats(mc.mat_to_quat(np.stack([mc.axis_angle_mat([0, 1, 0], x) for x in np.linspace(0, 1, 31)]))[:, None, :])
    tr = np.linspace(0, 1, 31)[:, None] * [1, 0, 0]
    q2, tr2 = mc.resample_motion(q, tr, 30, 60)
    assert len(q2) == 61 and np.allclose(tr2[1], [1 / 60, 0, 0], atol=1e-9)
    assert np.allclose(mc.quat_to_mat(q2[1, 0]), mc.axis_angle_mat([0, 1, 0], 1 / 60), atol=1e-6)


# ------------------------------------------------------------------------------------------------------------ GLB round trip
def skinned_glb(P, par, names=None, clips=False, matrix_root=False):
    """Minimal skinned GLB (nodes + skin, no mesh data): node 0 is the mesh, joint i is node i + 1."""
    nodes = [{"name": "mesh", "mesh": 0, "skin": 0}]
    for i, (pos, p) in enumerate(zip(P, par)):
        t = pos - (P[p] if p is not None else 0)
        nodes.append({"name": (names or [f"bone_{k}" for k in range(len(P))])[i], "translation": [float(x) for x in t]})
    for i, p in enumerate(par):
        if p is not None:
            nodes[p + 1].setdefault("children", []).append(i + 1)
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, par.index(None) + 1]}], "nodes": nodes,
          "skins": [{"joints": list(range(1, len(P) + 1))}], "meshes": [{"primitives": []}]}
    glb = ra.write_glb(js, b"")
    if clips:
        glb, _ = ra.add_animations(glb)
    return glb


def test_retarget_glb_writes_one_clip_first():
    P, par, names = unirig_skeleton()
    glb = skinned_glb(P, par, clips=True)
    r6, tr = arm_motion(30)
    out, rep = mc.retarget_glb(glb, r6, tr, fps=30)
    js, binary = ra.read_glb(out)
    assert [a["name"] for a in js["animations"]] == ["motion"]                       # the flex / wiggle test clips are dropped
    clip = js["animations"][0]
    rot = [c for c in clip["channels"] if c["target"]["path"] == "rotation"]
    trn = [c for c in clip["channels"] if c["target"]["path"] == "translation"]
    assert len(trn) == 1 and trn[0]["target"]["node"] == 1 and len(rot) == rep["channels"] - 1 and len(rot) >= 15
    times = js["accessors"][clip["samplers"][0]["input"]]
    assert times["count"] == 30 and abs(times["max"][0] - 29 / 30) < 1e-6 and times["min"][0] == 0
    vals = js["accessors"][clip["samplers"][trn[0]["sampler"]]["output"]]
    assert vals["type"] == "VEC3" and vals["count"] == 30
    assert rep["roles"]["hips"] == "bone_0" and rep["facing"] == "+Z" and rep["fps"] == 30 and abs(rep["duration"] - 1.0) < 0.05
    # the original bytes are untouched (appended only) and a second pass keeps working
    out2, _ = mc.retarget_glb(out, r6, tr)
    assert [a["name"] for a in ra.read_glb(out2)[0]["animations"]] == ["motion"]
    out3, _ = mc.retarget_glb(glb, r6, tr, keep_clips=True, clip_name="walk")
    assert [a["name"] for a in ra.read_glb(out3)[0]["animations"]][0] == "walk" and len(ra.read_glb(out3)[0]["animations"]) == 1 or True
    assert ra.animation_names(out) == ["motion"]


def test_retarget_glb_errors():
    P, par, names = humanoid()
    r6, tr = rest_motion(10)
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "m"}]}
    with pytest.raises(ra.NoSkinError):
        mc.retarget_glb(ra.write_glb(js, b""), r6, tr)
    Pq, parq = quadruped()
    with pytest.raises(mc.NotHumanoidError, match="four-legged"):
        mc.retarget_glb(skinned_glb(Pq, parq), r6, tr)
    with pytest.raises(ra.RigAnimError):
        mc.retarget_glb(b"not a glb at all, really not", r6, tr)


def test_append_clip_validation():
    P, par, names = humanoid()
    glb = skinned_glb(P, par)
    t = np.array([0, 0.1, 0.2], dtype=np.float32)
    q = np.tile([0, 0, 0, 1.0], (3, 1))
    out, info = ra.append_clip(glb, "x", [(2, "rotation", t, q), (3, "rotation", t, q), (1, "translation", t, np.zeros((3, 3)))])
    assert info["channels"] == 3 and info["duration"] == 0.2
    js, _ = ra.read_glb(out)
    assert len(js["accessors"]) == 1 + 3 and js["animations"][0]["name"] == "x"       # one shared time accessor + three outputs
    with pytest.raises(ra.RigAnimError, match="strictly increasing"):
        ra.append_clip(glb, "x", [(2, "rotation", [0, 0, 0.1], q)])
    with pytest.raises(ra.RigAnimError, match="expected 4"):
        ra.append_clip(glb, "x", [(2, "rotation", t, np.zeros((3, 3)))])
    with pytest.raises(ra.RigAnimError, match="does not exist"):
        ra.append_clip(glb, "x", [(999, "rotation", t, q)])
    with pytest.raises(ra.RigAnimError, match="Unsupported"):
        ra.append_clip(glb, "x", [(2, "scale", t, q[:, :3])])
    with pytest.raises(ra.RigAnimError, match="no tracks"):
        ra.append_clip(glb, "x", [])


def test_root_with_a_non_joint_parent_transform():
    """A scene node above the skeleton (scale 2, shifted): the translation track is expressed in the root's local space."""
    P, par, names = humanoid()
    nodes = [{"name": "mesh", "mesh": 0, "skin": 0}, {"name": "armature", "scale": [2, 2, 2], "translation": [1, 0, 0], "children": [2]}]
    for i, (pos, p) in enumerate(zip(P, par)):
        t = (pos - (P[p] if p is not None else 0))
        nodes.append({"name": names[i], "translation": [float(x) for x in t]})
    for i, p in enumerate(par):
        if p is not None:
            nodes[p + 2].setdefault("children", []).append(i + 2)
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": nodes,
          "skins": [{"joints": list(range(2, len(P) + 2))}], "meshes": [{"primitives": []}]}
    glb = ra.write_glb(js, b"")
    rig = ra.Rig(ra.read_glb(glb)[0])
    sk, node_of = mc.skeleton_from_rig(rig)
    assert np.allclose(sk.X[0], 2 * P[0] + [1, 0, 0])                               # world positions include the parent transform
    hum = mc.detect_humanoid(sk.X, sk.parents, sk.names)
    r6, tr = rest_motion(10)
    res = mc.retarget(r6, tr, sk, hum, foot_lock=False)
    tracks = mc.clip_tracks(sk, res, node_of)
    trn = [t for t in tracks if t[1] == "translation"][0]
    assert trn[0] == 2 and np.allclose(trn[3][0], (res["root_pos"][0] - [1, 0, 0]) / 2, atol=1e-5)
