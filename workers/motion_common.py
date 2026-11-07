"""Text-to-motion helpers (numpy + stdlib only; unit-tested in tests/test_motion.py, used by workers/motion_hy_worker.py).

  rot6d_to_mat / FK        HY-Motion's SMPL-H body (22 body joints used) as global rotations + positions
  detect_humanoid          finds pelvis / spine / head / arms / legs on ANY skeleton (UniRig's `bone_0 ...`) from joint positions
                           and parents alone (names are used first when they match a known convention); refuses clearly otherwise
  retarget                 global-rotation transfer with per-bone rest alignment, root translation scaled by leg length,
                           ground contact by the lowest foot, optional two-bone IK foot lock
  retarget_glb             rigged GLB bytes + HY-Motion output -> GLB bytes with one animation clip (via workers/rig_anim.py)

Conventions (the studio's): +Y up, the character faces +Z, its left side is +X (L = up x front). HY-Motion's rest pose is a T-pose
with identity local rotations on exactly that frame. Rotations are 3x3 column-vector matrices ("W" world, "L" local, "G" source
global, "Q" per-bone rest alignment, "R0" target rest world rotation); quaternions are glTF xyzw.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

import rig_anim
from rig_anim import RigAnimError

EPS = 1e-9
X, Y, Z = np.eye(3)


class NotHumanoidError(RigAnimError):
    """The skeleton is not a two-legged, two-armed upright character. `kind` is 'input' (user-fixable)."""
    kind = "input"


HINT = ("Text to motion needs a two-legged, two-armed character standing upright (T- or A-pose). "
        "Re-rig with another seed, or use a humanoid mesh.")

# ---------------------------------------------------------------------------------------------------- HY-Motion skeleton
# SMPL-H order, first 22 joints (the body; joints 22..51 are fingers and are ignored). Rest = HY-Motion's WoodenMesh j_template.
HY_NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
            "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
HY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
HY_REST = np.array([
    [-0.00180, -0.19083, 0.02822],
    [0.08058, -0.28077, 0.02512],
    [-0.08058, -0.28077, 0.02512],
    [0.00000, -0.09664, 0.03173],
    [0.09984, -0.65808, 0.01466],
    [-0.09785, -0.65818, 0.01466],
    [0.00000, 0.02598, 0.02919],
    [0.10107, -1.05622, -0.03746],
    [-0.09700, -1.05632, -0.03746],
    [0.00000, 0.10624, 0.00501],
    [0.11363, -1.12786, 0.10525],
    [-0.10918, -1.12802, 0.10525],
    [0.00000, 0.30207, -0.01495],
    [0.05469, 0.22507, -0.02023],
    [-0.05469, 0.22507, -0.02023],
    [0.00000, 0.39772, -0.01645],
    [0.16535, 0.23745, -0.01686],
    [-0.16535, 0.23745, -0.01686],
    [0.43344, 0.24939, -0.02956],
    [-0.43344, 0.24939, -0.02956],
    [0.70249, 0.25342, -0.04273],
    [-0.70249, 0.25342, -0.04273],
])
# the bone-direction child used for every source joint that drives something (end effectors: Head, wrists, toes)
HY_CHILD = {3: 6, 6: 9, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21, 1: 4, 2: 5, 4: 7, 5: 8, 7: 10, 8: 11}
HY_FEET = (7, 8, 10, 11)
HY_FPS = 30


def load_hy_rest(path) -> np.ndarray:
    """Rest joints of HY-Motion's WoodenMesh (`j_template.bin`, float32 [52,3]) -> first 22 rows; the embedded HY_REST is a copy."""
    j = np.fromfile(str(path), dtype="<f4").reshape(-1, 3)[:22].astype(np.float64)
    return j


# ---------------------------------------------------------------------------------------------------- rotation maths
def unit(v, fallback=(0.0, 1.0, 0.0)):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.maximum(n, EPS)
    if v.ndim == 1:
        return out if float(n[0]) > EPS else np.asarray(fallback, dtype=np.float64)
    return np.where(n > EPS, out, np.asarray(fallback, dtype=np.float64))


def rot6d_to_mat(r6) -> np.ndarray:
    """HY-Motion's 6D rotation (Zhou et al.; first two matrix COLUMNS stored row by row: [r00 r01 r10 r11 r20 r21]) -> [...,3,3]."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0::2], r6[..., 1::2]
    b1 = unit(a1, (1, 0, 0))
    b2 = unit(a2 - np.sum(b1 * a2, -1, keepdims=True) * b1, (0, 1, 0))
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], -1)


def mat_to_rot6d(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.stack([m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1], m[..., 2, 0], m[..., 2, 1]], -1)


def mat_to_quat(m) -> np.ndarray:
    """[...,3,3] -> unit quaternions xyzw [...,4] (vectorised, numerically safe branch per element)."""
    m = np.asarray(m, dtype=np.float64)
    sh = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    cand = np.stack([1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22, 1 + m00 + m11 + m22], 1)
    k = np.argmax(cand, 1)
    q = np.zeros((len(m), 4))
    for i in range(len(m)):
        a = k[i]
        s = 2.0 * math.sqrt(max(cand[i, a], EPS))
        r = m[i]
        if a == 0:
            q[i] = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif a == 1:
            q[i] = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        elif a == 2:
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
        else:
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
    return unit(q, (0, 0, 0, 1)).reshape(*sh, 4)


def quat_to_mat(q) -> np.ndarray:
    q = unit(np.asarray(q, dtype=np.float64), (0, 0, 0, 1))
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def canon_quats(q) -> np.ndarray:
    """q [T,...,4]: flip signs so consecutive frames stay in one hemisphere (LINEAR / slerp never takes the long way round)."""
    q = np.array(q, dtype=np.float64)
    for t in range(1, len(q)):
        flip = np.sum(q[t] * q[t - 1], -1) < 0
        q[t] = np.where(flip[..., None], -q[t], q[t])
    return q


def slerp(a, b, t):
    """Quaternion slerp (xyzw), broadcasting; t in [0,1]."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = np.sum(a * b, -1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d).clip(0, 1)
    th = np.arccos(d)
    s = np.sin(th)
    t = np.asarray(t, dtype=np.float64)[..., None]
    wa = np.where(s > 1e-6, np.sin((1 - t) * th) / np.maximum(s, 1e-6), 1 - t)
    wb = np.where(s > 1e-6, np.sin(t * th) / np.maximum(s, 1e-6), t)
    return unit(wa * a + wb * b, (0, 0, 0, 1))


def axis_angle_mat(axis, angle) -> np.ndarray:
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def arc_rotation(a, b, prefer_axis=None) -> np.ndarray:
    """Minimal rotation taking direction a onto direction b (Rodrigues). Antiparallel: pi about an axis perpendicular to a
    (`prefer_axis` projected, when given). Single vectors (3,) or batches (N,3) -> (3,3) / (N,3,3)."""
    a, b = unit(a), unit(b)
    if a.ndim == 2:
        return np.stack([arc_rotation(x, y, prefer_axis) for x, y in zip(a, b)])
    c = float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-9:
        ax = np.asarray(prefer_axis if prefer_axis is not None else (1, 0, 0), dtype=np.float64)
        ax = ax - a * (a @ ax)
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, (0, 0, 1) if abs(a[2]) < 0.9 else (1, 0, 0))
        return axis_angle_mat(ax, math.pi)
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) / (1 + c)


def arc_rotations(a, b) -> np.ndarray:
    """Vectorised arc_rotation for unit vectors a, b [N,3] (antiparallel pairs get pi about a perpendicular axis)."""
    a, b = unit(a), unit(b)
    c = np.sum(a * b, -1)
    v = np.cross(a, b)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    R = np.eye(3) + K + (K @ K) / np.maximum(1 + c, 1e-9)[..., None, None]
    for i in np.flatnonzero(c < -1 + 1e-9):
        R[i] = arc_rotation(a[i], b[i])
    return R


def basis(d, n):
    """Right-handed frame with columns [d, n_perp, d x n_perp]; None when n is parallel to d."""
    d = unit(d)
    n = np.asarray(n, dtype=np.float64)
    n = n - d * (d @ n)
    if np.linalg.norm(n) < 1e-6:
        return None
    n = unit(n)
    return np.stack([d, n, np.cross(d, n)], 1)


def angle_between(a, b) -> float:
    return math.degrees(math.acos(float(np.clip(unit(a) @ unit(b), -1, 1))))


# ---------------------------------------------------------------------------------------------------- skeleton + FK
def topo_order(parents) -> list[int]:
    """Parents before children for a parent-index list (-1 / None = root)."""
    J = len(parents)
    order, seen = [], [False] * J

    def visit(j):
        chain = []
        while j >= 0 and not seen[j]:
            chain.append(j)
            seen[j] = True
            p = parents[j]
            j = -1 if p is None else int(p)
        order.extend(reversed(chain))

    for j in range(J):
        visit(j)
    # `visit` pushes a joint after its ancestors only if they were unseen: fix up any remaining inversions by a stable sort on depth
    depth = [0] * J
    for j in range(J):
        d, p = 0, parents[j]
        while p is not None and p >= 0 and d < J:
            d, p = d + 1, parents[p]
        depth[j] = d
    return sorted(range(J), key=lambda j: (depth[j], j))


class Skeleton:
    """Target skeleton: rest world positions X [J,3], parents (-1 = root), rest world rotations R0 [J,3,3], names.
    `parent_rot` is the rest rotation of whatever sits above the roots (non-joint nodes), identity for UniRig."""

    def __init__(self, positions, parents, rest_rot=None, names=None, parent_rot=None, parent_mat=None):
        self.X = np.asarray(positions, dtype=np.float64)
        self.parents = [-1 if p is None else int(p) for p in parents]
        J = len(self.parents)
        self.J = J
        self.R0 = np.tile(np.eye(3), (J, 1, 1)) if rest_rot is None else np.asarray(rest_rot, dtype=np.float64)
        self.names = list(names) if names is not None else [f"bone_{i}" for i in range(J)]
        self.parent_mat = np.eye(4) if parent_mat is None else np.asarray(parent_mat, dtype=np.float64)   # world matrix above the roots
        self.parent_rot = np.eye(3) if parent_rot is None else np.asarray(parent_rot, dtype=np.float64)
        self.order = topo_order(self.parents)
        self.children = [[] for _ in range(J)]
        for j, p in enumerate(self.parents):
            if p >= 0:
                self.children[p].append(j)
        self.offsets = np.zeros((J, 3))                       # rest offset in the PARENT's rest frame
        for j, p in enumerate(self.parents):
            if p >= 0:
                self.offsets[j] = self.R0[p].T @ (self.X[j] - self.X[p])

    def subtree(self, j) -> list[int]:
        out, stack = [], [j]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(self.children[n])
        return out


def fk(local, offsets, parents, root_pos, order=None, root_parent_rot=None):
    """Forward kinematics. local [T,J,3,3], offsets [J,3] (in the parent frame), root_pos [T,3] = world position of every root
    joint -> (world rotations [T,J,3,3], world positions [T,J,3])."""
    T, J = local.shape[:2]
    order = order if order is not None else topo_order(parents)
    W = np.zeros((T, J, 3, 3))
    P = np.zeros((T, J, 3))
    rp = np.eye(3) if root_parent_rot is None else root_parent_rot
    for j in order:
        p = parents[j]
        if p is None or p < 0:
            W[:, j] = rp @ local[:, j]
            P[:, j] = root_pos
        else:
            W[:, j] = W[:, p] @ local[:, j]
            P[:, j] = P[:, p] + np.einsum("tij,j->ti", W[:, p], offsets[j])
    return W, P


def hy_fk(rot6d, transl):
    """HY-Motion output (rot6d [T,>=22,6] with the root rotation first, transl [T,3]) -> (global rotations [T,22,3,3],
    positions [T,22,3], local rotations [T,22,3,3]). The root translation moves the pelvis joint: pelvis = HY_REST[0] + transl."""
    L = rot6d_to_mat(np.asarray(rot6d, dtype=np.float64)[:, :22])
    off = np.zeros((22, 3))
    for j in range(1, 22):
        off[j] = HY_REST[j] - HY_REST[HY_PARENTS[j]]
    W, P = fk(L, off, HY_PARENTS, HY_REST[0] + np.asarray(transl, dtype=np.float64))
    return W, P, L


def resample_motion(local_q, transl, fps_in, fps_out):
    """Slerp / lerp a clip to another frame rate. local_q [T,J,4] xyzw, transl [T,3] -> same with T' frames."""
    T = len(local_q)
    if fps_in == fps_out or T < 2:
        return local_q, transl
    n = max(2, int(round((T - 1) / fps_in * fps_out)) + 1)
    u = np.linspace(0, T - 1, n)
    i0 = np.floor(u).astype(int).clip(0, T - 2)
    f = u - i0
    q = slerp(local_q[i0], local_q[i0 + 1], f[:, None] * np.ones((1, local_q.shape[1])))
    tr = transl[i0] * (1 - f)[:, None] + transl[i0 + 1] * f[:, None]
    return q, tr


# ---------------------------------------------------------------------------------------------------- humanoid detection
_NAME_ROLES = [   # (role, regex on the lower-case name without side tokens); first match wins
    ("toe", r"toe|ball"), ("foot", r"foot|ankle"), ("thigh", r"upleg|upperleg|thigh|hip$"), ("shin", r"lowerleg|shin|calf|knee|^leg$"),
    ("hand", r"hand|wrist"), ("forearm", r"forearm|lowerarm|elbow"), ("upperarm", r"upperarm|^arm$|uparm"),
    ("collar", r"clavicle|collar|shoulder"),
]


def _name_tokens(raw: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", raw)]


def _roles_by_name(names):
    """Name cross-check for Mixamo / VRoid / SMPL / Unreal / Blender style names -> {role: joint} or None when incomplete."""
    found: dict[str, int] = {}
    for j, raw in enumerate(names):
        tok = [t for t in _name_tokens(raw) if t not in ("mixamorig", "j", "bip", "c")]
        side = "l" if ("left" in tok or "l" in tok) else "r" if ("right" in tok or "r" in tok) else None
        core = "".join(t for t in tok if t not in ("left", "right", "l", "r"))
        if side is None:
            if core in ("hips", "pelvis"):
                found.setdefault("hips", j)
            elif core == "head":
                found.setdefault("head", j)
            continue
        for role, rx in _NAME_ROLES:
            if re.search(rx, core):
                found.setdefault(f"{side}_{role}", j)
                break
    for s_ in "lr":      # SMPL / VRoid: "shoulder" is the clavicle when an explicit upper arm exists, otherwise it is the upper arm
        if f"{s_}_upperarm" not in found and f"{s_}_collar" in found and f"{s_}_forearm" in found:
            found[f"{s_}_upperarm"] = found.pop(f"{s_}_collar")
    need = ["hips", "head"] + [f"{s_}_{r}" for s_ in "lr" for r in ("thigh", "shin", "foot", "upperarm", "forearm", "hand")]
    return found if all(k in found for k in need) else None


class _Geo:
    """Tree helpers on joint positions X [J,3] and parents (None / -1 = root)."""

    def __init__(self, X, parents):
        self.X = np.asarray(X, dtype=np.float64)
        self.J = len(self.X)
        self.parents = [-1 if p is None else int(p) for p in parents]
        self.children = [[] for _ in range(self.J)]
        for j, p in enumerate(self.parents):
            if p >= 0:
                self.children[p].append(j)
        self.adj = [list(c) for c in self.children]
        for j, p in enumerate(self.parents):
            if p >= 0:
                self.adj[j].append(p)
        self._reach: dict = {}

    def subtree(self, j):
        out, st = [], [j]
        while st:
            n = st.pop()
            out.append(n)
            st.extend(self.children[n])
        return out

    def branch(self, node, frm):
        """Joints reachable from `node` without passing through `frm` (undirected tree)."""
        out, st = [], [(node, frm)]
        while st:
            n, f = st.pop()
            out.append(n)
            st.extend((m, n) for m in self.adj[n] if m != f)
        return out

    def reach_y(self, node, frm):
        key = (node, frm)
        if key not in self._reach:
            self._reach[key] = float(self.X[self.branch(node, frm)][:, 1].max())
        return self._reach[key]


def detect_humanoid(joints, parents, names=None, facing="auto") -> dict:
    """Find the humanoid structure of a skeleton. joints [J,3] rest world positions (+Y up), parents (None / -1 = root).

    `facing`: 'auto' (feet / knees decide, default +Z), or '+Z', '-Z', '+X', '-X'. Returns
      {hips, spine[], chest, neck[], head, {l,r}_{collar|None,upperarm,forearm,hand,thigh,shin,foot,toe|None}, facing (unit vector),
       left (unit vector), leg_length, height, warnings[], method}
    Raises NotHumanoidError (with a hint) when it is not a two-legged, two-armed upright character."""
    X = np.asarray(joints, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or len(X) != len(parents):
        raise NotHumanoidError("Bad skeleton data.")
    J = len(X)
    if J < 10:
        raise NotHumanoidError(f"The skeleton has only {J} joints; a character needs at least 10. {HINT}")
    fronts = {"+Z": (0, 0, 1), "-Z": (0, 0, -1), "+X": (1, 0, 0), "-X": (-1, 0, 0)}
    if facing in fronts:
        order = [facing]
    else:
        order = ["+Z", "-Z", "+X", "-X"]
    last_err, ok = None, []
    for f in order:
        try:
            ok.append((f, _detect(X, parents, names, np.asarray(fronts[f], dtype=np.float64))))
        except NotHumanoidError as e:
            last_err = last_err or e
    if not ok:
        raise last_err
    if len(ok) > 1:
        # several succeed when the figure is front/back ambiguous (+Z / -Z): the feet decide, default +Z
        votes = []
        for f, r in ok:
            fw = np.asarray(fronts[f])
            s = 0.0
            for side in "lr":
                a, t, k, h = r[f"{side}_foot"], r.get(f"{side}_toe"), r[f"{side}_shin"], r[f"{side}_thigh"]
                if t is not None:
                    s += float((X[t] - X[a]) @ fw)
                s += 0.5 * float((X[k] - (X[h] + X[a]) / 2) @ fw)
            votes.append(s)
        best = int(np.argmax(votes))
        if votes[best] <= 1e-4 or (ok[0][0] == "+Z" and votes[0] >= votes[best] - 1e-4):
            best = 0
        ok = [ok[best]]
    f, r = ok[0]
    r["facing_name"] = f
    return r


def _detect(X, parents, names, F):
    g = _Geo(X, parents)
    J = g.J
    up = np.array([0.0, 1.0, 0.0])
    Lx = np.cross(up, F)                              # left = up x front
    lat = X @ Lx
    ymin, ymax = float(X[:, 1].min()), float(X[:, 1].max())
    H = ymax - ymin
    if H < 1e-6:
        raise NotHumanoidError(f"The skeleton is flat. {HINT}")
    warnings: list[str] = []

    # ---- 1. feet: leaf joints near the lowest point, grouped by the child branch of the pelvis they belong to
    leaves = [j for j in range(J) if not g.children[j]]
    low = [j for j in leaves if X[j, 1] < ymin + 0.15 * H]
    if len(low) < 2:
        raise NotHumanoidError(f"No two feet found at the bottom of the skeleton. {HINT}")

    def floor_children(j):
        low_set = set(low)
        return [c for c in g.children[j] if low_set & set(g.subtree(c))]

    pelvis, best_depth = None, -1
    depth = [0] * J
    for j in range(J):
        d, p = 0, g.parents[j]
        while p >= 0:
            d, p = d + 1, g.parents[p]
        depth[j] = d
    for j in range(J):
        fc = floor_children(j)
        if len(fc) < 2:
            continue
        cl = [float(np.mean([lat[l] for l in low if l in set(g.subtree(c))])) for c in fc]
        if max(cl) - min(cl) > 0.06 * H and depth[j] > best_depth:
            pelvis, best_depth = j, depth[j]
    if pelvis is None:
        raise NotHumanoidError(f"No joint splits into a left and a right leg reaching the floor. {HINT}")
    fc = floor_children(pelvis)
    cl = {c: float(np.mean([lat[l] for l in low if l in set(g.subtree(c))])) for c in fc}
    left_c, right_c = max(cl, key=cl.get), min(cl, key=cl.get)
    mid = lat[pelvis]
    if not (cl[left_c] > mid > cl[right_c]):
        raise NotHumanoidError(f"Both legs are on the same side of the pelvis. {HINT}")
    leg_sub = set(g.subtree(left_c)) | set(g.subtree(right_c))
    stray = [l for l in low if l not in leg_sub]
    if stray:
        sl = lat[stray]
        if (sl.max() - sl.min()) > 0.08 * H or len(stray) > 3 or any(abs(lat[l] - mid) > 0.10 * H for l in stray):
            raise NotHumanoidError("More than two limbs reach the floor (four-legged or a creature with extra legs). " + HINT)
        warnings.append("a tail-like chain reaches the floor and is ignored")
    for c in fc:
        if c not in (left_c, right_c):
            low_in = [l for l in low if l in set(g.subtree(c))]
            if low_in and (lat[low_in].max() - lat[low_in].min()) > 0.08 * H:
                raise NotHumanoidError("More than two limbs reach the floor (four-legged). " + HINT)

    def leg_chain(c):
        chain = [c]
        while g.children[chain[-1]]:
            ch = g.children[chain[-1]]
            reach = [(float(X[g.subtree(k)][:, 1].min()), k) for k in ch]
            lowest = min(r[0] for r in reach)
            cand = [k for y, k in reach if y < lowest + 0.1 * H]           # branches that reach the floor level
            if len(cand) > 1:                                              # toe vs heel: the forward one is the toe
                cand = [max(cand, key=lambda k: float((X[g.subtree(k)] @ F).max() - X[chain[-1]] @ F))]
            chain.append(cand[0] if cand else min(reach)[1])
        return chain

    def split_leg(chain):
        n = len(chain)
        arc = [0.0]
        for a, b in zip(chain[:-1], chain[1:]):
            arc.append(arc[-1] + float(np.linalg.norm(X[b] - X[a])))
        ankle_i = None
        for i in range(2, n):
            if arc[-1] - arc[i] <= 0.45 * arc[i]:
                ankle_i = i
                break
        if ankle_i is None:
            raise NotHumanoidError(f"A leg has too few joints (need hip, knee and ankle). {HINT}")
        knee_i = min(range(1, ankle_i), key=lambda i: abs(arc[i] - arc[ankle_i] / 2))
        toe = chain[ankle_i + 1] if ankle_i + 1 < n else None
        return {"thigh": chain[0], "shin": chain[knee_i], "foot": chain[ankle_i], "toe": toe, "chain": chain}

    legs = {"l": split_leg(leg_chain(left_c)), "r": split_leg(leg_chain(right_c))}
    leg_len = np.mean([np.linalg.norm(X[v["shin"]] - X[v["thigh"]]) + np.linalg.norm(X[v["foot"]] - X[v["shin"]]) for v in legs.values()])
    if leg_len < 0.15 * H:
        raise NotHumanoidError(f"The legs are too short for a character (under 15% of the height). {HINT}")

    # ---- 2. spine: from the pelvis upwards through the neighbour that reaches highest (the tree is walked undirected, so a
    # rig rooted at the chest works too)
    spine, prev, cur = [pelvis], None, pelvis
    while len(spine) <= J:
        cands = [m for m in g.adj[cur] if m != prev and m not in leg_sub and g.reach_y(m, cur) > X[cur, 1] + 0.02 * H]
        if not cands:
            break
        prev, cur = cur, max(cands, key=lambda m: g.reach_y(m, cur))
        spine.append(cur)
    top = spine[-1]
    if X[top, 1] - X[pelvis, 1] < 0.25 * H or angle_between(X[top] - X[pelvis], up) > 35:
        raise NotHumanoidError(f"No upright spine above the pelvis. {HINT}")
    if len(spine) < 3:
        raise NotHumanoidError(f"The torso has too few joints (need pelvis, chest and head). {HINT}")

    # ---- 3. arms: side branches off the spine with a wide lateral reach
    spine_set = set(spine)
    cand_arms = []
    for si, s in enumerate(spine[1:], 1):
        for m in g.adj[s]:
            if m in spine_set or m in leg_sub:
                continue
            br = g.branch(m, s)
            ext = lat[br] - mid
            k = int(np.argmax(np.abs(ext)))
            if abs(ext[k]) > 0.15 * H:
                cand_arms.append((si, m, float(ext[k])))
    left_arms = [c for c in cand_arms if c[2] > 0]
    right_arms = [c for c in cand_arms if c[2] < 0]
    if not left_arms and not right_arms:
        raise NotHumanoidError(f"No arms found (nothing branches sideways off the spine). {HINT}")
    if not left_arms or not right_arms:
        raise NotHumanoidError(f"Only one arm found. {HINT}")
    la = max(left_arms, key=lambda c: abs(c[2]))
    ra_ = max(right_arms, key=lambda c: abs(c[2]))

    def arm_chain(si, m):
        chain, prev_, cur_ = [spine[si], m], spine[si], m
        while True:
            nbr = [k for k in g.adj[cur_] if k != prev_]
            if not nbr:
                break
            if len(nbr) >= 3:
                break
            if len(nbr) == 2:
                bl = [float(np.linalg.norm(X[k] - X[cur_])) for k in nbr]
                if max(bl) < 0.6 * float(np.linalg.norm(X[cur_] - X[prev_])):
                    break
            k = max(nbr, key=lambda k: max(abs(lat[g.branch(k, cur_)] - mid)))
            prev_, cur_ = cur_, k
            chain.append(cur_)
            if len(chain) > J:
                break
        return chain[1:]            # without the spine joint it hangs from

    arms = {}
    for side, (si, m, _) in (("l", la), ("r", ra_)):
        ch = arm_chain(si, m)
        arc = sum(float(np.linalg.norm(X[b] - X[a])) for a, b in zip(ch[:-1], ch[1:]))
        collar = None
        if len(ch) >= 4 and np.linalg.norm(X[ch[1]] - X[ch[0]]) < 0.35 * arc:
            collar, ch = ch[0], ch[1:]
        if len(ch) < 3:
            raise NotHumanoidError(f"An arm has too few joints (need shoulder, elbow and hand). {HINT}")
        arc = [0.0]
        for a, b in zip(ch[:-1], ch[1:]):
            arc.append(arc[-1] + float(np.linalg.norm(X[b] - X[a])))
        elbow_i = min(range(1, len(ch) - 1), key=lambda i: abs(arc[i] - arc[-1] / 2))
        arms[side] = {"collar": collar, "upperarm": ch[0], "forearm": ch[elbow_i], "hand": ch[-1], "chain": ch, "si": si}

    # ---- 4. torso roles
    chest_i = min(arms["l"]["si"], arms["r"]["si"])
    if arms["l"]["si"] != arms["r"]["si"]:
        warnings.append("the arms branch off different spine joints; the lower one is the chest")
    if chest_i >= len(spine) - 1:
        raise NotHumanoidError(f"No head above the shoulders. {HINT}")
    above = spine[chest_i + 1:]
    head = above[-1]
    branching = [j for j in above[:-1] if len(g.adj[j]) - 1 >= 2]       # a head carries several children (hair, jaw, eyes)
    if branching:
        head = branching[0]
    elif len(above) >= 2:
        # a tiny end nub above the head joint is not the head
        a_, b_ = spine[-2], spine[-1]
        if np.linalg.norm(X[b_] - X[a_]) < 0.25 * np.linalg.norm(X[a_] - X[spine[-3]]) and not g.children[b_]:
            head = a_
    h_i = spine.index(head)
    roles = {"hips": pelvis, "spine": spine[1:chest_i], "chest": spine[chest_i], "neck": spine[chest_i + 1:h_i], "head": head}
    for side in "lr":
        a, l = arms[side], legs[side]
        roles.update({f"{side}_collar": a["collar"], f"{side}_upperarm": a["upperarm"], f"{side}_forearm": a["forearm"],
                      f"{side}_hand": a["hand"], f"{side}_thigh": l["thigh"], f"{side}_shin": l["shin"], f"{side}_foot": l["foot"],
                      f"{side}_toe": l["toe"]})

    # ---- 5. sanity checks
    for side in "lr":
        if X[roles[f"{side}_foot"], 1] >= X[pelvis, 1]:
            raise NotHumanoidError(f"A foot is above the pelvis. {HINT}")
        if X[roles[f"{side}_hand"], 1] < X[roles[f"{side}_shin"], 1]:
            raise NotHumanoidError(f"A hand is below the knee (not an upright pose). {HINT}")
    roles.update({"facing": F, "left": Lx, "leg_length": float(leg_len), "height": H, "warnings": warnings, "method": "geometry"})
    if names is not None:
        by_name = _roles_by_name(list(names))
        if by_name:
            hand_ok = lambda s_: by_name[f"{s_}_hand"] in (roles[f"{s_}_hand"], g.parents[roles[f"{s_}_hand"]])      # noqa: E731
            ok = by_name["hips"] == pelvis and by_name["head"] in (head, spine[-1]) and all(
                by_name[f"{s_}_thigh"] == roles[f"{s_}_thigh"] and hand_ok(s_) for s_ in "lr")
            roles["method"] = "geometry+names" if ok else "geometry (names disagree)"
    return roles


# ---------------------------------------------------------------------------------------------------- retargeting
def _source_map(hum) -> dict:
    """target joint -> (HY source joint, HY child joint or None for end effectors, target chain child or None)."""
    m: dict[int, tuple] = {}
    tor = [hum["hips"]] + list(hum["spine"]) + [hum["chest"]] + list(hum["neck"]) + [hum["head"]]
    nxt = {a: b for a, b in zip(tor[:-1], tor[1:])}
    m[hum["hips"]] = (0, 3, hum["spine"][0] if hum["spine"] else hum["chest"])
    n = len(hum["spine"])
    for i, j in enumerate(hum["spine"]):
        u = (i + 1) / (n + 1)
        m[j] = (3 if u < 0.5 else 6, None, nxt[j])
    m[hum["chest"]] = (9, None, nxt[hum["chest"]])
    for j in hum["neck"]:
        m[j] = (12, None, nxt[j])
    m[hum["head"]] = (15, None, None)
    for s, (c, u, e, w, t, k, a, f) in (("l", (13, 16, 18, 20, 1, 4, 7, 10)), ("r", (14, 17, 19, 21, 2, 5, 8, 11))):
        ua, fa, ha = hum[f"{s}_upperarm"], hum[f"{s}_forearm"], hum[f"{s}_hand"]
        if hum[f"{s}_collar"] is not None:
            m[hum[f"{s}_collar"]] = (c, None, ua)
        m[ua], m[fa], m[ha] = (u, None, fa), (e, None, ha), (w, None, None)
        th, sh, ft, to = hum[f"{s}_thigh"], hum[f"{s}_shin"], hum[f"{s}_foot"], hum[f"{s}_toe"]
        m[th], m[sh], m[ft] = (t, None, sh), (k, None, ft), (a, None, to)
        if to is not None:
            m[to] = (f, None, None)
    return m


def compute_alignment(sk: Skeleton, hum: dict):
    """Per mapped target joint j: (source joint, Q_j) with W_j(t) = G_src(t) Q_j R0_j. Returns (dict, info)."""
    smap = _source_map(hum)
    X = sk.X
    info = {"warnings": [], "angles": {}}
    Q: dict[int, np.ndarray] = {}
    f = np.asarray(hum["facing"], dtype=np.float64)
    Qf = axis_angle_mat(Y, math.atan2(f[0], f[2]))             # the source is turned to the target's front, so are its rest directions
    rest = HY_REST @ Qf.T
    d_src = lambda s: unit(rest[HY_CHILD[s]] - rest[s])         # noqa: E731

    # hips: two-axis (lateral + up)
    h = hum["hips"]
    aT, uT = unit(X[hum["l_thigh"]] - X[hum["r_thigh"]]), unit(X[smap[h][2]] - X[h])
    aS, uS = unit(rest[1] - rest[2]), unit(rest[3] - rest[0])
    BT, BS = basis(aT, uT), basis(aS, uS)
    Q[h] = BS @ BT.T if BT is not None and BS is not None else np.eye(3)

    # bend axes: arms and legs share the plane normal of (upper, lower) when the target limb is clearly bent
    bend: dict[int, tuple] = {}
    for s in "lr":
        for up, lo, end, sup, slo, send in ((f"{s}_upperarm", f"{s}_forearm", f"{s}_hand", 16 if s == "l" else 17, 18 if s == "l" else 19, 20 if s == "l" else 21),
                                            (f"{s}_thigh", f"{s}_shin", f"{s}_foot", 1 if s == "l" else 2, 4 if s == "l" else 5, 7 if s == "l" else 8)):
            d1, d2 = unit(X[hum[lo]] - X[hum[up]]), unit(X[hum[end]] - X[hum[lo]])
            s1, s2 = unit(rest[slo] - rest[sup]), unit(rest[send] - rest[slo])
            nS = np.cross(s1, s2) if angle_between(s1, s2) > 5 else None
            if nS is None:      # straight source limb: natural bend direction (elbows forward, knees back)
                nS = Qf @ (np.array([0.0, -1.0 if s == "l" else 1.0, 0.0]) if "arm" in up else np.array([1.0, 0.0, 0.0]))
            nT = np.cross(d1, d2) if angle_between(d1, d2) > 15 else None
            bend[hum[up]] = bend[hum[lo]] = (nT, nS)

    def lim(j, dT, sj):
        dS = d_src(sj)
        nT, nS = bend.get(j, (None, None))
        BT_ = basis(dT, nT) if nT is not None else None
        BS_ = basis(dS, nS) if nS is not None else None
        q = BS_ @ BT_.T if BT_ is not None and BS_ is not None else arc_rotation(dT, dS, prefer_axis=nS)
        return q

    for j in sk.order:
        if j not in smap or j == h:
            continue
        sj, _, child = smap[j]
        if child is not None and sj in HY_CHILD:
            dT = unit(X[child] - X[j])
            Q[j] = lim(j, dT, sj)
            info["angles"][sk.names[j]] = round(angle_between(dT, d_src(sj)), 1)
            if info["angles"][sk.names[j]] > 150:
                info["warnings"].append(f"{sk.names[j]}: bone points {info['angles'][sk.names[j]]:.0f} degrees away from the template direction")
        else:
            Q[j] = None                       # end effector (or a bone whose chain child is missing): inherits the parent's Q below
    # end effectors take the Q of the nearest mapped ancestor
    for j in sk.order:
        if j in smap and Q.get(j) is None:
            p = sk.parents[j]
            while p >= 0 and Q.get(p) is None:
                p = sk.parents[p]
            Q[j] = Q[p] if p >= 0 else np.eye(3)
    align = {j: (smap[j][0], Q[j]) for j in smap}
    return align, info


def _propagate(sk: Skeleton, W, mapped):
    """Fill the unmapped joints (rigid with their parent at rest) into W [T,J,3,3] in place."""
    for j in sk.order:
        if mapped[j]:
            continue
        p = sk.parents[j]
        W[:, j] = (W[:, p] @ (sk.R0[p].T @ sk.R0[j])) if p >= 0 else sk.R0[j]
    return W


def _locals(sk: Skeleton, W):
    L = np.zeros_like(W)
    for j in range(sk.J):
        p = sk.parents[j]
        L[:, j] = (np.swapaxes(W[:, p], -1, -2) if p >= 0 else sk.parent_rot.T) @ W[:, j]
    return L


def detect_contacts(foot_pos, floor, *, height=0.04, speed=0.015, min_run=5, max_gap=2):
    """foot_pos [T,3] of a foot's lowest point joint -> bool [T]: low and slow. Short gaps are bridged, short runs dropped."""
    T = len(foot_pos)
    v = np.zeros(T)
    v[1:] = np.linalg.norm(np.diff(foot_pos, axis=0), axis=1)
    v[0] = v[1] if T > 1 else 0
    v = np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    c = (foot_pos[:, 1] - floor < height) & (v < speed)
    out = c.copy()
    runs = _runs(~c)                            # bridge short non-contact gaps
    for a, b in runs:
        if b - a <= max_gap and a > 0 and b < T:
            out[a:b] = True
    return _drop_short(out, min_run)


def _runs(mask):
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _drop_short(mask, min_run):
    out = mask.copy()
    for a, b in _runs(mask):
        if b - a < min_run:
            out[a:b] = False
    return out


def two_bone_ik(H, K_fk, A_target, a, b, fallback_pole):
    """Hip H, desired ankle A_target, bone lengths a (hip-knee) and b (knee-ankle); the knee stays on the side of K_fk.
    Vectorised over frames [N,3] -> (knee [N,3], reached ankle [N,3])."""
    d = A_target - H
    c = np.linalg.norm(d, axis=-1)
    c_cl = np.clip(c, abs(a - b) + 1e-4, a + b - 1e-4)
    dn = d / np.maximum(c, EPS)[:, None]
    l = (a * a - b * b + c_cl * c_cl) / (2 * c_cl)
    hgt = np.sqrt(np.maximum(a * a - l * l, 0))
    pole = (K_fk - H)
    pole = pole - np.sum(pole * dn, -1, keepdims=True) * dn
    pn = np.linalg.norm(pole, axis=-1, keepdims=True)
    fb = fallback_pole - np.sum(fallback_pole * dn, -1, keepdims=True) * dn
    pole = np.where(pn > 1e-5, pole / np.maximum(pn, EPS), unit(fb))
    K = H + dn * l[:, None] + pole * hgt[:, None]
    A = H + dn * c_cl[:, None]
    return K, A


def retarget(rot6d, transl, sk: Skeleton, hum: dict, *, in_place=False, foot_lock=True, arm_offset_deg=0.0, floor_y=None,
             fps_in=HY_FPS, fps=HY_FPS) -> dict:
    """HY-Motion output -> animation of the target skeleton.

    rot6d [T,>=22,6], transl [T,3] (HY-Motion's `rot6d` / `transl`). Returns
      {local_q [T,J,4] xyzw (every joint), root_pos [T,3] (world position of the skeleton root joint), joint_pos [T,J,3] (world),
       fps, scale, floor_y, contacts {l,r: [T] bool}, align {joint name: degrees}, warnings []}"""
    rest = HY_REST
    G, xs, _ = hy_fk(rot6d, transl)
    T = len(G)
    # source into the target's facing
    f = np.asarray(hum["facing"], dtype=np.float64)
    ang = math.atan2(f[0], f[2])                                   # rotation about +Y taking +Z to f
    Qf = axis_angle_mat(Y, ang)
    G = Qf @ G @ Qf.T
    xs = xs @ Qf.T

    align, info = compute_alignment(sk, hum)
    mapped = np.zeros(sk.J, dtype=bool)
    W = np.zeros((T, sk.J, 3, 3))
    for j, (sj, Qj) in align.items():
        mapped[j] = True
        W[:, j] = G[:, sj] @ Qj @ sk.R0[j]
    if arm_offset_deg:
        th = math.radians(arm_offset_deg)
        for s, sign in (("l", 1.0), ("r", -1.0)):
            Ro = axis_angle_mat(f, sign * th)
            for r in ("upperarm", "forearm", "hand"):
                j = hum[f"{s}_{r}"]
                W[:, j] = Ro @ W[:, j]
    _propagate(sk, W, mapped)

    # ---- root translation
    X = sk.X
    hips = hum["hips"]
    root = hips
    while sk.parents[root] >= 0:
        root = sk.parents[root]
    feet_t = [hum[k] for k in ("l_foot", "r_foot", "l_toe", "r_toe") if hum.get(k) is not None]
    L0 = _locals(sk, W)
    _, pos0 = fk(L0, sk.offsets, sk.parents, np.zeros((T, 3)), sk.order, sk.parent_rot)
    hips_rel = pos0[:, hips]
    src_leg = np.mean([np.linalg.norm(rest[k] - rest[h]) + np.linalg.norm(rest[a] - rest[k]) for h, k, a in ((1, 4, 7), (2, 5, 8))])
    k_scale = hum["leg_length"] / max(float(src_leg), EPS)
    pel_s = xs[:, 0]
    foot_s = xs[:, list(HY_FEET)]
    lowest_s = foot_s[:, :, 1].min(1)
    floor_s = float(np.percentile(lowest_s, 1))
    lowest_rest_t = float(X[feet_t, 1].min())
    ground_t = lowest_rest_t if floor_y is None else float(floor_y)
    sole_t = lowest_rest_t - ground_t
    P = np.zeros((T, 3))
    if in_place:
        P[:, 0], P[:, 2] = X[hips, 0], X[hips, 2]
    else:
        P[:, 0] = X[hips, 0] + k_scale * (pel_s[:, 0] - pel_s[0, 0])
        P[:, 2] = X[hips, 2] + k_scale * (pel_s[:, 2] - pel_s[0, 2])
    lowest_rel = pos0[:, feet_t, 1].min(1)
    root_y = ground_t + sole_t + k_scale * (lowest_s - floor_s) - lowest_rel          # lowest foot joint follows the source's
    P[:, 1] = root_y + hips_rel[:, 1]
    root_pos = P - hips_rel
    # ---- foot lock
    contacts = {"l": np.zeros(T, bool), "r": np.zeros(T, bool)}
    if foot_lock and T > 5:
        Wm, Pm = fk(_locals(sk, W), sk.offsets, sk.parents, root_pos, sk.order, sk.parent_rot)
        for s, toe_i, ank_i in (("l", 10, 7), ("r", 11, 8)):
            hp, kn, an = hum[f"{s}_thigh"], hum[f"{s}_shin"], hum[f"{s}_foot"]
            toe_floor = float(np.percentile(xs[:, toe_i, 1], 1))
            c = detect_contacts(xs[:, toe_i], toe_floor)
            contacts[s] = c
            if not c.any():
                continue
            a_len, b_len = float(np.linalg.norm(Pm[0, kn] - Pm[0, hp])), float(np.linalg.norm(Pm[0, an] - Pm[0, kn]))
            blend = 3
            w = np.zeros(T)
            lock_pos = np.zeros((T, 3))
            for r0, r1 in _runs(c):
                lock = Pm[r0:r1, an].mean(0)
                for i in range(r0, r1):
                    w[i] = min(1.0, (i - r0 + 1) / (blend + 1), (r1 - i) / (blend + 1))
                lock_pos[r0:r1] = lock
            idx = np.flatnonzero(w > 0)
            A_fk = Pm[idx, an]
            A_t = A_fk + w[idx, None] * (lock_pos[idx] - A_fk)
            H = Pm[idx, hp]
            K_new, A_new = two_bone_ik(H, Pm[idx, kn], A_t, a_len, b_len, np.asarray(f))
            R1 = arc_rotations(Pm[idx, kn] - H, K_new - H)                       # thigh swing
            R2 = arc_rotations(np.einsum("nij,nj->ni", R1, Pm[idx, an] - Pm[idx, kn]), A_new - K_new)   # shin re-aim after the swing
            # intermediate (twist) joints are unmapped, hence rigid with the mapped joint above them: swinging the mapped joint
            # carries them, and the foot keeps its own world rotation
            W[idx, hp] = R1 @ W[idx, hp]
            W[idx, kn] = R2 @ R1 @ W[idx, kn]
        _propagate(sk, W, mapped)

    L = _locals(sk, W)
    Wf, Pf = fk(L, sk.offsets, sk.parents, root_pos, sk.order, sk.parent_rot)
    q = np.zeros((T, sk.J, 4))
    for j in range(sk.J):
        q[:, j] = mat_to_quat(L[:, j])
    q = canon_quats(q)
    if fps != fps_in:
        q, tr = resample_motion(q, root_pos, fps_in, fps)
        root_pos = tr
        L = quat_to_mat(q)
        _, Pf = fk(L, sk.offsets, sk.parents, root_pos, sk.order, sk.parent_rot)
    return {"local_q": q, "root_pos": root_pos, "joint_pos": Pf, "fps": fps, "scale": k_scale, "floor_y": ground_t, "contacts": contacts,
            "align": info["angles"], "warnings": info["warnings"], "root": root, "world_rot": Wf, "source_pos": xs}


def clip_tracks(sk: Skeleton, res: dict, node_of: list[int], *, eps=1e-5):
    """retarget() result -> [(node index, 'rotation'|'translation', times float32[T], values float32[T,3|4])] for rig_anim.append_clip.
    Joints whose local rotation never leaves its rest value get no channel; the root joint carries the translation."""
    q = res["local_q"]
    T = len(q)
    times = (np.arange(T) / res["fps"]).astype(np.float32)
    rest_q = mat_to_quat(np.stack([(sk.R0[sk.parents[j]].T if sk.parents[j] >= 0 else sk.parent_rot.T) @ sk.R0[j] for j in range(sk.J)]))
    tracks = []
    for j in range(sk.J):
        dq = np.abs(np.abs(np.sum(q[:, j] * rest_q[j], -1)) - 1).max()
        if dq > eps:
            tracks.append((node_of[j], "rotation", times, q[:, j].astype(np.float32)))
    root = res["root"]
    inv = np.linalg.inv(sk.parent_mat)
    local_t = (np.concatenate([res["root_pos"], np.ones((T, 1))], 1) @ inv.T)[:, :3]
    tracks.append((node_of[root], "translation", times, local_t.astype(np.float32)))
    return tracks


# ---------------------------------------------------------------------------------------------------- GLB glue
def skeleton_from_rig(rig) -> tuple[Skeleton, list[int]]:
    """rig_anim.Rig -> (Skeleton in joint order, node index of every joint)."""
    node_of = list(rig.joints)
    idx = {n: i for i, n in enumerate(node_of)}
    parents = [idx[rig.parent[n]] if rig.parent[n] is not None else -1 for n in node_of]
    X = np.array([rig.world(n)[:3, 3] for n in node_of])
    R0 = []
    for n in node_of:
        m = rig.world(n)[:3, :3]
        m = m / np.maximum(np.linalg.norm(m, axis=0), 1e-12)[None, :]
        R0.append(m)
    names = [rig.nodes[n].get("name") or f"bone_{i}" for i, n in enumerate(node_of)]
    root = next(n for n in node_of if rig.parent[n] is None)
    up = rig.node_parent.get(root)
    pm = rig.world(up) if up is not None else np.eye(4)
    prot = pm[:3, :3] / np.maximum(np.linalg.norm(pm[:3, :3], axis=0), 1e-12)[None, :]
    return Skeleton(X, parents, np.array(R0), names, parent_rot=prot, parent_mat=pm), node_of


def retarget_glb(glb: bytes | str | Path, rot6d, transl, *, clip_name="motion", fps=HY_FPS, facing="auto", in_place=False,
                 foot_lock=True, arm_offset_deg=0.0, keep_clips=False, floor_y=None) -> tuple[bytes, dict]:
    """Rigged GLB + HY-Motion output -> (GLB bytes with the clip as the FIRST animation, report dict). Raises NotHumanoidError /
    RigAnimError (no skin) for user-fixable problems."""
    data = Path(glb).read_bytes() if isinstance(glb, (str, Path)) else bytes(glb)
    js, binary = rig_anim.read_glb(data)
    rig = rig_anim.Rig(js)
    sk, node_of = skeleton_from_rig(rig)
    hum = detect_humanoid(sk.X, sk.parents, sk.names, facing=facing)
    if floor_y is None:
        floor_y = mesh_floor_y(js, binary, rig)
    res = retarget(rot6d, transl, sk, hum, in_place=in_place, foot_lock=foot_lock, arm_offset_deg=arm_offset_deg, floor_y=floor_y,
                   fps=fps)
    tracks = clip_tracks(sk, res, node_of)
    out, info = rig_anim.append_clip(data, clip_name, tracks, first=True, keep_existing=keep_clips)
    roles = {k: (sk.names[v] if isinstance(v, (int, np.integer)) else [sk.names[i] for i in v] if isinstance(v, list) else None)
             for k, v in hum.items() if k in ("hips", "spine", "chest", "neck", "head") or k[:2] in ("l_", "r_")}
    report = {"roles": roles, "facing": hum["facing_name"], "method": hum["method"], "scale": round(res["scale"], 4),
              "frames": len(res["local_q"]), "fps": res["fps"], "duration": round(len(res["local_q"]) / res["fps"], 3),
              "channels": info["channels"], "warnings": hum["warnings"] + res["warnings"], "align_deg": res["align"],
              "contact_frames": {k: int(v.sum()) for k, v in res["contacts"].items()}, "floor_y": round(float(res["floor_y"]), 4),
              "bytes_before": len(data), "bytes_after": len(out)}
    return out, report


def mesh_floor_y(js, binary, rig) -> float | None:
    """Lowest world Y of the mesh in the rest pose (POSITION accessor min, unskinned), or None when it cannot be read."""
    try:
        lo = None
        node_parent = rig.node_parent
        for ni, n in enumerate(js.get("nodes") or []):
            if n.get("mesh") is None:
                continue
            m = np.eye(4)
            chain, p = [], ni
            while p is not None:
                chain.append(p)
                p = node_parent.get(p)
            for c in reversed(chain):
                t, q, s = rig_anim._trs(js["nodes"][c])
                loc = np.eye(4)
                loc[:3, :3] = rig_anim.q_to_mat(q) * s[None, :]
                loc[:3, 3] = t
                m = m @ loc
            for prim in js["meshes"][n["mesh"]].get("primitives", []):
                acc = js["accessors"][prim["attributes"]["POSITION"]]
                if acc.get("min") is None:
                    continue
                a, b = np.array(acc["min"]), np.array(acc["max"])
                corners = np.array([[x, y, z, 1.0] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
                y = float((corners @ m.T)[:, 1].min())
                lo = y if lo is None else min(lo, y)
        return lo
    except Exception:  # noqa: BLE001
        return None
