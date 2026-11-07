"""Animation clips for rigged GLBs: a "flex test" and a gentle "wiggle", written with numpy + the stdlib only.

Used by the rig worker (right after the skin is written) and by the CPU tool `tool:mesh.flex_preview`
(app/plugins/mesh.py). The mesh, skin and every existing byte of the binary chunk are left alone: the clips are appended
as new bufferViews / accessors / `animations`, so the output is the input plus a few KB.

  flex    Walks the joint hierarchy level by level from the root. Every joint swings once, a sine back-and-forth of
          +-`angle_deg` (default 30) about an axis perpendicular to its bone (the bend axis, bone x world +Z, or x +Y for a bone
          that points along Z, tilted 39 degrees towards the other perpendicular so the swing shows from the front and the
          side), so a bad weight shows up as a dent / stretch on the neighbouring body part. Joints of one hierarchy level move together (siblings with
          opposite sign, so left and right limbs go in opposite directions); the level slot is 1.0 s for shallow rigs
          and shrinks to 0.6 s, then several levels share a slot, so even 60+ joints stay near 18 s. The clip starts and
          ends at rest, so it loops cleanly. The root joint swings at half amplitude and nothing is ever translated.
  wiggle  All joints at once, a loop of 4 s: a few degrees (default 3) of bend about the same axis plus 60 % sideways,
          phase-shifted by depth so a wave runs down every chain.

Rotations are composed with each node's rest rotation (q_rest * q_delta, delta in the joint's own frame), normalised,
LINEAR samplers, float32 times. Nodes that carry a `matrix` get an equivalent TRS (a glTF node must not be animated
while it has a matrix).

CLI: python rig_anim.py in.glb out.glb [flex,wiggle]
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x46546C67           # "glTF"
CH_JSON, CH_BIN = 0x4E4F534A, 0x004E4942
FLOAT, SCALAR, VEC4 = 5126, "SCALAR", "VEC4"
CLIPS = ("flex", "wiggle")
FLEX_SIDE = 0.8              # weight of the sideways component in the flex axis


class RigAnimError(ValueError):
    """The GLB cannot be animated (not a GLB, no skin, ...). `kind` is 'input' for user-fixable problems."""
    kind = "input"


class NoSkinError(RigAnimError):
    pass


# ------------------------------------------------------------------------------------------------------------ GLB io
def read_glb(data: bytes) -> tuple[dict, bytes]:
    """bytes -> (gltf json, binary chunk without trailing alignment cut; b'' when there is none)."""
    if len(data) < 20:
        raise RigAnimError("Not a GLB file (too short).")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or version != 2:
        raise RigAnimError("Not a binary glTF 2.0 (.glb) file.")
    if length > len(data):
        raise RigAnimError("The GLB is truncated.")
    pos, js, binary = 12, None, b""
    while pos + 8 <= length:
        clen, ctype = struct.unpack_from("<II", data, pos)
        body = data[pos + 8:pos + 8 + clen]
        if ctype == CH_JSON and js is None:
            js = json.loads(body.decode("utf-8").rstrip(" \x00"))
        elif ctype == CH_BIN and not binary:
            binary = bytes(body)
        pos += 8 + clen + (-clen % 4)
    if js is None:
        raise RigAnimError("The GLB has no JSON chunk.")
    return js, binary


def write_glb(js: dict, binary: bytes) -> bytes:
    text = json.dumps(js, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = bytes(binary) + b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    out = bytearray(struct.pack("<III", MAGIC, 2, total))
    out += struct.pack("<II", len(text), CH_JSON) + text
    if binary:
        out += struct.pack("<II", len(binary), CH_BIN) + binary
    return bytes(out)


# ------------------------------------------------------------------------------------------------------------ math
def _norm(v, fallback=(0.0, 1.0, 0.0)):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, dtype=np.float64)


def qmul(a, b):
    """Hamilton product of quaternions in glTF (x, y, z, w) order; broadcasts over leading axes."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def qaxis(axis, angles):
    """Unit quaternions for rotations of `angles` (radians, array) about the unit vector `axis`."""
    half = np.asarray(angles, dtype=np.float64) / 2
    ax = _norm(axis)
    return np.concatenate([ax[None, :] * np.sin(half)[:, None], np.cos(half)[:, None]], 1)


def q_to_mat(q):
    x, y, z, w = _norm(q, (0, 0, 0, 1))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_q(m):
    m = np.asarray(m, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    return _norm(q, (0, 0, 0, 1))


def _trs(node: dict):
    """(translation, rotation xyzw, scale) of a node; a `matrix` is decomposed (shear ignored)."""
    if node.get("matrix") is not None:
        m = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T      # glTF is column-major
        t, lin = m[:3, 3].copy(), m[:3, :3].copy()
        s = np.linalg.norm(lin, axis=0)
        s[s < 1e-12] = 1.0
        rot = lin / s
        if np.linalg.det(rot) < 0:
            s[0] = -s[0]
            rot[:, 0] = -rot[:, 0]
        return t, mat_to_q(rot), s
    return (np.asarray(node.get("translation", (0, 0, 0)), dtype=np.float64),
            _norm(node.get("rotation", (0, 0, 0, 1)), (0, 0, 0, 1)),
            np.asarray(node.get("scale", (1, 1, 1)), dtype=np.float64))


# ------------------------------------------------------------------------------------------------------------ rig model
class Rig:
    """Joint hierarchy of a GLB with rest rotations and, per joint, the bend axes (in the joint's own frame)."""

    def __init__(self, js: dict):
        nodes = js.get("nodes") or []
        skins = [s for s in js.get("skins") or [] if s.get("joints")]
        if not skins:
            raise NoSkinError("This mesh has no skeleton or skin; rig it first (Rig tab).")
        seen, order = set(), []
        for s in skins:
            for j in s["joints"]:
                if isinstance(j, int) and 0 <= j < len(nodes) and j not in seen:
                    seen.add(j)
                    order.append(j)
        if not order:
            raise NoSkinError("The skin lists no valid joint nodes.")
        self.js, self.nodes, self.joints = js, nodes, order
        self.node_parent = {c: p for p, n in enumerate(nodes) for c in n.get("children") or []}
        # world rest transform of every node (4x4, glTF column vectors)
        self._world: dict[int, np.ndarray] = {}
        self.rest_q: dict[int, np.ndarray] = {}
        for j in order:
            self.rest_q[j] = _trs(nodes[j])[1]
        # nearest joint ancestor
        jset = set(order)
        self.parent: dict[int, int | None] = {}
        for j in order:
            p, seen_up = self.node_parent.get(j), 0
            while p is not None and p not in jset and seen_up < 10000:
                p, seen_up = self.node_parent.get(p), seen_up + 1
            self.parent[j] = p
        self.children: dict[int, list[int]] = {j: [] for j in order}
        for j in order:
            if self.parent[j] is not None:
                self.children[self.parent[j]].append(j)
        self.depth: dict[int, int] = {}
        for j in order:
            d, p = 0, self.parent[j]
            while p is not None:
                d, p = d + 1, self.parent[p]
            self.depth[j] = d
        self.sibling: dict[int, int] = {j: (self.children[self.parent[j]].index(j) if self.parent[j] is not None else 0) for j in order}
        self._axes()

    def world(self, i: int) -> np.ndarray:
        if i in self._world:
            return self._world[i]
        chain, p = [], i
        while p is not None and p not in self._world:
            chain.append(p)
            p = self.node_parent.get(p)
            if len(chain) > 100000:
                raise RigAnimError("The node hierarchy has a cycle.")
        m = self._world[p] if p is not None else np.eye(4)
        for n in reversed(chain):
            t, q, s = _trs(self.nodes[n])
            loc = np.eye(4)
            loc[:3, :3] = q_to_mat(q) * s[None, :]
            loc[:3, 3] = t
            m = m @ loc
            self._world[n] = m
        return self._world[i]

    def _axes(self):
        """Per joint: `bend` (perpendicular to the bone) and `side` (the other perpendicular) as unit vectors in the joint's local frame."""
        self.bend, self.side, self.flex = {}, {}, {}
        for j in self.joints:
            w = self.world(j)
            pos = w[:3, 3]
            if self.children[j]:
                kids = np.array([self.world(c)[:3, 3] - pos for c in self.children[j]])
                d = kids.mean(0)
                if np.linalg.norm(d) < 1e-9 * (1 + np.abs(kids).max()):
                    d = kids[np.argmax(np.linalg.norm(kids, axis=1))]
            elif self.parent[j] is not None:
                d = pos - self.world(self.parent[j])[:3, 3]
            else:
                d = np.array([0.0, 1.0, 0.0])
            d = _norm(d)
            ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            bend = _norm(np.cross(d, ref))
            side = _norm(np.cross(d, bend))
            rw = w[:3, :3] / np.maximum(np.linalg.norm(w[:3, :3], axis=0), 1e-12)[None, :]
            self.bend[j], self.side[j] = _norm(rw.T @ bend), _norm(rw.T @ side)
            # the flex axis is the bend axis tilted towards the side axis: the swing is then visible from the front and from the side
            self.flex[j] = _norm(self.bend[j] + FLEX_SIDE * self.side[j])


# ------------------------------------------------------------------------------------------------------------ clips
def _canon(q):
    """Keep consecutive quaternions in one hemisphere so LINEAR (slerp) never takes the long way round."""
    q = q.copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0:
            q[i] = -q[i]
    return q


def flex_plan(rig: Rig, slot_s: float | None = None, max_s: float = 18.0) -> dict:
    """Timing of the flex clip: level -> slot index, slot length, total duration. Pure, for tests."""
    levels = sorted({rig.depth[j] for j in rig.joints})
    n_levels = len(levels)
    slot = slot_s or min(1.0, max(0.6, 14.0 / n_levels))
    per_slot = max(1, math.ceil(n_levels * slot / max_s))
    lead = tail = 0.25
    n_slots = math.ceil(n_levels / per_slot)
    return {"levels": n_levels, "slot": slot, "levels_per_slot": per_slot, "slots": n_slots,
            "lead": lead, "duration": lead + n_slots * slot + tail, "level_slot": {lv: i // per_slot for i, lv in enumerate(levels)}}


def build_flex(rig: Rig, angle_deg: float = 30.0, keys_per_slot: int = 12):
    """-> (clip dict {name, duration, tracks: [(joint node, times float32[K], quats float32[K,4])]}). Rest at both ends."""
    plan = flex_plan(rig)
    slot, lead, T = plan["slot"], plan["lead"], plan["duration"]
    tracks = []
    u = np.linspace(0.0, 1.0, keys_per_slot + 1)
    for j in rig.joints:
        k = plan["level_slot"][rig.depth[j]]
        amp = math.radians(angle_deg) / math.sqrt(plan["levels_per_slot"])
        if rig.parent[j] is None:
            amp *= 0.5
        sign = -1.0 if rig.sibling[j] % 2 else 1.0
        t0 = lead + k * slot
        times = np.concatenate([[0.0], t0 + u * slot, [T]])
        ang = np.concatenate([[0.0], sign * amp * np.sin(2 * math.pi * u), [0.0]])
        quats = _canon(qmul(rig.rest_q[j][None, :], qaxis(rig.flex[j], ang)))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        tracks.append((j, times.astype(np.float32), quats.astype(np.float32)))
    return {"name": "flex", "duration": float(T), "tracks": tracks, "plan": plan}


def build_wiggle(rig: Rig, angle_deg: float = 3.0, period_s: float = 4.0, fps: int = 8):
    """A seamless loop: the first and the last key are equal."""
    n = int(round(period_s * fps))
    times = np.arange(n + 1, dtype=np.float64) / fps
    ph = 2 * math.pi * times / period_s
    amp = math.radians(angle_deg)
    tracks = []
    for j in rig.joints:
        lag = -0.45 * rig.depth[j] + 0.8 * (rig.sibling[j] % 2)
        a = amp * np.sin(ph + lag)
        b = 0.6 * amp * np.cos(ph + lag)
        dq = qmul(qaxis(rig.bend[j], a), qaxis(rig.side[j], b))
        quats = _canon(qmul(rig.rest_q[j][None, :], dq))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        quats[-1] = quats[0]
        tracks.append((j, times.astype(np.float32), quats.astype(np.float32)))
    return {"name": "wiggle", "duration": float(period_s), "tracks": tracks}


# ------------------------------------------------------------------------------------------------------------ glTF writing
class _Blob:
    def __init__(self, js: dict, binary: bytes):
        self.js, self.buf = js, bytearray(binary)
        if js.get("buffers"):
            if js["buffers"][0].get("uri") is not None:
                raise RigAnimError("The GLB's buffer is external (uri); only self-contained GLBs are supported.")
        else:
            js["buffers"] = [{"byteLength": 0}]
        self.buf.extend(b"\x00" * (-len(self.buf) % 4))           # existing bytes stay where they are

    def accessor(self, arr: np.ndarray, atype: str, minmax: bool) -> int:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        self.buf.extend(b"\x00" * (-len(self.buf) % 4))
        off = len(self.buf)
        self.buf.extend(arr.tobytes())
        views = self.js.setdefault("bufferViews", [])
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        acc = {"bufferView": len(views) - 1, "componentType": FLOAT, "count": int(arr.shape[0]), "type": atype}
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"], acc["max"] = [float(x) for x in flat.min(0)], [float(x) for x in flat.max(0)]
        accs = self.js.setdefault("accessors", [])
        accs.append(acc)
        return len(accs) - 1

    def finish(self) -> bytes:
        self.buf.extend(b"\x00" * (-len(self.buf) % 4))
        self.js["buffers"][0]["byteLength"] = len(self.buf)
        return bytes(self.buf)


def _matrix_to_trs(js: dict, joints):
    """Animated nodes must not have a `matrix`: replace it by the identical translation / rotation / scale."""
    for j in joints:
        n = js["nodes"][j]
        if n.get("matrix") is not None:
            t, q, s = _trs(n)
            del n["matrix"]
            n["translation"], n["rotation"], n["scale"] = [float(x) for x in t], [float(x) for x in q], [float(x) for x in s]


def add_animations(src: bytes | str | Path, clips=CLIPS, *, angle_deg: float = 30.0, wiggle_deg: float = 3.0) -> tuple[bytes, dict]:
    """GLB (bytes or path) -> (new GLB bytes with the requested clips, info). Raises NoSkinError / RigAnimError.

    Clips whose name already exists in the file are skipped; if nothing is left to add a RigAnimError is raised."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    js, binary = read_glb(data)
    rig = Rig(js)
    existing = {a.get("name") for a in js.get("animations") or []}
    want = [c for c in clips if c in CLIPS]
    unknown = [c for c in clips if c not in CLIPS]
    if unknown:
        raise RigAnimError(f"Unknown clip(s): {', '.join(unknown)} (known: {', '.join(CLIPS)}).")
    want = [c for c in want if c not in existing]
    if not want:
        raise RigAnimError("This mesh already has these animation clips.")
    _matrix_to_trs(js, rig.joints)
    blob = _Blob(js, binary)
    built = []
    for name in want:
        built.append(build_flex(rig, angle_deg) if name == "flex" else build_wiggle(rig, wiggle_deg))
    anims = js.setdefault("animations", [])
    info = {"joints": len(rig.joints), "clips": []}
    for clip in built:
        samplers, channels, time_acc = [], [], {}
        for j, times, quats in clip["tracks"]:
            key = times.tobytes()
            if key not in time_acc:                          # joints that share a time axis share its accessor
                time_acc[key] = blob.accessor(times, SCALAR, True)
            samplers.append({"input": time_acc[key], "interpolation": "LINEAR", "output": blob.accessor(quats, VEC4, False)})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "rotation"}})
        anims.append({"name": clip["name"], "samplers": samplers, "channels": channels})
        info["clips"].append({"name": clip["name"], "duration": round(clip["duration"], 3), "channels": len(channels)})
    out = write_glb(js, blob.finish())
    info["bytes_before"], info["bytes_after"] = len(data), len(out)
    return out, info


def append_clip(src: bytes | str | Path, name: str, tracks, *, first: bool = True, keep_existing: bool = False) -> tuple[bytes, dict]:
    """Add ONE arbitrary clip to a skinned GLB -> (new GLB bytes, info). `tracks` = [(node index, 'rotation' | 'translation',
    times float[K], values float[K, 4 | 3])]; times must be strictly increasing, rotations unit quaternions (xyzw), values are the
    node's LOCAL rotation / translation (absolute, replacing the rest value). Tracks with the same times share one accessor.
    The clip becomes `animations[0]` (viewers that autoplay the first clip play it) unless `first=False`; the clips already in
    the file are dropped unless `keep_existing`. Meshes, skin and every existing byte stay (dropped clips leave a few KB unused)."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    js, binary = read_glb(data)
    rig = Rig(js)                                  # raises NoSkinError
    nodes = js["nodes"]
    prepared = []
    for node, path, times, values in tracks:
        if not isinstance(node, (int, np.integer)) or not 0 <= node < len(nodes):
            raise RigAnimError(f"Track target node {node} does not exist.")
        if path not in ("rotation", "translation"):
            raise RigAnimError(f"Unsupported track path '{path}' (rotation or translation).")
        times = np.asarray(times, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        width = 4 if path == "rotation" else 3
        if times.ndim != 1 or values.shape != (len(times), width) or len(times) < 1:
            raise RigAnimError(f"Track for node {node}: expected {width} values per key and one time per key.")
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise RigAnimError(f"Track for node {node}: key times must be strictly increasing.")
        if not np.all(np.isfinite(values)):
            raise RigAnimError(f"Track for node {node} contains NaN / inf.")
        if path == "rotation":
            values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        prepared.append((int(node), path, times, values))
    if not prepared:
        raise RigAnimError("The clip has no tracks.")
    _matrix_to_trs(js, {t[0] for t in prepared})
    blob = _Blob(js, binary)
    samplers, channels, time_acc = [], [], {}
    for node, path, times, values in prepared:
        key = times.tobytes()
        if key not in time_acc:
            time_acc[key] = blob.accessor(times, SCALAR, True)
        samplers.append({"input": time_acc[key], "interpolation": "LINEAR",
                         "output": blob.accessor(values, VEC4 if path == "rotation" else "VEC3", False)})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
    anims = [] if not keep_existing else list(js.get("animations") or [])
    clip = {"name": name, "samplers": samplers, "channels": channels}
    anims.insert(0, clip) if first else anims.append(clip)
    js["animations"] = anims
    out = write_glb(js, blob.finish())
    duration = float(max(t[2][-1] for t in prepared))
    return out, {"joints": len(rig.joints), "name": name, "channels": len(channels), "duration": round(duration, 3),
                 "bytes_before": len(data), "bytes_after": len(out)}


def has_skin(src: bytes | str | Path) -> bool:
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    js, _ = read_glb(data)
    return any(s.get("joints") for s in js.get("skins") or [])


def animation_names(src: bytes | str | Path) -> list[str]:
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    return [a.get("name") or f"animation {i}" for i, a in enumerate(read_glb(data)[0].get("animations") or [])]


def animate_file(src, dst, clips=CLIPS, **kw) -> dict:
    out, info = add_animations(src, clips, **kw)
    Path(dst).write_bytes(out)
    return info


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    print(json.dumps(animate_file(sys.argv[1], sys.argv[2], tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else CLIPS), indent=1))
