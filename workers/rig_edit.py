"""Edit an existing rig (numpy + stdlib only; unit-tested in tests/test_rig_edit.py).

Used by the CPU tool `tool:rig.edit` and `GET /api/x/rig/<asset>/skeleton` (app/plugins/rig.py) and by the rig worker's
`reskin` task (which reads the edited skeleton back with `skeleton_arrays`).

A skinned GLB goes in, a list of deterministic operations is applied to its skeleton, a NEW GLB comes out. The mesh,
materials, textures and every existing byte of the binary chunk stay (new data is appended). Rest pose is preserved: after
every edit the inverse bind matrices are recomputed so that `joint world * inverse bind` is unchanged for every existing
joint, i.e. the skinned mesh looks exactly as before; only the pivots / names / hierarchy change.

Units and axes are the studio's: glTF world units (metres), +Y up, the character faces +Z, its left side is +X.

Operations (JSON objects, `op` + fields; unknown fields are refused so typos do not pass silently):

  rename          {bone, to}
  rename_standard {scheme: mixamo | vrm | unreal}        humanoids only (roles from motion_common.detect_humanoid)
  move_joint      {bone, offset:[x,y,z] | delta:[x,y,z] | to:[x,y,z], children: stay | follow, mirror: bool}
                  offset = world units; delta = fractions of the bone length (distance to the parent joint); to = absolute
                  position. children `stay` (default) keeps the child joints where they are, `follow` moves the whole
                  subtree with the joint. mirror=true applies the mirrored move to the opposite-side joint (x about the root).
  delete_bone     {bone | match: "glob*", merge_weights_to: bone, include_children: bool}
                  the skin weights of a deleted bone go to its parent (or `merge_weights_to`), so the mesh stays bound.
                  Children are re-attached to the parent unless include_children (the whole subtree is deleted).
  add_bone        {parent, to:[x,y,z] | direction + length, name}
  add_chain       {parent, count, direction, length, total_length, name_prefix}      tails, ears, hair, antennas
  reparent        {bone, parent}                                  the joint keeps its world position

A bone is named by node name ("bone_8", "mixamorig:LeftArm"), by the index it has in the INPUT skeleton (int, as listed
by `describe`), or by a role word on a humanoid ("left forearm", "left elbow", "head", "hips", "spine 2"). Anything
ambiguous or unknown raises RigEditError listing the candidates. New bones have zero skin weight until a re-skin.
Existing animation clips are dropped (their targets change); `flex` / `wiggle` are rebuilt when the input had them.
"""
from __future__ import annotations

import fnmatch
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import rig_anim
from rig_anim import NoSkinError, RigAnimError, read_glb, write_glb  # noqa: F401  (re-exported for callers)

SCHEMES = ("mixamo", "vrm", "unreal")
MAX_OPS = 200
MAX_NAME = 64
DIRECTIONS = {"up": (0, 1, 0), "down": (0, -1, 0), "left": (1, 0, 0), "right": (-1, 0, 0),
              "forward": (0, 0, 1), "front": (0, 0, 1), "back": (0, 0, -1), "backward": (0, 0, -1), "behind": (0, 0, -1)}
OP_FIELDS = {
    "rename": ({"bone", "to"}, {"bone", "to"}),
    "rename_standard": ({"scheme"}, {"scheme"}),
    "move_joint": ({"bone", "offset", "delta", "to", "children", "mirror"}, {"bone"}),
    "delete_bone": ({"bone", "match", "merge_weights_to", "include_children"}, set()),
    "add_bone": ({"parent", "to", "direction", "length", "name"}, {"parent"}),
    "add_chain": ({"parent", "count", "direction", "length", "total_length", "name_prefix"}, {"parent", "count"}),
    "reparent": ({"bone", "parent"}, {"bone", "parent"}),
}
COMP = {5120: ("i1", 1), 5121: ("u1", 1), 5122: ("i2", 2), 5123: ("u2", 2), 5125: ("u4", 4), 5126: ("f4", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class RigEditError(RigAnimError):
    """A user-fixable problem with the edit (unknown / ambiguous bone, bad op, not a humanoid ...). `kind` is 'input'."""
    kind = "input"


# ------------------------------------------------------------------------------------------------------------ names
def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _squash_name(s: str) -> str:
    return re.sub(r"^mixamorig\d*", "", _squash(s))


def _vec3(v, what: str) -> np.ndarray:
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise RigEditError(f"{what} must be three numbers [x, y, z].") from None
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise RigEditError(f"{what} must be three finite numbers [x, y, z].")
    return a


def _direction(v, what="direction") -> np.ndarray:
    if isinstance(v, str):
        key = v.strip().lower().replace("-", " ").replace("_", " ")
        if key in DIRECTIONS:
            return np.asarray(DIRECTIONS[key], dtype=np.float64)
        raise RigEditError(f"Unknown {what} '{v}'. Use [x, y, z] or one of: {', '.join(sorted(DIRECTIONS))} "
                           "(+Y up, +Z in front of the character, +X its left).")
    a = _vec3(v, what)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise RigEditError(f"{what} must not be the zero vector.")
    return a / n


# part word -> role part (see ROLE_PARTS); sided parts need left / right
PART_WORDS = {
    "forearm": "forearm", "lowerarm": "forearm", "elbow": "forearm", "upperarm": "upperarm", "arm": "upperarm", "uparm": "upperarm",
    "shoulder": "shoulder", "clavicle": "collar", "collar": "collar", "hand": "hand", "wrist": "hand",
    "thigh": "thigh", "upperleg": "thigh", "upleg": "thigh", "hip": "thigh", "shin": "shin", "calf": "shin", "knee": "shin",
    "lowerleg": "shin", "leg": "shin", "foot": "foot", "ankle": "foot", "toe": "toe", "toes": "toe", "toebase": "toe", "ball": "toe",
}
CENTER_WORDS = {"hips": "hips", "pelvis": "hips", "chest": "chest", "upperchest": "chest", "neck": "neck", "head": "head", "spine": "spine"}
SIDED = ("collar", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe")


# ------------------------------------------------------------------------------------------------------------ accessors
def _acc_view(js, buf, idx, ncomp_override=None):
    """Writable numpy view (count, ncomp) onto an accessor of `buf` (a bytearray). The caller must `del` it before `buf` grows."""
    a = js["accessors"][idx]
    if a.get("sparse"):
        raise RigEditError("Sparse accessors on the skin data are not supported.")
    if "bufferView" not in a:
        raise RigEditError("An accessor of the skin has no data.")
    dt, size = COMP[a["componentType"]]
    n = NCOMP[a["type"]]
    v = js["bufferViews"][a["bufferView"]]
    if v.get("buffer", 0) != 0:
        raise RigEditError("Only single-buffer GLBs are supported.")
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride") or size * n
    return np.ndarray((a["count"], n), dtype="<" + dt, buffer=buf, offset=off, strides=(stride, size))


def _read_acc(js, buf, idx) -> np.ndarray:
    v = _acc_view(js, buf, idx)
    out = np.array(v)
    del v
    return out


def _is_norm_int(a) -> bool:
    return bool(a.get("normalized")) and a["componentType"] in (5121, 5123)


# ------------------------------------------------------------------------------------------------------------ the document
class RigDoc:
    """A skinned GLB loaded for editing. Node indices are glTF node indices; `order` is the skin's joint list."""

    def __init__(self, data: bytes):
        self.data = bytes(data)
        js, binary = read_glb(self.data)
        self.js = js
        self.buf = bytearray(binary)
        nodes = js.get("nodes") or []
        skins = [(i, s) for i, s in enumerate(js.get("skins") or []) if s.get("joints")]
        if not skins:
            raise rig_anim.NoSkinError("This mesh has no skeleton or skin; rig it first (Rig tab).")
        if len(skins) > 1 and any(s["joints"] != skins[0][1]["joints"] for _, s in skins[1:]):
            raise RigEditError("This GLB has several different skins; only files with one skeleton can be edited.")
        self.skin_ids = [i for i, _ in skins]
        self.skin = skins[0][1]
        if js.get("buffers") and js["buffers"][0].get("uri") is not None:
            raise RigAnimError("The GLB's buffer is external (uri); only self-contained GLBs are supported.")
        joints = []
        for j in self.skin["joints"]:
            if not isinstance(j, int) or not 0 <= j < len(nodes):
                raise RigEditError("The skin lists a joint node that does not exist.")
            if j in joints:
                raise RigEditError("The skin lists the same joint twice.")
            joints.append(j)
        self.joints0 = list(joints)
        self.order = list(joints)                       # current joints (original order, new ones appended)
        self.nparent = {c: p for p, n in enumerate(nodes) for c in n.get("children") or []}
        self.W0 = self._world_all()
        self.G = {j: self.W0[j].copy() for j in joints}
        self.G0 = {j: self.W0[j].copy() for j in joints}
        self.names = {j: (nodes[j].get("name") or f"joint_{i}") for i, j in enumerate(joints)}
        self.index0 = {j: i for i, j in enumerate(joints)}
        self.ibm0 = self._read_ibm()
        self.merge: dict[int, int] = {}                 # deleted joint -> joint that takes its weights
        self.removed: set[int] = set()                  # joints no longer in the skin
        self.added: list[int] = []
        self.dirty: set[int] = set()
        self.log: list[dict] = []
        self.flags = {"moved": False, "reparented": False, "deleted": False, "added": False, "renamed": False}
        self._hum = None
        self._hum_err = None
        self._hum_done = False

    # ---------------------------------------------------------------- structure
    def _world_all(self) -> dict:
        nodes = self.js.get("nodes") or []
        memo: dict[int, np.ndarray] = {}

        def world(n):
            chain, p = [], n
            while p is not None and p not in memo:
                chain.append(p)
                p = self.nparent.get(p)
                if len(chain) > 100000:
                    raise RigAnimError("The node hierarchy has a cycle.")
            m = memo[p] if p is not None else np.eye(4)
            for c in reversed(chain):
                t, q, s = rig_anim._trs(nodes[c])
                loc = np.eye(4)
                loc[:3, :3] = rig_anim.q_to_mat(q) * s[None, :]
                loc[:3, 3] = t
                m = m @ loc
                memo[c] = m
            return memo[n]
        for i in range(len(nodes)):
            world(i)
        return memo

    def _read_ibm(self) -> np.ndarray:
        J = len(self.joints0)
        acc = self.skin.get("inverseBindMatrices")
        if acc is None:
            return np.tile(np.eye(4), (J, 1, 1))
        flat = _read_acc(self.js, self.buf, acc).astype(np.float64)
        if flat.shape != (J, 16):
            raise RigEditError("The inverse bind matrices do not match the joint list.")
        return flat.reshape(J, 4, 4).transpose(0, 2, 1).copy()      # glTF is column-major

    def world(self, n: int) -> np.ndarray:
        return self.G[n] if n in self.G else self.W0[n]

    def set_world(self, n: int, m: np.ndarray):
        if n in self.G:
            self.G[n] = m
        else:
            self.W0[n] = m

    def pos(self, n: int) -> np.ndarray:
        return self.world(n)[:3, 3].copy()

    def live(self) -> list[int]:
        return [j for j in self.order if j not in self.removed]

    def jparent(self, n: int):
        """Nearest ancestor that is a live joint, or None."""
        live = set(self.live())
        p = self.nparent.get(n)
        while p is not None and p not in live:
            p = self.nparent.get(p)
        return p

    def jchildren(self, n: int) -> list[int]:
        return [j for j in self.live() if self.jparent(j) == n]

    def subtree(self, n: int) -> list[int]:
        out, st = [], [n]
        while st:
            x = st.pop()
            out.append(x)
            st.extend(self.jchildren(x))
        return out

    def node_descendants(self, n: int) -> list[int]:
        nodes = self.js["nodes"]
        out, st = [], list(nodes[n].get("children") or [])
        while st:
            x = st.pop()
            out.append(x)
            st.extend(nodes[x].get("children") or [])
        return out

    def length(self, n: int) -> float:
        p = self.jparent(n)
        if p is not None:
            d = float(np.linalg.norm(self.pos(n) - self.pos(p)))
            if d > 1e-9:
                return d
        kids = self.jchildren(n)
        if kids:
            d = float(np.mean([np.linalg.norm(self.pos(k) - self.pos(n)) for k in kids]))
            if d > 1e-9:
                return d
        return max(0.05, 0.1 * self.height())

    def height(self) -> float:
        X = np.array([self.pos(j) for j in self.live()])
        return float(X[:, 1].max() - X[:, 1].min()) if len(X) else 1.0

    # ---------------------------------------------------------------- humanoid roles
    def humanoid(self):
        """-> (roles {label: original joint node}, info dict) or (None, error message). Cached; computed on the INPUT skeleton."""
        if self._hum_done:
            return self._hum, self._hum_err
        self._hum_done = True
        try:
            import motion_common as mc
            X = np.array([self.G0[j][:3, 3] for j in self.joints0])
            pidx = []
            for j in self.joints0:
                p = self.nparent.get(j)
                while p is not None and p not in self.index0:
                    p = self.nparent.get(p)
                pidx.append(self.index0[p] if p is not None else None)
            names = [self.names[j] for j in self.joints0]
            h = mc.detect_humanoid(X, pidx, names)
        except RigAnimError as e:
            self._hum_err = str(e)
            return None, self._hum_err
        roles: dict[str, int] = {}
        J = self.joints0

        def put(label, i):
            if i is not None:
                roles[label] = J[int(i)]

        def many(label, items):
            items = list(items)
            for k, i in enumerate(items):
                put(label if len(items) == 1 else f"{label}_{k + 1}", i)
        put("hips", h["hips"])
        many("spine", h["spine"])
        put("chest", h["chest"])
        many("neck", h["neck"])
        put("head", h["head"])
        for s, side in (("l", "left"), ("r", "right")):
            for part in SIDED:
                put(f"{side}_{part}", h.get(f"{s}_{part}"))
        self._hum = {"roles": roles, "facing": h.get("facing_name"), "method": h.get("method"), "warnings": list(h.get("warnings") or [])}
        return self._hum, None

    def role_of(self, n: int) -> str | None:
        hum, _ = self.humanoid()
        if not hum:
            return None
        for label, j in hum["roles"].items():
            if j == n:
                return label
        return None

    # ---------------------------------------------------------------- bone references
    def _label(self, n: int) -> str:
        role = self.role_of(n) if n in self.index0 else None
        idx = f"#{self.index0[n]}" if n in self.index0 else "new"
        return f"{self.names[n]} ({idx}{', ' + role if role else ''})"

    def candidates_text(self, nodes, limit=12) -> str:
        t = ", ".join(self._label(n) for n in nodes[:limit])
        return t + (f", ... ({len(nodes)} in all)" if len(nodes) > limit else "")

    def resolve(self, ref, what="bone", allow_removed=False) -> int:
        """A bone reference (node name, input index, role word) -> node index of a live joint. Raises RigEditError."""
        live = self.live()
        if isinstance(ref, bool) or ref is None or (isinstance(ref, str) and not ref.strip()):
            raise RigEditError(f"{what}: give a bone name, index or role word (for example \"left forearm\").")
        if isinstance(ref, (int, np.integer)) or (isinstance(ref, float) and float(ref).is_integer()):
            i = int(ref)
            if not 0 <= i < len(self.joints0):
                raise RigEditError(f"{what}: joint index {i} does not exist (the input rig has joints 0..{len(self.joints0) - 1}).")
            n = self.joints0[i]
            if n in self.removed:
                raise RigEditError(f"{what}: joint #{i} ({self.names[n]}) was deleted by an earlier operation.")
            return n
        if not isinstance(ref, str):
            raise RigEditError(f"{what}: a bone is a name, an index or a role word, not {type(ref).__name__}.")
        s = ref.strip()
        exact = [n for n in live if self.names[n] == s]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise RigEditError(f"{what} '{s}' is ambiguous: {self.candidates_text(exact)}. Use the index.")
        ci = [n for n in live if self.names[n].lower() == s.lower()]
        if len(ci) == 1:
            return ci[0]
        if len(ci) > 1:
            raise RigEditError(f"{what} '{s}' is ambiguous: {self.candidates_text(ci)}. Use the exact name or the index.")
        sq = _squash_name(s)
        sqm = [n for n in live if sq and _squash_name(self.names[n]) == sq]
        if len(sqm) == 1:
            return sqm[0]
        if len(sqm) > 1:
            raise RigEditError(f"{what} '{s}' is ambiguous: {self.candidates_text(sqm)}. Use the exact name or the index.")
        if re.fullmatch(r"#?\d+", s):
            return self.resolve(int(s.lstrip("#")), what)
        role = self._resolve_role(s, what)
        if role is not None:
            return role
        sub = [n for n in live if sq and sq in _squash_name(self.names[n])]
        if len(sub) == 1:
            return sub[0]
        if len(sub) > 1:
            raise RigEditError(f"{what} '{s}' is ambiguous: {self.candidates_text(sub)}. Use the exact name or the index.")
        hum, err = self.humanoid()
        extra = ""
        if hum is None and re.search(r"left|right|arm|leg|hand|foot|head|elbow|knee|hip|spine|neck", s, re.I):
            extra = f" Role words need a recognised humanoid skeleton ({err})"
        raise RigEditError(f"No bone matches {what} '{s}'.{extra} Bones: {self.candidates_text(live, 40)}.")

    def _resolve_role(self, s: str, what: str):
        toks = [t for t in re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()) if t]
        if not toks:
            return None
        side = None
        rest = []
        for t in toks:
            if t in ("left", "l"):
                side = "left"
            elif t in ("right", "r"):
                side = "right"
            else:
                rest.append(t)
        word = "".join(rest)
        if word == "root":
            roots = [n for n in self.live() if self.jparent(n) is None]
            if len(roots) == 1:
                return roots[0]
            raise RigEditError(f"{what} 'root' is ambiguous: {self.candidates_text(roots)}.")
        m = re.fullmatch(r"([a-z]+?)(\d+)?", word)
        base, num = (m.group(1), m.group(2)) if m else (word, None)
        part = CENTER_WORDS.get(base) or PART_WORDS.get(base)
        if part is None:
            return None
        hum, err = self.humanoid()
        if hum is None:
            raise RigEditError(f"{what} '{s}' is a role word, but this skeleton is not recognised as a humanoid ({err}) "
                               "Use the bone name or index (get_rig lists them).")
        roles = hum["roles"]

        def pick(prefix):
            if num is not None:
                key = f"{prefix}_{int(num)}"
                if key in roles:
                    return roles[key]
            if prefix in roles:
                return roles[prefix]
            many = sorted(k for k in roles if re.fullmatch(rf"{prefix}_\d+", k))
            if many:
                raise RigEditError(f"{what} '{s}' is ambiguous: this rig has {len(many)} {prefix} joints "
                                   f"({', '.join(f'{k} = {self.names[roles[k]]}' for k in many)}). Say e.g. \"{many[0]}\" or use the name.")
            return None
        if base == "hip" and side is None:
            part = "hips"
        if part in SIDED or part == "shoulder":
            if side is None:
                want = "collar" if part == "shoulder" else part
                found = [roles[k] for k in roles if k.split("_", 1)[-1] == want]
                raise RigEditError(f"{what} '{s}' needs a side: say \"left {base}\" or \"right {base}\" (candidates: {self.candidates_text(found)}).")
            if part == "shoulder":
                part = "collar" if f"{side}_collar" in roles else "upperarm"
            n = roles.get(f"{side}_{part}")
            if n is None:
                raise RigEditError(f"{what} '{s}': this humanoid has no {side} {part} joint.")
            return n
        if side is not None:
            return None
        n = pick(part)
        if n is None:
            raise RigEditError(f"{what} '{s}': this humanoid has no {part} joint.")
        return n

    # ---------------------------------------------------------------- tree surgery
    def _scene_lists(self):
        return [s.setdefault("nodes", []) for s in self.js.get("scenes") or []]

    def _detach(self, n: int):
        p = self.nparent.get(n)
        if p is not None:
            ch = self.js["nodes"][p].get("children") or []
            if n in ch:
                ch.remove(n)
            if not ch:
                self.js["nodes"][p].pop("children", None)
        else:
            for lst in self._scene_lists():
                while n in lst:
                    lst.remove(n)
        self.nparent.pop(n, None)

    def _attach(self, n: int, parent, scenes_from=None):
        if parent is None:
            self.nparent.pop(n, None)
            lists = self._scene_lists()
            if not lists:
                self.js["scenes"] = [{"nodes": []}]
                self.js.setdefault("scene", 0)
                lists = self._scene_lists()
            if scenes_from is None:
                scenes_from = range(len(lists))
            for k in scenes_from:
                if n not in lists[k]:
                    lists[k].append(n)
        else:
            self.js["nodes"][parent].setdefault("children", []).append(n)
            self.nparent[n] = parent
        self.dirty.add(n)

    def _scenes_of(self, n):
        return [k for k, lst in enumerate(self._scene_lists()) if n in lst]

    def _shift_subtree(self, n: int, d: np.ndarray, include_self=True):
        for x in ([n] if include_self else []) + self.node_descendants(n):
            m = self.world(x).copy()
            m[:3, 3] += d
            self.set_world(x, m)

    # ---------------------------------------------------------------- operations
    def apply(self, op: dict, i: int):
        if not isinstance(op, dict) or not isinstance(op.get("op"), str):
            raise RigEditError(f"Operation {i + 1} must be an object with an \"op\" field (one of: {', '.join(OP_FIELDS)}).")
        name = op["op"].strip().lower()
        if name not in OP_FIELDS:
            raise RigEditError(f"Operation {i + 1}: unknown op '{op['op']}'. Known ops: {', '.join(OP_FIELDS)}.")
        allowed, required = OP_FIELDS[name]
        extra = sorted(set(op) - allowed - {"op"})
        if extra:
            raise RigEditError(f"Operation {i + 1} ({name}): unknown field(s) {', '.join(extra)}. Fields: {', '.join(sorted(allowed))}.")
        missing = sorted(k for k in required if op.get(k) in (None, ""))
        if missing:
            raise RigEditError(f"Operation {i + 1} ({name}): missing {', '.join(missing)}.")
        try:
            summary = getattr(self, f"op_{name}")(op)
        except RigEditError as e:
            raise RigEditError(f"Operation {i + 1} ({name}): {e}") from None
        self.log.append({"op": name, "summary": summary})

    def _check_name(self, new: str, ignore: int | None = None) -> str:
        if not isinstance(new, str) or not new.strip():
            raise RigEditError("The new name must be a non-empty string.")
        new = new.strip()
        if len(new) > MAX_NAME:
            raise RigEditError(f"Bone names are at most {MAX_NAME} characters.")
        if any(self.names[n] == new for n in self.live() if n != ignore):
            raise RigEditError(f"There already is a bone named '{new}'.")
        return new

    def _unique(self, base: str) -> str:
        names = {self.names[n] for n in self.live()}
        if base not in names:
            return base
        k = 2
        while f"{base}_{k}" in names:
            k += 1
        return f"{base}_{k}"

    def _rename(self, n: int, new: str):
        self.names[n] = new
        self.js["nodes"][n]["name"] = new
        self.flags["renamed"] = True

    def op_rename(self, op):
        n = self.resolve(op["bone"])
        old = self.names[n]
        new = self._check_name(op["to"], ignore=n)
        self._rename(n, new)
        return f"renamed {old} -> {new}"

    def op_rename_standard(self, op):
        scheme = str(op["scheme"]).strip().lower()
        if scheme not in SCHEMES:
            raise RigEditError(f"Unknown naming scheme '{op['scheme']}'. Use one of: {', '.join(SCHEMES)}.")
        hum, err = self.humanoid()
        if hum is None:
            raise RigEditError(f"Standard names need a humanoid skeleton (two legs, two arms, upright), and this one was not recognised: {err}")
        roles = hum["roles"]
        mapping: dict[int, str] = {}
        spine = [roles[k] for k in sorted(k for k in roles if k == "spine" or re.fullmatch(r"spine_\d+", k))]
        spine.sort(key=lambda n: self.index0[n])
        chain = ([] if "chest" not in roles else [roles["chest"]])
        torso_chain = spine + chain
        neck = sorted((roles[k] for k in roles if k == "neck" or re.fullmatch(r"neck_\d+", k)), key=lambda n: self.index0[n])
        if scheme == "mixamo":
            P = "mixamorig:"
            mapping[roles["hips"]] = P + "Hips"
            for k, n in enumerate(torso_chain):
                mapping[n] = P + ("Spine" if k == 0 else f"Spine{k}")
            for k, n in enumerate(neck):
                mapping[n] = P + ("Neck" if k == 0 else f"Neck{k}")
            mapping[roles["head"]] = P + "Head"
            part = {"collar": "Shoulder", "upperarm": "Arm", "forearm": "ForeArm", "hand": "Hand", "thigh": "UpLeg", "shin": "Leg", "foot": "Foot", "toe": "ToeBase"}
            for side, S in (("left", "Left"), ("right", "Right")):
                for k, v in part.items():
                    if f"{side}_{k}" in roles:
                        mapping[roles[f"{side}_{k}"]] = f"{P}{S}{v}"
        elif scheme == "vrm":
            mapping[roles["hips"]] = "hips"
            m = len(torso_chain)
            for k, n in enumerate(torso_chain):
                left = m - k          # 1 = last
                mapping[n] = ("upperChest" if m >= 3 and left == 1 else "chest" if m >= 2 and left == (2 if m >= 3 else 1)
                              else "spine" if k == 0 else f"spine_{k + 1}")
            for k, n in enumerate(neck):
                mapping[n] = "neck" if k == 0 else f"neck_{k + 1}"
            mapping[roles["head"]] = "head"
            part = {"collar": "Shoulder", "upperarm": "UpperArm", "forearm": "LowerArm", "hand": "Hand", "thigh": "UpperLeg", "shin": "LowerLeg", "foot": "Foot", "toe": "Toes"}
            for side in ("left", "right"):
                for k, v in part.items():
                    if f"{side}_{k}" in roles:
                        mapping[roles[f"{side}_{k}"]] = side + v
        else:
            mapping[roles["hips"]] = "pelvis"
            for k, n in enumerate(torso_chain):
                mapping[n] = f"spine_{k + 1:02d}"
            for k, n in enumerate(neck):
                mapping[n] = f"neck_{k + 1:02d}"
            mapping[roles["head"]] = "head"
            part = {"collar": "clavicle", "upperarm": "upperarm", "forearm": "lowerarm", "hand": "hand", "thigh": "thigh", "shin": "calf", "foot": "foot", "toe": "ball"}
            for side, sfx in (("left", "l"), ("right", "r")):
                for k, v in part.items():
                    if f"{side}_{k}" in roles:
                        mapping[roles[f"{side}_{k}"]] = f"{v}_{sfx}"
        mapping = {n: v for n, v in mapping.items() if n in self.live()}
        targets = set(mapping.values())
        # another joint that already uses a target name moves out of the way
        for n in self.live():
            if n not in mapping and self.names[n] in targets:
                self._rename(n, self._unique(self.names[n] + "_orig"))
        for n, v in mapping.items():
            self._rename(n, v)
        return f"renamed {len(mapping)} bones to {scheme} names (other bones keep their names)"

    def op_move_joint(self, op):
        n = self.resolve(op["bone"])
        given = [k for k in ("offset", "delta", "to") if op.get(k) is not None]
        if len(given) != 1:
            raise RigEditError("give exactly one of offset (world units), delta (fractions of the bone length) or to (absolute position).")
        mode = str(op.get("children") or "stay").lower()
        if mode not in ("stay", "follow"):
            raise RigEditError("children must be 'stay' or 'follow'.")
        if given[0] == "offset":
            d = _vec3(op["offset"], "offset")
        elif given[0] == "delta":
            d = _vec3(op["delta"], "delta") * self.length(n)
        else:
            d = _vec3(op["to"], "to") - self.pos(n)
        targets = [(n, d)]
        mirror_note = ""
        if op.get("mirror"):
            twin = self._mirror_of(n)
            if twin is None:
                mirror_note = " (no mirrored joint found, only this side moved)"
            else:
                targets.append((twin, d * np.array([-1.0, 1.0, 1.0])))
                mirror_note = f" and {self.names[twin]} (mirrored)"
        for node, dv in targets:
            if float(np.linalg.norm(dv)) < 1e-12:
                continue
            if mode == "follow":
                self._shift_subtree(node, dv)
                self.dirty.add(node)
            else:
                m = self.G[node].copy()
                m[:3, 3] += dv
                self.G[node] = m
                self.dirty.add(node)
                for c in self.js["nodes"][node].get("children") or []:
                    self.dirty.add(c)              # children keep their world position: their local offset changes
            self.flags["moved"] = True
        return f"moved {self.names[n]} by ({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}) m, children {mode}{mirror_note}"

    def _mirror_of(self, n: int):
        live = self.live()
        roots = [j for j in live if self.jparent(j) is None]
        cx = float(self.pos(roots[0])[0]) if roots else 0.0
        p = self.pos(n)
        if abs(p[0] - cx) < 0.01 * max(self.height(), 1e-6):
            return None
        q = np.array([2 * cx - p[0], p[1], p[2]])
        best, dist = None, 1e18
        for j in live:
            if j == n:
                continue
            d = float(np.linalg.norm(self.pos(j) - q))
            if d < dist:
                best, dist = j, d
        return best if best is not None and dist <= max(0.35 * self.length(n), 0.03 * self.height()) else None

    def op_delete_bone(self, op):
        if op.get("match") is not None:
            pat = str(op["match"])
            nodes = [n for n in self.live() if fnmatch.fnmatchcase(self.names[n].lower(), pat.lower())]
            if not nodes:
                raise RigEditError(f"no bone name matches '{pat}'. Bones: {self.candidates_text(self.live(), 40)}.")
        else:
            refs = op.get("bone")
            if refs in (None, ""):
                raise RigEditError("give bone (a name, index or role word) or match (a glob like \"gun*\").")
            refs = refs if isinstance(refs, list) else [refs]
            nodes = []
            for r in refs:
                n = self.resolve(r)
                if n not in nodes:
                    nodes.append(n)
        include = bool(op.get("include_children"))
        target_ref = op.get("merge_weights_to")
        done = []
        for n in nodes:
            if n in self.removed:
                continue
            done.append(self._delete_one(n, include, target_ref))
        return "deleted " + ", ".join(done) + (" (with their children)" if include else "") + "; weights merged into the parent"

    def _delete_one(self, n: int, include: bool, target_ref) -> str:
        doomed = self.subtree(n) if include else [n]
        for x in doomed:
            if any(k in self.js["nodes"][x] for k in ("mesh", "camera", "skin")):
                raise RigEditError(f"{self.names[x]} also carries a mesh or camera, so it cannot be deleted as a bone.")
        live = self.live()
        if len(doomed) >= len(live):
            raise RigEditError("this would delete every bone of the rig.")
        if target_ref not in (None, ""):
            tgt = self.resolve(target_ref, "merge_weights_to")
            if tgt in doomed:
                raise RigEditError("merge_weights_to must be a bone that is not deleted.")
        else:
            tgt = self.jparent(n)
            if tgt is None:
                kids = [k for k in self.jchildren(n) if k not in doomed]
                if len(kids) == 1:
                    tgt = kids[0]
                else:
                    raise RigEditError(f"{self.names[n]} is the root: pass merge_weights_to (a bone that takes its weights) "
                                       f"or delete it with a single child ({self.candidates_text(kids)}).")
        label = self.names[n]
        top_parent = self.nparent.get(n)
        top_scenes = self._scenes_of(n) if top_parent is None else None
        # children of the deleted bone(s) that survive go to the node parent of the top bone, world position kept
        for x in doomed:
            for c in list(self.js["nodes"][x].get("children") or []):
                if c in doomed:
                    continue
                self._detach(c)
                self._attach(c, top_parent, scenes_from=top_scenes)
        for x in reversed(doomed):
            self._detach(x)
            self.removed.add(x)
            self.merge[x] = tgt
        self.flags["deleted"] = True
        return label if len(doomed) == 1 else f"{label} (+{len(doomed) - 1})"

    def _new_bone(self, parent: int, pos: np.ndarray, name: str) -> int:
        nodes = self.js["nodes"]
        n = len(nodes)
        m = np.eye(4)
        m[:3, 3] = pos
        nodes.append({"name": name})
        self.names[n] = name
        self.G[n] = m
        self.G0[n] = m.copy()
        self.order.append(n)
        self.added.append(n)
        self._attach(n, parent)
        self.dirty.add(n)
        self.flags["added"] = True
        return n

    def op_add_bone(self, op):
        parent = self.resolve(op["parent"], "parent")
        if op.get("to") is not None:
            if op.get("direction") is not None:
                raise RigEditError("give either to (a position) or direction + length, not both.")
            pos = _vec3(op["to"], "to")
        else:
            if op.get("direction") is None:
                raise RigEditError("give to (a position [x, y, z]) or direction (and optionally length).")
            ln = float(op["length"]) if op.get("length") is not None else 0.5 * self.length(parent)
            if not 0 < ln < 1e3:
                raise RigEditError("length must be a positive number.")
            pos = self.pos(parent) + _direction(op["direction"]) * ln
        if float(np.linalg.norm(pos - self.pos(parent))) < 1e-9:
            raise RigEditError("the new bone would sit exactly on its parent; give a direction and length.")
        name = self._check_name(op["name"]) if op.get("name") else self._unique(f"bone_new_{len(self.added) + 1}")
        n = self._new_bone(parent, pos, name)
        return f"added {name} under {self.names[parent]} at ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"

    def op_add_chain(self, op):
        parent = self.resolve(op["parent"], "parent")
        try:
            count = int(op["count"])
        except (TypeError, ValueError):
            raise RigEditError("count must be a whole number.") from None
        if not 1 <= count <= 64:
            raise RigEditError("count must be between 1 and 64.")
        if op.get("direction") is not None:
            d = _direction(op["direction"])
        else:                                                       # continue the parent's own direction
            gp = self.jparent(parent)
            v = self.pos(parent) - self.pos(gp) if gp is not None else np.array([0.0, 1.0, 0.0])
            d = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else np.array([0.0, 1.0, 0.0])
        if op.get("total_length") is not None and op.get("length") is not None:
            raise RigEditError("give length (per bone) or total_length, not both.")
        if op.get("total_length") is not None:
            seg = float(op["total_length"]) / count
        elif op.get("length") is not None:
            seg = float(op["length"])
        else:
            seg = 0.5 * self.length(parent)
        if not 0 < seg < 1e3:
            raise RigEditError("length must be a positive number.")
        prefix = str(op.get("name_prefix") or "chain").strip()[:40] or "chain"
        cur, pos, made = parent, self.pos(parent), []
        for k in range(count):
            pos = pos + d * seg
            cur = self._new_bone(cur, pos, self._unique(f"{prefix}_{k + 1:02d}"))
            made.append(self.names[cur])
        return f"added {count} bones {made[0]} .. {made[-1]} under {self.names[parent]}, {seg:.4f} m each towards ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})"

    def op_reparent(self, op):
        n = self.resolve(op["bone"])
        p = self.resolve(op["parent"], "parent")
        if p in self.subtree(n):
            raise RigEditError(f"{self.names[p]} is {'the bone itself' if p == n else 'below ' + self.names[n]}: that would make a loop.")
        if self.jparent(n) is None:
            raise RigEditError(f"{self.names[n]} is the root bone and cannot get a parent.")
        if self.jparent(n) == p and self.nparent.get(n) == p:
            return f"{self.names[n]} already is a child of {self.names[p]}"
        self._detach(n)
        self._attach(n, p)
        self.flags["reparented"] = True
        return f"{self.names[n]} is now a child of {self.names[p]}"

    # ---------------------------------------------------------------- write
    def _set_local(self, n: int, loc: np.ndarray):
        node = self.js["nodes"][n]
        t, q, s = rig_anim._trs({"matrix": loc.T.reshape(-1).tolist()})
        node.pop("matrix", None)
        for k in ("translation", "rotation", "scale"):
            node.pop(k, None)
        if np.any(np.abs(t) > 1e-12):
            node["translation"] = [float(x) for x in t]
        if np.any(np.abs(q - np.array([0, 0, 0, 1.0])) > 1e-9):
            node["rotation"] = [float(x) for x in q]
        if np.any(np.abs(s - 1.0) > 1e-9):
            node["scale"] = [float(x) for x in s]

    def _append_view(self, arr: np.ndarray, target=None) -> int:
        self.buf.extend(b"\x00" * (-len(self.buf) % 4))
        off = len(self.buf)
        self.buf.extend(arr.tobytes())
        views = self.js.setdefault("bufferViews", [])
        v = {"buffer": 0, "byteOffset": off, "byteLength": int(arr.nbytes)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def _weight_sets(self):
        """[(JOINTS accessor idx, WEIGHTS accessor idx)] of every primitive that uses the skin (deduplicated)."""
        js = self.js
        pairs, seen = [], set()
        for n in js.get("nodes") or []:
            if n.get("skin") not in self.skin_ids or n.get("mesh") is None:
                continue
            for prim in js["meshes"][n["mesh"]].get("primitives") or []:
                at = prim.get("attributes") or {}
                k = 0
                while f"JOINTS_{k}" in at and f"WEIGHTS_{k}" in at:
                    key = (at[f"JOINTS_{k}"], at[f"WEIGHTS_{k}"])
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
                    k += 1
        return pairs

    def weight_share(self) -> np.ndarray:
        """Share of the total skin weight carried by each ORIGINAL joint (sums to 1), or zeros when there are no weights."""
        J = len(self.joints0)
        acc = np.zeros(J)
        for ja, wa in self._weight_sets():
            j = _read_acc(self.js, self.buf, ja).astype(np.int64)
            w = self._weights_float(wa)
            ok = (j >= 0) & (j < J)
            np.add.at(acc, j[ok], w[ok])
        tot = acc.sum()
        return acc / tot if tot > 0 else acc

    def _weights_float(self, wa) -> np.ndarray:
        a = self.js["accessors"][wa]
        w = _read_acc(self.js, self.buf, wa).astype(np.float64)
        if _is_norm_int(a):
            w /= 255.0 if a["componentType"] == 5121 else 65535.0
        return w

    def finish(self) -> tuple[bytes, dict]:
        js = self.js
        nodes = js["nodes"]
        live = self.live()
        # 1. local transforms of every node whose parent or world position changed
        for n in sorted(self.dirty):
            if n in self.removed:
                continue
            p = self.nparent.get(n)
            pm = self.world(p) if p is not None else np.eye(4)
            self._set_local(n, np.linalg.inv(pm) @ self.world(n))
        # 2. inverse bind matrices: joint world * inverse bind stays what it was for every existing joint
        bind = None
        for j in self.joints0:
            if j not in self.removed:
                bind = self.G0[j] @ self.ibm0[self.index0[j]]
                break
        if bind is None:
            bind = np.eye(4)
        ibm = []
        for j in live:
            if j in self.index0:
                same = np.allclose(self.G[j], self.G0[j], atol=1e-12)
                ibm.append(self.ibm0[self.index0[j]] if same else np.linalg.inv(self.G[j]) @ self.G0[j] @ self.ibm0[self.index0[j]])
            else:
                ibm.append(np.linalg.inv(self.G[j]) @ bind)
        ibm = np.array(ibm)
        # 3. skin weights: deleted joints merge into their target; joint indices follow the new joint order
        new_index = {j: i for i, j in enumerate(live)}

        def final_target(j):
            seen = 0
            while j in self.removed and seen < 10000:
                j, seen = self.merge[j], seen + 1
            return j
        remap = np.zeros(len(self.joints0), dtype=np.int64)
        for j in self.joints0:
            remap[self.index0[j]] = new_index[final_target(j)]
        changed_order = any(remap[i] != i for i in range(len(remap)))
        if changed_order:
            self._rewrite_weights(remap, len(live))
        # 4. node compaction (deleted joints without other duties disappear)
        drop = {j for j in self.removed if not any(k in nodes[j] for k in ("mesh", "camera", "skin"))}
        self._compact(drop, live)
        structural = any(self.flags[k] for k in ("moved", "reparented", "deleted", "added"))
        dropped = list(js.pop("animations", []) or []) if structural else []    # a rename keeps the clips: node indices are unchanged
        # IBM accessor
        flat = np.ascontiguousarray(ibm.transpose(0, 2, 1).reshape(len(live), 16), dtype="<f4")
        view = self._append_view(flat)
        js.setdefault("accessors", []).append({"bufferView": view, "componentType": 5126, "count": len(live), "type": "MAT4"})
        for sid in self.skin_ids:
            js["skins"][sid]["inverseBindMatrices"] = len(js["accessors"]) - 1
        js["buffers"][0]["byteLength"] = len(self.buf)
        out = write_glb(js, bytes(self.buf))
        return out, {"dropped_animations": [a.get("name") or f"animation {i}" for i, a in enumerate(dropped)]}

    def _rewrite_weights(self, remap: np.ndarray, count: int):
        js = self.js
        for ja, wa in self._weight_sets():
            jacc, wacc = js["accessors"][ja], js["accessors"][wa]
            J = _read_acc(js, self.buf, ja).astype(np.int64)
            W = self._weights_float(wa)
            J = np.clip(J, 0, len(remap) - 1)
            J = remap[J]
            J = np.where(W > 0, J, 0)
            order = np.argsort(J, axis=1, kind="stable")
            J = np.take_along_axis(J, order, 1)
            W = np.take_along_axis(W, order, 1)
            for k in range(1, J.shape[1]):               # equal joint ids are now neighbours: add them up in the last slot of the run
                same = (J[:, k] == J[:, k - 1]) & (W[:, k - 1] > 0)
                W[:, k] = np.where(same, W[:, k] + W[:, k - 1], W[:, k])
                W[:, k - 1] = np.where(same, 0.0, W[:, k - 1])
            order = np.argsort(-W, axis=1, kind="stable")
            J = np.take_along_axis(J, order, 1)
            W = np.take_along_axis(W, order, 1)
            J = np.where(W > 0, J, 0)
            # joints: same layout when the component type holds the new indices, otherwise a new accessor
            dt_max = {5121: 255, 5123: 65535}.get(jacc["componentType"], 0)
            if J.max(initial=0) <= dt_max:
                v = _acc_view(js, self.buf, ja)
                v[...] = J.astype(v.dtype)
                del v
            else:
                view = self._append_view(np.ascontiguousarray(J.astype("<u2")), 34962)
                js["accessors"].append({"bufferView": view, "componentType": 5123, "count": int(J.shape[0]), "type": jacc["type"]})
                self._repoint(ja, len(js["accessors"]) - 1)
            if _is_norm_int(wacc):
                scale = 255.0 if wacc["componentType"] == 5121 else 65535.0
                Wq = np.rint(np.clip(W, 0, 1) * scale)
                v = _acc_view(js, self.buf, wa)
                v[...] = Wq.astype(v.dtype)
            elif wacc["componentType"] == 5126:
                v = _acc_view(js, self.buf, wa)
                v[...] = W.astype(v.dtype)
            else:
                raise RigEditError("Unsupported WEIGHTS component type.")
            del v

    def _repoint(self, old: int, new: int):
        for m in self.js.get("meshes") or []:
            for prim in m.get("primitives") or []:
                at = prim.get("attributes") or {}
                for k, v in list(at.items()):
                    if v == old and k.startswith("JOINTS_"):
                        at[k] = new

    def _compact(self, drop: set, live: list):
        js = self.js
        nodes = js["nodes"]
        if drop:
            keep = [i for i in range(len(nodes)) if i not in drop]
            mp = {o: n for n, o in enumerate(keep)}
            new_nodes = []
            for o in keep:
                node = nodes[o]
                if node.get("children") is not None:
                    ch = [mp[c] for c in node["children"] if c in mp]
                    if ch:
                        node["children"] = ch
                    else:
                        node.pop("children")
                new_nodes.append(node)
            js["nodes"] = new_nodes
            for sc in js.get("scenes") or []:
                sc["nodes"] = [mp[c] for c in sc.get("nodes", []) if c in mp]
        else:
            mp = {i: i for i in range(len(nodes))}
        joints = [mp[j] for j in live if j in mp]
        for sid in self.skin_ids:
            sk = js["skins"][sid]
            sk["joints"] = joints
            skel = sk.get("skeleton")
            if skel is not None:
                sk["skeleton"] = mp.get(skel)
                if sk["skeleton"] is None:
                    sk.pop("skeleton")
        # every moved / removed node index also lives in the document: fix cameras-free references (none besides the above)


# ------------------------------------------------------------------------------------------------------------ public API
def parse_ops(ops) -> list:
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except ValueError:
            raise RigEditError("ops must be a JSON list of operations.") from None
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list) or not ops:
        raise RigEditError(f"ops must be a non-empty list of operations, for example [{{\"op\": \"rename_standard\", \"scheme\": \"mixamo\"}}]. "
                           f"Ops: {', '.join(OP_FIELDS)}.")
    if len(ops) > MAX_OPS:
        raise RigEditError(f"At most {MAX_OPS} operations per edit.")
    return ops


def validate_ops(ops) -> list[str]:
    """Structure check only (no skeleton needed): op names, unknown / missing fields, vector shapes -> list of problems (empty = fine).
    Bone names are resolved when the edit runs."""
    try:
        ops = parse_ops(ops)
    except RigEditError as e:
        return [str(e)]
    out = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or not isinstance(op.get("op"), str) or op["op"].strip().lower() not in OP_FIELDS:
            out.append(f"Operation {i + 1}: needs \"op\" to be one of {', '.join(OP_FIELDS)}.")
            continue
        name = op["op"].strip().lower()
        allowed, required = OP_FIELDS[name]
        extra = sorted(set(op) - allowed - {"op"})
        if extra:
            out.append(f"Operation {i + 1} ({name}): unknown field(s) {', '.join(extra)}; fields: {', '.join(sorted(allowed))}.")
        miss = sorted(k for k in required if op.get(k) in (None, ""))
        if miss:
            out.append(f"Operation {i + 1} ({name}): missing {', '.join(miss)}.")
        if name == "delete_bone" and not (op.get("bone") or op.get("match")):
            out.append(f"Operation {i + 1} (delete_bone): give bone or match.")
        if name == "move_joint" and sum(op.get(k) is not None for k in ("offset", "delta", "to")) != 1:
            out.append(f"Operation {i + 1} (move_joint): give exactly one of offset, delta, to.")
        if name == "rename_standard" and str(op.get("scheme", "")).lower() not in SCHEMES:
            out.append(f"Operation {i + 1} (rename_standard): scheme must be one of {', '.join(SCHEMES)}.")
        for k in ("offset", "delta", "to") if name in ("move_joint", "add_bone") else ():
            if op.get(k) is not None:
                try:
                    _vec3(op[k], k)
                except RigEditError as e:
                    out.append(f"Operation {i + 1} ({name}): {e}")
        for k in ("direction",) if name in ("add_bone", "add_chain") else ():
            if op.get(k) is not None:
                try:
                    _direction(op[k])
                except RigEditError as e:
                    out.append(f"Operation {i + 1} ({name}): {e}")
    return out[:8]


def edit(src: bytes | str | Path, ops, *, regen_clips: bool = True) -> tuple[bytes, dict]:
    """GLB (bytes or path) + ops -> (new GLB bytes, report). Raises NoSkinError / RigEditError (kind 'input')."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    ops = parse_ops(ops)
    doc = RigDoc(data)
    before = len(doc.live())
    for i, op in enumerate(ops):
        doc.apply(op, i)
    out, info = doc.finish()
    notes = []
    regenerated = []
    wanted = [c for c in info["dropped_animations"] if c in rig_anim.CLIPS]
    if regen_clips and wanted:
        try:
            out, ai = rig_anim.add_animations(out, wanted)
            regenerated = [c["name"] for c in ai["clips"]]
        except RigAnimError as e:
            notes.append(f"preview clips not rebuilt: {e}")
    if info["dropped_animations"]:
        left = [n for n in info["dropped_animations"] if n not in regenerated]
        notes.append("Animation clips were removed because the joints they move changed"
                     + (f" ({', '.join(left)} gone)" if left else "") + (f"; {', '.join(regenerated)} rebuilt for the new rig" if regenerated else "") + ".")
    zero = [doc.names[n] for n in doc.added if n not in doc.removed]
    suggested = any(doc.flags[k] for k in ("moved", "reparented", "deleted", "added"))
    if zero:
        notes.append(f"{len(zero)} new bone(s) have no skin weight yet: run a re-skin (task reskin) so the mesh follows them.")
    elif suggested and any(doc.flags[k] for k in ("moved", "reparented")):
        notes.append("The existing weights were kept; run a re-skin (task reskin) if the moved joints should reshape the weights.")
    report = {"ops": doc.log, "joints_before": before, "joints_after": len(doc.live()), "needs_reskin": bool(zero),
              "reskin_suggested": bool(suggested), "zero_weight_bones": zero,
              "animations_dropped": info["dropped_animations"], "animations_rebuilt": regenerated, "notes": notes,
              "bytes_before": len(data), "bytes_after": len(out), "renamed": doc.flags["renamed"]}
    return out, report


def describe(src: bytes | str | Path) -> dict:
    """The skeleton of a skinned GLB for people and LLMs: bones with index, parent, world position, role, weight share."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    doc = RigDoc(data)
    share = doc.weight_share()
    hum, err = doc.humanoid()
    rows = []
    for i, j in enumerate(doc.joints0):
        p = doc.jparent(j)
        rows.append({"index": i, "name": doc.names[j], "parent": doc.index0[p] if p is not None else None,
                     "position": [round(float(x), 5) for x in doc.pos(j)], "children": len(doc.jchildren(j)),
                     "length": round(doc.length(j), 5), "role": doc.role_of(j), "weight_share": round(float(share[i]), 5)})
    X = np.array([doc.pos(j) for j in doc.joints0])
    out = {"joints": rows, "count": len(rows),
           "bounds": {"min": [round(float(x), 4) for x in X.min(0)], "max": [round(float(x), 4) for x in X.max(0)]},
           "height": round(float(X[:, 1].max() - X[:, 1].min()), 4),
           "animations": [a.get("name") or f"animation {i}" for i, a in enumerate(doc.js.get("animations") or [])],
           "units": "glTF world units (metres); +Y up, the character faces +Z, its left side is +X",
           "humanoid": ({"ok": True, "facing": hum["facing"], "method": hum["method"], "warnings": hum["warnings"],
                         "roles": {k: doc.names[v] for k, v in hum["roles"].items()}} if hum else {"ok": False, "reason": err}),
           "weighted": bool(share.sum() > 0)}
    out["zero_weight"] = [r["name"] for r in rows if r["weight_share"] <= 0]
    return out


def skeleton_arrays(src: bytes | str | Path) -> dict:
    """Names, parents (index into the joint list, None = root) and world positions [J,3] (glTF frame) of the skin's joints.
    Used by the re-skin task. Raises RigEditError when the rig has no single root or two bones share a name."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    doc = RigDoc(data)
    names = [doc.names[j] for j in doc.joints0]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise RigEditError(f"Bone names must be unique to re-skin; duplicated: {', '.join(dup[:6])}. Rename them with tool rig.edit first.")
    parents = []
    for j in doc.joints0:
        p = doc.jparent(j)
        parents.append(doc.index0[p] if p is not None else None)
    roots = [i for i, p in enumerate(parents) if p is None]
    if len(roots) != 1:
        raise RigEditError(f"A re-skin needs a skeleton with exactly one root bone; this one has {len(roots)}.")
    return {"names": names, "parents": parents, "positions": np.array([doc.pos(j) for j in doc.joints0])}


# ------------------------------------------------------------------------------------------------------------ UniRig bridge
def unirig_order(parents) -> list[int]:
    """Joint permutation with the root first and every parent before its children (UniRig requires parent < child).
    Identity when the list already is in that order; otherwise a depth-first order that keeps sibling order."""
    J = len(parents)
    if parents and parents[0] is None and all(p is not None and p < i for i, p in enumerate(parents) if i > 0):
        return list(range(J))
    kids: dict[int, list[int]] = {i: [] for i in range(J)}
    roots = []
    for i, p in enumerate(parents):
        (roots if p is None else kids[p]).append(i)
    if len(roots) != 1:
        raise RigEditError(f"A re-skin needs a skeleton with exactly one root bone; this one has {len(roots)}.")
    out, st = [], [roots[0]]
    while st:
        n = st.pop()
        out.append(n)
        st.extend(reversed(kids[n]))
    if len(out) != J:
        raise RigEditError("The skeleton is not a single connected tree.")
    return out


def unirig_tails(joints, parents) -> np.ndarray:
    """Bone tails the way UniRig's skeleton decoder makes them (tokenizer/spec.py make_skeleton): a bone with one child ends at
    that child, a leaf extends half its own length away from the parent, a branching bone half its length along the parent
    direction (the root: half the mean child distance along +Z). `parents` must be in UniRig order (parent < child)."""
    X = np.asarray(joints, dtype=np.float64)
    J = len(X)
    kids: dict[int, list[int]] = {i: [] for i in range(J)}
    for i, p in enumerate(parents):
        if p is not None:
            kids[p].append(i)
    tails = X.copy()
    for i in range(J):
        p = parents[i]
        if len(kids[i]) == 1:
            tails[i] = X[kids[i][0]]
        elif len(kids[i]) == 0 or len(kids[i]) > 1 and p is not None:
            d = X[i] - X[p] if p is not None else np.zeros(3)
            if float(np.linalg.norm(d)) <= 1e-9:
                d = np.array([0.0, 0.0, 1.0])
            tails[i] = X[i] + 0.5 * d
        else:
            av = float(np.mean([np.linalg.norm(X[i] - X[c]) for c in kids[i]]))
            tails[i] = X[i] + np.array([0.0, 0.0, 0.5 * av])
    return tails


def edit_file(src, dst, ops, **kw) -> dict:
    out, rep = edit(src, ops, **kw)
    Path(dst).write_bytes(out)
    return rep


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "describe":
        print(json.dumps(describe(sys.argv[2]), indent=1))
    elif len(sys.argv) == 4:
        print(json.dumps(edit_file(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text())), indent=1))
    else:
        sys.exit("usage: rig_edit.py describe in.glb | rig_edit.py in.glb out.glb ops.json")
