"""Minimal binary FBX 7.4 writer for static meshes (Unreal and Unity packages of the engine export). numpy + struct + zlib only.

Written from the publicly documented binary layout (node records with absolute end offsets, typed properties, zlib arrays, the standard
footer); no Autodesk SDK and no third-party FBX code. Validated against ufbx (strict mode: version 7400, unit 0.01 m, +Y up, a LodGroup
with its levels, UCX nodes at the root, material and texture links, 0 warnings) and assimp; see docs/research/engine_export.md.

What it writes: Null / LodGroup / Mesh models (triangles, per-polygon-vertex normals, one UV set with v flipped for FBX, one or several
materials), Phong materials linked to Texture + Video objects (DiffuseColor, NormalMap) with RELATIVE file names only, GlobalSettings
(up axis Y, UnitScaleFactor 1 = centimetres: the caller writes vertices in centimetres). No skin, no animation, no vertex colours, no quads.

    sc = Scene(up="Y", unit_cm_per_unit=1.0)
    mat = sc.material("M_Rock", {"DiffuseColor": "Textures/T_Rock_BC.png", "NormalMap": "Textures/T_Rock_N.png"})
    grp = sc.group("SM_Rock", lod_levels=3)
    sc.mesh("SM_Rock_LOD0", V, F, N, UV, materials=[mat], parent=grp)
    sc.mesh("UCX_SM_Rock_LOD0_00", hullV, hullF, parent=0)          # parent 0 = the scene root
    sc.save("SM_Rock.fbx")
"""
from __future__ import annotations

import struct
import time
import zlib

import numpy as np

HEADER = b"Kaydara FBX Binary  \x00\x1a\x00"
FOOT_ID = bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e")
FOOT_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")
VERSION = 7400


class N:
    """One FBX node: name, properties [(type char, value)], children."""
    __slots__ = ("name", "props", "kids")

    def __init__(self, name, *props, kids=None):
        self.name, self.props, self.kids = name, list(props), list(kids or [])

    def add(self, name, *props, kids=None):
        n = N(name, *props, kids=kids)
        self.kids.append(n)
        return n


def _prop(p):
    t, v = p
    if t in "YCIFDL":
        return t.encode() + struct.pack({"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t], v)
    if t in "SR":
        b = v.encode("utf-8") if isinstance(v, str) else bytes(v)
        return t.encode() + struct.pack("<I", len(b)) + b
    if t in "fdilb":
        dt = {"f": "<f4", "d": "<f8", "i": "<i4", "l": "<i8", "b": "u1"}[t]
        raw = np.ascontiguousarray(v, dt).tobytes()
        n = len(raw) // np.dtype(dt).itemsize
        if len(raw) > 128:
            z = zlib.compress(raw, 6)
            return t.encode() + struct.pack("<III", n, 1, len(z)) + z
        return t.encode() + struct.pack("<III", n, 0, len(raw)) + raw
    raise ValueError(t)


def _node(n, off):
    props = b"".join(_prop(p) for p in n.props)
    name = n.name.encode()
    head = 13 + len(name)
    body = bytearray()
    pos = off + head + len(props)
    for k in n.kids:
        b = _node(k, pos)
        body += b
        pos += len(b)
    if n.kids or not n.props:
        body += b"\0" * 13
    end = off + head + len(props) + len(body)
    return struct.pack("<IIIB", end, len(n.props), len(props), len(name)) + name + props + bytes(body)


def write(path, roots):
    out = bytearray(HEADER + struct.pack("<I", VERSION))
    for r in roots:
        out += _node(r, len(out))
    out += b"\0" * 13
    out += FOOT_ID + b"\0" * 4
    out += b"\0" * ((16 - len(out) % 16) % 16 or 16)
    out += struct.pack("<I", VERSION) + b"\0" * 120 + FOOT_MAGIC
    with open(path, "wb") as fh:
        fh.write(out)


def p70(*entries):
    """entries: (name, type, label, flags, *values) -> a Properties70 node. Python floats are doubles, ints / bools are ints, str is text."""
    n = N("Properties70")
    for name, typ, label, flags, *vals in entries:
        ps = [("S", name), ("S", typ), ("S", label), ("S", flags)]
        for v in vals:
            if isinstance(v, bool):
                ps.append(("I", int(v)))
            elif isinstance(v, int):
                ps.append(("I", v))
            elif isinstance(v, float):
                ps.append(("D", v))
            else:
                ps.append(("S", str(v)))
        n.add("P", *ps)
    return n


def _name(base: str, cls: str) -> str:
    return f"{base}\x00\x01{cls}"


class Scene:
    """Collects objects and connections; `save(path)` writes the file. All ids are generated here; the root is id 0."""

    def __init__(self, up="Y", unit_cm_per_unit=1.0, creator="Open Asset Studio"):
        self.objects, self.conns = N("Objects"), N("Connections")
        self.counts: dict[str, int] = {}
        self._id = 1_000_000
        self.up, self.unit, self.creator = up, unit_cm_per_unit, creator
        self._tex: dict[str, tuple[int, int]] = {}

    def _new(self):
        self._id += 1
        return self._id

    def _count(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def link(self, child, parent, prop=None):
        if prop:
            self.conns.add("C", ("S", "OP"), ("L", child), ("L", parent), ("S", prop))
        else:
            self.conns.add("C", ("S", "OO"), ("L", child), ("L", parent))

    def _model(self, name, kind, parent):
        mid = self._new()
        m = self.objects.add("Model", ("L", mid), ("S", _name(name, "Model")), ("S", kind))
        m.add("Version", ("I", 232))
        m.kids.append(p70(("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0),
                          ("DefaultAttributeIndex", "int", "Integer", "", 0),
                          ("InheritType", "enum", "", "", 1)))
        m.add("Shading", ("C", True))
        m.add("Culling", ("S", "CullingOff"))
        self._count("Model")
        self.link(mid, parent)
        return mid

    def group(self, name, parent=0, lod_levels=0, thresholds=None):
        """A Null group, or a LodGroup (with its NodeAttribute and `lod_levels` display levels) when lod_levels > 0."""
        if not lod_levels:
            return self._model(name, "Null", parent)
        mid = self._model(name, "LodGroup", parent)
        aid = self._new()
        a = self.objects.add("NodeAttribute", ("L", aid), ("S", _name(name, "NodeAttribute")), ("S", "LodGroup"))
        th = thresholds or [None] * (lod_levels - 1)
        ents = [("MinMaxDistance", "bool", "", "", 0), ("MinDistance", "double", "Number", "", -100.0),
                ("MaxDistance", "double", "Number", "", 100.0), ("WorldSpace", "bool", "", "", 0)]
        for i in range(lod_levels - 1):
            ents.append((f"Thresholds|Level{i}", "Distance", "", "", float(th[i] or (i + 1) * 1000.0), "cm"))
        for i in range(lod_levels):
            ents.append((f"DisplayLevels|Level{i}", "enum", "", "", 0))
        a.kids.append(p70(*ents))
        a.add("TypeFlags", ("S", "LodGroup"))
        self._count("NodeAttribute")
        self.link(aid, mid)
        return mid

    def material(self, name, textures=None, color=(0.8, 0.8, 0.8)):
        """A Phong material; textures {slot: relative path} with slot DiffuseColor | NormalMap | EmissiveColor. Shared textures are written once."""
        mid = self._new()
        m = self.objects.add("Material", ("L", mid), ("S", _name(name, "Material")), ("S", ""))
        m.add("Version", ("I", 102))
        m.add("ShadingModel", ("S", "phong"))
        m.add("MultiLayer", ("I", 0))
        m.kids.append(p70(("DiffuseColor", "Color", "", "A", *map(float, color)), ("Diffuse", "Vector3D", "Vector", "", *map(float, color))))
        self._count("Material")
        for slot, rel in (textures or {}).items():
            if rel not in self._tex:
                tid, vid = self._new(), self._new()
                base = rel.rsplit("/", 1)[-1]
                v = self.objects.add("Video", ("L", vid), ("S", _name(base, "Video")), ("S", "Clip"))
                v.add("Type", ("S", "Clip"))
                v.kids.append(p70(("Path", "KString", "XRefUrl", "", rel)))
                v.add("UseMipMap", ("I", 0))
                v.add("Filename", ("S", rel))
                v.add("RelativeFilename", ("S", rel))
                t = self.objects.add("Texture", ("L", tid), ("S", _name(base, "Texture")), ("S", ""))
                t.add("Type", ("S", "TextureVideoClip"))
                t.add("Version", ("I", 202))
                t.add("TextureName", ("S", _name(base, "Texture")))
                t.kids.append(p70(("UseMaterial", "bool", "", "", 1)))
                t.add("Media", ("S", _name(base, "Video")))
                t.add("FileName", ("S", rel))
                t.add("RelativeFilename", ("S", rel))
                t.add("ModelUVTranslation", ("D", 0.0), ("D", 0.0))
                t.add("ModelUVScaling", ("D", 1.0), ("D", 1.0))
                t.add("Texture_Alpha_Source", ("S", "None"))
                t.add("Cropping", ("I", 0), ("I", 0), ("I", 0), ("I", 0))
                self._count("Texture")
                self._count("Video")
                self.link(vid, tid)
                self._tex[rel] = (tid, vid)
            self.link(self._tex[rel][0], mid, slot)
        return mid

    def mesh(self, name, V, F, Nrm=None, UV=None, materials=None, mat_per_face=None, parent=0, scale=1.0):
        """A triangle Mesh model. V (n,3) in metres * `scale` (100 -> centimetres), F (m,3), Nrm (n,3) per vertex, UV (n,2) glTF-style (v down),
        materials: list of material ids (slot order); mat_per_face: (m,) slot index per triangle (omit for one material)."""
        V = np.asarray(V, np.float64) * scale
        F = np.asarray(F, np.int64).reshape(-1, 3)
        gid = self._new()
        g = self.objects.add("Geometry", ("L", gid), ("S", _name(name, "Geometry")), ("S", "Mesh"))
        g.add("Vertices", ("d", V.reshape(-1)))
        pvi = F.copy()
        pvi[:, 2] = -pvi[:, 2] - 1                             # the last index of a polygon is stored as ~i
        g.add("PolygonVertexIndex", ("i", pvi.reshape(-1).astype(np.int32)))
        g.add("GeometryVersion", ("I", 124))
        layer = N("Layer", ("I", 0), kids=[N("Version", ("I", 100))])
        if Nrm is not None:
            le = g.add("LayerElementNormal", ("I", 0))
            le.add("Version", ("I", 101))
            le.add("Name", ("S", ""))
            le.add("MappingInformationType", ("S", "ByPolygonVertex"))
            le.add("ReferenceInformationType", ("S", "Direct"))
            le.add("Normals", ("d", np.asarray(Nrm, np.float64)[F.reshape(-1)].reshape(-1)))
            layer.add("LayerElement", kids=[N("Type", ("S", "LayerElementNormal")), N("TypedIndex", ("I", 0))])
        if UV is not None:
            le = g.add("LayerElementUV", ("I", 0))
            le.add("Version", ("I", 101))
            le.add("Name", ("S", "UVMap"))
            le.add("MappingInformationType", ("S", "ByPolygonVertex"))
            le.add("ReferenceInformationType", ("S", "IndexToDirect"))
            uv = np.asarray(UV, np.float64).copy()
            uv[:, 1] = 1.0 - uv[:, 1]                          # glTF has its UV origin at the top, FBX at the bottom
            le.add("UV", ("d", uv.reshape(-1)))
            le.add("UVIndex", ("i", F.reshape(-1).astype(np.int32)))
            layer.add("LayerElement", kids=[N("Type", ("S", "LayerElementUV")), N("TypedIndex", ("I", 0))])
        materials = list(materials or [])
        if materials:
            le = g.add("LayerElementMaterial", ("I", 0))
            le.add("Version", ("I", 101))
            le.add("Name", ("S", ""))
            if len(materials) > 1 and mat_per_face is not None:
                le.add("MappingInformationType", ("S", "ByPolygon"))
                le.add("ReferenceInformationType", ("S", "IndexToDirect"))
                le.add("Materials", ("i", np.asarray(mat_per_face, np.int32)))
            else:
                le.add("MappingInformationType", ("S", "AllSame"))
                le.add("ReferenceInformationType", ("S", "IndexToDirect"))
                le.add("Materials", ("i", np.zeros(1, np.int32)))
            layer.add("LayerElement", kids=[N("Type", ("S", "LayerElementMaterial")), N("TypedIndex", ("I", 0))])
        g.kids.append(layer)
        self._count("Geometry")
        mid = self._model(name, "Mesh", parent)
        self.link(gid, mid)
        for m in materials:
            self.link(m, mid)
        return mid

    def save(self, path):
        up = {"Y": (1, 1, 2, 1), "Z": (2, 1, 1, -1)}[self.up]   # UpAxis, UpAxisSign, FrontAxis, FrontAxisSign
        t = time.localtime()
        hdr = N("FBXHeaderExtension", kids=[
            N("FBXHeaderVersion", ("I", 1003)), N("FBXVersion", ("I", VERSION)), N("EncryptionType", ("I", 0)),
            N("CreationTimeStamp", kids=[N("Version", ("I", 1000))] + [N(k, ("I", v)) for k, v in zip(
                ("Year", "Month", "Day", "Hour", "Minute", "Second", "Millisecond"), tuple(t[:6]) + (0,))]),
            N("Creator", ("S", self.creator))])
        gs = N("GlobalSettings", kids=[N("Version", ("I", 1000)), p70(
            ("UpAxis", "int", "Integer", "", up[0]), ("UpAxisSign", "int", "Integer", "", up[1]),
            ("FrontAxis", "int", "Integer", "", up[2]), ("FrontAxisSign", "int", "Integer", "", up[3]),
            ("CoordAxis", "int", "Integer", "", 0), ("CoordAxisSign", "int", "Integer", "", 1),
            ("OriginalUpAxis", "int", "Integer", "", up[0]), ("OriginalUpAxisSign", "int", "Integer", "", 1),
            ("UnitScaleFactor", "double", "Number", "", float(self.unit)),
            ("OriginalUnitScaleFactor", "double", "Number", "", float(self.unit)))])
        defs = N("Definitions", kids=[N("Version", ("I", 100)), N("Count", ("I", sum(self.counts.values()) + 1)),
                                      N("ObjectType", ("S", "GlobalSettings"), kids=[N("Count", ("I", 1))])] +
                [N("ObjectType", ("S", k), kids=[N("Count", ("I", v))]) for k, v in self.counts.items()])
        write(path, [hdr, N("FileId", ("R", bytes(range(16)))), N("CreationTime", ("S", time.strftime("%Y-%m-%d %H:%M:%S:000"))),
                     N("Creator", ("S", self.creator)), gs, N("Documents", kids=[N("Count", ("I", 0))]), N("References"),
                     defs, self.objects, self.conns, N("Takes", kids=[N("Current", ("S", ""))])])
