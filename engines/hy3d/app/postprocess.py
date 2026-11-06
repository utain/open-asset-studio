"""Mesh post-processing, statistics and export (CPU only, no GPU lock needed).

Deliberately independent from the vendor `hy3dshape.postprocessors`: those write
undeleted temp .ply files on every call and `DegenerateFaceRemover` is a no-op.
Everything here works on in-memory trimesh / pymeshlab objects.
"""
from __future__ import annotations

import secrets
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

SUPPORTED_FORMATS = ("glb", "obj", "ply", "stl")

_WHITE = dict(baseColorFactor=[225, 225, 228, 255], metallicFactor=0.0, roughnessFactor=0.65)


# ----------------------------------------------------------------- loading
def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load any mesh file as a single Trimesh (scenes are concatenated)."""
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


# ------------------------------------------------------------------ cleaning
def remove_degenerate(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    """Drop zero-area / duplicate faces and unreferenced vertices."""
    before = len(mesh.faces)
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh, before - len(mesh.faces)


def _face_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    """Connected-component id per face (components joined by shared vertices)."""
    faces = np.asarray(mesh.faces)
    n_v = len(mesh.vertices)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_v, n_v))
    _, labels = connected_components(graph, directed=False)
    return labels[faces[:, 0]]


def remove_floaters(mesh: trimesh.Trimesh, ratio: float = 0.005) -> tuple[trimesh.Trimesh, int]:
    """Delete connected components with fewer than `ratio * total_faces` faces.

    Connectivity is by shared vertices (cheap, conservative). The largest
    component is always kept.
    """
    if len(mesh.faces) == 0:
        return mesh, 0
    face_label = _face_labels(mesh)
    counts = np.bincount(face_label)
    keep_label = counts >= max(1, ratio * len(face_label))
    keep_label[counts.argmax()] = True
    keep = keep_label[face_label]
    removed = int((~keep).sum())
    if removed == 0:
        return mesh, 0
    mesh = mesh.copy()
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh, removed


def reduce_faces(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation (pymeshlab, in memory). No-op if already small enough."""
    if len(mesh.faces) <= target:
        return mesh
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)))
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=int(target), qualitythr=1.0, preserveboundary=True, boundaryweight=3,
        preservenormal=True, preservetopology=True, autoclean=True,
    )
    cur = ms.current_mesh()
    return trimesh.Trimesh(cur.vertex_matrix(), cur.face_matrix(), process=False)


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Taubin smoothing (volume preserving, unlike plain Laplacian)."""
    if iterations <= 0:
        return mesh
    mesh = mesh.copy()
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(iterations))
    return mesh


def fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    trimesh.repair.fix_normals(mesh)
    return mesh


def run_pipeline(mesh: trimesh.Trimesh, *, floaters=False, floater_ratio=0.005, degenerate=False,
                 simplify=False, target_faces=10_000, smooth=0, normals=False) -> tuple[trimesh.Trimesh, dict]:
    """Apply the selected steps in a sensible order. Returns (mesh, report with timings)."""
    report: dict = {"faces_in": int(len(mesh.faces))}

    def step(name, fn):
        nonlocal mesh
        t = time.time()
        mesh, extra = fn(mesh)
        report[name] = {"seconds": round(time.time() - t, 2), **extra}

    def removal(fn, *args):
        return lambda m: (lambda r: (r[0], {"removed_faces": r[1]}))(fn(m, *args))

    if degenerate:
        step("degenerate", removal(remove_degenerate))
    if floaters:
        step("floaters", removal(remove_floaters, floater_ratio))
    if simplify:
        step("simplify", lambda m: (reduce_faces(m, int(target_faces)), {"target": int(target_faces)}))
    if smooth:
        step("smooth", lambda m: (smooth_mesh(m, int(smooth)), {"iterations": int(smooth)}))
    if normals:
        step("normals", lambda m: (fix_normals(m), {}))
    report["faces_out"] = int(len(mesh.faces))
    return mesh, report


# ---------------------------------------------------------------- statistics
def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    """JSON-safe geometry summary (all plain python numbers)."""
    ext = mesh.extents
    stats = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_min": [round(float(v), 4) for v in mesh.bounds[0]],
        "bbox_max": [round(float(v), 4) for v in mesh.bounds[1]],
        "extents": [round(float(v), 4) for v in ext],
        "watertight": bool(mesh.is_watertight),
        "surface_area": round(float(mesh.area), 4),
        "volume": round(float(mesh.volume), 5) if mesh.is_watertight else None,
    }
    try:
        stats["components"] = int(len(np.unique(_face_labels(mesh)))) if len(mesh.faces) else 0
    except Exception:  # statistics must never break a job
        pass
    return stats


# -------------------------------------------------------------------- export
def _white_visual(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attach a light-grey, non-metallic material so GLB viewers don't show default dark metal."""
    out = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(**_WHITE))
    return out


def export_mesh(mesh: trimesh.Trimesh, path: str | Path, fmt: str | None = None) -> Path:
    """Export a plain (untextured) mesh. GLB gets a neutral material and smooth normals."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "glb":
        out = _white_visual(mesh)
        _ = out.vertex_normals  # cache so the exporter writes NORMAL
        out.export(str(path), file_type="glb", include_normals=True)
    elif fmt == "obj":
        mesh.export(str(path), file_type="obj", include_normals=False)
    else:
        mesh.export(str(path), file_type=fmt)
    return path


def export_run_mesh(run_dir: str | Path, *, textured: bool, fmt: str, opts: dict) -> dict:
    """The 'Transform' action: post-process a run's mesh and export it.

    Returns {"download": path, "preview": glb path for the viewer, "report": {...}, "warnings": [...]}.
    Textured meshes keep their UVs, so geometry edits (simplify/floaters/smooth) are skipped for them.
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / "export"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{time.strftime('%H%M%S')}_{secrets.token_hex(2)}"  # unique even within one second
    warnings: list[str] = []

    if textured:
        glb = run_dir / "textured_mesh.glb"
        if not glb.exists():
            raise FileNotFoundError("This run has no textured mesh yet.")
        if any(opts.get(k) for k in ("floaters", "degenerate", "simplify", "smooth", "normals")):
            warnings.append("Mesh clean-up / simplify options were ignored: they would destroy the UV layout of a textured mesh.")
        if fmt == "glb":
            dst = out_dir / f"textured_mesh_{stamp}.glb"
            shutil.copy2(glb, dst)
        elif fmt == "obj":
            dst = out_dir / f"textured_mesh_obj_{stamp}.zip"
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
                for f in sorted(run_dir.glob("textured_mesh*")):
                    if f.suffix in (".obj", ".mtl", ".jpg", ".png"):
                        z.write(f, f.name)
        else:
            mesh = load_mesh(glb)
            if fmt == "ply":
                try:
                    mesh.visual = mesh.visual.to_color()
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"Could not bake texture to vertex colours ({e}); exported geometry only.")
            else:
                warnings.append("STL cannot carry colour or texture; exported geometry only.")
            dst = out_dir / f"textured_mesh_{stamp}.{fmt}"
            mesh.export(str(dst), file_type=fmt)
        return {"download": str(dst), "preview": str(glb), "report": {"warnings": warnings} if warnings else {}, "warnings": warnings}

    src = run_dir / "white_mesh.glb"
    if not src.exists():
        raise FileNotFoundError("This run has no mesh yet.")
    mesh = load_mesh(src)
    mesh, report = run_pipeline(
        mesh, floaters=opts.get("floaters", False), floater_ratio=opts.get("floater_ratio", 0.005),
        degenerate=opts.get("degenerate", False), simplify=opts.get("simplify", False),
        target_faces=opts.get("target_faces", 10_000), smooth=opts.get("smooth", 0),
        normals=opts.get("normals", False))
    dst = export_mesh(mesh, out_dir / f"white_mesh_{stamp}.{fmt}", fmt)
    preview = dst if fmt == "glb" else export_mesh(mesh, out_dir / f"preview_{stamp}.glb", "glb")
    report["result"] = mesh_stats(mesh)
    if warnings:
        report["warnings"] = warnings
    return {"download": str(dst), "preview": str(preview), "report": report, "warnings": warnings}
