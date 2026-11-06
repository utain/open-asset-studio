"""Run folders under outputs/: one directory per generation, params.json next to the results."""
from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from pathlib import Path

from PIL import Image

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[a-z]+_[0-9a-f]{4}$")
URL_PREFIX = "/outputs"


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- ids / paths ------------------------------------------------------
    def new_run(self, kind: str) -> tuple[str, Path]:
        for _ in range(20):
            rid = f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{secrets.token_hex(2)}"
            d = self.root / rid
            try:
                d.mkdir()
                return rid, d
            except FileExistsError:
                continue
        raise RuntimeError("could not allocate a run folder")

    def run_dir(self, run_id: str) -> Path:
        """Validated lookup; the id regex rules out '..' and absolute paths."""
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id: {run_id!r}")
        d = (self.root / run_id).resolve()
        if d.parent != self.root or not d.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return d

    def exists(self, run_id: str | None) -> bool:
        try:
            self.run_dir(run_id)
            return True
        except (ValueError, FileNotFoundError, TypeError):
            return False

    def url(self, run_id: str, name: str) -> str:
        return f"{URL_PREFIX}/{run_id}/{name}"

    # -- reading ------------------------------------------------------------
    def read_json(self, run_id: str, name: str) -> dict:
        try:
            return json.loads((self.run_dir(run_id) / name).read_text())
        except (OSError, ValueError):
            return {}

    def list_runs(self, limit: int = 200) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if not (d.is_dir() and RUN_ID_RE.match(d.name)):
                continue
            meta = self.read_json(d.name, "params.json")
            stats = self.read_json(d.name, "stats.json")
            out.append({
                "id": d.name,
                "kind": meta.get("kind", d.name.split("_")[1]),
                "source_run": meta.get("source_run"),
                "created": meta.get("created", ""),
                "has_white": (d / "white_mesh.glb").exists(),
                "has_textured": (d / "textured_mesh.glb").exists(),
                "faces": stats.get("number_of_faces"),
                "total_s": (stats.get("time") or {}).get("total"),
                "thumb": str(d / "thumb.png") if (d / "thumb.png").exists() else None,
                "controls": meta.get("controls", {}),
            })
            if len(out) >= limit:
                break
        return out

    # -- writing -------------------------------------------------------------
    @staticmethod
    def write_json(path: Path, data: dict):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        tmp.replace(path)

    def save_thumb(self, run_dir: Path, image: Image.Image):
        thumb = image.convert("RGBA")
        thumb.thumbnail((192, 192))
        thumb.save(run_dir / "thumb.png")

    def fork_for_texture(self, src_id: str) -> tuple[str, Path]:
        """New run that reuses the source run's mesh + input images (the 'Texture existing shape' flow)."""
        src = self.run_dir(src_id)
        rid, dst = self.new_run("retex")
        for name in ("white_mesh.glb", "input.png", "input_original.png", "thumb.png"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        return rid, dst

    def prune_orphans(self, min_age_s: float = 3600) -> int:
        """Remove run folders without params.json (left behind by a crash / kill) that are older than min_age_s."""
        n = 0
        for d in self.root.iterdir():
            try:
                if (d.is_dir() and RUN_ID_RE.match(d.name) and not (d / "params.json").exists()
                        and time.time() - d.stat().st_mtime > min_age_s):
                    shutil.rmtree(d, ignore_errors=True)
                    n += 1
            except OSError:
                continue
        return n

    def delete(self, run_id: str):
        shutil.rmtree(self.run_dir(run_id))
