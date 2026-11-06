"""Video group: registry entries (models.d/video.yaml), prompt styles, worker helpers and the ffmpeg tools (no GPU, no weights)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.registry import Registry

STUDIO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDIO / "workers"))


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("video")
    return Registry(STUDIO / "models.yaml", Settings(tmp / "settings.json"), hf_cache=tmp / "hf", lmstudio=tmp / "lm",
                    probe_cache=tmp / "probe.json")


def video_models(reg):
    return {k: m for k, m in reg.models.items() if m.get("backend") == "video"}


def test_loads_without_errors(reg):
    assert not [e for e in reg.errors if "video" in str(e).lower() or "wan" in str(e).lower()]
    assert len(video_models(reg)) >= 4


def test_tasks_declared(reg):
    for t in ("t2v", "i2v", "flf2v", "interpolate"):
        assert reg.tasks[t]["studio"] == "video" and reg.tasks[t]["modality"] == "video"


def test_backend_script_exists(reg):
    b = reg.backends["video"]
    assert (STUDIO / b["script"]).exists()
    assert (STUDIO / "workers" / "video_common.py").exists()


def test_every_model_has_basics(reg):
    import video_worker
    for mid, m in video_models(reg).items():
        assert m["license"].get("commercial") in (True, False) and m["license"].get("name"), mid
        assert m["memory_gb"] > 0 and m["est"]["run_s"] > 0, mid
        assert m["modality"] == "video" and m.get("family") == video_worker.family_of(m), mid
        names = [p["name"] for p in m["params"]]
        assert len(names) == len(set(names)), mid
        for task, preset in [(None, v) for v in m["presets"].values()]:
            assert set(preset) <= set(names), (mid, preset)
        for p in m["params"]:
            if p["type"] == "choice":
                vals = [c["value"] if isinstance(c, dict) else c for c in p["choices"]]
                assert p["default"] in vals, (mid, p["name"])
        if m["tasks"] != ["interpolate"]:
            assert {"seconds", "aspect", "batch", "seed"} <= set(names), mid


def test_aspect_presets_cover_social_formats(reg):
    for mid, m in video_models(reg).items():
        a = next((p for p in m["params"] if p["name"] == "aspect"), None)
        if a:
            assert {"16:9", "9:16", "1:1"} <= set(a["choices"]), mid


def test_weights_are_hf_and_license_flags(reg):
    for mid, m in video_models(reg).items():
        assert all("hf" in w for w in m["weights"]), mid
    assert reg.models["wan22-ti2v-5b"]["license"]["commercial"] is True
    assert reg.models["hunyuan15-480p-i2v-distilled"]["license"].get("verify") is True   # Tencent EU / UK / KR exclusion
    assert "EU" in reg.models["hunyuan15-480p-i2v-distilled"]["license"]["note"] or "European" in reg.models["hunyuan15-480p-i2v-distilled"]["license"]["note"]


def test_param_validation_clips_and_coerces(reg):
    p = reg.validate_params("wan22-ti2v-5b", {"seconds": 99, "aspect": "9:16", "resolution": "480", "batch": 9})
    assert p["seconds"] == 5 and p["aspect"] == "9:16" and p["resolution"] == 480 and p["batch"] == 4 and p["steps"] == 30
    with pytest.raises(ValueError):
        reg.validate_params("wan22-ti2v-5b", {"aspect": "7:3"})


def test_prompt_styles_exist_for_every_model(reg):
    styles = yaml.safe_load((STUDIO / "prompts" / "video.yaml").read_text())["styles"]
    for mid, m in video_models(reg).items():
        ps = m.get("prompt_style")
        for s in (ps.values() if isinstance(ps, dict) else [ps]):
            if m["tasks"] != ["interpolate"]:
                assert s in styles, (mid, s)
    for s in styles.values():
        assert s["system"].strip() and s["chips"]


# ------------------------------------------------------------------ worker helpers
def test_dims_and_frames():
    import video_common as vc
    assert vc.dims_from_aspect("16:9", 480, 16) == (848, 480)
    assert vc.dims_from_aspect("9:16", 480, 16) == (480, 848)
    assert vc.dims_from_aspect("1:1", 704, 32) == (704, 704)
    assert vc.dims_from_aspect("16:9", 480, 32, image_size=(1000, 1000)) == (480, 480)    # follow the input image
    assert vc.frame_count(5, 24, 4, 121) == 121
    assert vc.frame_count(5, 16, 4, 81) == 81
    assert vc.frame_count(10, 24, 8, 241) == 241
    assert vc.frame_count(60, 24, 4, 121) == 121           # capped
    assert vc.frame_count(0.1, 24, 8, 241) == 9            # at least one step
    for secs in (1, 2.5, 3.3, 7):
        assert (vc.frame_count(secs, 24, 8, 241) - 1) % 8 == 0 and (vc.frame_count(secs, 24, 4, 121) - 1) % 4 == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_write_mp4_h264_faststart_and_audio(tmp_path):
    import numpy as np

    import video_common as vc
    frames = (np.random.rand(10, 64, 96, 3) * 255).astype("uint8")
    audio = np.sin(np.linspace(0, 440 * 6.28, 24000, dtype="float32"))[None, :] * 0.3
    out = tmp_path / "x.mp4"
    info = vc.write_mp4(out, frames, 12, audio=audio, sample_rate=24000)
    assert info["frames"] == 10 and info["has_audio"]
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "h264,yuv420p" in pr and "aac" in pr
    head = out.read_bytes()[:64]
    assert b"ftyp" in head
    assert out.read_bytes().find(b"moov") < out.read_bytes().find(b"mdat")      # faststart: moov before mdat
    from app import media
    assert media.verify_video(out)["width"] == 96


def test_hook_progress_reports_and_cancels():
    import video_common as vc

    class Ctx:
        def __init__(self):
            self.ev, self.n = [], 0

        def progress(self, *a, **k):
            self.ev.append(a)

        def check(self):
            self.n += 1
            if self.n > 2:
                raise RuntimeError("cancelled")

    class Pipe:
        pass
    p, c = Pipe(), Ctx()
    vc.hook_progress(p, c, "take 1 · ")
    with p.progress_bar(total=5) as bar:
        bar.update()
        bar.update()
        with pytest.raises(RuntimeError):
            bar.update()
    assert c.ev[0][:3] == (1, 5, "denoising") and "take 1" in c.ev[0][3]


# ------------------------------------------------------------------ plugin
def test_plugin_helpers():
    from app.plugins import video as vp
    import math
    for sp in (0.25, 0.5, 1.0, 3.0, 4.0):
        prod = math.prod(float(x.split("=")[1]) for x in vp._atempo(sp).split(","))
        assert abs(prod - sp) < 1e-3 and all(0.5 <= float(x.split("=")[1]) <= 2.0 for x in vp._atempo(sp).split(","))
    assert vp._lic([{"license": {"name": "MIT", "commercial": True}}, {"license": {"name": "CC-BY-NC", "commercial": False}}])["name"] == "CC-BY-NC"
    assert vp._lic([{}]) is None
    with pytest.raises(Exception):
        vp._num("abc", 0, 1, 0, "x")
    assert vp.formats()["aspects"]["9:16"] == [1080, 1920]
