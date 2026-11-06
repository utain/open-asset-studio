"""Export presets: ffmpeg two-pass loudnorm + rate / bit depth / format for audio, crop / size / format for images.
Originals are never modified; everything is CPU only (ffmpeg + Pillow)."""
import hashlib
import shutil
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

from app import media
from app.assets import AssetError, AssetStore

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _bursts(path, sr=44100, stereo=True):
    rng = np.random.default_rng(1)
    x = np.zeros(sr * 8, dtype="float32")
    for k in range(4):
        n = sr
        x[k * 2 * sr:k * 2 * sr + n] = (rng.standard_normal(n) * 0.08 * np.hanning(n)).astype("float32")
    sf.write(str(path), np.stack([x, x], 1) if stereo else x, sr)
    return path


def _lufs(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return media.integrated_lufs(x, sr), sr


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@needs_ffmpeg
@pytest.mark.parametrize("target", [-14.0, -16.0, -23.0])
def test_loudnorm_hits_the_target(tmp_path, target):
    src = _bursts(tmp_path / "in.wav")
    out, rep = media.export_audio(src, tmp_path / "o", "x", {"format": "wav", "lufs": target, "sample_rate": 48000, "bit_depth": 24})
    lufs, sr = _lufs(out)
    assert abs(lufs - target) < 0.6, (lufs, rep)
    assert sr == 48000 and sf.info(str(out)).subtype == "PCM_24"
    assert rep["normalized"] and rep["input_lufs"] < -15 and abs(rep["output_lufs"] - target) < 0.6
    assert rep["output_true_peak"] <= -0.9


@needs_ffmpeg
@pytest.mark.parametrize("fmt,sub", [("wav", "PCM_16"), ("flac", "PCM_24"), ("mp3", None), ("ogg", None)])
def test_formats_and_sample_rate(tmp_path, fmt, sub):
    src = _bursts(tmp_path / "in.wav")
    out, rep = media.export_audio(src, tmp_path / "o", "clip", {"format": fmt, "sample_rate": 44100, "bit_depth": 24 if fmt == "flac" else 16, "lufs": -16})
    assert out.suffix == f".{fmt}" and out.stat().st_size > 1000
    info = sf.info(str(out))
    assert info.samplerate == 44100 and rep["sample_rate"] == 44100
    if sub:
        assert info.subtype == sub


@needs_ffmpeg
def test_no_loudness_target_keeps_level_and_rate(tmp_path):
    src = _bursts(tmp_path / "in.wav")
    out, rep = media.export_audio(src, tmp_path / "o", "x", {"format": "flac"})
    assert not rep["normalized"]
    assert abs(_lufs(out)[0] - _lufs(src)[0]) < 0.3 and sf.info(str(out)).samplerate == 44100


@needs_ffmpeg
def test_loudnorm_keeps_source_rate_when_unspecified(tmp_path):
    out, _ = media.export_audio(_bursts(tmp_path / "in.wav"), tmp_path / "o", "x", {"format": "wav", "lufs": -14})
    assert sf.info(str(out)).samplerate == 44100      # loudnorm works at 192 kHz internally; the file must not stay there


@needs_ffmpeg
def test_music_sample_rate_override(tmp_path):
    src = _bursts(tmp_path / "in.wav")
    o = {"format": "wav", "sample_rate": 48000, "music_sample_rate": 44100}
    assert sf.info(str(media.export_audio(src, tmp_path / "a", "x", o)[0])).samplerate == 48000
    assert sf.info(str(media.export_audio(src, tmp_path / "b", "x", o, music=True)[0])).samplerate == 44100


@needs_ffmpeg
def test_silence_is_converted_without_gain(tmp_path):
    sf.write(str(tmp_path / "s.wav"), np.zeros(44100, dtype="float32"), 44100)
    out, rep = media.export_audio(tmp_path / "s.wav", tmp_path / "o", "x", {"format": "wav", "lufs": -14})
    assert out.is_file() and not rep["normalized"] and "silent" in rep["note"]


def test_audio_option_validation(tmp_path):
    for bad in ({"format": "aac"}, {"sample_rate": 10}, {"bit_depth": 8}, {"lufs": 3}, {"lufs": -80}, {"channels": 5}, {"true_peak": 4}):
        with pytest.raises(media.ExportError):
            media.audio_export_options(bad)
    assert media.audio_export_options({"format": "FLAC", "bit_depth": 32})["bit_depth"] == 24


def test_image_aspect_crop_and_size(tmp_path):
    Image.new("RGB", (1000, 600), (10, 200, 30)).save(tmp_path / "a.png")
    out, rep = media.export_image(tmp_path / "a.png", tmp_path / "o", "i", {"format": "jpeg", "aspect": "9:16", "width": 1080, "height": 1920})
    assert out.suffix == ".jpg" and Image.open(out).size == (1080, 1920) and rep["source_width"] == 1000
    out, _ = media.export_image(tmp_path / "a.png", tmp_path / "o", "sq", {"format": "png", "aspect": "1:1"})
    assert Image.open(out).size == (600, 600)
    out, _ = media.export_image(tmp_path / "a.png", tmp_path / "o", "w", {"format": "webp", "max_side": 300})
    assert Image.open(out).size == (300, 180) and out.suffix == ".webp"
    out, _ = media.export_image(tmp_path / "a.png", tmp_path / "o", "n", {"format": "png", "max_side": 4000})
    assert Image.open(out).size == (1000, 600)      # never upscaled by max_side


def test_image_crop_is_centred_and_pad_adds_transparency(tmp_path):
    im = Image.new("RGB", (300, 100), (255, 0, 0))
    im.paste((0, 0, 255), (100, 0, 200, 100))       # blue centre third
    im.save(tmp_path / "a.png")
    out, _ = media.export_image(tmp_path / "a.png", tmp_path / "o", "c", {"format": "png", "aspect": "1:1"})
    c = Image.open(out)
    assert c.size == (100, 100) and c.getpixel((50, 50)) == (0, 0, 255)
    out, _ = media.export_image(tmp_path / "a.png", tmp_path / "o", "p", {"format": "png", "aspect": "1:1", "fit": "pad"})
    p = Image.open(out)
    assert p.size == (300, 300) and p.getpixel((5, 5))[3] == 0 and p.getpixel((150, 150))[:3] == (0, 0, 255)


def test_jpeg_flattens_alpha(tmp_path):
    Image.new("RGBA", (40, 40), (255, 0, 0, 0)).save(tmp_path / "a.png")
    out, _ = media.export_image(tmp_path / "a.png", tmp_path / "o", "j", {"format": "jpeg", "background": "#00ff00"})
    px = Image.open(out).getpixel((20, 20))
    assert px[1] > 200 and px[0] < 60


def test_image_option_validation(tmp_path):
    for bad in ({"format": "gif"}, {"aspect": "wide"}, {"width": 4}, {"background": "red"}):
        with pytest.raises(media.ExportError):
            media.image_export_options(bad)


@pytest.fixture
def store(tmp_path):
    return AssetStore(tmp_path / "projects")


@needs_ffmpeg
def test_render_and_zip_never_touch_the_original(store, tmp_path):
    wav = _bursts(tmp_path / "a.wav")
    png = tmp_path / "i.png"
    Image.new("RGB", (800, 400), (1, 2, 3)).save(png)
    a = store.ingest("default", wav, "audio", {"name": "Boom", "task": "sfx"}, move=False)
    i = store.ingest("default", png, "image", {"name": "Pic"}, move=False)
    ma = store.asset_dir("default", a["id"]) / a["main"]
    mi = store.asset_dir("default", i["id"]) / i["main"]
    before = (_sha(ma), _sha(mi), (store.asset_dir("default", a["id"]) / "meta.json").read_text())
    spec = {"audio": {"format": "mp3", "lufs": -16, "sample_rate": 44100}, "image": {"format": "jpeg", "aspect": "16:9", "width": 640, "height": 360}}
    r = store.render(a["id"], spec, tmp_path / "ex")
    assert r["path"].suffix == ".mp3" and r["report"]["normalized"] and r["path"].parent.parent == tmp_path / "ex"
    z = store.export([a["id"], i["id"]], tmp_path / "ex", convert=spec)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert "audio/boom.mp3" in names and "images/pic.jpg" in names and "manifest.json" in names
        man = __import__("json").loads(zf.read("manifest.json"))
        assert man["export"]["audio"]["lufs"] == -16 and man["assets"][0]["export"]["original"] == a["main"]
        zf.extract("images/pic.jpg", tmp_path / "unz")
    assert Image.open(tmp_path / "unz" / "images" / "pic.jpg").size == (640, 360)
    assert (_sha(ma), _sha(mi), (store.asset_dir("default", a["id"]) / "meta.json").read_text()) == before
    # no conversion requested -> the stored files are zipped untouched
    z2 = store.export([a["id"]], tmp_path / "ex")
    with zipfile.ZipFile(z2) as zf:
        assert "audio/boom.wav" in zf.namelist()
    # an export spec with nothing for this asset type is refused for a single render
    with pytest.raises(AssetError):
        store.render(i["id"], {"audio": {"format": "wav"}, "image": None}, tmp_path / "ex")


@needs_ffmpeg
def test_http_render_and_tool(tmp_path):
    import time

    import app.main as main
    from app.core import Core

    main.core = Core()
    try:
        with TestClient(main.app) as c:
            core = main.core
            a = core.assets.ingest("default", _bursts(tmp_path / "a.wav"), "audio", {"name": "Clip"}, move=False)
            r = c.post(f"/api/assets/{a['id']}/render", json={"export": {"preset": "video_online"}})
            assert r.status_code == 200, r.text
            j = r.json()
            assert j["name"].endswith(".wav") and abs(j["report"]["output_lufs"] + 14) < 0.6
            d = c.get(j["url"])
            assert d.status_code == 200 and d.headers["content-type"] == "audio/wav" and len(d.content) == j["bytes"]
            assert c.get("/api/export/render/zzzzzzzzzzzz/x.wav").status_code == 404
            assert c.get("/api/export/render/" + "0" * 12 + "/..%2Fx").status_code in (404, 400)
            assert c.post(f"/api/assets/{a['id']}/render", json={"export": {"preset": "original"}}).status_code == 400
            assert c.post(f"/api/assets/{a['id']}/render", json={"export": {"preset": "nope"}}).status_code == 400
            z = c.post("/api/export", json={"ids": [a["id"]], "export": {"preset": "podcast"}}).json()
            with zipfile.ZipFile(__import__("io").BytesIO(c.get(z["url"]).content)) as zf:
                assert "audio/clip.mp3" in zf.namelist()
            # tool job: saves derived NEW assets with parents, originals untouched
            job = c.post("/api/jobs", json={"tool": "asset.export", "params": {"preset": "broadcast"}, "inputs": [{"asset": a["id"]}]}).json()
            for _ in range(100):
                st = c.get(f"/api/jobs/{job['id']}").json()
                if st["status"] in ("done", "failed"):
                    break
                time.sleep(0.2)
            assert st["status"] == "done", st
            out = core.assets.get(st["outputs"][0])
            assert out["parents"] == [a["id"]] and out["id"] != a["id"] and out["info"]["export"]["target_lufs"] == -23.0
    finally:
        main.core = None
