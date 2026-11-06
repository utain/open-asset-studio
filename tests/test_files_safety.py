"""Path safety of the file routes, uploads and export downloads."""
import io
import os
import struct
import wave

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.assets import AssetError, AssetStore, safe_name


def _png(path, size=(32, 24), alpha=False):
    Image.new("RGBA" if alpha else "RGB", size, (200, 30, 30, 128) if alpha else (200, 30, 30)).save(path)
    return path


def _wav(path, secs=0.2, sr=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", (i * 997) % 20000 - 10000) for i in range(int(sr * secs))))
    return path


@pytest.fixture
def store(tmp_path):
    return AssetStore(tmp_path / "projects")


def test_ids_and_names_are_validated(store, tmp_path):
    a = store.ingest("default", _png(tmp_path / "x.png"), "image", {"name": "red"})
    ok = store.file_path("default", a["id"], a["main"])
    assert ok.is_file() and ok.parent.parent.parent.parent == store.root
    for bad in ["../meta.json", "..", "/etc/passwd", "a/b.png", "%2e%2e", ".hidden", "image.png/..", "meta.json.."]:
        with pytest.raises(AssetError):
            store.file_path("default", a["id"], bad)
    for bad_proj in ["../default", "Default", "", "a" * 60, "x/y", "."]:
        with pytest.raises(AssetError):
            store.file_path(bad_proj, a["id"], a["main"])
    for bad_id in ["../../x", "20261105-164419-zzzzzz", a["id"] + "/..", ""]:
        with pytest.raises(AssetError):
            store.file_path("default", bad_id, a["main"])
    # existing file that is not listed in meta.json is not served
    (store.asset_dir("default", a["id"]) / "secret.txt").write_text("x")
    with pytest.raises(AssetError):
        store.file_path("default", a["id"], "secret.txt")
    # right asset, wrong project
    store.create_project("other")
    with pytest.raises(AssetError):
        store.file_path("other", a["id"], a["main"])


def test_symlink_escape_is_refused(store, tmp_path):
    a = store.ingest("default", _png(tmp_path / "x.png"), "image")
    d = store.asset_dir("default", a["id"])
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (d / a["main"]).unlink()
    os.symlink(outside, d / a["main"])  # listed name, but now points outside the store
    with pytest.raises(AssetError):
        store.file_path("default", a["id"], a["main"])


def test_safe_name():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("my file (1).PNG") == "my_file_1.png"
    assert safe_name("") == "file"
    assert "/" not in safe_name("a/b\\c.wav")


def test_uploads_are_validated_by_content(store, tmp_path):
    fake = tmp_path / "evil.png"
    fake.write_text("<?php echo 1; ?>")
    with pytest.raises(AssetError):
        store.upload("default", "evil.png", fake)
    with pytest.raises(AssetError):
        store.upload("default", "script.sh", fake)
    bad_glb = tmp_path / "m.glb"
    bad_glb.write_bytes(b"NOTGLTF" * 10)
    with pytest.raises(AssetError):
        store.upload("default", "m.glb", bad_glb)
    a = store.upload("default", "../../sneaky name.png", _png(tmp_path / "good.png", alpha=True))
    assert a["main"] == "image.png" and a["original_filename"] == "sneaky_name.png"
    assert a["info"]["has_alpha"] and a["preview"]["thumb"] == "thumb.webp"
    w = store.upload("default", "beep.wav", _wav(tmp_path / "b.wav"))
    assert w["type"] == "audio" and w["preview"]["waveform"] == "waveform.json" and w["info"]["duration"] == 0.2


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.main as main
    from app.core import Core

    main.core = Core()
    with TestClient(main.app) as c:
        yield c, main.core
    main.core = None


def test_http_routes(client, tmp_path):
    c, core = client
    a = core.assets.ingest("default", _png(tmp_path / "x.png"), "image", {"name": "red"})
    r = c.get(f"/files/default/{a['id']}/{a['main']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    for url in [f"/files/default/{a['id']}/..%2Fmeta.json", f"/files/default/{a['id']}/%2e%2e",
                f"/files/..%2F..%2Fetc/{a['id']}/passwd", "/files/default/..%2F..%2F..%2Fetc/passwd",
                f"/files/default/{a['id']}/meta.json", f"/files/default/{a['id']}/..\\meta.json",
                "/api/export/..%2F..%2Fsettings.json", "/api/export/export-1.zip"]:
        assert c.get(url).status_code in (400, 404), url
    # upload via HTTP: content checked, server path never accepted
    r = c.post("/api/upload", files={"file": ("fake.png", io.BytesIO(b"not an image"), "image/png")}, data={"project": "default"})
    assert r.status_code == 415
    r = c.post("/api/upload", files={"file": ("ok.png", open(_png(tmp_path / "ok.png"), "rb"), "image/png")},
               data={"project": "../x"})
    assert r.status_code == 400
    r = c.get("/api/system/logs", params={"name": "../../../etc/passwd"})
    assert r.status_code == 200 and r.json()["lines"] == []
    # export zip of the asset, then download it by its generated name only
    r = c.post("/api/export", json={"ids": [a["id"]]})
    assert r.status_code == 200
    z = c.get(r.json()["url"])
    assert z.status_code == 200 and z.content[:2] == b"PK"
    assert (config.DATA / "exports").is_dir()


# ------------------------------------------------------------------ video assets (v2a inputs, muxed attachments)
import shutil
import subprocess

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg / ffprobe not installed")


def _mp4(path, secs=1.0, size="160x120"):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={secs}:size={size}:rate=10",
                        "-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return path


@needs_ffmpeg
def test_video_upload_is_validated_by_content(store, tmp_path):
    src = _mp4(tmp_path / "src.mp4", 1.0)
    v = store.upload("default", "../../my clip.MP4", shutil.copy(src, tmp_path / "ok.mp4"))
    assert v["type"] == "video" and v["main"] == "video.mp4" and v["original_filename"] == "my_clip.mp4"
    assert abs(v["info"]["duration"] - 1.0) < 0.2 and v["info"]["width"] == 160 and v["info"]["height"] == 120
    assert v["info"]["has_audio"] and v["info"]["bytes"] > 0
    assert v["preview"]["thumb"] == "thumb.webp" and store.file_path("default", v["id"], "thumb.webp").is_file()
    assert [f["mime"] for f in v["files"]] == ["video/mp4"]
    # a mov / mkv copy is accepted by its own magic; the wrong extension for the container is refused
    mkv = tmp_path / "ok.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", str(mkv)], check=True)
    assert store.upload("default", "ok.mkv", mkv)["type"] == "video"
    with pytest.raises(AssetError):
        store.upload("default", "renamed.webm", shutil.copy(src, tmp_path / "renamed.webm"))      # mp4 bytes, .webm name
    # things that are not videos
    for name, payload in [("evil.mp4", b"<?php echo 1; ?>" * 10), ("pic.mp4", _png(tmp_path / "p.png").read_bytes()),
                          ("empty.mp4", b""), ("trunc.mp4", b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"),
                          ("list.mkv", b"#EXTM3U\n#EXT-X-VERSION:3\nfile:///etc/passwd\n"),
                          ("concat.mp4", b"ffconcat version 1.0\nfile '/etc/passwd'\n")]:
        f = tmp_path / ("x_" + name)
        f.write_bytes(payload)
        with pytest.raises(AssetError):
            store.upload("default", name, f)
    # an audio-only mp4 (no video stream) is not a video
    a = tmp_path / "audio_only.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(a)], check=True)
    with pytest.raises(AssetError):
        store.upload("default", "audio_only.mp4", a)


def test_video_magic_bytes_without_ffprobe(store, tmp_path, monkeypatch):
    """No ffprobe / ffmpeg on the machine: containers are still checked by magic bytes, no poster is made."""
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    ok = tmp_path / "m.mp4"
    ok.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64)
    v = store.upload("default", "m.mp4", ok)
    assert v["type"] == "video" and v["info"]["bytes"] > 0 and "thumb" not in v["preview"]
    ebml = tmp_path / "m.webm"
    ebml.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
    assert store.upload("default", "m.webm", ebml)["type"] == "video"
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"MZ" + b"\x00" * 64)
    with pytest.raises(AssetError):
        store.upload("default", "bad.mp4", bad)


@needs_ffmpeg
def test_video_files_are_served_safely(client, tmp_path):
    c, core = client
    src = _mp4(tmp_path / "up.mp4")
    r = c.post("/api/upload", files={"file": ("clip.mp4", open(src, "rb"), "video/mp4")}, data={"project": "default"})
    assert r.status_code == 200 and r.json()["type"] == "video"
    v = r.json()
    r = c.get(f"/files/default/{v['id']}/{v['main']}")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and len(r.content) == v["info"]["bytes"]
    r = c.get(f"/files/default/{v['id']}/{v['main']}", headers={"Range": "bytes=0-99"})   # <video> seeks with ranges
    assert r.status_code == 206 and len(r.content) == 100
    assert c.get(f"/files/default/{v['id']}/{v['preview']['thumb']}").headers["content-type"] == "image/webp"
    for bad in ["..%2Fmeta.json", "%2e%2e", "video.mp4/..", "meta.json"]:
        assert c.get(f"/files/default/{v['id']}/{bad}").status_code in (400, 404), bad
    r = c.post("/api/upload", files={"file": ("fake.mp4", io.BytesIO(b"not a video"), "video/mp4")}, data={"project": "default"})
    assert r.status_code == 415
    assert c.get("/api/assets", params={"type": "video"}).json()["total"] >= 1
    # a muxed take_N_video.mp4 attached to an audio take (what the foley worker does): served as video, still listed-only
    wav = _wav(tmp_path / "take_1.wav")
    mux = tmp_path / "take_1_video.mp4"
    shutil.copy(src, mux)
    a = core.assets.ingest("default", wav, "audio", {"name": "foley take", "task": "v2a"}, attach=[mux])
    att = [f for f in a["files"] if f["role"] == "attachment"]
    assert [f["name"] for f in att] == ["take_1_video.mp4"] and att[0]["mime"] == "video/mp4"
    r = c.get(f"/files/default/{a['id']}/take_1_video.mp4")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    (core.assets.asset_dir("default", a["id"]) / "unlisted.mp4").write_bytes(b"x")
    assert c.get(f"/files/default/{a['id']}/unlisted.mp4").status_code == 404
    # export keeps videos in their own folder
    z = c.post("/api/export", json={"ids": [v["id"]]})
    assert z.status_code == 200
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(c.get(z.json()["url"]).content)).namelist()
    assert any(n.startswith("video/") and n.endswith(".mp4") for n in names)


def test_input_slots_are_validated_by_the_registry():
    from app.registry import Registry

    errs = []
    m = {"id": "x", "params": [], "input_slots": [
        {"role": "melody", "label": "Melody", "types": ["audio"], "required": False, "hint": "h"},
        {"role": "Bad Role", "types": ["audio"]}, {"role": "v", "types": ["pdf"]}, "nope",
        {"role": "video", "types": "video", "required": True, "tasks": "v2a", "unknown": 1}]}
    Registry._normalize(m, errs)
    assert [s["role"] for s in m["input_slots"]] == ["melody", "video"] and len(errs) == 3
    assert m["input_slots"][1] == {"role": "video", "label": "Video", "types": ["video"], "required": True, "hint": "", "tasks": ["v2a"]}
    # features.input_slots is the alias; models without slots get no key
    m2 = {"id": "y", "params": [], "features": {"input_slots": [{"role": "ref", "types": ["voice", "audio"], "required": True}]}}
    Registry._normalize(m2, errs)
    assert m2["input_slots"][0]["required"] is True
    m3 = {"id": "z", "params": []}
    Registry._normalize(m3, errs)
    assert "input_slots" not in m3
