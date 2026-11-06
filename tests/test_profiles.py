"""Use-case profiles: definitions, settings API, tagged prompt chips."""
import pytest
import yaml
from fastapi.testclient import TestClient

from app import config, profiles


@pytest.fixture
def client():
    import app.main as main
    from app.core import Core

    main.core = Core()
    with TestClient(main.app) as c:
        yield c, main.core
    main.core.settings.update({"profile": "general", "export": {"preset": None}, "license": {"commercial_only": False, "strict": False}})
    main.core = None


def test_profile_definitions_are_consistent():
    assert set(profiles.PROFILES) == set(profiles.PROFILE_IDS)
    assert profiles.DEFAULT_PROFILE == "general"
    for pid, p in profiles.PROFILES.items():
        assert p["export"] in profiles.EXPORT_PRESETS, pid
        assert p["export"] in p["export_presets"], pid
        assert set(p["export_presets"]) <= set(profiles.EXPORT_PRESETS), pid
        if p["aspect"]:
            assert p["aspect"] in p["aspects"], pid
    # game keeps today's behaviour: original files, no aspect override
    assert profiles.PROFILES["game"]["export"] == "original" and profiles.PROFILES["game"]["aspect"] is None
    assert profiles.PROFILES["game"]["hints"]["rgba"] and profiles.PROFILES["game"]["hints"]["loop"]
    assert profiles.PROFILES["video"]["aspect"] == "16:9" and profiles.PROFILES["social"]["aspect"] == "9:16"


def test_loudness_targets_per_profile():
    lufs = lambda pid: (profiles.EXPORT_PRESETS[profiles.PROFILES[pid]["export"]]["audio"] or {}).get("lufs")  # noqa: E731
    assert lufs("video") == -14.0 and lufs("podcast") == -16.0 and lufs("game") is None and lufs("film") == -23.0
    a = profiles.EXPORT_PRESETS["video_online"]["audio"]
    assert (a["sample_rate"], a["bit_depth"]) == (48000, 24)
    assert profiles.EXPORT_PRESETS["broadcast"]["audio"]["lufs"] == -23.0


def test_resolve_export():
    class S:
        def __init__(self, d): self.d = d
        def get(self, *k, default=None):
            v = self.d
            for x in k:
                if not isinstance(v, dict) or x not in v:
                    return default
                v = v[x]
            return v
    assert profiles.resolve_export({"preset": "original"})["audio"] is None
    r = profiles.resolve_export({"preset": "podcast", "audio": {"format": "wav"}})
    assert r["audio"]["format"] == "wav" and r["audio"]["lufs"] == -16.0 and r["image"]["format"] == "jpeg"
    assert profiles.resolve_export({"preset": "podcast", "audio": None})["audio"] is None
    assert profiles.resolve_export({}, S({"profile": "video"}))["preset"] == "video_online"
    assert profiles.resolve_export({}, S({"profile": "video", "export": {"preset": "broadcast"}}))["preset"] == "broadcast"
    with pytest.raises(ValueError):
        profiles.resolve_export({"preset": "nope"})


def test_settings_api(client):
    c, core = client
    r = c.get("/api/profiles").json()
    assert r["current"] == "general" and {p["id"] for p in r["profiles"]} == set(profiles.PROFILE_IDS)
    assert "video_online" in {p["id"] for p in r["export_presets"]}
    assert c.get("/api/bootstrap").json()["settings"]["profile"] == "general"
    s = c.patch("/api/settings", json={"profile": "video"}).json()
    assert s["profile"] == "video" and c.get("/api/profiles").json()["default_export"] == "video_online"
    assert c.patch("/api/settings", json={"profile": "nope"}).status_code == 400
    assert c.patch("/api/settings", json={"export": {"preset": "nope"}}).status_code == 400
    assert c.patch("/api/settings", json={"export": {"preset": "broadcast"}}).json()["export"]["preset"] == "broadcast"
    assert c.get("/api/profiles").json()["default_export"] == "broadcast"
    assert "preset" not in c.patch("/api/settings", json={"export": {"preset": None}}).json()["export"]


def _all_chips():
    for f in sorted(config.PROMPTS.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text()) or {}
        for name, chips in (doc.get("chipsets") or {}).items():
            for c in chips:
                yield f.name, name, c
        for sid, st in (doc.get("styles") or {}).items():
            for c in (st or {}).get("chips") or []:
                if isinstance(c, dict):
                    yield f.name, sid, c


def test_prompt_chips_profile_tags_are_valid():
    seen = {}
    for fname, group, c in _all_chips():
        for key in ("profiles", "default_for"):
            for p in c.get(key) or []:
                assert p in profiles.PROFILE_IDS, f"{fname}:{group}:{c['id']} {key} has unknown profile {p!r}"
        assert c.get("hint") and c.get("label")
        seen.setdefault(group, set()).add(c["id"])
    # media-oriented presets exist
    assert {"thumbnail", "broll", "product", "social_vertical", "storyboard", "concept_art", "cover_art"} <= seen["image_game"]
    assert {"podcast_sting", "ad_jingle", "bed_music"} <= seen["sound_music"]
    assert {"bed", "roomtone"} <= seen["sound_amb"]


def test_game_chips_stay_first_for_game_profile():
    doc = yaml.safe_load((config.PROMPTS / "image.yaml").read_text())
    chips = doc["chipsets"]["image_game"]
    ids = [c["id"] for c in chips]
    # the original game chip order is untouched (new media chips are appended after it)
    assert ids[:14] == ["3d_ready", "transparent", "stylized", "lowpoly", "realistic", "pixel", "icon", "isometric", "concept",
                        "turnaround", "sprite", "texture", "cozy", "dark"]
    # video-only chips are tagged so the UI can order / dim them
    thumb = next(c for c in chips if c["id"] == "thumbnail")
    assert "video" in thumb["profiles"] and "game" not in thumb["profiles"]


def test_improve_styles_expose_profile_tags(client):
    c, _ = client
    styles = c.get("/api/improve/styles").json()["styles"]
    qi = next(s for s in styles if s["id"] == "qwen_image")
    assert any(ch.get("profiles") for ch in qi["chips"])
