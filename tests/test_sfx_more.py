"""Registry-level checks for models.d/sfx_more.yaml (no GPU, no weights needed)."""
from app.registry import Registry

IDS = {"audioldm-l-full", "audioldm2", "audioldm2-large", "audioldm2-music", "tango2", "tangoflux",
       "make-an-audio-2", "stable-audio-open-1.0", "stable-audio-open-small"}


def test_sfx_more_entries_load():
    r = Registry()
    assert not r.errors
    mine = {m["id"]: m for m in r.list() if m.get("backend") in ("sfxmore", "sao")}
    assert IDS <= set(mine)
    for m in mine.values():
        assert m["memory_gb"] > 0 and m["license"].get("name")
        assert {p["name"] for p in m["params"]} >= {"seconds", "seed"}
    for mid in ("audioldm-l-full", "audioldm2", "tango2", "tangoflux"):
        assert mine[mid]["license"]["commercial"] is False
    for mid in ("stable-audio-open-1.0", "stable-audio-open-small"):   # gated on Hugging Face, but wired (workers/sao_worker.py)
        m = mine[mid]
        assert m["backend"] == "sao" and not m["features"].get("blocked")
        assert m["license"]["commercial"] is True and m["license"]["verify"] is True
        assert "hf auth login" in m["weights"][0]["fix"]
