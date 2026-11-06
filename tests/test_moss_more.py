"""OpenMOSS group (models.d/moss_more.yaml): entries load cleanly, tasks/backends/workers exist, every param validates its
own default, input slots are sane, and the pure helpers of the MOSS-TTSD script handling work. No GPU, no weights."""
import importlib.util
import sys

import pytest

from app import config
from app.registry import Registry

IDS = ["moss-ttsd-v1", "moss-tts-realtime", "moss-tts-local-v1.5", "moss-transcribe-diarize", "moss-audio-4b-instruct"]


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


@pytest.fixture(scope="module")
def worker():
    sys.path.insert(0, str(config.STUDIO / "workers"))
    spec = importlib.util.spec_from_file_location("moss_more_worker", config.STUDIO / "workers" / "moss_more_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_group_loads_without_errors(reg):
    assert not [e for e in reg.errors if "moss_more" in e]
    for mid in IDS:
        assert mid in reg.models, mid
        assert reg.models[mid]["_file"] == "moss_more.yaml"


def test_tasks_exist(reg):
    for mid in IDS:
        assert set(reg.models[mid]["tasks"]) <= set(reg.tasks), mid
    assert reg.tasks["transcribe"]["studio"] == "sound" and reg.tasks["audio_ask"]["studio"] == "sound"
    assert reg.models["moss-transcribe-diarize"]["default_for"] == ["transcribe"]


def test_backends_have_worker_scripts(reg):
    for mid in IDS:
        b = reg.backends[reg.models[mid]["backend"]]
        assert (config.STUDIO / b["script"]).exists(), mid
    assert reg.backends["moss_more"]["python"] == "envs/moss/bin/python"      # the shared env is only read
    assert reg.backends["moss_asr"]["python"] == "envs/moss_asr/bin/python"


def test_defaults_validate(reg):
    for mid in IDS:
        assert isinstance(reg.validate_params(mid, {}), dict), mid


def test_licences_apache_commercial(reg):
    for mid in IDS:
        assert reg.models[mid]["license"]["commercial"] is True, mid


def test_dialogue_slots_cover_four_speakers(reg):
    slots = reg.models["moss-ttsd-v1"].get("input_slots") or []
    for task in ("tts", "voice_clone"):
        roles = [s["role"] for s in slots if not s.get("tasks") or task in s["tasks"]]
        assert roles == ["ref", "ref2", "ref3", "ref4"], (task, roles)
    # voice_clone needs speaker 1, tts does not
    req = {t: [s["required"] for s in slots if t in s.get("tasks", []) and s["role"] == "ref"][0] for t in ("tts", "voice_clone")}
    assert req == {"tts": False, "voice_clone": True}
    names = {p["name"] for p in reg.models["moss-ttsd-v1"]["params"]}
    assert {"text", "ref_text", "ref_text2", "ref_text3", "ref_text4", "max_seconds"} <= names


def test_text_tasks_have_audio_slot(reg):
    for mid, task in (("moss-transcribe-diarize", "transcribe"), ("moss-audio-4b-instruct", "audio_ask")):
        slots = [s for s in reg.models[mid]["input_slots"] if task in s.get("tasks", [])]
        assert slots and slots[0]["role"] == "audio" and slots[0]["required"] is True


def test_mode_choices_validate(reg):
    out = reg.validate_params("moss-transcribe-diarize", {"mode": "speakers", "hotwords": "Tailscale, vLLM"})
    assert out["mode"] == "speakers"
    with pytest.raises(Exception):
        reg.validate_params("moss-transcribe-diarize", {"mode": "nonsense"})


# ------------------------------------------------------------------ worker helpers
def test_normalize_script_merges_and_cleans(worker):
    t = worker.normalize_script("[S1] Hello... world.\n[S1] Second part.\n[S2] ha ha ha, that is funny!")
    assert t.startswith("[S1]") and t.count("[S1]") == 1 and "[S2]" in t
    assert "..." not in t and "[laugh]" in t


def test_merge_tags_and_speakers(worker):
    assert worker.merge_tags("[S1]a [S1]b [S2]c [S1]d") == "[S1]a b[S2]c[S1]d"
    assert worker.script_speakers("[S2]x [S1]y [S2]z") == [1, 2]
    assert worker.script_speakers("no tags") == []


def test_tidy_trims_silence_mono_and_stereo(worker):
    import numpy as np

    sr = 1000
    x = np.zeros(sr * 3, dtype=np.float32)
    x[sr: sr * 2] = 0.5
    y = worker.tidy(x, sr)
    assert sr <= len(y) <= sr + 2 * int(0.12 * sr) + 2
    s = np.stack([x, x], axis=1)
    assert worker.tidy(s, sr).shape[1] == 2


def test_save_wav_stereo_and_clip_limit(worker, tmp_path):
    import numpy as np
    import soundfile as sf

    w = np.stack([np.sin(np.linspace(0, 50, 4800)) * 1.5, np.sin(np.linspace(0, 30, 4800))], axis=1).astype("float32")
    meta = worker.save_wav(tmp_path / "a.wav", w, 48000)
    assert meta["channels"] == 2 and meta["sample_rate"] == 48000 and "peak_limited" in meta
    data, sr = sf.read(tmp_path / "a.wav")
    assert data.shape[1] == 2 and abs(data).max() <= 0.9


def test_ref_transcript_falls_back_to_voice_meta(worker):
    class J:
        inputs = [{"role": "ref2", "meta": {"x_voice": {"ref_text": "Hello there."}}}]

        def input(self, role=None, type=None):  # noqa: A002
            return next((i for i in self.inputs if i["role"] == role), None)

    assert worker.ref_transcript(J(), "ref2", {}, "ref_text2") == "Hello there."
    assert worker.ref_transcript(J(), "ref2", {"ref_text2": "typed"}, "ref_text2") == "typed"
    assert worker.ref_transcript(J(), "ref", {}, "ref_text") == ""
