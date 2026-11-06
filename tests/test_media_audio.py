"""Media audio group (models.d/media_audio.yaml, workers/media_audio_common.py): no GPU, no weights needed."""
import sys
from pathlib import Path

import pytest

from app.config import Settings
from app.registry import Registry

STUDIO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDIO / "workers"))
import media_audio_common as C  # noqa: E402


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("media_audio")
    return Registry(STUDIO / "models.yaml", Settings(tmp / "settings.json"), hf_cache=tmp / "hf", lmstudio=tmp / "lm",
                    probe_cache=tmp / "probe.json")


def mine(reg):
    return {k: m for k, m in reg.models.items() if m.get("backend") == "media_audio"}


def test_registry_entries(reg):
    assert not [e for e in reg.errors if "media_audio" in str(e)]
    ms = mine(reg)
    assert {"whisper-subs-turbo", "whisper-subs-large-v3", "demucs-htdemucs", "demucs-htdemucs-ft", "demucs-htdemucs-6s",
            "melband-roformer-vocals", "dub-pipeline"} <= set(ms)
    for t in ("subtitles", "separate", "dub"):
        assert reg.tasks[t]["studio"] == "sound"
    for mid, m in ms.items():
        assert m["license"].get("commercial") in (True, False), mid
        assert m["memory_gb"] > 0 and m["weights"], mid
    assert (STUDIO / "workers" / "media_audio_worker.py").exists()
    assert reg.backends["media_audio"]["python"].endswith("envs/media_audio/bin/python")


def test_translate_only_on_large_v3(reg):
    assert reg.models["whisper-subs-large-v3"]["features"]["translate"] is True
    assert reg.models["whisper-subs-turbo"]["features"]["translate"] is False
    assert "translate" in {p["name"] for p in reg.models["whisper-subs-large-v3"]["params"]}
    assert "translate" not in {p["name"] for p in reg.models["whisper-subs-turbo"]["params"]}


def test_dub_param_defaults_validate(reg):
    p = reg.validate_params("dub-pipeline", {"target_language": "th", "max_speedup": 9, "lufs": -3})
    assert p["max_speedup"] == 1.8 and p["lufs"] == -8 and p["stage"] == "all"


# ------------------------------------------------------------------ subtitles
def test_time_formats():
    assert C.srt_time(3661.5) == "01:01:01,500"
    assert C.vtt_time(0.0004) == "00:00:00.000"


def words(text, t0=0.0, per=0.3, gap_after=None):
    out, t = [], t0
    for i, w in enumerate(text.split()):
        out.append({"text": (" " if i else "") + w, "start": round(t, 3), "end": round(t + per, 3)})
        t += per + ((gap_after or {}).get(i, 0.0))
    return out


def test_sentences_split_and_long_sentence_balanced():
    w = words("Hello there. This is a very long sentence that goes on and on and on, with a comma in the middle of it, "
              "and then keeps going until the end of the thought.")
    segs = C.words_to_segments(w, max_chars=40, max_lines=2)
    assert segs[0]["text"] == "Hello there."
    assert len(segs) >= 3
    for s in segs:
        assert all(len(line) <= 40 + 12 for line in s["text"].split("\n")), s["text"]
        assert len(s["text"].split("\n")) <= 2
    assert all(a["end"] <= b["start"] + 1e-6 or a["end"] - b["start"] < 0.9 for a, b in zip(segs, segs[1:]))


def test_pause_splits_and_srt_vtt():
    w = words("one two three four five", gap_after={2: 1.5})
    segs = C.words_to_segments(w, sentences=False)
    assert [s["text"] for s in segs] == ["one two three", "four five"]
    srt, vtt = C.to_srt(segs), C.to_vtt(segs)
    assert srt.startswith("1\n00:00:00,000 --> ") and "\n\n2\n" in srt
    assert vtt.startswith("WEBVTT") and "00:00:00.000 -->" in vtt


def test_thai_words_without_spaces_wrap_between_words():
    w = [{"text": t, "start": i * 0.2, "end": i * 0.2 + 0.2} for i, t in enumerate(["สวัสดี", "ครับ", "วันนี้", "เรา", "จะ", "คุยกัน", "เรื่อง", "การ", "ทำ", "วิดีโอ"] * 3)]
    segs = C.words_to_segments(w, max_chars=20, max_lines=2)
    assert len(segs) >= 2
    joined = "".join(s["text"].replace("\n", "") for s in segs)
    assert joined == "".join(x["text"] for x in w)


def test_clean_words_fixes_missing_and_backwards_times():
    out = C.clean_words([{"text": " a", "start": 1.0, "end": None}, {"text": " b", "start": 0.2, "end": 0.4}, {"text": " ", "start": 0, "end": 1}])
    assert len(out) == 2 and out[0]["end"] > out[0]["start"] and out[1]["start"] >= out[0]["end"] - 0.3


def test_dtw_monotonic():
    import numpy as np
    cost = np.ones((3, 8))
    for i, j in ((0, 1), (1, 4), (2, 6)):
        cost[i, j] = 0
    ti, tj = C.dtw(cost)
    assert list(ti) == sorted(ti) and list(tj) == sorted(tj) and ti[0] == 0 and tj[-1] == 7


# ------------------------------------------------------------------ dubbing math
def test_fit_plan_uses_gap_then_speeds_up_within_limit():
    segs = [{"start": 0.0, "end": 2.0}, {"start": 4.0, "end": 6.0}, {"start": 6.2, "end": 8.0}]
    p = C.fit_plan(segs, [1.5, 3.0, 3.0], total=10.0, max_speedup=1.3)
    assert p[0]["rate"] == 1.0                      # shorter than the slot: natural pace
    assert p[1]["room"] == pytest.approx(2.12, abs=0.01) and p[1]["rate"] == 1.3 and p[1]["overflow"] > 0   # 3.0 s in 2.12 s: capped
    g = C.fit_plan([{"start": 0.0, "end": 1.0}, {"start": 4.0, "end": 5.0}], [2.5, 1.0], total=8.0)
    assert g[0]["rate"] == 1.0 and g[0]["out_dur"] == 2.5       # longer than its slot but fits in the silence after it


def test_fit_plan_reports_overflow():
    p = C.fit_plan([{"start": 0.0, "end": 1.0}, {"start": 1.2, "end": 2.0}], [6.0, 1.0], total=5.0, max_speedup=1.2)
    assert p[0]["rate"] == 1.2 and p[0]["overflow"] > 3


def test_atempo_chain_limits():
    assert C.atempo_chain(1.3) == "atempo=1.30000"
    assert C.atempo_chain(3.0).count("atempo") == 2
    assert C.atempo_chain(0.4).startswith("atempo=0.5")


def test_speech_ref_window_prefers_9_seconds_of_adjacent_lines():
    segs = [{"start": 0, "end": 3, "text": "a"}, {"start": 3.2, "end": 6, "text": "b"}, {"start": 6.3, "end": 9.5, "text": "c"},
            {"start": 30, "end": 31, "text": "far"}]
    a, b, t = C.speech_ref_window(segs)
    assert 6 <= b - a <= 12 and "far" not in t


def test_language_mapping():
    assert C.norm_lang("Japanese") == "ja" and C.norm_lang("zh-cn") == "zh" and C.norm_lang("auto") is None
    assert C.norm_lang(False) == "no"            # YAML parses the Norwegian code `no` as False
    m = {"params": [{"name": "language", "type": "choice", "choices": ["Auto", "Chinese", "Japanese", "Thai"]}]}
    assert C.language_value(m, "ja") == "Japanese" and C.language_value(m, "th") == "Thai" and C.language_value(m, "ko") is None
    assert C.model_languages(m) == ["zh", "ja", "th"]
    assert C.model_languages({"features": {"languages": ["en", "zh-cn"]}, "params": []}) == ["en", "zh"]
    assert C.language_value({"params": [{"name": "language", "type": "choice", "choices": ["zh", "zh-cn"]}]}, "zh") == "zh"


def test_refine_edges_sentence_parked_in_silence():
    """A sentence the aligner parked early inside a pause (and overlapping the previous line's tail) starts at the sound."""
    np = pytest.importorskip("numpy")
    import media_audio_worker as W
    sr = 16000
    y = np.zeros(sr * 8, dtype=np.float32)
    rng = np.random.default_rng(0)
    y[sr * 0: int(sr * 2.0)] = rng.normal(0, 0.1, int(sr * 2.0))              # sentence 1: 0-2 s
    y[int(sr * 2.0): int(sr * 2.3)] = rng.normal(0, 0.001, int(sr * 0.3))    # its decaying tail
    y[sr * 5: sr * 7] = rng.normal(0, 0.1, sr * 2)                            # sentence 2: 5-7 s
    words = [{"text": " One", "start": 0.0, "end": 1.0}, {"text": " done.", "start": 1.0, "end": 2.4},
             {"text": " Two", "start": 2.2, "end": 4.5}, {"text": " here.", "start": 4.5, "end": 7.0}]
    out = W.refine_edges(words, y)
    assert 4.8 <= out[2]["start"] <= 5.05          # snapped from 2.2 (the tail) to the sound at 5 s
    assert out[1]["end"] <= 2.2                    # the first sentence ends with its sound, not the aligner's 2.4
    assert out[3]["end"] <= 7.2
