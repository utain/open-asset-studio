"""Dubbing helpers of app/plugins/media_audio.py (no GPU, no workers; ffmpeg only where noted)."""
import json
import shutil

import numpy as np
import pytest

from app.jobs import JobError
from app.plugins import media_audio as M
from app.textgen import TextGenError


def test_parse_segments_validates_and_orders():
    segs = M.parse_segments(json.dumps([{"start": 5, "end": 6, "text": "b", "translation": "B"}, {"start": 0, "end": 2, "text": "a"}]))
    assert [s["text"] for s in segs] == ["a", "b"] and [s["id"] for s in segs] == [0, 1] and segs[1]["translation"] == "B"
    assert M.parse_segments("") == []
    for bad in ("{", "[]x", json.dumps([{"start": 3, "end": 2}]), json.dumps([{"text": "x"}])):
        with pytest.raises(JobError):
            M.parse_segments(bad)


class FakeTextgen:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def generate(self, *, messages, schema, backend, validate=None, **kw):
        if self.fail:
            raise self.fail
        user = messages[-1]["content"]
        items = json.loads(user[user.index("["):]) if "Lines to translate:\n" in user else []
        items = json.loads(user.split("Lines to translate:\n")[1])
        self.calls.append(len(items))
        data = {"lines": [{"id": i["id"], "text": "T:" + i["text"]} for i in items]}
        assert not validate(data)
        return {"data": data, "backend": {"label": "fake"}, "seconds": 0.1}


def test_translate_batches_and_keeps_existing():
    lines = [{"id": i, "start": i, "end": i + 1, "text": f"l{i}", "translation": "keep" if i == 2 else ""} for i in range(30)]
    tg = FakeTextgen()
    info = M.translate_lines(tg, lines, "English", "Thai", None, batch_size=12)
    assert tg.calls == [12, 12, 5] and info["translated"] == 29
    assert lines[2]["translation"] == "keep" and lines[0]["translation"] == "T:l0"


def test_translate_error_propagates():
    with pytest.raises(TextGenError):
        M.translate_lines(FakeTextgen(TextGenError("no llm", "unavailable")), [{"id": 0, "start": 0, "end": 1, "text": "x", "translation": ""}], "A", "B", None)


def test_trim_and_place():
    sr = 16000
    x = np.concatenate([np.zeros(sr), 0.5 * np.sin(np.linspace(0, 2000, sr)), np.zeros(sr)]).astype(np.float32)
    y = M.trim_silence(x, sr)
    assert 0.9 * sr < len(y) < 1.2 * sr
    out = M.place_lines([(0.5, y)], 3.0, sr)
    assert len(out) == 3 * sr and out[: sr // 2 - 10].max() == 0 and out.max() > 0.4


def test_duck_curve_lowers_only_under_speech():
    sr = 8000
    sp = np.zeros(sr * 4, dtype=np.float32)
    sp[sr:2 * sr] = 0.3
    c = M.duck_curve(sp, sr, 6.0)
    assert c[: sr // 2].min() > 0.99 and c[int(1.5 * sr)] < 0.55 and c[-1] > 0.99
    assert M.duck_curve(sp, sr, 0).min() == 1.0


def test_soft_limit_ceiling():
    x = np.linspace(-3, 3, 1000).astype(np.float32)
    assert np.abs(M.soft_limit(x, -1.0)).max() <= 10 ** (-1 / 20) + 1e-6


def test_normalize_loudness_private_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(M.media, "export_audio", None, raising=False)
    sr = 44100
    t = np.arange(sr * 3) / sr
    x = (0.05 * np.sin(2 * np.pi * 440 * t))[None, :].repeat(2, 0).astype(np.float32)
    src = tmp_path / "a.wav"
    M.write_wav(src, x, sr)
    out, rep = M.normalize_loudness(src, -16.0, tmp_path / "n", lambda m: None)
    assert rep["method"] == "gain+softlimit" and abs(rep["output_lufs"] - (-16.0)) < 1.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_ffmpeg_tempo_shortens():
    sr = 24000
    x = (0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 2) / sr)).astype(np.float32)
    y = M.ffmpeg_tempo(x, sr, 1.25)
    assert abs(len(y) / sr - 2 / 1.25) < 0.1
    assert M.ffmpeg_tempo(x, sr, 1.0) is x
