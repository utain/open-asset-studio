"""Sound post tools (critical: they rewrite the audio users ship) and the export naming plan."""
import numpy as np
import pytest

from app.plugins.sound import fade, limit, lufs_of, make_loop, normalize_lufs, plan_export, process, resample, trim_silence

SR = 44100


def tone(sec=2.0, f=220.0, amp=0.3, ch=2, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    x = (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return np.stack([x] * ch, axis=1)


def test_trim_keeps_pad_and_drops_silence():
    x = np.concatenate([np.zeros((SR, 2), np.float32), tone(1.0), np.zeros((SR // 2, 2), np.float32)])
    y = trim_silence(x, SR, threshold_db=-50, pad_ms=30)
    assert abs(len(y) / SR - (1.0 + 0.06)) < 0.01


def test_loop_is_seamless_and_shorter_by_the_crossfade():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3 * SR, 2)) * 0.1).astype(np.float32)
    y, L = make_loop(x, SR, 500)
    assert L == SR // 2 and len(y) == len(x) - L
    # end -> start continuity: the sample after y[-1] in the source is x[n-L], which is exactly y[0]
    assert np.allclose(y[0], x[len(x) - L], atol=1e-6)
    assert np.allclose(y[-1], x[len(x) - L - 1])


def test_normalize_hits_target_and_respects_ceiling():
    x = tone(3.0, amp=0.05)
    y, rep = normalize_lufs(x, SR, target=-16, ceiling_db=-1)
    assert abs(lufs_of(y, SR) - (-16)) < 0.6
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    loud = tone(3.0, amp=0.9)
    y2, _ = normalize_lufs(loud, SR, target=-6, ceiling_db=-1)
    assert np.abs(y2).max() <= 10 ** (-1 / 20) + 1e-6   # limiter caps peaks even when the gain pushes past 0 dBFS


def test_limiter_noop_below_ceiling():
    x = tone(1.0, amp=0.2)
    assert np.array_equal(limit(x, SR, -1), x)


def test_fade_edges_and_resample_length():
    y = fade(tone(1.0), SR, 50, 100)
    assert abs(y[0, 0]) < 1e-6 and abs(y[-1, 0]) < 1e-3
    r = resample(tone(1.0), SR, 22050)
    assert len(r) == 22050


def test_process_chain_order_and_channels():
    x = np.concatenate([np.zeros((SR // 2, 2), np.float32), tone(2.0)])
    y, sr, labels, rep = process(x, SR, {"trim": {}, "sample_rate": 48000, "channels": "mono", "normalize": {"lufs": -18}})
    assert sr == 48000 and y.shape[1] == 1
    assert labels[0] == "trim" and labels[-1] == "-18 LUFS"
    assert rep["trimmed_s"] > 0.4


def _a(i, name, task="sfx", rating=0, created="2026-11-05T10:00:0", **kw):
    return {"id": f"20261105-1000{i:02d}-aaaaaa", "type": "audio", "name": name, "task": task, "rating": rating,
            "created": f"{created}{i % 10}", **kw}


def test_export_plan_names_events_and_takes():
    metas = [_a(1, "Door slam #1"), _a(2, "Door slam #2", rating=5), _a(3, "x", x_event="UI Click"),
             _a(4, "Hero line", task="tts", x_sheet={"character": "Knight", "event": "kn_001"}),
             _a(5, "Boss theme · loop -14 LUFS", task="music")]
    plan = {p["asset"]["id"]: p["file"] for p in plan_export(metas, "ogg")}
    assert plan[metas[1]["id"]] == "sfx/door_slam_01.ogg"        # higher rating -> take 01
    assert plan[metas[0]["id"]] == "sfx/door_slam_02.ogg"
    assert plan[metas[2]["id"]] == "sfx/ui_click_01.ogg"
    assert plan[metas[3]["id"]] == "voice/knight/kn_001_01.ogg"
    assert plan[metas[4]["id"]] == "music/boss_theme_01.ogg"
    assert len(set(plan.values())) == len(plan)


@pytest.mark.parametrize("fmt", ["wav", "ogg", "mp3"])
def test_write_formats_roundtrip(tmp_path, fmt):
    import soundfile as sf

    from app.plugins.sound import write_audio

    p = write_audio(tmp_path / f"t.{fmt}", tone(1.0), SR, fmt)
    x, sr = sf.read(str(p), always_2d=True)
    assert sr == SR and abs(len(x) / sr - 1.0) < 0.1 and np.abs(x).max() > 0.1
