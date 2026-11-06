"""Pure helpers of the media-audio tools (subtitles, stem separation, dubbing). Standard library only.

Imported by `workers/media_audio_worker.py` (envs/media_audio), by the dubbing handler `app/plugins/media_audio.py`
(envs/core, loaded by file path) and by `tests/test_media_audio*.py`. No numpy, no torch.

  srt_time / vtt_time / to_srt / to_vtt    subtitle files
  words_to_segments                        word timestamps -> cues (length / pause / punctuation limits, line wrapping)
  fit_plan                                 time-fit of dubbed lines into the original slots (speed-up limits)
  language_value                           map a language code to the value a TTS model's `language` param expects
"""
from __future__ import annotations

import math
import re

NO_SPACE = re.compile(r"[฀-๿぀-ヿ㐀-鿿豈-﫿가-힯຀-໿က-႟ក-៿]")
SENT_END = re.compile(r"[.!?。！？…]+[\"'”’)\]]*$")
CLAUSE_END = re.compile(r"[,;:、，；：—]+[\"'”’)\]]*$")

NO_SPACE_LANGS = {"th", "ja", "zh", "lo", "my", "km", "yue"}
LANG_NAMES = {"en": "English", "th": "Thai", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "fr": "French",
              "de": "German", "es": "Spanish", "it": "Italian", "ru": "Russian", "pt": "Portuguese", "vi": "Vietnamese",
              "id": "Indonesian", "ar": "Arabic", "hi": "Hindi", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
              "sv": "Swedish", "da": "Danish", "fi": "Finnish", "el": "Greek", "he": "Hebrew", "ms": "Malay",
              "sw": "Swahili", "cs": "Czech", "hu": "Hungarian", "no": "Norwegian", "uk": "Ukrainian", "ca": "Catalan"}
NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}
CODE_ALIASES = {"zh-cn": "zh", "zh-tw": "zh", "cmn": "zh", "en-us": "en", "en-gb": "en", "fr-fr": "fr", "pt-br": "pt",
                "jp": "ja", "chinese": "zh", "mandarin": "zh"}


def norm_lang(v) -> str | None:
    """'Japanese' / 'ja' / 'zh-cn' / 'en-US' -> 'ja' / 'ja' / 'zh' / 'en' (None when unknown)."""
    if v is None or v is False:   # YAML turns the code `no` into False
        return "no" if v is False else None
    s = str(v).strip().lower()
    if not s or s == "auto":
        return None
    s = CODE_ALIASES.get(s, s)
    if s in LANG_NAMES:
        return s
    if s in NAME_TO_CODE:
        return NAME_TO_CODE[s]
    base = s.split("-")[0].split("_")[0]
    return base if base in LANG_NAMES else None


def model_languages(model: dict) -> list[str]:
    """Language codes a TTS registry entry can speak: its `language` param choices, else `features.languages`."""
    out: list[str] = []
    for p in model.get("params") or []:
        if p.get("name") == "language" and p.get("type") == "choice":
            for c in p.get("choices") or []:
                n = norm_lang(c["value"] if isinstance(c, dict) else c)
                if n and n not in out:
                    out.append(n)
    if not out:
        langs = (model.get("features") or {}).get("languages")
        if isinstance(langs, list):
            for c in langs:
                n = norm_lang(c)
                if n and n not in out:
                    out.append(n)
    return out


def language_value(model: dict, code: str):
    """The value to send as `params.language` to a TTS model for the language `code` (None = model has no such param)."""
    code = norm_lang(code) or code
    for p in model.get("params") or []:
        if p.get("name") == "language" and p.get("type") == "choice":
            choices = [c["value"] if isinstance(c, dict) else c for c in p.get("choices") or []]
            best = None
            for c in choices:
                if norm_lang(c) == code:
                    # prefer the exact short code when the model lists several variants (zh / zh-cn)
                    if best is None or str(c).lower() == code:
                        best = c
            return best
    return None


# ------------------------------------------------------------------ subtitle files
def srt_time(t: float) -> str:
    ms = max(0, int(round(float(t) * 1000)))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(t: float) -> str:
    return srt_time(t).replace(",", ".")


def to_srt(cues: list[dict]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text'].strip()}\n")
    return "\n".join(out)


def to_vtt(cues: list[dict]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{vtt_time(c['start'])} --> {vtt_time(c['end'])}\n{c['text'].strip().replace('-->', '->')}\n")
    return "\n".join(out)


def _join(words: list[dict]) -> str:
    return "".join(w["text"] for w in words).strip()


def wrap_words(words: list[dict], max_chars: int, max_lines: int = 2) -> str:
    """Join the words of one cue into at most `max_lines` balanced lines, breaking only between words."""
    text = _join(words)
    if max_lines <= 1 or len(text) <= max_chars or len(words) < 2:
        return text
    lines_n = min(max_lines, max(2, math.ceil(len(text) / max_chars)))
    # cumulative character count at each word boundary
    pos, acc = [], 0
    for w in words:
        acc += len(w["text"])
        pos.append(acc)
    total = pos[-1]
    cuts, prev = [], 0
    for k in range(1, lines_n):
        goal = total * k / lines_n
        best = min(range(prev, len(words) - 1), key=lambda i: abs(pos[i] - goal), default=None)
        if best is None:
            break
        cuts.append(best + 1)
        prev = best + 1
    lines, a = [], 0
    for c in cuts + [len(words)]:
        if c > a:
            lines.append(_join(words[a:c]))
        a = c
    return "\n".join(l for l in lines if l)


def clean_words(words: list[dict], duration: float | None = None) -> list[dict]:
    """Make word timestamps monotonic and complete (Whisper can return end=None for the last word)."""
    out, last = [], 0.0
    for w in words:
        t = str(w.get("text", ""))
        if not t.strip():
            continue
        s = w.get("start")
        e = w.get("end")
        s = last if s is None else max(float(s), 0.0)
        if e is None:
            e = s + 0.3
        e = max(float(e), s + 0.02)
        if duration:
            s, e = min(s, duration), min(e, duration)
        if s < last - 0.3:      # timestamps jumped backwards: keep order
            s = last
            e = max(e, s + 0.02)
        out.append({"text": t, "start": round(s, 3), "end": round(e, 3)})
        last = out[-1]["end"]
    return out


def words_to_segments(words: list[dict], *, max_chars: int = 42, max_lines: int = 2, max_dur: float = 7.0,
                      min_dur: float = 0.9, pause: float = 0.7, sentences: bool = True) -> list[dict]:
    """Group timed words into cues.

    Phrases end at a pause longer than `pause` and (when `sentences`) at the end of a sentence. A phrase longer than
    `max_chars * max_lines` characters or `max_dur` seconds is cut into equal-looking cues, preferring clause
    punctuation and pauses as cut points. Cues shorter than `min_dur` are stretched into the following gap (never into
    the next cue). Returns [{start, end, text, words: [..]}] where `text` has the line breaks.
    """
    words = [w for w in words if str(w.get("text", "")).strip()]
    cap = max(8, max_chars * max_lines)
    phrases: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > pause:
            phrases.append(cur)
            cur = []
        cur.append(w)
        if sentences and SENT_END.search(w["text"].strip()):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    segs: list[list[dict]] = []
    for ph in phrases:
        segs.extend(_split_phrase(ph, cap, max_dur))
    # an orphan of one or two characters ("OK", a lone "." ) joins the previous cue when it fits
    merged: list[list[dict]] = []
    for s in segs:
        if merged and len(_join(s)) <= 3 and len(_join(merged[-1] + s)) <= cap and s[0]["start"] - merged[-1][-1]["end"] <= pause:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    out = []
    for k, s in enumerate(merged):
        start, end = s[0]["start"], s[-1]["end"]
        nxt = merged[k + 1][0]["start"] if k + 1 < len(merged) else None
        if end - start < min_dur:
            end = start + min_dur if nxt is None else min(start + min_dur, max(end, nxt - 0.05))
        out.append({"start": round(start, 3), "end": round(end, 3), "text": wrap_words(s, max_chars, max_lines),
                    "words": [{"text": w["text"], "start": w["start"], "end": w["end"]} for w in s]})
    return out


def _split_phrase(ph: list[dict], cap: int, max_dur: float) -> list[list[dict]]:
    total = len(_join(ph))
    dur = ph[-1]["end"] - ph[0]["start"]
    k = max(math.ceil(total / cap), math.ceil(dur / max_dur), 1)
    if k <= 1 or len(ph) < 2:
        return [ph]
    k = min(k, len(ph))
    pos, acc = [], 0
    for w in ph:
        acc += len(w["text"])
        pos.append(acc)                       # characters up to and including word i
    cuts, prev = [], 0
    for j in range(1, k):
        goal = total * j / k
        lo, hi = prev, len(ph) - 1 - (k - 1 - j)       # leave at least one word for each remaining piece
        best, best_score = None, 1e9
        for b in range(lo, hi):
            gap = ph[b + 1]["start"] - ph[b]["end"]
            bonus = (6 if CLAUSE_END.search(ph[b]["text"].strip()) else 0) + 10 * min(gap, 1.0)
            score = abs(pos[b] - goal) - bonus
            if score < best_score:
                best, best_score = b, score
        if best is None:
            break
        cuts.append(best + 1)
        prev = best + 1
    out, a = [], 0
    for c in cuts + [len(ph)]:
        if c > a:
            out.append(ph[a:c])
        a = c
    return out


# ------------------------------------------------------------------ dubbing time-fit
def fit_plan(segments: list[dict], durations: list[float], total: float | None = None, *, max_speedup: float = 1.35,
             min_gap: float = 0.08, max_slowdown: float = 1.0) -> list[dict]:
    """Decide where each dubbed line goes and how much it is sped up.

    segments[i] = {start, end} of the original line; durations[i] = length of the synthesized line (seconds, silence
    trimmed). A line may use the silence after its slot (up to the next line's start minus `min_gap`); beyond that it is
    sped up by at most `max_speedup` (ffmpeg atempo); what still does not fit overlaps the next line and is reported as
    `overflow` (seconds). Lines shorter than their slot keep their natural pace (`max_slowdown` < 1 would slow them,
    default off). Returns [{start, rate, out_dur, overflow}] with rate >= 1.0 meaning faster.
    """
    plan = []
    n = len(segments)
    for i, (seg, d) in enumerate(zip(segments, durations)):
        start = float(seg["start"])
        slot = max(float(seg["end"]) - start, 0.2)
        nxt = float(segments[i + 1]["start"]) if i + 1 < n else (total if total else start + max(d, slot) + 1.0)
        room = max(nxt - min_gap - start, slot)         # slot + the following silence
        d = max(float(d), 0.05)
        rate = 1.0
        if d > room:
            rate = min(d / room, max_speedup)
        elif max_slowdown < 1.0 and d < slot * 0.7:
            rate = max(max_slowdown, d / slot)
        out_dur = d / rate
        overflow = max(0.0, start + out_dur - (nxt - min_gap)) if i + 1 < n else max(0.0, start + out_dur - total) if total else 0.0
        plan.append({"start": round(start, 3), "rate": round(rate, 4), "out_dur": round(out_dur, 3),
                     "overflow": round(overflow, 3), "slot": round(slot, 3), "room": round(room, 3)})
    return plan


def atempo_chain(rate: float) -> str:
    """ffmpeg filter string for a speed change `rate` (each atempo stage is limited to 0.5 .. 2.0)."""
    parts = []
    r = float(rate)
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.5f}")
    return ",".join(parts)


def speech_ref_window(segments: list[dict], lo: float = 6.0, hi: float = 12.0, max_gap: float = 0.8) -> tuple[float, float, str]:
    """Pick the span of consecutive source lines (gaps <= max_gap) that is closest to 9 s long, within lo..hi seconds
    when possible. Returns (start, end, text): the reference clip of the original speaker and its transcript."""
    best, best_score = None, 1e9
    n = len(segments)
    for i in range(n):
        for j in range(i, n):
            if j > i and segments[j]["start"] - segments[j - 1]["end"] > max_gap:
                break
            s, e = float(segments[i]["start"]), float(segments[j]["end"])
            d = e - s
            if d > hi + 3:
                break
            score = abs(d - (lo + hi) / 2) + (0 if lo <= d <= hi else 3)
            if score < best_score:
                best, best_score = (s, min(e, s + hi), " ".join(str(x.get("text", "")).replace("\n", " ").strip() for x in segments[i:j + 1])), score
    if best is None and segments:
        s = segments[0]
        best = (float(s["start"]), min(float(s["end"]), float(s["start"]) + hi), str(s.get("text", "")))
    return best or (0.0, 0.0, "")


# ------------------------------------------------------------------ word alignment helpers (used by the worker)
def dtw(cost):
    """Monotonic alignment (Whisper's dtw): cost [N, M] numpy -> (text_indices, time_indices) of the best path."""
    import numpy as np

    n, m = cost.shape
    c = np.full((n + 1, m + 1), np.inf)
    c[0, 0] = 0.0
    tr = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cl = c.tolist()
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0, c1, c2 = cl[i - 1][j - 1], cl[i - 1][j], cl[i][j - 1]
            if c0 < c1 and c0 < c2:
                v, t = c0, 0
            elif c1 < c0 and c1 < c2:
                v, t = c1, 1
            else:
                v, t = c2, 2
            cl[i][j] = float(cost[i - 1, j - 1]) + v
            tr[i, j] = t
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        t = tr[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        elif t == 2:
            j -= 1
        else:
            break
    path = path[::-1]
    return np.array([p[0] for p in path]), np.array([p[1] for p in path])


def split_token_units(tok, token_ids: list[int], spaces: bool = True) -> list[tuple[str, list[int]]]:
    """Group Whisper text tokens into words: [(text with its leading space, [token ids])].

    Tokens are first merged until they decode to complete unicode characters; for space-delimited languages a group that
    starts with a space starts a new word and punctuation-only groups attach to the previous word; for Thai / Japanese /
    Chinese every complete group is a unit."""
    groups: list[tuple[str, list[int]]] = []
    cur: list[int] = []
    for t in token_ids:
        cur.append(t)
        txt = tok.decode(cur)
        if "\ufffd" not in txt:
            groups.append((txt, cur))
            cur = []
    if cur:
        groups.append((tok.decode(cur), cur))
    if not spaces:
        return groups
    words: list[tuple[str, list[int]]] = []
    for txt, ids in groups:
        if words and not txt.startswith(" "):        # sub-word piece or punctuation: continues the previous word
            words[-1] = (words[-1][0] + txt, words[-1][1] + ids)
        else:
            words.append((txt, list(ids)))
    return words


def interpolate_words(segs: list[dict], duration: float) -> list[dict]:
    """Fallback word times: spread each segment's characters evenly over its time span."""
    out = []
    for sg in segs:
        s = float(sg["start"] or 0.0)
        e = float(sg["end"]) if sg.get("end") is not None else min(duration, s + 5.0)
        text = sg["text"]
        parts = re.findall(r"\s*\S+", text) if " " in text.strip() else list(text)
        tot = sum(len(p) for p in parts) or 1
        t = s
        for p in parts:
            d = (e - s) * len(p) / tot
            out.append({"text": p, "start": t, "end": t + d})
            t += d
    return out
