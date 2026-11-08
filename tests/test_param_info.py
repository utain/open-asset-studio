"""Every parameter key of the merged registry has a beginner-friendly info text (app/param_info.py). No GPU, no weights."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.param_info import PARAM_INFO, PARAM_INFO_BY_TASK, param_info  # noqa: E402


def _usage() -> dict[str, set[str]]:
    """param key -> tasks of every model that has it, read from models.yaml + models.d/*.yaml (later files override by id)."""
    models: dict[str, dict] = {}
    for f in [ROOT / "models.yaml", *sorted((ROOT / "models.d").glob("*.yaml"))]:
        for m in (yaml.safe_load(f.read_text()) or {}).get("models") or []:
            if isinstance(m, dict) and m.get("id"):
                models[m["id"]] = m
    use: dict[str, set[str]] = {}
    for m in models.values():
        tasks = m.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        for p in m.get("params") or []:
            if isinstance(p, dict) and p.get("name"):
                use.setdefault(p["name"], set()).update(tasks)
    return use


def test_every_registry_param_is_covered():
    use = _usage()
    assert len(use) > 200
    missing = [(k, t) for k, tasks in sorted(use.items()) for t in sorted(tasks or {None}) if not param_info(k, t)]
    assert not missing, f"no info text for: {missing[:20]}"
    assert not [k for k in use if not PARAM_INFO.get(k)], "every key also needs a generic text"


def test_texts_are_sane():
    for key, text in [*PARAM_INFO.items(), *((f"{t}.{k}", v) for (t, k), v in PARAM_INFO_BY_TASK.items())]:
        assert isinstance(text, str) and text.strip(), key
        assert text == text.strip(), key
        assert 40 <= len(text) <= 600, f"{key}: {len(text)} chars"
        assert "\n" not in text, key


def test_task_specific_entries_refer_to_known_params():
    use = _usage()
    for (task, key) in PARAM_INFO_BY_TASK:
        assert key in PARAM_INFO, f"({task}, {key}) has no generic text"
        assert key in use, f"({task}, {key}): key is not used by any model"
        assert task in use[key], f"({task}, {key}): no model uses this key for that task"


def test_param_info_lookup():
    assert param_info("size", "t2i") == PARAM_INFO_BY_TASK[("t2i", "size")]
    assert param_info("size", "no_such_task") == PARAM_INFO["size"]
    assert param_info("seed") == PARAM_INFO["seed"]
    assert param_info("no_such_param") == ""
    assert param_info("no_such_param", "t2i") == ""
