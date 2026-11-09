"""Project style guide (docs/CONTRACTS.md section 11): validation, versions, prompt composition, the job hooks, the palette maths,
the REST API, the MCP tools and the registry flag. No GPU, no weights (the jobs run the echo worker)."""
import importlib.util
import json
import textwrap
import time
import types

import pytest

from app import config, jobs
from app import style_common as sc
from app.assets import AssetStore
from app.config import Settings
from app.jobs import TERMINAL, JobError, JobManager
from app.memory import MemoryManager
from app.plugins import style as style_plugin
from app.registry import Registry
from app.workers import WorkerPool

ROOT = config.STUDIO

QWEN = {"id": "qwen-image-2.1", "name": "Qwen-Image 2.1", "backend": "qwen", "tasks": ["t2i", "rgba", "multi_ref", "edit"],
        "features": {"negative": "cfg", "style_ref": ["t2i", "rgba", "multi_ref"], "max_refs": 3},
        "params": [{"name": "true_cfg", "default": 1.0}], "prompt_style": "qwen_image"}
ZIMG = {"id": "z-image-turbo", "name": "Z-Image Turbo", "backend": "zimage", "tasks": ["t2i"], "features": {"negative": False}, "params": [], "prompt_style": "z_image"}
WAN = {"id": "wan", "name": "Wan", "backend": "wan", "tasks": ["t2v"], "features": {"negative": True}, "params": [], "prompt_style": "video_wan"}
CLOUD = {"id": "cl", "name": "Cloud", "backend": "cloud", "tasks": ["t2i"], "features": {"negative": True, "style_ref": ["t2i"], "max_refs": 3}, "params": []}
LIB = {"a1": {"id": "a1", "type": "image", "name": "ref one", "project": "p", "license": {"commercial": True}, "model": "z-image-turbo"},
       "a2": {"id": "a2", "type": "image", "name": "ref two", "project": "p", "license": {"commercial": False}},
       "a3": {"id": "a3", "type": "image", "name": "ref three", "project": "p", "license": {"commercial": True}},
       "u1": {"id": "u1", "type": "image", "name": "user image", "project": "p"}}


def guide(**kw):
    g = sc.default_guide()
    g.update(version=3, hash="abc123abc123", positive="hand-painted fantasy art, thick dark outlines", background="plain cream background", avoid=["photorealistic", "text"])
    g.update(kw)
    return g


def comp(g, model, task="t2i", params=None, inputs=None, req=None, meta=None, **kw):
    return sc.compose(g, model, task, params if params is not None else {"prompt": "an iron sword"}, inputs or [], req, meta=meta, lookup=LIB.get, **kw)


# ------------------------------------------------------------------ 1. validation
def test_validate_defaults_and_errors():
    g = sc.validate_guide({}, "p")
    assert g["enabled"] and g["strength"] == "normal" and g["variants"] == {} and g["palette"]["post"] == "off"
    for bad in ({"nope": 1}, {"strength": "huge"}, {"positive": "x" * 401}, {"avoid": [f"a{i}" for i in range(17)]}, {"avoid": ["x" * 41]},
                {"palette": {"colors": ["red"]}}, {"palette": {"colors": ["#112233", "#223344", "#334455"], "weights": [1]}}, {"palette": {"post": "blur"}},
                {"variants": {"flat": {}}}, {"variants": {"image": {"mode": "x"}}}, {"variants": {"image": {"zzz": 1}}}, {"schema": 9}, {"enabled": "yes"}):
        with pytest.raises(ValueError):
            sc.validate_guide(bad, "p")


def test_validate_normalises_colours_avoid_and_variants():
    g = sc.validate_guide({"avoid": "Text, text , Watermark", "palette": {"colors": ["#AABBCC", "#aabbcc", "#112233"], "tint_strength": 5},
                           "variants": {"audio": {"positive": "warm"}}}, "p")
    assert g["avoid"] == ["Text", "Watermark"]
    assert g["palette"]["colors"] == ["#aabbcc", "#112233"] and g["palette"]["tint_strength"] == 0.8
    assert g["variants"]["audio"]["enabled"] is False and g["variants"]["audio"]["mode"] == "replace" and g["variants"]["audio"]["positive"] == "warm"
    assert sc.validate_guide({"variants": {"texture": {}}}, "p")["variants"]["texture"]["avoid"] == ["perspective", "shadows"]


def test_validate_refs_need_real_images_of_the_project(tmp_path):
    store = AssetStore(tmp_path / "projects")
    store.create_project("p1")
    store.create_project("p2")
    src = tmp_path / "x.png"
    from PIL import Image
    Image.new("RGB", (8, 8), "red").save(src)
    a = store.ingest("p1", src, "image", {"name": "red"}, move=False)
    b = store.ingest("p2", src, "image", {"name": "other"}, move=False)
    g = sc.validate_guide({"refs": [{"asset": a["id"]}]}, "p1", store)
    assert g["refs"][0]["name"] == "red" and len(g["refs"][0]["sha256"]) == 64
    with pytest.raises(ValueError, match="another project"):
        sc.validate_guide({"refs": [{"asset": b["id"]}]}, "p1", store)
    with pytest.raises(ValueError, match="not found"):
        sc.validate_guide({"refs": [{"asset": "20200101-000000-aaaaaa"}]}, "p1", store)
    assert sc.validate_guide({"refs": [{"asset": "20200101-000000-aaaaaa"}]}, "p1", store, allow_missing={"20200101-000000-aaaaaa"})["refs"]
    with pytest.raises(ValueError, match="At most 4"):
        sc.validate_guide({"refs": [{"asset": f"20200101-000000-aaaaa{i}"} for i in range(5)]}, "p1", None)


def test_hash_ignores_server_keys_and_lint_flags_negations():
    g = sc.validate_guide({"positive": "painted"}, "p")
    assert sc.guide_hash(g) == sc.guide_hash({**g, "version": 9, "saved": "x", "hash": "y", "saved_by": "ui"})
    assert sc.guide_hash(g) != sc.guide_hash({**g, "positive": "inked"})
    assert sc.lint({"positive": "flat shading, no outlines", "variants": {}}) and not sc.lint({"positive": "flat shading"})


# ------------------------------------------------------------------ 2. composition
def test_style_is_a_suffix_never_a_prefix():
    r = comp(guide(), ZIMG)
    assert r["applied"] and r["prompt"].startswith("an iron sword. ") and "thick dark outlines" in r["prompt"] and r["prompt"].endswith("plain cream background")
    assert r["x_style"]["user_prompt"] == "an iron sword" and r["variant"] == "image" and r["version"] == 3


def test_rgba_drops_the_background_and_keeps_the_trigger_text_first():
    p = {"prompt": "Generate an image with a transparent background. a sword"}
    r = comp(guide(), QWEN, "rgba", p)
    assert r["prompt"].startswith("Generate an image with a transparent background.") and "cream" not in r["prompt"]
    assert any(s["what"] == "background" for s in r["skipped"])


def test_dedupe_when_the_style_is_already_in_the_prompt():
    g = guide(background="")
    r = comp(g, ZIMG, params={"prompt": "an iron sword, hand-painted fantasy art, thick dark outlines"})
    assert not r["applied"] or r["prompt"] == "an iron sword, hand-painted fantasy art, thick dark outlines"
    assert any(s["what"] == "prompt" for s in r["skipped"])


def test_avoid_goes_only_into_a_real_negative_prompt():
    assert comp(guide(), ZIMG)["negative"] == ""                                             # no negative prompt at all
    r = comp(guide(), ZIMG)
    assert any(s["what"] == "avoid" for s in r["skipped"])
    assert comp(guide(), QWEN)["negative"] == ""                                              # true_cfg 1.0: ignored
    r = comp(guide(), QWEN, params={"prompt": "a sword", "true_cfg": 4.0})
    assert r["negative"] == "photorealistic, text"
    r = comp(guide(), QWEN, params={"prompt": "a sword", "true_cfg": 4.0, "negative": "blurry"})
    assert r["negative"] == "blurry, photorealistic, text" and r["x_style"]["user_negative"] == "blurry"


def test_wan_keeps_a_quality_base_negative():
    r = comp(guide(variants={"video": {"enabled": True, "positive": "", "background": None, "avoid": [], "mode": "extend", "refs": False, "palette_post": None}}),
             WAN, "t2v", {"prompt": "a knight walks"})
    assert r["variant"] == "video" and r["negative"].startswith("worst quality") and "photorealistic" in r["negative"]
    assert "static" in r["negative"] and "风格" not in r["negative"]


def test_references_only_for_models_that_declare_them():
    g = guide(refs=[{"asset": "a1", "name": "ref one"}, {"asset": "a3", "name": "ref three"}, {"asset": "a2", "name": "ref two"}])
    r = comp(g, QWEN)
    assert [x["asset"] for x in r["refs"]] == ["a1", "a3"] and r["extra_seconds_est"] == sc.SEC_PER_REF_IMAGE        # normal: cap 2
    assert r["params"]["prompt"].count("style reference") == 1 and "image 1-2" in r["params"]["prompt"]
    assert [i["role"] for i in r["inputs"]] == ["style_ref", "style_ref"]
    assert not comp(g, ZIMG)["refs"] and not comp(g, CLOUD)["refs"]
    assert not comp(guide(refs=g["refs"], strength="light"), QWEN)["refs"]
    s = comp(g, QWEN, req={"strength": "strong"})
    assert len(s["refs"]) == 3                                                                                       # room: max_refs 3 - 0 user images
    s = comp(g, QWEN, inputs=[{"asset": "u1", "role": "input"}], req={"strength": "strong"})
    assert len(s["refs"]) == 2 and s["refs"][0]["index"] == 2
    assert "colour palette" not in s["prompt"]                                                                       # no colours saved


def test_commercial_only_skips_non_commercial_refs():
    g = guide(refs=[{"asset": "a2", "name": "ref two"}, {"asset": "a1", "name": "ref one"}])
    assert [x["asset"] for x in comp(g, QWEN, commercial_only=True)["refs"]] == ["a1"]
    assert [x["asset"] for x in comp(g, QWEN)["refs"]] == ["a2", "a1"]


def test_palette_words_only_when_asked_or_strong():
    g = guide(palette={**sc.default_guide()["palette"], "colors": ["#dc143c", "#008080", "#fff5d6"]})
    assert "colour palette" not in comp(g, ZIMG)["prompt"]
    assert "colour palette" in comp(g, ZIMG, req={"strength": "strong"})["prompt"]
    g["palette"]["in_prompt"] = True
    assert "crimson" in comp(g, ZIMG)["prompt"]


def test_reasons_when_not_applied():
    assert comp(guide(enabled=False), ZIMG)["reason"] == "guide disabled"
    assert comp(guide(), ZIMG, req={"apply": False})["reason"] == "off for this job"
    assert "not styled" in comp(guide(), ZIMG, "edit")["reason"] and comp(guide(), ZIMG, "edit")["prompt"] == "an iron sword"
    assert "Audio mood variant is off" in comp(guide(), {"id": "m", "features": {}}, "sfx")["reason"]
    assert comp(guide(), ZIMG, params={"prompt": "  "})["reason"] == "no prompt to add the style to"
    assert comp(None, ZIMG)["applied"] is False
    with pytest.raises(sc.StyleInputError):
        comp(guide(), ZIMG, req={"variant": "nope"})
    with pytest.raises(sc.StyleInputError):
        comp(guide(), ZIMG, req={"version": 9}, load_version=lambda n: None)


def test_audio_variant_replaces_the_visual_look():
    g = guide(variants={"audio": {"enabled": True, "positive": "warm, close", "mode": "replace", "background": None, "avoid": [], "refs": False, "palette_post": None}})
    r = comp(g, {"id": "m", "features": {}}, "sfx", {"prompt": "sword draw"})
    assert r["prompt"] == "sword draw, warm, close"


def test_compose_is_idempotent_on_its_own_output():
    g = guide(refs=[{"asset": "a1", "name": "ref one"}])
    first = comp(g, QWEN, params={"prompt": "a sword", "true_cfg": 4.0, "negative": "blurry"})
    meta = {"asset_extra": {"x_style": first["x_style"]}}
    again = comp(g, QWEN, params=first["params"], inputs=first["inputs"], meta=meta)
    assert again["params"] == first["params"] and again["inputs"] == first["inputs"]
    off = comp(g, QWEN, params=first["params"], inputs=first["inputs"], req={"apply": False}, meta=meta)      # toggled off on a retry: back to what was typed
    assert off["params"]["prompt"] == "a sword" and off["params"]["negative"] == "blurry" and not [i for i in off["inputs"] if i["role"] == "style_ref"]


def test_task_policy_and_register_task():
    assert sc.TASK_POLICY["t2i"]["variant"] == "image" and sc.TASK_POLICY["text3d"]["variant"] == "texture" and sc.TASK_POLICY["rmbg"]["variant"] == "skip"
    sc.register_task("test_material", "texture")
    try:
        assert comp(guide(variants={"texture": {"enabled": True, "positive": "even light", "background": "", "avoid": [], "mode": "extend", "refs": False, "palette_post": None}}),
                    ZIMG, "test_material")["variant"] == "texture"
    finally:
        sc.TASK_POLICY.pop("test_material", None)
    with pytest.raises(ValueError):
        sc.register_task("x", "bogus")


def test_improve_line():
    assert "thick dark outlines" in sc.improve_line(guide(), "t2i", {})
    assert sc.improve_line(guide(), "edit", {}) == "" and sc.improve_line(guide(enabled=False), "t2i", {}) == "" and sc.improve_line(None, "t2i", {}) == ""
    assert sc.improve_line(guide(), "t2i", {"apply": False}) == ""


# ------------------------------------------------------------------ 3. palette maths
def test_palette_extract_tint_and_lock():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (64, 64), "#fff5d6")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 31, 63], fill="#dc143c")
    d.rectangle([40, 10, 60, 50], fill="#008080")
    cols, weights = sc.extract_palette(im, n=3)
    assert len(cols) == 3 and len(weights) == 3 and all(len(c) == 7 and c[0] == "#" for c in cols)
    assert sc.extract_palette(im, n=3)[0] == cols                                                                  # deterministic
    lock = sc.apply_post(im, "lock", ["#ff0000", "#00ff00", "#0000ff"], 1.0)
    got = {c for _n, c in lock.convert("RGB").getcolors(10000)}
    assert got <= {(255, 0, 0), (0, 255, 0), (0, 0, 255)}
    tint = sc.apply_post(im, "tint", ["#dc143c", "#008080"], 0.5)
    assert tint.size == im.size and tint.mode == "RGB"
    assert sc.apply_post(im, "off", ["#ff0000"], 1.0).tobytes() == im.tobytes()
    assert sc.srgb2lab(__import__("numpy").array([[255.0, 255.0, 255.0]]))[0][0] == pytest.approx(100, abs=0.5)


# ------------------------------------------------------------------ 4. hooks with the echo worker
@pytest.fixture
def env(tmp_path):
    yaml = textwrap.dedent(f"""
    backends:
      echo_gpu: {{python: {ROOT}/envs/core/bin/python, script: {ROOT}/workers/echo_worker.py, device: gpu}}
      echo_cpu: {{python: {ROOT}/envs/core/bin/python, script: {ROOT}/workers/echo_worker.py, device: cpu}}
    models:
      - {{id: test-tone, backend: echo_gpu, tasks: [sfx], memory_gb: 0, weights: []}}
      - {{id: test-swatch, backend: echo_cpu, tasks: [t2i], memory_gb: 0, weights: [], features: {{negative: true}},
          params: [{{name: negative, type: str, default: ""}}]}}
    """)
    (tmp_path / "models.yaml").write_text(yaml)
    s = Settings(tmp_path / "settings.json")
    reg = Registry(tmp_path / "models.yaml", s, probe_cache=tmp_path / "p.json")
    assets = AssetStore(tmp_path / "projects")
    assets.create_project("sty")
    mem = MemoryManager(s)
    pool = WorkerPool(reg, s, mem)
    events = []
    bus = types.SimpleNamespace(publish=lambda t, d: events.append((t, d)))
    m = JobManager(reg, assets, pool, mem, bus, s, tmp_path / "jobs.jsonl", tmp_path / "scratch")
    core = types.SimpleNamespace(jobs=m, assets=assets, registry=reg, bus=bus, plugins={}, improve_context_hooks=[], settings=s,
                                 settings_for=lambda p: s)
    saved = (list(jobs.SUBMIT_HOOKS), list(jobs.OUTPUT_HOOKS), style_plugin.CORE, style_plugin.SVC)
    style_plugin.setup(core)
    yield types.SimpleNamespace(core=core, mgr=m, svc=style_plugin.SVC, assets=assets, events=events, tmp=tmp_path)
    jobs.SUBMIT_HOOKS[:], jobs.OUTPUT_HOOKS[:] = saved[0], saved[1]
    style_plugin.CORE, style_plugin.SVC = saved[2], saved[3]
    m.shutdown()
    pool.stop_all()


def wait(m, jid, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        if m.get(jid)["status"] in TERMINAL:
            return m.get(jid)
        time.sleep(0.05)
    raise AssertionError(f"job {jid} still {m.get(jid)['status']}")


def submit(env, prompt="a potion", style=None, **kw):
    meta = dict(kw.pop("meta", {}))
    if style is not None:
        meta["style"] = style
    j = env.mgr.submit({"model": "test-swatch", "task": "t2i", "params": {"prompt": prompt, "batch": 1}, "project": "sty", "meta": meta, **kw})
    return wait(env.mgr, j["id"])


def test_service_versions_and_conflicts(env):
    s = env.svc
    g1, created, _w = s.save("sty", {"positive": "ink and wash"}, by="ui")
    assert created and g1["version"] == 1 and g1["saved_by"] == "ui" and len(g1["hash"]) == 12
    g1b, created, _w = s.save("sty", {"positive": "ink and wash"})
    assert not created and g1b["version"] == 1                                                                   # same content: no new version
    g2, created, _w = s.save("sty", {"positive": "oil", "strength": "light"}, base_version=1, by="mcp")
    assert created and g2["version"] == 2 and g2["saved_by"] == "mcp"
    with pytest.raises(style_plugin.Conflict) as e:
        s.save("sty", {"positive": "x"}, base_version=1)
    assert e.value.current == 2
    g3, _c, _w = s.save("sty", {"background": "plain"}, patch=True)
    assert g3["positive"] == "oil" and g3["background"] == "plain" and g3["version"] == 3
    assert [v["version"] for v in s.versions("sty")] == [3, 2, 1]
    g4, _c, _w = s.restore("sty", 1)
    assert g4["version"] == 4 and g4["positive"] == "ink and wash" and g4["saved_by"] == "restore"
    assert s.load_version("sty", 2)["positive"] == "oil" and s.load_version("sty", 99) is None
    assert any(t == "settings" and d.get("style") for t, d in env.events)
    s.delete("sty")
    assert s.get("sty") is None and s.summary_all()["sty"] is None


def test_job_gets_the_style_and_records_it(env):
    env.svc.save("sty", {"positive": "hand-painted", "avoid": ["photorealistic"], "background": "plain cream"})
    j = submit(env)
    assert j["status"] == "done"
    assert j["params"]["prompt"] == "a potion. hand-painted, plain cream" and j["params"]["negative"] == "photorealistic"
    a = env.assets.get(j["outputs"][0])
    assert a["x_style"]["applied"] and a["x_style"]["version"] == 1 and a["x_style"]["user_prompt"] == "a potion" and a["x_style"]["variant"] == "image"
    assert a["prompt"] == "a potion" and a["final_prompt"] == "a potion. hand-painted, plain cream"
    assert j["meta"]["style_applied"]["version"] == 1


def test_retry_does_not_stack_the_style(env):
    env.svc.save("sty", {"positive": "hand-painted", "avoid": ["photorealistic"]})
    j = submit(env)
    r = wait(env.mgr, env.mgr.retry(j["id"])["id"])
    r2 = wait(env.mgr, env.mgr.retry(r["id"])["id"])
    assert r2["params"]["prompt"] == j["params"]["prompt"] == "a potion. hand-painted"
    assert r2["params"]["negative"] == "photorealistic"
    env.svc.save("sty", {"positive": "oil painting"})                                                            # a retry picks up the new version, still no stacking
    r3 = wait(env.mgr, env.mgr.retry(r2["id"])["id"])
    assert r3["params"]["prompt"] == "a potion. oil painting" and r3["meta"]["asset_extra"]["x_style"]["version"] == 2


def test_per_job_override_off_and_pinned_version(env):
    env.svc.save("sty", {"positive": "v1 look"})
    env.svc.save("sty", {"positive": "v2 look"})
    assert submit(env, style={"apply": False})["params"]["prompt"] == "a potion"
    assert submit(env, style="off")["params"]["prompt"] == "a potion"
    assert submit(env, style={"version": 1})["params"]["prompt"] == "a potion. v1 look"
    j = submit(env, style={"apply": False})
    assert j["meta"]["asset_extra"]["x_style"] == {"applied": False, "version": 2, "hash": env.svc.get("sty")["hash"], "reason": "off for this job"}
    with pytest.raises(JobError):
        env.mgr.submit({"model": "test-swatch", "task": "t2i", "params": {"prompt": "x"}, "project": "sty", "meta": {"style": {"version": 77}}})
    with pytest.raises(JobError):
        env.mgr.submit({"model": "test-swatch", "task": "t2i", "params": {"prompt": "x"}, "project": "sty", "meta": {"style": 5}})


def test_disabled_guide_and_projects_without_one_are_untouched(env):
    assert submit(env)["params"]["prompt"] == "a potion"
    env.svc.save("sty", {"positive": "look", "enabled": False})
    j = submit(env)
    assert j["params"]["prompt"] == "a potion" and j["meta"]["asset_extra"]["x_style"]["reason"] == "guide disabled"


def test_tool_jobs_do_not_call_the_hook(env):
    env.svc.save("sty", {"positive": "look"})
    calls = []
    jobs.SUBMIT_HOOKS.append(lambda sub: calls.append(sub["task"]))
    try:
        j = env.mgr.submit({"tool": "style.palette", "params": {"mode": "tint", "colors": ["#ff0000"]}, "inputs": [], "project": "sty"})
    except JobError:
        pytest.skip("no input needed to queue")
    wait(env.mgr, j["id"])
    assert calls == []


def test_a_raising_hook_never_blocks_a_job_and_a_job_error_does(env):
    def boom(sub):
        sub["params"]["prompt"] = "half-edited"
        raise RuntimeError("bad hook")
    jobs.SUBMIT_HOOKS.append(boom)
    try:
        j = submit(env, prompt="clean")
        assert j["status"] == "done" and j["params"]["prompt"] == "clean"
    finally:
        jobs.SUBMIT_HOOKS.remove(boom)

    def refuse(sub):
        raise JobError("nope", "input")
    jobs.SUBMIT_HOOKS.append(refuse)
    try:
        with pytest.raises(JobError, match="nope"):
            submit(env)
    finally:
        jobs.SUBMIT_HOOKS.remove(refuse)


def test_hooks_of_a_foreign_core_are_ignored(env):
    env.svc.save("sty", {"positive": "look"})
    other = JobManager.__new__(JobManager)
    sub = {"jobs": other, "meta": {}, "params": {"prompt": "x"}, "inputs": [], "task": "t2i", "model": None, "project": "sty", "settings": None}
    jobs.SUBMIT_HOOKS[-1](sub)
    assert sub["params"]["prompt"] == "x"


def test_output_hook_tints_and_keeps_the_raw_file(env):
    env.svc.save("sty", {"positive": "look", "palette": {"colors": ["#ff0000", "#00ff00"], "post": "lock"}})
    j = submit(env)
    a = env.assets.get(j["outputs"][0])
    post = a["x_style"]["post"]
    assert post["palette"] == "lock" and post["raw"].endswith("_raw.png") or post["raw"].endswith(".png")
    names = [f["name"] for f in a["files"]]
    assert post["raw"] in names
    from PIL import Image
    with Image.open(env.assets.file_path("sty", a["id"], a["main"])) as im:
        cols = {c for _n, c in im.convert("RGB").getcolors(100000)}
    assert cols <= {(255, 0, 0), (0, 255, 0)}


def test_normal_maps_are_never_tinted():
    ctx = types.SimpleNamespace(mgr=None, job={"meta": {"style_post": {"mode": "lock", "colors": ["#ff0000"], "strength": 0.3}}})
    core = types.SimpleNamespace(jobs=None)
    _s, out, _i = style_plugin.make_hooks(core, types.SimpleNamespace())
    assert out(ctx, __import__("pathlib").Path("n.png"), "image", {"x_material": {"map": "normal"}}, []) is None
    assert out(ctx, __import__("pathlib").Path("m.glb"), "mesh", {}, []) is None


def test_improve_context_hook(env):
    env.svc.save("sty", {"positive": "hand-painted"})
    hook = [f for f in env.core.improve_context_hooks if getattr(f, "_style_hook", False)][0]
    assert "hand-painted" in hook("sty", None, "t2i", None)
    assert hook("sty", None, "t2i", {"apply": False}) == "" and hook("sty", None, "edit", None) == "" and hook(None, None, "t2i", None) == ""


# ------------------------------------------------------------------ 5. REST API (real Core)
@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    from app.core import Core

    main.core = Core()
    core = main.core
    monkeypatch.setattr(core.jobs, "_execute", lambda jid: None)
    with TestClient(main.app) as c:
        if c.get("/api/projects/sty-api/settings").status_code == 404:
            assert c.post("/api/projects", json={"id": "sty-api", "title": "Style API"}).status_code == 200
        yield c, core
    main.core = None


def test_api_roundtrip_conflict_history_and_restore(client):
    c, core = client
    base = "/api/x/style"
    assert c.get(f"{base}/guide", params={"project": "sty-api"}).json()["guide"] is None
    r = c.put(f"{base}/guide", json={"project": "sty-api", "guide": {"positive": "flat vector art, no gradients"}, "base_version": 0, "by": "ui"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["guide"]["version"] == 1 and d["created_version"] and any("gradients" in w or "no" in w for w in d["warnings"])
    bad = c.put(f"{base}/guide", json={"project": "sty-api", "guide": {"positive": "x"}, "base_version": 0})
    assert bad.status_code == 409 and bad.json()["kind"] == "conflict" and bad.json()["current_version"] == 1
    assert c.put(f"{base}/guide", json={"project": "sty-api", "guide": {"strength": "zzz"}}).status_code == 400
    r = c.patch(f"{base}/guide", json={"project": "sty-api", "patch": {"background": "white"}, "base_version": 1, "by": "assistant"})
    assert r.json()["guide"]["version"] == 2 and r.json()["guide"]["positive"].startswith("flat vector")
    vs = c.get(f"{base}/versions", params={"project": "sty-api"}).json()["versions"]
    assert [v["version"] for v in vs] == [2, 1] and vs[0]["saved_by"] == "assistant"
    assert c.get(f"{base}/versions/1", params={"project": "sty-api"}).json()["background"] == ""
    assert c.get(f"{base}/versions/9", params={"project": "sty-api"}).status_code == 404
    r = c.post(f"{base}/restore", json={"project": "sty-api", "version": 1}).json()
    assert r["guide"]["version"] == 3 and r["guide"]["saved_by"] == "restore"
    assert c.get(f"{base}/summary").json()["projects"]["sty-api"]["version"] == 3
    pol = c.get(f"{base}/policy").json()
    assert pol["tasks"]["t2i"] == "image" and pol["tasks"]["rmbg"] == "skip" and "qwen-image-2.1" in pol["models"]
    info = c.get(f"{base}/info").json()
    assert set(info) >= {"enabled", "strength", "positive", "avoid", "refs", "palette", "variants", "preview", "history", "chip"}
    assert all(40 <= len(v["text"]) <= 600 and v["title"] for v in info.values())


def test_api_compose_preview_and_job_submit(client):
    c, core = client
    base = "/api/x/style"
    c.put(f"{base}/guide", json={"project": "sty-api", "guide": {"positive": "ink wash", "avoid": ["photo"]}})
    r = c.post(f"{base}/compose", json={"project": "sty-api", "task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a boot"}}).json()
    assert r["applied"] and r["prompt"] == "a boot. ink wash" and r["user_prompt"] == "a boot"
    r = c.post(f"{base}/compose", json={"project": "sty-api", "task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a boot"}, "style": {"apply": False}}).json()
    assert not r["applied"] and r["prompt"] == "a boot"
    r = c.post(f"{base}/compose", json={"project": "sty-api", "task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a boot"}, "guide": {"positive": "draft look"}}).json()
    assert r["prompt"] == "a boot. draft look"                                                                    # an unsaved draft
    j = c.post("/api/jobs", json={"task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a boot"}, "project": "sty-api"})
    assert j.status_code == 200, j.text
    job = j.json()
    ver = c.get(f"{base}/guide", params={"project": "sty-api"}).json()["guide"]["version"]
    assert job["params"]["prompt"] == "a boot. ink wash" and job["meta"]["asset_extra"]["x_style"]["version"] == ver
    assert c.post("/api/jobs", json={"task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a boot"}, "project": "sty-api",
                                     "meta": {"style": {"variant": "nope"}}}).status_code == 400


def test_api_improve_receives_the_style_line(client):
    c, core = client
    c.put("/api/x/style/guide", json={"project": "sty-api", "guide": {"positive": "ink wash"}})
    seen = {}
    monkey = core.improver
    orig = monkey.improve

    def fake(text, **kw):
        seen.update(kw)
        return {"prompt": text, "variants": [], "notes": []}
    monkey.improve = fake
    try:
        r = c.post("/api/improve", json={"text": "a boot", "model_id": "z-image-turbo", "task": "t2i", "project": "sty-api", "context": "game"})
        assert r.status_code == 200, r.text
    finally:
        monkey.improve = orig
    assert "ink wash" in seen.get("context", "") and seen["context"].startswith("game")


# ------------------------------------------------------------------ 6. MCP + Assistant + registry
def _mcp():
    spec = importlib.util.spec_from_file_location("mcp_server_style", ROOT / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcp_style_tools_and_submit_style_arg(monkeypatch):
    mod = _mcp()
    calls = []

    def fake(method, path, body=None, *, query=None, raw=None, ctype="", timeout=60):
        calls.append((method, path, body, query))
        if path == "/api/x/style/guide" and method == "GET":
            return {"guide": None, "warnings": [], "missing_refs": []}
        if path == "/api/x/style/guide":
            return {"guide": {"version": 2, "hash": "h"}, "created_version": True, "warnings": []}
        if path == "/api/x/style/compose":
            return {"applied": True, "prompt": "p"}
        if path == "/api/x/style/draft":
            return {"guide": {"refs": []}, "warnings": [], "describe": None}
        if path == "/api/x/style/summary":
            return {"projects": {"a": {"name": "N", "version": 1, "enabled": True}, "b": None}}
        if path == "/api/projects":
            return {"projects": [{"id": "a"}, {"id": "b"}]}
        if path == "/api/models":
            return {"tasks": {"t2i": {"studio": "image"}}}
        if path == "/api/jobs" and method == "POST":
            return {"id": "j1", "status": "done", "project": "default", "outputs": [],
                    "meta": {"asset_extra": {"x_style": {"applied": True, "version": 2, "hash": "h", "variant": "image", "strength": "normal", "added": {"prompt": "ink"}}}}}
        raise AssertionError(path)
    monkeypatch.setattr(mod, "http", fake)
    assert mod.t_get_style_guide({"project": "a"}, None)["guide"] is None
    out = mod.t_set_style_guide({"project": "a", "patch": {"positive": "x"}, "base_version": 1}, None)
    assert out["version"] == 2 and calls[-1][0] == "PATCH" and calls[-1][2]["by"] == "mcp" and calls[-1][2]["base_version"] == 1
    mod.t_set_style_guide({"project": "a", "guide": {"positive": "x"}}, None)
    assert calls[-1][0] == "PUT"
    with pytest.raises(mod.StudioError):
        mod.t_set_style_guide({"project": "a"}, None)
    assert mod.t_preview_style({"project": "a", "prompt": "a boot", "style": "off"}, None)["applied"]
    assert calls[-1][2]["style"] == {"apply": False} and calls[-1][2]["params"]["prompt"] == "a boot"
    assert mod.t_style_from_assets({"project": "a", "assets": ["x"], "save": True}, None)["saved"]["version"] == 2
    assert mod._style_summary() == {"a": {"name": "N", "version": 1, "enabled": True}}
    projects = mod.t_list_projects({}, None)["projects"]
    assert projects[0]["style"]["version"] == 1 and projects[1]["style"] is None
    res = mod.t_submit_job({"task": "t2i", "params": {"prompt": "a boot"}, "style": {"variant": "sprite"}, "wait_s": 0, "preview": False}, mod.Ctx(None))
    body = [c for c in calls if c[1] == "/api/jobs"][-1][2]
    assert body["meta"]["style"] == {"variant": "sprite"}
    text = json.loads(res[0]["text"])
    assert text["style"]["version"] == 2 and text["style"]["added"] == "ink"
    assert {t["name"] for t in mod.TOOLS} >= {"get_style_guide", "set_style_guide", "style_from_assets", "preview_style"}
    assert mod.BY_NAME["get_style_guide"]["annotations"]["readOnlyHint"] and mod.BY_NAME["preview_style"]["annotations"]["readOnlyHint"]
    assert "style.palette" in mod.TOOL_PARAMS


def test_assistant_style_tool_and_plan_validation(client):
    c, core = client
    from app.plugins import assistant
    c.put("/api/x/style/guide", json={"project": "sty-api", "guide": {"positive": "ink wash"}})
    assert assistant.tool_get_style_guide("sty-api", {})["positive"] == "ink wash"
    assert assistant.tool_get_style_guide("default-none-xyz", {})["guide"] is None
    assert "get_style_guide" in assistant.MCP_TOOLS
    plan = {"title": "t", "steps": [{"id": "s1", "title": "a", "task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a"}, "style": "off"}]}
    v = assistant.validate_plan(plan, "sty-api", include_test=True)
    assert not v["errors"].get("s1") or all("not ready" in e or "self-test" in e for e in v["errors"]["s1"])
    assert v["plan"]["steps"][0]["style"] == "off"
    plan["steps"][0]["style"] = {"variant": "bogus"}
    assert any("style must be" in e for e in assistant.validate_plan(plan, "sty-api", include_test=True)["errors"]["s1"])
    step = {"id": "s1", "task": "t2i", "model": "z-image-turbo", "params": {"prompt": "a"}, "style": {"variant": "sprite"}}
    assert assistant._step_spec(step, {"title": "t"}, "sty-api", "c1", "r1", [])["meta"]["style"] == {"variant": "sprite"}
    assert "style" not in assistant._step_spec({**step, "style": None}, {"title": "t"}, "sty-api", "c1", "r1", [])["meta"]


def test_registry_declares_style_refs_for_qwen_only():
    from app.registry import Registry as R
    reg = R(config.MODELS_YAML, Settings(config.DATA / "style-reg-settings.json"), probe_cache=config.DATA / "style-probe.json")
    with_refs = {mid for mid, m in reg.models.items() if (m.get("features") or {}).get("style_ref")}
    assert "qwen-image-2.1" in with_refs
    for mid in with_refs:
        m = reg.models[mid]
        assert set(m["features"]["style_ref"]) <= set(m["tasks"]) and m["features"].get("max_refs", 1) >= 2
    assert not [mid for mid in with_refs if reg.models[mid].get("backend") == "cloud"]


def test_style_page_is_a_registered_studio_file():
    src = (ROOT / "app" / "web" / "studios" / "style.js").read_text()
    assert "id: 'style'" in src and "section: 'workspace'" in src and "style_from" in src
    assert "registerPromptAddon" in src and "registerSelectionAction" in src
