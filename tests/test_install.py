"""The installer (install.sh -> scripts/install.py): recipes, packs, torch index choice, lock filtering, dry run,
weights planning and download error handling. No GPU, no network, nothing is installed."""
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("studio_install", ROOT / "scripts" / "install.py")
inst = importlib.util.module_from_spec(_spec)
sys.modules["studio_install"] = inst
_spec.loader.exec_module(inst)

from app.registry import Registry  # noqa: E402

# Backends of the registry that have no environment to install (the installer knows why).
EXPECTED_MISSING = {"cloud", "comfyui", "music_levo"}


@pytest.fixture(scope="module")
def recipes():
    r, errors = inst.load_recipes()
    assert errors == []
    return r


@pytest.fixture(scope="module")
def registry():
    return Registry()


@pytest.fixture(scope="module")
def ctx(recipes, registry):
    c = inst.Ctx(dry_run=True, torch_index="cu130")
    c.recipes, c.reg, c.packs, c.sizes = recipes, registry, inst.load_packs(), inst.load_sizes()
    return c


# ------------------------------------------------------------------------------------------------ recipes
def test_every_recipe_parses_and_validates(recipes):
    files = {p.stem for p in (ROOT / "install").glob("*.yaml")} - {"packs", "sizes"}
    assert files == set(recipes), f"recipes that failed validation: {files - set(recipes)}"


def test_recipe_files_follow_format_md():
    """The keys install/FORMAT.md documents are the keys the validator accepts."""
    fmt = (ROOT / "install" / "FORMAT.md").read_text()
    for key in inst.RECIPE_KEYS:
        assert re.search(rf"^{key}:", fmt, re.M) or key in fmt, f"FORMAT.md does not mention '{key}'"


def test_registry_backends_have_a_recipe_or_are_expected_missing(recipes, registry):
    missing = {b for b in registry.backends if b not in recipes and b not in EXPECTED_MISSING}
    assert missing == set(), f"backends without install/<backend>.yaml: {sorted(missing)}"
    assert set(inst.EXPECTED_MISSING) <= EXPECTED_MISSING | set(inst.EXPECTED_MISSING)
    for b in EXPECTED_MISSING:
        assert b not in recipes or b in registry.backends


def test_recipe_env_matches_registry_python(recipes, registry):
    for bid, b in registry.backends.items():
        py = b.get("python")
        if bid not in recipes or not py or b.get("kind"):
            continue
        m = re.match(r"envs/([^/]+)/bin/python$", py)
        assert m, f"{bid}: unexpected python path {py}"
        assert recipes[bid]["env"] == m.group(1), f"{bid}: recipe env {recipes[bid]['env']} != registry {py}"


def test_shared_envs_point_at_a_real_recipe(recipes):
    for bid, r in recipes.items():
        sh = r["shares_env_with"]
        if sh:
            assert sh in recipes and recipes[sh]["env"] == r["env"] and not recipes[sh]["shares_env_with"]
            assert not r["lock"] and not r["torch"]


def test_every_lock_file_belongs_to_a_recipe_or_is_known(recipes):
    """An env-lock file nobody installs is probably a forgotten recipe."""
    used = {r["lock"] for r in recipes.values() if r["lock"]} | {"vllm_omni_overlay"}
    for p in (ROOT / "env-lock").glob("*.txt"):
        assert p.stem in used, f"env-lock/{p.name} is not referenced by any install recipe"


def test_pre_existing_backends_have_recipes(recipes):
    """The backends whose env-lock existed in the first commit."""
    first = ["audio", "clonea_chatterbox", "clonea_codec", "clonea_openvoice", "clonea_xtts", "clonea_zonos", "cloneb_amphion",
             "cloneb_hf", "cloneb_index", "core", "foley", "hy3d", "moss", "music_acestep", "music_audiocraft", "music_diffrhythm",
             "music_heartmula", "music_magenta", "music_yue", "omni_glm", "omni_kimi", "omni_llama2", "omni_minicpm", "omni_moshi",
             "omni_qwen", "omni_step", "qwen", "qwentts", "sfxmore", "ttslight", "ttslight_parler", "vc", "vc_diffsinger", "vllm_omni"]
    assert [b for b in first if b not in recipes] == []


@pytest.mark.parametrize("bad, fragment", [
    ({"backend": "x", "steps": ["sudo apt install foo"]}, "sudo"),
    ({"backend": "x", "steps": ["hf download org/repo"]}, "downloads model weights"),
    ({"backend": "x", "lock": "core", "mystery": 1}, "unknown key"),
    ({"backend": "x", "lock": "core", "check": "import ("}, "not valid python"),
    ({"backend": "x", "lock": "core", "lock_flags": ["no-deps"]}, "not a flag"),
    ({"backend": "x", "lock": "does-not-exist"}, "does not exist"),
    ({"backend": "x", "lock": "core", "vendor": ["no-such-repo-anywhere"]}, "matches nothing"),
    ({"backend": "x", "lock": "core", "python": "3.11"}, "3.12"),
    ({"backend": "x"}, "nothing to do"),
    ({"backend": "x", "shares_env_with": "core", "lock": "core"}, "also has its own"),
    ({"backend": "x", "lock": "core", "steps_linux-arm": ["true"]}, "steps_"),
    ({"backend": "y", "lock": "core"}, "does not match the file name"),
    ({"backend": "x", "lock": "core", "size_gb": "big"}, "size_gb"),
    ({"backend": "x", "lock": "core", "tested": ["windows"]}, "tested platform"),
    ({"backend": "x", "lock": "core", "torch": ["torch>=2"]}, "torch entry"),
])
def test_validate_recipe_rejects(bad, fragment):
    errs = inst.validate_recipe(bad, "x.yaml")
    assert any(fragment in e for e in errs), errs


def test_validate_recipe_accepts_minimal_and_manual_only():
    assert inst.validate_recipe({"backend": "x", "lock": "core"}, "x.yaml") == []
    assert inst.validate_recipe({"backend": "x", "manual": ["do it by hand"]}, "x.yaml") == []
    assert inst.validate_recipe({"backend": "x", "lock": "core", "steps_linux-aarch64": ["true"]}, "x.yaml") == []


def test_manual_only_recipes_are_marked_manual(recipes):
    assert inst.recipe_kind(recipes["vllm_omni"]) == "manual"
    assert inst.recipe_kind(recipes["llm"]) == "manual"
    assert inst.recipe_kind(recipes["moss_more"]) == "shared"
    assert inst.recipe_kind(recipes["qwen"]) == "env"


def test_recipe_vendor_filters_resolve_to_vendor_lock(recipes):
    for bid, r in recipes.items():
        for f in r["vendor"]:
            assert inst.vendor_matches([f]), f"{bid}: vendor filter {f}"


def test_locks_that_install_vendor_checkouts_list_them(recipes):
    """A lock with `-e ./vendor/X` needs the clone first (the recipe's `vendor:`)."""
    for bid, r in recipes.items():
        if not r["lock"]:
            continue
        for ln in (ROOT / "env-lock" / f"{r['lock']}.txt").read_text().splitlines():
            m = re.match(r"^(?:-e\s+)?\./(vendor/[^/\s]+)", ln.strip())
            if m:
                assert any(f.lower() in m.group(1).lower() for f in r["vendor"]), f"{bid}: {ln} needs a vendor entry"


# ------------------------------------------------------------------------------------------------ packs
def test_packs_refer_to_existing_models_and_backends(ctx):
    assert sum(1 for p in ctx.packs.values() if p.get("default")) == 1
    for name in ("starter", "image", "3d", "video", "sound", "voice", "music", "omni", "all"):
        assert name in ctx.packs
    for name, pk in ctx.packs.items():
        for inc in pk["include"]:
            assert inc in ctx.packs, f"{name}: include {inc}"
        for b in pk["backends"]:
            assert b in ctx.recipes, f"{name}: backend {b}"
        for mid in pk["models"]:
            assert mid in ctx.reg.models, f"{name}: unknown model {mid}"
            assert inst.model_problem(ctx, mid) is None, f"{name}: {mid}: {inst.model_problem(ctx, mid)}"
            assert inst.recipe_kind(ctx.recipes[ctx.reg.models[mid]["backend"]]) != "manual", f"{name}: {mid} needs a manual install"


def test_packs_other_than_all_hold_commercial_models_only(ctx):
    for name, pk in ctx.packs.items():
        if pk.get("everything"):
            continue
        for mid in pk["models"]:
            assert ctx.reg.models[mid]["license"].get("commercial"), f"{name}: {mid} is not commercial"


def test_starter_needs_no_login(ctx):
    models, _ = inst.pack_models(ctx, "starter")
    assert not [m for m in models if m in ctx.sizes["gated"]]
    assert 4 <= len(models) <= 8


def test_pack_expansion_and_all(ctx):
    models, _ = inst.pack_models(ctx, "all")
    assert len(models) > 60 and len(models) == len(set(models))
    assert all(inst.model_problem(ctx, m) is None for m in models)
    assert not any((ctx.reg.models[m].get("features") or {}).get("blocked") for m in models)


def test_pack_include_loop_is_an_error(ctx):
    ctx2 = SimpleNamespace(**ctx.__dict__)
    ctx2.packs = {"a": {"models": [], "backends": [], "include": ["b"]}, "b": {"models": [], "backends": [], "include": ["a"]}}
    with pytest.raises(inst.InstallError, match="loop"):
        inst.pack_models(ctx2, "a")


def test_sizes_cover_the_pack_models(ctx):
    for name, pk in ctx.packs.items():
        for mid in pk["models"]:
            hf = [w for w in ctx.reg.models[mid]["weights"] if "hf" in w]
            if hf:
                assert mid in ctx.sizes["weights_gb"], f"install/sizes.yaml has no size for {mid}"


# ------------------------------------------------------------------------------------------------ torch index / GPU
@pytest.mark.parametrize("cuda, idx", [("13.0", "cu130"), ("13.1", "cu130"), ("14.2", "cu130"), ("13", "cu130"),
                                       ("12.9", "cu128"), ("12.8", "cu128")])
def test_torch_index_for_cuda(cuda, idx):
    assert inst.torch_index_for_cuda(cuda) == idx


@pytest.mark.parametrize("cuda", ["12.7", "12.4", "11.8", "", None, "garbage"])
def test_torch_index_for_old_or_unknown_cuda_is_an_error(cuda):
    with pytest.raises(inst.InstallError):
        inst.torch_index_for_cuda(cuda)


def test_ctx_torch_index_option_and_validation():
    assert inst.Ctx(torch_index="cu128").torch_index() == "cu128"
    with pytest.raises(inst.InstallError):
        inst.Ctx(torch_index="cu118").torch_index()


def test_ctx_torch_index_from_nvidia_smi(monkeypatch):
    monkeypatch.delenv("STUDIO_TORCH_INDEX", raising=False)
    monkeypatch.setattr(inst, "detect_gpu", lambda: {"ok": True, "name": "X", "driver": "570", "cuda": "12.8"})
    assert inst.Ctx().torch_index() == "cu128"
    monkeypatch.setattr(inst, "detect_gpu", lambda: {"ok": True, "name": "X", "driver": "470", "cuda": "11.4"})
    with pytest.raises(inst.InstallError, match="12.8"):
        inst.Ctx().torch_index()
    monkeypatch.setattr(inst, "detect_gpu", lambda: {"ok": False, "error": "nvidia-smi not found"})
    with pytest.raises(inst.InstallError, match="nvidia-smi"):
        inst.Ctx().torch_index()
    assert inst.Ctx(dry_run=True).torch_index() == "cu130"   # a dry run on a machine without a GPU still shows the plan


def test_detect_gpu_parses_nvidia_smi(monkeypatch):
    smi = ("| NVIDIA-SMI 580.1  Driver Version: 580.1  CUDA Version: 13.0 |\n")

    def fake_run(cmd, **kw):
        if len(cmd) == 1:
            return SimpleNamespace(stdout=smi, stderr="", returncode=0)
        return SimpleNamespace(stdout="NVIDIA GB10, 580.1, [N/A], 12.1\n", stderr="", returncode=0)
    monkeypatch.setattr(inst.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(inst.subprocess, "run", fake_run)
    g = inst.detect_gpu()
    assert g["ok"] and g["cuda"] == "13.0" and g["driver"] == "580.1" and g["memory_gb"] is None and g["cc"] == "12.1"
    monkeypatch.setattr(inst.shutil, "which", lambda n: None)
    assert not inst.detect_gpu()["ok"]


# ------------------------------------------------------------------------------------------------ lock filtering
LOCK = """# comment
-e ./vendor/Foo
accelerate==1.0
nvidia-cublas==13.0.0.19
nvidia-cudnn-cu13==9.13.0.50
nvidia-ml-py==12.1
torch==2.9.1+cu130
torchaudio==2.9.1
torchcodec==0.17.0+cu130
triton==3.5.1
transformers==5.0.0
"""


def test_lock_is_untouched_on_cu130_and_for_envs_without_torch():
    assert inst.filter_lock(LOCK, "cu130", True) == LOCK
    assert inst.filter_lock(LOCK, "cu128", False) == LOCK


def test_lock_is_filtered_on_cu128():
    out = inst.filter_lock(LOCK, "cu128", True).splitlines()
    assert "torch==2.9.1+cu130" not in out and "torchaudio==2.9.1" not in out and "triton==3.5.1" not in out
    assert not any(ln.startswith(("nvidia-cublas", "nvidia-cudnn")) for ln in out)
    assert "nvidia-ml-py==12.1" in out       # not a torch dependency: kept
    assert "accelerate==1.0" in out and "-e ./vendor/Foo" in out and "# comment" in out
    assert "+cu" not in "\n".join(out)


def test_real_locks_keep_their_non_torch_packages_on_cu128():
    for p in (ROOT / "env-lock").glob("*.txt"):
        text = p.read_text()
        out = inst.filter_lock(text, "cu128", inst.lock_has_torch(text))
        if inst.lock_has_torch(text):
            assert not re.search(r"^(torch|torchaudio|torchvision|triton|nvidia-cudnn-cu13)==", out, re.M), p.name
            assert len(out.splitlines()) > len(text.splitlines()) // 2


# ------------------------------------------------------------------------------------------------ dry run
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Installer pointed at an empty envs/ folder: nothing is installed, nothing may be created."""
    monkeypatch.setattr(inst, "ENVS", tmp_path / "envs")
    monkeypatch.delenv("STUDIO_DL_PREFIX", raising=False)
    return tmp_path


def test_dry_run_pack_prints_the_steps_and_changes_nothing(sandbox, capsys):
    rc = inst.main(["pack", "starter", "--weights", "--dry-run", "--torch-index", "cu130"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "plan: pack starter" in out and "dry run: nothing was changed" in out
    assert re.search(r"uv venv --python 3\.12 \S*envs/core", out)
    assert "uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.0" in out
    assert "uv pip install" in out and "-r env-lock/qwen.txt" in out
    assert "./run_studio.sh" in out and "http://127.0.0.1:" in out
    assert not (sandbox / "envs").exists()


def test_vendor_clones_are_planned_when_missing(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(inst, "REPO", sandbox)       # no vendor/ folder here: everything has to be cloned
    ctx = inst.Ctx(dry_run=True)
    inst.fetch_vendor(ctx, inst.load_recipes()[0]["ttslight_parler"])
    out = capsys.readouterr().out
    assert "vendor/parler-tts" in out and "vendor/MeloTTS" in out
    assert "bash scripts/fetch_vendor.sh parler-tts MeloTTS" in out


def test_dry_run_cu128_uses_the_other_index_and_filters_the_lock(sandbox, capsys):
    rc = inst.main(["backend", "qwentts", "--dry-run", "--torch-index", "cu128"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "whl/cu128" in out and "whl/cu130" not in out
    assert "lock filtered for cu128" in out and ".lock.filtered.txt" in out
    assert not (sandbox / "envs").exists()


def test_dry_run_model_installs_the_backend_and_checks_the_weights(sandbox, capsys):
    rc = inst.main(["model", "kokoro-82m", "--dry-run", "--torch-index", "cu130"])
    out = capsys.readouterr().out
    assert rc == 0 and "plan: model kokoro-82m" in out and "backends: core, ttslight" in out
    assert "weights" in out


def test_dry_run_shared_env_installs_the_owner_first(sandbox, capsys):
    rc = inst.main(["backend", "moss_more", "--dry-run", "--torch-index", "cu130"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.index("==> backend moss (") < out.index("==> backend moss_more")


def test_dry_run_manual_backend_prints_its_manual_steps(sandbox, capsys):
    rc = inst.main(["backend", "vllm_omni", "--dry-run", "--torch-index", "cu130"])
    out = capsys.readouterr().out
    assert rc == 0 and "cannot be installed automatically" in out and "envs/vllm_omni" in out


def test_unknown_things_are_clear_errors(sandbox, capsys):
    assert inst.main(["pack", "nope", "--dry-run", "--torch-index", "cu130"]) == inst.EXIT_PREREQ
    assert "unknown pack" in capsys.readouterr().err
    assert inst.main(["backend", "nope", "--dry-run", "--torch-index", "cu130"]) == inst.EXIT_PREREQ
    assert "no recipe" in capsys.readouterr().err
    assert inst.main(["model", "no-such-model", "--dry-run", "--torch-index", "cu130"]) == inst.EXIT_PREREQ
    assert "unknown model" in capsys.readouterr().err
    assert inst.main(["model", "songgeneration-levo", "--dry-run", "--torch-index", "cu130"]) == inst.EXIT_PREREQ
    assert "blocked" in capsys.readouterr().err
    assert inst.main(["frobnicate"]) == inst.EXIT_PREREQ


def test_list_shows_packs_backends_and_models(capsys):
    assert inst.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "starter" in out and "ttslight_parler" in out and "z-image-turbo" in out and "vllm_omni" in out
    assert inst.main(["list", "packs"]) == 0
    assert inst.main(["list", "bogus"]) == inst.EXIT_PREREQ


def test_validate_command_is_green(capsys):
    assert inst.main(["validate"]) == 0
    assert "0 problem(s)" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ weights
@pytest.fixture
def wctx(tmp_path):
    """A registry with one cached repo and some missing ones, in a temp HF cache."""
    hf = tmp_path / "hf"
    snap = hf / "models--org--cached" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    blob = hf / "models--org--cached" / "blobs" / "b1"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x")
    (snap / "model.safetensors").symlink_to(blob)
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({
        "backends": {"b": {"python": str(tmp_path / "py"), "script": str(tmp_path / "w.py")}},
        "models": [
            {"id": "have", "backend": "b", "tasks": ["t"], "weights": [{"hf": "org/cached"}]},
            {"id": "need1", "backend": "b", "tasks": ["t"], "weights": [{"hf": "org/cached"}, {"hf": "org/new", "files": ["a/*.safetensors"]}]},
            {"id": "need2", "backend": "b", "tasks": ["t"], "weights": [{"hf": "org/new", "files": ["b.bin"]}, {"hf": "org/whole"}]},
            {"id": "need3", "backend": "b", "tasks": ["t"], "weights": [{"hf": "org/new"}]},
            {"id": "cmd", "backend": "b", "tasks": ["t"], "weights": [{"path": str(tmp_path / "x.pth"), "download_cmd": ["true"], "fix": "curl x"}]},
            {"id": "manual", "backend": "b", "tasks": ["t"], "weights": [{"path": str(tmp_path / "y.pth"), "fix": "put y there"}]},
        ]}))
    c = inst.Ctx(dry_run=True)
    c.reg = Registry(tmp_path / "models.yaml", None, hf_cache=hf, lmstudio=tmp_path / "lm", probe_cache=tmp_path / "p.json")
    return c


def test_plan_weights_skips_cached_repos_and_merges_files(wctx):
    hf, cmds, notes = inst.plan_weights(wctx, ["have", "need1", "need2"])
    assert [d.repo for d in hf] == ["org/new", "org/whole"]           # cached repo skipped, one entry per repo
    assert hf[0].patterns == ["a/*.safetensors", "b.bin"] and hf[0].models == ["need1", "need2"]
    assert hf[1].patterns is None                                     # no `files:` -> whole repo
    assert "--include" not in hf[1].fix() and "--include 'b.bin'" in hf[0].fix()
    hf, _, _ = inst.plan_weights(wctx, ["need1", "need3"])
    assert hf[0].patterns is None                                      # one model wants everything -> whole repo
    _, cmds, notes = inst.plan_weights(wctx, ["cmd", "manual"])
    assert [c[0] for c in cmds] == ["cmd"] and [n[0] for n in notes] == ["manual"] and "put y there" in notes[0][1]


def test_fetch_weights_downloads_one_repo_at_a_time_and_skips_cached(wctx, monkeypatch, tmp_path):
    wctx.dry_run = False
    calls = []
    monkeypatch.setattr(inst, "ENVS", tmp_path / "envs")
    (tmp_path / "envs" / "core" / "bin").mkdir(parents=True)
    (tmp_path / "envs" / "core" / "bin" / "python").write_text("")
    monkeypatch.setenv("STUDIO_DL_PREFIX", "flock /tmp/some.lock")

    def fake_run(ctx, cmd, **kw):
        calls.append((cmd, kw.get("env", {})))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(inst, "run", fake_run)
    inst.fetch_weights(wctx, ["have", "need1", "need2"])
    repos = [c[0][4] for c in calls]
    assert repos == ["org/new", "org/whole"]
    for cmd, env in calls:
        assert cmd[:3] == ["flock", "/tmp/some.lock", str(tmp_path / "envs" / "core" / "bin" / "python")]
        assert cmd[3].endswith("hf_fetch.py")
        assert env["HF_HUB_OFFLINE"] == "0" and env["HF_HUB_DISABLE_XET"] == "1"
    assert "have" in wctx.weights_skipped and wctx.weights_done == ["org/new", "org/whole"]


@pytest.mark.parametrize("code, fragment", [(10, "hf auth login"), (11, "rate limiting")])
def test_download_stops_on_gated_and_rate_limit(wctx, monkeypatch, tmp_path, code, fragment):
    wctx.dry_run = False
    monkeypatch.setattr(inst, "ENVS", tmp_path / "envs")
    (tmp_path / "envs" / "core" / "bin").mkdir(parents=True)
    (tmp_path / "envs" / "core" / "bin" / "python").write_text("")
    calls = []

    def fake_run(ctx, cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(inst, "run", fake_run)
    with pytest.raises(inst.Stop) as e:
        inst.fetch_weights(wctx, ["need2"])
    assert fragment in str(e.value)
    assert len(calls) == 1                       # stopped at the first refused repo: no further downloads
    if code == 10:
        assert "https://huggingface.co/org/new" in str(e.value)
        assert e.value.code == inst.EXIT_STOPPED
    assert wctx.weights_failed


def test_download_not_found_continues_but_is_reported(wctx, monkeypatch, tmp_path):
    wctx.dry_run = False
    monkeypatch.setattr(inst, "ENVS", tmp_path / "envs")
    (tmp_path / "envs" / "core" / "bin").mkdir(parents=True)
    (tmp_path / "envs" / "core" / "bin" / "python").write_text("")
    seq = iter([12, 0])
    monkeypatch.setattr(inst, "run", lambda ctx, cmd, **kw: SimpleNamespace(returncode=next(seq)))
    inst.fetch_weights(wctx, ["need2"])
    assert wctx.weights_failed and wctx.weights_done == ["org/whole"]


def test_hf_fetch_status_mapping():
    spec = importlib.util.spec_from_file_location("hf_fetch", ROOT / "scripts" / "hf_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    resp = SimpleNamespace(status_code=429)
    assert mod._status_of(SimpleNamespace(response=resp)) == 429
    assert mod._status_of(Exception("401 Client Error: Unauthorized")) == 401
    assert mod._status_of(Exception("boom")) is None


def test_weights_gb_missing_counts_only_what_is_absent(wctx):
    wctx.sizes = {"weights_gb": {"have": 5.0, "need1": 3.0, "need2": 4.0}, "gated": set()}
    assert inst.weights_gb_missing(wctx, ["have", "need1", "need2"]) == (7.0, 0)
    assert inst.weights_gb_missing(wctx, ["need3"]) == (0.0, 1)


# ------------------------------------------------------------------------------------------------ misc
def test_install_state_uses_marker_and_symlinks(tmp_path, monkeypatch, recipes):
    monkeypatch.setattr(inst, "ENVS", tmp_path / "envs")
    r = recipes["qwentts"]
    assert inst.backend_state(r) == "missing"
    (tmp_path / "envs" / "qwentts" / "bin").mkdir(parents=True)
    (tmp_path / "envs" / "qwentts" / "bin" / "python").write_text("")
    assert inst.backend_state(r) == "present"
    (tmp_path / "envs" / "qwentts" / inst.MARKER).write_text('{"hash": "%s"}' % r["_hash"])
    assert inst.backend_state(r) == "installed"
    (tmp_path / "envs" / "qwentts" / inst.MARKER).write_text('{"hash": "old"}')
    assert inst.backend_state(r) == "present"
    other = tmp_path / "other"
    (other / "bin").mkdir(parents=True)
    (other / "bin" / "python").write_text("")
    (tmp_path / "envs" / "audio").symlink_to(other)
    assert inst.backend_state(recipes["audio"]) == "linked"
    assert inst.backend_state(recipes["vllm_omni"]) == "manual"


def test_install_sh_exists_and_is_executable():
    sh = ROOT / "install.sh"
    assert sh.is_file() and sh.stat().st_mode & 0o111
    text = sh.read_text()
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in text
    assert "uv run" in text and "--python 3.12" in text and "scripts/install.py" in text


# ---------------------------------------------------------------- registry fix text, long paths
def test_registry_fix_points_at_installer(registry):
    m = {"id": "kokoro-82m"}
    # backend with a recipe -> installer command; backend without one -> the registry's own text
    assert Registry._installer_fix(m, b={"name": "ttslight"}, fix="old") == "./install.sh model kokoro-82m"
    assert Registry._installer_fix(m, b={"name": "cloud"}, fix="old") == "old"
    # Hugging Face weights / download_cmd -> installer first, raw command as the alternative
    assert Registry._installer_fix(m, w={"hf": "a/b"}, fix="hf download a/b") == "./install.sh model kokoro-82m   # or: hf download a/b"
    assert Registry._installer_fix(m, w={"hf": "a/b"}, fix="") == "./install.sh model kokoro-82m"
    # a plain file the user has to supply keeps its specific text
    assert Registry._installer_fix(m, w={"path": "x"}, fix="Put the file at x") == "Put the file at x"


def test_long_install_path_warns(monkeypatch):
    monkeypatch.setattr(inst, "REPO", Path("/home/u/open-asset-studio"))
    assert inst.long_path_warning() is None
    monkeypatch.setattr(inst, "REPO", Path("/tmp/" + "x" * 120))
    assert "160" in inst.long_path_warning()
