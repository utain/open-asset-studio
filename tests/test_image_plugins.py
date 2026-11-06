"""Pipelines flow store (path safety, validation) and the ComfyUI availability check."""
import pytest
from fastapi import HTTPException

from app import config
from app.plugins import image as image_plugin
from app.plugins import pipelines as pl
from app.registry import Registry

FID = "f20261105-120000-abcd12"


@pytest.mark.parametrize("bad", ["../etc", "f20261105-120000-ab/../x", "", "f2026", "F20261105-120000-abcd12", "f20261105-120000-abcd12.json"])
def test_flow_ids_are_validated(bad):
    with pytest.raises(HTTPException):
        pl.flow_path("default", bad)


def test_flow_project_is_validated():
    from app.assets import AssetError
    with pytest.raises(AssetError):
        pl.flow_path("../x", FID)


def test_flow_roundtrip_and_limits():
    rec = pl.put_flow(FID, {"project": "default", "kind": "concept3d", "title": "Chest", "step": 2, "state": {"drafts": {"picked": "x"}},
                            "evil": "dropped"})
    assert rec["id"] == FID and "evil" not in rec and rec["status"] == "active"
    assert (config.DATA / "pipelines" / "default" / f"{FID}.json").is_file()
    assert pl.get_flow(FID, "default")["state"]["drafts"]["picked"] == "x"
    assert [f["id"] for f in pl.list_flows("default")["flows"]] == [FID]
    with pytest.raises(HTTPException):
        pl.put_flow(FID, {"project": "default", "kind": "bogus"})
    with pytest.raises(HTTPException):
        pl.put_flow(FID, {"project": "default", "kind": "concept3d", "state": {"x": "y" * (pl.MAX_BYTES + 10)}})
    pl.delete_flow(FID, "default")
    with pytest.raises(HTTPException):
        pl.get_flow(FID, "default")


def test_comfyui_status_follows_reachability(tmp_path, monkeypatch):
    monkeypatch.delenv("STUDIO_NO_COMFYUI", raising=False)
    reg = Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path / "probe.json")
    if "flux2-klein-comfy" not in reg.models:
        pytest.skip("no comfyui model registered")

    def down(url):
        raise OSError("connection refused")

    image_plugin.check_comfyui(reg, fetch=down)
    st = reg.status["flux2-klein-comfy"]
    if st["code"] != "probe_failed":   # e.g. the comfyui worker script missing -> another reason, also not ready
        assert st["status"] != "ready"
    else:
        assert "not reachable" in st["reason"]
    image_plugin.check_comfyui(reg, fetch=lambda url: b"{}")
    assert reg.status["flux2-klein-comfy"]["code"] != "probe_failed"
    reg.reload()   # a registry refresh keeps the last reachability verdict
    image_plugin.check_comfyui(reg, fetch=down)
    reg.reload()
    assert reg.status["flux2-klein-comfy"]["status"] != "ready"
