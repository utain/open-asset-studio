"""Critical: downloads only accept registry model ids, need licence acceptance, and never take repos/commands from the API."""
import inspect

import pytest

from app import downloads as D


class FakeRegistry:
    def __init__(self, present=False):
        self.present = present
        self.models = {"m1": {"id": "m1", "name": "M1", "weights": [{"hf": "org/repo", "files": ["a/*.safetensors"]}],
                              "license": {"name": "X", "commercial": False}}}

    def get(self, mid):
        if mid not in self.models:
            raise KeyError(mid)
        return self.models[mid]

    def _check_weight(self, w):
        return (self.present, "x", "fix")


class Bus:
    def __init__(self):
        self.events = []

    def publish(self, t, d):
        self.events.append((t, d))


def make(present=False):
    d = D.Downloads(FakeRegistry(present), Bus())
    d.wake = type("W", (), {"set": lambda self: None, "wait": lambda self, t=None: None, "clear": lambda self: None})()
    return d


def test_requires_licence_acceptance():
    d = make()
    with pytest.raises(D.DownloadError):
        d.start("m1", accept_license=False)
    assert "m1" not in d.state


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        make().start("../../etc", accept_license=True)


def test_nothing_missing_rejected():
    with pytest.raises(D.DownloadError):
        make(present=True).start("m1", accept_license=True)


def test_queue_and_cancel_queued():
    d = make()
    st = d.start("m1", accept_license=True)
    assert st["status"] == "queued" and list(d.queue) == ["m1"]
    assert d.start("m1", accept_license=True) is st  # idempotent while active
    d.cancel("m1")
    assert d.state["m1"]["status"] == "cancelled" and not d.queue


def test_api_takes_only_model_id():
    from app import main
    assert set(main.DownloadReq.model_fields) == {"accept_license"}
    for fn in (main.download_plan, main.download_start, main.download_cancel):
        assert list(inspect.signature(fn).parameters)[0] == "model_id"


def test_hf_args_from_registry():
    assert D.Downloads._hf_args({"hf": "o/r", "files": ["x/*.bin"]}) == ["--repo", "o/r", "--include", "x/*.bin"]
