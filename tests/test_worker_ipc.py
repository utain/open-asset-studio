"""Worker IPC: JSON-lines framing survives stray prints, big messages, errors; the worker dies with its parent."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from app import config
from app.workers import WorkerCancelled, WorkerError, WorkerProcess

ROOT = config.STUDIO
PY = str(ROOT / "envs" / "core" / "bin" / "python")

NOISY = textwrap.dedent('''
import os, sys, time
from studio_worker import Backend, InputError, serve

class Noisy(Backend):
    name = "noisy"
    def load(self, model, ctx):
        print("library banner on stdout")            # must not corrupt the protocol
        os.write(1, b"raw fd1 write {not json}\\n")
        sys.stderr.write("some warning\\n")
        return True
    def run(self, job, ctx):
        p = job.params
        if p.get("crash"):
            os._exit(3)
        if p.get("bad"):
            raise InputError("bad input")
        if p.get("hang"):
            while True:
                time.sleep(0.1)   # ignores cancel: the core must kill it after the grace period
        for i in range(int(p.get("steps", 3))):
            ctx.check(); time.sleep(0.01)
            print("step noise", i)
            ctx.progress(i + 1, int(p.get("steps", 3)), "work", "x" * int(p.get("big", 0)))
        f = job.out_dir / "out.txt"; f.write_text("ok")
        ctx.output(f, "text", note="ünïcödé ✓")
        return {"meta": {"echo": p.get("echo")}}

if __name__ == "__main__":
    serve(Noisy())
''')


@pytest.fixture
def noisy(tmp_path):
    script = tmp_path / "noisy_worker.py"
    script.write_text(NOISY)
    b = {"name": "noisy", "python": PY, "script": str(script), "kill_on_cancel": True}
    w = WorkerProcess(b, None, log_dir=tmp_path)
    yield w, tmp_path
    w.stop(wait=3)


def test_framing_progress_and_result(noisy):
    w, tmp = noisy
    events = []
    big = 200_000
    res = w.run("j1", {"id": "m1", "name": "M1"}, "t", {"steps": 4, "big": big, "echo": "é" * 1000}, [], str(tmp / "o"),
                on_event=events.append)
    prog = [e for e in events if e["ev"] == "progress" and e.get("stage") == "work"]
    assert len(prog) == 4 and len(prog[-1]["msg"]) == big
    assert res["meta"]["echo"] == "é" * 1000
    assert res["files"][0]["note"] == "ünïcödé ✓" and open(res["files"][0]["path"]).read() == "ok"
    assert "m1" in w.loaded
    assert any(e["ev"] == "loaded" for e in events)
    log = (tmp / "worker-noisy.log").read_text()
    assert "library banner on stdout" in log and "raw fd1 write" in log  # went to the log, not the protocol


def test_errors_and_crash(noisy):
    w, tmp = noisy
    with pytest.raises(WorkerError) as e:
        w.run("j2", {"id": "m1"}, "t", {"bad": 1}, [], str(tmp / "o"))
    assert e.value.kind == "input" and "bad input" in str(e.value)
    assert w.alive()  # a job error does not kill the worker
    with pytest.raises(WorkerError) as e:
        w.run("j3", {"id": "m1"}, "t", {"crash": 1}, [], str(tmp / "o"))
    assert "exited unexpectedly" in str(e.value)
    # next request transparently starts a fresh worker
    assert w.run("j4", {"id": "m1"}, "t", {"steps": 1}, [], str(tmp / "o"))["files"]


def test_cancel_kills_unresponsive_worker(noisy):
    import threading

    w, tmp = noisy
    w.settings = type("S", (), {"get": lambda self, *k, default=None: 1 if k[-1] == "cancel_grace_s" else default})()
    ev = threading.Event()
    threading.Timer(0.8, ev.set).start()
    t = time.time()
    with pytest.raises(WorkerCancelled):
        w.run("j5", {"id": "m1"}, "t", {"hang": 1}, [], str(tmp / "o"), cancel=ev)
    assert time.time() - t < 8 and not w.alive()


PARENT = textwrap.dedent('''
import sys, time
sys.path.insert(0, sys.argv[1])
from app.workers import WorkerProcess
w = WorkerProcess({"name": "echo", "python": sys.argv[2], "script": sys.argv[1] + "/workers/echo_worker.py"}, None,
                  log_dir=sys.argv[3])
w.start()
print(w.proc.pid, flush=True)
time.sleep(600)
''')


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_worker_dies_with_parent(tmp_path, sig):
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    parent = subprocess.Popen([PY, str(script), str(ROOT), PY, str(tmp_path)], stdout=subprocess.PIPE, text=True,
                              start_new_session=True)
    child = int(parent.stdout.readline())
    time.sleep(0.5)
    os.kill(child, 0)  # alive
    parent.send_signal(sig)
    parent.wait(timeout=10)
    t = time.time()
    while time.time() - t < 6:
        try:
            os.kill(child, 0)
            with open(f"/proc/{child}/status") as f:
                if "zombie" in f.read():
                    break
        except (ProcessLookupError, FileNotFoundError):
            break
        time.sleep(0.1)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("worker survived its parent")
