"""fal.ai queue API: POST https://queue.fal.run/{endpoint} -> poll status_url -> GET response_url.

Auth `Authorization: Key <FAL_KEY>`. Status values IN_QUEUE | IN_PROGRESS | COMPLETED (a failed request is COMPLETED
with `error`). Status/result/cancel URLs drop the endpoint's sub-path, so we always use the URLs fal returns.
fal does not return a per-request cost: the registry estimate is recorded (marked estimated).
"""
from __future__ import annotations

from .base import CloudRun
from .common import batch_count, build_input
from ..providers import ProviderError


def _submit(r: CloudRun, endpoint: str, body: dict) -> dict:
    sub = r.post_json(r.url(f"/{endpoint}", "queue"), body)
    rid = sub.get("request_id")
    if not rid:
        raise ProviderError("fal.ai did not return a request id.", "provider")
    r.request_ids.append(rid)
    r.submitted = True
    r.log(f"fal request {rid} ({endpoint})")
    base = r.url(f"/{'/'.join(endpoint.split('/')[:2])}/requests/{rid}", "queue")
    status_url = sub.get("status_url") or base + "/status"
    resp_url = sub.get("response_url") or base
    cancel_url = sub.get("cancel_url") or base + "/cancel"
    r.cancel_remote = lambda: r.request("PUT", cancel_url)
    return {"id": rid, "status": status_url, "response": resp_url}


def _wait(r: CloudRun, h: dict, frac0: float, frac1: float) -> dict:
    def step():
        st = r.get_json(h["status"] + ("&" if "?" in h["status"] else "?") + "logs=1")
        s = st.get("status")
        if s == "IN_QUEUE":
            pos = st.get("queue_position")
            r.progress(frac0, f"In the fal.ai queue{f' · position {pos}' if pos is not None else ''}", stage="queued")
            return None
        if s == "IN_PROGRESS":
            logs = st.get("logs") or []
            msg = (logs[-1].get("message") if logs and isinstance(logs[-1], dict) else None) or "Generating on fal.ai"
            r.progress(frac0 + (frac1 - frac0) * 0.5, str(msg)[:120], stage="generating")
            return None
        if s == "COMPLETED":
            if st.get("error"):
                raise ProviderError(f"fal.ai: {st.get('error')}", "input" if st.get("error_type") else "provider")
            return st
        return None
    r.poll(step, interval=1.5)
    res = r.get_json(h["response"])
    if isinstance(res, dict) and res.get("detail") and not any(k in res for k in ("images", "image", "audio", "model_glb", "model_mesh")):
        raise ProviderError(f"fal.ai: {res['detail']}", "provider")
    return res


def run(r: CloudRun):
    cfg = r.m.get("request") or {}
    endpoint = r.remote()
    want = {"image": "image", "mesh": "mesh", "audio": "audio"}[r.m["modality"]]
    n = batch_count(r)
    loops = 1 if (cfg.get("n") or want != "image") else n
    if want == "audio" and not cfg.get("n"):
        loops = n
    got = 0
    for i in range(loops):
        body = build_input(r, cfg, n=n if cfg.get("n") else None)
        if loops > 1 and "seed" in (cfg.get("map") or {}) and int(r.params.get("seed", -1)) >= 0:
            body[(cfg.get("map") or {})["seed"]] = int(r.params["seed"]) + i
        lo, hi = i / loops, (i + 1) / loops
        r.progress(0.05 + 0.85 * lo, f"Submitting to fal.ai ({endpoint})", stage="submit")
        h = _submit(r, endpoint, body)
        res = _wait(r, h, 0.05 + 0.85 * lo, 0.05 + 0.85 * hi)
        if isinstance(res, dict) and res.get("seed") is not None:
            r.extra_meta.setdefault("seeds", []).append(res.get("seed"))
        k = r.fetch_outputs(res, want, prefix=f"fal{i + 1}")
        if not k:
            raise ProviderError(f"fal.ai returned no {want} file.", "provider", "See the job log; retry.")
        got += k
    return got
