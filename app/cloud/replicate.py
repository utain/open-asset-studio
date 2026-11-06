"""Replicate predictions API (Bearer token).

Official models: POST /v1/models/{owner}/{name}/predictions {input}. Community models (`community: true`): the latest
version id is read from GET /v1/models/{owner}/{name} and POST /v1/predictions {version, input} is used.
Poll GET /v1/predictions/{id} (starting | processing | succeeded | failed | canceled); cancel POST .../cancel.
Inputs are sent as data: URIs (Replicate accepts them; fine for images up to a few MB).
Actual cost: `pricing.per_predict_second` x metrics.predict_time for time-billed models, else the estimate.
"""
from __future__ import annotations

from urllib.parse import urlparse

from .base import CloudRun
from .common import batch_count, build_input
from ..providers import ProviderError

_VERSIONS: dict[str, str] = {}


def _version(r: CloudRun, slug: str) -> str:
    if slug not in _VERSIONS:
        d = r.get_json(r.url(f"/models/{slug}"))
        v = (d.get("latest_version") or {}).get("id")
        if not v:
            raise ProviderError(f"Replicate model {slug} has no published version.", "provider")
        _VERSIONS[slug] = v
    return _VERSIONS[slug]


def run(r: CloudRun):
    cfg = r.m.get("request") or {}
    slug = r.remote()
    want = {"image": "image", "mesh": "mesh", "audio": "audio"}[r.m["modality"]]
    n = batch_count(r)
    loops = 1 if cfg.get("n") else n
    total_s = 0.0
    got = 0
    for i in range(loops):
        inp = build_input(r, cfg, n=n if cfg.get("n") else None)
        if loops > 1 and "seed" in (cfg.get("map") or {}) and int(r.params.get("seed", -1)) >= 0:
            inp[(cfg.get("map") or {})["seed"]] = int(r.params["seed"]) + i
        r.progress(0.05 + 0.85 * i / loops, f"Submitting to Replicate ({slug})", stage="submit")
        if r.m.get("community"):
            pred = r.post_json(r.url("/predictions"), {"version": _version(r, slug), "input": inp})
        else:
            pred = r.post_json(r.url(f"/models/{slug}/predictions"), {"input": inp}, headers={"Prefer": "wait=5"})
        pid = pred.get("id")
        if not pid:
            raise ProviderError("Replicate did not return a prediction id.", "provider")
        r.request_ids.append(pid)
        r.submitted = True
        r.log(f"replicate prediction {pid}")
        get_url = (pred.get("urls") or {}).get("get") or r.url(f"/predictions/{pid}")
        cancel_url = (pred.get("urls") or {}).get("cancel") or r.url(f"/predictions/{pid}/cancel")
        r.cancel_remote = lambda u=cancel_url: r.request("POST", u)
        state = {"p": pred}

        def step():
            p = state["p"]
            s = p.get("status")
            if s == "succeeded":
                return p
            if s in ("failed", "canceled"):
                raise ProviderError(f"Replicate prediction {s}: {r.p.secrets_in(str(p.get('error') or ''))[:300]}",
                                    "input" if s == "failed" else "provider")
            logs = (p.get("logs") or "").strip().splitlines()
            r.progress(0.05 + 0.85 * (i + 0.5) / loops, (logs[-1][:120] if logs else f"Replicate: {s or 'starting'}"),
                       stage="generating" if s == "processing" else "queued")
            state["p"] = r.get_json(get_url)
            return None
        done = r.poll(step, interval=1.5)
        total_s += float((done.get("metrics") or {}).get("predict_time") or 0)
        api_host = urlparse(r.url("")).hostname
        out = done.get("output")
        files_auth = any(urlparse(u).hostname == api_host for u in ([out] if isinstance(out, str) else
                                                                    [x for x in (out or []) if isinstance(x, str)]))
        k = r.fetch_outputs(out, want, prefix=f"rep{i + 1}", auth=files_auth)
        if not k:
            raise ProviderError(f"Replicate returned no {want} file.", "provider")
        got += k
    pps = (r.m.get("pricing") or {}).get("per_predict_second")
    if pps and total_s:
        r.cost_usd = round(total_s * float(pps), 5)
        r.cost_note = f"{total_s:.1f} s of compute"
    return got
