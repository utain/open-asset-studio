"""Cloud model jobs (handler "cloud") + provider key tests. Registered by app/plugins/providers.py.

Why in-process and not a workers/cloud_worker.py subprocess: cloud jobs only wait on HTTP (no GPU, little memory),
the API keys then never leave the core process (no key in a worker's argv / env / stdin), cancel is a flag checked
between polls, and outputs go straight into the asset store. They run in their own "cloud" thread pool
(jobs.cloud_concurrency, default 4), never in the GPU queue, so they run alongside local GPU jobs.
"""
from __future__ import annotations

import secrets
import time

import httpx

from ..jobs import JobCancelled, JobContext, JobError
from ..providers import PROVIDERS, ProviderError, Providers, estimate_cost, fmt_usd
from . import elevenlabs, fal, gemini, mesh3d, openai, openrouter, replicate
from .base import CloudRun

ADAPTERS = {"fal": fal.run, "replicate": replicate.run, "openai": openai.run, "gemini": gemini.run,
            "elevenlabs": elevenlabs.run, "meshy": mesh3d.meshy, "tripo": mesh3d.tripo, "openrouter": openrouter.run}


def adapter_for(m: dict):
    a = m.get("adapter") or m.get("provider", "")
    if a.startswith("openai_compat:"):
        a = "openai"
    fn = ADAPTERS.get(a)
    if fn is None:
        raise JobError(f"No cloud adapter for '{a}'.", "env")
    return fn


def precheck(P: Providers, spec: dict, model: dict | None, params: dict, inputs: list, dry: bool = False) -> dict:
    """Runs in JobManager.submit before the job exists: switch, key, upload consent, budget, confirm.
    dry=True (the UI's ask-first call): same checks, but no reservation and no consent is recorded."""
    if not model:
        raise JobError("Cloud jobs need a model.", "input")
    pid = model.get("provider") or ""
    label = P.label(pid)
    if model.get("modality") == "text":
        raise JobError(f"{model['name']} is a text model: use it from the prompt improver / assistant.", "input")
    local = P.is_local_provider(pid)
    if not local and not P.cloud_allowed():
        raise ProviderError("Cloud models are switched off.", "cloud_disabled", "Turn on “Allow cloud” on the Providers page.")
    if pid in PROVIDERS and not P.get_key(pid):
        raise ProviderError(f"No API key for {label}.", "missing_key", "Add it on the Providers page.")
    n_in = len(inputs or [])
    if n_in and not P.upload_acknowledged(pid):
        if spec.get("confirm_upload"):
            if not dry:
                P.acknowledge_upload(pid)
        else:
            info = P.info(pid)
            raise ProviderError(
                f"This job uploads {n_in} file{'s' if n_in > 1 else ''} from your library to {label}.", "consent",
                "Confirm once per provider to allow sending your images / audio off this machine.",
                {"provider": pid, "provider_label": label, "files": n_in, "terms": info.get("terms", ""),
                 "terms_url": info.get("terms_url")})
    est = estimate_cost(model.get("pricing"), params, n_in)
    confirmed = bool(spec.get("confirm_cost"))
    token = None
    if not local:
        P.spend.check(est, pid, confirmed=confirmed, model_id=model["id"])
        if not dry:
            token = "r" + secrets.token_hex(6)
            P.spend.reserve(token, est)
    return {"cost": {"estimate_usd": est, "currency": "USD", "provider": pid, "provider_label": label,
                     "confirmed": confirmed, "local": local, "pricing_checked": model.get("pricing_checked")},
            "_cost_reserve": token}


def run_job(P: Providers, ctx: JobContext):
    m = ctx.model
    if not m:
        raise JobError("Unknown model.", "input")
    pid = m["provider"]
    job = ctx.job
    cost = job.setdefault("cost", {})
    token = job.get("_cost_reserve")
    local = P.is_local_provider(pid)
    r = None
    try:
        if not local and not P.cloud_allowed():
            raise ProviderError("Cloud models were switched off while this job waited.", "cloud_disabled",
                                "Turn on “Allow cloud” on the Providers page and retry.")
        fn = adapter_for(m)
        r = CloudRun(ctx, P, m)
        ctx.progress(0.01, f"Contacting {r.label}", stage="submit")
        t0 = time.time()
        try:
            fn(r)
        except JobCancelled:
            if r.submitted and not local:
                est = cost.get("estimate_usd") or 0
                P.spend.record(est, pid, m["id"], job["project"], job["id"], estimated=True,
                               note="cancelled after submit: the provider may still charge")
                cost.update(usd=est, estimated=True, note="cancelled after submit (may be charged)")
            raise
        if not r.files:
            raise ProviderError(f"{r.label} returned no output.", "provider")
        est = cost.get("estimate_usd")
        usd = r.cost_usd if r.cost_usd is not None else float(est or 0)
        estimated = r.cost_usd is None
        if not local:
            P.spend.record(usd, pid, m["id"], job["project"], job["id"], estimated=estimated, note=r.cost_note)
        cost.update(usd=round(usd, 5), actual_usd=r.cost_usd, estimated=estimated, note=r.cost_note,
                    request_ids=r.request_ids[:20], seconds=round(time.time() - t0, 1))
        ctx.log(f"{r.label}: {fmt_usd(usd)}{' (estimate)' if estimated else ''}{' · ' + r.cost_note if r.cost_note else ''}")
        ctx.progress(0.97, "Saving outputs", stage="saving")
        info = P.info(pid)
        per = usd / len(r.files)
        cloud_meta = {"provider": pid, "provider_label": r.label, "remote_model": r.remote(),
                      "request_ids": r.request_ids[:20], "terms": (m.get("cloud") or {}).get("terms") or info.get("terms", ""),
                      "terms_url": info.get("terms_url"), "uploaded_inputs": len(r.inputs)}
        for i, f in enumerate(r.files):
            fm = dict(f["meta"])
            meta = ctx.asset_meta(fm, fm.get("index", i))
            meta["cloud"] = cloud_meta
            meta["cost"] = {"usd": round(per, 5), "job_usd": round(usd, 5), "estimated": estimated, "currency": "USD"}
            seeds = r.extra_meta.get("seeds") or []
            if i < len(seeds) and seeds[i] is not None:
                meta["seed"] = seeds[i]
            if f["type"] == "voice" and r.extra_meta.get("x_voice"):
                meta["x_voice"] = r.extra_meta["x_voice"]
            ctx.add_output(f["path"], f["type"], meta, attach=f.get("attach"))
    finally:
        if r is not None:
            r.close()
        if token:
            P.spend.release(token)


# ------------------------------------------------------------------ key tests (cheap, free calls)
def _get(P: Providers, pid: str, path: str, kind: str = "api", **kw) -> httpx.Response:
    url = P.base(pid, kind) + path
    try:
        return httpx.get(url, headers={**P.headers(pid), "User-Agent": "open-asset-studio/0.1"}, timeout=20.0,
                         follow_redirects=True, **kw)
    except httpx.HTTPError as e:
        raise ProviderError(f"Cannot reach {P.label(pid)} ({type(e).__name__}).", "network") from None


def test_provider(P: Providers, pid: str) -> dict:
    """{ok, message, details}: a free authenticated call. Never returns the key or account identity."""
    if not P.known(pid):
        raise ProviderError(f"Unknown provider '{pid}'.", "input")
    if not P.get_key(pid) and pid in PROVIDERS:
        return {"ok": False, "message": "No key set.", "details": {}}
    t0 = time.time()
    det: dict = {}
    try:
        if pid == "fal":
            r = _get(P, pid, "/models/pricing", params={"endpoint_id": "fal-ai/flux-2-pro"})
            if r.status_code < 400:
                pr = (r.json().get("prices") or [{}])[0]
                det["sample_price"] = f"flux-2-pro: ${pr.get('unit_price')} per {pr.get('unit')}" if pr else None
                b = _get(P, pid, "/account/billing", params={"expand": "credits"})
                if b.status_code < 400:
                    cr = (b.json().get("credits") or {})
                    det["balance"] = f"{cr.get('current_balance')} {cr.get('currency', 'USD')}"
                else:
                    det["balance"] = "needs a key with billing:read scope"
        elif pid == "replicate":
            r = _get(P, pid, "/account")
            if r.status_code < 400:
                det["account_type"] = r.json().get("type")
        elif pid == "openrouter":
            r = _get(P, pid, "/key")
            if r.status_code < 400:
                d = r.json().get("data") or {}
                det.update(usage_usd=d.get("usage"), limit_usd=d.get("limit"), limit_remaining_usd=d.get("limit_remaining"),
                           free_tier=d.get("is_free_tier"))
        elif pid == "openai":
            r = _get(P, pid, "/models")
            if r.status_code < 400:
                ids = [x.get("id", "") for x in r.json().get("data") or []]
                det["models"] = len(ids)
                det["image_models"] = [x for x in ids if "image" in x][:8]
        elif pid == "gemini":
            r = _get(P, pid, "/models", params={"pageSize": 50})
            if r.status_code < 400:
                ids = [x.get("name", "").split("/")[-1] for x in r.json().get("models") or []]
                det["models"] = len(ids)
                det["image_models"] = [x for x in ids if "image" in x][:8]
        elif pid == "elevenlabs":
            r = _get(P, pid, "/user/subscription")
            if r.status_code < 400:
                d = r.json()
                det.update(tier=d.get("tier"), characters_used=d.get("character_count"), character_limit=d.get("character_limit"),
                           status=d.get("status"))
        elif pid == "meshy":
            r = _get(P, pid, "/v1/balance")
            if r.status_code < 400:
                det["credits"] = r.json().get("balance")
        elif pid == "tripo":
            r = _get(P, pid, "/account/balance")
            if r.status_code < 400:
                d = r.json().get("data") or {}
                det.update(credits=d.get("balance"), frozen=d.get("frozen"))
        else:   # openai_compat:<name>
            r = _get(P, pid, "/models")
            if r.status_code < 400:
                det["models"] = [x.get("id") for x in (r.json().get("data") or [])][:40]
    except ValueError:
        return {"ok": False, "message": f"{P.label(pid)} answered with something that is not JSON.", "details": {}}
    ms = int((time.time() - t0) * 1000)
    if r.status_code < 400:
        msg = f"Key works ({ms} ms)." if P.get_key(pid) else f"Endpoint answered ({ms} ms)."
        return {"ok": True, "message": msg, "details": {k: v for k, v in det.items() if v is not None}}
    txt = P.secrets_in(r.text[:200])
    if r.status_code in (401, 403):
        return {"ok": False, "message": f"Key rejected ({r.status_code}).", "details": {"answer": txt}}
    return {"ok": False, "message": f"{P.label(pid)} answered {r.status_code}.", "details": {"answer": txt}}


def endpoint_models(P: Providers, pid: str) -> list[str]:
    r = _get(P, pid, "/models")
    if r.status_code >= 400:
        raise ProviderError(f"{P.label(pid)}: GET /models answered {r.status_code}.", "provider")
    try:
        return [x.get("id") for x in (r.json().get("data") or []) if x.get("id")][:100]
    except ValueError:
        raise ProviderError("The endpoint's /models answer is not JSON.", "provider") from None
