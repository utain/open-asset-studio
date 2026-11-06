"""Cloud providers plugin: creates `core.providers` (app/providers.py), registers the "cloud" job handler and the
registry hooks (status missing_key / cloud_disabled / ready, user OpenAI-compatible endpoints, price overrides),
and serves the Providers page API under /api/x/providers. No route ever returns an API key.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import cloud
from ..jobs import HANDLERS, TERMINAL, JobContext, handler
from ..providers import PROVIDERS, ProviderError, Providers, estimate_cost

router = APIRouter()
CORE = None
P: Providers | None = None


def _P() -> Providers:
    assert P is not None, "providers plugin not set up"
    return P


def setup(core):
    global CORE, P
    CORE = core
    P = Providers(registry=core.registry, bus=core.bus)
    core.providers = P
    reg = core.registry
    reg.status_hooks["cloud"] = P.model_status
    if P.model_hook not in reg.model_hooks:
        reg.model_hooks.append(P.model_hook)
    if P.endpoint_models not in reg.model_sources:
        reg.model_sources.append(P.endpoint_models)
    P.spend.is_active = lambda tok: any(j.get("_cost_reserve") == tok and j.get("status") not in TERMINAL
                                        for j in list(core.jobs.jobs.values()))
    P.on_spend = lambda e: core.bus.publish("spend", e)
    reg.reload()
    HANDLERS["cloud"]["precheck"] = lambda spec, model, params, inputs: cloud.precheck(P, spec, model, params, inputs)


@handler("cloud", gpu=False, pool="cloud", label="Cloud model")
def _cloud(ctx: JobContext):
    cloud.run_job(_P(), ctx)


# ------------------------------------------------------------------ helpers
def _cloud_models() -> list[dict]:
    reg = CORE.registry
    out = []
    for m in reg.models.values():
        if not P.is_cloud_model(m):
            continue
        out.append({"id": m["id"], "name": m["name"], "provider": m.get("provider"), "provider_label": P.label(m.get("provider")),
                    "modality": m.get("modality"), "tasks": m.get("tasks"), "remote_model": m.get("remote_model") or m.get("remote"),
                    "pricing": m.get("pricing"), "pricing_default": m.get("pricing_default", m.get("pricing")),
                    "pricing_overridden": m.get("pricing_overridden"), "pricing_checked": m.get("pricing_checked"),
                    "pricing_source": m.get("pricing_source"), "status": reg.status.get(m["id"], {}),
                    "license": m.get("license"), "unverified": m.get("unverified"),
                    "example_usd": estimate_cost(m.get("pricing"), {p["name"]: p.get("default") for p in m.get("params") or []
                                                                    if "default" in p})})
    out.sort(key=lambda m: (m["provider"] or "", m["modality"] or "", m["name"]))
    return out


# ------------------------------------------------------------------ routes
@router.get("")
@router.get("/")
def overview():
    p = _P()
    return {**p.public(), "spend": p.spend.summary(), "models": _cloud_models()}


class KeyReq(BaseModel):
    key: str


@router.put("/keys/{pid}")
def key_put(pid: str, req: KeyReq):
    _P().set_key(pid, req.key)
    return {"provider": pid, "key": _P().key_info(pid)}


@router.delete("/keys/{pid}")
def key_delete(pid: str):
    _P().delete_key(pid)
    return {"provider": pid, "key": _P().key_info(pid)}


@router.post("/test/{pid}")
def key_test(pid: str):
    return {"provider": pid, **cloud.test_provider(_P(), pid)}


class AllowReq(BaseModel):
    on: bool


@router.put("/allow")
def allow(req: AllowReq):
    _P().set_allowed(req.on)
    return {"allow_cloud": _P().cloud_allowed()}


class LimitsReq(BaseModel):
    budget_usd: float | None = None
    confirm_usd: float | None = None


@router.put("/limits")
def limits(req: LimitsReq):
    _P().set_limits(req.budget_usd, req.confirm_usd)
    return {"budget_usd": _P().state["budget_usd"], "confirm_usd": _P().state["confirm_usd"], "spend": _P().spend.summary()}


@router.get("/spend")
def spend(month: str | None = None):
    return _P().spend.summary(month)


class EndpointReq(BaseModel):
    name: str
    base_url: str
    label: str = ""
    key: str | None = None
    models: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=lambda: ["llm"])
    price_per_image: float | None = None


@router.put("/endpoints")
def endpoint_put(req: EndpointReq):
    ep = _P().put_endpoint(req.name, req.base_url, req.models, req.modalities, req.label, req.key, req.price_per_image)
    return {"endpoint": {**ep, "key": _P().key_info(f"openai_compat:{ep['name']}")}}


@router.delete("/endpoints/{name}")
def endpoint_delete(name: str):
    _P().delete_endpoint(name)
    return {"ok": True}


@router.post("/endpoints/{name}/models")
def endpoint_models(name: str):
    pid = f"openai_compat:{name}"
    if not _P().endpoint(pid):
        raise ProviderError(f"No endpoint '{name}'.", "input")
    return {"models": cloud.endpoint_models(_P(), pid)}


class PriceReq(BaseModel):
    pricing: dict


@router.put("/prices/{model_id}")
def price_put(model_id: str, req: PriceReq):
    m = CORE.registry.get(model_id)
    if not _P().is_cloud_model(m):
        raise ProviderError("Only cloud models have prices.", "input")
    _P().set_price_override(model_id, req.pricing)
    return {"model": model_id, "pricing": CORE.registry.get(model_id).get("pricing")}


@router.delete("/prices/{model_id}")
def price_delete(model_id: str):
    CORE.registry.get(model_id)
    _P().set_price_override(model_id, None)
    return {"model": model_id, "pricing": CORE.registry.get(model_id).get("pricing")}


class AckReq(BaseModel):
    on: bool = True


@router.put("/upload-ack/{pid}")
def upload_ack(pid: str, req: AckReq):
    if not _P().known(pid):
        raise ProviderError(f"Unknown provider '{pid}'.", "input")
    _P().acknowledge_upload(pid, req.on)
    return {"provider": pid, "upload_ack": (_P().state.get("upload_ack") or {}).get(pid)}


@router.get("/clis")
def clis(refresh: bool = False):
    return {"clis": _P().clis(refresh)}


class PrecheckReq(BaseModel):
    task: str | None = None
    model: str
    params: dict = Field(default_factory=dict)
    inputs: list = Field(default_factory=list)
    confirm_cost: bool = False
    confirm_upload: bool = False


@router.post("/precheck")
def precheck(req: PrecheckReq):
    """Dry run of the submit-time checks, so the UI can ask (cost / upload consent) before POST /api/jobs.
    200 {ok: true, cost} or 200 {ok: false, kind, error, fix, ...data}; never reserves budget or records consent."""
    reg = CORE.registry
    m = reg.get(req.model)
    if not _P().is_cloud_model(m):
        return {"ok": True, "cost": None}
    try:
        params = reg.validate_params(req.model, req.params)
        r = cloud.precheck(_P(), req.model_dump(), m, params, [i for i in req.inputs if i], dry=True)
    except ProviderError as e:
        return {**(e.data or {}), "ok": False, "kind": e.kind, "error": str(e), "fix": e.fix}
    except ValueError as e:
        return {"ok": False, "kind": "input", "error": str(e), "fix": ""}
    return {"ok": True, "cost": r.get("cost")}


class EstimateReq(BaseModel):
    model: str
    params: dict = Field(default_factory=dict)
    inputs: int = 0


@router.post("/estimate")
def estimate(req: EstimateReq):
    m = CORE.registry.get(req.model)
    params = CORE.registry.validate_params(req.model, req.params)
    return {"model": req.model, "estimate_usd": estimate_cost(m.get("pricing"), params, req.inputs),
            "pricing": m.get("pricing"), "provider": m.get("provider")}


@router.get("/catalog")
def catalog():
    """Static provider catalog (labels, env var names, links): for docs and the other plugins."""
    return {"providers": {k: {kk: vv for kk, vv in v.items() if kk not in ("auth",)} for k, v in PROVIDERS.items()}}
