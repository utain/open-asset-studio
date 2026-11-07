"""Commercial-use licence filter (docs/CONTRACTS.md "Licence filter").

Every registry model carries `license: {name, commercial, note?, verify?}`. Three states for the UI and the API:

  commercial  commercial use is allowed                    (`commercial: true`)
  check       allowed on paper, but the entry says the terms must be verified (`commercial: true, verify: true`)
  noncommercial  research / non-commercial licence         (`commercial: false`, or a missing licence)

With the setting `license.commercial_only` on, the job API refuses `noncommercial` models (and `check` models too when
`license.strict` is on) with a clear error of kind "license"; the web pickers hide or disable the same models.
"""
from __future__ import annotations

COMMERCIAL, CHECK, NONCOMMERCIAL = "commercial", "check", "noncommercial"


def state(lic: dict | None) -> str:
    if not isinstance(lic, dict) or not lic.get("commercial"):
        return NONCOMMERCIAL
    return CHECK if lic.get("verify") else COMMERCIAL


def settings_of(settings) -> tuple[bool, bool]:
    if settings is None:
        return False, False
    return bool(settings.get("license", "commercial_only", default=False)), bool(settings.get("license", "strict", default=False))


def blocked(model: dict | None, settings) -> bool:
    """True when `model` may not be used under the current commercial-only setting."""
    only, strict = settings_of(settings)
    if not only or not model:
        return False
    st = state(model.get("license"))
    return st == NONCOMMERCIAL or (strict and st == CHECK)


def refusal(model: dict, project: str | None = None) -> str:
    lic = model.get("license") or {}
    st = state(lic)
    name = lic.get("name") or "unknown licence"
    why = "is not licensed for commercial use" if st == NONCOMMERCIAL else "needs its licence verified before commercial use"
    where = f" (set by project '{project}')" if project else ""
    return (f"{model.get('name') or model.get('id')} {why} ({name}), and 'Commercial-safe models only' is on{where}. "
            "Pick a model with a commercial licence, or turn the setting off (Settings > General, "
            "or the shield button in the top bar).")


def check(model: dict, settings):
    """Raise jobs.JobError(kind='license') when the model is refused. Imported lazily to avoid a cycle."""
    if blocked(model, settings):
        from .jobs import JobError

        # name the project only when it is the project that switched the filter on (its own `license` override)
        proj = getattr(settings, "project", None)
        proj = proj if proj and (getattr(settings, "overrides", None) or {}).get("license") else None
        raise JobError(refusal(model, proj), "license", "Settings > license.commercial_only = false")
