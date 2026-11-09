"""Who is calling: the two checks that guard actions on THIS machine (same rules as app/plugins/mcp.py).

`origin_ok`  a browser page on another site must not drive the API (CSRF / DNS-rebinding); clients without an Origin header pass.
`is_local`   same machine only. A local reverse proxy (`tailscale serve`, nginx ...) connects from 127.0.0.1 but forwards other
             machines, so any proxy header makes the caller remote.
Used by the engine delivery router (app/plugins/engine.py) and the redaction of delivery paths in GET /api/settings.
"""
from __future__ import annotations

from urllib.parse import urlsplit

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "tailscale-user-login")


def origin_ok(request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    o = urlsplit(origin)
    host = (request.headers.get("host") or "").split(":")[0]
    return o.hostname in LOCAL_HOSTS or (bool(host) and o.hostname == host)


def is_local(request) -> bool:
    if any(h in request.headers for h in PROXY_HEADERS):
        return False
    return bool(request.client) and request.client.host in ("127.0.0.1", "::1", "localhost")
