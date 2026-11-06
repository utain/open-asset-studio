"""MCP over HTTP, served by the core itself: POST /api/x/mcp (MCP "Streamable HTTP" transport, protocol 2025-06-18).

Starts and stops with the studio, so an MCP client only needs the URL:
  claude mcp add --transport http studio http://127.0.0.1:7860/api/x/mcp        (this repo's .mcp.json does it)

The tools are the ones of scripts/mcp_server.py (single source of truth; that file is also the stdio transport). They are
thin clients of the public REST API over loopback, so jobs take the same queue / memory / licence path as the web UI.
Requests get a JSON response; a tools/call from a client that accepts text/event-stream gets an SSE stream with
notifications/progress events followed by the result. No sessions, no server-initiated stream (GET -> 405).
Same trust model as the REST API (no auth, see SECURITY.md); requests from a browser page of another origin are refused
(DNS-rebinding / CSRF guard).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import config

router = APIRouter()
_spec = importlib.util.spec_from_file_location("studio_public_mcp", config.STUDIO / "scripts" / "mcp_server.py")
MCP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MCP)
MCP.DATA = config.DATA
# Own threads: a tool call waits on the core's REST API, which runs in FastAPI's thread pool. Running the calls in that
# same pool could fill it with waiting submit_job calls (up to 600 s each) and starve the API they wait on.
POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mcp")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _api_url() -> str:
    return os.environ.get("STUDIO_MCP_API") or f"http://127.0.0.1:{os.environ.get('STUDIO_PORT', '7860')}"


def _origin_ok(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:                         # CLI / desktop clients send no Origin
        return True
    o = urlsplit(origin)
    host = (request.headers.get("host") or "").split(":")[0]
    return o.hostname in LOCAL_HOSTS or (bool(host) and o.hostname == host)


def _is_local(request: Request) -> bool:
    return bool(request.client) and request.client.host in ("127.0.0.1", "::1", "localhost")


def _rpc_error(mid, code: int, message: str, status: int = 200):
    return JSONResponse({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}, status_code=status)


@router.post("")
async def mcp_post(request: Request):
    if not _origin_ok(request):
        return _rpc_error(None, -32600, "Origin not allowed", 403)
    try:
        msg = json.loads(await request.body())
    except ValueError:
        return _rpc_error(None, -32700, "Parse error", 400)
    MCP.API = _api_url()
    local = _is_local(request)             # only same-machine clients may name files on this machine (import_file)
    loop = asyncio.get_running_loop()

    def run(m, emit=None):
        return loop.run_in_executor(POOL, lambda: MCP.handle(m, emit=emit, allow_paths=local))

    if isinstance(msg, list):              # JSON-RPC batch (older protocol revisions)
        outs = [o for o in [await run(m) for m in msg if isinstance(m, dict)] if o]
        return JSONResponse(outs) if outs else Response(status_code=202)
    if not isinstance(msg, dict):
        return _rpc_error(None, -32600, "Invalid request", 400)
    if msg.get("id") is None:              # notification or a response to us: accepted, nothing to say
        return Response(status_code=202)
    streaming = msg.get("method") == "tools/call" and "text/event-stream" in request.headers.get("accept", "") \
        and ((msg.get("params") or {}).get("_meta") or {}).get("progressToken") is not None
    if not streaming:
        out = await run(msg)
        return JSONResponse(out)

    q: asyncio.Queue = asyncio.Queue()     # async end to end: a sync generator would hold a FastAPI pool thread per stream
    fut = run(msg, emit=lambda n: loop.call_soon_threadsafe(q.put_nowait, ("note", n)))

    def done(f):
        try:
            q.put_nowait(("final", f.result()))
        except Exception as e:  # noqa: BLE001
            q.put_nowait(("final", {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32603, "message": str(e)}}))

    fut.add_done_callback(done)            # an asyncio future: the callback runs on the loop

    async def events():
        while True:
            try:
                kind, m = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield f"event: message\ndata: {json.dumps(m, ensure_ascii=False)}\n\n"
            if kind == "final":
                return

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("")
def mcp_get():
    return Response(status_code=405, headers={"Allow": "POST"})


@router.delete("")
def mcp_delete():
    return Response(status_code=405, headers={"Allow": "POST"})
