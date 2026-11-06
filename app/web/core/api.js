// REST client of the studio core (docs/CONTRACTS.md). Every call throws ApiError {message, status, kind, fix}.
export class ApiError extends Error {
  constructor(message, status, kind, fix, data) { super(message); this.status = status; this.kind = kind; this.fix = fix; this.data = data || {}; }
}

// Submit guard (set by core/cost.js): when a job is refused with kind "confirm" (cost over the threshold) or
// "consent" (inputs would be uploaded to a cloud provider), the guard asks the user and returns the spec to resubmit
// (with confirm_cost / confirm_upload) or null to cancel. Every studio gets this through api.submit / api.retry.
let submitGuard = null, submitBefore = null;
export function setSubmitGuard(fn, before = null) { submitGuard = fn; submitBefore = before; }
async function guarded(send, spec, ask = true) {
  let s = spec;
  if (ask && submitBefore) {   // ask first (dry-run precheck), so the normal flow never hits a 409
    s = await submitBefore(s);
    if (!s) { const c = new ApiError('Cancelled: the job was not started.', 0, 'cancelled'); c.cancelled = true; throw c; }
  }
  for (let i = 0; i < 4; i++) {
    try { return await send(s); } catch (e) {
      if (!submitGuard || !['confirm', 'consent'].includes(e.kind)) throw e;
      const next = await submitGuard(e, s);
      if (!next) { const c = new ApiError('Cancelled: the job was not started.', 0, 'cancelled'); c.cancelled = true; throw c; }
      s = next;
    }
  }
  return send(s);
}

async function req(method, url, body, opts = {}) {
  const init = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers['content-type'] = 'application/json'; }
  if (opts.signal) init.signal = opts.signal;
  let r;
  try { r = await fetch(url, init); } catch (e) { if (e.name === 'AbortError') throw e; throw new ApiError('The studio server is not reachable.', 0, 'network'); }
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('json') ? await r.json().catch(() => null) : await r.text();
  if (!r.ok) {
    const msg = (data && (data.error || data.detail)) || (typeof data === 'string' && data) || `HTTP ${r.status}`;
    throw new ApiError(typeof msg === 'string' ? msg : JSON.stringify(msg), r.status, data?.kind, data?.fix, data && typeof data === 'object' ? data : null);
  }
  return data;
}

const qs = o => { const p = new URLSearchParams(); for (const [k, v] of Object.entries(o || {})) if (v !== undefined && v !== null && v !== '') p.set(k, v); const s = p.toString(); return s ? '?' + s : ''; };

export const api = {
  get: (u, q) => req('GET', u + qs(q)), post: (u, b) => req('POST', u, b ?? {}), put: (u, b) => req('PUT', u, b),
  patch: (u, b) => req('PATCH', u, b), del: (u) => req('DELETE', u),

  bootstrap: () => req('GET', '/api/bootstrap'),
  // models / settings
  models: (q) => req('GET', '/api/models' + qs(q)),
  model: (id) => req('GET', `/api/models/${encodeURIComponent(id)}`),
  refreshModels: () => req('POST', '/api/models/refresh'),
  downloads: () => req('GET', '/api/downloads'),
  downloadPlan: (id) => req('GET', `/api/models/${encodeURIComponent(id)}/download`),
  downloadStart: (id, accept_license) => req('POST', `/api/models/${encodeURIComponent(id)}/download`, { accept_license }),
  downloadCancel: (id) => req('POST', `/api/models/${encodeURIComponent(id)}/download/cancel`),
  probe: (backend) => req('POST', '/api/models/probe', { backend }),
  setDefault: (task, model) => req('PUT', '/api/settings/defaults', { task, model }),
  settings: () => req('GET', '/api/settings'),
  patchSettings: (p) => req('PATCH', '/api/settings', p),
  presets: (model) => req('GET', `/api/presets/${encodeURIComponent(model)}`),
  savePreset: (model, name, params) => req('PUT', `/api/presets/${encodeURIComponent(model)}`, { name, params }),
  deletePreset: (model, name) => req('DELETE', `/api/presets/${encodeURIComponent(model)}/${encodeURIComponent(name)}`),
  // projects / assets
  projects: () => req('GET', '/api/projects'),
  createProject: (id, title) => req('POST', '/api/projects', { id, title }),
  assets: (q) => req('GET', '/api/assets' + qs(q)),
  asset: (id) => req('GET', `/api/assets/${id}`),
  patchAsset: (id, p) => req('PATCH', `/api/assets/${id}`, p),
  deleteAsset: (id) => req('DELETE', `/api/assets/${id}`),
  lineage: (id) => req('GET', `/api/assets/${id}/lineage`),
  exportAssets: (ids, opts = {}) => req('POST', '/api/export', { ids, ...opts }),
  setThumb: (id, blob) => { const f = new FormData(); f.append('file', blob, 'thumb.webp'); return req('POST', `/api/assets/${id}/thumb`, f); },
  upload(file, { project = 'default', name = '', tags = '', type = '' } = {}, onProgress) {
    return new Promise((resolve, reject) => {
      const f = new FormData(); f.append('file', file); f.append('project', project); f.append('name', name); f.append('tags', tags); f.append('type', type);
      const x = new XMLHttpRequest(); x.open('POST', '/api/upload');
      x.upload.onprogress = e => onProgress && e.lengthComputable && onProgress(e.loaded / e.total);
      x.onload = () => { let d = null; try { d = JSON.parse(x.responseText); } catch { /* */ } x.status < 300 ? resolve(d) : reject(new ApiError(d?.error || `Upload failed (${x.status})`, x.status, d?.kind)); };
      x.onerror = () => reject(new ApiError('Upload failed: server not reachable', 0, 'network'));
      x.send(f);
    });
  },
  // jobs
  submit: (spec) => guarded(s => req('POST', '/api/jobs', s), spec),
  jobs: (q) => req('GET', '/api/jobs' + qs(q)),
  job: (id) => req('GET', `/api/jobs/${id}`),
  cancel: (id) => req('POST', `/api/jobs/${id}/cancel`),
  retry: (id) => guarded(s => req('POST', `/api/jobs/${id}/retry`, s), {}, false),
  clearJobs: () => req('DELETE', '/api/jobs'),
  // improver
  improve: (body, signal) => req('POST', '/api/improve', body, { signal }),
  improveStyles: () => req('GET', '/api/improve/styles'),
  llmStatus: () => req('GET', '/api/improve/status'),
  llmRelease: () => req('POST', '/api/improve/release'),
  llmWarmup: () => req('POST', '/api/improve/warmup'),
  // text backends + writing tools (app/plugins/assistant.py)
  textBackends: (check_auth = true) => req('GET', '/api/x/assistant/backends' + qs({ check_auth })),
  setTextBackends: (p) => req('PUT', '/api/x/assistant/settings', p),
  write: (body) => req('POST', '/api/x/assistant/write', body),
  // system
  system: () => req('GET', '/api/system'),
  unload: (body) => req('POST', '/api/system/unload', body),
  logs: (name, lines = 300) => req('GET', '/api/system/logs' + qs({ name, lines })),
  selftest: (kind, project) => req('POST', '/api/system/selftest', { kind, project }),
};

/** URL of a file of an asset (meta from the API). name defaults to the main file. */
export function fileUrl(asset, name, download = false) {
  if (!asset) return '';
  const n = name || asset.main;
  return `/files/${asset.project}/${asset.id}/${encodeURIComponent(n)}${download ? '?download=1' : ''}${asset.updated && !download ? (download ? '&' : '?') + 'v=' + encodeURIComponent(asset.updated) : ''}`;
}
export function thumbUrl(asset) {
  if (!asset) return '';
  if (asset.preview?.thumb) return fileUrl(asset, asset.preview.thumb);
  return asset.type === 'image' ? fileUrl(asset) : '';
}
const wfCache = new Map();
export async function waveform(asset) {
  if (!asset?.preview?.waveform) return null;
  const key = asset.id;
  if (!wfCache.has(key)) wfCache.set(key, fetch(fileUrl(asset, asset.preview.waveform)).then(r => r.json()).catch(() => null));
  return wfCache.get(key);
}
