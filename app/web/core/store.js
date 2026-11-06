// App state + event bus + SSE connection. Studios read state and subscribe; they never poll the server for jobs.
import { api } from './api.js';
import { licenseBlocked } from './license.js';

const listeners = new Map();
/** bus.on(type, fn) -> unsubscribe. Types: job, job:<id>, asset, assistant, models, system, llm, project, theme, route, sse,
 *  profile (profile object), license ({commercial_only, strict}), export (default export preset id) */
export const bus = {
  on(type, fn) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(fn); return () => listeners.get(type)?.delete(fn); },
  emit(type, data) { for (const fn of [...(listeners.get(type) || [])]) { try { fn(data); } catch (e) { console.error(`[bus] ${type} handler failed`, e); } } },
};

export const prefs = {
  get(k, d) { try { const v = localStorage.getItem('gas.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('gas.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

export const store = {
  project: prefs.get('project', 'default'),
  projects: [],
  tasks: {},
  defaults: {},
  models: [],            // registry entries (public form), refreshed on "models" events
  modelsById: new Map(),
  jobs: new Map(),       // live jobs (id -> job), fed by SSE
  downloads: new Map(),  // model id -> download state, fed by SSE "download" events
  counts: { queued: 0, running: 0 },
  memory: null,
  llm: null,
  ui: {},
  profile: 'general',    // use-case profile id (settings.json "profile"); definitions in store.profiles
  profiles: {},          // id -> {label, summary, aspect, aspects, export, export_presets, hints}
  exportPresets: {},     // id -> {label, summary, audio, image}
  license: { commercial_only: false, strict: false },
  exportDefault: 'original',
  tools: {},
  version: '',
  sse: 'connecting',
};

export function setProject(id) {
  store.project = id; prefs.set('project', id); bus.emit('project', id);
}

// ------------------------------------------------------------------ profile + licence filter (settings.json)
/** The active use-case profile definition: {id, label, summary, aspect, aspects, export, export_presets, hints}.
 *  Studios read defaults from it (`getProfile().aspect` for image / video aspect, `.hints`) and re-read on bus 'profile'. */
export function getProfile() { return store.profiles[store.profile] || { id: store.profile, label: store.profile, aspect: null, aspects: [], export: 'original', export_presets: ['original'], hints: {} }; }
/** First of `choices` the profile prefers as aspect ratio ('16:9', '9:16', '1:1' …), or null to keep the model default. */
export function profileAspect(choices) {
  const a = getProfile().aspect;
  return a && (choices || []).some(c => String(typeof c === 'object' ? c.value : c) === a) ? a : null;
}
/** Export preset used by plain "Export" actions (settings export.preset, else the profile's). */
export function defaultExportPreset() { return store.exportDefault || getProfile().export || 'original'; }
/** Is `model` refused under "Commercial-safe models only"? */
export function isLicenseBlocked(model) { return licenseBlocked(model, store.license); }

function applySettings(s) {
  if (!s) return;
  const prof = s.profile && store.profiles[s.profile] ? s.profile : (s.profile || store.profile);
  const profChanged = prof !== store.profile;
  const lic = { commercial_only: !!s.license?.commercial_only, strict: !!s.license?.strict };
  const licChanged = lic.commercial_only !== store.license.commercial_only || lic.strict !== store.license.strict;
  const exp = s.export?.preset || null;
  const expChanged = (exp || null) !== (store.exportOverride || null);
  store.profile = prof; store.license = lic; store.exportOverride = exp;
  store.exportDefault = exp || store.profiles[prof]?.export || 'original';
  if (profChanged) bus.emit('profile', getProfile());
  if (licChanged) { bus.emit('license', store.license); bus.emit('models', store.models); }
  if (expChanged || profChanged) bus.emit('export', store.exportDefault);
}
export async function loadProfiles(settings) {
  try {
    const r = await api.profiles();
    store.profiles = Object.fromEntries(r.profiles.map(p => [p.id, p]));
    store.exportPresets = Object.fromEntries(r.export_presets.map(p => [p.id, p]));
    applySettings({ profile: r.current, license: r.license, export: settings?.export });
  } catch { /* older core: stay on the general profile */ }
}
/** Change settings.json keys the profile / licence UI owns: patchProfileSettings({profile?, license?, export?}). */
export async function patchProfileSettings(patch) {
  const s = await api.patchSettings(patch);
  applySettings(s);
  return s;
}
export const setProfile = id => patchProfileSettings({ profile: id });
export const setCommercialOnly = (on, strict) => patchProfileSettings({ license: { commercial_only: !!on, ...(strict === undefined ? {} : { strict: !!strict }) } });

export async function loadModels() {
  const r = await api.models({ include_test: true });
  store.models = r.models; store.tasks = r.tasks; store.defaults = r.defaults;
  store.modelsById = new Map(r.models.map(m => [m.id, m]));
  bus.emit('models', store.models);
  return store.models;
}

/** Models usable for a task (test models only when the UI setting allows them or `includeTest`). */
export function modelsFor(task, { includeTest = false, modality, includeBlocked = false } = {}) {
  const showTest = includeTest || store.ui?.show_test_models;
  return store.models.filter(m => (!task || m.tasks.includes(task)) && (!modality || m.modality === modality) && (showTest || !m.test)
    && (includeBlocked || !licenseBlocked(m, store.license)));
}

export function promptStyleOf(model, task) {
  const ps = model?.prompt_style;
  if (!ps) return null;
  if (typeof ps === 'string') return ps;
  return ps[task] || Object.values(ps)[0];
}

// ------------------------------------------------------------------ SSE
let es = null, retryT = null;
export function connectEvents() {
  if (es) es.close();
  es = new EventSource('/api/events');
  es.addEventListener('hello', () => { store.sse = 'live'; bus.emit('sse', 'live'); });
  es.addEventListener('job', e => {
    const j = JSON.parse(e.data);
    store.jobs.set(j.id, j);
    if (store.jobs.size > 300) { for (const [id, v] of store.jobs) { if (['done', 'failed', 'cancelled'].includes(v.status)) { store.jobs.delete(id); if (store.jobs.size <= 250) break; } } }
    bus.emit('job', j); bus.emit('job:' + j.id, j);
  });
  es.addEventListener('asset', e => bus.emit('asset', JSON.parse(e.data)));
  es.addEventListener('settings', e => { try { applySettings(JSON.parse(e.data)); } catch { /* ignore */ } });
  es.addEventListener('assistant', e => bus.emit('assistant', JSON.parse(e.data)));
  es.addEventListener('models', () => loadModels().catch(() => {}));
  es.addEventListener('providers', e => bus.emit('providers', JSON.parse(e.data)));   // keys / switch / limits changed
  es.addEventListener('spend', e => bus.emit('spend', JSON.parse(e.data)));           // a paid call was recorded
  es.addEventListener('download', e => { const d = JSON.parse(e.data); store.downloads.set(d.model, d); bus.emit('download', d); bus.emit('download:' + d.model, d); });
  es.addEventListener('llm', e => { store.llm = JSON.parse(e.data); bus.emit('llm', store.llm); });
  es.addEventListener('system', e => {
    const d = JSON.parse(e.data); store.memory = d.memory; store.counts = d.jobs; store.llm = d.llm;
    bus.emit('system', d);
  });
  es.onerror = () => {
    store.sse = 'offline'; bus.emit('sse', 'offline');
    es.close(); clearTimeout(retryT); retryT = setTimeout(connectEvents, 2500);
  };
}

/** Resolve when job `id` reaches a terminal state (SSE-driven). */
export function waitJob(id) {
  return new Promise(res => {
    const j = store.jobs.get(id);
    if (j && ['done', 'failed', 'cancelled'].includes(j.status)) return res(j);
    const off = bus.on('job:' + id, jj => { if (['done', 'failed', 'cancelled'].includes(jj.status)) { off(); res(jj); } });
  });
}
