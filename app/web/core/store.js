// App state + event bus + SSE connection. Studios read state and subscribe; they never poll the server for jobs.
import { api } from './api.js';
import { licenseBlocked } from './license.js';

const listeners = new Map();
/** bus.on(type, fn) -> unsubscribe. Types: job, job:<id>, asset, assistant, models, system, llm, project, theme, route, sse,
 *  profile (profile object), license ({commercial_only, strict}), export (default export preset id),
 *  settings (the effective settings of the current project were refetched: system + its overrides) */
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
  effective: {},         // effective settings of the current project (system + its overrides): profile, license, export, prompt, defaults, ...
  effectiveProject: null, // the project whose effective settings are in `effective` / `overridden` / `defaults`
  overridden: new Set(), // dotted keys the current project overrides ("license.commercial_only", "defaults.t2i", ...)
  tools: {},
  version: '',
  sse: 'connecting',
};

export function setProject(id) {
  store.project = id; prefs.set('project', id); bus.emit('project', id);
  refreshEffective().catch(() => {});      // the project's overrides decide profile, licence filter, export and default models
  loadModels().catch(() => {});
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

/** Effective value of a dotted setting key for the current project, e.g. getSetting('prompt.context.image'). */
export function getSetting(key, dflt = undefined) {
  let d = store.effective;
  for (const k of String(key).split('.')) { if (d == null || typeof d !== 'object' || !(k in d)) return dflt; d = d[k]; }
  return d == null ? dflt : d;
}
/** Does the current project override this setting (or, for a group such as 'license', any setting inside it)? */
export function isOverridden(key) {
  if (store.overridden.has(key)) return true;
  const pre = key + '.';
  for (const k of store.overridden) if (k.startsWith(pre)) return true;
  return false;
}
function leaves(obj, prefix = '', out = {}) {
  for (const [k, v] of Object.entries(obj || {})) {
    if (v && typeof v === 'object' && !Array.isArray(v)) leaves(v, `${prefix}${k}.`, out); else out[`${prefix}${k}`] = v;
  }
  return out;
}
function nest(flat) {
  const out = {};
  for (const [k, v] of Object.entries(flat)) {
    const parts = k.split('.'); let d = out;
    for (const p of parts.slice(0, -1)) d = d[p] = d[p] || {};
    d[parts.at(-1)] = v;
  }
  return out;
}
let effSeq = 0;
/** Refetch the effective settings of the current project (system settings + its overrides) and apply them. The SSE
 *  `settings` event only triggers this: a system event can carry values the current project overrides. */
export async function refreshEffective() {
  const seq = ++effSeq, project = store.project;
  let r;
  try { r = await api.projectSettings(project); } catch (e) {
    if (e.status !== 404 && e.status !== 405) throw e;
    r = { effective: await api.settings(), overridden: [], resolved: null };      // older core, or the project vanished: system settings
  }
  if (seq !== effSeq) return;                  // a newer refresh (or project switch) is in flight
  store.overridden = new Set(r.overridden || []);
  store.effective = r.effective || {};
  store.effectiveProject = project;
  const defs = r.resolved?.defaults, defsChanged = !!defs && JSON.stringify(defs) !== JSON.stringify(store.defaults);
  if (defsChanged) store.defaults = defs;     // before applySettings: its profile / license events re-read the defaults
  applySettings(r.effective);
  if (defsChanged) bus.emit('models', store.models);
  bus.emit('settings', { project, overridden: store.overridden });
}

function applySettings(s) {
  if (!s) return;
  if (s.ui) {
    const showTest = !!s.ui.show_test_models, was = !!store.ui?.show_test_models;
    store.ui = { ...(store.ui || {}), ...s.ui };
    if (showTest !== was) bus.emit('models', store.models);
  }
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
    // `settings` (bootstrap) is already the effective settings of the current project; /api/profiles is the system's
    applySettings(settings?.profile ? settings : { profile: r.current, license: r.license, export: settings?.export });
    if (settings) { store.effective = { ...store.effective, ...settings }; store.overridden = new Set(settings.overridden || []); store.effectiveProject = store.project; }
  } catch { /* older core: stay on the general profile */ }
}
/** Change settings the profile / licence UI owns: patchEffective({profile?, license?, export?, ...}). Each leaf goes to the
 *  layer that is in effect: the project's overrides when the current project overrides that key, else settings.json. */
export async function patchEffective(patch) {
  const flat = leaves(patch), sys = {}, proj = {};
  for (const [k, v] of Object.entries(flat)) (isOverridden(k) ? proj : sys)[k] = v;
  let s = null;
  if (Object.keys(sys).length) s = await api.patchSettings(nest(sys));
  if (Object.keys(proj).length) await api.patchProjectSettings(store.project, nest(proj));
  await refreshEffective();
  return s;
}
export const patchProfileSettings = patchEffective;      // older name, same behaviour
/** Set project overrides of the current project: setProjectSettings({prompt: {context: {image: 'cozy farm'}}}); null resets a key. */
export async function setProjectSettings(patch) {
  const r = await api.patchProjectSettings(store.project, patch);
  await refreshEffective();
  return r;
}
export const setProfile = id => patchEffective({ profile: id });
export const setCommercialOnly = (on, strict) => patchEffective({ license: { commercial_only: !!on, ...(strict === undefined ? {} : { strict: !!strict }) } });

export async function loadModels() {
  const r = await api.models({ include_test: true, project: store.project });
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
  es.addEventListener('settings', e => {
    try { const d = JSON.parse(e.data); if (d.scope === 'project' && d.project !== store.project) return; } catch { /* refetch anyway */ }
    refreshEffective().catch(() => {});
  });
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
