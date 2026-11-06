// Studio module registry, hash router and send-to. See docs/CONTRACTS.md section 6.
//
// A studio module is web/studios/<id>.js with `export default defineStudio({...})`. The shell imports every file
// listed by /api/bootstrap, so adding a page never requires editing core files.
import { bus } from './store.js';

const studios = new Map();
const pendingPayloads = new Map();
let current = null;

/**
 * def = {
 *   id, title, icon, order, color (css var), section: 'studio' | 'workspace' | 'system', nav: true,
 *   mobile: true (show in the phone bottom bar), badge?: () => string,
 *   receives: [{action, label, icon, types: ['image'|'audio'|'mesh'|'voice'], multiple?: bool}],
 *   mount(el, ctx) -> optional {route(r), receive(payload), show(), hide()}  (called once, on first visit)
 * }
 */
export function defineStudio(def) {
  return { section: 'studio', nav: true, mobile: false, order: 50, receives: [], ...def };
}
export function registerStudio(def) { studios.set(def.id, def); return def; }
export function getStudio(id) { return studios.get(id); }
export function allStudios() { return [...studios.values()].sort((a, b) => a.order - b.order); }

export function parseHash(h = location.hash) {
  const raw = h.replace(/^#\/?/, '');
  const [path, q = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  return { studio: parts[0] || 'home', sub: parts[1] || '', rest: parts.slice(2), query: Object.fromEntries(new URLSearchParams(q)) };
}
export function navigate(studio, sub = '', query = null) {
  let h = `#/${studio}${sub ? '/' + encodeURIComponent(sub) : ''}`;
  if (query && Object.keys(query).length) h += '?' + new URLSearchParams(query).toString();
  if (location.hash === h) bus.emit('route', parseHash(h)); else location.hash = h;
}
export function currentRoute() { return current; }
export function _setCurrent(r) { current = r; }

/**
 * Send assets to another studio: sendTo('3d', assetId, {action: 'shape'}) or sendTo('image', [a, b], {action: 'edit'}).
 * The target studio's instance.receive({action, assets: [meta...], params, from}) is called after navigation.
 */
export function sendTo(studioId, assetIds, { action = 'open', params = {}, sub = '', from } = {}) {
  const ids = Array.isArray(assetIds) ? assetIds : (assetIds ? [assetIds] : []);
  const token = Math.random().toString(36).slice(2, 10);
  pendingPayloads.set(token, { action, assetIds: ids, params, from: from || current?.studio });
  navigate(studioId, sub, { send: token });
}
export function takePayload(token) { const p = pendingPayloads.get(token); pendingPayloads.delete(token); return p; }

/** Send-to actions available for an asset (from every registered studio's `receives`). */
export function sendActions(asset) {
  const out = [];
  for (const s of allStudios()) {
    for (const r of s.receives || []) {
      if (!r.types || r.types.includes(asset.type)) out.push({ studio: s.id, studioTitle: s.title, color: s.color, ...r });
    }
  }
  return out;
}
