// DOM helpers, toasts, popovers, menus, modals, formatters. No framework: h() builds elements.
import { icon } from './icons.js';
export { icon };

// append / prepend / replaceChildren ignore null, undefined and false (so `cond ? el : null` works everywhere).
for (const name of ['append', 'prepend', 'replaceChildren']) {
  const orig = Element.prototype[name];
  Element.prototype[name] = function (...nodes) { return orig.apply(this, nodes.flat(Infinity).filter(n => n != null && n !== false)); };
}

/** h('div.card.pad#id', {on:{click}, html, style, dataset, ...attrs}, ...children) */
export function h(sel, attrs, ...kids) {
  if (attrs == null || typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) { kids.unshift(attrs); attrs = {}; }
  const m = sel.match(/^([a-z0-9-]+)?((?:[.#][\w-]+)*)$/i);
  const el = document.createElement((m && m[1]) || 'div');
  if (m && m[2]) for (const part of m[2].match(/[.#][\w-]+/g)) {
    if (part[0] === '.') el.classList.add(part.slice(1)); else el.id = part.slice(1);
  }
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'on') for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === 'class') String(v).split(/\s+/).filter(Boolean).forEach(c => el.classList.add(c));
    else if (k === 'style' && typeof v === 'object') { for (const [sk, sv] of Object.entries(v)) { if (sv == null) continue; if (sk.startsWith('--')) el.style.setProperty(sk, sv); else el.style[sk] = sv; } }
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'value' && 'value' in el) el.value = v;
    else if (k in el && typeof v !== 'string' && k !== 'list') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, kids);
  return el;
}
export function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export function ic(name, size = 18, cls = '') { const t = document.createElement('template'); t.innerHTML = icon(name, size, cls); return t.content.firstChild; }
export function clear(el) { while (el.firstChild) el.firstChild.remove(); return el; }
export function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
export function btn(label, opts = {}) {
  const { icon: i, kind = '', size = '', onClick, title, disabled, type = 'button' } = opts;
  const b = h(`button.btn${kind ? '.btn-' + kind : ''}${size ? '.btn-' + size : ''}${!label ? '.btn-icon' : ''}`, { type, title: title || (typeof label === 'string' ? undefined : undefined), disabled, on: onClick ? { click: onClick } : {} });
  if (i) b.append(ic(i, size === 'sm' ? 15 : 17));
  if (label) b.append(h('span', label));
  if (title) b.title = title;
  if (!label && title) b.setAttribute('aria-label', title);
  return b;
}

// ------------------------------------------------------------------ formatters
export function fmtBytes(n) { if (n == null) return '-'; const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`; }
export function fmtDur(s) { if (s == null || isNaN(s)) return '-'; s = Number(s); if (s < 1) return `${Math.round(s * 1000)} ms`; if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`; const m = Math.floor(s / 60), r = Math.round(s % 60); if (m < 60) return `${m}:${String(r).padStart(2, '0')} min`; return `${Math.floor(m / 60)} h ${m % 60} min`; }
export function fmtClock(s) { s = Math.max(0, Number(s) || 0); const m = Math.floor(s / 60); return `${m}:${(s % 60).toFixed(1).padStart(4, '0')}`; }
export function fmtAgo(iso) {
  if (!iso) return ''; const t = new Date(iso).getTime(); const d = (Date.now() - t) / 1000;
  if (d < 45) return 'just now'; if (d < 3600) return `${Math.round(d / 60)} min ago`; if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  if (d < 7 * 86400) return `${Math.round(d / 86400)} d ago`; return new Date(t).toLocaleDateString();
}
export const debounce = (fn, ms = 250) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export async function copyText(t) {
  try { await navigator.clipboard.writeText(t); } catch { const a = h('textarea', { value: t, style: { position: 'fixed', opacity: 0 } }); document.body.append(a); a.select(); document.execCommand('copy'); a.remove(); }
  toast('Copied to clipboard', { kind: 'ok', timeout: 1600 });
}
export const MOD_ICON = { image: 'image', mesh: 'cuboid', audio: 'audio-lines', voice: 'mic', video: 'video', text: 'type', other: 'package' };
export const isThai = s => /[฀-๿]/.test(s || '');

// ------------------------------------------------------------------ toasts
let toastBox;
export function toast(msg, opts = {}) {
  const { kind = 'info', title, timeout = kind === 'err' ? 8000 : 4200, action } = opts;
  if (!toastBox) { toastBox = h('div.toasts', { role: 'status', 'aria-live': 'polite' }); document.body.append(toastBox); }
  const iconName = { ok: 'circle-check', err: 'circle-alert', warn: 'triangle-alert', info: 'info' }[kind] || 'info';
  const t = h(`div.toast.${kind}`, ic(iconName, 18), h('div.grow', title ? h('div.t-title', title) : null, h('div.t-msg', msg)),
    action ? btn(action.label, { size: 'sm', onClick: () => { action.onClick(); t.remove(); } }) : null,
    btn('', { icon: 'x', size: 'sm', kind: 'ghost', title: 'Dismiss', onClick: () => t.remove() }));
  toastBox.append(t);
  if (timeout) setTimeout(() => t.remove(), timeout);
  return t;
}
export function toastError(e, title = 'Something went wrong') {
  if (e?.cancelled) return;   // the user declined a cost / upload confirmation
  const msg = e?.message || String(e);
  toast(e?.fix ? `${msg} ${e.fix}` : msg, { kind: 'err', title });
}

// ------------------------------------------------------------------ popover / menu
let openPop = null;
export function closePopover() { if (openPop) { const p = openPop; openPop = null; p.close(); } }
export function popover(anchor, content, opts = {}) {
  closePopover();
  const el = h('div.popover', { role: opts.role || 'dialog' }, content);
  if (opts.width) el.style.width = opts.width;
  document.body.append(el);
  const place = () => {
    const r = anchor.getBoundingClientRect(); const pw = el.offsetWidth, ph = el.offsetHeight; const vw = innerWidth, vh = innerHeight;
    let left = opts.align === 'end' ? r.right - pw : r.left; left = Math.max(8, Math.min(left, vw - pw - 8));
    let top = r.bottom + 6; if (top + ph > vh - 8 && r.top - ph - 6 > 8) top = r.top - ph - 6; top = Math.max(8, Math.min(top, vh - ph - 8));
    el.style.left = left + 'px'; el.style.top = top + 'px';
  };
  place();
  const onDoc = e => { if (!el.contains(e.target) && !anchor.contains(e.target)) closePopover(); };
  const onKey = e => { if (e.key === 'Escape') closePopover(); };
  setTimeout(() => { document.addEventListener('pointerdown', onDoc, true); document.addEventListener('keydown', onKey); }, 0);
  addEventListener('resize', place);
  const api = { el, place, close: () => { el.remove(); document.removeEventListener('pointerdown', onDoc, true); document.removeEventListener('keydown', onKey); removeEventListener('resize', place); opts.onClose?.(); if (openPop === api) openPop = null; } };
  openPop = api;
  return api;
}
/** items: [{label, icon, sub, onClick, danger, on, disabled} | {sep:true} | {head:'Title'}] */
export function menu(anchor, items, opts = {}) {
  const box = h('div', { role: 'menu' });
  for (const it of items) {
    if (!it) continue;
    if (it.sep) { box.append(h('div.menu-sep')); continue; }
    if (it.head) { box.append(h('div.menu-head', it.head)); continue; }
    const b = h(`button.menu-item${it.on ? '.on' : ''}`, { role: 'menuitem', disabled: it.disabled, style: it.danger ? { color: 'var(--err)' } : null,
      on: { click: () => { closePopover(); it.onClick?.(); } } },
    it.icon ? ic(it.icon, 16) : null, h('div.grow', h('div', it.label), it.sub ? h('div.sub', it.sub) : null), it.on ? ic('check', 15) : null);
    box.append(b);
  }
  return popover(anchor, box, { role: 'menu', ...opts });
}

// ------------------------------------------------------------------ modal
export function modal({ title, body, foot, width = 560, onClose, icon: i } = {}) {
  const scrim = h('div.scrim');
  const close = () => { scrim.remove(); m.remove(); document.removeEventListener('keydown', onKey); onClose?.(); };
  const onKey = e => { if (e.key === 'Escape') close(); };
  const m = h('div.modal', { role: 'dialog', 'aria-modal': 'true', style: { '--mw': width + 'px' } },
    h('div.modal-head', i ? ic(i, 18) : null, h('h3.grow', title || ''), btn('', { icon: 'x', kind: 'ghost', size: 'sm', title: 'Close', onClick: () => close() })),
    h('div.modal-body', body), foot ? h('div.modal-foot', foot) : null);
  scrim.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.append(scrim, m);
  return { el: m, close };
}
export function confirmDialog(message, { title = 'Are you sure?', ok = 'OK', danger = false } = {}) {
  return new Promise(res => {
    let done = false;
    const md = modal({ title, width: 420, body: h('p', { style: { margin: 0 } }, message), onClose: () => { if (!done) res(false); },
      foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }),
        btn(ok, { kind: danger ? 'danger' : 'primary', onClick: () => { done = true; md.close(); res(true); } })] });
  });
}
export function promptDialog(label, { title = '', value = '', ok = 'Save', placeholder = '' } = {}) {
  return new Promise(res => {
    let done = false;
    const inp = h('input.input', { value, placeholder });
    const submit = () => { done = true; md.close(); res(inp.value.trim()); };
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
    const md = modal({ title: title || label, width: 420, body: h('label.field', h('span.field-label', label), inp), onClose: () => { if (!done) res(null); },
      foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn(ok, { kind: 'primary', onClick: submit })] });
    setTimeout(() => inp.focus(), 30);
  });
}

// ------------------------------------------------------------------ small widgets
export function seg(options, value, onChange) {
  const el = h('div.seg', { role: 'radiogroup' });
  const render = v => { el.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === String(v))); };
  for (const o of options) {
    const val = typeof o === 'object' ? o.value : o; const lab = typeof o === 'object' ? o.label : String(o);
    el.append(h('button', { type: 'button', role: 'radio', dataset: { v: String(val) }, on: { click: () => { render(val); onChange(val); } } }, lab));
  }
  render(value);
  el.set = render;
  return el;
}
export function toggle(checked, onChange, label) {
  const inp = h('input', { type: 'checkbox', checked: !!checked, on: { change: () => onChange(inp.checked) } });
  if (label) inp.setAttribute('aria-label', label);
  const el = h('span.switch', inp, h('span'));
  el.input = inp;
  return el;
}
export function stars(value, onChange, size = 15) {
  const el = h('span.stars', { role: 'radiogroup', 'aria-label': 'Rating' });
  const draw = v => el.querySelectorAll('button').forEach((b, i) => b.classList.toggle('on', i < v));
  for (let i = 1; i <= 5; i++) el.append(h('button', { type: 'button', title: `${i} star${i > 1 ? 's' : ''}`, on: { click: e => { e.stopPropagation(); const nv = value === i ? 0 : i; value = nv; draw(nv); onChange(nv); } } }, ic('star', size)));
  draw(value || 0);
  el.set = v => { value = v; draw(v); };
  return el;
}
export function progressBar(frac, indeterminate = false) {
  const i = h('i', { style: { width: `${Math.round((frac || 0) * 100)}%` } });
  const el = h(`div.progress${indeterminate ? '.indet' : ''}`, i);
  el.set = (f, ind) => { i.style.width = `${Math.round((f || 0) * 100)}%`; el.classList.toggle('indet', !!ind); };
  return el;
}
export function empty(iconName, title, text, action) {
  return h('div.empty', ic(iconName, 34), h('h4', title), text ? h('p', text) : null, action ? h('div', { style: { marginTop: '14px' } }, action) : null);
}
export function statusBadge(st) {
  const s = st?.status || 'broken'; const code = st?.code || s;
  const map = { ready: ['ok', 'Ready'], missing_weights: ['warn', 'Weights missing'], missing_env: ['warn', 'Env missing'], broken: ['err', 'Broken'],
    missing_key: ['warn', 'Needs API key'], cloud_disabled: ['', 'Cloud off'] };
  let [k, label] = map[s] || ['err', s];
  if (code === 'no_worker') { k = ''; label = 'Coming soon'; }
  return h(`span.badge${k ? '.' + k : '.outline'}`, { title: st?.reason || '' }, h(`span.status-dot.${code === 'no_worker' ? 'no_worker' : s}`), label);
}
export function licenseBadge(lic) {
  if (!lic) return null;
  return lic.commercial
    ? h('span.badge.ok', { title: lic.name + (lic.note ? ` — ${lic.note}` : '') }, ic('shield-check', 12), lic.verify ? 'Commercial?' : 'Commercial')
    : h('span.badge.err', { title: lic.name + (lic.note ? ` — ${lic.note}` : '') }, ic('ban', 12), 'Non-commercial');
}
