// Model picker: models that can do the current task, with feature / license / status badges. Not-ready models are
// shown greyed out with the reason and the fix command (and cannot be picked).
import { store, bus, modelsFor, prefs } from '../store.js';
import { h, ic, btn, popover, closePopover, statusBadge, licenseBadge, copyText, MOD_ICON } from '../ui.js';
import { cloudBadge, priceBadge, priceLabel, isCloud } from '../cost.js';
import { navigate } from '../studio.js';

const CLOUD_CODES = ['missing_key', 'cloud_disabled'];
function cloudFix(m) {
  return h('div.mp-fix', h('div', m.status.reason), h('div.row', h('span.dim', { style: { fontSize: '12px' } }, m.status.fix || ''),
    btn('Providers', { size: 'sm', icon: 'key', onClick: e => { e.stopPropagation(); closePopover(); navigate('providers'); } })));
}

const FEATURE_BADGES = [
  ['rgba', 'RGBA'], ['text_render', 'Text'], ['max_refs', f => (f > 1 ? `${f} refs` : null)], ['lyrics', 'Lyrics'],
  ['thai', f => (f === true ? 'Thai' : null)], ['needs_ref', f => (f ? 'Clone' : null)], ['instruct', 'Instruct'],
  ['tags', 'Tags'], ['vision', 'Vision'], ['pbr', 'PBR'], ['inpaint', 'Inpaint'], ['native_size', f => `${f}px`],
  ['remote', f => `via ${f}`],
];
export function featureBadges(m) {
  const out = [];
  const f = m.features || {};
  for (const [k, lab] of FEATURE_BADGES) {
    if (!(k in f) || f[k] === false || f[k] == null) continue;
    const t = typeof lab === 'function' ? lab(f[k]) : lab;
    if (t) out.push(h('span.badge.outline', t));
  }
  if (f.negative === true || f.negative === 'cfg') out.push(h('span.badge.outline', 'Negative'));
  for (const t of (m.tags || []).filter(t => ['fastest', 'fast', 'quality'].includes(t))) out.push(h('span.badge.accent', t));
  return out;
}
export function memBadge(m) { return m.memory_gb ? h('span.badge.outline', { title: 'Free memory needed' }, ic('memory-stick', 12), `${m.memory_gb} GB`) : null; }

/**
 * new ModelPicker({task, value, onChange, includeTest, label}) -> .el ; .value ; .model ; .setTask(task) ; .set(id)
 * Remembers the last pick per task (localStorage) and falls back to the registry default.
 */
export class ModelPicker {
  constructor({ task, value = null, onChange, includeTest = false, label = 'Model' } = {}) {
    this.task = task; this.onChange = onChange; this.includeTest = includeTest; this.label = label;
    this.el = h('div.field');
    this.btnEl = h('button.mp-btn', { type: 'button', 'aria-haspopup': 'listbox', on: { click: () => this.open() } });
    this.el.append(h('div.field-label', label), this.btnEl);
    this.value = value || this._initial();
    this._render();
    this._off = bus.on('models', () => { if (!this.model) this.value = this._initial(); this._render(); });
  }
  _initial() {
    const remembered = prefs.get(`model.${this.task}`);
    const list = modelsFor(this.task, { includeTest: this.includeTest });
    const ok = id => list.find(m => m.id === id && m.status.status === 'ready');
    return (ok(remembered) && remembered) || (ok(store.defaults[this.task]) && store.defaults[this.task])
      || list.find(m => m.status.status === 'ready')?.id || store.defaults[this.task] || list[0]?.id || null;
  }
  get model() { return store.modelsById.get(this.value) || null; }
  setTask(task) { this.task = task; const cur = this.model; if (!cur || !cur.tasks.includes(task)) this.value = this._initial(); this._render(); this.onChange?.(this.model, true); }
  set(id, silent = false) { this.value = id; prefs.set(`model.${this.task}`, id); this._render(); if (!silent) this.onChange?.(this.model); }
  _render() {
    const m = this.model;
    this.btnEl.replaceChildren();
    if (!m) { this.btnEl.append(h('div.grow.muted', 'No model for this task')); return; }
    this.btnEl.className = `mp-btn mod-${m.modality}`;
    this.btnEl.append(h('div.mp-icon', ic(MOD_ICON[m.modality] || 'cpu', 18)),
      h('div.grow', { style: { minWidth: 0 } }, h('div.row', h('span.mp-name.ellipsis', m.name), m.status.status !== 'ready' ? statusBadge(m.status) : null),
        h('div.mp-sub.ellipsis', isCloud(m) ? [m.cloud.label, priceLabel(m)].filter(Boolean).join(' · ') : (m.speed || m.summary || ''))),
      licenseBadge(m.license), ic('chevron-down', 16, 'dim'));
  }
  open() {
    const list = modelsFor(this.task, { includeTest: this.includeTest });
    const box = h('div.mp-list', { role: 'listbox' }, h('div.menu-head', `${store.tasks[this.task]?.label || this.task} · ${list.length} models`));
    for (const m of list) {
      const ready = m.status.status === 'ready';
      const item = h(`div.mp-item.mod-${m.modality}${m.id === this.value ? '.on' : ''}${ready ? '' : '.off'}`, { role: 'option', 'aria-selected': m.id === this.value, tabindex: ready ? 0 : -1 },
        h('div.mp-icon', ic(MOD_ICON[m.modality] || 'cpu', 17)),
        h('div.grow', { style: { minWidth: 0 } },
          h('div.row', h('b.ellipsis', m.name), store.defaults[this.task] === m.id ? h('span.badge.accent', 'default') : null, h('span.spacer'), statusBadge(m.status)),
          h('div.mp-sum', m.summary || ''),
          h('div.badges', cloudBadge(m), priceBadge(m), licenseBadge(m.license), memBadge(m), m.speed ? h('span.badge.outline', { title: m.speed }, ic('clock', 12), m.speed.split(/[;(]/)[0].slice(0, 34)) : null, ...featureBadges(m)),
          !ready && CLOUD_CODES.includes(m.status.code) ? cloudFix(m) : null,
          !ready && m.status.code !== 'no_worker' && !CLOUD_CODES.includes(m.status.code) ? h('div.mp-fix', h('div', m.status.reason), m.status.fix ? h('div.row', h('code.mono.ellipsis', m.status.fix), h('button.btn.btn-ghost.btn-sm', { type: 'button', on: { click: e => { e.stopPropagation(); copyText(m.status.fix); } } }, ic('copy', 13))) : null) : null,
          !ready && m.status.code === 'no_worker' ? h('div.mp-fix', { style: { color: 'var(--text-3)' } }, 'Backend worker not implemented yet.') : null));
      if (ready) {
        const pick = () => { closePopover(); this.set(m.id); };
        item.addEventListener('click', pick);
        item.addEventListener('keydown', e => { if (e.key === 'Enter') pick(); });
      }
      box.append(item);
    }
    if (!list.length) box.append(h('div.empty', 'No model supports this task yet.'));
    popover(this.btnEl, box, { width: 'min(460px, calc(100vw - 16px))' });
  }
  destroy() { this._off(); }
}
