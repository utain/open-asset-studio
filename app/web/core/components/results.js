// Asset cards and the results grid (image / mesh / audio / voice), with selection, quick rating and live job
// placeholders. Used by Library, Home and every studio's center column.
import { api, thumbUrl } from '../api.js';
import { bus } from '../store.js';
import { h, ic, btn, stars, fmtAgo, fmtDur, MOD_ICON, progressBar, toastError, empty } from '../ui.js';
import { audioRow } from './audio.js';

const modClass = t => `mod-${t === 'mesh' ? 'mesh' : t}`;

export function assetCard(asset, opts = {}) {
  const { onOpen, onToggleSelect, selectable = false, selected = false, showModel = true } = opts;
  const el = h(`div.acard.${modClass(asset.type)}${asset.type === 'audio' || asset.type === 'voice' ? '.audio' : ''}`, { tabindex: 0, dataset: { id: asset.id } });
  const render = a => {
    el.replaceChildren();
    if (a.type === 'audio' || a.type === 'voice') {
      el.append(audioRow(a));
    } else {
      const t = thumbUrl(a);
      const th = h(`div.thumb${a.type === 'mesh' ? '.mesh' : ''}${a.type === 'image' && a.info?.has_alpha ? '.checker' : ''}`);
      if (t) th.append(h('img', { src: t, alt: a.name || '', loading: 'lazy', decoding: 'async' }));
      else th.append(h('div.ph', ic(MOD_ICON[a.type] || 'package', 46)));
      el.append(th);
      const tl = h('div.tl');
      if (a.type === 'mesh') tl.append(h('span.badge.ov', 'GLB'));
      if (a.type === 'video') tl.append(h('span.badge.ov', ic('video', 11), a.info?.duration ? fmtDur(a.info.duration) : 'VIDEO'));
      if (a.info?.has_alpha) tl.append(h('span.badge.ov', 'RGBA'));
      el.append(tl);
    }
    const tr = h('div.tr' + (a.favorite ? '.show' : ''));
    tr.append(h('button.btn.btn-icon.btn-sm.ov', { type: 'button', title: a.favorite ? 'Unfavourite' : 'Favourite', style: { border: 0, color: a.favorite ? '#ff6b8b' : '#fff' },
      on: { click: e => { e.stopPropagation(); api.patchAsset(a.id, { favorite: !a.favorite }).catch(toastError); } } }, ic('heart', 15)));
    if (selectable) {
      const sb = h('button.selbox', { type: 'button', title: 'Select', on: { click: e => { e.stopPropagation(); onToggleSelect?.(a); } } }, ic('check', 14));
      tr.append(sb);
    }
    el.append(tr);
    const info = a.type === 'audio' || a.type === 'voice'
      ? [a.info?.lufs != null ? `${a.info.lufs.toFixed(1)} LUFS` : null, a.info?.sample_rate ? `${(a.info.sample_rate / 1000).toFixed(1)} kHz` : null]
      : a.type === 'image' ? [a.info?.width ? `${a.info.width}×${a.info.height}` : null] : a.type === 'mesh' ? [a.info?.faces ? `${(a.info.faces / 1000).toFixed(1)}k faces` : null]
        : a.type === 'video' ? [a.info?.width ? `${a.info.width}×${a.info.height}` : null, a.info?.bytes ? `${(a.info.bytes / 2 ** 20).toFixed(1)} MB` : null] : [];
    el.append(h('div.meta',
      h('div.row', h('div.name.ellipsis.grow', { title: a.name }, a.name || a.id),
        stars(a.rating || 0, v => api.patchAsset(a.id, { rating: v }).catch(toastError), 13)),
      h('div.sub', h('span.ellipsis.grow', [showModel && a.model_name ? a.model_name : a.type, ...info.filter(Boolean)].join(' · ')),
        h('span.nowrap', fmtAgo(a.created)))));
  };
  render(asset);
  el.addEventListener('click', () => onOpen?.(el._asset));
  el.addEventListener('keydown', e => { if (e.key === 'Enter') onOpen?.(el._asset); if (e.key === ' ' && selectable && e.target === el) { e.preventDefault(); onToggleSelect?.(el._asset); } });
  el._asset = asset;
  el.update = a => { el._asset = a; render(a); el.classList.toggle('sel', el._sel); };
  el.setSelected = v => { el._sel = v; el.classList.toggle('sel', v); };
  el.setSelected(selected);
  return el;
}

function pendingCard(job, type) {
  const bar = progressBar(job.progress?.frac || 0, true);
  const msg = h('div.sub.ellipsis', job.progress?.msg || 'Queued');
  const el = h(`div.acard.pending.${modClass(type)}`, h('div.thumb.skeleton', { style: type === 'audio' ? { aspectRatio: '4 / 1' } : null }, h('div.ph', ic(MOD_ICON[type] || 'loader', 34, 'spin'))),
    h('div.meta', h('div.name.ellipsis', job.title || job.model_name || 'Generating'), bar, msg));
  el.update = j => {
    const p = j.progress || {}; const ind = j.status === 'queued' || p.stage === 'loading' || !p.frac;
    bar.set(p.frac || 0, ind);
    msg.textContent = j.status === 'queued' ? (j.queue_position ? `Queued · #${j.queue_position}` : 'Queued') : (p.msg || p.stage || 'Running');
  };
  return el;
}

/**
 * new ResultsGrid({onOpen, selectable, type, emptyState}) -> .el
 *   set(assets) / add(asset, {prepend}) / update(asset) / remove(id) / trackJob(job) / selected() / clearSelection()
 *   onSelectionChange(fn)
 * Keeps itself in sync with asset events (rating / favourite / delete) for the cards it shows.
 */
export class ResultsGrid {
  constructor({ onOpen, selectable = true, type = null, emptyState = null } = {}) {
    this.onOpen = onOpen; this.selectable = selectable; this.cards = new Map(); this.sel = new Set(); this.selFns = []; this.changeFns = [];
    this.el = h('div.results');
    this.gridEl = h('div.grid' + (type === 'audio' ? '.audio-grid' : ''));
    this.emptyEl = emptyState || empty('layers', 'Nothing here yet', 'Results appear here as soon as a job finishes.');
    this.el.append(this.gridEl, this.emptyEl);
    this.pending = new Map();
    this._off = bus.on('asset', ev => {
      const c = this.cards.get(ev.asset.id);
      if (!c) return;
      if (ev.action === 'deleted') this.remove(ev.asset.id); else c.update(ev.asset);
    });
    this._refreshEmpty();
  }
  _refreshEmpty() { this.emptyEl.classList.toggle('hidden', this.cards.size + this.pending.size > 0); for (const f of this.changeFns || []) f(this.cards.size); }
  onChange(fn) { this.changeFns.push(fn); }
  _card(a) {
    return assetCard(a, { onOpen: x => this.onOpen?.(x, this), selectable: this.selectable, selected: this.sel.has(a.id),
      onToggleSelect: x => this.toggleSelect(x.id) });
  }
  set(assets) {
    this.gridEl.replaceChildren(...[...this.pending.values()]); this.cards.clear();
    for (const id of [...this.sel]) if (!assets.find(a => a.id === id)) this.sel.delete(id);
    for (const a of assets) { const c = this._card(a); this.cards.set(a.id, c); this.gridEl.append(c); }
    this._refreshEmpty(); this._emitSel();
  }
  add(a, { prepend = true } = {}) {
    if (this.cards.has(a.id)) { this.update(a); return; }
    const c = this._card(a); this.cards.set(a.id, c);
    if (prepend) { const firstReal = [...this.gridEl.children].find(x => !x.classList.contains('pending')); this.gridEl.insertBefore(c, firstReal || null); } else this.gridEl.append(c);
    this._refreshEmpty();
  }
  update(a) { this.cards.get(a.id)?.update(a); }
  remove(id) { this.cards.get(id)?.remove(); this.cards.delete(id); this.sel.delete(id); this._refreshEmpty(); this._emitSel(); }
  assets() { return [...this.cards.values()].map(c => c._asset); }
  setActive(id) { for (const [k, c] of this.cards) c.classList.toggle('active', k === id); }
  /** Placeholder card(s) for a running job; replaced by its outputs when it finishes. */
  trackJob(job, { type = 'image', count = 1, onDone } = {}) {
    const cards = [];
    for (let i = 0; i < Math.max(1, Math.min(count, 8)); i++) { const c = pendingCard(job, type); cards.push(c); this.gridEl.prepend(c); }
    this.pending.set(job.id, cards);
    this._refreshEmpty();
    const seen = new Set();
    const off = bus.on('job:' + job.id, async j => {
      cards.forEach(c => c.update(j));
      const fresh = (j.outputs || []).filter(id => !seen.has(id));
      if (fresh.length) {
        fresh.forEach(id => seen.add(id));
        try {
          const r = await api.assets({ ids: fresh.join(','), limit: 50 });
          for (const a of r.items) { const ph = cards.shift(); ph?.remove(); this.add(a); }
        } catch (e) { /* asset events will still update the library */ }
      }
      if (['done', 'failed', 'cancelled'].includes(j.status)) {
        off(); cards.forEach(c => c.remove()); this.pending.delete(job.id); this._refreshEmpty();
        onDone?.(j);
      }
    });
    return () => { off(); cards.forEach(c => c.remove()); this.pending.delete(job.id); this._refreshEmpty(); };
  }
  toggleSelect(id) { if (this.sel.has(id)) this.sel.delete(id); else this.sel.add(id); this.cards.get(id)?.setSelected(this.sel.has(id)); this._emitSel(); }
  selectAll() { for (const id of this.cards.keys()) { this.sel.add(id); this.cards.get(id).setSelected(true); } this._emitSel(); }
  clearSelection() { for (const id of this.sel) this.cards.get(id)?.setSelected(false); this.sel.clear(); this._emitSel(); }
  selected() { return [...this.sel]; }
  onSelectionChange(fn) { this.selFns.push(fn); }
  _emitSel() { for (const f of this.selFns) f(this.selected()); }
  destroy() { this._off(); }
}

/** One-line description of an asset for lists. */
export function assetSubtitle(a) {
  if (a.type === 'audio' || a.type === 'voice') return `${fmtDur(a.info?.duration)} · ${a.model_name || a.source}`;
  if (a.type === 'image') return `${a.info?.width || '?'}×${a.info?.height || '?'} · ${a.model_name || a.source}`;
  return a.model_name || a.source || a.type;
}
