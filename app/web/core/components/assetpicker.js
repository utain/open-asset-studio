// Asset picker (library browser in a modal + upload) and an input slot for studio forms.
import { api, thumbUrl } from '../api.js';
import { store, bus } from '../store.js';
import { h, ic, btn, modal, toastError, debounce, empty, MOD_ICON } from '../ui.js';
import { ResultsGrid } from './results.js';

/** pickAssets({types, multiple, title, exclude}) -> Promise<string[] | null> */
export function pickAssets({ types = null, multiple = false, title = 'Pick an asset', exclude = [] } = {}) {
  return new Promise(resolve => {
    let done = false;
    const q = h('input.input', { placeholder: 'Search name, prompt, tag…', type: 'search' });
    const grid = new ResultsGrid({ selectable: true, onOpen: a => { if (!multiple) { done = true; md.close(); resolve([a.id]); } else grid.toggleSelect(a.id); },
      emptyState: empty('library', 'No matching assets', 'Generate something or upload a file.') });
    const load = async () => {
      try {
        const r = await api.assets({ project: store.project, type: types ? [...new Set(types)].join(',') : undefined, q: q.value || undefined, limit: 120 });
        grid.set(r.items.filter(a => !exclude.includes(a.id)));
      } catch (e) { toastError(e); }
    };
    q.addEventListener('input', debounce(load, 250));
    const file = h('input', { type: 'file', class: 'hidden', multiple, accept: acceptFor(types), on: { change: async () => {
      for (const f of file.files) {
        try { const a = await api.upload(f, { project: store.project }); grid.add(a); grid.toggleSelect(a.id); if (!multiple) { done = true; md.close(); resolve([a.id]); return; } } catch (e) { toastError(e, 'Upload failed'); }
      }
    } } });
    const use = btn('Use selected', { kind: 'primary', onClick: () => { done = true; md.close(); resolve(grid.selected()); } });
    use.disabled = true;
    grid.onSelectionChange(s => { use.disabled = !s.length; use.querySelector('span').textContent = s.length > 1 ? `Use ${s.length} selected` : 'Use selected'; });
    const md = modal({ title, icon: 'library', width: 920, onClose: () => { grid.destroy(); if (!done) resolve(null); },
      body: h('div.col', { style: { gap: '12px' } }, h('div.row', q, btn('Upload', { icon: 'upload', onClick: () => file.click() }), file), grid.el),
      foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), multiple ? use : null] });
    load();
    setTimeout(() => q.focus(), 50);
  });
}

function acceptFor(types) {
  if (!types) return '.png,.jpg,.jpeg,.webp,.wav,.flac,.ogg,.mp3,.mp4,.mov,.webm,.mkv,.glb';
  const m = { image: '.png,.jpg,.jpeg,.webp', audio: '.wav,.flac,.ogg,.mp3', voice: '.wav,.flac,.ogg,.mp3', video: '.mp4,.mov,.webm,.mkv', mesh: '.glb' };
  return [...new Set(types.map(t => m[t]).filter(Boolean))].join(',');
}

/**
 * new AssetSlot({label, types, multiple, max, role, onChange}) -> .el ; .ids ; .set(ids) ; .inputs() -> [{asset, role}]
 * Drop files on it, click to pick from the library. Studios pass slot.inputs() as job `inputs`.
 */
export class AssetSlot {
  constructor({ label = 'Input', types = ['image'], multiple = false, max = 10, role = 'input', onChange, hint = '' } = {}) {
    this.types = types; this.multiple = multiple; this.max = max; this.role = role; this.onChange = onChange; this.ids = []; this.metas = new Map();
    this.box = h('div.slot', { tabindex: 0, role: 'button', 'aria-label': `${label}: pick or drop a file` });
    this.el = h('div.field', h('div.field-label', label, multiple ? h('span.dim', `up to ${max}`) : null), this.box, hint ? h('span.field-help', hint) : null);
    this.box.addEventListener('click', () => this.pick());
    this.box.addEventListener('keydown', e => { if (e.key === 'Enter') this.pick(); });
    this.box.addEventListener('dragover', e => { e.preventDefault(); this.box.classList.add('drag'); });
    this.box.addEventListener('dragleave', () => this.box.classList.remove('drag'));
    this.box.addEventListener('drop', async e => {
      e.preventDefault(); this.box.classList.remove('drag');
      const aid = e.dataTransfer.getData('text/x-asset-id');
      if (aid) { this.add([aid]); return; }
      for (const f of e.dataTransfer.files) { try { const a = await api.upload(f, { project: store.project }); this.metas.set(a.id, a); this.add([a.id]); } catch (err) { toastError(err, 'Upload failed'); } }
    });
    this._off = bus.on('asset', ev => { if (ev.action === 'deleted' && this.ids.includes(ev.asset.id)) this.set(this.ids.filter(i => i !== ev.asset.id)); });
    this.render();
  }
  async pick() { const ids = await pickAssets({ types: this.types, multiple: this.multiple, title: 'Choose input' }); if (ids?.length) this.multiple ? this.add(ids) : this.set(ids); }
  add(ids) { this.set(this.multiple ? [...new Set([...this.ids, ...ids])].slice(0, this.max) : ids.slice(0, 1)); }
  async set(ids) {
    this.ids = ids;
    for (const id of ids) if (!this.metas.has(id)) { try { this.metas.set(id, await api.asset(id)); } catch { /* deleted */ } }
    this.render(); this.onChange?.(this.ids);
  }
  inputs() { return this.ids.map(id => ({ asset: id, role: this.role })); }
  render() {
    this.box.replaceChildren();
    if (!this.ids.length) { this.box.append(h('div.mp-icon', ic(MOD_ICON[this.types[0]] || 'upload', 18)), h('div.grow', h('div', { style: { fontWeight: 600 } }, 'Drop a file or pick from the library'), h('div.dim', { style: { fontSize: '12px' } }, this.types.join(' / ')))); return; }
    for (const id of this.ids) {
      const m = this.metas.get(id);
      const t = thumbUrl(m);
      this.box.append(h('div.row', { style: { gap: '6px' } }, t ? h('img', { src: t, alt: '', class: m?.info?.has_alpha ? 'checker' : '' }) : h('div.mp-icon', ic(MOD_ICON[m?.type] || 'package', 16)),
        !this.multiple ? h('div.grow.ellipsis', m?.name || id) : null,
        h('button.btn.btn-ghost.btn-sm.btn-icon', { type: 'button', title: 'Remove', on: { click: e => { e.stopPropagation(); this.set(this.ids.filter(x => x !== id)); } } }, ic('x', 14))));
    }
    if (this.multiple && this.ids.length < this.max) this.box.append(h('div.dim', ic('plus', 14), ' add'));
  }
  destroy() { this._off(); }
}
