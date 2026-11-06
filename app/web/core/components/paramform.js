// Schema-driven parameter form (registry `params`), basic fields first and "Advanced" collapsed, presets
// (Fast / Balanced / Quality from the registry + user-saved). Values of fields that exist in several models
// (steps, seed, size, batch, seconds, ...) are kept when the model changes; per-studio memory in localStorage.
import { api } from '../api.js';
import { prefs, bus, profileAspect, getProfile } from '../store.js';
import { h, ic, seg, toggle, btn, menu, promptDialog, toast, toastError, confirmDialog } from '../ui.js';

const BUILTIN = ['fast', 'balanced', 'quality'];
const choiceVal = c => (typeof c === 'object' ? c.value : c);
const choiceLab = c => (typeof c === 'object' ? c.label : String(c));

/**
 * new ParamForm({model, task, memoryKey, onChange, exclude: ['prompt', ...], renderers: {type: fn}})
 *   .el  .values() -> {name: value}  .setModel(model, task)  .setTask(task)  .setValues(obj)  .applyPreset(name)
 * `renderers` lets a studio add custom field types: renderers.mytype = (param, value, set) => element.
 */
export class ParamForm {
  constructor({ model = null, task = null, memoryKey = 'default', onChange, exclude = [], renderers = {} } = {}) {
    this.model = model; this.task = task; this.memoryKey = memoryKey; this.onChange = onChange; this.exclude = new Set(exclude);
    this.renderers = renderers;
    this.mem = prefs.get(`params.${memoryKey}`, {});     // shared values by param name (survive model switches)
    this.vals = {};
    this.userPresets = {};
    this.advOpen = prefs.get('params.advOpen', false);
    this.el = h('div.pf');
    if (model) this.setModel(model, task);
    // use-case profile: a profile change sets the aspect ratio of image / video models (16:9 video, 9:16 social, 1:1 podcast cover)
    this._hadAspect = !!getProfile().aspect;
    this._offProfile = bus.on('profile', prof => {
      const p = (this.model?.params || []).find(x => x.name === 'aspect');
      const a = p && profileAspect(p.choices);
      if (a) this.setValues({ aspect: a });
      else if (p && this._hadAspect && !prof?.aspect && p.default !== undefined) this.setValues({ aspect: p.default });   // back to game / general: the model default
      this._hadAspect = !!prof?.aspect;
    });
  }
  params() {
    return (this.model?.params || []).filter(p => !this.exclude.has(p.name) && (!p.tasks || !this.task || p.tasks.includes(this.task))
      && (!p.show_if || Object.entries(p.show_if).every(([k, v]) => this.vals[k] === v)));
  }
  _valid(p, v) {
    if (v === undefined || v === null || v === '') return false;
    if (p.type === 'choice') return (p.choices || []).some(c => String(choiceVal(c)) === String(v));
    if (p.type === 'int' || p.type === 'float') return typeof v === 'number' && !isNaN(v);
    if (p.type === 'bool') return typeof v === 'boolean';
    if (p.type === 'seed') return Number.isInteger(v);
    return true;
  }
  _coerce(p, v) {
    if (p.type === 'int') { v = Math.round(Number(v)); if (p.min != null) v = Math.max(p.min, v); if (p.max != null) v = Math.min(p.max, v); }
    else if (p.type === 'float') { v = Number(v); if (p.min != null) v = Math.max(p.min, v); if (p.max != null) v = Math.min(p.max, v); }
    else if (p.type === 'choice') { const c = (p.choices || []).find(c => String(choiceVal(c)) === String(v)); v = c !== undefined ? choiceVal(c) : p.default; }
    else if (p.type === 'seed') v = parseInt(v, 10);
    return v;
  }
  setModel(model, task = this.task) {
    this.model = model; this.task = task;
    const next = {};
    for (const p of model?.params || []) {
      const remembered = this.mem[p.name];
      // first use of an `aspect` choice: start from the profile's aspect ratio (none for game / general: model default)
      if (p.name === 'aspect' && remembered === undefined && p.type === 'choice') { const a = profileAspect(p.choices); if (a) { next[p.name] = a; continue; } }
      next[p.name] = this._valid(p, remembered) ? this._coerce(p, remembered) : (p.default ?? (p.type === 'seed' ? -1 : p.type === 'bool' ? false : ''));
      if (p.type === 'seed' && next[p.name] == null) next[p.name] = -1;
    }
    this.vals = next;
    this.userPresets = {};
    if (model && !model.test) api.presets(model.id).then(r => { this.userPresets = r.user || {}; this.render(); }).catch(() => {});
    this.render();
    this._changed(false);
  }
  setTask(task) { this.task = task; this.render(); }
  values() { const out = {}; for (const p of this.params()) out[p.name] = this.vals[p.name]; return out; }
  allValues() { return { ...this.vals }; }
  setValues(obj, { silent = false } = {}) {
    for (const [k, v] of Object.entries(obj || {})) {
      const p = (this.model?.params || []).find(x => x.name === k);
      if (p) { this.vals[k] = this._coerce(p, v); this.mem[k] = this.vals[k]; }
    }
    prefs.set(`params.${this.memoryKey}`, this.mem);
    this.render();
    if (!silent) this._changed(true);
  }
  _set(name, v) {
    this.vals[name] = v; this.mem[name] = v; prefs.set(`params.${this.memoryKey}`, this.mem);
    if ((this.model?.params || []).some(p => p.show_if && name in p.show_if)) this.render(); else this._markPreset();
    this._changed(true);
  }
  _changed(user) { this.onChange?.(this.values(), user); }
  presetMatch() {
    const all = { ...(this.model?.presets || {}), ...this.userPresets };
    for (const [name, pv] of Object.entries(all)) {
      const keys = Object.keys(pv); if (!keys.length) continue;
      if (keys.every(k => String(this.vals[k]) === String(pv[k]))) return name;
    }
    if (this.model?.presets?.balanced && !Object.keys(this.model.presets.balanced).length) {
      const defaults = (this.model.params || []).every(p => p.type === 'seed' || String(this.vals[p.name]) === String(p.default));
      if (defaults) return 'balanced';
    }
    return null;
  }
  applyPreset(name) {
    const pv = this.model?.presets?.[name] ?? this.userPresets[name];
    if (!pv) return;
    const vals = { ...pv };
    if (BUILTIN.includes(name)) for (const p of this.model.params) if (!(p.name in vals) && p.type !== 'seed' && p.default !== undefined && !['batch', 'seconds', 'text', 'lyrics', 'instruct', 'instruction', 'ref_text', 'transparent', 'aspect'].includes(p.name) && p.group !== 'basic') vals[p.name] = p.default;
    this.setValues(vals);
  }
  _markPreset() {
    const cur = this.presetMatch();
    this.presetEl?.querySelectorAll('[data-preset]').forEach(b => b.classList.toggle('on', b.dataset.preset === cur));
  }
  _presetBar() {
    const m = this.model; if (!m) return null;
    const builtin = Object.keys(m.presets || {});
    const user = Object.keys(this.userPresets || {});
    if (!builtin.length && !user.length && m.test) return null;
    const bar = h('div.presets');
    for (const n of BUILTIN.filter(n => builtin.includes(n)).concat(builtin.filter(n => !BUILTIN.includes(n)))) {
      bar.append(h('button.chip', { type: 'button', dataset: { preset: n }, on: { click: () => this.applyPreset(n) } }, n[0].toUpperCase() + n.slice(1)));
    }
    for (const n of user) bar.append(h('button.chip', { type: 'button', dataset: { preset: n }, on: { click: () => this.applyPreset(n), contextmenu: e => { e.preventDefault(); this._presetMenu(e.currentTarget, n); } } }, ic('star', 12), n));
    if (!m.test) {
      const more = h('button.chip', { type: 'button', title: 'Save / manage presets', on: { click: () => this._presetMenu(more) } }, ic('plus', 13), 'Preset');
      bar.append(more);
    }
    this.presetEl = bar;
    return h('div.field', h('div.field-label', ic('sliders-horizontal', 14), 'Presets'), bar);
  }
  _presetMenu(anchor, only) {
    const user = Object.keys(this.userPresets || {});
    menu(anchor, [
      !only ? { label: 'Save current settings as preset…', icon: 'plus', onClick: async () => {
        const name = await promptDialog('Preset name', { title: 'Save preset', placeholder: 'e.g. UI clicks, Boss theme' });
        if (!name) return;
        const vals = this.values(); for (const p of this.params()) if (p.type === 'seed' || ['text', 'lyrics', 'instruct', 'instruction', 'ref_text'].includes(p.name)) delete vals[p.name];
        try { const r = await api.savePreset(this.model.id, name, vals); this.userPresets = r.user; this.render(); toast(`Preset “${name}” saved`, { kind: 'ok' }); } catch (e) { toastError(e); }
      } } : null,
      ...(only ? [only] : user).map(n => ({ label: `Delete “${n}”`, icon: 'trash-2', danger: true, onClick: async () => {
        if (!(await confirmDialog(`Delete the preset “${n}”?`, { ok: 'Delete', danger: true }))) return;
        try { const r = await api.deletePreset(this.model.id, n); this.userPresets = r.user; this.render(); } catch (e) { toastError(e); }
      } })),
    ]);
  }
  _field(p) {
    const v = this.vals[p.name];
    const set = nv => this._set(p.name, nv);
    const label = h('span.field-label', p.label || p.name);
    const help = p.help ? h('span.field-help', p.help) : null;
    if (this.renderers[p.type]) return h('div.field', label, this.renderers[p.type](p, v, set, this), help);
    switch (p.type) {
      case 'int': case 'float': {
        const step = p.step ?? (p.type === 'int' ? 1 : 0.1);
        const valEl = h('span.val');
        const fmt = x => `${p.type === 'float' && step < 1 ? Number(x).toFixed(String(step).split('.')[1]?.length || 1) : x}${p.unit ? ' ' + p.unit : ''}`;
        valEl.textContent = fmt(v);
        label.append(valEl);
        if (p.min != null && p.max != null && (p.max - p.min) / step <= 2000) {
          const r = h('input', { type: 'range', min: p.min, max: p.max, step, value: v, 'aria-label': p.label });
          const pct = () => r.style.setProperty('--p', `${((r.value - p.min) / (p.max - p.min)) * 100}%`);
          pct();
          r.addEventListener('input', () => { pct(); valEl.textContent = fmt(r.value); });
          r.addEventListener('change', () => set(this._coerce(p, r.value)));
          r.addEventListener('dblclick', () => { r.value = p.default; pct(); valEl.textContent = fmt(p.default); set(p.default); });
          return h('label.field', label, r, help);
        }
        const n = h('input.input.input-sm', { type: 'number', min: p.min, max: p.max, step, value: v, on: { change: e => { const c = this._coerce(p, e.target.value); e.target.value = c; valEl.textContent = fmt(c); set(c); } } });
        return h('label.field', label, n, help);
      }
      case 'seed': {
        const n = h('input.input.input-sm.grow', { type: 'number', value: v, min: -1, placeholder: 'random', on: { change: e => set(parseInt(e.target.value || '-1', 10)) } });
        const rnd = h('button.btn.btn-sm', { type: 'button', title: 'Random each time (-1)', class: v === -1 ? 'on' : '', on: { click: () => { n.value = -1; set(-1); } } }, ic('dice-5', 15), 'Random');
        const lock = h('button.btn.btn-sm.btn-icon', { type: 'button', title: 'Pick a fixed seed', on: { click: () => { const s = Math.floor(Math.random() * 2 ** 31); n.value = s; set(s); } } }, ic('lock', 15));
        label.append(h('span.val', v === -1 ? 'random' : 'fixed'));
        return h('div.field', label, h('div.row', n, rnd, lock), help);
      }
      case 'choice': {
        const short = (p.choices || []).length <= 5 && (p.choices || []).every(c => choiceLab(c).length <= 8);
        if (short) return h('div.field', label, seg((p.choices || []).map(c => ({ value: choiceVal(c), label: choiceLab(c) + (p.unit && /^\d+$/.test(choiceLab(c)) ? '' : '') })), v, nv => set(nv)), help);
        const s = h('select.select', { 'aria-label': p.label, on: { change: e => set(this._coerce(p, e.target.value)) } },
          ...(p.choices || []).map(c => h('option', { value: String(choiceVal(c)), selected: String(choiceVal(c)) === String(v) }, choiceLab(c) + (p.unit && /^\d+$/.test(choiceLab(c)) ? ` ${p.unit}` : ''))));
        return h('label.field', label, s, help);
      }
      case 'bool':
        return h('div.field', h('div.row', h('span.field-label.grow', p.label || p.name), toggle(v, nv => set(nv), p.label)), help);
      case 'textarea': case 'lyrics': {
        const t = h('textarea.textarea', { value: v ?? '', rows: p.type === 'lyrics' ? 8 : 3, placeholder: p.placeholder || '', style: p.type === 'lyrics' ? { fontFamily: 'var(--mono)', fontSize: '12.5px' } : null,
          on: { input: e => { this.vals[p.name] = e.target.value; this.mem[p.name] = e.target.value; }, change: e => set(e.target.value) } });
        const extra = p.type === 'lyrics' ? h('div.chips', ...['[intro]', '[verse]', '[chorus]', '[bridge]', '[instrumental]', '[outro]'].map(tag =>
          h('button.chip', { type: 'button', on: { click: () => { const pos = t.selectionStart ?? t.value.length; t.value = t.value.slice(0, pos) + (pos && t.value[pos - 1] !== '\n' ? '\n' : '') + tag + '\n' + t.value.slice(pos); set(t.value); t.focus(); } } }, tag))) : null;
        return h('label.field', label, t, extra, help);
      }
      case 'asset': case 'assets':
        return h('div.field', label, h('div.field-help', 'Asset inputs are handled by the studio (inputs[]).'), help);
      default: {
        const i = h('input.input', { value: v ?? '', placeholder: p.placeholder || '', on: { input: e => { this.vals[p.name] = e.target.value; this.mem[p.name] = e.target.value; }, change: e => set(e.target.value) } });
        return h('label.field', label, i, help);
      }
    }
  }
  render() {
    const ps = this.params();
    const basic = ps.filter(p => (p.group || 'basic') !== 'advanced');
    const adv = ps.filter(p => p.group === 'advanced');
    const kids = [this._presetBar(), ...basic.map(p => this._field(p))];
    if (adv.length) {
      const d = h('details.pf-adv', { open: this.advOpen }, h('summary', ic('chevron-right', 14), `Advanced`, h('span.dim', `(${adv.length})`)), h('div.pf', ...adv.map(p => this._field(p))));
      d.addEventListener('toggle', () => { this.advOpen = d.open; prefs.set('params.advOpen', d.open); });
      kids.push(d);
    }
    if (!ps.length && this.model) kids.push(h('div.field-help', 'This model has no settings.'));
    this.el.replaceChildren(...kids.filter(Boolean));
    this._markPreset();
  }
}

/** Time estimate (seconds) for a model from its registry `est` (+ batch), or null. */
export function estimateSeconds(model, values = {}, loaded = false) {
  const e = model?.est; if (!e) return null;
  const n = Math.max(1, Number(values.batch || 1));
  return (loaded ? 0 : (e.load_s || 0)) + (e.run_s || 0) * n;
}
