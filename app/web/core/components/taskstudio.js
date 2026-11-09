// TaskStudio: the standard studio page (DESIGN 3 "Studio page layout") built from the shared components.
//   left:   task tabs · model picker · inputs · prompt box (+ Improve) · parameter form · presets · Generate (+ ETA)
//   center: results of this studio (live job placeholders, cards, multi-select -> compare / export)
//   drawer: asset details (shared)
// Studios can use it as is (see studios/selftest.js) or compose the same components themselves.
import { api } from '../api.js';
import { store, bus, prefs, promptStyleOf, getProfile, isLicenseBlocked } from '../store.js';
import { h, ic, btn, toast, toastError, fmtDur, empty, clear } from '../ui.js';
import { ModelPicker } from './modelpicker.js';
import { ParamForm, estimateSeconds } from './paramform.js';
import { PromptBox, promptMeta } from './promptbox.js';
import { ResultsGrid } from './results.js';
import { AssetSlot } from './assetpicker.js';
import { openAsset, exportAssets } from './drawer.js';
import { compare } from './compare.js';
import { jobCard } from './jobs.js';
import { costSpan } from '../cost.js';
import { uiInfo } from '../uiinfo.js';
import { TabGroups } from './tabgroups.js';

/**
 * new TaskStudio({
 *   studio: 'sound',                       // studio id (stored in asset meta.studio, used for "Regenerate")
 *   tasks: [{id: 'sfx', label: 'SFX', icon, group: 'create', result: 'audio', prompt: true, promptLabel, placeholder,
 *            inputs: [{key: 'ref', label: 'Reference', types: ['audio'], role: 'ref', multiple: false, required: false}],
 *            exclude: ['text'],             // params rendered elsewhere (e.g. the prompt box)
 *            promptParam: null,             // send the prompt as this param name instead of `prompt`
 *            tool: null}],                  // submit as a CPU/GPU tool instead of a model job
 *   groups: [{id: 'create', label: 'Create'}],  // optional: ordered workflow steps; tasks with `group` show as pills under their step
 *   numberedGroups: false,                 // number the steps (1 Create · 2 Refine …) for a real pipeline
 *   includeTest: false,
 *   beforeSubmit(spec, kit) -> spec | null // last chance to adjust (return null to abort)
 *   extraPanel(kit, task) -> Element|null  // inserted after the parameters
 *   centerHeader(kit) -> Element|null
 * })
 *   .el .task .picker .prompt .form .grid  .setTask(id)  .route(route)  .receive(payload)  .generate()
 */
export class TaskStudio {
  constructor(o) {
    this.o = o; this.studio = o.studio;
    this.tasks = o.tasks;
    this.task = prefs.get(`task.${o.studio}`, o.tasks[0].id);
    if (!this.tasks.find(t => t.id === this.task)) this.task = this.tasks[0].id;
    this.slots = {};
    this.el = h('div.studio');
    this.tg = new TabGroups({ items: o.tasks, groups: o.groups, value: this.task, numbered: o.numberedGroups, label: 'Tasks', onSelect: id => this.setTask(id),
      remember: { get: g => prefs.get(`taskgrp.${o.studio}.${g}`), set: (g, id) => prefs.set(`taskgrp.${o.studio}.${g}`, id) } });
    this.tabs = this.tg.el;
    this.panelBody = h('div.sp-body');
    this.genBtn = h('button.btn.btn-primary.gen-btn', { type: 'button', on: { click: () => this.generate() } });
    this.jobsEl = h('div.col', { style: { gap: '8px' } });
    this.panel = h('div.studio-panel', this.tabs, this.panelBody, h('div.sp-foot', this.jobsEl, h('div', { style: { height: '8px' } }), this.genBtn));
    this.selBar = h('div.row');
    this.center = h('div.studio-center');
    this.el.append(this.panel, this.center);
    this.grid = new ResultsGrid({ selectable: true, type: this.cur().result, onOpen: a => openAsset(a, { list: this.grid.assets().map(x => x.id) }),
      emptyState: empty('sparkles', 'Your results appear here', 'Pick a model, describe what you want, press Generate. Every take lands in the Library with its full recipe.') });
    this.grid.onSelectionChange(s => this._renderSel(s));
    this.grid.onChange(() => this.headTitle && this._renderSel(this.grid.selected()));
    this.picker = new ModelPicker({ task: this.task, includeTest: o.includeTest, onChange: m => this._onModel(m) });
    this.form = new ParamForm({ memoryKey: o.studio, onChange: () => this._eta() });
    this.prompt = new PromptBox({ memoryKey: o.studio, label: 'Prompt', onSubmit: () => this.generate(),
      onApply: extras => this.form.setValues(extras), imageAsset: () => this._firstImageInput(), context: () => this.o.context?.(this) || '',
      params: () => this.form.allValues(), inputs: () => Object.values(this.slots || {}).flatMap(s => s.inputs()) });
    this.prompt.ta?.addEventListener('input', () => { clearTimeout(this._etaT); this._etaT = setTimeout(() => this._eta(), 350); });
    this._buildCenter();
    this._buildPanel();
    this._offAsset = bus.on('asset', ev => {
      if (ev.action === 'created' && ev.asset.studio === this.studio && ev.asset.project === store.project && this.cur().id === ev.asset.task && !this.grid.cards.has(ev.asset.id) && !this._tracking) this.grid.add(ev.asset);
    });
    this._offProj = bus.on('project', () => this.loadResults());
    this._offProfile = bus.on('profile', () => { if (this.promptOn) this.prompt.ta.placeholder = this._placeholder(this.cur()); });
    this._offLic = bus.on('license', () => this._eta());
    this.loadResults();
  }
  cur() { return this.tasks.find(t => t.id === this.task) || this.tasks[0]; }
  setTask(id) {
    if (!this.tasks.find(t => t.id === id)) return;
    this.task = id; prefs.set(`task.${this.studio}`, id);
    this.tg.set(id);
    this.picker.setTask(id);
    this._buildPanel();
    this.grid.gridEl.classList.toggle('audio-grid', this.cur().result === 'audio');
    this.loadResults();
  }
  _buildPanel() {
    const t = this.cur();
    clear(this.panelBody);
    this.panelBody.append(this.picker.el);
    for (const s of Object.values(this.slots)) s.destroy();
    this.slots = {};
    for (const inp of t.inputs || []) {
      const slot = new AssetSlot({ label: inp.label, types: inp.types, multiple: inp.multiple, max: inp.max || 10, role: inp.role || 'input', hint: inp.hint, task: t.id, info: inp.info, onChange: () => this._eta() });
      this.slots[inp.key || inp.role || 'input'] = slot;
      this.panelBody.append(slot.el);
    }
    const m = this.picker.model;
    this.promptOn = t.prompt !== false && (t.prompt === true || !!promptStyleOf(m, this.task));
    if (this.promptOn) {
      this.prompt.ta.placeholder = this._placeholder(t);
      this.panelBody.append(this.prompt.el);
    }
    this.form.exclude = new Set(t.exclude || []);
    this.panelBody.append(h('div.field', h('div.field-label', ic('settings-2', 14), 'Settings', uiInfo('settings'))), this.form.el);
    const extra = this.o.extraPanel?.(this, t);
    if (extra) this.panelBody.append(extra);
    this._onModel(m, true);
  }
  /** `placeholder` may be a string or a function (profile) -> string, so the hint follows the use-case profile. */
  _placeholder(t) {
    const p = typeof t.placeholder === 'function' ? t.placeholder(getProfile()) : t.placeholder;
    return p || 'Describe what you want… (Thai or English)';
  }
  _onModel(m, fromBuild = false) {
    // the pick itself is remembered by ModelPicker.set() (per project); silent re-resolves must not overwrite that memory
    this.form.setModel(m, this.task);
    this.prompt.setModel(m, this.task);
    this._eta();
  }
  _firstImageInput() {
    for (const s of Object.values(this.slots)) { const id = s.ids.find(i => s.metas.get(i)?.type === 'image'); if (id) return id; }
    return null;
  }
  _eta() {
    const m = this.picker.model;
    const loaded = (store.memory?.loaded || []).some(e => e.key?.endsWith(':' + m?.id));
    const s = estimateSeconds(m, this.form.allValues(), loaded);
    const blocked = isLicenseBlocked(m);
    const ready = m?.status?.status === 'ready' && !blocked;
    const nIn = Object.values(this.slots || {}).reduce((n, sl) => n + sl.ids.length, 0);
    this.genBtn.replaceChildren(ic('sparkles', 18), h('span', ready || !m ? 'Generate' : (m.status.status === 'cloud_disabled' ? 'Cloud disabled' : m.status.status === 'missing_key' ? 'Needs API key' : 'Model not ready')),
      s ? h('span.eta', `· ~${fmtDur(s)}`) : null, ready ? costSpan(m, { ...this.form.allValues(), prompt: this.prompt?.value }, nIn) : null);
    this.genBtn.disabled = !m || !ready;
    this.genBtn.title = m && !ready ? (blocked ? 'Commercial-safe models only is on: this model is not licensed for commercial use.' : (m.status.reason || '')) : 'Generate (Ctrl/⌘ + Enter in the prompt)';
  }
  _buildCenter() {
    this.headTitle = h('div.grow', { style: { minWidth: 0 } });
    const head = h('div.sc-head', this.headTitle, this.selBar, this.o.centerHeader?.(this));
    this.center.append(head, h('div.sc-body', this.grid.el));
    this._renderSel([]);
  }
  _renderSel(sel) {
    clear(this.selBar);
    const t = this.cur();
    this.headTitle.replaceChildren(h('div', { style: { fontWeight: 650 } }, t.label), h('div.dim', { style: { fontSize: '12px' } }, `${this.grid.cards.size} result(s) in ${store.project}`));
    if (!sel.length) return;
    this.selBar.append(h('span.badge.accent', `${sel.length} selected`),
      sel.length === 2 ? btn('Compare', { size: 'sm', icon: 'columns-2', onClick: () => compare(sel[0], sel[1]) }) : null,
      btn('Export', { size: 'sm', icon: 'package', onClick: () => exportAssets(sel) }),
      btn('', { size: 'sm', kind: 'ghost', icon: 'x', title: 'Clear selection', onClick: () => this.grid.clearSelection() }));
  }
  async loadResults() {
    try {
      const r = await api.assets({ project: store.project, studio: this.studio, task: this.task, limit: 120 });
      this.grid.set(r.items);
      this._renderSel(this.grid.selected());
    } catch (e) { toastError(e); }
  }
  spec() {
    const t = this.cur(); const m = this.picker.model;
    const params = { ...this.form.values() };
    const meta = { studio: this.studio };
    if (this.promptOn) {
      const pm = promptMeta(this.prompt);
      Object.assign(meta, pm);
      params[t.promptParam || 'prompt'] = this.prompt.value;
      if (this.prompt.negative) params.negative = this.prompt.negative;
    }
    const inputs = Object.values(this.slots).flatMap(s => s.inputs());
    return { task: t.tool ? undefined : this.task, tool: t.tool || undefined, model: m?.id, params, inputs, project: store.project, meta,
      title: (this.promptOn && this.prompt.value ? this.prompt.value.slice(0, 60) : '') || `${m?.name || t.label}` };
  }
  async generate() {
    const t = this.cur(); const m = this.picker.model;
    if (!m) return toast('Pick a model first.', { kind: 'warn' });
    for (const inp of t.inputs || []) {
      const s = this.slots[inp.key || inp.role || 'input'];
      if (inp.required && !s.ids.length) { toast(`Add the ${inp.label.toLowerCase()} first.`, { kind: 'warn' }); return; }
    }
    if (this.promptOn && t.promptRequired !== false && !this.prompt.value) { this.prompt.focus(); toast('Write a prompt first.', { kind: 'warn' }); return; }
    let spec = this.spec();
    if (this.o.beforeSubmit) { spec = await this.o.beforeSubmit(spec, this); if (!spec) return; }
    try {
      const job = await api.submit(spec);
      store.jobs.set(job.id, job);
      this._tracking = true;
      const n = Number(spec.params.batch || 1);
      this.grid.trackJob(job, { type: t.result || m.modality, count: n, onDone: j => {
        this._tracking = false;
        if (j.status === 'failed') toast(j.error?.message || 'Job failed', { kind: 'err', title: 'Generation failed' });
      } });
      const card = jobCard(job, { compact: true });
      this.jobsEl.prepend(card);
      const off = bus.on('job:' + job.id, j => { if (['done', 'cancelled'].includes(j.status)) { setTimeout(() => card.remove(), 1800); off(); } if (j.status === 'failed') off(); });
      while (this.jobsEl.children.length > 3) this.jobsEl.lastChild.remove();
    } catch (e) { toastError(e, 'Cannot start the job'); }
  }
  /** Shell calls this on each navigation into the studio (#/<studio>/<task>?...). */
  route(r) {
    // #/<studio>/<task>, #/<studio>?task=<task>, or #/<studio>/<step> (a step id opens its last-used task)
    const want = r.sub || r.query?.task;
    if (!want) return;
    const id = this.tasks.find(t => t.id === want) ? want : this.tg.selectGroup(want);
    if (id && id !== this.task) this.setTask(id);
  }
  /** sendTo payload: {action, assets: [meta], params}. regenerate -> restore task/model/params/prompt; else fill inputs. */
  async receive(p) {
    const assets = p.assets || [];
    if (p.action === 'prompt') {   // from the Home idea box: {params: {prompt, task, lyrics?, text?}}
      if (p.params?.task && this.tasks.find(t => t.id === p.params.task)) this.setTask(p.params.task);
      if (this.promptOn && p.params?.prompt) this.prompt.set(p.params.prompt);
      const extra = { ...p.params }; delete extra.prompt; delete extra.task;
      if (Object.keys(extra).length) this.form.setValues(extra);
      return;
    }
    if (p.action === 'regenerate' && assets[0]) {
      const a = assets[0];
      if (a.task && this.tasks.find(t => t.id === a.task)) this.setTask(a.task);
      if (a.model && store.modelsById.get(a.model)?.tasks.includes(this.task)) this.picker.set(a.model, true), this._onModel(this.picker.model);
      this.form.setValues({ ...(a.params || {}), seed: a.seed ?? a.params?.seed });
      if (this.promptOn) { this.prompt.set(a.improved_prompt || a.prompt || a.x_style?.user_prompt || a.params?.prompt || ''); this.prompt.setNegative(a.x_style?.user_negative ?? a.negative ?? ''); }
      const styleRefs = a.x_style?.added?.refs || [];     // reference images the project style attached are not inputs of the user's recipe
      const parents = (a.parents || []).filter(id => !styleRefs.includes(id));
      const slots = Object.values(this.slots);
      if (parents.length && slots[0]) slots[0].set(parents.slice(0, slots[0].multiple ? slots[0].max : 1));
      toast('Settings restored from the asset: tweak and generate.', { kind: 'info' });
      return;
    }
    const target = (this.tasks.find(t => t.receives?.includes(p.action)) || this.cur());
    if (target.id !== this.task) this.setTask(target.id);
    const slot = Object.values(this.slots)[0];
    if (slot && assets.length) slot.set(assets.map(a => a.id).slice(0, slot.multiple ? slot.max : 1));
  }
  destroy() { this._offAsset(); this._offProj(); this._offProfile(); this._offLic(); this.grid.destroy(); this.picker.destroy(); }
}
