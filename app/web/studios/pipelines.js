// Pipelines: guided multi-step flows built on the normal jobs (every result is an ordinary asset with lineage).
//   Concept → 3D prop : idea → ✨ improve → 4 fast variations → pick → optional refine edit → transparent cut-out +
//                       recenter → shape → texture → GLB
//   Character kit     : concept → cut-out → 3D shape → texture → designed voice → barks in that voice → SFX → kit zip
//                       (voice description, sample line, barks and SFX default to an ✨ AI read of the character)
// Each step: pick a model / settings, run, look at the result, pick / approve and continue (or redo / skip). The flow
// record (step, picks, job and asset ids) is stored server-side per project (app/plugins/pipelines.py), so a flow can
// be resumed later or from another machine. "Auto-finish" runs the remaining 3D steps with the current settings.
import * as S from '../core/sdk.js';
import { css, EDIT_PRESETS, alphaBgToggle, bestModel, readyModels, runJob, finishJob, meshViewer, autoThumb, flowId } from './i3d/common.js';
const { h, ic, btn } = S;

const KINDS = {
  concept3d: { title: 'Concept → 3D prop', icon: 'cuboid', blurb: 'Idea → 4 fast concepts → pick → refine → transparent cut-out → shape → texture → GLB.',
    steps: ['idea', 'drafts', 'refine', 'cutout', 'shape', 'texture', 'export'], placeholder: 'A treasure chest with gold trim and a glowing gem lock… (Thai or English)', chips: ['3d_ready'] },
  character: { title: 'Character kit', icon: 'user', blurb: 'Concept → 3D model → a designed voice → barks in that voice → footsteps / attack SFX.',
    steps: ['idea', 'drafts', 'cutout', 'shape', 'texture', 'voice', 'barks', 'sfx', 'summary'], placeholder: 'A cheerful dwarf blacksmith with a braided red beard and a big hammer…', chips: ['3d_ready', 'turnaround'] },
};
const API = '/api/x/pipelines/flows';
const DEFAULT_BARKS = 'Ha! Take that!\nOver here, follow me!\nI could use a break…\nWatch out, behind you!';
const DEFAULT_SFX = 'heavy boots footsteps on stone floor, close, single steps\nwhoosh of a large hammer swing, fast\nhammer striking a metal anvil, bright metallic clang with ring\nleather armour rustle as the character moves';

export default S.defineStudio({
  id: 'pipelines', title: 'Pipelines', icon: 'workflow', color: 'var(--accent)', order: 40, short: 'Flows',
  mount(el) {
    css();
    el.style.height = '100%';
    const P = new Page(el);
    return { route: r => P.route(r), show: () => P.loadList() };
  },
});

class Page {
  constructor(el) {
    this.side = h('aside.pl-side');
    this.main = h('div.pl-main');
    el.append(h('div.pl', this.side, this.main));
    this.flow = null;
    this.disposers = [];
    this.open = null;             // step key expanded by the user (null = the active step)
    this.autopilot = false;
    this.running = new Set();     // step keys with a job in flight (this page)
    this._save = S.debounce(() => this.saveNow(), 500);
    this.renderSide();
    this.renderEmpty();
    S.bus.on('project', () => { this.flow = null; this.renderEmpty(); this.loadList(); });
    S.bus.on('models', () => { if (this.flow) this.render(); });
  }

  // ---------------------------------------------------------------- side: start + saved flows
  renderSide() {
    this.listEl = h('div.pl-flows', h('div.dim', { style: { fontSize: '12.5px' } }, 'Loading…'));
    this.side.replaceChildren(
      h('div.section-head', { style: { margin: 0 } }, h('h3.section-title', ic('rocket', 16), 'Start a flow')),
      h('div.pl-start', ...Object.entries(KINDS).map(([k, d]) => h('button.pl-kind', { type: 'button', on: { click: () => this.start(k) } },
        h('div.mp-icon', ic(d.icon, 18)), h('div.grow', h('div.pl-kind-t', d.title), h('div.pl-kind-s', d.blurb))))),
      h('div.section-head', { style: { margin: '6px 0 0' } }, h('h3.section-title', ic('history', 16), 'Your flows'), h('span.spacer'),
        btn('', { size: 'sm', kind: 'ghost', icon: 'refresh-cw', title: 'Reload', onClick: () => this.loadList() })),
      this.listEl);
    this.loadList();
  }
  async loadList() {
    try {
      const r = await S.api.get(API, { project: S.store.project });
      this.listEl.replaceChildren(...(r.flows.length ? r.flows.map(f => h(`div.pl-flow${this.flow?.id === f.id ? '.on' : ''}`, { on: { click: () => this.openFlow(f.id) }, tabindex: 0 },
        f.thumb ? h('img', { src: `/files/${S.store.project}/${f.thumb}/thumb.webp`, alt: '', loading: 'lazy', on: { error: e => e.target.replaceWith(h('div.pl-ph', ic(KINDS[f.kind]?.icon || 'workflow', 16))) } }) : h('div.pl-ph', ic(KINDS[f.kind]?.icon || 'workflow', 16)),
        h('div.grow', { style: { minWidth: 0 } }, h('div.ellipsis', { style: { fontWeight: 600, fontSize: '13px' } }, f.title || KINDS[f.kind]?.title),
          h('div.dim.ellipsis', { style: { fontSize: '11.5px' } }, `${KINDS[f.kind]?.title || f.kind} · ${f.status === 'done' ? 'done' : `step ${(f.step || 0) + 1}`} · ${S.fmtAgo(f.updated)}`)))) :
        [h('div.dim', { style: { fontSize: '12.5px' } }, 'No flows yet in this project.')]));
    } catch (e) { this.listEl.replaceChildren(h('div.dim', e.message)); }
  }
  route(r) {
    if (r.sub && KINDS[r.sub] && (!this.flow || this.flow.kind !== r.sub || r.query?.new)) {
      if (!this.flow || this.flow.kind !== r.sub) this.start(r.sub);
    }
  }
  renderEmpty() {
    this.main.replaceChildren(h('div.page-pad', { style: { padding: 0 } },
      h('h1.page-title', 'Pipelines'), h('p.page-sub', 'Guided multi-step flows. Each step is a normal job: pick, approve and continue. Every result lands in the Library with its full recipe and lineage.'),
      h('div.grid', { style: { marginTop: '18px', '--card-min': '280px' } }, ...Object.entries(KINDS).map(([k, d]) => h('div.card.card-pad', h('h3', ic(d.icon, 17), d.title),
        h('p.dim', { style: { fontSize: '13px' } }, d.blurb), h('div.pl-mini', ...d.steps.map((s, i) => h('span.badge.outline', `${i + 1}. ${STEPS[s].short || STEPS[s].title}`))),
        h('div', { style: { marginTop: '12px' } }, btn('Start', { kind: 'primary', icon: 'arrow-right', onClick: () => this.start(k) })))))));
  }

  // ---------------------------------------------------------------- flow lifecycle
  start(kind) {
    this.flow = { id: flowId(), kind, title: KINDS[kind].title, project: S.store.project, step: 0, status: 'active', state: {} };
    this.open = null;   // saved on the first change (typing the idea), so clicking Start leaves no empty record
    this.render();
  }
  async openFlow(id) {
    try {
      this.flow = await S.api.get(`${API}/${id}`, { project: S.store.project });
      this.flow.state ||= {};
      this.open = null;
      this.render();
      this.loadList();
      // resume jobs that were running when the page was left
      for (const [k, st] of Object.entries(this.flow.state)) if (st?.running && st.job) this.resume(k, st.job);
    } catch (e) { S.toastError(e); }
  }
  async saveNow() {
    if (!this.flow) return;
    try { await S.api.put(`${API}/${this.flow.id}`, { ...this.flow, project: this.flow.project || S.store.project }); } catch (e) { console.warn('flow save failed', e); }
  }
  save() { this._save(); }
  st(key) { return (this.flow.state[key] ||= {}); }
  steps() { return KINDS[this.flow.kind].steps; }
  idx(key) { return this.steps().indexOf(key); }
  approve(key, extra = {}) {
    Object.assign(this.st(key), { status: 'done', ...extra });
    const i = this.idx(key);
    if (this.flow.step <= i) this.flow.step = Math.min(i + 1, this.steps().length - 1);
    if (i === this.steps().length - 1) this.flow.status = 'done';
    this.open = null;
    this.saveNow().then(() => this.loadList()); this.render();
    if (this.steps()[this.flow.step] && !this.autopilot) setTimeout(() => this.main.querySelector('.pl-step.active')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  }
  skip(key) { this.approve(key, { status: 'skipped' }); }
  goto(key) { this.open = this.open === key ? null : key; this.render(); }
  meta(studio, key, extra = {}) {
    return { studio, tags: ['pipeline', this.flow.kind], asset_extra: { x_flow: this.flow.id, x_flow_step: key }, ...extra };
  }
  /** Run one job for a step: state.job / running are persisted so a reload can resume; resolves with the assets. */
  async job(key, spec) {
    const st = this.st(key);
    this.running.add(key);
    try {
      const res = await runJob({ project: S.store.project, ...spec }, { onJob: j => { st.job = j.id; st.running = true; this.save(); this.render(); } });
      st.running = false;
      return res.assets;
    } catch (e) {
      st.running = false;
      st.error = e.cancelled ? 'Cancelled' : e.message;
      throw e;
    } finally { this.running.delete(key); this.save(); }
  }
  async resume(key, jobId) {
    const st = this.st(key);
    try {
      const { assets } = await finishJob(jobId);
      st.running = false;
      STEPS[key].onResumed?.(this, st, assets);
    } catch (e) { st.running = false; st.error = e.message; }
    this.save(); this.render();
  }
  async assets(ids) {
    ids = [...new Set((ids || []).filter(Boolean))];
    if (!ids.length) return [];
    const r = await S.api.assets({ ids: ids.join(','), limit: 200 });
    return ids.map(i => r.items.find(a => a.id === i)).filter(Boolean);
  }
  allAssetIds() {
    const out = [];
    for (const st of Object.values(this.flow.state || {})) for (const k of ['picked', 'output', 'final', 'voice', 'cut']) if (typeof st?.[k] === 'string') out.push(st[k]);
    for (const st of Object.values(this.flow.state || {})) for (const id of st?.kept || []) out.push(id);
    return [...new Set(out)];
  }
  track(c) { if (c?.destroy) this.disposers.push(c); return c; }

  // ---------------------------------------------------------------- rendering
  render() {
    for (const d of this.disposers.splice(0)) { try { d.destroy(); } catch { /* */ } }
    const f = this.flow; if (!f) return this.renderEmpty();
    const K = KINDS[f.kind];
    const title = h('h2.grow', f.title || K.title);
    const head = h('div.pl-head', h('div.mp-icon', ic(K.icon, 18)), title,
      h('span.badge.outline', K.title), f.status === 'done' ? h('span.badge.ok', ic('check', 12), 'done') : null,
      btn('Rename', { size: 'sm', kind: 'ghost', icon: 'pencil', onClick: async () => { const t = await S.promptDialog('Flow name', { value: f.title }); if (t) { f.title = t; this.save(); this.render(); this.loadList(); } } }),
      btn('Export kit', { size: 'sm', icon: 'package', title: 'Zip every picked / final asset of this flow', onClick: () => { const ids = this.allAssetIds(); if (!ids.length) return S.toast('Nothing to export yet.', { kind: 'warn' }); S.exportAssets(ids); } }),
      btn('', { size: 'sm', kind: 'ghost', icon: 'trash-2', title: 'Delete this flow (its assets stay in the Library)', onClick: async () => {
        if (!(await S.confirmDialog('Delete this flow record? Its assets stay in the Library.', { ok: 'Delete', danger: true }))) return;
        await S.api.del(`${API}/${f.id}?project=${encodeURIComponent(S.store.project)}`).catch(S.toastError); this.flow = null; this.renderEmpty(); this.loadList();
      } }));
    const list = h('div.pl-steps');
    this.steps().forEach((key, i) => list.append(this.stepCard(key, i)));
    this.main.replaceChildren(head, list);
  }
  stepCard(key, i) {
    const D = STEPS[key]; const st = this.st(key);
    const active = i === this.flow.step && this.flow.status !== 'done';
    const expanded = this.open ? this.open === key : active;
    const cls = `.pl-step${active ? '.active' : ''}${st.status === 'done' ? '.done' : ''}${st.status === 'skipped' ? '.skipped' : ''}`;
    const badge = st.running ? h('span.badge.info', ic('loader', 12, 'spin'), 'running') : st.status === 'done' ? h('span.badge.ok', ic('check', 12), 'done')
      : st.status === 'skipped' ? h('span.badge.outline', 'skipped') : active ? h('span.badge.accent', 'current') : i > this.flow.step ? h('span.badge.outline', 'later') : null;
    const needs = D.sound && !readyModels(D.task(this, st)).length ? h('span.badge.warn', 'needs Sound studio') : null;
    const card = h(`section${cls}`, { dataset: { step: key } },
      h('div.pl-step-h', { on: { click: () => i <= this.flow.step && this.goto(key) }, title: i <= this.flow.step ? 'Show / hide this step' : '' },
        h('div.pl-num', st.status === 'done' ? ic('check', 14) : String(i + 1)),
        h('div.grow', { style: { minWidth: 0 } }, h('div.pl-step-t', D.title, D.optional ? h('span.dim', { style: { fontWeight: 400, fontSize: '12px' } }, '  optional') : null),
          h('div.pl-step-s', D.summary?.(this, st) || D.sub || '')),
        needs, badge));
    if (expanded && i <= this.flow.step) {
      const body = h('div.pl-step-b');
      try { body.append(D.body(this, st)); } catch (e) { console.error(e); body.append(h('div.err', String(e))); }
      if (st.job && (st.running || st.error)) body.prepend(this.jobBox(key, st));
      card.append(body);
    }
    return card;
  }
  jobBox(key, st) {
    const box = h('div');
    S.api.job(st.job).then(j => { S.store.jobs.set(j.id, j); box.replaceChildren(S.jobCard(j)); }).catch(() => box.replaceChildren(st.error ? h('div.err', st.error) : null));
    return box;
  }
  runBtn(label, fn, { icon = 'sparkles', kind = 'primary', disabled = false } = {}) {
    const b = btn(label, { kind, icon, disabled, onClick: async () => {
      b.disabled = true;
      try { await fn(); } catch (e) { if (!e.cancelled) S.toast(e.fix ? `${e.message} ${e.fix}` : e.message, { kind: 'err', title: 'Step failed' }); this.render(); }
      finally { b.disabled = false; }
    } });
    return b;
  }

  // ---------------------------------------------------------------- shared building blocks
  modelBlock(st, task, { prefer = [], memoryKey, exclude = [], defaults = {}, label = 'Model' } = {}) {
    const form = this.track(new S.ParamForm({ memoryKey, exclude }));
    const picker = this.track(new S.ModelPicker({ task, label, value: (st.model && S.store.modelsById.get(st.model)?.tasks.includes(task) && S.store.modelsById.get(st.model)?.status?.status === 'ready') ? st.model : bestModel(task, prefer),
      onChange: m => { st.model = m?.id; form.setModel(m, task); form.setValues(defaults, { silent: true }); if (st.params) form.setValues(st.params, { silent: true }); this.save(); } }));
    form.onChange = (v, user) => { if (user) { st.params = form.values(); this.save(); } };
    const m = picker.model;
    if (m) { form.setModel(m, task); form.setValues(defaults, { silent: true }); if (st.params) form.setValues(st.params, { silent: true }); }
    st.model = picker.value;
    return { picker, form, el: h('div.col', { style: { gap: '14px' } }, picker.el, h('details.pf-adv', { open: !!st.showSettings, on: { toggle: e => { st.showSettings = e.currentTarget.open; } } },
      h('summary', ic('chevron-right', 14), 'Settings', h('span.dim', '(presets + every knob)')), h('div', { style: { marginTop: '10px' } }, form.el))) };
  }
  pickGrid(st, ids, { field = 'picked', onPick, multiple = false } = {}) {
    const grid = h('div.grid.pl-pick', { style: { '--card-min': '170px' } });
    this.assets(ids).then(list => {
      autoThumb(list);
      grid.replaceChildren(...list.map(a => {
        const c = S.assetCard(a, { onOpen: x => {
          if (multiple) { st[field] = (st[field] || []).includes(x.id) ? st[field].filter(i => i !== x.id) : [...(st[field] || []), x.id]; }
          else st[field] = x.id;
          this.save(); onPick ? onPick(x) : this.render();
        } });
        const on = multiple ? (st[field] || []).includes(a.id) : st[field] === a.id;
        c.classList.toggle('picked', on);
        c.title = 'Click to pick · double-click for details';
        c.addEventListener('dblclick', e => { e.stopPropagation(); S.openAsset(a, { list: list.map(x => x.id) }); });
        return c;
      }));
      if (!list.length) grid.replaceChildren(h('div.dim', { style: { fontSize: '12.5px' } }, 'No results yet.'));
    }).catch(e => grid.replaceChildren(h('div.dim', e.message)));
    return grid;
  }
  inputOf(key) {   // the image a step works on = the output of the previous image step
    const s = this.flow.state;
    if (key === 'refine') return s.drafts?.picked;
    if (key === 'cutout') return (s.refine?.status === 'done' && s.refine?.picked) || s.drafts?.picked;
    if (key === 'shape') return s.cutout?.final || s.cutout?.picked;
    if (key === 'texture') return s.shape?.output;
    if (key === 'export') return s.texture?.output || s.shape?.output;
    return null;
  }
  preview(id, opts = {}) {
    const box = h('div.pl-prev');
    if (!id) return box;
    this.assets([id]).then(([a]) => {
      if (!a) return box.replaceChildren(h('div.dim', 'The asset was deleted.'));
      if (a.type === 'image') box.replaceChildren(h('div.i3d-img' + (a.info?.has_alpha ? '.checker' : ''), { on: { click: () => S.openAsset(a) }, title: 'Open with zoom' }, h('img', { src: S.fileUrl(a), alt: a.name || '' })),
        h('div.dim.ellipsis', { style: { fontSize: '12px', marginTop: '4px' } }, `${a.name || a.id} · ${a.info?.width}×${a.info?.height}${a.info?.has_alpha ? ` · RGBA ${Math.round((a.info.transparent_fraction || 0) * 100)}% transparent` : ''}`));
      else if (a.type === 'mesh' || (a.files || []).some(f => f.name.endsWith('_preview.glb'))) { const v = meshViewer({ height: opts.height || 380 }); v.set(a); box.replaceChildren(v); box.style.maxWidth = 'none'; }
      else box.replaceChildren(S.assetCard(a, { onOpen: x => S.openAsset(x) }));
    }).catch(() => {});
    return box;
  }
  async autoFinish(fromKey) {
    this.autopilot = true;
    try {
      for (const key of this.steps().slice(this.idx(fromKey))) {
        const D = STEPS[key];
        if (!D.auto) break;
        const st = this.st(key);
        if (st.status === 'done' || st.status === 'skipped') continue;
        await D.auto(this, st);
      }
      S.toast('Auto-finish done: check the results, redo any step if needed.', { kind: 'ok' });
    } catch (e) { if (!e.cancelled) S.toast(e.message, { kind: 'err', title: 'Auto-finish stopped' }); }
    finally { this.autopilot = false; this.render(); }
  }
}

// =================================================================================== steps
const STEPS = {
  // ------------------------------------------------------------------ 1. idea
  idea: {
    title: 'Idea', short: 'Idea', sub: 'Describe it (Thai or English), ✨ improve it into a 3D-ready prompt.',
    summary: (P, st) => st.prompt ? `“${st.prompt.slice(0, 90)}${st.prompt.length > 90 ? '…' : ''}”` : null,
    body(P, st) {
      const K = KINDS[P.flow.kind];
      if (S.prefs.get(`chips.pl-idea-${P.flow.kind}`) == null) S.prefs.set(`chips.pl-idea-${P.flow.kind}`, K.chips);
      const draftModel = S.store.modelsById.get(P.flow.state.drafts?.model || bestModel('t2i', ['z-image-turbo']));
      const box = new S.PromptBox({ label: P.flow.kind === 'character' ? 'Character' : 'Prop', placeholder: K.placeholder, memoryKey: `pl-idea-${P.flow.kind}`,
        model: draftModel, task: 't2i', style: 'image3d', context: () => st.context || '', onSubmit: () => go(),
        onChange: v => { if (v === (st.prompt || '')) return; st.prompt = v; P.save(); } });   // set() also fires onChange
      box.set(st.prompt || '');
      if (st.improved_prompt && st.prompt === st.improved_prompt && st.original) box.improved = { original: st.original, applied: st.prompt, style: 'image3d' };
      const ctxIn = h('input.input.input-sm', { placeholder: 'Art direction, e.g. "stylized hand-painted RPG"', value: st.context || '', on: { change: e => { st.context = e.target.value; P.save(); } } });
      const go = () => {
        if (!box.value) { box.focus(); return S.toast('Write the idea first.', { kind: 'warn' }); }
        const pm = S.promptMeta(box);
        Object.assign(st, { prompt: box.value, original: pm.prompt, improved_prompt: pm.improved_prompt || null });
        if (!P.flow.title || P.flow.title === K.title) P.flow.title = (pm.prompt || box.value).slice(0, 60);
        P.approve('idea');
        P.loadList();
      };
      return h('div.col', { style: { gap: '12px', maxWidth: '760px' } }, box.el,
        h('label.field', h('span.field-label', ic('palette', 14), 'Art direction (optional, used by ✨ Improve)'), ctxIn),
        h('div.pl-actions', btn('Continue', { kind: 'primary', icon: 'arrow-right', onClick: go }), h('span.dim', { style: { fontSize: '12px' } }, 'Tip: ✨ Improve with the 3D-ready chip writes a single-object, fully visible, plain-background prompt.')));
    },
  },

  // ------------------------------------------------------------------ 2. fast variations
  drafts: {
    title: 'Fast variations', short: 'Variations', sub: 'Four quick concepts with a fast model; pick the best one.',
    summary: (P, st) => st.picked ? 'picked 1 of ' + (st.kept || []).length : (st.kept?.length ? `${st.kept.length} results` : null),
    onResumed: (P, st, assets) => { st.kept = [...(st.kept || []), ...assets.map(a => a.id)]; },
    async run(P, st, mb) {
      const m = mb?.picker.model || S.store.modelsById.get(st.model);
      if (!m) throw new Error('No text-to-image model is ready.');
      const idea = P.flow.state.idea || {};
      const params = { ...(mb ? mb.form.values() : st.params || {}), prompt: idea.prompt, seed: -1 };
      if (m.params.some(p => p.name === 'threed_ready') && params.threed_ready === undefined) params.threed_ready = true;
      st.params = { ...params }; delete st.params.prompt;
      const assets = await P.job('drafts', { task: 't2i', model: m.id, params, title: `Concepts: ${(idea.original || idea.prompt || '').slice(0, 40)}`,
        meta: P.meta('image', 'drafts', { prompt: idea.original || idea.prompt, improved_prompt: idea.improved_prompt }) });
      st.kept = [...(st.kept || []), ...assets.map(a => a.id)];
      if (!st.picked && assets[0] && P.autopilot) st.picked = assets[0].id;
      if (assets[0] && !P.flow.state.thumb) P.flow.state.thumb = assets[0].id;
      P.save(); P.render();
      return assets;
    },
    body(P, st) {
      const mb = P.modelBlock(st, 't2i', { prefer: ['z-image-turbo'], memoryKey: 'pl-drafts', defaults: st.params ? {} : { batch: 4, threed_ready: true } });
      const n = mb.form.values().batch || 1;
      const add = async () => { const ids = await S.pickAssets({ types: ['image'], title: 'Use an existing image' }); if (ids?.length) { st.kept = [...(st.kept || []), ids[0]]; st.picked = ids[0]; P.save(); P.render(); } };
      return h('div.col', { style: { gap: '14px' } },
        h('div.pl-cols', mb.el, h('div.col', { style: { gap: '10px' } },
          h('div.dim', { style: { fontSize: '12.5px' } }, `Prompt: ${(P.flow.state.idea?.prompt || '').slice(0, 220)}`),
          h('div.pl-actions', P.runBtn(st.kept?.length ? `More variations (${n})` : `Generate ${n} variation${n > 1 ? 's' : ''}`, () => STEPS.drafts.run(P, st, mb), { disabled: P.running.has('drafts') || st.running }),
            btn('Use an existing image', { icon: 'library', onClick: add })))),
        st.kept?.length ? h('div.field-label', 'Click to pick · double-click for details') : null,
        st.kept?.length ? P.pickGrid(st, [...st.kept].reverse()) : null,
        h('div.pl-actions', btn('Use the picked concept', { kind: 'primary', icon: 'arrow-right', disabled: !st.picked, onClick: () => { P.flow.state.thumb = st.picked; P.approve('drafts'); P.loadList(); } }),
          st.picked ? btn('Open in Image studio', { size: 'sm', icon: 'image', onClick: () => S.sendTo('image', st.picked, { action: 'edit' }) }) : null));
    },
  },

  // ------------------------------------------------------------------ 3. refine (optional)
  refine: {
    title: 'Refine with an edit', short: 'Refine', optional: true, sub: 'Optional instruction edit (Qwen-Image keeps geometry and text).',
    summary: (P, st) => st.status === 'skipped' ? 'kept the concept as is' : st.instruction ? `“${st.instruction.slice(0, 80)}”` : null,
    onResumed: (P, st, assets) => { st.kept = [...(st.kept || []), ...assets.map(a => a.id)]; },
    body(P, st) {
      const src = P.inputOf('refine');
      const mb = P.modelBlock(st, 'edit', { prefer: ['qwen-image-2.1'], memoryKey: 'pl-refine', defaults: st.params ? {} : { batch: 1 } });
      const box = new S.PromptBox({ label: 'Instruction', placeholder: 'Make the gem glow blue, keep everything else identical…', memoryKey: 'pl-refine', model: mb.picker.model, task: 'edit',
        imageAsset: () => src, onChange: v => { st.instruction = v; P.save(); } });
      box.set(st.instruction || '');
      const run = async () => {
        if (!box.value) { box.focus(); throw new Error('Write the edit instruction first.'); }
        const m = mb.picker.model; const pm = S.promptMeta(box);
        st.instruction = box.value;
        const assets = await P.job('refine', { task: 'edit', model: m.id, params: { ...mb.form.values(), prompt: box.value }, inputs: [{ asset: src, role: 'image' }],
          title: `Refine: ${box.value.slice(0, 40)}`, meta: P.meta('image', 'refine', { prompt: pm.prompt, improved_prompt: pm.improved_prompt }) });
        st.kept = [...(st.kept || []), ...assets.map(a => a.id)];
        st.picked = assets[0]?.id || st.picked;
        P.save(); P.render();
      };
      return h('div.col', { style: { gap: '14px' } },
        h('div.pl-cols', h('div.col', { style: { gap: '12px' } }, P.preview(src), mb.el),
          h('div.col', { style: { gap: '12px' } }, box.el,
            h('div.img-presets', ...EDIT_PRESETS.slice(0, 7).map(([label, text]) => h('button.chip', { type: 'button', title: text, on: { click: () => { box.set(text); st.instruction = text; P.save(); } } }, label))),
            h('div.pl-actions', P.runBtn('Run the edit', run, { icon: 'pencil', disabled: st.running })))),
        st.kept?.length ? P.pickGrid(st, [src, ...[...st.kept].reverse()]) : null,
        h('div.pl-actions', btn('Use the picked image', { kind: 'primary', icon: 'arrow-right', disabled: !st.picked, onClick: () => P.approve('refine') }),
          btn('Skip: keep the concept', { icon: 'chevron-right', onClick: () => P.skip('refine') })));
    },
  },

  // ------------------------------------------------------------------ 4. cut-out + recenter
  cutout: {
    title: 'Transparent cut-out + recenter', short: 'Cut-out', sub: 'Isolate the subject on real transparency, centre it with a margin for 3D.',
    summary: (P, st) => st.final ? `${st.method || 'cut-out'} + recenter` : null,
    async run(P, st, opts = {}) {
      const src = P.inputOf('cutout');
      if (!src) throw new Error('Pick a concept first.');
      const method = st.method || (bestModel('extract') ? 'extract' : 'rmbg');
      let cut = src;
      if (method === 'extract') {
        const m = bestModel('extract', ['qwen-image-2.1']); if (!m) throw new Error('No extract model is ready: use “rembg (fast)”.');
        const a = await P.job('cutout', { task: 'extract', model: m, params: { prompt: st.what || '' }, inputs: [{ asset: src, role: 'image' }],
          title: `Cut-out: ${st.what || 'main subject'}`, meta: P.meta('image', 'cutout') });
        cut = a[0]?.id;
      } else if (method === 'rmbg') {
        const m = bestModel('rmbg', ['rembg-u2net']); if (!m) throw new Error('No background-removal model is ready.');
        const a = await P.job('cutout', { task: 'rmbg', model: m, params: {}, inputs: [{ asset: src, role: 'input' }], title: 'Remove background', meta: P.meta('image', 'cutout') });
        cut = a[0]?.id;
      }
      if (!cut) throw new Error('The cut-out produced no image.');
      st.cut = cut;
      const rp = { ...(opts.mb ? opts.mb.form.values() : st.params || {}) };
      const r = await P.job('cutout', { task: 'recenter', model: bestModel('recenter', ['recenter-3d']), params: rp, inputs: [{ asset: cut, role: 'input' }], title: 'Recenter for 3D', meta: P.meta('image', 'cutout') });
      st.final = r[0]?.id;
      st.kept = [...new Set([...(st.kept || []), cut, st.final].filter(Boolean))];
      P.save(); P.render();
    },
    auto: async (P, st) => { await STEPS.cutout.run(P, st); P.approve('cutout'); },
    body(P, st) {
      const src = P.inputOf('cutout');
      const hasExtract = !!bestModel('extract');
      st.method ||= hasExtract ? 'extract' : 'rmbg';
      const mb = P.modelBlock(st, 'recenter', { memoryKey: 'pl-recenter', label: 'Recenter tool' });
      const what = h('input.input', { value: st.what || '', placeholder: 'What to keep, e.g. "the chest" (empty = main subject)', on: { change: e => { st.what = e.target.value; P.save(); } } });
      const prevWrap = h('div.col', { style: { gap: '8px' } });
      if (st.final) prevWrap.append(h('div.row.wrap', h('span.field-label.grow', 'Result (alpha background)'), alphaBgToggle(prevWrap, 'pl.alphaBg')), P.preview(st.final));
      return h('div.col', { style: { gap: '14px' } },
        h('div.pl-cols', h('div.col', { style: { gap: '12px' } }, P.preview(src),
          h('div.field', h('div.field-label', 'Method'), S.seg([
            { value: 'extract', label: 'Qwen extract (best)' }, { value: 'rmbg', label: 'rembg (fast)' }, { value: 'none', label: 'Already clean' }], st.method, v => { st.method = v; P.save(); P.render(); })),
          st.method === 'extract' ? h('label.field', h('span.field-label', 'Subject'), what) : null,
          !hasExtract && st.method === 'extract' ? h('div.err', 'Qwen-Image is not ready here: use rembg.') : null,
          mb.el),
          h('div.col', { style: { gap: '12px' } }, prevWrap,
            h('div.pl-actions', P.runBtn(st.final ? 'Redo cut-out' : 'Cut out + recenter', () => STEPS.cutout.run(P, st, { mb }), { icon: 'scissors', disabled: st.running })))),
        h('div.pl-actions', btn('Use this for 3D', { kind: 'primary', icon: 'arrow-right', disabled: !st.final, onClick: () => P.approve('cutout') }),
          st.final && STEPS.shape.auto ? P.runBtn('Auto-finish to GLB', async () => { P.approve('cutout'); await P.autoFinish('shape'); }, { kind: '', icon: 'zap' }) : null));
    },
  },

  // ------------------------------------------------------------------ 5. shape
  shape: {
    title: '3D shape', short: 'Shape', sub: 'Hunyuan3D 2.1 image → mesh (about a minute).',
    summary: (P, st) => st.output ? 'mesh ready' : null,
    async run(P, st, mb) {
      const src = P.inputOf('shape'); if (!src) throw new Error('Finish the cut-out first.');
      const m = bestModel('shape', ['hunyuan3d-2.1']); if (!m) throw new Error('Hunyuan3D is not ready.');
      const params = { ...(mb ? mb.form.values() : st.params || {}), textured: false };
      const a = await P.job('shape', { task: 'shape', model: m, params, inputs: [{ asset: src, role: 'image' }], title: '3D shape', meta: P.meta('3d', 'shape') });
      st.output = a[0]?.id; st.kept = [...(st.kept || []), st.output].filter(Boolean);
      P.save(); P.render();
    },
    auto: async (P, st) => { await STEPS.shape.run(P, st); P.approve('shape'); },
    body(P, st) {
      const mb = P.modelBlock(st, 'shape', { prefer: ['hunyuan3d-2.1'], memoryKey: 'pl-shape', exclude: ['textured'], label: '3D model' });
      return h('div.col', { style: { gap: '14px' } },
        h('div.pl-cols', h('div.col', { style: { gap: '12px' } }, P.preview(P.inputOf('shape')), mb.el),
          h('div.col', { style: { gap: '12px' } }, st.output ? P.preview(st.output) : h('div.i3d-viewer', h('div.empty', ic('cuboid', 30), h('p', 'The mesh appears here.'))),
            h('div.pl-actions', P.runBtn(st.output ? 'Redo shape (new seed)' : 'Make the shape', () => STEPS.shape.run(P, st, mb), { icon: 'cuboid', disabled: st.running })))),
        h('div.pl-actions', btn('Approve the shape', { kind: 'primary', icon: 'arrow-right', disabled: !st.output, onClick: () => P.approve('shape') }),
          st.output ? P.runBtn('Auto-finish: texture + GLB', async () => { P.approve('shape'); await P.autoFinish('texture'); }, { kind: '', icon: 'zap' }) : null));
    },
  },

  // ------------------------------------------------------------------ 6. texture
  texture: {
    title: 'PBR texture', short: 'Texture', sub: 'Paint albedo + metallic / roughness from the concept (3-5 min).',
    summary: (P, st) => st.output ? 'textured mesh ready' : null,
    async run(P, st, mb) {
      const mesh = P.inputOf('texture'); if (!mesh) throw new Error('Make the shape first.');
      const m = bestModel('texture', ['hunyuan3d-2.1']);
      const a = await P.job('texture', { task: 'texture', model: m, params: { ...(mb ? mb.form.values() : st.params || {}) }, inputs: [{ asset: mesh, role: 'mesh' }], title: 'PBR texture', meta: P.meta('3d', 'texture') });
      st.output = a[0]?.id; st.kept = [...(st.kept || []), st.output].filter(Boolean);
      P.save(); P.render();
    },
    auto: async (P, st) => { await STEPS.texture.run(P, st); P.approve('texture'); },
    body(P, st) {
      const mb = P.modelBlock(st, 'texture', { prefer: ['hunyuan3d-2.1'], memoryKey: 'pl-texture', label: '3D model' });
      return h('div.col', { style: { gap: '14px' } },
        h('div.pl-cols', h('div.col', { style: { gap: '12px' } }, mb.el),
          h('div.col', { style: { gap: '12px' } }, P.preview(st.output || P.inputOf('texture')),
            h('div.pl-actions', P.runBtn(st.output ? 'Repaint (new settings)' : 'Paint the texture', () => STEPS.texture.run(P, st, mb), { icon: 'palette', disabled: st.running })))),
        h('div.pl-actions', btn('Approve the texture', { kind: 'primary', icon: 'arrow-right', disabled: !st.output, onClick: () => P.approve('texture') }),
          btn('Skip: keep it untextured', { icon: 'chevron-right', onClick: () => P.skip('texture') })));
    },
  },

  // ------------------------------------------------------------------ 7. export
  export: {
    title: 'Export GLB', short: 'GLB', sub: 'Engine-ready file (GLB with PBR, or OBJ / PLY / STL; optional decimation).',
    summary: (P, st) => st.output ? `${(st.format || 'glb').toUpperCase()} ready` : null,
    async run(P, st, mb) {
      const mesh = P.inputOf('export'); if (!mesh) throw new Error('Make the mesh first.');
      const params = { ...(mb ? mb.form.values() : st.params || { export_type: 'glb', include_texture: true }) };
      const a = await P.job('export', { task: 'mesh_export', model: bestModel('mesh_export', ['mesh-export']), params, inputs: [{ asset: mesh, role: 'mesh' }], title: `Export ${params.export_type || 'glb'}`, meta: P.meta('3d', 'export') });
      st.output = a[0]?.id; st.format = params.export_type; st.kept = [...(st.kept || []), st.output].filter(Boolean);
      P.save(); P.render();
    },
    auto: async (P, st) => { await STEPS.export.run(P, st); P.approve('export'); },
    body(P, st) {
      const mb = P.modelBlock(st, 'mesh_export', { memoryKey: 'pl-export', label: 'Export tool', defaults: st.params ? {} : { export_type: 'glb', include_texture: true } });
      const dl = st.output ? btn('Download', { kind: 'primary', icon: 'download', onClick: async () => { const [a] = await P.assets([st.output]); if (a) location.href = S.fileUrl(a, a.main, true); } }) : null;
      return h('div.col', { style: { gap: '14px' } },
        h('div.pl-cols', mb.el, h('div.col', { style: { gap: '12px' } }, st.output ? P.preview(st.output) : P.preview(P.inputOf('export')),
          h('div.pl-actions', P.runBtn(st.output ? 'Export again' : 'Export', () => STEPS.export.run(P, st, mb), { icon: 'package', disabled: st.running }), dl,
            st.output ? btn('Open in 3D studio', { icon: 'cuboid', onClick: () => S.sendTo('3d', st.output, { action: 'export' }) }) : null))),
        h('div.pl-actions', btn(P.flow.kind === 'concept3d' ? 'Finish' : 'Continue', { kind: 'primary', icon: 'check', disabled: !st.output, onClick: () => P.approve('export') })));
    },
  },

  // ------------------------------------------------------------------ character: voice
  voice: {
    title: 'Designed voice', short: 'Voice', sound: true, task: () => 'voice_design', sub: 'A voice made from a text description of the character (✨ AI suggests it from step 1).',
    summary: (P, st) => st.voice ? 'voice picked' : null,
    onResumed: (P, st, assets) => { st.kept = [...(st.kept || []), ...assets.map(a => a.id)]; },
    body(P, st) {
      if (!readyModels('voice_design').length) return needsSound(P, 'voice', 'voice design');
      const mb = P.modelBlock(st, 'voice_design', { memoryKey: 'pl-voice', exclude: ['text', 'instruct', 'instruction', 'ref_text'], label: 'Voice model' });
      const idea = P.flow.state.idea || {};
      const line = h('input.input', { value: st.line ?? DEFAULT_LINE, on: { change: e => { st.line = e.target.value; P.save(); } } });
      const box = new S.PromptBox({ label: 'Voice description', memoryKey: 'pl-voice', model: mb.picker.model, task: 'voice_design', context: () => `Game character: ${idea.original || idea.prompt || ''}`,
        placeholder: 'A gruff, warm male voice in his fifties, low pitch, slow and confident…', onChange: v => { st.desc = v; P.save(); },
        onApply: extras => { if (extras.text) { line.value = extras.text; st.line = extras.text; } const d = extras.instruct || extras.instruction; if (d) { box.set(d); st.desc = d; } P.save(); } });
      box.set(st.desc || '');
      const run = async () => {
        const desc = box.value; if (!desc) { box.focus(); throw new Error('Describe the voice first (✨ Improve can write it from the character).'); }
        st.desc = desc; st.line = line.value;
        const m = mb.picker.model;
        const a = await P.job('voice', { task: 'voice_design', model: m.id, params: { ...mb.form.values(), prompt: desc, instruct: desc, instruction: desc, text: line.value }, title: 'Character voice',
          meta: P.meta('sound', 'voice', { prompt: desc }) });
        st.kept = [...(st.kept || []), ...a.map(x => x.id)];
        st.voice = a[0]?.id || st.voice;
        P.save(); P.render();
      };
      return h('div.col', { style: { gap: '14px' } }, profileBar(P),
        h('div.pl-cols', mb.el, h('div.col', { style: { gap: '12px' } }, box.el, h('label.field', h('span.field-label', 'Sample line'), line),
          h('div.pl-actions', P.runBtn(st.kept?.length ? 'Another take' : 'Design the voice', run, { icon: 'mic', disabled: st.running })))),
        st.kept?.length ? P.pickGrid(st, [...st.kept].reverse(), { field: 'voice' }) : null,
        h('div.pl-actions', btn('Use the picked voice', { kind: 'primary', icon: 'arrow-right', disabled: !st.voice, onClick: () => P.approve('voice') }),
          btn('Skip', { icon: 'chevron-right', onClick: () => P.skip('voice') })));
    },
  },

  // ------------------------------------------------------------------ character: barks
  barks: {
    title: 'Barks in that voice', short: 'Barks', sound: true, task: (P, st) => barkTask(P, st), sub: 'Short in-character lines, one take set per line.',
    summary: (P, st) => st.kept?.length ? `${st.kept.length} take(s)` : null,
    body(P, st) {
      const task = barkTask(P, st);
      if (!readyModels(task).length) return needsSound(P, 'barks', task === 'voice_clone' ? 'voice cloning' : 'text to speech');
      const voice = P.flow.state.voice?.voice;
      const mb = P.modelBlock(st, task, { memoryKey: `pl-barks-${task}`, exclude: ['text', 'instruct', 'instruction', 'ref_text'], label: task === 'voice_clone' ? 'Clone model' : 'TTS model' });
      const lines = h('textarea.textarea.pl-lines', { rows: 5, value: st.lines ?? DEFAULT_BARKS, on: { change: e => { st.lines = e.target.value; P.save(); } } });
      const run = async () => {
        const list = lines.value.split('\n').map(s => s.trim()).filter(Boolean).slice(0, 30);
        if (!list.length) throw new Error('Write at least one line.');
        st.lines = lines.value;
        const m = mb.picker.model;
        const jobs = list.map(text => P.job('barks', { task, model: m.id, params: { ...mb.form.values(), text, prompt: text, ...(task === 'voice_clone' ? { ref_text: P.flow.state.voice?.line || '' } : {}) },
          inputs: task === 'voice_clone' && voice ? [{ asset: voice, role: 'ref' }] : [], title: `Bark: ${text.slice(0, 40)}`, meta: P.meta('sound', 'barks', { prompt: text }) })
          .then(a => { st.kept = [...(st.kept || []), ...a.map(x => x.id)]; P.save(); P.render(); }));
        await Promise.allSettled(jobs);
      };
      const modeSeg = readyModels('voice_clone').length && voice ? S.seg([{ value: 'voice_clone', label: 'Clone the designed voice' }, { value: 'tts', label: 'Preset TTS voice' }], task, v => { st.mode = v; P.save(); P.render(); }) : null;
      return h('div.col', { style: { gap: '14px' } }, profileBar(P),
        h('div.pl-cols', h('div.col', { style: { gap: '12px' } }, modeSeg, mb.el), h('div.col', { style: { gap: '12px' } },
          h('label.field', h('span.field-label', 'Lines (one per line)'), lines),
          voice ? h('div.dim', { style: { fontSize: '12px' } }, 'Reference voice: the voice picked in the previous step.') : h('div.dim', { style: { fontSize: '12px' } }, 'No designed voice: a preset TTS voice is used.'),
          h('div.pl-actions', P.runBtn('Record the barks', run, { icon: 'mic' })))),
        st.kept?.length ? P.pickGrid(st, [...st.kept].reverse(), { field: 'keep', multiple: true }) : null,
        h('div.pl-actions', btn('Continue', { kind: 'primary', icon: 'arrow-right', disabled: !st.kept?.length, onClick: () => P.approve('barks') }), btn('Skip', { icon: 'chevron-right', onClick: () => P.skip('barks') })));
    },
  },

  // ------------------------------------------------------------------ character: sfx
  sfx: {
    title: 'Footsteps / attack SFX', short: 'SFX', sound: true, task: () => 'sfx', sub: 'One sound-effect job per event line.',
    summary: (P, st) => st.kept?.length ? `${st.kept.length} take(s)` : null,
    body(P, st) {
      if (!readyModels('sfx').length) return needsSound(P, 'sfx', 'sound effects');
      const mb = P.modelBlock(st, 'sfx', { memoryKey: 'pl-sfx', label: 'SFX model', defaults: st.params ? {} : { batch: 2 } });
      const lines = h('textarea.textarea.pl-lines', { rows: 5, value: st.lines ?? DEFAULT_SFX, on: { change: e => { st.lines = e.target.value; P.save(); } } });
      const run = async () => {
        const list = lines.value.split('\n').map(s => s.trim()).filter(Boolean).slice(0, 30);
        if (!list.length) throw new Error('Write at least one event.');
        st.lines = lines.value;
        const m = mb.picker.model;
        await Promise.allSettled(list.map(prompt => P.job('sfx', { task: 'sfx', model: m.id, params: { ...mb.form.values(), prompt }, title: `SFX: ${prompt.slice(0, 40)}`, meta: P.meta('sound', 'sfx', { prompt }) })
          .then(a => { st.kept = [...(st.kept || []), ...a.map(x => x.id)]; P.save(); P.render(); })));
      };
      return h('div.col', { style: { gap: '14px' } }, profileBar(P),
        h('div.pl-cols', mb.el, h('div.col', { style: { gap: '12px' } }, h('label.field', h('span.field-label', 'Events (one per line)'), lines),
          h('div.pl-actions', P.runBtn('Generate the SFX', run, { icon: 'audio-lines' })))),
        st.kept?.length ? P.pickGrid(st, [...st.kept].reverse(), { field: 'keep', multiple: true }) : null,
        h('div.pl-actions', btn('Continue', { kind: 'primary', icon: 'arrow-right', disabled: !st.kept?.length, onClick: () => P.approve('sfx') }), btn('Skip', { icon: 'chevron-right', onClick: () => P.skip('sfx') })));
    },
  },

  // ------------------------------------------------------------------ character: summary
  summary: {
    title: 'Character kit', short: 'Kit', sub: 'Everything made in this flow; export it as one zip.',
    body(P, st) {
      const ids = P.allAssetIds();
      const grid = h('div.pl-sum');
      P.assets(ids).then(list => { autoThumb(list); grid.replaceChildren(...list.map(a => S.assetCard(a, { onOpen: x => S.openAsset(x, { list: list.map(y => y.id) }) }))); });
      return h('div.col', { style: { gap: '14px' } }, grid,
        h('div.pl-actions', btn('Export the kit (zip)', { kind: 'primary', icon: 'package', onClick: () => ids.length ? S.exportAssets(ids) : S.toast('Nothing to export yet.', { kind: 'warn' }) }),
          btn('Mark as done', { icon: 'check', onClick: () => P.approve('summary') })));
    },
  },
};

function barkTask(P, st) {
  const voice = P.flow.state.voice?.voice;
  if (st.mode && readyModels(st.mode).length) return st.mode;
  return voice && readyModels('voice_clone').length ? 'voice_clone' : 'tts';
}

function needsSound(P, key, what) {
  return h('div.pl-needs', ic('triangle-alert', 18), h('div.grow', { style: { minWidth: '200px' } },
    h('b', 'Needs the Sound studio'), h('div.dim', { style: { fontSize: '12.5px' } }, `No ${what} model is ready on this machine yet. Once the Sound studio's models are installed this step works here directly (it calls the same jobs).`)),
  h('div.pl-actions', btn('Open Sound studio', { size: 'sm', icon: 'audio-lines', onClick: () => S.navigate('sound') }),
    btn('Check again', { size: 'sm', icon: 'refresh-cw', onClick: async () => { await S.loadModels().catch(() => {}); P.render(); } }),
    btn('Skip', { size: 'sm', icon: 'chevron-right', onClick: () => P.skip(key) })));
}

// =================================================================================== ✨ character → audio defaults
// One text-backend call (POST /api/x/assistant/write, kind character_audio) reads the character from step 1 and proposes
// the voice description + sample line, the barks and the SFX events. They become the defaults of the voice / barks / sfx
// steps. The automatic run (when the first sound step opens) only fills fields the user has not edited; "Analyze again"
// replaces them. Options (auto on/off, language of the lines, text backend) are per-browser prefs.
const DEFAULT_LINE = 'Hello there, traveller. Need something forged?';
const FACTORY = { 'voice.line': DEFAULT_LINE, 'barks.lines': DEFAULT_BARKS, 'sfx.lines': DEFAULT_SFX };
const PROFILE_LANGS = [['en', 'English'], ['th', 'Thai'], ['ja', 'Japanese'], ['zh', 'Chinese'], ['ko', 'Korean'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German']];

function profileInput(flow) {
  const i = flow.state.idea || {};
  const character = i.original || i.prompt || '';
  return { character, description: i.prompt && i.prompt !== character ? i.prompt : '', context: i.context || '' };
}
const profileKey = (flow, lang) => JSON.stringify([profileInput(flow), lang]);

function applyProfile(state, d, { force = false } = {}) {
  const targets = [
    ['voice', 'desc', d.voice?.description],
    ['voice', 'line', d.voice?.sample_line],
    ['barks', 'lines', (d.barks || []).map(b => String(b.line || '').trim()).filter(Boolean).join('\n')],
    ['sfx', 'lines', (d.sfx || []).map(s => String(s.prompt || '').trim()).filter(Boolean).join('\n')],
  ];
  for (const [key, field, value] of targets) {
    if (!value) continue;
    const st = (state[key] ||= {});
    const cur = st[field];
    const untouched = !cur || cur === st.auto?.[field] || cur === FACTORY[`${key}.${field}`];
    if (force || (untouched && st.status !== 'done')) st[field] = value;
    st.auto = { ...(st.auto || {}), [field]: value };
  }
}

async function analyzeCharacter(P, { force = false } = {}) {
  const flow = P.flow;
  if (!flow || P.profiling === flow.id) return;
  const inp = profileInput(flow);
  if (!inp.character) { if (force) S.toast('Describe the character in step 1 first.', { kind: 'warn' }); return; }
  const lang = S.prefs.get('pl-profile-lang', 'en');
  const prof = (flow.state.profile ||= {});
  prof.error = null;
  P.profiling = flow.id;
  P.render();
  try {
    const r = await S.api.write({ kind: 'character_audio', project: flow.project || S.store.project, backend: S.prefs.get('pl-profile-backend', '') || undefined,
      input: { ...inp, language: lang, count: 6, sfx_count: 4 } });
    Object.assign(prof, { data: r.data, key: profileKey(flow, lang), backend: r.backend, seconds: r.seconds, usd: r.usd, warnings: r.warnings || [] });
    applyProfile(flow.state, r.data || {}, { force });
  } catch (e) {
    prof.error = e.fix ? `${e.message} ${e.fix}` : e.message;
  } finally {
    if (P.profiling === flow.id) P.profiling = null;
    if (P.flow === flow) { P.save(); P.render(); }
    else S.api.put(`${API}/${flow.id}`, { ...flow, project: flow.project || S.store.project }).catch(() => {});   // user switched flows meanwhile
  }
}

function profileBar(P) {
  const flow = P.flow;
  const prof = flow.state.profile || {};
  const lang = S.prefs.get('pl-profile-lang', 'en');
  const auto = S.prefs.get('pl-profile-auto', true);
  const running = P.profiling === flow.id;
  const hasChar = !!profileInput(flow).character;
  if (auto && hasChar && !running && !prof.data && !prof.error) setTimeout(() => analyzeCharacter(P), 0);
  const stale = prof.data && prof.key !== profileKey(flow, lang);
  const d = prof.data || {};
  const text = running ? [ic('loader', 14, 'spin'), ' Reading the character to suggest the voice, barks and SFX… (CLI / cloud backends take 5-30 s)']
    : prof.error ? [h('b', { style: { color: 'var(--err)' } }, 'Could not analyze the character: '), prof.error]
    : prof.data ? [h('b', '✨ Defaults from the character'), d.summary ? `: ${d.summary}` : '', h('span.dim', ` · ${prof.backend?.label || 'AI'}${prof.seconds != null ? ` · ${prof.seconds}s` : ''}${prof.usd ? ` · $${Number(prof.usd).toFixed(4)}` : ''}`),
      stale ? h('div', { style: { color: 'var(--warn)' } }, 'The character (or language) changed since: analyze again to update.') : null]
    : !hasChar ? ['No character description in step 1: the defaults below are generic.']
    : ['✨ Let AI read the character and fill the voice description, sample line, barks and SFX.'];
  const backend = S.backendPicker({ purpose: 'write', value: S.prefs.get('pl-profile-backend', ''), small: true, onChange: v => S.prefs.set('pl-profile-backend', v) });
  const langSel = h('select.select.select-sm', { 'aria-label': 'Language of the lines', on: { change: e => { S.prefs.set('pl-profile-lang', e.target.value); P.render(); } } },
    ...PROFILE_LANGS.map(([v, l]) => h('option', { value: v, selected: v === lang }, l)));
  const opts = h('details', { open: !!P.profileOpts, on: { toggle: e => { P.profileOpts = e.currentTarget.open; } } },
    h('summary', { style: { cursor: 'pointer' } }, 'Options'),
    h('div.pl-actions', { style: { marginTop: '8px' } },
      h('label.row', { style: { gap: '6px' } }, S.toggle(auto, v => { S.prefs.set('pl-profile-auto', v); }, 'Analyze automatically'), 'Auto-analyze'),
      h('label.row', { style: { gap: '6px' } }, 'Lines in', langSel), backend));
  return h('div.pl-profile', ic('sparkles', 16), h('div.pl-profile-t', ...text),
    btn(prof.data ? 'Analyze again' : 'Analyze character', { size: 'sm', icon: 'sparkles', disabled: running || !hasChar,
      title: prof.data ? 'Replaces the voice description, sample line, barks and SFX lines with new suggestions' : '',
      onClick: () => analyzeCharacter(P, { force: !!prof.data }) }), opts);
}
