// Project style guide: the Style page, the "Style" chip of every prompt box and the Library's "Style guide from selection" action.
// One guide per project (data/projects/<p>/style/guide.json, versioned). It is applied on the SERVER at submit time (jobs.SUBMIT_HOOKS), so
// this file only edits and previews it; the chip only sends the per-job choice as job meta `style` ({apply:false} | {variant, strength}).
// REST: /api/x/style/*  (app/plugins/style.py). Contract: docs/CONTRACTS.md section 11.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const API = '/api/x/style';
const VARIANTS = [['image', 'Image'], ['texture', 'Texture & materials'], ['sprite', 'Sprite'], ['ui_icon', 'UI icon'], ['video', 'Video'], ['audio', 'Audio mood']];
const VLABEL = Object.fromEntries(VARIANTS);
const PALETTE_VARIANTS = ['image', 'texture', 'sprite', 'ui_icon'];
const STRENGTH_HINT = { light: 'text only', normal: 'text + up to 2 references', strong: 'text + up to 4 references + palette words' };
// Variant tab order by the active use-case profile (the first one is the one that opens).
const PROFILE_ORDER = {
  game: ['sprite', 'texture', 'image', 'ui_icon', 'video', 'audio'],
  video: ['video', 'image', 'audio', 'sprite', 'texture', 'ui_icon'], film: ['video', 'image', 'audio', 'sprite', 'texture', 'ui_icon'],
  social: ['image', 'video', 'audio', 'sprite', 'texture', 'ui_icon'], podcast: ['audio', 'image', 'video', 'sprite', 'texture', 'ui_icon'],
};
const SAMPLE = { image: 'an iron short sword', texture: 'mossy cobblestone wall', sprite: 'a goblin archer', ui_icon: 'a health potion', video: 'a knight walking through a misty forest', audio: 'a sword being drawn from its sheath' };

const CSS = `
.sty-page { display: flex; flex-direction: column; gap: 14px; }
.sty-head { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 10px 16px; }
.sty-head .grow { min-width: min(100%, 260px); }
.sty-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; min-width: 0; }
.sty-tools .lab { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-2); font-weight: 550; white-space: nowrap; }
.sty-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 400px); gap: 16px; align-items: start; }
.sty-grid > * { min-width: 0; }
.sty-col { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.sty-side { position: sticky; top: 10px; }
.sty-card { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.sty-card > header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.sty-card > header h3 { margin: 0; font-size: 14.5px; }
.sty-card > header .spacer { flex: 1; }
.sty-fld { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.sty-fld .field-label { flex-wrap: wrap; }
.sty-fld > .seg { align-self: flex-start; }
.sty-fld .cnt { margin-left: auto; font-weight: 400; color: var(--text-3); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.sty-fld .cnt.over { color: var(--err); }
.sty-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; min-width: 0; }
.sty-row .grow { min-width: 140px; }
.sty-warn { display: flex; flex-direction: column; gap: 4px; }
.sty-warn div { display: flex; gap: 6px; align-items: flex-start; font-size: 12.5px; color: var(--warn); background: var(--warn-soft); border-radius: 8px; padding: 5px 9px; overflow-wrap: anywhere; }
.sty-warn .ic { flex: none; margin-top: 2px; }
.sty-refs { display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 10px; }
.sty-ref { position: relative; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--bg-2); display: flex; flex-direction: column; min-width: 0; }
.sty-ref .im { aspect-ratio: 1; background: var(--panel-3); display: grid; place-items: center; color: var(--text-3); }
.sty-ref .im img { width: 100%; height: 100%; object-fit: cover; display: block; }
.sty-ref .nm { padding: 4px 7px; font-size: 11.5px; color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sty-ref .x { position: absolute; top: 4px; right: 4px; width: 24px; height: 24px; display: grid; place-items: center; border-radius: 50%; border: 0; cursor: pointer; background: var(--scrim); color: #fff; }
.sty-ref .x:hover { background: var(--err); }
.sty-ref.gone { border-color: var(--warn); }
.sty-ref .b { position: absolute; left: 4px; top: 4px; }
.sty-add { display: grid; place-items: center; gap: 4px; aspect-ratio: 1; border: 1px dashed var(--line); border-radius: 10px; color: var(--text-2); background: none; cursor: pointer; font: inherit; font-size: 12px; }
.sty-add:hover { border-color: var(--accent); color: var(--text); }
.sty-sw { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.sty-swatch { position: relative; width: 44px; height: 44px; border-radius: 10px; border: 1px solid var(--line); overflow: hidden; flex: none; }
.sty-swatch input[type=color] { position: absolute; inset: -6px; width: calc(100% + 12px); height: calc(100% + 12px); border: 0; padding: 0; cursor: pointer; background: none; }
.sty-swatch .x { position: absolute; top: 1px; right: 1px; width: 18px; height: 18px; border-radius: 50%; display: none; place-items: center; border: 0; cursor: pointer; background: var(--scrim); color: #fff; padding: 0; z-index: 1; }
.sty-swatch:hover .x, .sty-swatch:focus-within .x { display: grid; }
@media (hover: none) { .sty-swatch .x { display: grid; } }
.sty-hex { font-family: var(--mono); font-size: 11px; color: var(--text-3); }
.sty-range { display: flex; align-items: center; gap: 10px; min-width: 0; }
.sty-range input[type=range] { flex: 1; min-width: 80px; accent-color: var(--accent); }
.sty-var { display: flex; flex-direction: column; gap: 12px; padding-top: 4px; }
.sty-prev .pv { border: 1px solid var(--line-soft); background: var(--bg-2); border-radius: 10px; padding: 9px 11px; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; white-space: pre-wrap; }
.sty-prev .pv mark, .sty-pop .pv mark { background: var(--accent-soft); color: var(--text); border-radius: 4px; padding: 0 2px; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.sty-prev .pv.neg { color: var(--text-2); }
.sty-prev h4, .sty-pop h4 { margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--text-3); font-weight: 600; }
.sty-prev .refs { display: flex; flex-wrap: wrap; gap: 6px; }
.sty-prev .refs img { width: 44px; height: 44px; object-fit: cover; border-radius: 8px; border: 1px solid var(--line); }
.sty-prev .why { font-size: 12px; color: var(--text-3); display: flex; flex-direction: column; gap: 2px; }
.sty-start { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.sty-start button { text-align: left; border: 1px solid var(--line); background: var(--panel); border-radius: 12px; padding: 14px; cursor: pointer; color: var(--text); font: inherit; display: flex; flex-direction: column; gap: 6px; }
.sty-start button:hover { border-color: var(--accent); background: var(--panel-2); }
.sty-start button b { display: flex; align-items: center; gap: 8px; font-size: 14px; }
.sty-start button span { color: var(--text-3); font-size: 12.5px; }
.sty-hist { display: flex; flex-direction: column; gap: 8px; }
.sty-hrow { border: 1px solid var(--line-soft); border-radius: 10px; padding: 9px 12px; display: flex; flex-direction: column; gap: 6px; }
.sty-hrow.cur { border-color: var(--accent); }
.sty-hrow .top { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }
.sty-hrow .sum { color: var(--text-2); font-size: 12.5px; overflow-wrap: anywhere; }
.sty-hrow pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--mono); font-size: 11.5px; color: var(--text-2); background: var(--bg-2); padding: 8px 10px; border-radius: 8px; }
.sty-chip { display: inline-flex; align-items: center; gap: 5px; height: 26px; padding: 0 9px 0 7px; border-radius: 999px; border: 1px solid var(--accent); background: var(--accent-soft); color: var(--accent); font: inherit; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; flex: none; }
.sty-chip:hover { background: var(--accent-soft); box-shadow: 0 0 0 3px var(--accent-soft); }
.sty-chip.off { border-color: var(--warn); background: var(--warn-soft); color: var(--warn); }
.sty-chip.na { border-color: var(--line); background: none; color: var(--text-3); font-weight: 500; }
.sty-chip .v { font-weight: 500; opacity: .8; font-variant-numeric: tabular-nums; }
.sty-pop { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.sty-pop .pv { border: 1px solid var(--line-soft); background: var(--bg-2); border-radius: 9px; padding: 7px 10px; font-size: 12.5px; line-height: 1.45; max-height: 150px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; }
.sty-pop .hd { display: flex; align-items: center; gap: 8px; font-weight: 650; }
.sty-pop .rw { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: space-between; }
.sty-pop .rw > span.l { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-2); font-weight: 550; }
.sty-pop .dim { font-size: 12px; color: var(--text-3); }
.sty-pop a { cursor: pointer; }
@media (max-width: 1000px) {
  .sty-grid { grid-template-columns: minmax(0, 1fr); }
  .sty-side { position: static; }
}
@media (max-width: 520px) {
  .sty-card { padding: 12px; }
  .sty-refs { grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); }
}
`;

// The chip lives in every prompt box, so the CSS is injected when this module loads (the shell imports every studio file at start), not when the page is first opened.
if (!document.getElementById('sty-css')) document.head.append(h('style', { id: 'sty-css' }, CSS));

// ------------------------------------------------------------------ shared state (page + chip)
const clone = o => JSON.parse(JSON.stringify(o));
const SERVER = ['version', 'hash', 'saved', 'saved_by'];
const content = g => { const c = clone(g); for (const k of SERVER) delete c[k]; return c; };
const newGuide = () => ({ schema: 1, name: 'Project style', enabled: true, strength: 'normal', positive: '', background: '', avoid: [],
  palette: { colors: [], source: null, in_prompt: false, post: 'off', tint_strength: 0.35 }, refs: [], variants: {} });
const vDefaults = vid => {
  const d = { enabled: true, positive: '', background: null, avoid: [], mode: 'extend', refs: true, palette_post: null };
  if (vid === 'texture') Object.assign(d, { background: '', avoid: ['perspective', 'shadows'], refs: false });
  else if (vid === 'video') d.refs = false;
  else if (vid === 'audio') Object.assign(d, { enabled: false, mode: 'replace', refs: false });
  return d;
};
const vEff = (g, vid) => ({ ...vDefaults(vid), ...((g?.variants || {})[vid] || {}) });

const GUIDES = new Map();                   // project -> Promise<{guide, warnings, missing_refs, refs_info}>
function fetchGuide(project, force = false) {
  if (force || !GUIDES.has(project)) GUIDES.set(project, S.api.get(`${API}/guide`, { project }).catch(() => { GUIDES.delete(project); return { guide: null, warnings: [], missing_refs: [], refs_info: [] }; }));
  return GUIDES.get(project);
}
let policyP = null;
const getPolicy = () => policyP || (policyP = S.api.get(`${API}/policy`).catch(() => { policyP = null; return { tasks: {}, models: {}, labels: {}, test_subjects: '' }; }));
let infoP = null;
const getInfo = () => infoP || (infoP = S.api.get(`${API}/info`).catch(() => { infoP = null; return {}; }));

/** A little (i) button for style info key `key`; it fills in when the texts have loaded. */
function tip(key) {
  const host = h('span.sty-tip', { style: { display: 'inline-flex' } });
  getInfo().then(info => { const b = S.infoBtn(info[key]); if (b) host.append(b); });
  return host;
}

// A guide changed somewhere (this tab, another tab, MCP, the assistant): forget the cache, refetch, tell everyone.
const onSettings = S.debounce(async () => {
  GUIDES.clear();
  const r = await fetchGuide(S.store.project);
  S.bus.emit('style', r);
}, 200);
S.bus.on('settings', onSettings);
S.bus.on('project', onSettings);

const assetMetas = new Map();
function assetMeta(id) {
  if (!assetMetas.has(id)) assetMetas.set(id, S.api.asset(id).catch(() => null));
  return assetMetas.get(id);
}
function thumb(id, cls = '') {
  const box = h(`div.im${cls ? '.' + cls : ''}`, ic('image', 22));
  assetMeta(id).then(a => { const u = a && S.thumbUrl(a); if (u) box.replaceChildren(h('img', { src: u, alt: '', loading: 'lazy' })); });
  return box;
}
function tagInput(list, onChange, { max = 16, maxLen = 40, placeholder = 'Type a term, press Enter' } = {}) {
  const wrap = h('div.tag-input');
  const input = h('input', { placeholder, 'aria-label': placeholder, maxLength: maxLen });
  const paint = () => {
    wrap.replaceChildren(...list.map((t, i) => h('span.tag', t, h('button', { type: 'button', 'aria-label': `Remove ${t}`, on: { click: () => { list.splice(i, 1); onChange(); paint(); } } }, ic('x', 12)))), input);
  };
  const add = () => {
    for (const t of input.value.split(',').map(x => x.trim()).filter(Boolean)) {
      if (list.length >= max) { S.toast(`At most ${max} terms.`, { kind: 'warn', timeout: 2200 }); break; }
      if (!list.some(x => x.toLowerCase() === t.toLowerCase())) list.push(t.slice(0, maxLen));
    }
    input.value = ''; onChange(); paint();
  };
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); add(); }
    else if (e.key === 'Backspace' && !input.value && list.length) { list.pop(); onChange(); paint(); input.focus(); }
  });
  input.addEventListener('blur', () => { if (input.value.trim()) add(); });
  wrap.addEventListener('click', () => input.focus());
  paint();
  return wrap;
}
/** User text, then the added style text highlighted. */
function renderPrompt(r) {
  const p = r.prompt || '', u = r.user_prompt;
  if (u && p.startsWith(u) && p.length > u.length) {
    const rest = p.slice(u.length), sep = rest.match(/^[\s.,;]+/)?.[0] || '';
    return h('div.pv', u + sep, h('mark', rest.slice(sep.length)));        // the joining ". " belongs to the user's text, only the added style is marked
  }
  return h('div.pv', p || '(empty)');
}

// ------------------------------------------------------------------ the prompt-box chip
S.registerPromptAddon({
  id: 'style',
  mount(box) {
    let g = null, pol = null, model = box.model, task = box.task;
    const el = h('button.sty-chip.hidden', { type: 'button', on: { click: () => openPop() } });
    const key = () => `style.job.${S.store.project}.${box.o.memoryKey || 'default'}`;
    const pref = () => {
      const p = S.prefs.get(key(), {});
      if (p.apply === false && g && p.ver !== g.version) { S.prefs.set(key(), {}); return {}; }     // a new guide version switches the style back on
      return p;
    };
    const setPref = p => { S.prefs.set(key(), p); paint(); };
    const mid = () => (model && typeof model === 'object' ? model.id : model) || undefined;
    const variantFor = () => {
      const p = pref();
      const v = p.variant || pol?.tasks?.[task];
      return v && v !== 'skip' ? v : null;
    };
    function state() {
      if (!g || !g.enabled) return 'hidden';
      const v = variantFor();
      if (!v || !vEff(g, v).enabled) return 'na';
      return pref().apply === false ? 'off' : 'on';
    }
    function paint() {
      const st = state();
      el.className = `sty-chip${st === 'off' ? ' off' : st === 'na' ? ' na' : ''}${st === 'hidden' ? ' hidden' : ''}`;
      if (st === 'hidden') return;
      el.replaceChildren(ic('palette', 14), st === 'off' ? 'Style off' : st === 'na' ? 'Style n/a' : 'Style', g ? h('span.v', `v${g.version}`) : null);
      el.title = st === 'off' ? 'The project style is switched off for this job. Click to change.'
        : st === 'na' ? 'The project style does not apply to this kind of job. Click for details.' : `Project style “${g.name}” will be added to this job. Click to change.`;
      el.setAttribute('aria-label', el.title);
    }
    async function load() {
      const [r, p] = await Promise.all([fetchGuide(S.store.project), getPolicy()]);
      g = r.guide; pol = p; paint();
    }
    const unsub = S.bus.on('style', r => { if (r) { g = r.guide; paint(); } });
    const unsubP = S.bus.on('project', () => { g = null; paint(); load(); });
    load();
    function req() {
      const p = pref(), o = {};
      if (p.apply === false) o.apply = false;
      else { if (p.variant) o.variant = p.variant; if (p.strength) o.strength = p.strength; }
      return Object.keys(o).length ? o : null;
    }
    function openPop() {
      const p = pref();
      const prev = h('div.col', { style: { gap: '6px' } }, h('div.dim', 'Loading the preview…'));
      const apply = S.toggle(p.apply !== false, v => { const q = { ...pref() }; if (v) { delete q.apply; delete q.ver; } else { q.apply = false; q.ver = g.version; } setPref(q); refresh(); }, 'Apply to this job');
      const vsel = h('select.select.input-sm', { 'aria-label': 'Variant for this job', style: { width: 'auto', minWidth: '120px' },
        on: { change: () => { const q = { ...pref() }; if (vsel.value) q.variant = vsel.value; else delete q.variant; setPref(q); refresh(); } } },
        h('option', { value: '' }, `Auto${pol?.tasks?.[task] && pol.tasks[task] !== 'skip' ? ' (' + VLABEL[pol.tasks[task]] + ')' : ''}`),
        ...VARIANTS.map(([v, l]) => h('option', { value: v, selected: p.variant === v }, l)));
      const str = S.seg([{ value: '', label: 'Default' }, ...['light', 'normal', 'strong'].map(s => ({ value: s, label: s[0].toUpperCase() + s.slice(1) }))], p.strength || '', v => { const q = { ...pref() }; if (v) q.strength = v; else delete q.strength; setPref(q); refresh(); });
      async function refresh() {
        try {
          const params = { ...(box.o.params?.() || {}), prompt: box.value };
          if (box.negative) params.negative = box.negative;
          const r = await S.api.post(`${API}/compose`, { project: S.store.project, task, model: mid(), params, inputs: box.o.inputs?.() || [], style: req() || undefined });
          const bits = [];
          if (!r.applied) bits.push(h('div.dim', r.reason ? `Not applied: ${r.reason}.` : 'Not applied.'));
          else {
            bits.push(h('h4', 'Prompt the model gets'), renderPrompt(r));
            if (r.negative) bits.push(h('h4', 'Negative'), h('div.pv', r.negative));
            if (r.refs?.length) bits.push(h('div.dim', `${r.refs.length} reference image${r.refs.length > 1 ? 's' : ''} attached (about +${r.extra_seconds_est || 0} s).`));
            for (const s of r.skipped || []) bits.push(h('div.dim', `Skipped ${s.what}: ${s.why}.`));
            for (const n of r.notes || []) bits.push(h('div.dim', n));
          }
          prev.replaceChildren(...bits);
        } catch (e) { prev.replaceChildren(h('div.dim', e.message || 'No preview.')); }
      }
      const content = h('div.sty-pop',
        h('div.hd', ic('palette', 16), h('span.grow', g.name), h('span.badge.outline', `v${g.version}`), tip('chip')),
        h('div.rw', h('span.l', 'Apply to this job', tip('chip_apply')), apply),
        h('div.rw', h('span.l', 'Variant', tip('chip_variant')), vsel),
        h('div.rw', h('span.l', 'Strength', tip('chip_strength')), str),
        prev,
        h('a', { on: { click: () => { S.closePopover(); S.navigate('style'); } } }, 'Edit the style guide'));
      S.popover(el, content, { width: 'min(340px, calc(100vw - 24px))' });
      refresh();
    }
    return {
      el,
      meta() { const r = state() === 'hidden' ? null : req(); return r ? { style: r } : {}; },
      improve() { const r = state() === 'hidden' ? null : req(); return r ? { style: r } : {}; },
      setModel(m, t) { model = m; task = t || task; paint(); },
      destroy() { unsub?.(); unsubP?.(); },
    };
  },
});

// ------------------------------------------------------------------ Library: "Style guide from selection"
S.registerSelectionAction({
  id: 'style-from-selection', label: 'Style guide from selection', icon: 'palette', types: ['image'], min: 1, max: 4,
  run: ids => S.sendTo('style', ids, { action: 'style_from' }),
});

// ------------------------------------------------------------------ the page
export default S.defineStudio({
  id: 'style', title: 'Style guide', icon: 'palette', section: 'workspace', order: 55, mobile: false,
  receives: [{ action: 'style_from', label: 'Use as style', icon: 'palette', types: ['image'] }],
  async mount(el) {
    const api = S.api;
    const page = h('div.page-pad.sty-page');
    const headHost = h('div'), body = h('div');
    page.append(headHost, body);
    el.append(page);

    let visible = false, loaded = false;
    let guide = null, draft = null, resp = { warnings: [], missing_refs: [], refs_info: [] };
    let policy = { tasks: {}, models: {}, labels: {}, test_subjects: '' };
    let activeVar = null, tabs = null, previewBox = null, prevTok = 0, saving = false;
    const pv = { task: 't2i', model: '', variant: '', prompt: '' };

    const dirty = () => !!draft && (!guide || JSON.stringify(content(draft)) !== JSON.stringify(content(guide)));
    const pid = () => S.store.project;
    const proj = () => S.store.projects.find(p => p.id === pid());
    const order = () => { const o = PROFILE_ORDER[S.store.profile]; return o || VARIANTS.map(v => v[0]); };
    const edit = () => { drawHead(); schedulePreview(); };
    const schedulePreview = S.debounce(() => runPreview(), 300);

    // ---------------------------------------------------------------- load / save
    async function load(force = true) {
      const [r, p] = await Promise.all([fetchGuide(pid(), force), getPolicy()]);
      policy = p; resp = r; guide = r.guide; draft = guide ? clone(guide) : null; loaded = true;
      if (!activeVar || !order().includes(activeVar)) activeVar = order()[0];
      draw();
    }
    function applyResp(r) {
      resp = r; guide = r.guide; draft = guide ? clone(guide) : null; GUIDES.set(pid(), Promise.resolve(r));
      draw();
    }
    const payload = () => content(draft);
    async function save(base = guide ? guide.version : 0) {
      if (saving || !draft) return;
      saving = true; drawHead();
      try {
        const r = await api.put(`${API}/guide`, { project: pid(), guide: payload(), base_version: base, by: 'ui' });
        applyResp(r);
        S.toast(r.created_version ? `Style guide saved (version ${r.guide.version})` : 'Nothing changed, no new version', { kind: 'ok', timeout: 2000 });
      } catch (e) {
        if (e.status === 409) conflict(e.data?.current_version);
        else S.toastError(e);
      } finally { saving = false; drawHead(); }
    }
    function conflict(cur) {
      const md = S.modal({ title: 'The guide was changed meanwhile', icon: 'triangle-alert', width: 460,
        body: h('p', { style: { margin: 0 } }, `Version ${cur} was saved from somewhere else (another tab, the Assistant or an MCP client) while you were editing. You can keep editing, load that version (your edits are lost) or save yours on top as a new version.`),
        foot: [btn('Keep editing', { kind: 'ghost', onClick: () => md.close() }),
          btn('Load theirs', { onClick: () => { md.close(); load(true).catch(S.toastError); } }),
          btn('Save mine on top', { kind: 'primary', onClick: () => { md.close(); save(cur); } })] });
    }

    // ---------------------------------------------------------------- header
    function drawHead() {
      const p = proj();
      const d = dirty();
      const tools = [];
      if (draft) {
        tools.push(
          h('span.lab', 'Enabled', tip('enabled'), S.toggle(draft.enabled, v => { draft.enabled = v; edit(); }, 'Style guide enabled')),
          h('span.lab', 'Strength', tip('strength'), S.seg([['light', 'Light'], ['normal', 'Normal'], ['strong', 'Strong']].map(([value, label]) => ({ value, label })), draft.strength, v => { draft.strength = v; edit(); })),
          guide ? h('span.badge.outline', { title: `Content hash ${guide.hash}` }, `v${guide.version}`) : h('span.badge.warn', 'Not saved yet'),
          d && guide ? h('span.badge.warn', 'Unsaved changes') : null,
          btn('History', { icon: 'history', onClick: () => openHistory() }),
          btn(saving ? 'Saving…' : 'Save', { kind: 'primary', icon: 'check', disabled: !d || saving, onClick: () => save() }));
      }
      headHost.replaceChildren(h('div.sty-head',
        h('div.grow', h('h1.page-title', 'Style guide'), h('p.page-sub', `One look for everything in project “${p?.title || pid()}”. It is added to the prompt of every image, texture, video and sound job on the server, so Retry and the Assistant use it too.`)),
        h('div.sty-tools', ...tools)));
    }

    // ---------------------------------------------------------------- body
    function draw() {
      drawHead();
      S.closeOrphanInfo();
      if (!loaded) { body.replaceChildren(S.empty('palette', 'Loading the style guide…')); return; }
      if (!draft) { body.replaceChildren(drawStart()); return; }
      const warns = (resp.warnings || []).slice();
      const left = h('div.sty-col', lookCard(warns), refsCard(), paletteCard(), variantsCard());
      previewBox = h('div.sty-prev.col', { style: { gap: '10px' } });
      body.replaceChildren(h('div.sty-grid', left, h('div.sty-side', previewCard())));
      runPreview();
    }

    function field(label, infoKey, control, extra = null) {
      return h('div.sty-fld', h('div.field-label', label, infoKey ? tip(infoKey) : null, extra), control);
    }
    function counter(input, cap) {
      const c = h('span.cnt');
      const upd = () => { c.textContent = `${input.value.length}/${cap}`; c.classList.toggle('over', input.value.length > cap); };
      input.addEventListener('input', upd); upd();
      return c;
    }

    function drawStart() {
      const others = S.store.projects.filter(p => p.id !== pid());
      return h('div.col', { style: { gap: '14px' } },
        S.empty('palette', 'No style guide in this project yet', 'A style guide keeps every image, texture, video and sound in this project looking like it belongs together. Pick a way to begin.'),
        h('div.sty-start',
          h('button', { type: 'button', on: { click: () => startFromImages() } }, h('b', ic('image', 18), 'From reference images'), h('span', 'Pick 1 to 4 finished pictures. The palette is read from the first one, and they become the reference images.')),
          h('button', { type: 'button', on: { click: () => { draft = newGuide(); guide = null; draw(); } } }, h('b', ic('pencil', 18), 'Write it myself'), h('span', 'Start with an empty guide and describe the look in words.')),
          others.length ? h('button', { type: 'button', on: { click: e => copyFrom(e.currentTarget, others) } }, h('b', ic('copy', 18), 'Copy from another project', tip('copy_project')), h('span', 'Reuse the text, avoid list, palette and variants of another project.')) : null));
    }
    async function startFromImages(ids = null) {
      ids = ids || await S.pickAssets({ types: ['image'], multiple: true, title: 'Pick 1 to 4 images that show the style' });
      if (!ids?.length) return;
      if (ids.length > 4) { S.toast('Only the first 4 images are used.', { kind: 'warn', timeout: 2500 }); ids = ids.slice(0, 4); }
      try {
        const r = await api.post(`${API}/draft`, { project: pid(), assets: ids, describe: false });
        draft = r.guide; if (guide) { draft.name = guide.name; } resp = { ...resp, warnings: r.warnings || [] };
        if (!loaded) loaded = true;
        draw();
        S.toast('Draft made from your images. Edit it and press Save.', { kind: 'ok', timeout: 3200 });
      } catch (e) { S.toastError(e); }
    }
    function copyFrom(anchor, others) {
      S.menu(anchor, others.map(p => ({ label: p.title || p.id, sub: p.id, icon: 'folder', onClick: async () => {
        try {
          const r = await fetchGuide(p.id, true);
          if (!r.guide) { S.toast(`Project “${p.title || p.id}” has no style guide.`, { kind: 'warn' }); return; }
          const g = content(r.guide); g.refs = []; g.palette.source = null;
          draft = g; guide = null; draw();
          S.toast('Copied. Reference images stay with their project; add new ones if you like, then Save.', { kind: 'ok', timeout: 3600 });
        } catch (e) { S.toastError(e); }
      } })));
    }

    // ---- Look
    function lookCard(warns) {
      const pos = h('textarea.textarea', { rows: 3, maxLength: 400, placeholder: 'hand-painted fantasy game art, thick dark outlines, flat two-tone shading, warm lighting', 'aria-label': 'The look', value: draft.positive,
        on: { input: () => { draft.positive = pos.value; edit(); } } });
      const bg = h('input.input', { maxLength: 160, placeholder: 'plain cream parchment background', 'aria-label': 'Background', value: draft.background, on: { input: () => { draft.background = bg.value; edit(); } } });
      const name = h('input.input', { maxLength: 60, 'aria-label': 'Guide name', value: draft.name, on: { input: () => { draft.name = name.value; edit(); } } });
      const hasRef = (draft.refs || []).length > 0;
      const describe = btn('Describe from reference', { size: 'sm', icon: 'wand-sparkles', disabled: !hasRef, title: hasRef ? 'Let the prompt model write the look from your first reference image' : 'Add a reference image first',
        onClick: () => describeFromRef(describe) });
      return h('div.card.sty-card',
        h('header', ic('palette', 17), h('h3', 'Look'), h('span.spacer'), describe, tip('describe')),
        warns.length ? h('div.sty-warn', ...warns.map(w => h('div', ic('triangle-alert', 14), w))) : null,
        field('Name', null, name),
        field('The look', 'positive', pos, counter(pos, 400)),
        field('Background', 'background', bg, counter(bg, 160)),
        field('Avoid', 'avoid', tagInput(draft.avoid, edit)));
    }
    async function describeFromRef(b) {
      const first = draft.refs[0]?.asset;
      if (!first) return;
      b.disabled = true;
      const old = b.textContent;
      b.lastChild.textContent = 'Describing…';
      try {
        const r = await api.post(`${API}/draft`, { project: pid(), assets: [first], describe: true });
        const g = r.guide;
        const have = (draft.positive || draft.background || draft.avoid.length);
        if (have && !(await S.confirmDialog('Replace the look, background and avoid list you wrote with the description of the reference?', { title: 'Replace your text?', ok: 'Replace' }))) return;
        draft.positive = g.positive || draft.positive; draft.background = g.background || draft.background;
        if (g.avoid?.length) draft.avoid = g.avoid;
        draw();
        S.toast(r.describe?.llm ? `Described with ${r.describe.llm}. Check it, then Save.` : 'Described. Check it, then Save.', { kind: 'ok', timeout: 3200 });
      } catch (e) { S.toastError(e); } finally { b.disabled = false; b.lastChild.textContent = old; }
    }

    // ---- References
    function refsCard() {
      const info = Object.fromEntries((resp.refs_info || []).map(r => [r.asset, r]));
      const tiles = (draft.refs || []).map((r, i) => {
        const inf = info[r.asset] || {};
        const gone = inf.exists === false;
        return h(`div.sty-ref${gone ? '.gone' : ''}`, { title: inf.model_name ? `${inf.name || r.name || r.asset} · made with ${inf.model_name}` : (r.name || r.asset) },
          thumb(r.asset),
          gone ? h('span.badge.warn.b', 'missing') : inf.changed ? h('span.badge.warn.b', 'changed') : null,
          h('button.x', { type: 'button', 'aria-label': 'Remove reference', on: { click: () => { draft.refs.splice(i, 1); draw(); } } }, ic('x', 13)),
          h('div.nm', inf.name || r.name || r.asset));
      });
      const add = (draft.refs || []).length < 4 ? h('button.sty-add', { type: 'button', on: { click: async () => {
        const ids = await S.pickAssets({ types: ['image'], multiple: true, title: 'Pick reference images', exclude: draft.refs.map(r => r.asset) });
        if (!ids?.length) return;
        for (const id of ids) if (draft.refs.length < 4) { const a = await assetMeta(id); draft.refs.push({ asset: id, name: a?.name }); }
        resp = { ...resp, refs_info: (resp.refs_info || []).filter(r => draft.refs.some(x => x.asset === r.asset)) };
        draw();
      } } }, ic('plus', 20), 'Add') : null;
      return h('div.card.sty-card',
        h('header', ic('image', 17), h('h3', 'Reference images'), h('span.badge.outline', `${draft.refs.length}/4`), tip('refs')),
        h('div.sty-refs', ...tiles, add),
        h('p.field-help', { style: { margin: 0 } }, 'Used on models that accept style references (Qwen-Image 2.1 for now), at Normal strength or stronger. Each one adds about 20 s per image.'));
    }

    // ---- Palette
    function paletteCard() {
      const pal = draft.palette;
      const sw = h('div.sty-sw');
      const paintSw = () => {
        sw.replaceChildren(...pal.colors.map((c, i) => {
          const inp = h('input', { type: 'color', value: c, 'aria-label': `Palette colour ${i + 1}`, on: { input: () => { pal.colors[i] = inp.value; wrap.style.background = inp.value; hex.textContent = inp.value; edit(); } } });
          const wrap = h('div.sty-swatch', { style: { background: c } }, inp, h('button.x', { type: 'button', 'aria-label': 'Remove colour', on: { click: () => { pal.colors.splice(i, 1); delete pal.weights; paintSw(); edit(); } } }, ic('x', 11)));
          const hex = h('span.sty-hex', c);
          return h('div.col', { style: { alignItems: 'center', gap: '3px' } }, wrap, hex);
        }),
        pal.colors.length < 12 ? h('button.sty-add', { type: 'button', style: { width: '44px', aspectRatio: '1', height: '44px' }, 'aria-label': 'Add a colour', title: 'Add a colour', on: { click: () => { pal.colors.push('#808080'); delete pal.weights; paintSw(); edit(); } } }, ic('plus', 18)) : null,
        !pal.colors.length ? h('span.field-help', 'No colours yet.') : null);
      };
      paintSw();
      const refOpts = (draft.refs || []);
      const ex = btn('Extract from a reference', { size: 'sm', icon: 'wand-sparkles', disabled: !refOpts.length, title: refOpts.length ? 'Pick the most distinct colours of a reference image' : 'Add a reference image first', onClick: async () => {
        const cur = refOpts.findIndex(r => r.asset === pal.source);
        const id = refOpts[(cur + 1) % refOpts.length].asset;      // a second click reads the next reference
        try {
          const r = await api.post(`${API}/palette/extract`, { project: pid(), asset: id, n: 6 });
          pal.colors = r.colors; pal.weights = r.weights; pal.source = id; paintSw(); edit();
          S.toast('Palette read from the reference.', { kind: 'ok', timeout: 1800 });
        } catch (e) { S.toastError(e); }
      } });
      const tint = h('input', { type: 'range', min: 0.1, max: 0.8, step: 0.05, value: pal.tint_strength, 'aria-label': 'Tint amount', on: { input: () => { pal.tint_strength = Number(tint.value); tv.textContent = pal.tint_strength.toFixed(2); edit(); } } });
      const tv = h('span.mono', pal.tint_strength.toFixed(2));
      const tintRow = h('div.sty-fld', h('div.field-label', 'Tint amount', tip('tint_strength')), h('div.sty-range', tint, tv));
      const post = S.seg([['off', 'Off'], ['tint', 'Tint'], ['lock', 'Lock']].map(([value, label]) => ({ value, label })), pal.post, v => { pal.post = v; tintRow.classList.toggle('hidden', v !== 'tint'); edit(); });
      tintRow.classList.toggle('hidden', pal.post !== 'tint');
      const inPrompt = S.toggle(pal.in_prompt, v => { pal.in_prompt = v; edit(); }, 'Name colours in the prompt');
      return h('div.card.sty-card',
        h('header', ic('palette', 17), h('h3', 'Palette'), h('span.badge.outline', `${pal.colors.length}/12`), tip('palette'), h('span.spacer'), ex, tip('palette_extract')),
        sw,
        h('div.sty-row', h('span.lab', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', fontSize: '12.5px', fontWeight: 550, color: 'var(--text-2)' } }, 'Name colours in the prompt', tip('palette_in_prompt'), inPrompt)),
        field('After generating', 'palette_post', h('div.sty-row', post, h('span.field-help', 'Changes finished pictures; the original is kept.'))),
        tintRow);
    }

    // ---- Variants
    function variantItems() {
      return order().map(v => {
        const mod = !!(draft.variants || {})[v] && JSON.stringify({ ...vDefaults(v), ...draft.variants[v] }) !== JSON.stringify(vDefaults(v));
        return { id: v, label: VLABEL[v] + (mod ? ' •' : ''), title: mod ? 'Changed from the default' : 'Same as the default' };
      });
    }
    function variantsCard() {
      const host = h('div.sty-var');
      const drawVar = () => host.replaceChildren(...variantBody(activeVar));
      tabs = new S.TabGroups({ items: variantItems(), value: activeVar, label: 'Variants', onSelect: id => { activeVar = id; drawVar(); } });
      host.refresh = () => { tabs.setItems(variantItems(), activeVar); };
      drawVar();
      variantsHost = host;
      return h('div.card.sty-card', { style: { padding: 0, overflow: 'hidden' } },
        h('header', { style: { padding: '14px 16px 0' } }, ic('sliders-horizontal', 17), h('h3', 'Variants'), tip('variants'), h('span.spacer'),
          h('span.field-help', 'Only what differs from the main guide.')),
        tabs.el, h('div', { style: { padding: '0 16px 14px' } }, host));
    }
    let variantsHost = null;
    function vset(vid, k, val) {
      draft.variants = draft.variants || {};
      draft.variants[vid] = { ...vEff(draft, vid), [k]: val };
      variantsHost?.refresh?.();
      edit();
    }
    function variantBody(vid) {
      const v = vEff(draft, vid);
      const bgMode = v.background === null ? 'inherit' : v.background === '' ? 'none' : 'custom';
      const bgIn = h('input.input', { maxLength: 160, placeholder: 'a different background', value: v.background || '', 'aria-label': 'Background for this variant', on: { input: () => vset(vid, 'background', bgIn.value || ' ') } });
      bgIn.classList.toggle('hidden', bgMode !== 'custom');
      const bgSeg = S.seg([['inherit', 'Inherit'], ['none', 'None'], ['custom', 'Custom']].map(([value, label]) => ({ value, label })), bgMode, m => {
        bgIn.classList.toggle('hidden', m !== 'custom');
        vset(vid, 'background', m === 'inherit' ? null : m === 'none' ? '' : (bgIn.value.trim() || 'plain background'));
        if (m === 'custom') { bgIn.value = vEff(draft, vid).background || ''; bgIn.focus(); }
      });
      const pos = h('textarea.textarea', { rows: 2, maxLength: 300, placeholder: vid === 'sprite' ? 'crisp silhouette, side view' : vid === 'audio' ? 'warm, close, intimate' : 'extra look for this kind of job', value: v.positive, 'aria-label': 'Extra look',
        on: { input: () => vset(vid, 'positive', pos.value) } });
      const avoid = v.avoid.slice();
      const rows = [
        h('div.sty-row', h('span.lab', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', fontSize: '12.5px', fontWeight: 550, color: 'var(--text-2)' } }, `Use the ${VLABEL[vid]} variant`, tip('variant_enabled'), S.toggle(v.enabled, x => vset(vid, 'enabled', x), 'Use this variant')),
          h('span.spacer'), btn('Reset', { size: 'sm', kind: 'ghost', icon: 'rotate-ccw', title: 'Back to the default for this variant', onClick: () => { delete draft.variants?.[vid]; variantsHost?.refresh?.(); edit(); drawVarAgain(); } })),
        field('Combine with the main look', 'variant_mode', S.seg([['extend', 'Extend'], ['replace', 'Replace']].map(([value, label]) => ({ value, label })), v.mode, m => vset(vid, 'mode', m))),
        field('Extra look', 'variant_positive', pos, counter(pos, 300)),
        field('Background', 'variant_background', h('div.col', { style: { gap: '6px' } }, bgSeg, bgIn)),
        field('Extra avoid terms', 'variant_avoid', tagInput(avoid, () => vset(vid, 'avoid', avoid.slice()))),
        h('div.sty-row', h('span.lab', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', fontSize: '12.5px', fontWeight: 550, color: 'var(--text-2)' } }, 'Use reference images', tip('variant_refs'), S.toggle(v.refs, x => vset(vid, 'refs', x), 'Use reference images')))];
      if (PALETTE_VARIANTS.includes(vid)) {
        rows.push(field('Palette post-process', 'variant_post', S.seg([['', 'Inherit'], ['off', 'Off'], ['tint', 'Tint'], ['lock', 'Lock']].map(([value, label]) => ({ value, label })), v.palette_post || '', m => vset(vid, 'palette_post', m || null))));
      }
      return rows;
    }
    function drawVarAgain() { if (variantsHost) variantsHost.replaceChildren(...variantBody(activeVar)); }

    // ---- Preview
    function previewCard() {
      const taskIds = Object.entries(policy.tasks || {}).filter(([t, v]) => v !== 'skip' && S.modelsFor(t).length).map(([t]) => t);
      if (!taskIds.includes(pv.task)) pv.task = taskIds.includes('t2i') ? 't2i' : taskIds[0] || 't2i';
      const taskSel = h('select.select.input-sm', { 'aria-label': 'Task', on: { change: () => { pv.task = taskSel.value; pv.model = ''; pv.prompt = ''; fillModels(); samplePh(); runPreview(); } } },
        ...taskIds.map(t => h('option', { value: t, selected: t === pv.task }, policy.labels?.[t] || t)));
      const modelSel = h('select.select.input-sm', { 'aria-label': 'Model', on: { change: () => { pv.model = modelSel.value; runPreview(); } } });
      const fillModels = () => modelSel.replaceChildren(h('option', { value: '' }, 'Default model'), ...S.modelsFor(pv.task).map(m => h('option', { value: m.id, selected: m.id === pv.model }, m.name || m.id)));
      fillModels();
      const varSel = h('select.select.input-sm', { 'aria-label': 'Variant', on: { change: () => { pv.variant = varSel.value; runPreview(); } } },
        h('option', { value: '' }, 'Variant: automatic'), ...VARIANTS.map(([v, l]) => h('option', { value: v, selected: pv.variant === v }, l)));
      const sample = h('input.input.input-sm', { 'aria-label': 'Sample prompt', value: pv.prompt, on: { input: () => { pv.prompt = sample.value; schedulePreview(); } } });
      const samplePh = () => { sample.placeholder = SAMPLE[policy.tasks?.[pv.task]] || 'a sample prompt'; };
      samplePh();
      const test = btn('Test on 4 subjects', { size: 'sm', icon: 'sparkles', disabled: !guide || dirty(), title: !guide ? 'Save the guide first' : dirty() ? 'Save your changes first: the test uses the saved guide' : '',
        onClick: () => testSubjects() });
      return h('div.card.sty-card.sty-prev',
        h('header', ic('eye', 17), h('h3', 'Preview'), tip('preview'), h('span.spacer'), test, tip('test_subjects')),
        h('div.sty-row', h('div.grow', taskSel), h('div.grow', modelSel)),
        h('div.sty-row', h('div.grow', varSel), h('div.grow', sample)),
        previewBox);
    }
    async function runPreview() {
      if (!previewBox || !draft) return;
      const tok = ++prevTok;
      try {
        const kind = policy.tasks?.[pv.task];
        const params = { prompt: pv.prompt.trim() || SAMPLE[kind] || 'an iron short sword' };
        const r = await api.post(`${API}/compose`, { project: pid(), task: pv.task, model: pv.model || undefined, params, guide: payload(), style: pv.variant ? { variant: pv.variant } : undefined });
        if (tok !== prevTok) return;
        const bits = [];
        if (!r.applied) bits.push(h('div.sty-warn', h('div', ic('triangle-alert', 14), r.reason ? `Not applied: ${r.reason}.` : 'Not applied.')));
        bits.push(h('h4', `Prompt for ${r.model || 'the default model'}`), renderPrompt(r));
        if (r.applied) {
          if (r.negative) bits.push(h('h4', 'Negative prompt'), h('div.pv.neg', r.negative));
          if (r.refs?.length) bits.push(h('h4', `Reference images (${r.refs.length}) · about +${r.extra_seconds_est || 0} s`), h('div.refs', ...r.refs.map(x => { const i = h('img', { alt: x.name || '', title: x.name || '' }); assetMeta(x.asset).then(a => { if (a) i.src = S.thumbUrl(a); }); return i; })));
          if (r.post && r.post.mode && r.post.mode !== 'off') bits.push(h('div.why', `After generating: palette ${r.post.mode}.`));
          const why = [...(r.skipped || []).map(s => `${s.what}: ${s.why}`), ...(r.notes || [])];
          if (why.length) bits.push(h('div.why', ...why.map(w => h('span', w))));
          bits.push(h('div.why', `Variant ${VLABEL[r.variant] || r.variant}, strength ${r.strength}${r.version ? `, version ${r.version}` : ''}.`));
        }
        previewBox.replaceChildren(...bits);
      } catch (e) { if (tok === prevTok) previewBox.replaceChildren(h('div.sty-warn', h('div', ic('triangle-alert', 14), e.message || 'No preview.'))); }
    }
    async function testSubjects() {
      const subjects = (policy.test_subjects || '').split(',').map(s => s.trim()).filter(Boolean);
      const task = ['t2i', 'rgba'].includes(pv.task) ? pv.task : 't2i';
      const model = ['t2i', 'rgba'].includes(pv.task) && pv.model ? pv.model : undefined;
      if (!(await S.confirmDialog(`Generate ${subjects.length} pictures (${subjects.join(', ')}) with the saved guide? They run one after another in the queue.`, { title: 'Test the style', ok: `Generate ${subjects.length}` }))) return;
      let n = 0;
      for (const s of subjects) {
        try {
          const job = await api.submit({ task, model, params: { prompt: s }, project: pid(), meta: { tags: ['style-test'] } });
          S.store.jobs.set(job.id, job); n++;
        } catch (e) { S.toastError(e); break; }
      }
      if (n) S.toast(`${n} test job${n > 1 ? 's' : ''} queued. The results appear in the Library.`, { kind: 'ok', timeout: 3600, action: { label: 'Open Library', onClick: () => S.navigate('library') } });
    }

    // ---------------------------------------------------------------- history
    async function openHistory(focus = null) {
      let vs;
      try { vs = (await api.get(`${API}/versions`, { project: pid() })).versions; } catch (e) { S.toastError(e); return; }
      const list = h('div.sty-hist');
      const md = S.modal({ title: 'Style guide history', icon: 'history', width: 640, body: list, foot: [btn('Close', { kind: 'ghost', onClick: () => md.close() })] });
      if (!vs.length) list.append(S.empty('history', 'No versions yet', 'Every save keeps a copy here.'));
      for (const v of vs) {
        const detail = h('div.hidden');
        const cur = guide && guide.version === v.version;
        const row = h(`div.sty-hrow${cur ? '.cur' : ''}`, { id: `sty-v${v.version}` },
          h('div.top', h('b', `Version ${v.version}`), cur ? h('span.badge.accent', 'current') : null, h('span.badge.outline', v.saved_by || 'api'), h('span.dim', { title: v.saved }, v.saved ? S.fmtAgo(v.saved) : ''), h('span.spacer', { style: { flex: 1 } }),
            btn('View', { size: 'sm', kind: 'ghost', icon: 'eye', onClick: async () => {
              if (!detail.classList.contains('hidden')) { detail.classList.add('hidden'); return; }
              try { const g = await api.get(`${API}/versions/${v.version}`, { project: pid() }); detail.replaceChildren(h('pre', describeGuide(g))); detail.classList.remove('hidden'); } catch (e) { S.toastError(e); }
            } }),
            cur ? null : btn('Restore', { size: 'sm', icon: 'rotate-ccw', onClick: async () => {
              if (dirty() && !(await S.confirmDialog('You have unsaved changes. Restoring replaces them.', { title: 'Restore this version?', ok: 'Restore' }))) return;
              try { const r = await api.post(`${API}/restore`, { project: pid(), version: v.version }); applyResp(r); md.close(); S.toast(`Version ${v.version} restored as version ${r.guide.version}.`, { kind: 'ok', timeout: 2600 }); } catch (e) { S.toastError(e); }
            } })),
          h('div.sum', v.summary || ''), detail);
        list.append(row);
        if (focus === v.version) setTimeout(() => { row.scrollIntoView({ block: 'center' }); row.querySelector('button')?.click(); }, 60);
      }
    }
    function describeGuide(g) {
      const L = [`name: ${g.name}`, `enabled: ${g.enabled}   strength: ${g.strength}`, `look: ${g.positive || '-'}`, `background: ${g.background || '-'}`, `avoid: ${(g.avoid || []).join(', ') || '-'}`,
        `references: ${(g.refs || []).map(r => r.name || r.asset).join(', ') || '-'}`, `palette: ${(g.palette?.colors || []).join(' ') || '-'}  post: ${g.palette?.post}  in prompt: ${g.palette?.in_prompt}`];
      for (const [vid, v] of Object.entries(g.variants || {})) L.push(`variant ${vid}: ${v.enabled === false ? 'off, ' : ''}${v.mode}, look: ${v.positive || '-'}, refs: ${v.refs}`);
      return L.join('\n');
    }

    // ---------------------------------------------------------------- events
    S.bus.on('style', r => { if (visible && r && !dirty()) { resp = r; guide = r.guide; draft = guide ? clone(guide) : null; loaded = true; draw(); } });
    S.bus.on('project', () => { loaded = false; guide = draft = null; draw(); if (visible) load(false).catch(S.toastError); });
    S.bus.on('models', () => { if (visible && draft) { prevTok++; draw(); } });
    addEventListener('beforeunload', e => { if (dirty()) { e.preventDefault(); e.returnValue = ''; } });

    drawHead();
    body.append(S.empty('palette', 'Loading the style guide…'));
    try { await load(false); } catch (e) { S.toastError(e); }
    return {
      show() { visible = true; if (!dirty()) load(true).catch(() => {}); },
      hide() { visible = false; S.closeOrphanInfo(); },
      route(r) {
        visible = true;
        const v = Number(r.query?.v);
        if (v) setTimeout(() => openHistory(v), 80);
      },
      async receive(p) {
        visible = true;
        const ids = (p.assets || []).map(a => a.id).filter(Boolean);
        if (!ids.length) return;
        if (guide && !(await S.confirmDialog('Start a new draft from these images? The editor is replaced by the draft; nothing is saved until you press Save, and the saved guide stays as it is.', { title: 'Use as style', ok: 'Start the draft' }))) return;
        await startFromImages(ids);
      },
    };
  },
});
