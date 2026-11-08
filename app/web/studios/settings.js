// Settings: every configurable value of the studio, system-wide (settings.json) and per project (project.json "settings").
// The page is generated from GET /api/settings/schema (app/settings_schema.py): label, type, range, scope, apply level.
// Saving: PATCH /api/settings (System scope) or PATCH /api/projects/<id>/settings (Project scope); null resets a key.
// A project override wins over the system value; the project is always the one in the top bar. docs/CONTRACTS.md section 10.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const CSS = `
.set-page { display: flex; flex-direction: column; gap: 14px; }
.set-head { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 16px; }
.set-head .grow { min-width: min(100%, 260px); }
.set-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; min-width: 0; }
.set-search { display: flex; align-items: center; gap: 8px; height: 34px; padding: 0 10px; border-radius: 9px; background: var(--panel); border: 1px solid var(--line); color: var(--text-3); min-width: 0; width: min(260px, 100%); }
.set-search input { border: 0; outline: 0; background: none; color: var(--text); flex: 1; min-width: 0; font: inherit; }
.set-search:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.set-layout { display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 18px; align-items: start; }
.set-layout > * { min-width: 0; }
.set-nav { position: sticky; top: 10px; display: flex; flex-direction: column; gap: 2px; }
.set-nav a { display: flex; align-items: center; gap: 9px; padding: 7px 10px; border-radius: 9px; color: var(--text-2); cursor: pointer; font-weight: 500; text-decoration: none; min-width: 0; }
.set-nav a:hover { background: var(--panel-2); color: var(--text); text-decoration: none; }
.set-nav a.on { background: var(--panel-3); color: var(--text); }
.set-nav a .n { margin-left: auto; font-size: 11px; color: var(--accent); background: var(--accent-soft); border-radius: 20px; padding: 0 6px; font-variant-numeric: tabular-nums; }
.set-groups { display: flex; flex-direction: column; gap: 14px; }
.set-group { scroll-margin-top: 12px; padding: 0; overflow: hidden; }
.set-group > header { display: flex; flex-direction: column; gap: 2px; padding: 14px 16px 10px; }
.set-group > header h3 { display: flex; align-items: center; gap: 8px; margin: 0; font-size: 15px; }
.set-group > header p { margin: 0; color: var(--text-3); font-size: 12.5px; }
.set-sub { padding: 8px 16px 4px; font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-3); font-weight: 600; border-top: 1px solid var(--line-soft); }
.set-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 320px); gap: 8px 18px; align-items: start; padding: 11px 16px; border-top: 1px solid var(--line-soft); }
.set-row > * { min-width: 0; }
.set-row.changed { box-shadow: inset 3px 0 0 var(--accent); }
.set-row.ro { background: color-mix(in srgb, var(--panel-2) 55%, transparent); }
.set-lab { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 8px; font-weight: 600; font-size: 13.5px; }
.set-lab .badge { font-weight: 500; }
.set-help { margin-top: 3px; font-size: 12px; color: var(--text-3); line-height: 1.45; overflow-wrap: anywhere; }
.set-help code, .set-ro { font-family: var(--mono); font-size: 12px; }
.set-ctl { display: flex; align-items: center; gap: 8px; min-width: 0; }
.set-ctl > .select, .set-ctl > .input, .set-ctl > .textarea, .set-ctl > .tg-pick { flex: 1 1 auto; min-width: 0; width: 100%; }
.set-ctl .tg-pick { display: flex; flex-direction: column; gap: 4px; }
.set-num { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; }
.set-num .input { flex: 1; min-width: 0; }
.set-num .unit { color: var(--text-3); font-size: 12px; white-space: nowrap; }
.set-ctl .textarea { min-height: 54px; resize: vertical; }
.set-ro { color: var(--text-2); display: flex; gap: 6px; align-items: flex-start; overflow-wrap: anywhere; }
.set-err { color: var(--err); font-size: 12px; margin-top: 3px; }
.set-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.set-banner { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 0 16px 10px; padding: 8px 12px; border-radius: 10px; background: var(--warn-soft); color: var(--text); font-size: 12.5px; }
.set-banner .grow { min-width: 160px; }
.set-note { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; color: var(--text-2); font-size: 12.5px; padding: 4px 2px; }
.set-foot { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.set-env td { vertical-align: top; }
.set-env td:first-child { white-space: nowrap; }
.set-env .v { font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
.set-fld { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.set-cloud { padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; border-top: 1px solid var(--line-soft); }
@media (max-width: 899px) {
  .set-layout { grid-template-columns: minmax(0, 1fr); gap: 10px; }
  .set-nav { position: static; flex-direction: row; overflow-x: auto; gap: 6px; padding-bottom: 4px; max-width: 100%; scrollbar-width: thin; }
  .set-nav a { flex: none; border: 1px solid var(--line); background: var(--panel); padding: 5px 11px; white-space: nowrap; }
  .set-nav a.on { border-color: var(--accent); background: var(--accent-soft); }
}
@media (max-width: 719px) {
  .set-row { grid-template-columns: minmax(0, 1fr); padding: 11px 12px; }
  .set-group > header { padding: 12px 12px 8px; }
  .set-sub { padding-left: 12px; padding-right: 12px; }
  .set-banner { margin: 0 12px 10px; }
  .set-cloud { padding: 12px; }
}
`;

const STUDIO_LABEL = { image: 'Image', '3d': '3D', sound: 'Sound', video: 'Video', assistant: 'Assistant', system: 'System', other: 'Other' };
const APPLY_BADGE = { new_worker: ['outline', 'new workers', 'Applies to workers that start after the change.'], llm_start: ['warn', 'LLM restart', 'Applies when the prompt LLM next starts.'],
  restart: ['warn', 'studio restart', 'Needs a restart of the studio to apply.'] };

const getPath = (o, key) => { let d = o; for (const k of key.split('.')) { if (d == null || typeof d !== 'object') return undefined; d = d[k]; } return d; };
const nest = (key, v) => { const parts = key.split('.'); const out = {}; let d = out; for (const p of parts.slice(0, -1)) d = d[p] = {}; d[parts.at(-1)] = v; return out; };
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

export default S.defineStudio({
  id: 'settings', title: 'Settings', icon: 'settings-2', order: 95, section: 'system',
  async mount(el) {
    document.head.append(h('style', CSS));
    const api = S.api;
    const page = h('div.page-pad.set-page');
    const headHost = h('div'), root = h('div');       // header (search keeps focus while typing) and the body that is redrawn
    page.append(headHost, root);
    el.append(page);

    let scope = 'system';              // 'system' | 'project'
    let schema = null, sys = null, proj = null, prov = null;
    let q = '', changedOnly = false, active = 'general', failed = null;
    const notes = new Map();           // group id -> Set of apply kinds saved since the page opened (banners)
    const rowEls = new Map();          // field key -> row element of the last draw
    let visible = false;

    // ------------------------------------------------------------------ data
    async function load() {
      try {
        const [sc, s] = await Promise.all([api.settingsSchema(), api.settings()]);
        schema = sc; sys = s;
        try { proj = await api.projectSettings(S.store.project); } catch (e) { proj = null; if (scope === 'project') throw e; }
        try { prov = await api.get('/api/x/providers'); } catch { prov = null; }
        failed = null;
      } catch (e) { failed = e; }
      draw();
    }
    const reload = S.debounce(load, 60);

    // ------------------------------------------------------------------ per-field state
    const fieldsOf = () => (schema?.fields || []).filter(f => f.scope.includes(scope));
    function state(f) {
      const sysV = getPath(sys, f.key) ?? null;
      if (scope === 'system') {
        const def = f.default ?? null;
        return { value: sysV, sysV, changed: !same(sysV, def) && f.editable !== false && f.type !== 'json' };
      }
      const over = !!proj && proj.overridden.includes(f.key);
      return { value: over ? (getPath(proj.overrides, f.key) ?? null) : sysV, sysV, over, changed: over };
    }
    const matches = f => {
      if (!q) return true;
      const hay = `${f.label} ${f.key} ${f.help || ''} ${f.subgroup || ''}`.toLowerCase();
      return q.toLowerCase().split(/\s+/).filter(Boolean).every(t => hay.includes(t));
    };

    // ------------------------------------------------------------------ info balloons
    // The (i) button and its balloon are the shared component core/components/infotip.js (S.infoBtn); this page only builds the specs.
    const { infoBtn, closeInfo } = S;
    const APPLY_WORDS = { live: 'right away', new_worker: 'for workers started after the change', llm_start: 'the next time the prompt LLM starts', restart: 'after the studio is restarted' };
    function fieldSpec(f) {
      const text = f.info || f.help || '';                    // an older core has no `info`: fall back to the one-line help
      const facts = [];
      if (f.type !== 'json') facts.push(['Default', fmt(f, f.default ?? null)]);
      if ((f.type === 'int' || f.type === 'float') && f.min != null && f.max != null) facts.push(['Range', `${f.min} to ${f.max}${f.unit ? ' ' + f.unit : ''}`]);
      else if (f.type === 'text' || f.type === 'string') { if (f.maxlen) facts.push(['Length', `up to ${f.maxlen} characters`]); }
      if (f.editable === false) facts.push(['Changed by', 'editing settings.json by hand']);
      facts.push(['Takes effect', APPLY_WORDS[f.apply] || f.apply || 'right away']);
      facts.push(['Scope', f.scope?.includes('project') ? 'System, or overridden per project' : 'System only']);
      facts.push(['Key', f.key, true]);
      return { title: f.label, text, facts };
    }
    const groupSpec = g => (g.info ? { title: g.label, text: g.info } : null);
    const envSpec = e => (e.info ? { title: e.label, text: e.info, facts: [['Variable', e.key, true], ['Change it', 'restart the studio with a different value']] } : null);
    const CLOUD_INFO = { allow: S.UI_INFO.cloud_allow, budget: S.UI_INFO.cloud_budget, confirm: S.UI_INFO.cloud_confirm };    // texts live in core/uiinfo.js (shared with the Providers page)

    // ------------------------------------------------------------------ saving
    async function save(f, value, { quiet = false } = {}) {
      try {
        if (scope === 'system') await api.patchSettings(nest(f.key, value));
        else await api.patchProjectSettings(S.store.project, nest(f.key, value));
        if (f.apply !== 'live') { (notes.get(f.group) || notes.set(f.group, new Set()).get(f.group)).add(f.apply); }
        await S.refreshEffective().catch(() => {});
        reload();
        if (!quiet) S.toast(value === null ? `${f.label}: ${scope === 'project' ? 'using the system value' : 'reset to default'}` : `${f.label} saved`, { kind: 'ok', timeout: 1400 });
      } catch (e) { S.toastError(e, `Could not save ${f.label}`); draw(true); }
    }

    // ------------------------------------------------------------------ formatting
    function fmt(f, v) {
      if (v == null) return f.type === 'model' ? 'registry default' : f.key === 'export.preset' ? 'profile default' : f.type === 'backend' ? 'automatic' : f.type === 'text' ? 'empty' : f.nullable ? 'automatic' : '–';
      if (f.type === 'bool') return v ? 'on' : 'off';
      if (f.type === 'choice') return (f.choices || []).find(c => c.value === v)?.label || String(v);
      if (f.type === 'model') return S.store.modelsById.get(v)?.name || String(v);
      if (f.type === 'list_str') return v.join(' ');
      if (f.type === 'json') return JSON.stringify(v);
      return `${v}${f.unit ? ' ' + f.unit : ''}`;
    }

    // ------------------------------------------------------------------ controls
    function control(f, st) {
      const disabled = f.editable === false;
      const val = st.value;
      const wrap = h('div.set-ctl');
      const t = f.type;
      if (disabled) {
        wrap.append(h('div.set-ro', ic('lock', 13), h('span', t === 'list_str' ? (val || []).join(' ') : String(val ?? '–'))));
        return wrap;
      }
      if (t === 'bool') {
        const tg = S.toggle(!!val, v => save(f, v), f.label);
        wrap.append(tg, h('span.dim', { style: { fontSize: '12.5px' } }, val ? 'On' : 'Off'));
      } else if (t === 'int' || t === 'float') {
        const err = h('div.set-err.hidden');
        const inp = h('input.input', { type: 'number', min: f.min, max: f.max, step: f.step ?? (t === 'int' ? 1 : 'any'), value: val ?? '', placeholder: val == null ? 'automatic' : '', 'aria-label': f.label });
        const commit = S.debounce(() => {
          const raw = inp.value.trim();
          const bad = m => { err.textContent = m; err.classList.remove('hidden'); };
          if (raw === '') { if (f.nullable) { err.classList.add('hidden'); save(f, null); } else bad('Enter a value.'); return; }
          const n = Number(raw);
          if (!Number.isFinite(n)) return bad('Enter a number.');
          if (t === 'int' && !Number.isInteger(n)) return bad('Enter a whole number.');
          if ((f.min != null && n < f.min) || (f.max != null && n > f.max)) return bad(`Between ${f.min} and ${f.max}${f.unit ? ' ' + f.unit : ''}.`);
          err.classList.add('hidden');
          if (!same(n, val)) save(f, n);
        }, 400);
        inp.addEventListener('input', commit);
        wrap.append(h('div', { style: { flex: 1, minWidth: 0 } }, h('div.set-num', inp, f.unit ? h('span.unit', f.unit) : null), err));
      } else if (t === 'string') {
        const err = h('div.set-err.hidden');
        const listId = f.suggestions?.length ? `set-dl-${f.key.replace(/\W/g, '-')}` : null;
        const inp = h('input.input', { type: 'text', value: val ?? '', maxlength: f.maxlen || 200, placeholder: f.nullable ? 'automatic' : '', 'aria-label': f.label, list: listId, spellcheck: false });
        const commit = S.debounce(() => {
          const raw = inp.value.trim();
          const bad = m => { err.textContent = m; err.classList.remove('hidden'); };
          if (raw === '') { if (f.nullable) { err.classList.add('hidden'); if (val != null) save(f, null); } else bad('Enter a value.'); return; }
          if (f.pattern && !new RegExp(`^(?:${f.pattern})$`).test(raw)) return bad('Contains characters that are not allowed.');
          if (f.file_check && !/^(lmstudio:|hf:|\/|~)/.test(raw)) return bad('Start with lmstudio:, hf: or an absolute path.');
          err.classList.add('hidden');
          if (raw !== val) save(f, raw);
        }, 400);
        inp.addEventListener('input', commit);
        const file = f.file_check && val ? (f.file ? h(`span.badge.${f.file.found ? 'ok' : 'err'}`, { title: f.file.path || '' }, f.file.found ? 'file found' : 'file missing') : null) : null;
        wrap.append(h('div', { style: { flex: 1, minWidth: 0 } }, inp, listId ? h('datalist', { id: listId }, ...f.suggestions.map(x => h('option', { value: x }))) : null, file ? h('div', { style: { marginTop: '4px' } }, file) : null, err));
      } else if (t === 'text') {
        const ta = h('textarea.textarea', { rows: 2, maxlength: f.maxlen || 300, value: val ?? '', placeholder: scope === 'project' ? 'Inherit' : '', 'aria-label': f.label });
        const commit = S.debounce(() => { const raw = ta.value.trim(); if (raw !== (val ?? '')) save(f, raw === '' ? null : raw); }, 400);
        ta.addEventListener('input', commit);
        wrap.append(ta);
      } else if (t === 'choice') {
        const sel = h('select.select', { 'aria-label': f.label, on: { change: () => save(f, sel.value === '' ? null : sel.value) } });
        if (f.nullable) {
          const inherit = scope === 'project' ? `Inherit: ${fmtInherit(f, st.sysV)}` : `Default (${fmtInherit(f, null)})`;
          sel.append(h('option', { value: '', selected: val == null }, inherit));
        }
        for (const c of f.choices || []) sel.append(h('option', { value: c.value, selected: c.value === val, title: c.help || '' }, c.label));
        wrap.append(sel);
      } else if (t === 'model') {
        const sel = h('select.select', { 'aria-label': f.label, on: { change: () => save(f, sel.value === '' ? null : sel.value) } });
        const reg = S.store.defaults?.[f.task];
        sel.append(h('option', { value: '', selected: val == null },
          scope === 'project' ? `Inherit: ${st.sysV ? S.store.modelsById.get(st.sysV)?.name || st.sysV : 'registry default'}` : `Registry default${reg && !S.isOverridden('defaults.' + f.task) ? ` (${S.store.modelsById.get(reg)?.name || reg})` : ''}`));
        const ms = S.modelsFor(f.task, { includeBlocked: true, includeTest: true }).slice();
        const rank = m => (m.status.status === 'ready' ? 0 : 1) + (S.isLicenseBlocked(m) ? 2 : 0);
        ms.sort((a, b) => rank(a) - rank(b));
        for (const m of ms) {
          const tag = S.isLicenseBlocked(m) ? ' · blocked by licence filter' : m.status.status !== 'ready' ? ` · ${m.status.status.replace('_', ' ')}` : '';
          sel.append(h('option', { value: m.id, selected: m.id === val, style: rank(m) ? { color: 'var(--text-3)' } : null }, m.name + tag));
        }
        if (val && !ms.find(m => m.id === val)) sel.append(h('option', { value: val, selected: true }, val));
        wrap.append(sel);
      } else if (t === 'backend') {
        const pk = S.backendPicker({ purpose: f.key.split('.')[1], value: val || '', allowDefault: true, label: f.label, onChange: v => { if ((v || null) !== val) save(f, v || null); } });
        wrap.append(pk);
      } else {
        wrap.append(h('div.set-ro', JSON.stringify(val)));
      }
      return wrap;
    }
    const fmtInherit = (f, v) => {
      if (v != null) return fmt(f, v);
      if (f.key === 'export.preset') return S.store.exportPresets[S.getProfile().export]?.label || S.getProfile().export || 'original';
      return fmt(f, null);
    };

    // ------------------------------------------------------------------ rows
    function jsonRow(f) {
      const v = getPath(sys, f.key);
      const items = Array.isArray(v) ? v.map(x => (typeof x === 'string' ? x : x?.model ? `${x.provider || x.backend || ''}${x.provider ? '/' : ''}${x.model}` : x?.id || x?.label || JSON.stringify(x)))
        : v && typeof v === 'object' ? Object.keys(v) : [];
      return h('div.set-ctl', { style: { flexDirection: 'column', alignItems: 'stretch' } },
        items.length ? h('div.set-chips', ...items.slice(0, 12).map(x => h('span.badge.outline', { style: { maxWidth: '100%' } }, h('span.ellipsis', x))), items.length > 12 ? h('span.dim', `+${items.length - 12}`) : null) : h('span.dim', { style: { fontSize: '12.5px' } }, 'None'),
        btn('Manage text backends', { size: 'sm', icon: 'sparkles', onClick: () => S.navigate('assistant') }));
    }
    function row(f) {
      const st = state(f);
      const ro = f.editable === false;
      const looser = scope === 'project' && st.over && (f.key === 'license.commercial_only' || f.key === 'license.strict') && getPath(sys, f.key) === true && st.value === false;
      const badges = [];
      if (APPLY_BADGE[f.apply]) { const [k, lab, tip] = APPLY_BADGE[f.apply]; badges.push(h(`span.badge.${k}`, { title: tip }, lab)); }
      if (ro) badges.push(h('span.badge.outline', { title: 'Executed or shared by the studio, so it is only changed by editing settings.json.' }, ic('lock', 11), 'file-only'));
      else if (scope === 'system' && st.changed) badges.push(h('span.badge.info', 'Changed'));
      else if (scope === 'project') badges.push(st.over ? h('span.badge.accent', 'Overridden') : h('span.badge.outline', { style: { color: 'var(--text-3)' } }, 'Inherited'));
      if (looser) badges.push(h('span.badge.warn', { title: 'The system setting is on; this project turns it off.' }, ic('triangle-alert', 11), 'less strict than system'));
      const resetBtn = !ro && st.changed ? btn('', { icon: 'rotate-ccw', size: 'sm', kind: 'ghost', title: scope === 'project' ? 'Use system value' : 'Reset to default', onClick: () => save(f, null) }) : null;
      const helpBits = [];
      if (f.help) helpBits.push(f.help);
      if (!ro && f.type !== 'json') helpBits.push(scope === 'project' ? `System: ${fmt(f, st.sysV)}` : `Default: ${fmt(f, f.default ?? null)}`);
      if (ro && f.key.endsWith('.binary')) helpBits.push(`Edit settings.json: ${f.key}`);
      const ctl = f.type === 'json' ? jsonRow(f) : control(f, st);
      if (f.type !== 'json') ctl.append(resetBtn || h('span', { style: { width: '30px', flex: 'none' }, 'aria-hidden': 'true' }));
      const r = h(`div.set-row${st.changed ? '.changed' : ''}${ro ? '.ro' : ''}`, { dataset: { key: f.key } },
        h('div', h('div.set-lab', h('span', f.label), infoBtn(fieldSpec(f)), ...badges), h('div.set-help', helpBits.join(' · '))), ctl);
      rowEls.set(f.key, r);
      return r;
    }

    // ------------------------------------------------------------------ custom groups
    function cloudBody() {
      if (!prov) return h('div.set-cloud', h('div.dim', 'The Providers plugin is not available.'));
      const sp = prov.spend || {};
      const sw = S.toggle(prov.allow_cloud, async v => {
        try { await api.put('/api/x/providers/allow', { on: v }); await S.loadModels(); S.toast(v ? 'Cloud models enabled' : 'Cloud models disabled', { kind: v ? 'ok' : 'info', timeout: 1800 }); reload(); } catch (e) { S.toastError(e); sw.input.checked = !v; }
      }, 'Allow cloud');
      const budget = h('input.input', { type: 'number', min: 0, step: 1, value: prov.budget_usd, 'aria-label': 'Monthly budget (USD)' });
      const conf = h('input.input', { type: 'number', min: 0, step: 0.05, value: prov.confirm_usd, 'aria-label': 'Confirm above (USD)' });
      const saveLim = btn('Save limits', { size: 'sm', icon: 'check', onClick: async () => {
        const b = Number(budget.value), c = Number(conf.value);
        if (!(b >= 0) || !(c >= 0)) return S.toast('Limits must be 0 or more.', { kind: 'warn' });
        try { await api.put('/api/x/providers/limits', { budget_usd: b, confirm_usd: c }); S.toast('Limits saved', { kind: 'ok', timeout: 1400 }); reload(); } catch (e) { S.toastError(e); }
      } });
      return h('div.set-cloud',
        h('div.row', { style: { gap: '12px', alignItems: 'flex-start' } }, sw, h('div.grow', h('div.set-lab', h('b', prov.allow_cloud ? 'Cloud models allowed' : 'Cloud models off'), infoBtn(CLOUD_INFO.allow)),
          h('div.set-help', prov.allow_cloud ? 'Jobs on cloud models may cost money and send prompts (and inputs you pick) to the provider.' : 'Everything runs locally. Cloud models are refused until you turn this on.'))),
        h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' } },
          h('div.set-fld', h('span.field-label', 'Monthly budget (USD)', infoBtn(CLOUD_INFO.budget)), budget), h('div.set-fld', h('span.field-label', 'Confirm above (USD)', infoBtn(CLOUD_INFO.confirm)), conf)),
        h('div.set-foot', saveLim, h('span.dim', { style: { fontSize: '12.5px' } }, `This month: $${Number(sp.total_usd || 0).toFixed(2)} spent${sp.pending_usd ? ` + $${Number(sp.pending_usd).toFixed(2)} pending` : ''} of $${Number(sp.budget_usd ?? prov.budget_usd).toFixed(2)}`),
          h('span.spacer'), btn('API keys & endpoints', { size: 'sm', kind: 'ghost', icon: 'cloud', onClick: () => S.navigate('providers') })),
        h('div.set-help', 'API keys never appear on this page. Per-project cloud limits are not available: these limits are shared by all projects.'));
    }
    function envBody() {
      const copy = v => btn('', { icon: 'copy', size: 'sm', kind: 'ghost', title: 'Copy', onClick: () => S.copyText(v) });
      const rows = (schema.env || []).map(e => h('tr', h('td', h('div.set-lab', h('b', e.label), infoBtn(envSpec(e))), h('div.dim.mono', { style: { fontSize: '11px' } }, e.key)),
        h('td', e.value ? h('div.v', e.value) : h('span.dim', 'not set'), e.help ? h('div.set-help', e.help) : null), h('td', e.value ? copy(e.value) : null)));
      const files = schema.files || {};
      const frow = (lab, v) => h('tr', h('td', h('b', lab)), h('td', h('div.v', v)), h('td', copy(v)));
      return h('div', { style: { padding: '0 16px 14px' } },
        h('div.set-note', ic('info', 14), 'Set when starting ./run_studio.sh; restart the studio to change them. Secrets and API keys are never shown.'),
        h('div.tbl-wrap', h('table.tbl.set-env', h('tbody', ...rows, frow('settings.json', files.settings_json || ''), frow('Data folder', files.data || ''), frow('Logs folder', files.logs || ''),
          frow('Project file', files.project_json_template || '')))));
    }

    // ------------------------------------------------------------------ draw
    function groupBanner(g) {
      const kinds = notes.get(g.id);
      if (!kinds || !kinds.size) return null;
      const out = [];
      const mk = (text, action) => h('div.set-banner', ic('info', 15), h('span.grow', text), action,
        btn('', { icon: 'x', size: 'sm', kind: 'ghost', title: 'Dismiss', onClick: e => { e.currentTarget.closest('.set-banner').remove(); } }));
      if (kinds.has('llm_start')) out.push(mk('Takes effect when the prompt LLM next starts.', btn('Restart LLM now', { size: 'sm', icon: 'refresh-cw', onClick: async () => {
        try { await api.llmRelease(); S.toast('Prompt LLM stopped; it starts with the new settings on the next Improve.', { kind: 'ok' }); } catch (e) { S.toastError(e); } } })));
      if (kinds.has('new_worker')) out.push(mk('Applies to workers that start after this change.', btn('Stop idle workers', { size: 'sm', icon: 'square', onClick: async () => {
        try {
          const s = await api.system(); let n = 0;
          for (const w of s.workers || []) if (w.running && !w.busy) { await api.unload({ backend: w.backend }); n++; }
          S.toast(n ? `${n} idle worker(s) stopped; they restart with the new values.` : 'No idle worker is running.', { kind: 'ok' });
        } catch (e) { S.toastError(e); } } })));
      if (kinds.has('restart')) out.push(mk('Restart the studio to apply this (check the queue first, running jobs would be interrupted).', null));
      return out;
    }

    function draw(force = false) {
      if (failed || !schema) {
        const notFound = failed && (failed.status === 404 || failed.status === 405);
        root.replaceChildren(S.empty('settings-2', notFound ? 'Settings need a newer studio core' : 'Settings are unavailable',
          notFound ? 'This studio core does not serve /api/settings/schema yet. Restart the studio, then reload this page.' : (failed?.message || 'Loading…'),
          btn('Try again', { icon: 'refresh-cw', onClick: load })));
        return;
      }
      const act = document.activeElement;
      const keepKey = !force && act && root.contains(act) ? act.closest('[data-key]')?.dataset.key : null;
      const keepEl = keepKey ? rowEls.get(keepKey) : null;
      rowEls.clear();

      const fields = fieldsOf();
      const groups = schema.groups.filter(g => g.scopes.includes(scope));
      const cards = [], counts = {};
      for (const g of groups) {
        const custom = g.id === 'cloud' || g.id === 'env';
        let fs = fields.filter(f => f.group === g.id && matches(f));
        if (changedOnly) fs = fs.filter(f => state(f).changed);
        counts[g.id] = fieldsOf().filter(f => f.group === g.id && state(f).changed).length;
        if (custom) {
          if (changedOnly) continue;
          if (q && !`${g.label} ${g.help}`.toLowerCase().includes(q.toLowerCase())) continue;
        } else if (!fs.length) continue;
        const body = [];
        if (custom) body.push(g.id === 'cloud' ? cloudBody() : envBody());
        else {
          let sub = null;
          for (const f of fs) {
            const sg = f.subgroup ? (g.id === 'models' ? STUDIO_LABEL[f.subgroup] || f.subgroup : f.subgroup) : null;
            if (sg !== sub && sg) body.push(h('div.set-sub', sg));
            sub = sg;
            body.push(row(f));
          }
        }
        cards.push(h('section.card.set-group', { id: `set-g-${g.id}`, dataset: { group: g.id } },
          h('header', h('h3', ic(g.icon || 'settings-2', 16), g.label, infoBtn(groupSpec(g))), g.help ? h('p', g.help) : null), groupBanner(g), body));
      }
      if (keepEl) for (const c of cards) { const n = c.querySelector(`[data-key="${CSS.escape(keepKey)}"]`); if (n) { n.replaceWith(keepEl); rowEls.set(keepKey, keepEl); } }

      const nav = h('nav.set-nav', { 'aria-label': 'Settings groups' }, ...groups.filter(g => cards.some(c => c.dataset.group === g.id)).map(g =>
        h(`a${active === g.id ? '.on' : ''}`, { href: `#/settings/${g.id}`, on: { click: e => { e.preventDefault(); goto(g.id); } } }, ic(g.icon || 'settings-2', 16), h('span.ellipsis', g.label), counts[g.id] ? h('span.n', String(counts[g.id])) : null)));
      const foot = scope === 'project' ? h('div.set-foot',
        h('span.set-note', ic('info', 14), 'Memory, workers, prompt LLM and cloud are system-wide.', h('a', { href: '#/settings?scope=system', on: { click: e => { e.preventDefault(); setScope('system'); } } }, 'Open system settings')),
        h('span.spacer'),
        btn('Reset all project overrides', { size: 'sm', kind: 'danger', icon: 'rotate-ccw', disabled: !proj?.overridden?.length, onClick: resetAll })) : null;
      root.replaceChildren(h('div.set-layout', nav,
        h('div.set-groups', ...(cards.length ? cards : [S.empty('search', 'No setting matches', changedOnly ? 'Nothing is changed in this scope.' : 'Try another word.')]), foot)));
      S.closeOrphanInfo();            // the row of an open balloon was replaced
    }

    async function resetAll() {
      const p = S.store.projects.find(x => x.id === S.store.project);
      if (!(await S.confirmDialog(`Remove every setting override of project “${p?.title || S.store.project}”? It will follow the system settings again.`, { title: 'Reset project overrides', ok: 'Reset all', danger: true }))) return;
      try { await api.resetProjectSettings(S.store.project); await S.refreshEffective(); reload(); S.toast('Project overrides removed', { kind: 'ok', timeout: 1600 }); } catch (e) { S.toastError(e); }
    }

    function drawHead() {
      const p = S.store.projects.find(x => x.id === S.store.project);
      const seg = S.seg([{ value: 'system', label: 'System' }, { value: 'project', label: `Project: ${p?.title || S.store.project}` }], scope, setScope);
      seg.style.maxWidth = '100%';
      const search = h('input', { type: 'search', placeholder: 'Search settings…', value: q, 'aria-label': 'Search settings', on: { input: S.debounce(e => { q = e.target.value.trim(); draw(true); }, 200) } });
      const chg = h('label.row', { style: { gap: '6px', fontSize: '12.5px', color: 'var(--text-2)', whiteSpace: 'nowrap' } }, S.toggle(changedOnly, v => { changedOnly = v; draw(true); }, 'Changed only'), 'Changed only');
      headHost.replaceChildren(h('div.set-head',
        h('div.grow', h('h1.page-title', 'Settings'), h('p.page-sub', scope === 'system' ? 'Studio-wide values, stored in settings.json.' : `Overrides for project “${p?.title || S.store.project}”. They win over the system values; the project is chosen in the top bar.`)),
        h('div.set-tools', seg, h('label.set-search', ic('search', 15), search), chg)));
    }

    function setScope(s) {
      scope = s;
      try { history.replaceState(null, '', `#/settings${active ? '/' + active : ''}${s === 'project' ? '?scope=project' : ''}`); } catch { /* ignore */ }
      drawHead();
      if (s === 'project' && !proj) load(); else draw(true);
    }
    function goto(id) {
      active = id;
      root.querySelectorAll('.set-nav a').forEach(a => a.classList.toggle('on', a.getAttribute('href') === `#/settings/${id}`));
      document.getElementById(`set-g-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ------------------------------------------------------------------ events
    const refresh = () => { if (visible) reload(); };
    S.bus.on('settings', refresh);
    S.bus.on('project', () => { proj = null; drawHead(); if (visible) load(); });
    S.bus.on('models', () => { if (visible && schema) draw(); });
    S.bus.on('providers', refresh);

    drawHead();
    root.append(S.empty('settings-2', 'Loading settings…'));
    await load();
    return {
      show() { visible = true; reload(); },
      hide() { visible = false; closeInfo(); },
      route(r) {
        visible = true;
        const s = r.query?.scope === 'project' ? 'project' : 'system';
        const g = r.sub || null;
        const changed = s !== scope;
        scope = s;
        if (g) active = g;
        if (changed) { drawHead(); if (scope === 'project' && !proj) load(); else draw(true); }
        if (g) setTimeout(() => goto(g), 60);
      },
    };
  },
});
