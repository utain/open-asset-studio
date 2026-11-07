// Text backends (local LLM / API LLMs / subscription CLIs) and the ✍ Write helper.
//   backendPicker({purpose, value, onChange, allowDefault})  -> <select> element (.value, .refresh())
//   writeHelper(kind, {onInsert, input, title, insertLabel})  -> modal: barks | sfx_events | asset_list | asset_meta
// Backends come from GET /api/x/assistant/backends (app/plugins/assistant.py); writing via POST /api/x/assistant/write.
import { api } from '../api.js';
import { store, prefs, bus } from '../store.js';
import { h, ic, btn, modal, toast, toastError } from '../ui.js';

let listP = null, listProj = null;
/** Text backends + the default of each purpose. The defaults are the current project's (its overrides win over settings.json). */
export function textBackends(force = false) {
  if (force || !listP || listProj !== store.project) { listProj = store.project; listP = api.textBackends(true, store.project).catch(e => { listP = null; throw e; }); }
  return listP;
}

/** Options for a <select>: [{value, label, disabled, title, b}] (claude gets one entry per model). */
function optionsOf(list) {
  const out = [];
  for (const b of list.backends) {
    const off = !b.available;
    const why = off ? ` — ${b.reason}` : '';
    if (b.kind === 'cli' && b.models?.length && b.installed) {
      for (const m of b.models) out.push({ value: `${b.id}@${m}`, label: `${b.label} · ${m}${b.tested ? '' : ' (untested)'}${why}`, disabled: off, b, group: 'Subscription CLI' });
    } else {
      out.push({ value: b.id, label: `${b.label}${b.kind === 'api' ? ` · ${b.detail}` : ''}${b.kind === 'cli' && !b.tested ? ' (untested)' : ''}${why}`, disabled: off, b,
        group: b.kind === 'local' ? 'Local' : b.kind === 'cli' ? 'Subscription CLI' : 'API' });
    }
  }
  return out;
}
export function backendLabel(list, id) {
  if (!id) return '';
  const opts = optionsOf(list);
  const o = opts.find(x => x.value === id) || opts.find(x => x.value === id.split('@')[0]) || opts.find(x => x.value.startsWith(id + '@'));
  return o ? o.label.split(' — ')[0] : id;
}
export function isCli(id) { return String(id || '').startsWith('cli:'); }

export function backendPicker({ purpose = 'write', value = '', onChange, allowDefault = true, label = 'Text backend', small = false } = {}) {
  const sel = h(`select.select${small ? '.select-sm' : ''}`, { 'aria-label': label, on: { change: () => { draw(); onChange?.(sel.value); } } });
  const note = h('div.tg-tos.hidden');
  const wrap = h('div.tg-pick', sel, note);
  let list = null;
  const draw = () => {
    const id = sel.value || list?.defaults?.[purpose] || '';
    const tested = optionsOf(list || { backends: [] }).find(o => o.value === id)?.b?.tested;
    note.classList.toggle('hidden', !isCli(id));
    note.replaceChildren(ic('info', 12), ' ', 'Uses your subscription: counts against its usage limits, personal use only.', tested === false ? ' This CLI integration is untested.' : '');
    note.title = list?.tos || '';
  };
  const fill = async (force) => {
    try { list = await textBackends(force); } catch (e) { sel.replaceChildren(h('option', { value: '' }, 'Backends unavailable')); return; }
    const cur = sel.value || value || '';
    const opts = optionsOf(list);
    const def = list.defaults?.[purpose];
    sel.replaceChildren(
      allowDefault ? h('option', { value: '' }, `Default (${backendLabel(list, def) || 'local'})`) : null,
      ...['Local', 'Subscription CLI', 'API'].map(g => {
        const os = opts.filter(o => o.group === g); if (!os.length) return null;
        return h('optgroup', { label: g }, ...os.map(o => h('option', { value: o.value, disabled: o.disabled, title: o.b.reason || o.b.cost || '' }, o.label)));
      }));
    sel.value = cur;
    if (sel.value !== cur && cur) { const alt = opts.find(o => o.value.startsWith(cur + '@')); if (alt) sel.value = alt.value; }
    if (!sel.value && !allowDefault) sel.value = def || 'local';
    draw();
  };
  fill(false);
  wrap.select = sel;
  wrap.refresh = () => fill(true);
  // "Default (...)" names the current project's default: refetch when the project or its settings change (drop the listener once detached)
  let seen = false;
  const off = bus.on('settings', () => { if (wrap.isConnected) { seen = true; fill(true); } else if (seen) off(); });
  Object.defineProperty(wrap, 'value', { get: () => sel.value, set: v => { sel.value = v || ''; draw(); } });
  return wrap;
}

/** "via <backend> · 3.2 s" badge row for generated text. */
export function generatedBy(r) {
  if (!r?.backend) return null;
  const b = r.backend;
  return h('div.tg-by', ic(b.kind === 'cli' ? 'terminal' : b.kind === 'api' ? 'server' : 'cpu', 12),
    h('span', `Written by ${b.label}`), r.seconds != null ? h('span.dim', `· ${r.seconds}s`) : null,
    r.usd ? h('span.dim', `· $${Number(r.usd).toFixed(4)}`) : null,
    b.subscription ? h('span.badge.warn', { title: 'Counts against your subscription plan limits (personal use).' }, 'plan quota') : null,
    b.tested === false ? h('span.badge.outline', 'untested backend') : null);
}

const LANGS = [['en', 'English'], ['th', 'Thai'], ['ja', 'Japanese'], ['zh', 'Chinese'], ['ko', 'Korean'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German']];
const SITUATIONS = ['greeting', 'idle', 'combat_start', 'attack', 'hit', 'low_health', 'victory', 'defeat', 'taunt', 'quest', 'shop'];

const KINDS = {
  barks: { title: 'Write character lines', icon: 'mic', insert: 'Add to line sheet' },
  sfx_events: { title: 'Design doc → SFX event list', icon: 'zap', insert: 'Add to event list' },
  asset_list: { title: 'Design doc → asset list', icon: 'list', insert: 'Use these' },
  asset_meta: { title: 'Name, tag and describe assets', icon: 'tag', insert: 'Apply to assets' },
};

function ta(attrs, value) { const t = h('textarea.textarea', attrs); t.value = value ?? ''; return t; }

/**
 * Open the Write helper. onInsert(items, result) receives the selected rows:
 *   barks -> [{line, emotion, situation, character, language}] · sfx_events -> [{event, prompt, seconds, category}]
 *   asset_list -> [{name, kind, prompt, notes}] · asset_meta -> applied directly (PATCH), onInsert gets the items.
 */
export function writeHelper(kind, { onInsert, input = {}, title, insertLabel, assets = [] } = {}) {
  const K = KINDS[kind];
  const key = `write.${kind}`;
  const saved = prefs.get(key, {});
  const v = { ...saved, ...input };
  const form = h('div.col', { style: { gap: '10px' } });
  const fields = {};
  const field = (name, lab, el) => { fields[name] = el; form.append(h('label.field', h('span.field-label', lab), el)); return el; };
  if (kind === 'barks') {
    field('character', 'Character', h('input.input', { value: v.character || '', placeholder: 'e.g. Brannoc the dwarf blacksmith' }));
    field('description', 'Description', ta({ rows: 3, placeholder: 'Who they are, look, role in the game…' }, v.description));
    field('personality', 'Personality', h('input.input', { value: v.personality || '', placeholder: 'gruff, proud, secretly kind, dry humour' }));
    const chosen = new Set(v.situations || ['greeting', 'idle', 'combat_start', 'hit', 'victory']);
    const chips = h('div.chips', ...SITUATIONS.map(s => h(`button.chip${chosen.has(s) ? '.on' : ''}`, { type: 'button', on: { click: e => { chosen.has(s) ? chosen.delete(s) : chosen.add(s); e.currentTarget.classList.toggle('on'); } } }, s.replace('_', ' '))));
    fields.situations = { get value() { return [...chosen]; } };
    form.append(h('div.field', h('span.field-label', 'Situations'), chips));
    const lang = h('select.select', ...LANGS.map(([c, n]) => h('option', { value: c, selected: (v.language || 'en') === c }, n)));
    const count = h('input.input', { type: 'number', min: 1, max: 40, value: v.count || 8 });
    fields.language = lang; fields.count = count;
    form.append(h('div.row', h('label.field.grow', h('span.field-label', 'Language'), lang), h('label.field', { style: { width: '110px' } }, h('span.field-label', 'Lines'), count)));
  } else if (kind === 'sfx_events' || kind === 'asset_list') {
    field('text', 'Game design text', ta({ rows: 8, placeholder: 'Paste a section of the design doc: setting, rooms, enemies, items, interactions…', style: { fontSize: '13px' } }, v.text));
    const count = h('input.input', { type: 'number', min: 1, max: 60, value: v.count || 12 });
    fields.count = count;
    form.append(h('div.row', h('label.field.grow', h('span.field-label', 'Extra direction (optional)'), fields.context = h('input.input', { value: v.context || '', placeholder: kind === 'asset_list' ? 'low-poly, stylized, mobile game' : 'retro 16-bit feel' })),
      h('label.field', { style: { width: '110px' } }, h('span.field-label', 'About'), count)));
  } else if (kind === 'asset_meta') {
    let ids = assets.map(a => a.id || a);
    const lst = h('div.dim', { style: { fontSize: '12.5px' } });
    const drawIds = () => lst.replaceChildren(ids.length ? `${ids.length} asset(s) selected` : 'No assets selected.');
    drawIds();
    form.append(h('div.row.wrap', lst, btn('Pick assets…', { size: 'sm', icon: 'library', onClick: async () => {
      const { pickAssets } = await import('./assetpicker.js');
      const r = await pickAssets({ multiple: true, title: 'Assets to name and tag' }); if (r?.length) { ids = r; drawIds(); }
    } })));
    fields.asset_ids = { get value() { return ids; } };
    field('context', 'Context (optional)', h('input.input', { value: v.context || '', placeholder: 'project / naming convention, e.g. “Medieval village pack”' }));
  }
  const picker = backendPicker({ purpose: 'write', value: prefs.get('write.backend', '') });
  form.append(h('div.field', h('span.field-label', ic('bot', 13), 'Written by'), picker));
  const out = h('div.col.tg-out', { style: { gap: '8px' } });
  let result = null, sel = [];
  const go = btn('Write', { kind: 'primary', icon: 'pen-line' });
  const ins = btn(insertLabel || K.insert, { kind: 'primary', icon: 'plus', disabled: true });
  const md = modal({ title: title || K.title, icon: K.icon, width: 680, body: h('div.col', { style: { gap: '14px' } }, form, out),
    foot: [btn('Close', { kind: 'ghost', onClick: () => md.close() }), go, ins] });
  const collect = () => { const o = {}; for (const [k, el] of Object.entries(fields)) o[k] = el.value; if (o.count) o.count = Number(o.count); return o; };
  go.addEventListener('click', async () => {
    const inp = collect();
    prefs.set(key, { ...inp, asset_ids: undefined, text: inp.text?.slice(0, 4000) });
    prefs.set('write.backend', picker.value);
    if ((kind === 'sfx_events' || kind === 'asset_list') && !inp.text?.trim()) return toast('Paste some design text first.', { kind: 'warn' });
    if (kind === 'barks' && !inp.character?.trim() && !inp.description?.trim()) return toast('Describe the character first.', { kind: 'warn' });
    if (kind === 'asset_meta' && !inp.asset_ids?.length) return toast('Pick assets first.', { kind: 'warn' });
    go.disabled = true; go.replaceChildren(ic('loader', 15, 'spin'), h('span', 'Writing…'));
    out.replaceChildren(h('div.dim', ic('loader', 14, 'spin'), ' Writing… (CLI and cloud backends take 5-30 s)'));
    try {
      result = await api.write({ kind, project: store.project, backend: picker.value || undefined, input: inp });
      render(inp);
    } catch (e) { out.replaceChildren(h('div', { style: { color: 'var(--err)' } }, e.message, e.fix ? h('div.dim', e.fix) : null)); }
    finally { go.disabled = false; go.replaceChildren(ic('pen-line', 15), h('span', result ? 'Write again' : 'Write')); }
  });
  const render = (inp) => {
    const d = result.data || {};
    const rows = d.lines || d.events || d.assets || d.items || [];
    sel = rows.map(() => true);
    const list = h('div.tg-rows');
    rows.forEach((r, i) => {
      const cb = h('input', { type: 'checkbox', checked: true, 'aria-label': 'Include', on: { change: e => { sel[i] = e.target.checked; ins.disabled = !sel.some(Boolean); } } });
      let body;
      if (kind === 'barks') body = [h('div', `“${r.line}”`), h('div.dim', [r.situation, r.emotion].filter(Boolean).join(' · '))];
      else if (kind === 'sfx_events') body = [h('div.row', h('b.mono', r.event), h('span.badge.outline', r.category), h('span.dim', `${r.seconds}s`)), h('div.dim', r.prompt)];
      else if (kind === 'asset_list') body = [h('div.row', h('b.mono', r.name), h('span.badge.accent', r.kind)), h('div.dim', r.prompt), r.notes ? h('div.dim', { style: { fontStyle: 'italic' } }, r.notes) : null];
      else body = [h('div.row', h('b', r.name), h('span.dim.mono', r.id)), h('div.chips', ...(r.tags || []).map(t => h('span.chip', t))), h('div.dim', r.description)];
      list.append(h('label.tg-row', cb, h('div.grow', ...body)));
    });
    out.replaceChildren(generatedBy(result), rows.length ? list : h('div.dim', 'Nothing came back.'), d.notes ? h('div.dim', ic('info', 12), ' ', d.notes) : null,
      result.warnings?.length ? h('div', { style: { color: 'var(--warn)', fontSize: '12px' } }, result.warnings.join('; ')) : null);
    ins.disabled = !rows.length;
    ins.onclick = async () => {
      const picked = rows.filter((_, i) => sel[i]);
      if (kind === 'barks') picked.forEach(r => { r.character = inp.character; r.language = inp.language; });
      if (kind === 'asset_meta') {
        let n = 0;
        for (const r of picked) { try { await api.patchAsset(r.id, { name: r.name, tags: r.tags, notes: r.description }); n++; } catch (e) { toastError(e); } }
        toast(`${n} asset(s) updated.`, { kind: 'ok' });
      }
      onInsert?.(picked, result);
      md.close();
    };
  };
  return md;
}

export const WRITER_CSS = `
.tg-pick { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.tg-pick .select { width: 100%; min-width: 0; }
.select.select-sm { height: 28px; font-size: 12.5px; padding-top: 2px; padding-bottom: 2px; }
.tg-tos { font-size: 11.5px; color: var(--warn); line-height: 1.4; }
.tg-by { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: 11.5px; color: var(--text-2); min-width: 0; }
.tg-rows { display: flex; flex-direction: column; gap: 6px; max-height: 46vh; overflow: auto; }
.tg-row { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; border: 1px solid var(--line-soft); border-radius: 9px; background: var(--panel-2); font-size: 13px; cursor: pointer; min-width: 0; }
.tg-row > input { margin-top: 3px; }
.tg-row .dim { font-size: 12px; overflow-wrap: anywhere; }
`;
if (typeof document !== 'undefined' && !document.getElementById('tg-css')) { const s = document.createElement('style'); s.id = 'tg-css'; s.textContent = WRITER_CSS; document.head.append(s); }
