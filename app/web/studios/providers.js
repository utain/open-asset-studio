// Providers: cloud API keys (stored server-side in data/secrets.json, never sent back), the global "Allow cloud"
// switch, monthly budget + confirm threshold, spend this month, user OpenAI-compatible endpoints, price overrides,
// upload consent per provider, and detected subscription CLIs (display only).
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const CSS = `
.prov-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap: 12px; }
.prov-card { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.prov-card .ph { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.prov-card .ph b { font-size: 15px; }
.prov-card .keyline { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 13px; }
.prov-card .acts { display: flex; gap: 6px; flex-wrap: wrap; }
.prov-card .links { display: flex; gap: 12px; flex-wrap: wrap; font-size: 12.5px; }
.prov-card .terms { font-size: 12px; color: var(--text-3); }
.prov-test { font-size: 12.5px; padding: 8px 10px; border-radius: 9px; background: var(--panel-2); overflow-wrap: anywhere; }
.prov-test.ok { border-left: 3px solid var(--ok); } .prov-test.bad { border-left: 3px solid var(--err); }
.prov-top { display: flex; gap: 12px; flex-wrap: wrap; align-items: stretch; }
.prov-top > .card { flex: 1 1 300px; min-width: 0; }
.prov-switch { display: flex; align-items: center; gap: 12px; }
.prov-switch .big { font-size: 16px; font-weight: 650; }
.budget-bar { height: 10px; border-radius: 6px; background: var(--panel-2); overflow: hidden; }
.budget-bar i { display: block; height: 100%; background: var(--accent); }
.budget-bar i.over { background: var(--err); } .budget-bar i.near { background: var(--warn); }
.prov-limits { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; align-items: end; }
.prov-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 12px; }
.prov-form { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: 10px; }
.prov-sec h3 { display: flex; gap: 8px; align-items: center; margin: 0; font-size: 15px; }
.prov-tbl td, .prov-tbl th { vertical-align: top; }
.prov-tbl .pr { font-variant-numeric: tabular-nums; white-space: nowrap; }
.prov-off { padding: 10px 12px; border-radius: 10px; background: color-mix(in srgb, var(--warn) 12%, transparent); font-size: 13px; display: flex; gap: 8px; }
`;

export default S.defineStudio({
  id: 'providers', title: 'Providers', icon: 'cloud', order: 85, section: 'system',
  async mount(el) {
    document.head.append(h('style', CSS));
    let D = null;            // last /api/x/providers answer
    const tests = new Map(); // provider id -> test result
    let clis = null;
    const root = h('div.page-pad.col', { style: { gap: '16px', maxWidth: '1500px' } });
    el.append(root);
    const api = S.api;

    async function load() {
      try { D = await api.get('/api/x/providers'); render(); } catch (e) { root.replaceChildren(S.empty('cloud', 'Providers unavailable', e.message)); }
    }
    async function loadClis(refresh = false) {
      try { clis = (await api.get('/api/x/providers/clis', refresh ? { refresh: true } : undefined)).clis; render(); } catch (e) { S.toastError(e); }
    }
    S.bus.on('providers', () => load());
    S.bus.on('spend', () => load());

    // ---------------------------------------------------------------- top: switch + budget
    function topCards() {
      const sw = S.toggle(D.allow_cloud, async v => {
        try { await api.put('/api/x/providers/allow', { on: v }); await S.loadModels(); S.toast(v ? 'Cloud models enabled' : 'Cloud models disabled', { kind: v ? 'ok' : 'info', timeout: 2000 }); load(); }
        catch (e) { S.toastError(e); sw.input.checked = !v; }
      }, 'Allow cloud');
      const sp = D.spend;
      const frac = sp.budget_usd ? Math.min(1, (sp.total_usd + sp.pending_usd) / sp.budget_usd) : 0;
      const budget = h('input.input', { type: 'number', min: 0, step: 1, value: D.budget_usd, 'aria-label': 'Monthly budget (USD)' });
      const confirm = h('input.input', { type: 'number', min: 0, step: 0.05, value: D.confirm_usd, 'aria-label': 'Confirm above (USD)' });
      const save = btn('Save', { size: 'sm', icon: 'check', onClick: async () => {
        try { await api.put('/api/x/providers/limits', { budget_usd: Number(budget.value), confirm_usd: Number(confirm.value) }); S.toast('Limits saved', { kind: 'ok', timeout: 1600 }); load(); } catch (e) { S.toastError(e); }
      } });
      return h('div.prov-top',
        h('section.card.card-pad.col', { style: { gap: '10px' } },
          h('div.prov-switch', sw, h('div.grow', h('div.big', D.allow_cloud ? 'Cloud models allowed' : 'Cloud models off', S.uiInfo('cloud_allow')),
            h('div.dim', { style: { fontSize: '12.5px' } }, D.allow_cloud ? 'Jobs on cloud models may cost money and send prompts (and inputs you pick) to the provider.' : 'Everything runs locally. Cloud models show “Cloud off” and their jobs are refused.'))),
          D.mock ? h('div.prov-off', ic('flask-conical', 14), h('span', 'Test mode: STUDIO_CLOUD_MOCK points every provider at a local mock server.')) : null),
        h('section.card.card-pad.col', { style: { gap: '10px' } },
          h('div.row', h('div.stat.grow', h('b', S.fmtUsd(sp.total_usd)), h('span', `spent in ${sp.month}${sp.pending_usd ? ` · ${S.fmtUsd(sp.pending_usd)} running` : ''}${sp.estimated_count ? ` · ${sp.estimated_count} estimated` : ''}`)),
            h('div.stat', { style: { textAlign: 'right' } }, h('b', sp.budget_usd ? S.fmtUsd(sp.budget_usd) : '∞'), h('span', 'monthly budget'))),
          h('div.budget-bar', h(`i${frac >= 1 ? '.over' : frac > 0.8 ? '.near' : ''}`, { style: { width: `${frac * 100}%` } })),
          h('div.prov-limits',
            h('div.field', h('span.field-label', 'Monthly budget ($, 0 = none)', S.uiInfo('cloud_budget')), budget),
            h('div.field', h('span.field-label', 'Ask to confirm above ($)', S.uiInfo('cloud_confirm')), confirm), save)));
    }

    // ---------------------------------------------------------------- provider cards
    function keyDialog(p) {
      const inp = h('input.input', { type: 'password', autocomplete: 'off', spellcheck: false, placeholder: p.key.env_names?.[0] ? `paste the key (or set ${p.key.env_names[0]})` : 'paste the key' });
      const md = S.modal({ title: `${p.key.set ? 'Replace' : 'Add'} ${p.label} key`, icon: 'key', width: 460,
        body: h('div.col', { style: { gap: '10px' } }, h('div.field-label', 'API key', S.uiInfo('api_key')), inp,
          h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, 'Stored in data/secrets.json (mode 600) on this machine. The studio never sends it back to a browser; it is only used for calls to ', h('b', p.label), '.'),
          p.keys_url ? h('a', { href: p.keys_url, target: '_blank', rel: 'noopener' }, ic('external-link', 12), ' Get a key') : null),
        foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn('Save key', { kind: 'primary', icon: 'check', onClick: save })] });
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') save(); });
      setTimeout(() => inp.focus(), 30);
      async function save() {
        const k = inp.value.trim(); if (!k) return;
        try { await api.put(`/api/x/providers/keys/${encodeURIComponent(p.id)}`, { key: k }); inp.value = ''; md.close(); S.toast(`${p.label} key saved`, { kind: 'ok', timeout: 1800 }); tests.delete(p.id); await S.loadModels(); load(); }
        catch (e) { S.toastError(e); }
      }
    }
    async function test(p) {
      tests.set(p.id, { pending: true }); render();
      try { tests.set(p.id, await api.post(`/api/x/providers/test/${encodeURIComponent(p.id)}`)); } catch (e) { tests.set(p.id, { ok: false, message: e.message }); }
      render();
    }
    function testBox(pid) {
      const t = tests.get(pid); if (!t) return null;
      if (t.pending) return h('div.prov-test', ic('loader', 13, 'spin'), ' Testing…');
      const det = Object.entries(t.details || {}).map(([k, v]) => `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(', ') || '–' : v}`);
      return h(`div.prov-test.${t.ok ? 'ok' : 'bad'}`, h('b', t.message), det.length ? h('div.dim', { style: { marginTop: '3px' } }, det.join(' · ')) : null);
    }
    function keyLine(k) {
      if (!k.set) return h('div.keyline', h('span.badge.warn', ic('key', 12), 'no key'), k.env_names?.length ? h('span.dim', `or set ${k.env_names.join(' / ')}`) : null);
      return h('div.keyline', h('span.badge.ok', ic('key', 12), `set${k.last4 ? ` (…${k.last4})` : ''}`), h('span.dim', k.source === 'env' ? `from env ${k.env}` : 'saved on this machine'));
    }
    function providerCard(p) {
      return h('div.card.card-pad.prov-card',
        h('div.ph', h('b.grow', p.label), ...p.modalities.map(m => h('span.badge.outline', m === 'mesh' ? '3D' : m)), h('span.dim', { style: { fontSize: '12px' } }, `${p.models} models`)),
        keyLine(p.key),
        h('div.acts',
          btn(p.key.set && p.key.source === 'file' ? 'Replace key' : 'Add key', { size: 'sm', icon: 'key', onClick: () => keyDialog(p) }),
          p.key.source === 'file' ? btn('Delete', { size: 'sm', kind: 'ghost', icon: 'trash-2', onClick: async () => {
            if (!(await S.confirmDialog(`Delete the saved ${p.label} key?`, { ok: 'Delete', danger: true, title: 'Delete key' }))) return;
            try { await api.del(`/api/x/providers/keys/${encodeURIComponent(p.id)}`); tests.delete(p.id); await S.loadModels(); load(); } catch (e) { S.toastError(e); }
          } }) : null,
          btn('Test', { size: 'sm', icon: 'flask-conical', title: p.test, disabled: !p.key.set, onClick: () => test(p) }), S.uiInfo('provider_test')),
        testBox(p.id),
        p.upload_ack ? h('div.keyline', h('span.badge.outline', ic('upload', 12), 'uploads allowed'), h('span.dim', S.fmtAgo(p.upload_ack)),
          btn('Revoke', { size: 'sm', kind: 'ghost', onClick: async () => { await api.put(`/api/x/providers/upload-ack/${p.id}`, { on: false }).catch(S.toastError); load(); } })) : null,
        h('div.terms', p.terms),
        h('div.links', h('a', { href: p.keys_url, target: '_blank', rel: 'noopener' }, 'API keys'), h('a', { href: p.pricing_url, target: '_blank', rel: 'noopener' }, 'Pricing'),
          h('a', { href: p.terms_url, target: '_blank', rel: 'noopener' }, 'Terms'),
          p.base_overridden ? h('span.dim.mono', { style: { fontSize: '11px', overflowWrap: 'anywhere' } }, p.base_url) : null));
    }

    // ---------------------------------------------------------------- user endpoints
    function endpointForm(ep = null) {
      const f = {
        name: h('input.input', { value: ep?.name || '', placeholder: 'lab', disabled: !!ep, 'aria-label': 'Name' }),
        label: h('input.input', { value: ep?.label || '', placeholder: 'Lab llama-server', 'aria-label': 'Label' }),
        base: h('input.input', { value: ep?.base_url || '', placeholder: 'http://127.0.0.1:7880/v1', 'aria-label': 'Base URL' }),
        key: h('input.input', { type: 'password', autocomplete: 'off', placeholder: ep?.key?.set ? 'key set: leave empty to keep' : 'optional', 'aria-label': 'API key' }),
        models: h('input.input', { value: (ep?.models || []).join(', '), placeholder: 'model ids, comma separated', 'aria-label': 'Models' }),
        price: h('input.input', { type: 'number', min: 0, step: 0.001, value: ep?.price_per_image ?? '', placeholder: '0', 'aria-label': 'Price per image' }),
      };
      const mods = new Set(ep?.modalities || ['llm']);
      const modBox = h('div.row', ...['llm', 'image'].map(m => h('label.row', { style: { gap: '6px', fontSize: '13px' } },
        h('input', { type: 'checkbox', checked: mods.has(m), on: { change: e => { e.target.checked ? mods.add(m) : mods.delete(m); } } }), m === 'llm' ? 'Text (LLM)' : 'Images')));
      const fetchBtn = btn('Fetch models', { size: 'sm', icon: 'refresh-cw', title: 'Save, then read GET /models from the endpoint', onClick: async () => {
        if (!(await saveEp(false))) return;
        try { const r = await api.post(`/api/x/providers/endpoints/${encodeURIComponent(f.name.value.trim().toLowerCase())}/models`); f.models.value = r.models.join(', '); S.toast(`${r.models.length} model(s) found: pick the ones to keep and Save`, { kind: 'ok' }); }
        catch (e) { S.toastError(e, 'Cannot list models'); }
      } });
      async function saveEp(close = true) {
        try {
          await api.put('/api/x/providers/endpoints', { name: f.name.value.trim().toLowerCase(), label: f.label.value.trim(), base_url: f.base.value.trim(), key: f.key.value.trim() || null,
            models: f.models.value.split(/[,\n]/).map(s => s.trim()).filter(Boolean), modalities: [...mods], price_per_image: f.price.value === '' ? null : Number(f.price.value) });
          f.key.value = '';
          if (close) { md.close(); S.toast('Endpoint saved', { kind: 'ok', timeout: 1600 }); }
          await S.loadModels(); load(); return true;
        } catch (e) { S.toastError(e, 'Cannot save the endpoint'); return false; }
      }
      const fld = (l, x, help, info) => h('div.field', h('span.field-label', l, info ? S.uiInfo(info) : null), x, help ? h('span.field-help', help) : null);
      const md = S.modal({ title: ep ? `Edit ${ep.label || ep.name}` : 'Add an OpenAI-compatible endpoint', icon: 'plug', width: 620,
        body: h('div.col', { style: { gap: '12px' } },
          h('div.prov-form', fld('Name (id)', f.name, 'provider id: openai_compat:<name>', 'endpoint_name'), fld('Label', f.label, null, 'endpoint_label'), fld('Base URL', f.base, 'ends with /v1 (…/chat/completions, …/images/generations, …/models)', 'endpoint_base'),
            fld('API key', f.key, 'optional; stored like the other keys', 'endpoint_key'), fld('Price per image ($)', f.price, 'for image endpoints; 0 = free', 'endpoint_price')),
          fld('Models', f.models, null, 'endpoint_models'), h('div.field', h('span.field-label', 'Serves', S.uiInfo('endpoint_serves')), modBox),
          h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, 'Local / LAN endpoints (127.0.0.1, 192.168.x, *.local) work even with cloud off and cost nothing. Text models appear in the prompt improver / assistant; image models in Image › Generate.')),
        foot: [fetchBtn, h('span.spacer'), btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn('Save', { kind: 'primary', icon: 'check', onClick: () => saveEp(true) })] });
    }
    function endpointsSec() {
      const eps = D.endpoints || [];
      return h('section.card.card-pad.col.prov-sec', { style: { gap: '12px' } },
        h('div.section-head', { style: { margin: 0, flexWrap: 'wrap' } }, h('h3.grow', ic('plug', 16), 'Your OpenAI-compatible endpoints'), btn('Add endpoint', { size: 'sm', icon: 'plus', onClick: () => endpointForm() })),
        eps.length ? h('div.prov-grid', ...eps.map(ep => h('div.card.card-pad.prov-card',
          h('div.ph', h('b.grow', ep.label || ep.name), ...ep.modalities.map(m => h('span.badge.outline', m)), ep.local ? h('span.badge.ok', 'local') : h('span.badge.cloud', ic('cloud', 12), 'remote')),
          h('div.mono.dim', { style: { fontSize: '12px', overflowWrap: 'anywhere' } }, ep.base_url),
          keyLine({ ...ep.key, env_names: [] }),
          h('div.dim', { style: { fontSize: '12.5px' } }, ep.models.length ? `${ep.models.length} model(s): ${ep.models.slice(0, 6).join(', ')}${ep.models.length > 6 ? ' …' : ''}` : 'No models listed yet: Edit → Fetch models.'),
          h('div.acts', btn('Test', { size: 'sm', icon: 'flask-conical', onClick: () => test({ id: ep.id }) }), btn('Edit', { size: 'sm', icon: 'pencil', onClick: () => endpointForm(ep) }),
            btn('Delete', { size: 'sm', kind: 'ghost', icon: 'trash-2', onClick: async () => {
              if (!(await S.confirmDialog(`Delete the endpoint “${ep.label || ep.name}” and its key?`, { ok: 'Delete', danger: true }))) return;
              try { await api.del(`/api/x/providers/endpoints/${encodeURIComponent(ep.name)}`); await S.loadModels(); load(); } catch (e) { S.toastError(e); }
            } })),
          testBox(ep.id)))) : h('div.dim', { style: { fontSize: '13px' } }, 'None yet. Add a llama-server / vLLM / LM Studio / LiteLLM URL to use its models for text (and images where it supports /images/generations).'));
    }

    // ---------------------------------------------------------------- prices
    function priceDialog(m) {
      const ta = h('textarea.textarea.mono', { rows: 6, value: JSON.stringify(m.pricing || {}, null, 1), spellcheck: false });
      const md = S.modal({ title: `Price of ${m.name}`, icon: 'circle-dollar-sign', width: 540,
        body: h('div.col', { style: { gap: '10px' } }, h('div.field-label', 'Price (JSON)', S.uiInfo('price_override')),
          h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, `List price checked ${m.pricing_checked || '?'}`, m.pricing_source ? [' · ', h('a', { href: m.pricing_source, target: '_blank', rel: 'noopener' }, 'source')] : null,
            '. Keys: per_image, per_megapixel, first_megapixel, per_request, per_second, per_minute, per_1k_chars, credits + per_credit, input_per_mtok / output_per_mtok, by {param: {value: {...}}}, add {param: {value: usd}}.'),
          ta, m.pricing_overridden ? h('div.dim', { style: { fontSize: '12px' } }, 'Default: ', h('code', JSON.stringify(m.pricing_default))) : null),
        foot: [m.pricing_overridden ? btn('Reset to list price', { kind: 'ghost', onClick: async () => { try { await api.del(`/api/x/providers/prices/${encodeURIComponent(m.id)}`); md.close(); await S.loadModels(); load(); } catch (e) { S.toastError(e); } } }) : null,
          h('span.spacer'), btn('Cancel', { kind: 'ghost', onClick: () => md.close() }),
          btn('Save', { kind: 'primary', icon: 'check', onClick: async () => {
            let pr; try { pr = JSON.parse(ta.value); } catch { S.toast('Not valid JSON', { kind: 'err' }); return; }
            try { await api.put(`/api/x/providers/prices/${encodeURIComponent(m.id)}`, { pricing: pr }); md.close(); S.toast('Price override saved', { kind: 'ok', timeout: 1600 }); await S.loadModels(); load(); } catch (e) { S.toastError(e); }
          } })] });
    }
    let priceFilter = S.prefs.get('providers.pf', '');
    function pricesSec() {
      const ms = (D.models || []).filter(m => !priceFilter || m.provider === priceFilter || m.modality === priceFilter);
      const sel = h('select.select', { style: { width: 'auto' }, 'aria-label': 'Filter', on: { change: e => { priceFilter = e.target.value; S.prefs.set('providers.pf', priceFilter); render(); } } },
        h('option', { value: '' }, 'All'), ...['image', 'mesh', 'audio', 'text'].map(v => h('option', { value: v, selected: priceFilter === v }, v === 'mesh' ? '3D' : v)),
        ...D.providers.map(p => h('option', { value: p.id, selected: priceFilter === p.id }, p.label)));
      const rows = ms.map(m => {
        const sm = S.store.modelsById.get(m.id) || { pricing: m.pricing, modality: m.modality, cloud: {} };
        return h('tr', h('td', h('b', m.name), h('div.dim.mono', { style: { fontSize: '11px' } }, typeof m.remote_model === 'string' ? m.remote_model : Object.values(m.remote_model || {})[0] || m.id),
          m.unverified ? h('div', { style: { fontSize: '11px', color: 'var(--warn)' }, title: m.unverified }, ic('info', 11), ' unverified') : null),
          h('td', m.provider_label), h('td', S.statusBadge(m.status)),
          h('td.pr', S.priceLabel(sm) || '–', m.pricing_overridden ? h('div', h('span.badge.accent', 'override')) : null),
          h('td.pr', m.example_usd != null ? S.fmtUsd(m.example_usd) : '–'),
          h('td.dim', { style: { fontSize: '12px' } }, m.pricing_checked || '–', m.pricing_source ? h('div', h('a', { href: m.pricing_source, target: '_blank', rel: 'noopener' }, 'source')) : null),
          h('td', m.license?.commercial ? h('span.badge.ok', 'commercial') : h('span.badge.err', 'non-comm.')),
          h('td', btn('', { size: 'sm', icon: 'pencil', title: 'Override the price', onClick: () => priceDialog(m) })));
      });
      return h('section.card.card-pad.col.prov-sec', { style: { gap: '12px' } },
        h('div.section-head', { style: { margin: 0, flexWrap: 'wrap' } }, h('h3.grow', ic('circle-dollar-sign', 16), `Cloud models and prices (${ms.length})`), sel),
        h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, 'List prices from the providers\' pages on the date shown; they change. Estimates use these numbers; jobs record the actual cost when the API reports it (OpenAI usage, OpenRouter, Meshy / Tripo credits, Replicate compute time). “Example” = one job with the default settings.'),
        h('div.tbl-wrap', h('table.tbl.prov-tbl', h('thead', h('tr', ...['Model', 'Provider', 'Status', 'Price', 'Example', 'Checked', 'Use', ''].map(x => h('th', x)))), h('tbody', ...rows))));
    }

    // ---------------------------------------------------------------- spend
    function spendSec() {
      const sp = D.spend;
      const tbl = (title, rows, label = r => r.key) => h('div.col', { style: { gap: '6px', minWidth: 0 } }, h('div.field-label', title),
        rows.length ? h('div.tbl-wrap', h('table.tbl', h('tbody', ...rows.slice(0, 12).map(r => h('tr', h('td.ellipsis', { style: { maxWidth: '220px' } }, label(r)), h('td.pr', S.fmtUsd(r.usd)), h('td.dim', `${r.count}×`)))))) : h('div.dim', { style: { fontSize: '12.5px' } }, 'Nothing yet.'));
      return h('section.card.card-pad.col.prov-sec', { style: { gap: '12px' } },
        h('div.section-head', { style: { margin: 0 } }, h('h3.grow', ic('wallet', 16), `Spend · ${sp.month}`), h('span.pr', h('b', S.fmtUsd(sp.total_usd)))),
        h('div.prov-cols', tbl('By provider', sp.by_provider, r => r.label || r.key), tbl('By model', sp.by_model, r => S.store.modelsById.get(r.key)?.name || r.key), tbl('By project', sp.by_project)),
        sp.recent.length ? h('details', h('summary', { style: { cursor: 'pointer', fontSize: '13px', color: 'var(--text-2)' } }, `Recent calls (${sp.recent.length})`),
          h('div.tbl-wrap', { style: { marginTop: '8px' } }, h('table.tbl', h('thead', h('tr', ...['When', 'Provider', 'Model', 'Project', 'Cost'].map(x => h('th', x)))),
            h('tbody', ...sp.recent.map(e => h('tr', h('td.nowrap.dim', e.ts.replace('T', ' ')), h('td', e.provider), h('td.ellipsis', { style: { maxWidth: '200px' } }, S.store.modelsById.get(e.model)?.name || e.model || '–'),
              h('td', e.project || '–'), h('td.pr', S.fmtUsd(e.usd), e.estimated ? h('span.dim', ' est.') : null))))))) : null,
        sp.months.length > 1 ? h('div.dim', { style: { fontSize: '12.5px' } }, 'Earlier: ', sp.months.slice(1).map(m => `${m.month} ${S.fmtUsd(m.usd)}`).join(' · ')) : null);
    }

    // ---------------------------------------------------------------- subscription CLIs
    function clisSec() {
      const body = !clis ? h('div.dim', { style: { fontSize: '13px' } }, 'Detecting…') : h('div.tbl-wrap', h('table.tbl.prov-tbl',
        h('thead', h('tr', ...['CLI', 'Installed', 'Version', 'Logged in', 'Install / login'].map(x => h('th', x)))),
        h('tbody', ...clis.map(c => h('tr', h('td', h('b', c.label), h('div.dim.mono', { style: { fontSize: '11px' } }, c.bin || c.id)),
          h('td', c.installed ? h('span.badge.ok', 'yes') : h('span.badge.outline', 'no')),
          h('td.mono', { style: { fontSize: '12px' } }, c.version || '–'),
          h('td', c.logged_in === true ? h('span.badge.ok', 'yes') : c.logged_in === false ? h('span.badge.warn', 'no') : h('span.dim', '?'), c.auth_note ? h('div.dim', { style: { fontSize: '11.5px' } }, c.auth_note) : null),
          h('td', h('div.row', { style: { gap: '4px', flexWrap: 'nowrap' } }, h('code.mono', { style: { fontSize: '11.5px', overflowWrap: 'anywhere' } }, c.installed ? c.login : c.install),
            btn('', { size: 'sm', kind: 'ghost', icon: 'copy', title: 'Copy', onClick: () => S.copyText(c.installed ? c.login : c.install) })),
            h('a', { href: c.docs, target: '_blank', rel: 'noopener', style: { fontSize: '12px' } }, 'docs')))))));
      return h('section.card.card-pad.col.prov-sec', { style: { gap: '12px' } },
        h('div.section-head', { style: { margin: 0, flexWrap: 'wrap' } }, h('h3.grow', ic('terminal', 16), 'Subscription CLIs'), btn('Re-detect', { size: 'sm', icon: 'refresh-cw', onClick: () => { clis = null; render(); loadClis(true); } })),
        h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, 'Coding-agent CLIs that run on your own subscription. The assistant can use them as text backends; the login check never spends quota (no prompts are sent).'),
        body);
    }

    function render() {
      if (!D) return;
      root.replaceChildren(
        h('div.section-head', { style: { margin: 0, flexWrap: 'wrap' } }, h('div.grow', h('h1.page-title', 'Providers'),
          h('p.page-sub', 'Cloud APIs and your own endpoints: keys, cost limits, prices and terms. Local models never need any of this.'))),
        topCards(),
        !D.allow_cloud ? h('div.prov-off', ic('info', 14), h('span', 'Cloud is off: keys can be added and tested, but cloud jobs are refused until you switch it on.')) : null,
        h('section.col.prov-sec', { style: { gap: '10px' } }, h('h3', ic('key', 16), 'API providers'), h('div.prov-grid', ...D.providers.map(providerCard))),
        endpointsSec(), spendSec(), pricesSec(), clisSec());
    }
    await load();
    loadClis();
    return { show: () => load() };
  },
});
