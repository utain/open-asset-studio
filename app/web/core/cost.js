// Cloud cost helpers: the price estimate shown on Generate buttons (same formula as app/providers.py
// estimate_cost), price labels / cloud badges for model pickers, and the confirm / upload-consent dialogs that
// api.submit runs when the core refuses a job with kind "confirm" or "consent".
import { setSubmitGuard, api } from './api.js';
import { store } from './store.js';
import { h, ic, btn, modal } from './ui.js';
import { navigate } from './studio.js';

export const isCloud = m => !!(m && m.cloud);

function mergeBy(pr, v) {
  const p = { ...pr }; delete p.by;
  for (const [name, table] of Object.entries(pr.by || {})) {
    const key = typeof v[name] === 'boolean' ? String(v[name]) : String(v[name]);
    for (const [k, sub] of Object.entries(table || {})) if (String(k).toLowerCase() === key.toLowerCase() && sub && typeof sub === 'object') Object.assign(p, sub);
  }
  return p;
}
function megapixels(v) {
  const MP = 1024 * 1024; const size = v.size;
  const res = String(v.resolution || '').toUpperCase().replace(/ /g, '');
  if (typeof size === 'string' && /x/i.test(size)) { const [w, hh] = size.toLowerCase().split('x').map(Number); if (w && hh) return (w * hh) / MP; }
  if (size != null && size !== '' && size !== 'auto' && !isNaN(Number(size))) {
    const [a, b] = String(v.aspect || '1:1').split(':').map(Number); const L = Number(size);
    if (a && b) return (L * L * Math.min(a, b) / Math.max(a, b)) / MP;
  }
  const m = res.match(/^(\d+(?:\.\d+)?)MP$/); if (m) return Number(m[1]);
  return { '0.5K': 0.25, 512: 0.25, '1K': 1, '2K': 4, '4K': 16 }[res] ?? 1;
}

/** USD estimate for one job (null = unknown). values = the form values (+ prompt / text). */
export function estimateCost(model, values = {}, nInputs = 0) {
  const pr = model?.pricing; if (!pr) return null;
  const p = mergeBy(pr, values);
  const n = Math.max(1, Number(values.batch || values.num_images || 1));
  let usd = 0, known = false;
  const has = k => p[k] != null;
  if (has('per_image')) { usd += p.per_image * n; known = true; }
  if (has('per_megapixel') || has('first_megapixel')) {
    const mp = Math.max(1, Math.ceil(megapixels(values) - 1e-6)); const first = p.first_megapixel ?? p.per_megapixel;
    usd += n * (first + (mp - 1) * (p.per_megapixel ?? first)); known = true;
  }
  if (has('per_input_image')) { usd += p.per_input_image * nInputs; known = true; }
  if (has('per_input_megapixel')) { usd += p.per_input_megapixel * nInputs; known = true; }
  if (has('per_request')) { usd += p.per_request * n; known = true; }
  if (has('per_second') || has('per_minute')) {
    const s = Number(values.seconds ?? values.duration ?? p.default_seconds ?? 10);
    usd += n * s * ((p.per_second || 0) + (p.per_minute || 0) / 60); known = true;
  }
  if (has('per_1k_chars')) { const t = String(values.text || values.prompt || ''); usd += n * Math.max(t.length, 1) / 1000 * p.per_1k_chars; known = true; }
  if (has('credits')) { usd += n * p.credits * (p.per_credit ?? 0.01); known = true; }
  if (has('input_per_mtok') || has('output_per_mtok')) {
    const tin = String(values.prompt || values.text || '').length / 3.5 + 200; const tout = Number(values.max_tokens || 800);
    usd += (tin * (p.input_per_mtok || 0) + tout * (p.output_per_mtok || 0)) / 1e6; known = true;
  }
  for (const [name, table] of Object.entries(p.add || {})) {
    const key = String(values[name]);
    for (const [k, add] of Object.entries(table || {})) if (String(k).toLowerCase() === key.toLowerCase()) { usd += Number(add) * n; known = true; }
  }
  if (!known) return null;
  return Math.round(Math.max(usd, p.min_usd || 0) * 10000) / 10000;
}

export function fmtUsd(v) {
  if (v == null || isNaN(v)) return '$?';
  if (v === 0) return '$0';
  if (v < 0.01) return `$${v.toFixed(4).replace(/0+$/, '')}`;
  return v < 0.1 ? `$${v.toFixed(3)}` : `$${v.toFixed(2)}`;
}

/** Short price description of a cloud model: "$0.04 / image", "$0.30 / model", "30 credits (~$0.60)". */
export function priceLabel(m) {
  const p = m?.pricing; if (!p) return m?.cloud?.local ? 'free (local)' : '';
  const u = { image: 'image', mesh: 'model', audio: 'clip', text: 'call' }[m.modality] || 'job';
  if (p.input_per_mtok != null || p.output_per_mtok != null) return `${fmtUsd(p.input_per_mtok)} / ${fmtUsd(p.output_per_mtok)} per 1M tok`;
  if (p.per_image != null) return `${fmtUsd(p.per_image)} / image`;
  if (p.first_megapixel != null) return `${fmtUsd(p.first_megapixel)} + ${fmtUsd(p.per_megapixel)}/MP`;
  if (p.per_megapixel != null) return `${fmtUsd((p.per_request || 0) + p.per_megapixel)} / MP`;
  if (p.per_request != null) return `${fmtUsd(p.per_request)} / ${u}`;
  if (p.per_second != null) return `${fmtUsd(p.per_second)} / s`;
  if (p.per_minute != null) return `${fmtUsd(p.per_minute)} / min`;
  if (p.per_1k_chars != null) return `${fmtUsd(p.per_1k_chars)} / 1k chars`;
  if (p.credits != null) return `${p.credits} credits (~${fmtUsd(p.credits * (p.per_credit ?? 0.01))})`;
  return '';
}

export function cloudBadge(m) {
  if (!isCloud(m)) return null;
  return h('span.badge.cloud', { title: `${m.cloud.local ? 'Your endpoint' : 'Runs in the cloud'}: ${m.cloud.label}. ${m.cloud.terms || ''}` },
    ic(m.cloud.local ? 'server' : 'cloud', 12), `${m.cloud.local ? 'Endpoint' : 'Cloud'} · ${m.cloud.label}`);
}
export function priceBadge(m) {
  const t = priceLabel(m); if (!t) return null;
  return h('span.badge.outline.price', { title: `${m.pricing_overridden ? 'Your price override' : 'List price'}${m.pricing_checked ? ` (checked ${m.pricing_checked})` : ''}. Prices change: see the Providers page.` }, ic('circle-dollar-sign', 12), t);
}

/** "≈ $0.04" element for a Generate button (null for local models). */
export function costSpan(m, values, nInputs = 0) {
  if (!isCloud(m) || m.cloud.local) return null;
  const v = estimateCost(m, values, nInputs);
  return h('span.eta.cost', { title: 'Estimated cost (list price). The actual cost is recorded when the provider reports it.' }, v == null ? '· price ?' : `· ≈ ${fmtUsd(v)}`);
}

/** Cost line of a job card. */
export function jobCost(j) {
  const c = j?.cost; if (!c || c.local) return null;
  if (c.usd != null) return h('span.badge.outline.price', { title: c.note || (c.estimated ? 'List-price estimate (the provider did not report a cost)' : 'Cost reported by the provider') }, ic('circle-dollar-sign', 12), `${fmtUsd(c.usd)}${c.estimated ? ' est.' : ''}`);
  if (c.estimate_usd != null) return h('span.badge.outline.price', { title: 'Estimated before running' }, ic('circle-dollar-sign', 12), `≈ ${fmtUsd(c.estimate_usd)}`);
  return null;
}

// ------------------------------------------------------------------ confirm / consent dialogs
function ask({ title, icon, body, ok, danger = false }) {
  return new Promise(res => {
    let done = false;
    const md = modal({ title, icon, width: 480, body, onClose: () => { if (!done) res(false); },
      foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn(ok, { kind: danger ? 'danger' : 'primary', onClick: () => { done = true; md.close(); res(true); } })] });
  });
}

async function guard(err, spec) {
  const d = err.data || {};
  if (err.kind === 'confirm') {
    const est = d.estimate_usd;
    const ok = await ask({ title: 'Confirm cloud cost', icon: 'circle-dollar-sign', ok: est != null ? `Run for ≈ ${fmtUsd(est)}` : 'Run anyway',
      body: h('div.col', { style: { gap: '10px' } },
        h('p', { style: { margin: 0 } }, err.message),
        h('dl.kv', h('dt', 'Provider'), h('dd', d.provider_label || d.provider || '–'),
          h('dt', 'This month'), h('dd', `${fmtUsd(d.month_usd)}${d.pending_usd ? ` + ${fmtUsd(d.pending_usd)} running` : ''} of ${d.budget_usd ? fmtUsd(d.budget_usd) : 'no'} budget`),
          h('dt', 'Ask above'), h('dd', `${fmtUsd(d.confirm_usd)} per job`)),
        h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, 'List prices can be stale; the provider bills its current price. Change the threshold on the Providers page.'),
        h('a', { href: '#/providers', on: { click: () => navigate('providers') } }, 'Open Providers')) });
    return ok ? { ...spec, confirm_cost: true } : null;
  }
  if (err.kind === 'consent') {
    const ok = await ask({ title: `Send files to ${d.provider_label || 'the provider'}?`, icon: 'upload', ok: 'Allow and run',
      body: h('div.col', { style: { gap: '10px' } },
        h('p', { style: { margin: 0 } }, err.message),
        h('div.cloud-note', ic('info', 14), h('span', 'Your input images / audio leave this machine and are processed by ',
          h('b', d.provider_label || d.provider), '. They may be stored by the provider for a while (see its terms). This is asked once per provider; you can revoke it on the Providers page.')),
        d.terms ? h('p.dim', { style: { margin: 0, fontSize: '12.5px' } }, d.terms) : null,
        d.terms_url ? h('a', { href: d.terms_url, target: '_blank', rel: 'noopener' }, 'Provider terms') : null) });
    return ok ? { ...spec, confirm_upload: true } : null;
  }
  return null;
}
async function before(spec) {
  const m = store.modelsById.get(spec?.model);
  if (!isCloud(m) || m.cloud.local || spec.tool) return spec;
  let s = spec;
  for (let i = 0; i < 3; i++) {
    let r;
    try { r = await api.post('/api/x/providers/precheck', { task: s.task, model: s.model, params: s.params, inputs: s.inputs, confirm_cost: !!s.confirm_cost, confirm_upload: !!s.confirm_upload }); }
    catch { return s; }                       // precheck unavailable: the submit's own checks still apply
    if (r.ok || !['confirm', 'consent'].includes(r.kind)) return s;   // other refusals come back from the submit
    s = await guard({ kind: r.kind, message: r.error, data: r }, s);
    if (!s) return null;
  }
  return s;
}
setSubmitGuard(guard, before);

const CSS = `
.badge.cloud { background: color-mix(in srgb, #4aa3ff 16%, transparent); color: color-mix(in srgb, #4aa3ff 70%, var(--text)); border: 1px solid color-mix(in srgb, #4aa3ff 35%, transparent); }
.badge.price { color: var(--text-2); }
.status-dot.missing_key { background: var(--warn); } .status-dot.cloud_disabled { background: var(--text-3); }
.eta.cost { color: inherit; opacity: .9; }
.cloud-note { display: flex; gap: 8px; align-items: flex-start; padding: 10px 12px; border-radius: 10px; background: color-mix(in srgb, var(--warn) 12%, transparent); font-size: 13px; }
.cloud-note .ic { flex: none; margin-top: 2px; }
`;
document.head.append(h('style', CSS));
