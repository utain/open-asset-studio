// Model weight downloads: confirm dialog (size, disk, licence + accept), live progress widget, list for the Queue page.
// Server side: app/downloads.py. Only registry models can be downloaded; the API takes a model id.
import { api } from '../api.js';
import { store, bus } from '../store.js';
import { h, ic, btn, modal, toast, toastError, progressBar, fmtBytes, fmtDur, licenseBadge, copyText } from '../ui.js';

const ACTIVE = ['queued', 'planning', 'running'];
export const canDownload = m => m?.status?.status === 'missing_weights' && m.status.downloadable !== false;

const fixLine = fix => fix ? (/^https?:/.test(fix)
  ? h('a', { href: fix, target: '_blank', rel: 'noopener' }, ic('external-link', 12), ' ', fix)
  : h('div.fixwrap', h('code.fixcode', fix), btn('', { size: 'sm', icon: 'copy', title: 'Copy', onClick: () => copyText(fix) }))) : null;

/** Confirm dialog: what will be downloaded, disk space, licence (must be accepted). */
export function downloadDialog(m) {
  const body = h('div.col', { style: { gap: '12px' } }, h('div.row', progressBar(0, true), h('span.dim', 'Asking Hugging Face for the file list…')));
  let accepted = false;
  const go = btn('Download', { kind: 'primary', icon: 'download', disabled: true, onClick: async () => {
    go.disabled = true;
    try { const st = await api.downloadStart(m.id, true); store.downloads.set(m.id, st); bus.emit('download:' + m.id, st); bus.emit('download', st); md.close(); toast(`${m.name}: download ${st.status}`, { kind: 'ok', timeout: 2500 }); }
    catch (e) { toastError(e); go.disabled = false; }
  } });
  const md = modal({ title: `Download ${m.name}`, icon: 'download', width: 560, body, foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), go] });
  api.downloadPlan(m.id).then(p => {
    const lic = p.license || {};
    const gated = p.items.some(i => i.gated);
    const chk = h('input', { type: 'checkbox', id: 'dl-accept', on: { change: () => { accepted = chk.checked; go.disabled = !(accepted && p.enough_disk && p.items.length); } } });
    body.replaceChildren(
      p.items.length ? h('div.tbl-wrap', h('table.tbl', h('thead', h('tr', h('th', 'Files'), h('th', 'Size'), h('th', 'Already here'))),
        h('tbody', ...p.items.map(i => h('tr', h('td', h('div.mono', { style: { fontSize: '12px', overflowWrap: 'anywhere' } }, i.label), h('div.dim', { style: { fontSize: '11.5px' } }, `${i.files} file${i.files > 1 ? 's' : ''}${i.kind === 'cmd' ? ' · size is an estimate' : ''}`)),
          h('td.nowrap', fmtBytes(i.total)), h('td.nowrap.dim', i.present ? fmtBytes(i.present) : '–')))))) : h('div.dim', 'Nothing can be downloaded automatically for this model.'),
      h('div.row.wrap', { style: { gap: '18px' } }, h('div.stat', h('b', fmtBytes(p.need)), h('span', 'to download')), h('div.stat', h('b', fmtBytes(p.free)), h('span', 'free on disk')),
        p.enough_disk ? null : h('span.badge.err', ic('triangle-alert', 12), 'Not enough disk space')),
      p.blocked.length ? h('div.card.card-pad.col', { style: { gap: '6px', borderColor: 'var(--warn)' } }, h('b', ic('triangle-alert', 14), ' Still needed after the download (manual step)'),
        ...p.blocked.map(b => h('div.col', { style: { gap: '4px' } }, h('div', b.label), fixLine(b.fix)))) : null,
      p.note ? h('div.dim', { style: { fontSize: '12.5px' } }, ic('info', 12), ' ', p.note) : null,
      h('div.card.card-pad.col', { style: { gap: '8px' } },
        h('div.row.wrap', h('b.grow', 'Licence'), licenseBadge(lic)),
        h('div', lic.name || 'unknown'), lic.note ? h('div.dim', { style: { fontSize: '12.5px' } }, lic.note) : null,
        lic.commercial ? null : h('div', { style: { color: 'var(--err)', fontSize: '12.5px' } }, 'Not for commercial use: assets made with this model cannot ship in a commercial game.'),
        gated ? h('div.dim', { style: { fontSize: '12.5px' } }, ic('lock', 12), ' Gated on Hugging Face: your token (hf auth login) must have accepted the terms at ',
          ...p.items.filter(i => i.gated).map(i => h('a', { href: i.url, target: '_blank', rel: 'noopener' }, i.repo))) : null,
        h('label.row', { for: 'dl-accept', style: { gap: '8px', cursor: 'pointer' } }, chk, h('span', 'I have read and accept this licence'))),
      h('div.dim', { style: { fontSize: '12px' } }, 'Only the download process goes online; generation stays offline. You can cancel and resume later.'));
  }).catch(e => {
    body.replaceChildren(h('div', { style: { color: 'var(--err)' } }, ic('circle-alert', 14), ' ', e.message), fixLine(e.fix));
  });
  return md;
}

/** Inline status/progress for one model; updates itself from SSE. */
export function downloadWidget(m, { compact = false } = {}) {
  const el = h('div.dl-widget.col', { style: { gap: '6px', marginTop: '6px' } });
  const bar = progressBar(0);
  const draw = () => {
    const d = store.downloads.get(m.id);
    if (d && ACTIVE.includes(d.status)) {
      const frac = d.total ? d.done / d.total : 0;
      bar.set(frac, d.status !== 'running' || !d.total);
      el.replaceChildren(bar, h('div.row.wrap', { style: { gap: '8px', fontSize: '12px' } },
        h('span.grow', d.status === 'queued' ? 'Waiting for another download…' : d.status === 'planning' ? 'Preparing…'
          : `${fmtBytes(d.done)} / ${fmtBytes(d.total)} (${Math.round(frac * 100)}%)${d.rate > 1 ? ` · ${fmtBytes(d.rate)}/s` : ''}${d.eta ? ` · ${fmtDur(d.eta)} left` : ''}`),
        btn('Cancel', { size: 'sm', kind: 'ghost', icon: 'x', onClick: () => api.downloadCancel(m.id).catch(toastError) })));
      return;
    }
    const kids = [];
    if (d?.status === 'error') kids.push(h('div', { style: { color: 'var(--err)', fontSize: '12px', overflowWrap: 'anywhere' } }, ic('circle-alert', 12), ' ', d.message), fixLine(d.fix));
    if (d?.status === 'cancelled') kids.push(h('div.dim', { style: { fontSize: '12px' } }, d.total ? `Stopped at ${fmtBytes(d.done)} of ${fmtBytes(d.total)}. ` : '', 'Download again to resume.'));
    if (d?.status === 'done' && m.status?.status !== 'ready') kids.push(h('div.dim', { style: { fontSize: '12px' } }, d.message), fixLine(d.fix));
    if (canDownload(m)) kids.push(h('div', btn(d && d.status !== 'done' ? 'Resume download' : 'Download', { size: 'sm', kind: compact ? '' : 'primary', icon: 'download', onClick: () => downloadDialog(m) })));
    el.replaceChildren(...kids);
  };
  const off = bus.on('download:' + m.id, () => { if (!el.isConnected && el.dataset.mounted) { off(); return; } el.dataset.mounted = '1'; draw(); });
  draw();
  requestAnimationFrame(() => { el.dataset.mounted = '1'; });
  return el;
}

/** Queue page section: every download of this session. */
export function downloadsList() {
  const el = h('div.col', { style: { gap: '10px' } });
  const draw = () => {
    const ds = [...store.downloads.values()].sort((a, b) => (b.queued || 0) - (a.queued || 0));
    if (!ds.length) { el.replaceChildren(h('div.dim', 'No downloads yet. Missing models can be downloaded from the Models page.')); return; }
    el.replaceChildren(...ds.map(d => {
      const m = store.modelsById.get(d.model) || { id: d.model, name: d.name, status: {} };
      const badge = { queued: 'outline', planning: 'info', running: 'accent', done: 'ok', error: 'err', cancelled: 'outline' }[d.status] || 'outline';
      return h('div.card.card-pad.col', { style: { gap: '4px' } }, h('div.row', h('b.grow', m.name || d.model), h(`span.badge.${badge}`, d.status)),
        d.item ? h('div.dim.mono', { style: { fontSize: '11.5px', overflowWrap: 'anywhere' } }, d.item) : null, downloadWidget(m, { compact: true }));
    }));
  };
  const seen = new Map();
  const off = bus.on('download', d => {
    if (!el.isConnected && el.dataset.mounted) { off(); return; }
    el.dataset.mounted = '1';
    if (seen.get(d.model) !== d.status) { seen.set(d.model, d.status); draw(); }
  });
  for (const d of store.downloads.values()) seen.set(d.model, d.status);
  draw();
  return el;
}
