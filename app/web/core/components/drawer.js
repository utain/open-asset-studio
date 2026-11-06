// Asset detail drawer: preview (image zoom on a checkerboard, GLB in model-viewer, audio waveform), metadata,
// prompts, lineage, rating / tags / notes, send-to actions, regenerate, compare, export, delete.
import { api, fileUrl, thumbUrl } from '../api.js';
import { bus, store, defaultExportPreset } from '../store.js';
import { exportDialog } from './exportdialog.js';
import { h, ic, btn, stars, toast, toastError, confirmDialog, copyText, fmtBytes, fmtDur, fmtAgo, MOD_ICON, licenseBadge, menu, clear } from '../ui.js';
import { sendActions, sendTo, getStudio } from '../studio.js';
import { waveSurfer } from './audio.js';
import { videoBox, videoAttachments } from './video.js';
import { fmtUsd } from '../cost.js';

let drawerEl = null, bodyEl = null, headEl = null, current = null, list = [], ws = null;

let mvLoaded = null;
export function loadModelViewer() {
  if (!mvLoaded) mvLoaded = new Promise((res, rej) => { const s = document.createElement('script'); s.type = 'module'; s.src = 'vendor/model-viewer.min.js'; s.onload = res; s.onerror = rej; document.head.append(s); });
  return mvLoaded;
}

function ensure() {
  if (drawerEl) return;
  headEl = h('div.drawer-head');
  bodyEl = h('div.drawer-body');
  drawerEl = h('aside.drawer', { 'aria-label': 'Asset details' }, headEl, bodyEl);
  document.body.append(drawerEl);
  addEventListener('keydown', e => {
    if (!drawerEl.classList.contains('open') || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) return;
    if (e.key === 'Escape') closeAsset();
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') step(1);
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') step(-1);
  });
  bus.on('asset', ev => {
    if (!current || ev.asset.id !== current.id) return;
    if (ev.action === 'deleted') closeAsset(); else { current = ev.asset; renderMetaOnly(); }
  });
}

function step(d) {
  if (!current || list.length < 2) return;
  const i = list.indexOf(current.id); if (i < 0) return;
  openAsset(list[(i + d + list.length) % list.length], { list });
}

export function closeAsset() {
  if (!drawerEl) return;
  drawerEl.classList.remove('open'); current = null;
  if (ws) { ws.destroy(); ws = null; }
  bus.emit('drawer', null);
}

/** openAsset(idOrMeta, {list: [ids] for ←/→ navigation}) */
export async function openAsset(a, opts = {}) {
  ensure();
  if (opts.list) list = opts.list;
  try { current = typeof a === 'string' ? await api.asset(a) : a; } catch (e) { toastError(e, 'Cannot open asset'); return; }
  drawerEl.classList.add('open');
  bus.emit('drawer', current);
  render();
}

function render() {
  const a = current;
  if (ws) { ws.destroy(); ws = null; }
  const i = list.indexOf(a.id);
  headEl.replaceChildren(
    h(`span.badge.mod.mod-${a.type}`, ic(MOD_ICON[a.type] || 'package', 12), a.type),
    h('div.grow.ellipsis', { style: { fontWeight: 650 } }, a.name || a.id),
    list.length > 1 && i >= 0 ? h('span.dim.mono', { style: { fontSize: '11.5px' } }, `${i + 1}/${list.length}`) : null,
    list.length > 1 ? btn('', { icon: 'chevron-left', kind: 'ghost', size: 'sm', title: 'Previous (←)', onClick: () => step(-1) }) : null,
    list.length > 1 ? btn('', { icon: 'chevron-right', kind: 'ghost', size: 'sm', title: 'Next (→)', onClick: () => step(1) }) : null,
    btn('', { icon: 'x', kind: 'ghost', size: 'sm', title: 'Close (Esc)', onClick: closeAsset }));
  clear(bodyEl);
  bodyEl.append(preview(a), ...videoAttachments(a).map(f => videoBox(a, { name: f.name, compact: true })), h('div#drawer-meta.col', { style: { gap: '14px' } }));
  renderMetaOnly();
}

function preview(a) {
  if (a.type === 'image') return imagePreview(a);
  if (a.type === 'mesh') return meshPreview(a);
  if (a.type === 'audio' || a.type === 'voice') return audioPreview(a);
  if (a.type === 'video') return videoBox(a);
  return h('div.pv', h('div.empty', ic(MOD_ICON[a.type] || 'package', 30), h('p', a.main)));
}

function imagePreview(a) {
  const img = h('img', { src: fileUrl(a), alt: a.name || '', draggable: false });
  const box = h(`div.pv-img${a.info?.has_alpha ? '.checker' : ''}`, img);
  let z = 1, x = 0, y = 0, drag = null;
  const apply = () => { img.style.transform = `translate(${x}px, ${y}px) scale(${z})`; box.classList.toggle('zoomed', z > 1); zl.textContent = `${Math.round(z * 100)}%`; };
  const zoomAt = (nz, cx, cy) => { nz = Math.max(1, Math.min(12, nz)); x = cx - (cx - x) * (nz / z); y = cy - (cy - y) * (nz / z); z = nz; if (z === 1) { x = 0; y = 0; } apply(); };
  box.addEventListener('wheel', e => { e.preventDefault(); const r = box.getBoundingClientRect(); zoomAt(z * (e.deltaY < 0 ? 1.2 : 1 / 1.2), e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  box.addEventListener('click', e => { if (drag?.moved) return; if (z === 1) { const r = box.getBoundingClientRect(); zoomAt(3, e.clientX - r.left, e.clientY - r.top); } });
  box.addEventListener('pointerdown', e => { if (z === 1) return; drag = { sx: e.clientX - x, sy: e.clientY - y, moved: false }; box.setPointerCapture(e.pointerId); });
  box.addEventListener('pointermove', e => { if (!drag) return; x = e.clientX - drag.sx; y = e.clientY - drag.sy; drag.moved = true; apply(); });
  box.addEventListener('pointerup', () => { setTimeout(() => { drag = null; }, 0); });
  const zl = h('span.badge.ov', '100%');
  const tools = h('div.pv-tools', zl,
    h('button.btn.btn-sm.btn-icon.ov', { type: 'button', title: 'Zoom in', style: { border: 0 }, on: { click: e => { e.stopPropagation(); zoomAt(z * 1.5, box.clientWidth / 2, box.clientHeight / 2); } } }, ic('zoom-in', 15)),
    h('button.btn.btn-sm.btn-icon.ov', { type: 'button', title: 'Fit', style: { border: 0 }, on: { click: e => { e.stopPropagation(); z = 1; x = y = 0; apply(); } } }, ic('maximize-2', 15)));
  return h('div.pv', box, tools);
}

function meshPreview(a) {
  const wrap = h('div.pv', h('div.empty', ic('loader', 24, 'spin'), h('p', 'Loading 3D viewer…')));
  loadModelViewer().then(() => {
    const mv = h('model-viewer', { src: fileUrl(a), 'camera-controls': true, 'auto-rotate': true, 'shadow-intensity': '0.8', exposure: '1', 'interaction-prompt': 'none', alt: a.name || '3D model' });
    const snap = btn('Use view as thumbnail', { size: 'sm', icon: 'image', onClick: async () => {
      try { const blob = await mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9, idealAspect: true }); await api.setThumb(a.id, blob); toast('Thumbnail updated', { kind: 'ok' }); } catch (e) { toastError(e); }
    } });
    wrap.replaceChildren(mv, h('div.pv-tools', snap));
    if (!a.preview?.thumb) mv.addEventListener('load', () => setTimeout(async () => {
      try { const blob = await mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9 }); await api.setThumb(a.id, blob); } catch { /* optional */ }
    }, 600), { once: true });
  }).catch(() => wrap.replaceChildren(h('div.empty', 'The 3D viewer could not be loaded.')));
  return wrap;
}

function audioPreview(a) {
  const wv = h('div', { style: { minHeight: '96px' } });
  const playB = btn('', { icon: 'play', kind: 'primary', title: 'Play / pause', onClick: () => ws?.playPause() });
  const loopB = btn('', { icon: 'repeat', title: 'Loop', onClick: () => { if (!ws) return; ws._loop = !ws._loop; loopB.classList.toggle('btn-primary', ws._loop); } });
  const t = h('span.mono.dim', '0:00.0');
  const box = h('div.pv.pv-audio', wv, h('div.row', { style: { marginTop: '10px' } }, playB, loopB, t, h('span.spacer'),
    a.info?.lufs != null ? h('span.badge.outline', `${a.info.lufs.toFixed(1)} LUFS`) : null,
    a.info?.peak_db != null ? h(`span.badge.${a.info.clipped ? 'err' : 'outline'}`, `peak ${a.info.peak_db.toFixed(1)} dB`) : null));
  waveSurfer(wv, a).then(w => {
    ws = w;
    const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
    w.on('timeupdate', s => { t.textContent = `${fmt(s)} / ${fmt(w.getDuration() || a.info?.duration || 0)}`; });
    w.on('play', () => playB.replaceChildren(ic('pause', 17)));
    w.on('pause', () => playB.replaceChildren(ic('play', 17)));
    t.textContent = `0:00.0 / ${fmt(a.info?.duration || 0)}`;
  }).catch(e => wv.replaceChildren(h('div.dim', `Waveform unavailable: ${e.message}`)));
  return box;
}

function originStudio(a) {
  if (a.studio && getStudio(a.studio)) return a.studio;
  const t = store.tasks[a.task]; if (t?.studio && getStudio(t.studio)) return t.studio;
  return null;
}

function renderMetaOnly() {
  const a = current; const root = bodyEl.querySelector('#drawer-meta'); if (!root) return;
  clear(root);
  // actions
  const acts = sendActions(a);
  const actRow = h('div.row.wrap');
  for (const s of acts.slice(0, 6)) actRow.append(btn(s.label, { size: 'sm', icon: s.icon || 'send', onClick: () => { sendTo(s.studio, a.id, { action: s.action }); closeAsset(); } }));
  const origin = originStudio(a);
  if (origin && a.model && a.source !== 'upload') actRow.append(btn('Regenerate', { size: 'sm', icon: 'refresh-cw', title: 'Open the studio with exactly these settings', onClick: () => { sendTo(origin, a.id, { action: 'regenerate', params: a.params }); closeAsset(); } }));
  actRow.append(btn('', { size: 'sm', icon: 'download', title: 'Download', onClick: () => { location.href = fileUrl(a, a.main, true); } }),
    btn('', { size: 'sm', icon: 'more-horizontal', title: 'More', onClick: e => menu(e.currentTarget, [
      { label: 'Compare with…', icon: 'columns-2', onClick: () => import('./compare.js').then(m => m.pickAndCompare(a)) },
      { label: 'Export as zip', icon: 'package', onClick: () => exportAssets([a.id]) },
      ['audio', 'voice', 'image'].includes(a.type) ? { label: 'Export / convert…', icon: 'sliders-horizontal', onClick: () => exportAssets([a.id], { options: true }) } : null,
      { label: 'Copy asset id', icon: 'copy', onClick: () => copyText(a.id) },
      { sep: true },
      { label: 'Delete', icon: 'trash-2', danger: true, onClick: async () => {
        if (!(await confirmDialog(`Delete “${a.name || a.id}”? It is moved to the project's .trash folder.`, { title: 'Delete asset', ok: 'Delete', danger: true }))) return;
        try { await api.deleteAsset(a.id); toast('Asset deleted', { kind: 'ok' }); closeAsset(); } catch (err) { toastError(err); }
      } },
    ], { align: 'end' }) }));
  root.append(actRow);

  // rating / favourite / tags / notes
  const fav = btn('', { size: 'sm', kind: 'ghost', icon: 'heart', title: 'Favourite', onClick: () => api.patchAsset(a.id, { favorite: !a.favorite }).catch(toastError) });
  if (a.favorite) { fav.style.color = '#ff6b8b'; fav.querySelector('svg').style.fill = '#ff6b8b'; }
  root.append(h('div.row', stars(a.rating || 0, v => api.patchAsset(a.id, { rating: v }).catch(toastError), 18), fav, h('span.spacer'), licenseBadge(a.license)));
  root.append(tagEditor(a));
  const notes = h('textarea.textarea', { rows: 2, placeholder: 'Notes…', value: a.notes || '', on: { change: e => api.patchAsset(a.id, { notes: e.target.value }).catch(toastError) } });
  root.append(notes);

  // prompts
  const pr = [];
  if (a.prompt) pr.push(['Prompt', a.prompt]);
  if (a.improved_prompt) pr.push(['Improved', a.improved_prompt]);
  if (a.final_prompt && a.final_prompt !== a.prompt && a.final_prompt !== a.improved_prompt) pr.push(['Sent to model', a.final_prompt]);
  if (a.negative) pr.push(['Negative', a.negative]);
  if (pr.length) root.append(h('div.drawer-sec', h('h4', 'Prompt'), ...pr.map(([k, v]) => h('div', { style: { marginBottom: '8px' } },
    h('div.row', h('span.dim', { style: { fontSize: '12px' } }, k), h('span.spacer'), btn('', { size: 'sm', kind: 'ghost', icon: 'copy', title: 'Copy', onClick: () => copyText(v) })),
    h('div', { style: { fontSize: '13px', overflowWrap: 'anywhere' } }, v)))));

  // generation
  const gen = h('dl.kv');
  const kv = (k, v) => { if (v !== undefined && v !== null && v !== '') gen.append(h('dt', k), h('dd', v)); };
  kv('Model', a.model_name || a.model);
  kv('Task', store.tasks[a.task]?.label || a.task);
  kv('Seed', a.seed != null ? h('span.mono', String(a.seed)) : null);
  kv('Source', a.source);
  if (a.cloud) kv('Provider', h('span', ic('cloud', 12), ` ${a.cloud.provider_label || a.cloud.provider}${a.cloud.remote_model ? ` · ${a.cloud.remote_model}` : ''}`));
  if (a.cost) kv('Cost', `${fmtUsd(a.cost.usd)}${a.cost.estimated ? ' (estimate)' : ''}${a.cost.job_usd && a.cost.job_usd !== a.cost.usd ? ` · job ${fmtUsd(a.cost.job_usd)}` : ''}`);
  if (a.cloud?.request_ids?.length) kv('Request', h('span.mono', { style: { fontSize: '11.5px' } }, a.cloud.request_ids.join(', ')));
  kv('Created', `${new Date(a.created).toLocaleString()} (${fmtAgo(a.created)})`);
  if (a.job) kv('Job', h('span.mono', a.job));
  root.append(h('div.drawer-sec', h('h4', 'Generation'), gen));
  if (a.cloud) root.append(h('div.drawer-sec', h('h4', 'Terms (cloud)'), h('div.cloud-note', ic('info', 14), h('span',
    a.license?.note && a.license.note !== a.cloud.terms ? `${a.license.name || 'Model'}: ${a.license.note} ` : '',
    a.cloud.terms || 'The provider\'s terms apply.', a.cloud.terms_url ? [' ', h('a', { href: a.cloud.terms_url, target: '_blank', rel: 'noopener' }, 'Terms')] : null,
    a.cloud.uploaded_inputs ? h('div.dim', { style: { marginTop: '4px', fontSize: '12px' } }, `${a.cloud.uploaded_inputs} input file(s) were sent to ${a.cloud.provider_label}.`) : null)),
    a.license && !a.license.commercial ? h('div.err', { style: { marginTop: '6px', fontSize: '12.5px' } }, 'This model\'s output is not cleared for commercial use.') : null));
  const params = Object.entries(a.params || {}).filter(([k]) => !['prompt', 'negative'].includes(k));
  if (params.length) {
    const pv = h('dl.kv');
    for (const [k, v] of params) pv.append(h('dt', k), h('dd.mono', typeof v === 'object' ? JSON.stringify(v) : String(v)));
    root.append(h('details.drawer-sec', h('summary', { style: { cursor: 'pointer', color: 'var(--text-2)', fontSize: '12.5px', fontWeight: 600 } }, `Parameters (${params.length})`), h('div', { style: { marginTop: '8px' } }, pv)));
  }
  // file info
  const inf = h('dl.kv'); const I = a.info || {};
  const ikv = (k, v) => { if (v !== undefined && v !== null && v !== '') inf.append(h('dt', k), h('dd', v)); };
  if (a.type === 'image') { ikv('Size', I.width ? `${I.width} × ${I.height} px` : null); ikv('Alpha', I.has_alpha ? `yes (${Math.round((I.transparent_fraction || 0) * 100)}% transparent)` : 'no'); }
  if (a.type === 'audio' || a.type === 'voice') { ikv('Duration', fmtDur(I.duration)); ikv('Format', I.sample_rate ? `${I.sample_rate} Hz · ${I.channels} ch` : null); ikv('Loudness', I.lufs != null ? `${I.lufs} LUFS` : null); ikv('Peak', I.peak_db != null ? `${I.peak_db} dBFS${I.clipped ? ' (clipped)' : ''}` : null); ikv('Silence', I.silence_fraction != null ? `${Math.round(I.silence_fraction * 100)}%` : null); }
  if (a.type === 'video') { ikv('Duration', fmtDur(I.duration)); ikv('Size', I.width ? `${I.width} × ${I.height} px${I.fps ? ` · ${I.fps} fps` : ''}` : null); ikv('Codec', [I.codec, I.has_audio ? `audio ${I.audio_codec || ''}`.trim() : 'no audio'].filter(Boolean).join(' · ')); }
  if (a.type === 'mesh') { ikv('Vertices', I.vertices?.toLocaleString()); ikv('Faces', I.faces?.toLocaleString()); ikv('Materials', I.materials); ikv('Textures', I.textures); }
  for (const f of a.files || []) ikv(f.role === 'main' ? 'File' : 'Attachment', h('a', { href: fileUrl(a, f.name, true) }, `${f.name} · ${fmtBytes(f.bytes)}`));
  root.append(h('div.drawer-sec', h('h4', 'File'), inf));
  // lineage
  const lin = h('div.drawer-sec', h('h4', 'Lineage'), h('div.dim', { style: { fontSize: '12.5px' } }, 'Loading…'));
  root.append(lin);
  api.lineage(a.id).then(L => {
    const item = (m, rel) => {
      if (m.missing) return h('div.lin-item', h('div.lin-ph', ic('ban', 16)), h('div.grow.dim', `${m.id} (deleted)`));
      const t = thumbUrl(m);
      return h(`div.lin-item.mod-${m.type}`, { on: { click: () => openAsset(m.id, { list }) } },
        t ? h('img', { src: t, alt: '' }) : h('div.lin-ph', ic(MOD_ICON[m.type] || 'package', 16)),
        h('div.grow', { style: { minWidth: 0 } }, h('div.ellipsis', m.name || m.id), h('div.dim', { style: { fontSize: '11.5px' } }, `${rel} · ${m.model || m.type}`)));
    };
    const kids = [...L.ancestors.map(m => item(m, 'from')), ...L.children.map(m => item(m, 'used in'))];
    lin.replaceChildren(h('h4', 'Lineage'), kids.length ? h('div.lineage', ...kids) : h('div.dim', { style: { fontSize: '12.5px' } }, 'No parents or derived assets yet.'),
      L.siblings?.length ? h('div.dim', { style: { fontSize: '12px', marginTop: '6px' } }, `${L.siblings.length} other take(s) in the same batch`) : null);
  }).catch(() => lin.remove());
}

function tagEditor(a) {
  const tags = [...(a.tags || [])];
  const box = h('div.tag-input');
  const inp = h('input', { placeholder: tags.length ? '' : 'Add tags…', 'aria-label': 'Add tag' });
  const save = () => api.patchAsset(a.id, { tags }).catch(toastError);
  const draw = () => {
    box.replaceChildren(...tags.map((t, i) => h('span.tag', t, h('button', { type: 'button', title: 'Remove', on: { click: () => { tags.splice(i, 1); draw(); save(); } } }, ic('x', 12)))), inp);
  };
  inp.addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ',') && inp.value.trim()) { e.preventDefault(); const t = inp.value.trim().replace(/,$/, ''); if (!tags.includes(t)) tags.push(t); inp.value = ''; draw(); save(); inp.focus(); }
    if (e.key === 'Backspace' && !inp.value && tags.length) { tags.pop(); draw(); save(); inp.focus(); }
  });
  draw();
  return box;
}

/** Zip of the assets. With the profile's default export preset (settings) audio / images are converted into derived
 *  files inside the zip (originals untouched); `{options: true}` opens the export dialog (preset, loudness, format…).
 *  The game / general profiles default to "original": the stored files, exactly as before. */
export async function exportAssets(ids, { options = false, preset } = {}) {
  if (options) return exportDialog(ids);
  try {
    const p = preset || defaultExportPreset();
    const r = await api.exportAssets(ids, p && p !== 'original' ? { export: { preset: p } } : {});
    const label = p && p !== 'original' ? ` · ${store.exportPresets[p]?.label || p}` : '';
    toast(`${r.count} asset(s) · ${fmtBytes(r.bytes)}${label}`, { kind: 'ok', title: 'Export ready' });
    const a = document.createElement('a'); a.href = r.url; a.download = r.name; document.body.append(a); a.click(); a.remove();
  } catch (e) { toastError(e, 'Export failed'); }
}
export function currentAsset() { return current; }
