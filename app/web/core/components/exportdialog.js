// Export dialog: pick an export preset (the profile's first) or tune loudness / sample rate / bit depth / format for audio
// and aspect / size / format for images, then download one converted file or a zip. Originals are never modified: the
// server writes derived files (POST /api/assets/<id>/render, POST /api/export with `export`).
import { api } from '../api.js';
import { store, prefs, getProfile, isOverridden } from '../store.js';
import { h, ic, btn, modal, seg, toast, toastError, fmtBytes } from '../ui.js';
import { uiInfo } from '../uiinfo.js';

const SR = [['', 'Keep'], ['44100', '44.1 kHz'], ['48000', '48 kHz'], ['96000', '96 kHz']];
const BD = [['', 'Default'], ['16', '16-bit'], ['24', '24-bit'], ['32', '32-bit float']];
const LUFS = [['', 'No change'], ['-14', '-14 LUFS (online video / social)'], ['-16', '-16 LUFS (podcast)'], ['-23', '-23 LUFS (broadcast / film)'], ['-18', '-18 LUFS (quiet)']];
const ASPECT = [['', 'Keep'], ['1:1', '1:1'], ['16:9', '16:9'], ['9:16', '9:16'], ['4:3', '4:3'], ['3:4', '3:4'], ['3:2', '3:2'], ['2.39:1', '2.39:1 (cinema)']];
const SIZE = [['', 'Keep size'], ['m:1080', 'Long side 1080'], ['m:1920', 'Long side 1920'], ['m:2048', 'Long side 2048'], ['m:4096', 'Long side 4096'],
  ['1920x1080', '1920 x 1080'], ['1280x720', '1280 x 720'], ['1080x1920', '1080 x 1920'], ['1080x1080', '1080 x 1080'], ['3000x3000', '3000 x 3000']];

const sel = (opts, value, onChange, label) => h('select.select', { 'aria-label': label, on: { change: e => onChange(e.target.value) } },
  ...opts.map(([v, l]) => h('option', { value: v, selected: String(value ?? '') === v }, l)));
// a div, not a <label>: the (i) button would otherwise become the control that clicking the label activates (the selects carry an aria-label)
const field = (label, el, info) => h('div.field', h('span.field-label', label, info ? uiInfo(info) : null), el);
const sizeKey = i => (i?.width && i?.height ? `${i.width}x${i.height}` : i?.max_side ? `m:${i.max_side}` : '');
const fromPreset = p => ({
  audio: { on: !!p?.audio, format: p?.audio?.format || 'wav', sample_rate: String(p?.audio?.sample_rate ?? ''), bit_depth: String(p?.audio?.bit_depth ?? ''),
    lufs: p?.audio?.lufs != null ? String(p.audio.lufs) : '', music_sample_rate: p?.audio?.music_sample_rate || null },
  image: { on: !!p?.image, format: p?.image?.format || 'png', aspect: p?.image?.aspect || '', fit: p?.image?.fit || 'crop', size: sizeKey(p?.image), quality: p?.image?.quality || null },
});

function specOf(st) {
  const a = st.audio, i = st.image;
  const audio = a.on ? { format: a.format, sample_rate: a.sample_rate ? Number(a.sample_rate) : null, bit_depth: a.bit_depth ? Number(a.bit_depth) : null, lufs: a.lufs !== '' ? Number(a.lufs) : null,
    true_peak: -1.0, ...(a.music_sample_rate ? { music_sample_rate: a.music_sample_rate } : {}) } : null;
  const image = i.on ? { format: i.format, aspect: i.aspect || null, fit: i.fit,
    ...(i.size.startsWith('m:') ? { max_side: Number(i.size.slice(2)) } : i.size ? { width: Number(i.size.split('x')[0]), height: Number(i.size.split('x')[1]) } : {}),
    ...(i.quality ? { quality: i.quality } : {}) } : null;
  return { audio, image };
}

function triggerDownload(url, name) { const a = document.createElement('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); }

/** exportDialog(ids) -> opens the dialog. Resolves nothing; downloads start from the buttons. */
export async function exportDialog(ids, { title = 'Export' } = {}) {
  let metas = [];
  try { metas = (await api.assets({ ids: ids.join(','), limit: 200 })).items; } catch (e) { toastError(e); return; }
  const audioN = metas.filter(m => m.type === 'audio' || m.type === 'voice').length;
  const imageN = metas.filter(m => m.type === 'image').length;
  const otherN = metas.length - audioN - imageN;
  const prof = getProfile();
  const order = [...new Set([...(prof.export_presets || []), ...Object.keys(store.exportPresets)])].filter(id => store.exportPresets[id]);
  // 1. the preset picked before in this project, 2. the project's own export preset, 3. the old global pick, 4. the default
  const mine = prefs.get(`export.preset.${store.project}`, null), legacy = prefs.get('export.preset', null);
  let preset = order.includes(mine) ? mine : (isOverridden('export.preset') && order.includes(store.exportDefault)) ? store.exportDefault
    : order.includes(legacy) ? legacy : null;
  if (!preset) preset = store.exportDefault || order[0] || 'original';
  let st = fromPreset(store.exportPresets[preset]);
  const body = h('div.col', { style: { gap: '14px' } });
  const sum = h('div.dim', { style: { fontSize: '12.5px' } });

  const draw = () => {
    const P = store.exportPresets[preset];
    sum.textContent = P?.summary || '';
    const a = st.audio, i = st.image;
    const parts = [];
    parts.push(h('div.field', h('span.field-label', 'Preset', uiInfo('export_preset'), h('span.dim', { style: { fontWeight: 400 } }, ` · ${prof.label} profile first`)),
      h('select.select', { 'aria-label': 'Export preset', on: { change: e => { preset = e.target.value; prefs.set(`export.preset.${store.project}`, preset); st = fromPreset(store.exportPresets[preset]); draw(); } } },
        ...order.map(id => h('option', { value: id, selected: id === preset }, store.exportPresets[id].label + (id === (store.exportDefault || prof.export) ? '  (default)' : ''))))), sum);
    if (audioN) {
      const grid = h('div.export-grid',
        field('Format', seg(['wav', 'mp3', 'ogg', 'flac'], a.format, v => { a.format = v; draw(); }), 'export_audio_format'),
        field('Loudness', sel(LUFS, a.lufs, v => { a.lufs = v; }), 'export_loudness'),
        field('Sample rate', sel(SR, a.sample_rate, v => { a.sample_rate = v; a.music_sample_rate = null; }), 'export_sample_rate'),
        ['wav', 'flac'].includes(a.format) ? field('Bit depth', sel(BD, a.bit_depth, v => { a.bit_depth = v; }), 'export_bit_depth') : null);
      parts.push(h('section.card.card-pad.col', { style: { gap: '10px' } },
        h('label.row', { style: { gap: '8px' } }, h('input', { type: 'checkbox', checked: a.on, on: { change: e => { a.on = e.target.checked; draw(); } } }), h('b', ic('audio-lines', 15), ` Audio (${audioN})`), uiInfo('export_audio')),
        a.on ? grid : h('div.dim', { style: { fontSize: '12.5px' } }, 'Original audio files.'),
        a.on && a.lufs !== '' ? h('div.dim', { style: { fontSize: '12px' } }, 'Two-pass ffmpeg loudnorm to the target, true peak -1 dBTP. Silent files get no gain.') : null));
    }
    if (imageN) {
      const grid = h('div.export-grid',
        field('Format', seg([{ value: 'png', label: 'PNG' }, { value: 'jpeg', label: 'JPEG' }, { value: 'webp', label: 'WebP' }], i.format, v => { i.format = v; draw(); }), 'export_image_format'),
        field('Aspect', sel(ASPECT, i.aspect, v => { i.aspect = v; }), 'export_aspect'),
        field('Size', sel(SIZE, i.size, v => { i.size = v; }), 'export_size'),
        field('When the aspect differs', seg([{ value: 'crop', label: 'Crop' }, { value: 'pad', label: 'Pad' }], i.fit, v => { i.fit = v; }), 'export_fit'));
      parts.push(h('section.card.card-pad.col', { style: { gap: '10px' } },
        h('label.row', { style: { gap: '8px' } }, h('input', { type: 'checkbox', checked: i.on, on: { change: e => { i.on = e.target.checked; draw(); } } }), h('b', ic('image', 15), ` Images (${imageN})`), uiInfo('export_images')),
        i.on ? grid : h('div.dim', { style: { fontSize: '12.5px' } }, 'Original image files.'),
        i.on && i.format === 'jpeg' ? h('div.dim', { style: { fontSize: '12px' } }, 'JPEG has no transparency: transparent areas are flattened onto white.') : null));
    }
    if (otherN) parts.push(h('div.dim', { style: { fontSize: '12.5px' } }, `${otherN} other asset(s) (3D, video…) are included unchanged.`));
    parts.push(h('div.dim', { style: { fontSize: '12px' } }, ic('shield-check', 12), ' Your library files are never modified: conversions are written to new files.'));
    body.replaceChildren(...parts);
  };
  draw();

  const busyBtn = async (b, fn) => { b.disabled = true; const t = b.textContent; b.querySelector('span') && (b.querySelector('span').textContent = 'Working…'); try { await fn(); } catch (e) { toastError(e, 'Export failed'); } b.disabled = false; b.querySelector('span') && (b.querySelector('span').textContent = t); };
  const single = metas.length === 1 && (audioN + imageN) === 1;
  const zipBtn = btn(single ? 'Download zip' : 'Export zip', { icon: 'package', kind: single ? '' : 'primary', onClick: () => busyBtn(zipBtn, async () => {
    const r = await api.exportAssets(ids, { export: specOf(st) });
    toast(`${r.count} asset(s) · ${fmtBytes(r.bytes)}`, { kind: 'ok', title: 'Export ready' });
    triggerDownload(r.url, r.name); md.close();
  }) });
  const fileBtn = single ? btn('Download file', { icon: 'download', kind: 'primary', onClick: () => busyBtn(fileBtn, async () => {
    const spec = specOf(st);
    if (!spec.audio && !spec.image) { toast('Nothing to convert: use the original download in the asset panel.', { kind: 'warn' }); return; }
    const r = await api.renderAsset(ids[0], spec);
    const rep = r.report || {};
    const bits = [rep.output_lufs != null ? `${rep.output_lufs} LUFS` : null, rep.width ? `${rep.width} x ${rep.height}` : null, fmtBytes(r.bytes)].filter(Boolean).join(' · ');
    toast(bits, { kind: 'ok', title: rep.note ? `Converted (${rep.note})` : 'Converted' });
    triggerDownload(r.url, r.name); md.close();
  }) }) : null;
  const md = modal({ title: `${title} · ${metas.length} asset${metas.length === 1 ? '' : 's'}`, icon: 'package', width: 600, body, foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), zipBtn, fileBtn] });
}

/** Convert and download a single asset with the default preset (or open the dialog). */
export async function downloadConverted(asset) { return exportDialog([asset.id], { title: 'Download' }); }
