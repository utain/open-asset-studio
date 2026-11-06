// A/B compare: images (side by side or wipe slider), audio (A/B toggle with synced position), meshes (2-up).
import { api, fileUrl } from '../api.js';
import { h, ic, btn, modal, seg, stars, toastError } from '../ui.js';
import { loadModelViewer } from './drawer.js';
import { pickAssets } from './assetpicker.js';

export async function compare(a, b) {
  try {
    if (typeof a === 'string') a = await api.asset(a);
    if (typeof b === 'string') b = await api.asset(b);
  } catch (e) { toastError(e); return; }
  const body = h('div.col', { style: { gap: '12px' } });
  let mode = a.type === 'image' && b.type === 'image' ? 'wipe' : 'side';
  const render = () => {
    body.replaceChildren();
    if (a.type === 'image' && b.type === 'image') body.append(h('div.row', seg([{ value: 'wipe', label: 'Wipe' }, { value: 'side', label: 'Side by side' }], mode, v => { mode = v; render(); })));
    if (a.type === 'image' && b.type === 'image' && mode === 'wipe') body.append(wipe(a, b));
    else if ((a.type === 'audio' || a.type === 'voice') && (b.type === 'audio' || b.type === 'voice')) body.append(abAudio(a, b));
    else body.append(h('div.cmp', side(a, 'A'), side(b, 'B')));
    body.append(h('div.cmp', rateRow(a, 'A'), rateRow(b, 'B')));
  };
  render();
  const m = modal({ title: 'Compare', icon: 'columns-2', width: 1000, body, onClose: () => { audioEl?.pause(); } });
  return m;
}

let audioEl = null;
function rateRow(x, label) {
  return h('div.row', h('span.badge.accent', label), h('span.ellipsis.grow', x.name || x.id), stars(x.rating || 0, v => api.patchAsset(x.id, { rating: v }).catch(toastError)));
}
function side(x, label) {
  if (x.type === 'image') return h(`div.cmp-wipe${x.info?.has_alpha ? '.checker' : ''}`, { style: { background: x.info?.has_alpha ? null : 'var(--panel-2)' } }, h('img', { src: fileUrl(x), alt: label }), h('span.cmp-label', { style: { left: '8px' } }, label));
  if (x.type === 'mesh') {
    const box = h('div.pv', h('div.empty', ic('loader', 20, 'spin')));
    loadModelViewer().then(() => box.replaceChildren(h('model-viewer', { src: fileUrl(x), 'camera-controls': true, 'auto-rotate': true, alt: label })));
    return h('div.col', h('span.badge.accent', label), box);
  }
  return h('div.pv', h('div.empty', x.name || x.id));
}
function wipe(a, b) {
  const box = h(`div.cmp-wipe${a.info?.has_alpha || b.info?.has_alpha ? '.checker' : ''}`, { style: { maxHeight: '70vh', margin: '0 auto', width: 'min(100%, 70vh)' } },
    h('img', { src: fileUrl(a), alt: 'A' }), h('img.b', { src: fileUrl(b), alt: 'B' }), h('div.bar'),
    h('span.cmp-label', { style: { left: '8px' } }, 'A'), h('span.cmp-label', { style: { right: '8px' } }, 'B'));
  const move = e => { const r = box.getBoundingClientRect(); const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); box.style.setProperty('--x', `${x * 100}%`); };
  let down = false;
  box.addEventListener('pointerdown', e => { down = true; box.setPointerCapture(e.pointerId); move(e); });
  box.addEventListener('pointermove', e => { if (down) move(e); });
  box.addEventListener('pointerup', () => { down = false; });
  return box;
}
function abAudio(a, b) {
  audioEl = new Audio(); audioEl.preload = 'auto';
  let cur = 'A'; let loop = true;
  const src = { A: fileUrl(a), B: fileUrl(b) };
  audioEl.src = src.A; audioEl.loop = loop;
  const t = h('span.mono.dim', '0.0 s');
  audioEl.addEventListener('timeupdate', () => { t.textContent = `${audioEl.currentTime.toFixed(1)} s`; });
  const sw = seg([{ value: 'A', label: `A · ${a.name || ''}`.slice(0, 26) }, { value: 'B', label: `B · ${b.name || ''}`.slice(0, 26) }], 'A', v => {
    const pos = audioEl.currentTime, playing = !audioEl.paused; cur = v; audioEl.src = src[v];
    audioEl.addEventListener('loadedmetadata', () => { audioEl.currentTime = Math.min(pos, audioEl.duration || pos); if (playing) audioEl.play(); }, { once: true });
  });
  const play = btn('Play', { kind: 'primary', icon: 'play', onClick: () => { if (audioEl.paused) audioEl.play(); else audioEl.pause(); } });
  audioEl.addEventListener('play', () => { play.replaceChildren(ic('pause', 17), h('span', 'Pause')); });
  audioEl.addEventListener('pause', () => { play.replaceChildren(ic('play', 17), h('span', 'Play')); });
  const lp = btn('Loop', { icon: 'repeat', kind: 'primary', onClick: () => { loop = !loop; audioEl.loop = loop; lp.classList.toggle('btn-primary', loop); } });
  const keys = e => { if (e.key === 'a' || e.key === 'A') sw.querySelector('[data-v="A"]').click(); if (e.key === 'b' || e.key === 'B') sw.querySelector('[data-v="B"]').click(); };
  addEventListener('keydown', keys);
  const wrap = h('div.col', { style: { gap: '12px' } }, h('div.row.wrap', play, lp, sw, t), h('div.dim', { style: { fontSize: '12.5px' } }, 'Switch with A / B keys: playback continues at the same position (blind level-matched listening is up to you).'));
  const mo = new MutationObserver(() => { if (!wrap.isConnected) { removeEventListener('keydown', keys); audioEl.pause(); mo.disconnect(); } });
  setTimeout(() => mo.observe(document.body, { childList: true, subtree: true }), 0);
  return wrap;
}

export async function pickAndCompare(a) {
  const ids = await pickAssets({ types: [a.type === 'voice' ? 'audio' : a.type, a.type], title: 'Compare with…', exclude: [a.id] });
  if (ids?.length) compare(a, ids[0]);
}
