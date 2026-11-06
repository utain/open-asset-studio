// Audio: one shared player for every card (only one sound plays at a time), canvas waveforms from the
// precomputed peaks (waveform.json, no decoding in the browser), and a wavesurfer view for the detail drawer.
import { fileUrl, waveform } from '../api.js';
import { bus } from '../store.js';
import { h, ic, fmtClock } from '../ui.js';

const audio = new Audio();
audio.preload = 'none';
let currentId = null;
let raf = 0;
const views = new Set();

function tick() {
  for (const v of views) { if (!v.c.isConnected && v.seen) { views.delete(v); v.ro.disconnect(); continue; } v.draw(); }
  if (!audio.paused) raf = requestAnimationFrame(tick); else raf = 0;
}
audio.addEventListener('play', () => { if (!raf) raf = requestAnimationFrame(tick); bus.emit('player', { id: currentId, playing: true }); });
audio.addEventListener('pause', () => { for (const v of views) v.draw(); bus.emit('player', { id: currentId, playing: false }); });
audio.addEventListener('ended', () => { for (const v of views) v.draw(); bus.emit('player', { id: currentId, playing: false }); });

export const player = {
  get current() { return currentId; },
  get playing() { return !audio.paused; },
  get time() { return audio.currentTime; },
  get duration() { return audio.duration || 0; },
  play(asset, { at = null, loop = null } = {}) {
    if (currentId !== asset.id) { audio.src = fileUrl(asset); currentId = asset.id; }
    if (loop !== null) audio.loop = loop;
    if (at !== null) audio.currentTime = at;
    return audio.play().catch(() => {});
  },
  toggle(asset) { if (currentId === asset.id && !audio.paused) audio.pause(); else this.play(asset); },
  stop() { audio.pause(); },
  seekFrac(asset, f) { const d = asset.info?.duration || audio.duration || 0; this.play(asset, { at: Math.max(0, Math.min(d, f * d)) }); },
  setLoop(v) { audio.loop = !!v; },
  get loop() { return audio.loop; },
};

function cssVar(name, el = document.documentElement) { return getComputedStyle(el).getPropertyValue(name).trim(); }

/** Canvas waveform of an audio asset, synced with the shared player. Click = seek + play. */
export function waveCanvas(asset, { height = 48 } = {}) {
  const c = h('canvas.wave', { style: { height: height + 'px' }, 'aria-label': 'Waveform (click to play from here)' });
  let peaks = null;
  const view = {
    c, seen: false, ro: null,
    draw() {
      if (c.isConnected) view.seen = true;
      const w = c.clientWidth, hh = c.clientHeight; if (!w || !hh) return;
      const dpr = devicePixelRatio || 1;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(hh * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); }
      const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hh);
      const base = cssVar('--text-3'), played = cssVar('--c-sound');
      const p = peaks?.peaks?.[0] || [];
      const bars = Math.max(1, Math.floor(w / 3));
      const dur = asset.info?.duration || 0;
      const frac = currentId === asset.id && dur ? Math.min(1, audio.currentTime / dur) : -1;
      for (let i = 0; i < bars; i++) {
        let v = 0.04;
        if (p.length) { const a = Math.floor(i / bars * p.length), b = Math.max(a + 1, Math.floor((i + 1) / bars * p.length)); for (let k = a; k < b; k++) v = Math.max(v, p[k]); }
        const bh = Math.max(2, v * (hh - 4));
        g.fillStyle = frac >= 0 && i / bars <= frac ? played : base;
        g.globalAlpha = frac >= 0 && i / bars <= frac ? 1 : 0.65;
        g.fillRect(i * 3, (hh - bh) / 2, 2, bh);
      }
      g.globalAlpha = 1;
    },
  };
  c.addEventListener('click', e => { e.stopPropagation(); const r = c.getBoundingClientRect(); player.seekFrac(asset, (e.clientX - r.left) / r.width); });
  views.add(view);
  waveform(asset).then(p => { peaks = p; view.draw(); });
  view.ro = new ResizeObserver(() => view.draw()); view.ro.observe(c);
  c.redraw = () => view.draw();
  return c;
}

/** Play button bound to the shared player. */
export function playButton(asset, size = 38) {
  const b = h('button.play-btn', { type: 'button', title: 'Play / pause (space)', style: { width: size + 'px', height: size + 'px' }, on: { click: e => { e.stopPropagation(); player.toggle(asset); } } });
  const draw = () => { const on = player.current === asset.id && player.playing; b.classList.toggle('on', on); b.replaceChildren(ic(on ? 'pause' : 'play', size > 32 ? 18 : 15)); };
  draw();
  const off = bus.on('player', () => { if (!b.isConnected) { off(); return; } draw(); });
  return b;
}

/** Waveform + transport for the detail drawer / compare (wavesurfer.js, vendored). */
export async function waveSurfer(el, asset, { height = 96, loop = false } = {}) {
  const { default: WaveSurfer } = await import('../../vendor/wavesurfer.esm.js');
  const wf = await waveform(asset);
  const ws = WaveSurfer.create({
    container: el, height, url: fileUrl(asset), peaks: wf?.peaks, duration: wf?.duration,
    waveColor: cssVar('--text-3'), progressColor: cssVar('--c-sound'), cursorColor: cssVar('--text'),
    barWidth: 2, barGap: 1, barRadius: 2, normalize: true, dragToSeek: true,
  });
  player.stop();
  ws.on('finish', () => { if (ws._loop) ws.play(); });
  ws._loop = loop;
  return ws;
}

/** Compact transport row used by audio cards. */
export function audioRow(asset) {
  const dur = asset.info?.duration;
  return h('div.wave-row', playButton(asset), h('div.grow', waveCanvas(asset)), h('span.dim.mono', { style: { fontSize: '11.5px' } }, dur != null ? fmtClock(dur) : ''));
}

// global shortcut: space toggles the current sound when focus is not in a field
addEventListener('keydown', e => {
  if (e.code !== 'Space' || !currentId || /INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable) return;
  e.preventDefault(); if (audio.paused) audio.play().catch(() => {}); else audio.pause();
});
