// Material viewer (not a studio module: it lives in mat/). Registered as the drawer preview of `material` assets by studios/material.js
// (S.registerPreview). 3D view with live tiling (the texture repeat is changed on the loaded model, no reload), a flat 3x3 tile view
// of any map to judge seams, the map strip, facts and actions.
import * as S from '../../core/sdk.js';
import { mvScene } from '../i3d/common.js';
const { h, ic, btn } = S;

let cssDone = false;
export function css() {
  if (cssDone) return; cssDone = true;
  document.head.append(h('link', { rel: 'stylesheet', href: 'studios/mat/mat.css' }));
}

export const MAPS = [
  { id: 'basecolor', label: 'Base colour', file: 'basecolor.png' },
  { id: 'normal', label: 'Normal', file: 'normal.png' },
  { id: 'height', label: 'Height', file: 'height.png' },
  { id: 'roughness', label: 'Roughness', file: 'roughness.png' },
  { id: 'metallic', label: 'Metallic', file: 'metallic.png' },
  { id: 'ao', label: 'AO', file: 'ao.png' },
  { id: 'orm', label: 'ORM', file: 'orm.png' },
];
const TEX_KEYS = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap'];
const REPEAT = 1000;      // THREE.RepeatWrapping

/** Seam score -> badge class (docs/CONTRACTS.md: <= 1.05 seamless, <= 1.25 acceptable, above that visible). */
export const seamClass = v => (v <= 1.05 ? 'ok' : v <= 1.25 ? 'warn' : 'err');

/** Set the texture repeat of every map of a loaded <model-viewer> (no reload). false when its three.js scene is not reachable. */
export function applyTiling(mv, n) {
  const sc = mvScene(mv);
  if (!sc) return false;
  let any = false;
  sc.root.traverse(o => {
    if (!o.isMesh) return;
    for (const m of [].concat(o.material)) for (const k of TEX_KEYS) {
      const t = m && m[k];
      if (!t || !t.repeat) continue;
      t.wrapS = t.wrapT = REPEAT; t.repeat.set(n, n); t.needsUpdate = true; any = true;
    }
  });
  sc.scene.queueRender?.();
  return any;
}

const pref = (k, d) => S.prefs.get(`mat.${k}`, d);
const setPref = (k, v) => S.prefs.set(`mat.${k}`, v);

/** The drawer preview of a material asset. Returns an element synchronously; the 3D viewer loads lazily. */
export function materialPreview(a) {
  css();
  const X = a.x_material || {};
  const has = f => (a.files || []).some(x => x.name === f);
  const maps = MAPS.filter(m => has(m.file));
  let mode = pref('view', '3d'), shape = pref('shape', 'sphere'), tiles = Math.max(1, Math.min(8, Number(pref('tiles', 2)) || 2)), mapId = pref('map', 'basecolor'), edges = !!pref('edges', false);
  if (!maps.some(m => m.id === mapId)) mapId = 'basecolor';
  if (!has(`preview_${shape}.gltf`)) { shape = 'sphere'; }
  let mv = null, seq = 0;

  const stage = h('div.mat-stage');
  const nEl = h('span.mat-n', String(tiles));
  const shapeSeg = S.seg([{ value: 'sphere', label: 'Sphere' }, { value: 'cube', label: 'Cube' }, { value: 'plane', label: 'Plane' }], shape, v => { shape = v; setPref('shape', v); draw(); });
  const viewSeg = S.seg([{ value: '3d', label: '3D' }, { value: 'tile', label: 'Tile 3×3' }], mode, v => { mode = v; setPref('view', v); draw(); });
  const dec = btn('', { size: 'sm', icon: 'minus', title: 'Repeat less', onClick: () => setTiles(tiles - 1) });
  const inc = btn('', { size: 'sm', icon: 'plus', title: 'Repeat more', onClick: () => setTiles(tiles + 1) });
  const reset = btn('', { size: 'sm', icon: 'rotate-ccw', title: 'Reset the view', onClick: () => { if (mv) { mv.cameraOrbit = 'auto auto auto'; mv.cameraTarget = 'auto auto auto'; mv.fieldOfView = 'auto'; mv.jumpCameraToGoal?.(); } } });
  const edgeBtn = h('label.row', { style: { gap: '6px', fontSize: '12.5px' }, title: 'Draw the borders between the nine tiles' },
    h('input', { type: 'checkbox', checked: edges, on: { change: e => { edges = e.target.checked; setPref('edges', edges); draw(); } } }), 'Tile edges');
  const stepper = h('span.mat-step', { title: 'How many times the texture repeats across the shape' }, dec, nEl, inc, S.uiInfo('material.tiling'));
  const bar = h('div.mat-bar', viewSeg, S.uiInfo('material.view'), h('span.spacer'));
  const bar2 = h('div.mat-bar');

  function setTiles(n) {
    tiles = Math.max(1, Math.min(8, n)); nEl.textContent = String(tiles); setPref('tiles', tiles);
    if (mv && !applyTiling(mv, tiles)) { /* scene not ready or not reachable: the load handler applies it */ }
  }

  function draw() {
    const mine = ++seq;
    viewSeg.set(mode);
    drawStrip();
    bar2.replaceChildren();
    mv = null;
    if (mode === 'tile') {
      const m = maps.find(x => x.id === mapId) || maps[0];
      const t = h(`div.mat-tiles${edges ? '.edges' : ''}`, { style: { backgroundImage: `url("${S.fileUrl(a, m.file)}")` }, role: 'img', 'aria-label': `${m.label} repeated 3 by 3` });
      stage.replaceChildren(t, h('span.badge.ov.mat-tag', `${m.label} · 3×3`));
      bar2.append(edgeBtn, h('span.dim', { style: { fontSize: '12px' } }, 'Look for lines or jumps where tiles meet.'));
      return;
    }
    bar2.append(shapeSeg, S.uiInfo('material.shape'), stepper, reset);
    stage.replaceChildren(h('div.empty', ic('loader', 24, 'spin'), h('p', 'Loading 3D viewer…')));
    if (!has(`preview_${shape}.gltf`)) { stage.replaceChildren(h('div.empty', h('p', 'This material has no 3D preview. Use the 3×3 tile view.'))); return; }
    S.loadModelViewer().then(() => {
      if (mine !== seq) return;
      const el = h('model-viewer', { src: S.fileUrl(a, `preview_${shape}.gltf`), 'camera-controls': true, 'shadow-intensity': '0.7', exposure: '1.05', 'interaction-prompt': 'none',
        'environment-image': 'neutral', alt: `${a.name || 'Material'} on a ${shape}`, 'camera-orbit': shape === 'plane' ? '0deg 35deg auto' : 'auto auto auto' });
      el.addEventListener('load', () => { if (!applyTiling(el, tiles) && tiles !== 1) S.toast('Live tiling is not available in this viewer build: use the 3×3 tile view.', { kind: 'warn' }); }, { once: true });
      el.addEventListener('error', () => { if (mine === seq) stage.replaceChildren(h('div.empty', h('p', 'The 3D preview could not be loaded.'))); });
      mv = el;
      stage.replaceChildren(el);
    }).catch(() => { if (mine === seq) stage.replaceChildren(h('div.empty', h('p', 'The 3D viewer could not be loaded.'))); });
  }

  // ---- map strip (listbox)
  const strip = h('div.mat-strip', { role: 'listbox', 'aria-label': 'Maps' });
  const drawStrip = () => {
    strip.replaceChildren(...maps.map(m => h('button.mat-map', { type: 'button', role: 'option', 'aria-selected': String(m.id === mapId && mode === 'tile'), title: `${m.label}: click to inspect it tiled`,
      on: { click: () => { mapId = m.id; setPref('map', m.id); mode = 'tile'; setPref('view', 'tile'); draw(); } } },
      h('img', { src: S.fileUrl(a, m.file), alt: '', loading: 'lazy', decoding: 'async' }), h('span', m.label))));
  };
  strip.addEventListener('keydown', e => {
    if (!['ArrowRight', 'ArrowLeft'].includes(e.key)) return;
    const b = [...strip.querySelectorAll('button')]; const i = b.indexOf(document.activeElement); if (i < 0) return;
    e.preventDefault(); e.stopPropagation(); b[(i + (e.key === 'ArrowRight' ? 1 : -1) + b.length) % b.length].focus();
  });

  // ---- facts
  const seams = X.seam ? Object.entries(X.seam) : [];
  const facts = h('div.mat-facts',
    X.size ? h('span.badge.outline', `${X.size} × ${X.size} px`) : null,
    X.normal ? h('span.badge.outline', `normal ${String(X.normal).toLowerCase() === 'opengl' ? 'OpenGL' : X.normal}`) : null,
    X.source ? h('span.badge.outline', X.source === 'text' ? 'from text' : X.source === 'photo' ? 'from a photo' : 'variation') : null,
    X.tiling?.method ? h('span.badge.outline', `tiling: ${String(X.tiling.method).replace(/[_+]/g, ' ')}`) : null,
    ...seams.map(([k, v]) => h(`span.badge.${seamClass(Number(v))}`, { title: `Seam score of the ${k} map (1.0 = seamless)` }, `${k} seam ${Number(v).toFixed(2)}`)),
    null);
  const warn = (a.warnings || []).length ? h('div.mat-note', { style: { color: 'var(--warn)' } }, ic('triangle-alert', 13), ' ', a.warnings.join(' · ')) : null;

  // ---- actions
  const sampleBtn = btn('Sample GLB', { size: 'sm', icon: 'cuboid', title: 'A GLB of this material on the chosen shape, with the textures inside', onClick: async () => {
    sampleBtn.disabled = true;
    try {
      const job = await S.api.submit({ tool: 'material.sample_glb', params: { shape, tiles, size: 1024 }, inputs: [{ asset: a.id, role: 'input' }], project: a.project || S.store.project,
        title: `${a.name || 'Material'} (${shape} ×${tiles})`, meta: { studio: 'material' } });
      S.store.jobs.set(job.id, job);
      S.toast('Writing the sample GLB…', { kind: 'info' });
      const j = await S.waitJob(job.id);
      if (j.status === 'failed') S.toast(j.error?.message || 'Job failed', { kind: 'err', title: 'Sample GLB failed' });
      else if (j.outputs?.[0]) { S.toast('Sample GLB saved to the library', { kind: 'ok' }); S.openAsset(j.outputs[0]); }
    } catch (e) { S.toastError(e, 'Cannot make the sample GLB'); }
    sampleBtn.disabled = false;
  } });
  const thumbBtn = btn('Use view as thumbnail', { size: 'sm', icon: 'image', onClick: async () => {
    if (!mv || mode !== '3d') { S.toast('Switch to the 3D view first.', { kind: 'info' }); return; }
    try { const blob = await mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9, idealAspect: true }); await S.api.setThumb(a.id, blob); S.toast('Thumbnail updated', { kind: 'ok' }); } catch (e) { S.toastError(e); }
  } });
  const actions = h('div.mat-actions',
    btn('Download', { size: 'sm', kind: 'primary', icon: 'download', title: 'Choose a layout for Unreal, Unity, Godot or glTF', onClick: () => S.exportAssets([a.id], { options: true }) }),
    btn('Vary material', { size: 'sm', icon: 'shuffle', title: 'Make four variations of this material in the Materials studio (the Variations button below repeats the job with new seeds)', onClick: () => { S.closeAsset(); S.sendTo('material', a.id, { action: 'material_vary' }); } }),
    sampleBtn, thumbBtn);

  const side = h('div.mat-side',
    h('div.mat-sec-title', 'Maps', S.uiInfo('material.maps')), strip,
    h('div.mat-sec-title', 'Facts', S.uiInfo('material.seam')), facts, warn, actions);
  const main = h('div.mat-main', bar, stage, bar2);
  const root = h('div.mat-pv', h('div.mat-grid', main, side));
  draw();
  return root;
}
