// Shared helpers of the Image, 3D and Pipelines pages (not a studio module: it lives in a sub-folder so the shell
// does not import it as a page). Mesh viewer, automatic mesh thumbnails, alpha background toggle, job helpers.
import * as S from '../../core/sdk.js';
const { h, ic, btn } = S;

let cssDone = false;
export function css() {
  if (cssDone) return; cssDone = true;
  document.head.append(h('link', { rel: 'stylesheet', href: 'studios/i3d/i3d.css' }));
}

// ------------------------------------------------------------------ edit presets (engines/hy3d image_params.EDIT_PRESETS)
export const EDIT_PRESETS = [
  ['Plain background', 'Replace the background with a plain pure white background. Keep the subject exactly as it is.'],
  ['3D-ready cleanup', 'Show the subject alone as a single centered object, fully visible, in a three-quarter front view, clean even lighting, no cast shadow, plain white background.'],
  ['Neutral pose', 'Turn the character into a clean neutral A-pose reference: full body, facing the camera, arms slightly away from the body, legs straight, plain background, same outfit and identity.'],
  ['Change colour', 'Change the main color of the subject to deep red. Keep its shape, details and material unchanged.'],
  ['Change material', 'Change the material of the subject to brushed metal. Keep its shape and proportions unchanged.'],
  ['Cartoon style', 'Redraw the image in a clean cartoon style with bold outlines and flat colors. Keep the composition.'],
  ['Low-poly style', 'Redraw the subject as a low-poly 3D render with visible flat-shaded facets. Keep the composition.'],
  ['Clay render', 'Turn the subject into an untextured matte grey clay render with soft even lighting.'],
  ['Remove object', 'Remove the object mentioned and fill the area naturally with the surrounding background.'],
];

// ------------------------------------------------------------------ models
export const readyModels = task => S.modelsFor(task).filter(m => m.status?.status === 'ready');
export function bestModel(task, prefer = []) {
  const ready = readyModels(task);
  for (const id of prefer) if (ready.find(m => m.id === id)) return id;
  const d = S.store.defaults[task];
  if (ready.find(m => m.id === d)) return d;
  return ready[0]?.id || null;
}

// ------------------------------------------------------------------ alpha background toggle
const BG = [{ value: 'checker', label: 'Checker' }, { value: 'white', label: 'White' }, { value: 'black', label: 'Black' }, { value: 'grey', label: 'Grey' }];
/** Segmented control that sets the background behind transparent images inside `target`. */
export function alphaBgToggle(target, key = 'i3d.alphaBg') {
  const apply = v => { for (const b of BG) target.classList.toggle(`i3d-bg-${b.value}`, b.value === v); };
  const cur = S.prefs.get(key, 'checker');
  apply(cur);
  const el = S.seg(BG, cur, v => { S.prefs.set(key, v); apply(v); });
  el.title = 'Background behind transparent images';
  el.classList.add('i3d-bgseg');
  return el;
}

// ------------------------------------------------------------------ jobs
/** Submit a job spec; resolves with {job, assets} when it is terminal (throws on failure / cancel). */
export async function runJob(spec, { onJob } = {}) {
  const job = await S.api.submit(spec);
  S.store.jobs.set(job.id, job);
  onJob?.(job);
  return finishJob(job.id);
}
export async function finishJob(id) {
  let j = S.store.jobs.get(id);
  if (!j || !['done', 'failed', 'cancelled'].includes(j.status)) {
    try { j = await S.api.job(id); S.store.jobs.set(id, j); } catch { /* use SSE */ }
  }
  if (!j || !['done', 'failed', 'cancelled'].includes(j.status)) j = await S.waitJob(id);
  if (j.status === 'failed') { const e = new Error(j.error?.message || 'Job failed'); e.fix = j.error?.fix; e.job = j; throw e; }
  if (j.status === 'cancelled') { const e = new Error('Cancelled'); e.cancelled = true; e.job = j; throw e; }
  const assets = j.outputs?.length ? (await S.api.assets({ ids: j.outputs.join(','), limit: 50 })).items : [];
  const ordered = (j.outputs || []).map(i => assets.find(a => a.id === i)).filter(Boolean);
  return { job: j, assets: ordered };
}

// ------------------------------------------------------------------ mesh thumbnails
const thumbQueue = [];
const thumbSeen = new Set();
let thumbBusy = false;
let thumbHost = null;
/** Render a model-viewer snapshot for meshes without a thumbnail (one at a time, off-screen) and store it. */
export function autoThumb(assets) {
  for (const a of [].concat(assets || [])) {
    if (!a || a.type !== 'mesh' || a.preview?.thumb || thumbSeen.has(a.id) || !/\.glb$/i.test(a.main || '')) continue;
    thumbSeen.add(a.id); thumbQueue.push(a);
  }
  pumpThumbs();
}
async function pumpThumbs() {
  if (thumbBusy || !thumbQueue.length) return;
  thumbBusy = true;
  const a = thumbQueue.shift();
  try {
    await S.loadModelViewer();
    if (!thumbHost) { thumbHost = h('div.i3d-thumbhost', { 'aria-hidden': 'true' }); document.body.append(thumbHost); }
    const mv = h('model-viewer', { src: S.fileUrl(a), loading: 'eager', reveal: 'auto', 'camera-orbit': '35deg 70deg auto', exposure: '1.05', 'shadow-intensity': '0.6', 'interaction-prompt': 'none', alt: '' });
    thumbHost.replaceChildren(mv);
    await new Promise((res, rej) => { const t = setTimeout(() => rej(new Error('timeout')), 45000); mv.addEventListener('load', () => { clearTimeout(t); res(); }, { once: true }); mv.addEventListener('error', () => { clearTimeout(t); rej(new Error('load error')); }, { once: true }); });
    await new Promise(r => setTimeout(r, 400));
    const blob = await mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9, idealAspect: false });
    await S.api.setThumb(a.id, blob);
  } catch (e) { /* thumbnails are optional */ }
  finally { thumbHost?.replaceChildren(); thumbBusy = false; setTimeout(pumpThumbs, 50); }
}

// ------------------------------------------------------------------ edges / wireframe overlay
/** Draw the mesh's triangle edges on a loaded <model-viewer>. view: 'edges' (over the surface) | 'wire' (lines only, see-through).
 *  model-viewer has no wireframe option, so this reaches into its three.js scene (internal symbol); returns false if unavailable. */
function applyWire(mv, view) {
  const sym = Object.getOwnPropertySymbols(mv).find(k => k.description === 'scene');
  const scene = sym && mv[sym], root = scene && (scene.model || scene._model);
  if (!root) return false;
  const meshes = [];
  root.traverse(o => { if (o.isMesh && o.geometry && !o.userData.i3dWire) meshes.push(o); });
  const dark = document.documentElement.classList.contains('dark') || document.body.classList.contains('dark') ||
    (!document.documentElement.classList.contains('light') && matchMedia('(prefers-color-scheme: dark)').matches);
  const line = view === 'wire' ? (dark ? 0xd8dbe2 : 0x2b2f36) : 0x1f2328;
  for (const m of meshes) {
    for (const mat of [].concat(m.material)) { mat.visible = view !== 'wire'; mat.polygonOffset = true; mat.polygonOffsetFactor = 1; mat.polygonOffsetUnits = 1; }
    const wm = [].concat(m.material)[0].clone();
    for (const k of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap', 'alphaMap', 'bumpMap']) if (k in wm) wm[k] = null;
    Object.assign(wm, { wireframe: true, transparent: true, depthWrite: false, depthTest: view !== 'wire', opacity: view === 'wire' ? 0.08 : 0.45, visible: true });
    if (wm.metalness !== undefined) { wm.metalness = 0; wm.roughness = 1; }
    // unlit flat lines: black albedo, no environment reflection, colour comes only from emissive
    wm.color?.set(0x000000); wm.emissive?.set(line); wm.envMapIntensity = 0; wm.toneMapped = false;
    wm.needsUpdate = true;
    const w = m.clone(false); w.material = wm; w.castShadow = w.receiveShadow = false;
    w.position.set(0, 0, 0); w.rotation.set(0, 0, 0); w.scale.set(1, 1, 1); w.userData.i3dWire = true;
    m.add(w);
  }
  if (view === 'wire') mv.setAttribute('shadow-intensity', '0');   // a solid shadow under see-through lines looks wrong
  scene.queueRender?.();
  return true;
}

// ------------------------------------------------------------------ mesh viewer
/** Big model-viewer with view toggles: textured / white shape / edges / wireframe, auto-rotate, light, stats, downloads. */
export function meshViewer({ height = 420, empty: emptyText = 'Generate or pick a mesh to preview it here.', actions } = {}) {
  const box = h('div.i3d-viewer', { style: { '--vh': `${height}px` } });
  const bar = h('div.i3d-vbar');
  const stats = h('div.i3d-vstats.dim');
  const el = h('div.i3d-viewer-wrap', box, bar, stats);
  let cur = null, mode = 'textured', rotate = S.prefs.get('i3d.rotate', true), light = S.prefs.get('i3d.light', 'neutral');
  const hasWhite = a => (a?.files || []).some(f => f.name === 'white_mesh.glb');
  const geo = () => mode !== 'textured';   // shape, edges and wireframe all show the untextured mesh when there is one
  const srcOf = a => (geo() && hasWhite(a) ? S.fileUrl(a, 'white_mesh.glb') : S.fileUrl(a, a.files?.find(f => f.name.endsWith('_preview.glb'))?.name || a.main));
  const render = () => {
    const a = cur;
    if (!a) { box.replaceChildren(S.empty('cuboid', 'No mesh yet', emptyText)); bar.replaceChildren(); stats.replaceChildren(); return; }
    box.replaceChildren(h('div.empty', ic('loader', 22, 'spin'), h('p', 'Loading the 3D viewer…')));
    S.loadModelViewer().then(() => {
      if (cur !== a) return;
      const mv = h('model-viewer', { src: srcOf(a), 'camera-controls': true, 'auto-rotate': rotate || null, 'shadow-intensity': '0.8',
        exposure: light === 'bright' ? '1.4' : light === 'dark' ? '0.7' : '1', 'environment-image': 'neutral', 'interaction-prompt': 'none', alt: a.name || '3D model', 'touch-action': 'pan-y' });
      if (mode === 'edges' || mode === 'wire') {   // poll: model-viewer's 'load' event is not reliable for a re-created element
        const view = mode, t0 = Date.now();
        const tick = () => { if (cur !== a || !mv.isConnected || Date.now() - t0 > 120000) return;
          if (mv.loaded && applyWire(mv, view)) return; setTimeout(tick, 150); };
        setTimeout(tick, 150);
      }
      mv.addEventListener('load', () => setTimeout(async () => {   // the visible viewer doubles as the thumbnail camera
        if (mode === 'edges' || mode === 'wire') return;   // don't save a wireframe as the thumbnail
        if (a.preview?.thumb || thumbSeen.has(a.id) || cur !== a) return;
        thumbSeen.add(a.id);
        try { await S.api.setThumb(a.id, await mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9 })); } catch { thumbSeen.delete(a.id); autoThumb(a); }
      }, 700), { once: true });
      box.replaceChildren(mv);
      el._mv = mv;
    }).catch(() => box.replaceChildren(h('div.empty', 'The 3D viewer could not be loaded.')));
    const I = a.info || {};
    stats.replaceChildren(h('span.ellipsis', { style: { fontWeight: 600, color: 'var(--text)' } }, a.name || a.id),
      I.faces ? h('span', `${I.faces.toLocaleString()} faces`) : null, I.vertices ? h('span', `${I.vertices.toLocaleString()} verts`) : null,
      I.textures != null ? h('span', `${I.textures} texture(s)`) : null, a.worker?.stats?.time?.total ? h('span', `made in ${S.fmtDur(a.worker.stats.time.total)}`) : null);
    bar.replaceChildren(
      S.seg([{ value: 'textured', label: 'Textured' }, hasWhite(a) ? { value: 'white', label: 'Shape' } : null,
        { value: 'edges', label: 'Edges' }, { value: 'wire', label: 'Wireframe' }].filter(Boolean), mode, v => { mode = v; render(); }),
      btn('', { size: 'sm', icon: rotate ? 'pause' : 'refresh-cw', title: rotate ? 'Stop rotating' : 'Auto-rotate', onClick: () => { rotate = !rotate; S.prefs.set('i3d.rotate', rotate); render(); } }),
      btn('', { size: 'sm', icon: 'sun', title: `Light: ${light}`, onClick: () => { light = { neutral: 'bright', bright: 'dark', dark: 'neutral' }[light]; S.prefs.set('i3d.light', light); render(); } }),
      h('span.spacer'),
      btn('', { size: 'sm', icon: 'image', title: 'Use this view as the thumbnail', onClick: async () => {
        try { const blob = await el._mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9 }); await S.api.setThumb(a.id, blob); S.toast('Thumbnail updated', { kind: 'ok' }); } catch (e) { S.toastError(e); }
      } }),
      btn('', { size: 'sm', icon: 'download', title: 'Download GLB', onClick: () => { location.href = S.fileUrl(a, a.main, true); } }),
      btn('', { size: 'sm', icon: 'eye', title: 'Details', onClick: () => S.openAsset(a) }),
      ...(actions ? actions(a) : []));
  };
  el.set = a => { if (a && a.type !== 'mesh' && !(a.files || []).some(f => f.name.endsWith('_preview.glb'))) return; if (cur?.id !== a?.id) mode = 'textured'; cur = a; render(); };
  el.current = () => cur;
  render();
  return el;
}

// ------------------------------------------------------------------ image preview with alpha bg + zoom
export function imagePreview(a, { height } = {}) {
  const img = h('img', { src: S.fileUrl(a), alt: a.name || '', draggable: false });
  const box = h(`div.i3d-img${a.info?.has_alpha ? '.checker' : ''}`, { style: height ? { maxHeight: `${height}px` } : null, title: 'Click to open with zoom', on: { click: () => S.openAsset(a) } }, img);
  return box;
}

// ------------------------------------------------------------------ flow ids
export function flowId() {
  const d = new Date(); const p = n => String(n).padStart(2, '0');
  return `f${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${Math.random().toString(36).slice(2, 8)}`;
}
