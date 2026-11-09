// Shared helpers of the Image, 3D and Pipelines pages (not a studio module: it lives in a sub-folder so the shell
// does not import it as a page). Mesh viewer, automatic mesh thumbnails, alpha background toggle, job helpers.
import * as S from '../../core/sdk.js';
import { rigEditor } from './rigedit.js';
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
/** The app theme is `data-theme` on <html> (dark unless set to light). */
const isDark = () => document.documentElement.dataset.theme !== 'light';
/** The scene of a <model-viewer> (internal symbol) and its model root, or null while they are not ready. */
function mvScene(mv) {
  const sym = Object.getOwnPropertySymbols(mv).find(k => k.description === 'scene');
  const scene = sym && mv[sym], root = scene && (scene.model || scene._model);
  return root ? { scene, root } : null;
}
/** Index buffer that draws quad edges only, for a retopo mesh: its first 2n triangles are n quads (triangles 2k and 2k+1 share the
 *  diagonal), and the primitive carries `extras.oas_quad_pairs = n` (three.js GLTFLoader copies primitive extras to geometry.userData).
 *  Every edge becomes a degenerate triangle (a, b, b), so a wireframe material draws exactly the segment a-b. Null when not applicable. */
function quadEdgeIndex(geo) {
  const n = geo?.userData?.oas_quad_pairs | 0, idx = geo?.index?.array;
  if (n <= 0 || !idx || !geo.attributes?.position) return null;
  const N = geo.attributes.position.count, seen = new Set(), out = [];
  const add = (a, b) => { const k = Math.min(a, b) * N + Math.max(a, b); if (!seen.has(k)) { seen.add(k); out.push(a, b, b); } };
  for (let t = 0, T = idx.length / 3; t < T; t++) {
    const a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
    if (t >= 2 * n) { add(a, b); add(b, c); add(c, a); }
    else if (t % 2 === 0) { add(a, b); add(b, c); }            // triangle (q0, q1, q2) of a pair: skip the diagonal q2-q0
    else { add(b, c); add(c, a); }                             // triangle (q0, q2, q3): skip the diagonal q0-q2
  }
  return out;
}

/** Draw the mesh's triangle edges on a loaded <model-viewer> (quad edges only for a retopo mesh, see quadEdgeIndex). view: 'edges' (over the surface) | 'wire' (lines only, see-through).
 *  model-viewer has no wireframe option, so this reaches into its three.js scene (internal symbol); returns false if unavailable. */
export function applyWire(mv, view) {
  const sym = Object.getOwnPropertySymbols(mv).find(k => k.description === 'scene');
  const scene = sym && mv[sym], root = scene && (scene.model || scene._model);
  if (!root) return false;
  const meshes = [];
  root.traverse(o => { if (o.isMesh && o.geometry && !o.userData.i3dWire) meshes.push(o); });
  const dark = isDark();
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
    const w = m.clone(false); w.material = wm;
    const qi = quadEdgeIndex(m.geometry);
    if (qi) { const g = m.geometry.clone(); g.setIndex(qi); w.geometry = g; }
    w.castShadow = w.receiveShadow = false;
    w.position.set(0, 0, 0); w.rotation.set(0, 0, 0); w.scale.set(1, 1, 1); w.userData.i3dWire = true;
    m.add(w);
  }
  if (view === 'wire') mv.setAttribute('shadow-intensity', '0');   // a solid shadow under see-through lines looks wrong
  scene.queueRender?.();
  return true;
}

// ------------------------------------------------------------------ bones overlay
/** Skeleton of the loaded GLB: {bones, links:[[parent, child]]} (empty when the model has no skin), or null while the scene is not ready. */
function findSkeleton(mv) {
  const sc = mvScene(mv); if (!sc) return null;
  const set = new Set();
  sc.root.traverse(o => { if (o.isSkinnedMesh && o.skeleton) for (const b of o.skeleton.bones) set.add(b); });
  const bones = [...set], links = [];
  for (const b of bones) if (b.parent && set.has(b.parent)) links.push([b.parent, b]);
  return { ...sc, bones, links };
}

/** Draw the skeleton of a loaded <model-viewer> on top of the mesh (depth test off) and make the surface semi-transparent.
 *  One line object (a bone = a thin octahedron from the parent joint to the child joint, a joint = a small cross) whose
 *  vertices are recomputed from the live bone matrices in onBeforeRender, so it follows any playing animation.
 *  three.js classes come from existing scene objects (no direct import). Returns false if unavailable. */
function applyBones(mv, sk) {
  const { scene, root, bones, links } = sk;
  if (!bones.length || root.userData.i3dBones) return !!root.userData.i3dBones;
  const proto = bones[0], V3 = proto.position.constructor, M4 = proto.matrixWorld.constructor;
  let tmpl = null;
  root.traverse(o => { if (!tmpl && o.isMesh && o.material && !o.userData.i3dWire) tmpl = o; });
  if (!tmpl) return false;
  const Geo = tmpl.geometry.constructor, Attr = tmpl.geometry.attributes.position.constructor;
  const nSeg = links.length * 12 + bones.length * 3;      // octahedron = 12 edges, cross = 3 edges
  const pos = new Float32Array(nSeg * 9);                            // every segment is a triangle (a, b, b): wireframe draws just the line a-b
  const geo = new Geo(); const attr = new Attr(pos, 3); attr.setUsage?.(35048); geo.setAttribute('position', attr);
  const mat = [].concat(tmpl.material)[0].clone();
  for (const k of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap', 'alphaMap', 'bumpMap']) if (k in mat) mat[k] = null;
  Object.assign(mat, { wireframe: true, transparent: true, opacity: 1, depthTest: false, depthWrite: false, visible: true, side: 2 });
  if (mat.metalness !== undefined) { mat.metalness = 0; mat.roughness = 1; }
  mat.color?.set(0x000000); mat.emissive?.set(isDark() ? 0xffc247 : 0xc2410c); mat.envMapIntensity = 0; mat.toneMapped = false;
  mat.polygonOffset = false; mat.needsUpdate = true;
  let P = Object.getPrototypeOf(tmpl);                               // the template is usually a SkinnedMesh (flag set per instance): its parent class is Mesh
  if (tmpl.isSkinnedMesh || tmpl.isInstancedMesh) P = Object.getPrototypeOf(P);
  const ov = new (P.constructor)(geo, mat);
  ov.isSkinnedMesh = ov.isInstancedMesh = false;                     // belt and braces: the renderer must treat it as a plain mesh
  ov.frustumCulled = false; ov.renderOrder = 1000; ov.castShadow = ov.receiveShadow = false; ov.userData.i3dBones = true; ov.userData.i3dWire = true;
  const inv = new M4(), a = new V3(), b = new V3(), d = new V3(), u = new V3(), v = new V3(), q = new V3(), r = new V3(), w = new V3();
  let k = 0, avg = 0;
  const seg = (p, c) => { pos[k++] = p.x; pos[k++] = p.y; pos[k++] = p.z; pos[k++] = c.x; pos[k++] = c.y; pos[k++] = c.z; pos[k++] = c.x; pos[k++] = c.y; pos[k++] = c.z; };
  const at = (o, bone) => { o.set(bone.matrixWorld.elements[12], bone.matrixWorld.elements[13], bone.matrixWorld.elements[14]).applyMatrix4(inv); return o; };
  ov.onBeforeRender = () => {
    inv.copy(root.matrixWorld).invert(); k = 0;
    if (!avg) {                                       // joint size from the mean bone length, measured on the first drawn frame
      let s = 0; for (const [p, c] of links) s += at(a, p).distanceTo(at(b, c));
      avg = (s / Math.max(1, links.length)) || 0.05;
    }
    for (const [p, c] of links) {
      at(a, p); at(b, c); d.copy(b).sub(a); const L = d.length() || 1e-6; d.multiplyScalar(1 / L);
      u.set(Math.abs(d.x) < 0.8 ? 1 : 0, Math.abs(d.x) < 0.8 ? 0 : 1, 0).cross(d).normalize().multiplyScalar(L * 0.12);
      v.copy(d).cross(u).normalize().multiplyScalar(L * 0.12);
      q.copy(a).addScaledVector(d, L * 0.15);        // the widest ring sits near the parent joint, like a bone in a DCC tool
      for (const [s, t] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { r.copy(q).addScaledVector(u, s).addScaledVector(v, t); seg(a, r); seg(r, b); }
      for (const [x, y] of [[[1, 0], [0, 1]], [[0, 1], [-1, 0]], [[-1, 0], [0, -1]], [[0, -1], [1, 0]]]) {
        r.copy(q).addScaledVector(u, x[0]).addScaledVector(v, x[1]); w.copy(q).addScaledVector(u, y[0]).addScaledVector(v, y[1]); seg(r, w);
      }
    }
    const j = avg * 0.1;
    for (const bone of bones) {
      at(a, bone);
      for (let i = 0; i < 3; i++) { b.copy(a); w.copy(a); b.setComponent(i, a.getComponent(i) - j); w.setComponent(i, a.getComponent(i) + j); seg(b, w); }
    }
    attr.needsUpdate = true;
  };
  root.add(ov); root.userData.i3dBones = ov;
  // ghost the surface so the skeleton reads through it (restored by re-creating the element: toggles re-render)
  root.traverse(o => { if (o.isMesh && !o.userData.i3dWire) for (const m of [].concat(o.material)) { if (!m.visible) continue; m.transparent = true; m.opacity = Math.min(m.opacity ?? 1, 0.38); m.needsUpdate = true; } });
  scene.queueRender?.();
  return true;
}

// ------------------------------------------------------------------ mesh viewer
/** Extension point for other i3d modules (engine.js): a function (asset, rerender) -> {bar?: Element[], src?: () => Promise<url|null>} | null.
 *  `bar` elements are added to the viewer's toolbar, `src` may return another model URL (a blob) to show instead of the asset's file,
 *  `stats` is a text that replaces the faces / verts figures (the file's own counts include hidden LOD and hull nodes). */
export const viewerHooks = [];

/** Big model-viewer with view toggles: textured / white shape / edges / wireframe, auto-rotate, light, stats, downloads. */
export function meshViewer({ height = 420, empty: emptyText = 'Generate or pick a mesh to preview it here.', actions } = {}) {
  const box = h('div.i3d-viewer', { style: { '--vh': `${height}px` } });
  const bar = h('div.i3d-vbar');
  const stats = h('div.i3d-vstats.dim');
  const edBox = h('div.rige-host', { hidden: true });        // the rig editor panel (rigedit.js) lives here
  const el = h('div.i3d-viewer-wrap', box, bar, stats, edBox);
  let cur = null, mode = 'textured', rotate = S.prefs.get('i3d.rotate', true), light = S.prefs.get('i3d.light', 'neutral');
  let playing = true, clip = null;       // animation state: kept while the element is re-created (mode / light changes)
  let editing = false, ed = null, edSt = { id: null, log: [], redo: [], sel: null };   // rig editor: open flag, instance, undo log (kept while the element is re-created)
  let bones = S.prefs.get('i3d.bones', false);   // skeleton overlay on / off (a pref; the button only shows for a GLB that has a skeleton)
  const hasWhite = a => (a?.files || []).some(f => f.name === 'white_mesh.glb');
  const geo = () => mode !== 'textured';   // shape, edges and wireframe all show the untextured mesh when there is one
  const srcOf = a => (geo() && hasWhite(a) ? S.fileUrl(a, 'white_mesh.glb') : S.fileUrl(a, a.files?.find(f => f.name.endsWith('_preview.glb'))?.name || a.main));
  /** Play / pause + clip picker for a GLB with animations (e.g. the flex / wiggle clips of a rigged mesh); hidden otherwise. */
  const animControls = (mv, box) => {
    const names = [...(mv.availableAnimations || [])];
    if (!names.length) { box.hidden = true; return; }
    if (!names.includes(clip)) clip = names[0];
    mv.animationName = clip;
    if (playing) mv.play(); else { mv.pause(); mv.currentTime = 0; }
    const pp = btn('', { size: 'sm', icon: playing ? 'pause' : 'play', title: playing ? 'Pause animation' : 'Play animation', onClick: () => {
      playing = !playing; if (playing) mv.play(); else mv.pause();
      pp.replaceChildren(ic(playing ? 'pause' : 'play', 15)); pp.title = playing ? 'Pause animation' : 'Play animation'; pp.setAttribute('aria-label', pp.title);
    } });
    const pick = names.length > 1 ? h('select.select.i3d-animsel', { title: 'Animation clip', 'aria-label': 'Animation clip',
      on: { change: e => { clip = e.target.value; mv.animationName = clip; mv.currentTime = 0; if (playing) mv.play(); } } },
      names.map(n => h('option', { value: n, selected: n === clip }, n))) : h('span.dim.i3d-animname', clip);
    box.replaceChildren(pp, pick);
    box.hidden = false;
  };
  const render = () => {
    const a = cur;
    ed?.destroy(); ed = null; edBox.hidden = true; edBox.replaceChildren();
    if (edSt.id !== a?.id) edSt = { id: a?.id || null, log: [], redo: [], sel: null };
    if (!a) { box.replaceChildren(S.empty('cuboid', 'No mesh yet', emptyText)); bar.replaceChildren(); stats.replaceChildren(); return; }
    const animBox = h('span.i3d-anim', { hidden: true });
    const boneBtn = btn('Bones', { size: 'sm', kind: bones ? 'primary' : '', title: 'Show the skeleton (follows the animation)', onClick: () => { bones = !bones; if (!bones) editing = false; S.prefs.set('i3d.bones', bones); render(); } });
    boneBtn.hidden = true; boneBtn.setAttribute('aria-pressed', String(bones));
    const boneInfo = S.uiInfo('bones'), editInfo = S.uiInfo('edit_rig');
    boneInfo.hidden = true; editInfo.hidden = true;
    const editBtn = btn('Edit rig', { size: 'sm', icon: 'pencil', kind: editing ? 'primary' : '', title: 'Edit the skeleton: move, rename, delete or add bones, then re-skin',
      onClick: () => { editing = !editing; render(); } });
    editBtn.hidden = true; editBtn.setAttribute('aria-pressed', String(editing));
    box.replaceChildren(h('div.empty', ic('loader', 22, 'spin'), h('p', 'Loading the 3D viewer…')));
    const hooks = viewerHooks.map(f => { try { return f(a, render); } catch { return null; } }).filter(Boolean);
    S.loadModelViewer().then(async () => {
      if (cur !== a) return;
      let alt = null;                                  // a hook may swap the model (engine export: another LOD, hull overlay)
      for (const hk of hooks) { if (hk.src && !alt) alt = await hk.src(); }
      if (cur !== a) return;
      const mv = h('model-viewer', { src: alt || srcOf(a), 'camera-controls': true, 'auto-rotate': (rotate && !editing) || null, 'shadow-intensity': '0.8',
        exposure: light === 'bright' ? '1.4' : light === 'dark' ? '0.7' : '1', 'environment-image': 'neutral', 'interaction-prompt': 'none', alt: a.name || '3D model', 'touch-action': 'pan-y',
        autoplay: true });           // a GLB with animation clips plays the first one; static meshes are unaffected
      const t0a = Date.now();      // poll like the wireframe overlay below: the 'load' event is unreliable for a re-created element
      const tickAnim = () => { if (cur !== a || !mv.isConnected || Date.now() - t0a > 120000) return; if (mv.loaded) animControls(mv, animBox); else setTimeout(tickAnim, 150); };
      setTimeout(tickAnim, 150);
      let skel = false;
      const tb0 = Date.now();      // skeleton: show the Bones button when the GLB has one, draw the overlay when it is on
      const tickBones = () => { if (cur !== a || !mv.isConnected || Date.now() - tb0 > 120000) return;
        const sk = mv.loaded && findSkeleton(mv); if (!sk) return setTimeout(tickBones, 150);
        if (!sk.bones.length) return;
        boneBtn.hidden = false; boneInfo.hidden = false; skel = true;
        editBtn.hidden = !bones || !/\.glb$/i.test(a.main || ''); editInfo.hidden = editBtn.hidden;
        if (bones && !applyBones(mv, sk)) return setTimeout(tickBones, 150);
        if (bones && editing && !ed && !editBtn.hidden) {      // the editor needs the Bones overlay; it shows the skeleton at rest
          playing = false; try { mv.pause(); mv.currentTime = 0; } catch { /* no animation */ }
          ed = rigEditor({ asset: a, mv, host: edBox, overlayHost: box, scene: sk.scene, root: sk.root, st: edSt, runJob,
            onClose: () => { editing = false; render(); } });
        } };
      setTimeout(tickBones, 150);
      if (mode === 'edges' || mode === 'wire') {   // poll: model-viewer's 'load' event is not reliable for a re-created element
        const view = mode, t0 = Date.now();
        const tick = () => { if (cur !== a || !mv.isConnected || Date.now() - t0 > 120000) return;
          if (mv.loaded && applyWire(mv, view)) return; setTimeout(tick, 150); };
        setTimeout(tick, 150);
      }
      mv.addEventListener('load', () => setTimeout(async () => {   // the visible viewer doubles as the thumbnail camera
        if (mode === 'edges' || mode === 'wire' || (bones && skel) || alt) return;   // don't save a wireframe / skeleton / hull overlay as the thumbnail
        if (a.preview?.thumb || thumbSeen.has(a.id) || cur !== a) return;
        thumbSeen.add(a.id);
        try {
          const anim = mv.availableAnimations?.length > 0;
          if (anim) { mv.pause(); mv.currentTime = 0; await new Promise(r => setTimeout(r, 150)); }   // the thumbnail shows the rest pose
          const blob = await mv.toBlob({ mimeType: 'image/webp', qualityArgument: 0.9 });
          if (anim && playing) mv.play();
          await S.api.setThumb(a.id, blob);
        } catch { thumbSeen.delete(a.id); autoThumb(a); }
      }, 700), { once: true });
      box.replaceChildren(mv);
      el._mv = mv;
    }).catch(() => box.replaceChildren(h('div.empty', 'The 3D viewer could not be loaded.')));
    const hookStats = hooks.map(hk => hk.stats).find(Boolean);           // e.g. the engine export: the shown LOD's triangles, not every hidden LOD and hull in the file
    const I = a.info || {}, RT = a.worker?.retopo;     // a retopo result: quads are the unit, not triangles
    stats.replaceChildren(h('span.ellipsis', { style: { fontWeight: 600, color: 'var(--text)' } }, a.name || a.id),
      RT ? h('span', `${RT.faces.toLocaleString()} polys (${RT.quad_pct}% quads) · ${RT.triangles.toLocaleString()} tris · ${RT.vertices.toLocaleString()} verts · ${RT.tex_size}² maps`)
        : hookStats ? h('span', hookStats) : I.faces ? h('span', `${I.faces.toLocaleString()} faces`) : null,
      !RT && !hookStats && I.vertices ? h('span', `${I.vertices.toLocaleString()} verts`) : null,
      I.textures != null ? h('span', `${I.textures} texture(s)`) : null, a.worker?.stats?.time?.total ? h('span', `made in ${S.fmtDur(a.worker.stats.time.total)}`) : null);
    bar.replaceChildren(
      S.seg([{ value: 'textured', label: 'Textured' }, hasWhite(a) ? { value: 'white', label: 'Shape' } : null,
        { value: 'edges', label: 'Edges' }, { value: 'wire', label: 'Wireframe' }].filter(Boolean), mode, v => { mode = v; render(); }),
      S.uiInfo('viewer_mode'),
      btn('', { size: 'sm', icon: rotate ? 'pause' : 'refresh-cw', title: rotate ? 'Stop rotating' : 'Auto-rotate', onClick: () => { rotate = !rotate; S.prefs.set('i3d.rotate', rotate); render(); } }),
      btn('', { size: 'sm', icon: 'sun', title: `Light: ${light}`, onClick: () => { light = { neutral: 'bright', bright: 'dark', dark: 'neutral' }[light]; S.prefs.set('i3d.light', light); render(); } }),
      animBox,
      boneBtn, boneInfo,
      editBtn, editInfo,
      ...hooks.flatMap(hk => hk.bar || []),
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
