// Retopo + bake UI of the 3D studio (not a studio module: it lives in i3d/): the source-info card under the mesh slot and the
// synced before / after compare modal. The pipeline itself is workers/retopo_worker.py (docs/research/retopo.md).
import * as S from '../../core/sdk.js';
import { applyWire } from './common.js';
const { h, ic, btn } = S;

export const E_RIGGED = 'This model is rigged (it has a skeleton). Retopology rebuilds every vertex, so the skin and its animations would be lost. Run Retopo on the unrigged model (its source in the lineage), then Rig the result.';

const isRigged = a => !!(a?.worker?.rigged || a?.info?.rigged);
const hasAnim = a => !!((a?.worker?.animations || []).length || (a?.info?.animations || []).length);

/** Measured on the DGX Spark (20 CPU cores) for a 550k-triangle character: wall time and peak RAM of the bake, by texture size. */
const EST = { 512: [5, 0.9], 1024: [6, 1.0], 2048: [14, 1.3], 4096: [41, 2.1] };
export function estimate(texSize, inputTris = 0) {
  const [s, gb] = EST[texSize] || EST[2048];
  const extraM = Math.max(0, inputTris - 550000) / 1e6;
  return { seconds: Math.round(s + 4 * extraM), gb: Math.round((gb + 0.3 * extraM) * 10) / 10 };
}
/** Facts about a source asset from its meta only (no GLB is fetched). */
export function sourceFacts(a) {
  const files = a?.files || [];
  const whiteFile = files.some(f => f.name === 'white_mesh.glb');
  const whiteFaces = a?.worker?.stats?.white?.faces || 0;
  const tris = a?.info?.faces || 0;
  return { tris, whiteFile, whiteFaces, dense: Math.max(tris, whiteFaces), textures: a?.info?.textures || 0 };
}
const fmtN = n => (n == null ? '-' : Number(n).toLocaleString());

/** Small card under the mesh slot: what the source offers (triangles, stored shape mesh, colour) and the estimated time / RAM.
 *  Returns {el, update}; call update() whenever the slot or a setting changes. Rigged input: warning + "use the unrigged source". */
export function sourcePanel(kit) {
  const el = h('div.rt-src', { hidden: true });
  let seq = 0;
  const update = async () => {
    const slot = kit.slots?.mesh;
    const id = slot?.ids?.[0], a = id && slot.metas.get(id);
    const mine = ++seq;
    if (!a) { el.hidden = true; return; }
    el.hidden = false;
    const f = sourceFacts(a);
    const p = kit.form?.allValues?.() || {};
    const e = estimate(Number(p.tex_size) || 2048, f.dense);
    const rows = [];
    rows.push(h('div.rt-row', h('span.rt-k', 'Source'),
      h('span', `${f.tris ? `${fmtN(f.tris)} triangles` : 'triangle count unknown'} · shape mesh: ${f.whiteFile ? (f.whiteFaces ? `${fmtN(f.whiteFaces)} triangles` : 'yes') : 'none'} · colour: ${f.textures ? 'texture' : 'vertex colours or none'}`)));
    rows.push(h('div.rt-row', h('span.rt-k', 'Estimate'), h('span', `≈ ${e.seconds} s · ≈ ${e.gb} GB RAM`), h('span.dim', ' on this machine (20 CPU cores)')));
    if (f.dense && f.dense < 45000) rows.push(h('div.rt-row.rt-warn', ic('triangle-alert', 14), h('span', `Only ${fmtN(f.dense)} triangles: retopo needs a denser model than the target (about 4x the face target). Lower the target or use the original dense shape.`)));
    if (isRigged(a) || hasAnim(a)) {
      const row = h('div.rt-row.rt-warn', ic('triangle-alert', 14), h('span', isRigged(a) ? 'Rigged: retopology would drop the skeleton.' : 'Animated: retopology would drop the animation.'));
      rows.push(row);
      const pid = a.parents?.[0];
      if (pid) {
        try {
          const par = await S.api.asset(pid);
          if (mine !== seq) return;
          if (par?.type === 'mesh' && !isRigged(par) && !hasAnim(par)) {
            row.append(btn('Use the unrigged source', { size: 'sm', icon: 'rotate-ccw', onClick: async () => { await slot.set([par.id]); } }));
          }
        } catch { /* the parent is gone */ }
      }
    }
    if (mine === seq) el.replaceChildren(...rows);
  };
  el.update = update;
  setTimeout(update, 0);
  return el;
}

/** Hook the panel into the TaskStudio: update on slot change and on every settings / model change (kit._eta runs for those). */
export function attachSourcePanel(kit) {
  const panel = sourcePanel(kit);
  const slot = kit.slots.mesh;
  const prev = slot.onChange;
  slot.onChange = ids => { prev?.(ids); panel.update(); };
  if (!kit._retopoEtaHooked) {
    kit._retopoEtaHooked = true;
    const eta = kit._eta.bind(kit);
    kit._eta = () => { eta(); kit._retopoPanel?.update(); };
  }
  kit._retopoPanel = panel;
  slot.el.after(panel);       // right under the mesh slot (extraPanel itself lands below the settings)
  return panel;
}

/** beforeSubmit for the retopo task: refuse a rigged input, set the job title. */
export function retopoBeforeSubmit(spec, kit) {
  const id = spec.inputs.find(i => i.role === 'mesh')?.asset;
  const a = kit.slots.mesh?.metas.get(id);
  if (isRigged(a) || hasAnim(a)) { S.toast(E_RIGGED, { kind: 'err', timeout: 9000 }); return null; }
  const p = spec.params;
  spec.title = `Retopo ${Math.round((p.target_faces || 12000) / 1000)}k · ${p.tex_size || 2048}`.slice(0, 60);
  return spec;
}

// ------------------------------------------------------------------ before / after compare
const CLAY = [0.8, 0.8, 0.8, 1];
/** Two synced <model-viewer>s: the source ("Before") and the retopo result ("After"). Textured | Clay | Edges, normal map switch. */
export async function retopoCompare(srcRef, a) {
  let src;
  try { src = typeof srcRef === 'string' ? await S.api.asset(srcRef) : srcRef; } catch (e) { S.toastError(e); return; }
  try { await S.loadModelViewer(); } catch (e) { S.toastError(e); return; }
  const R = a.worker?.retopo || {};
  const hasWhite = (src.files || []).some(f => f.name === 'white_mesh.glb');
  let mode = 'textured', normalOn = true, cam = null, syncing = false, gen = 0;

  const paneB = h('div.rt-pane'), paneA = h('div.rt-pane');
  const srcUrl = () => (mode !== 'textured' && hasWhite ? S.fileUrl(src, 'white_mesh.glb') : S.fileUrl(src));
  const mkMv = (url, label) => h('model-viewer', { src: url, 'camera-controls': true, 'shadow-intensity': '0.8', exposure: '1', 'environment-image': 'neutral',
    'interaction-prompt': 'none', alt: label, 'touch-action': 'pan-y', 'camera-orbit': cam?.orbit || null, 'camera-target': cam?.target || null, 'field-of-view': cam?.fov || null });
  const label = (t, sub) => h('div.rt-label', h('span.badge.accent', t), sub ? h('span.dim.ellipsis', sub) : null);

  /** After a model has loaded: clay material, normal switch, edges overlay. */
  const finish = (mv, g, { isAfter }) => {
    const t0 = Date.now();
    const tick = () => {
      if (g !== gen || !mv.isConnected || Date.now() - t0 > 120000) return;
      if (!mv.loaded) return setTimeout(tick, 150);
      try {
        for (const m of mv.model?.materials || []) {
          if (mode === 'clay') {
            m.pbrMetallicRoughness.baseColorTexture?.setTexture?.(null);
            m.pbrMetallicRoughness.setBaseColorFactor(CLAY);
            m.pbrMetallicRoughness.setMetallicFactor(0);
          }
          if (isAfter && !normalOn) m.normalTexture?.setTexture?.(null);
        }
      } catch { /* a model without materials */ }
      if (mode === 'edges' && !applyWire(mv, 'edges')) return setTimeout(tick, 150);
    };
    setTimeout(tick, 100);
  };
  const link = (from, to) => from.addEventListener('camera-change', e => {
    if (syncing || e.detail?.source !== 'user-interaction') return;
    syncing = true;
    try {
      const o = from.getCameraOrbit(), tg = from.getCameraTarget(), fov = from.getFieldOfView();
      cam = { orbit: `${o.theta}rad ${o.phi}rad ${o.radius}m`, target: `${tg.x}m ${tg.y}m ${tg.z}m`, fov: `${fov}deg` };
      to.cameraOrbit = cam.orbit; to.cameraTarget = cam.target; to.fieldOfView = cam.fov;
      to.jumpCameraToGoal?.();
    } catch { /* not ready */ }
    syncing = false;
  });
  const build = () => {
    const g = ++gen;
    const mb = mkMv(srcUrl(), 'Before'), ma = mkMv(S.fileUrl(a), 'After');
    paneB.replaceChildren(label('Before', mode !== 'textured' && hasWhite ? 'dense shape mesh' : src.name), mb);
    paneA.replaceChildren(label('After', a.name), ma);
    link(mb, ma); link(ma, mb);
    finish(mb, g, { isAfter: false }); finish(ma, g, { isAfter: true });
  };

  const normalSw = S.toggle(true, v => { normalOn = v; build(); }, 'Normal map');
  const normalRow = h('label.rt-sw', normalSw, h('span', 'Normal map'), S.uiInfo('retopo_normal_toggle'));
  const modeSeg = S.seg([{ value: 'textured', label: 'Textured' }, { value: 'clay', label: 'Clay' }, { value: 'edges', label: 'Edges' }], mode, v => { mode = v; build(); });
  const head = h('div.rt-head', modeSeg, S.uiInfo('retopo_clay'), h('span.rt-spacer'), normalRow, S.uiInfo('retopo_compare'));

  const I = src.info || {}, J = a.info || {};
  const wdur = x => (x?.worker?.stats?.time?.total ? S.fmtDur(x.worker.stats.time.total) : '-');
  const rows = [
    ['Polys', '-', fmtN(R.faces)], ['Quads', '-', R.quad_pct != null ? `${R.quad_pct}%` : '-'],
    ['Triangles', fmtN(I.faces), fmtN(R.triangles ?? J.faces)], ['Vertices', fmtN(I.vertices), fmtN(R.vertices ?? J.vertices)],
    ['Textures', fmtN(I.textures), `${fmtN(J.textures)}${R.tex_size ? ` · ${R.tex_size}²` : ''}`],
    ['File size', I.bytes != null ? S.fmtBytes(I.bytes) : '-', J.bytes != null ? S.fmtBytes(J.bytes) : '-'], ['Made in', wdur(src), wdur(a)],
  ];
  const table = h('div.rt-table', h('span'), h('b', 'Before'), h('b', 'After'),
    ...rows.flatMap(([k, b, af]) => [h('span.rt-k', k), h('span', b), h('span', af)]));
  const notes = [];
  if (R.dist_mean_frac != null) notes.push(h('div.rt-fid', `Surface distance: mean ${(R.dist_mean_frac * 100).toFixed(3)}%, worst 1 % ${((R.dist_p99_frac ?? 0) * 100).toFixed(2)}% of the model size.`,
    R.high?.file ? h('span.dim', ` Detail source: ${R.high.file} (${fmtN(R.high.faces)} triangles).`) : null));
  const warns = a.worker?.warnings || [];
  if (warns.length) notes.push(h('ul.rt-warns', warns.map(w => h('li', w))));

  const body = h('div.rt-cmp', head, h('div.rt-panes', paneB, paneA), table, ...notes);
  const md = S.modal({ title: 'Retopo: before / after', icon: 'columns-2', width: 1100, body, onClose: () => { gen++; } });
  md.el.classList.add('rt-modal');
  build();
  return md;
}
