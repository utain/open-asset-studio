// Game-engine export UI of the 3D studio (Export step > "Game engine"; not a studio module: it lives in i3d/).
//   targetCard   what the chosen engine will get, live from the form (before the job runs)
//   resultPanel  after a job: LOD table, collision, textures, warnings, licence, Download package, Deliver to a game project
//   viewer hook  LOD switcher and collision-hull overlay for a mesh asset with meta.worker.engine (client-side edit of mesh.glb)
// The pipeline itself is workers/engine_worker.py (docs/research/engine_export.md); delivery is app/plugins/engine.py (docs/CONTRACTS.md section 12).
import * as S from '../../core/sdk.js';
import { viewerHooks } from './common.js';
const { h, ic, btn } = S;

const fmtN = n => (n == null ? '-' : Number(n).toLocaleString());
const isRigged = a => !!(a?.worker?.rigged || a?.info?.rigged);
const hasAnim = a => !!((a?.worker?.animations || []).length || (a?.info?.animations || []).length);
export const engineOf = a => (a?.type === 'mesh' && a?.worker?.engine && typeof a.worker.engine === 'object' ? a.worker.engine : null);
const TARGET_LABEL = { unreal: 'Unreal Engine', unity: 'Unity', godot: 'Godot', gltf: 'glTF / web' };

// ------------------------------------------------------------------ before the job: what you get
const COLLISION = {
  auto: (rigged, p) => (rigged ? 'one capsule (reported in the manifest; skeletal meshes carry no UCX)' : `convex collision pieces (up to ${p.max_hulls || 16})`),
  none: () => 'no collision', box: () => 'one box', obb: () => 'one oriented box', capsule: () => 'one capsule',
  convex_hull: () => 'one convex hull', decomposition: (_r, p) => `convex collision pieces (up to ${p.max_hulls || 16})`,
};
/** Plain-words summary of the package for the current form values. rigged: the first input is a rigged / animated mesh. */
export function describeTarget(p, rigged) {
  const t = p.target || 'unreal', lods = Number(p.lods) || 0;
  const col = (COLLISION[p.collision] || COLLISION.auto)(rigged, p);
  const lodTxt = t === 'godot' ? 'no LODs (Godot builds its own on import)' : lods ? `${lods} LOD level${lods > 1 ? 's' : ''}` : 'no LODs';
  const size = p.size_mode === 'height' ? `scaled to ${p.height_m} m tall` : 'size unchanged';
  const tex = p.tex_max && p.tex_max !== 'keep' ? `, textures up to ${p.tex_max}px` : '';
  const rows = [];
  if (rigged) {
    rows.push(['Files', 'GLB for every engine: skin, skeleton and clips are kept' + (t === 'unreal' || t === 'unity' ? '; each LOD is a separate _LODn.glb' : t === 'gltf' ? ' (MSFT_lod levels inside)' : '')]);
    rows.push(['LODs', lodTxt]);
    rows.push(['Collision', col]);
  } else if (t === 'unreal') {
    rows.push(['Files', `FBX (centimetres, Y up) with ${lodTxt}; SM_ names, UCX_ collision`]);
    rows.push(['Collision', col]);
    rows.push(['Textures', 'T_ files: _BC, _N (DirectX normal), _ORM (occlusion / roughness / metallic)']);
  } else if (t === 'unity') {
    const pipe = { urp: 'URP: _MetallicSmoothness + _Occlusion', hdrp: 'HDRP: _MaskMap', both: 'URP and HDRP maps' }[p.unity_pipeline] || 'URP and HDRP maps';
    rows.push(['Files', `FBX with ${lodTxt}, collision hull meshes and an optional editor script`]);
    rows.push(['Collision', col]);
    rows.push(['Textures', `_BaseMap, _Normal (OpenGL), ${pipe}`]);
  } else if (t === 'godot') {
    rows.push(['Files', 'GLB with ' + lodTxt]);
    rows.push(['Collision', col === 'no collision' ? col : `${col}, named -convcolonly`]);
    rows.push(['Textures', '_albedo, _normal, _orm (occlusion / roughness / metallic)']);
  } else {
    rows.push(['Files', `GLB with ${lodTxt} (MSFT_lod) and a separate collision GLB`]);
    rows.push(['Collision', col]);
    rows.push(['Textures', 'embedded in the GLB, also as plain PNG files']);
  }
  rows.push(['Placement', `${size}, pivot ${String(p.pivot || 'bottom_center').replace('_', ' ')}${p.yaw ? `, turned ${p.yaw}°` : ''}${tex}`]);
  return rows;
}

/** The "What you get" card under the settings. Returns {el, update}; update() re-reads the form and the mesh slot. */
export function targetCard(kit) {
  const el = h('div.eng-card');
  const update = () => {
    const slot = kit.slots?.mesh;
    const first = slot && slot.metas.get(slot.ids?.[0]);
    const rigged = isRigged(first) || hasAnim(first);
    const p = kit.form?.allValues?.() || {};
    const rows = describeTarget(p, rigged);
    el.replaceChildren(
      h('div.eng-title', ic('gamepad-2', 15), h('b', TARGET_LABEL[p.target] || 'Game engine'), S.uiInfo('engine_target')),
      ...rows.map(([k, v]) => h('div.rt-row', h('span.rt-k', k), h('span', v))),
      rigged && p.collision && !['auto', 'none', 'capsule'].includes(p.collision)
        ? h('div.rt-row.rt-warn', ic('triangle-alert', 14), h('span', 'A rigged mesh gets a capsule; the other collision shapes are for static props.')) : null);
  };
  el.update = update;
  return el;
}

// ------------------------------------------------------------------ delivery
let destCache = null;
async function destinations(force = false) {
  if (destCache && !force) return destCache;
  try { destCache = await S.api.get('/api/x/engine/destinations'); } catch { destCache = { destinations: [], local: false }; }
  return destCache;
}

function deliverRow(a) {
  const box = h('div.eng-deliver', { hidden: true });
  const E = engineOf(a);
  if (!E) return box;
  destinations().then(d => {
    const list = (d.destinations || []).filter(x => x.engine === 'any' || x.engine === E.target);
    if (!list.length) return;
    let cur = list[0];
    const sel = h('select.select', { 'aria-label': 'Game project', on: { change: e => { cur = list.find(x => x.id === e.target.value) || list[0]; draw(); } } },
      list.map(x => h('option', { value: x.id }, x.label)));
    const owBox = h('input', { type: 'checkbox', 'aria-label': 'Replace earlier delivery' });
    const note = h('div.dim.eng-note');
    const go = btn('Deliver', { icon: 'send', kind: 'primary', size: 'sm', onClick: () => deliver(false) });
    const where = x => `${x.subdir}/${E.name}/`;
    const draw = () => {
      go.disabled = !cur.can_deliver;
      note.textContent = cur.can_deliver
        ? `Copies the files into ${where(cur)} of "${cur.label}"${cur.path ? ` (${cur.path})` : ''}. Nothing is deleted; files you edited are kept.`
        : 'Deliver works only on the computer the studio runs on (this destination is local-only).';
    };
    const run = async overwrite => {
      go.disabled = true;
      try {
        const r = await S.api.post('/api/x/engine/deliver', { asset_id: a.id, destination: cur.id, overwrite });
        const n = r.written.length, u = r.unchanged.length;
        S.toast(n ? `Delivered ${n} file${n > 1 ? 's' : ''} to ${r.relpath}/${u ? ` (${u} already up to date)` : ''}.` : `Already up to date in ${r.relpath}/.`, { kind: 'ok', title: cur.label });
        a.worker = { ...a.worker, engine: { ...E, deliveries: [...(E.deliveries || []), { destination: cur.id, at: new Date().toISOString() }] } };
      } catch (e) {
        if (e.data?.code === 'exists' && !overwrite) {
          const ok = await S.confirmDialog(`${e.message}\n\nReplace the files the studio delivered earlier? Files you edited are kept.`, { title: 'Replace earlier delivery?', ok: 'Replace' });
          if (ok) { go.disabled = false; return run(true); }
        } else S.toastError(e, 'Delivery refused');
      } finally { draw(); }
    };
    const deliver = async overwrite => {
      const ok = await S.confirmDialog(`Copy the ${TARGET_LABEL[E.target] || 'engine'} package of "${E.name}" into the game project "${cur.label}"?\n\nFolder: ${where(cur)}${cur.path ? `\nProject: ${cur.path}` : ''}\n\nNew files are added; files from an earlier delivery are replaced only if you tick "Replace earlier delivery" and you did not edit them. Nothing is deleted.`,
        { title: 'Deliver to game project', ok: 'Deliver' });
      if (ok) run(overwrite || owBox.checked);
    };
    draw();
    box.replaceChildren(
      h('div.eng-row', h('span.eng-k', ic('send', 14), 'Deliver to', S.uiInfo('engine_deliver')), list.length > 1 ? sel : h('b', list[0].label), go),
      h('label.eng-ow', owBox, h('span', 'Replace earlier delivery'), S.uiInfo('engine_overwrite')), note);
    box.hidden = false;
  });
  return box;
}

// ------------------------------------------------------------------ after the job: report
const reportCache = new Map();
async function fetchReport(a) {
  const k = `${a.id}:${a.updated || ''}`;
  if (reportCache.has(k)) return reportCache.get(k);
  let r = null;
  try { const res = await fetch(S.fileUrl(a, 'engine_report.json')); if (res.ok) r = await res.json(); } catch { /* optional */ }
  reportCache.set(k, r);
  return r;
}

/** Report card of a finished engine export (mesh asset with meta.worker.engine). Returns {el, set(asset|null)}. */
export function resultPanel() {
  const el = h('div.eng-result', { hidden: true });
  let seq = 0;
  const set = async a => {
    const E = engineOf(a), mine = ++seq;
    if (!E) { el.hidden = true; el.replaceChildren(); return; }
    el.hidden = false;
    const lodRows = (E.lods || []).map((l, i) => [`LOD${i}`, fmtN(l.tris), `${Math.round((l.ratio ?? 1) * 100)}%`, i ? (l.error != null ? `${(l.error * 100).toFixed(2)}%` : '-') : '-', l.screen_size != null ? l.screen_size : '-']);
    const warns = a.worker?.warnings || [];
    const pkg = E.package, pkgFile = (a.files || []).find(f => f.name === pkg);
    const body = [
      h('div.eng-title', ic('list-checks', 15), h('b', `${TARGET_LABEL[E.target] || E.target} package`), h('span.dim', E.name), S.uiInfo('engine_report'), h('span.eng-sp'), S.ui.licenseBadge(a.license)),
      h('div.eng-table', h('b', 'Level'), h('b', 'Triangles'), h('b', 'Of LOD0'), h('b', 'Shape error'), h('b', 'Screen size'),
        ...lodRows.flatMap(r => r.map(c => h('span', c)))),
      h('div.rt-row', h('span.rt-k', 'Collision'), h('span', E.collision === 'none' || !E.collision ? 'none' : `${E.collision}${E.hulls ? ` · ${E.hulls} piece${E.hulls > 1 ? 's' : ''}` : ''}${E.volume_ratio != null ? ` · fills ${Math.round(E.volume_ratio * 100)}% of the model` : ''}`)),
      h('div.rt-row', h('span.rt-k', 'Files'), h('span', `${(E.files || []).length} in the package${E.textures ? `, ${E.textures} texture map(s)` : ''}`)),
      warns.length ? h('ul.rt-warns', warns.map(w => h('li', w))) : null,
    ];
    const tex = h('div.eng-tex');
    const dl = h('div.eng-actions',
      pkgFile ? btn(`Download package${pkgFile.bytes ? ` (${S.fmtBytes(pkgFile.bytes)})` : ''}`, { icon: 'download', size: 'sm', title: pkg, onClick: () => { location.href = S.fileUrl(a, pkg, true); } }) : null,
      S.uiInfo('engine_download'),
      (a.files || []).some(f => f.name === 'engine_report.json') ? btn('Report', { icon: 'list-checks', size: 'sm', kind: 'ghost', title: 'engine_report.json', onClick: () => window.open(S.fileUrl(a, 'engine_report.json'), '_blank') }) : null);
    el.replaceChildren(...body, tex, dl, deliverRow(a));
    const rep = await fetchReport(a);
    if (mine !== seq || !rep) return;
    const rows = (rep.textures || []).map(t => [t.file, t.packing || t.slot, t.size ? `${t.size[0]}×${t.size[1]}` : '-']);
    if (rows.length) tex.replaceChildren(h('div.eng-sub', 'Textures'), h('div.eng-table.eng-table3', ...rows.flatMap(r => r.map(c => h('span', c)))));
  };
  return { el, set };
}

/** Wire the card + result panel into the TaskStudio (extraPanel of the engine_export task). Returns the container element. */
export function enginePanel(kit, viewer) {
  const card = targetCard(kit), result = resultPanel();
  const root = h('div.eng-panel', card, result.el);
  const slot = kit.slots.mesh, prev = slot.onChange;
  slot.onChange = ids => { prev?.(ids); card.update(); };
  if (!kit._engEtaHooked) {
    kit._engEtaHooked = true;
    const eta = kit._eta.bind(kit);
    kit._eta = () => { eta(); kit._engCard?.update(); };
  }
  kit._engCard = card;
  kit._engResult = result;
  // project defaults (Settings > Game engine): pre-fill the engine and the character height once, unless the user already chose them
  const pe = S.store.effective?.engine || {};
  const mem = kit.form?.mem || {};
  const pre = {};
  if (pe.target && mem.target === undefined && kit.form.vals?.target !== pe.target) pre.target = pe.target;
  if (pe.height_m && mem.height_m === undefined && kit.form.vals?.height_m !== pe.height_m) pre.height_m = pe.height_m;
  if (Object.keys(pre).length) setTimeout(() => kit.form.setValues(pre, { silent: false }), 0);
  setTimeout(() => { card.update(); const a = viewer?.current?.(); if (a) result.set(a); }, 0);
  return root;
}

export function engineBeforeSubmit(spec, kit) {
  const p = spec.params, n = spec.inputs.length;
  const first = kit.slots.mesh?.metas.get(spec.inputs.find(i => i.role === 'mesh')?.asset);
  p.batch = n;
  spec.title = `Game engine · ${TARGET_LABEL[p.target] || p.target} · ${n > 1 ? `${n} meshes` : (p.name || first?.name || 'mesh')}`.slice(0, 60);
  return spec;
}

// ------------------------------------------------------------------ viewer: LOD switcher + collision hulls (client-side GLB edit)
const view = new Map();                       // asset id -> {lod, hulls}: kept while the viewer element is re-created
const HULL_RGB = [[0.95, 0.45, 0.20], [0.20, 0.65, 0.95], [0.40, 0.80, 0.35], [0.85, 0.30, 0.70], [0.95, 0.80, 0.20], [0.35, 0.40, 0.95]];
let baseCache = { key: null, buf: null }, lastUrl = null;

/** The GLB with LOD `lod` shown on the LOD0 node (swap of the mesh index) and, if `hulls`, the hull nodes added to the scene with translucent materials. */
export function editGlb(buf, { lod = 0, hulls = false } = {}) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546C67) return null;
  const jl = dv.getUint32(12, true);
  const j = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl)));
  const oas = j.extras?.oas_engine;
  if (!oas) return null;
  const bin = new Uint8Array(buf, 20 + jl);                                    // the rest: BIN chunk header + data, copied as it is
  if (lod > 0 && oas.lods?.[lod - 1] != null) {
    const a = j.nodes[oas.lod0], b = j.nodes[oas.lods[lod - 1]];
    a.mesh = b.mesh;
  }
  if (hulls && oas.hulls?.length) {
    j.materials = j.materials || [];
    const scene = j.scenes[j.scene || 0];
    oas.hulls.forEach((nid, k) => {
      const [r, g, b] = HULL_RGB[k % HULL_RGB.length];
      j.materials.push({ name: `hull_${k}`, alphaMode: 'BLEND', doubleSided: true, pbrMetallicRoughness: { baseColorFactor: [r, g, b, 0.42], metallicFactor: 0, roughnessFactor: 1 } });
      const prim = j.meshes[j.nodes[nid].mesh].primitives[0];
      prim.material = j.materials.length - 1;
      if (oas.root != null && j.nodes[oas.root]) (j.nodes[oas.root].children = j.nodes[oas.root].children || []).push(nid);
      else scene.nodes.push(nid);
    });
  }
  let js = new TextEncoder().encode(JSON.stringify(j));
  if (js.length % 4) { const p = new Uint8Array(js.length + (4 - js.length % 4)).fill(0x20); p.set(js); js = p; }
  const out = new Uint8Array(12 + 8 + js.length + bin.length);
  const od = new DataView(out.buffer);
  od.setUint32(0, 0x46546C67, true); od.setUint32(4, 2, true); od.setUint32(8, out.length, true);
  od.setUint32(12, js.length, true); od.setUint32(16, 0x4E4F534A, true);
  out.set(js, 20); out.set(bin, 20 + js.length);
  return out;
}

async function viewerSrc(a, st) {
  if (!st.lod && !st.hulls) return null;
  const key = `${a.id}:${a.updated || ''}`;
  if (baseCache.key !== key) baseCache = { key, buf: await (await fetch(S.fileUrl(a))).arrayBuffer() };
  const out = editGlb(baseCache.buf, st);
  if (!out) return null;
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(new Blob([out], { type: 'model/gltf-binary' }));
  return lastUrl;
}

/** meshViewer hook (see viewerHooks in common.js): the LOD segmented control and the Hulls toggle for an engine export. */
export function engineViewerHook(a, rerender) {
  const E = engineOf(a);
  if (!E) return null;
  const st = view.get(a.id) || { lod: 0, hulls: false };
  view.set(a.id, st);
  const lods = E.lods || [];
  const bar = [];
  if (lods.length > 1) {
    bar.push(S.seg(lods.map((l, i) => ({ value: i, label: `LOD${i} · ${l.tris >= 10000 ? Math.round(l.tris / 1000) + 'k' : fmtN(l.tris)}` })), st.lod, v => { st.lod = Number(v); rerender(); }), S.uiInfo('engine_lod'));
  }
  if ((E.hulls || 0) > 0 && (a.files || []).length) {
    const b = btn('Hulls', { size: 'sm', icon: 'box', kind: st.hulls ? 'primary' : '', title: 'Show the collision pieces over the model', onClick: () => { st.hulls = !st.hulls; rerender(); } });
    b.setAttribute('aria-pressed', String(st.hulls));
    bar.push(b, S.uiInfo('engine_hulls'));
  }
  const cur = lods[st.lod] || lods[0];
  const stats = cur ? `${fmtN(cur.tris)} triangles (${cur.name || 'LOD' + st.lod})${st.hulls && E.hulls ? ` + ${E.hulls} collision piece${E.hulls > 1 ? 's' : ''}` : ''}` : null;
  return { bar, stats, src: () => viewerSrc(a, st).catch(() => null) };
}
viewerHooks.push(engineViewerHook);

// ------------------------------------------------------------------ Library: "Export for engine…" in the multi-select bar
S.registerSelectionAction({
  id: 'engine-export', label: 'Export for game engine…', icon: 'gamepad-2', types: ['mesh'], min: 1, max: 12,
  run: ids => S.sendTo('3d', ids, { action: 'engine' }),
});
