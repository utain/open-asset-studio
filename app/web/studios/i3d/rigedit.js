// Manual rig editor of the 3D viewer ("Edit rig", shown while the Bones overlay is on).
//
// The skeleton comes from GET /api/x/rig/<asset>/skeleton (rest-pose joints in glTF space: metres, +Y up, the character
// faces +Z, its left is +X). Joints are drawn as dots over the model-viewer (projected through the viewer's own three.js
// camera); click a dot to select it, drag it on the screen plane or use the nudge buttons to move it, rename it, delete it,
// add a child. Every change is an entry of an undo log; "Apply" compiles the log into the ops of the CPU tool `rig.edit`
// (workers/rig_edit.py) and runs it, which writes a NEW mesh asset. New bones have no skin weight: "Re-skin" runs task `reskin`.
// Existing joints are always referenced by their index in the input skeleton (stable across ops), new ones by name.
import * as S from '../../core/sdk.js';
const { h, ic, btn } = S;

const r4 = v => Math.round(v * 1e4) / 1e4;
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len3 = a => Math.hypot(a[0], a[1], a[2]);
// character-relative directions (+Y up, +Z front, +X the character's left)
const DIRS = [['Up', [0, 1, 0]], ['Down', [0, -1, 0]], ['Left', [1, 0, 0]], ['Right', [-1, 0, 0]], ['Front', [0, 0, 1]], ['Back', [0, 0, -1]]];

/** Replay an edit log on the skeleton `info` (describe() of rig_edit): the joints as they look now, the compiled rig.edit ops
 *  (adjacent moves of one bone are merged) and one summary line per log entry. Pure function: the single source of truth. */
export function replay(info, log) {
  const J = new Map();
  for (const j of info.joints) J.set(`i${j.index}`, { key: `i${j.index}`, index: j.index, name: j.name, parent: j.parent == null || j.parent < 0 ? null : `i${j.parent}`, pos: [...j.position], isNew: false, role: j.role || '' });
  const kids = k => [...J.values()].filter(j => j.parent === k);
  const subtree = k => { const out = [], st = kids(k); while (st.length) { const c = st.pop(); out.push(c); st.push(...kids(c.key)); } return out; };
  const ref = j => (j.isNew ? j.name : j.index);
  const ops = [], lines = [];
  for (const e of log) {
    const j = J.get(e.key);
    if (e.type === 'add') {
      const p = J.get(e.parent); if (!p) { lines.push(null); continue; }
      ops.push({ op: 'add_bone', parent: ref(p), to: e.pos.map(r4), name: e.name });
      J.set(e.key, { key: e.key, index: null, name: e.name, parent: e.parent, pos: [...e.pos], isNew: true, role: '' });
      lines.push(`Add ${e.name} under ${p.name}`); continue;
    }
    if (!j) { lines.push(null); continue; }
    if (e.type === 'move') {
      const d = sub3(e.to, j.pos); j.pos = [...e.to];
      if (e.follow) for (const c of subtree(j.key)) c.pos = add3(c.pos, d);
      const op = { op: 'move_joint', bone: ref(j), to: e.to.map(r4), ...(e.follow ? { children: 'follow' } : {}) };
      const last = ops[ops.length - 1];
      if (last && last.op === 'move_joint' && last.bone === op.bone && last.children === op.children) ops[ops.length - 1] = op; else ops.push(op);
      lines.push(`Move ${j.name}`);
    } else if (e.type === 'rename') {
      ops.push({ op: 'rename', bone: ref(j), to: e.name }); lines.push(`Rename ${j.name} to ${e.name}`); j.name = e.name;
    } else if (e.type === 'delete') {
      ops.push({ op: 'delete_bone', bone: ref(j) }); lines.push(`Delete ${j.name}`);
      for (const c of kids(j.key)) c.parent = j.parent;
      J.delete(j.key);
    }
  }
  return { joints: [...J.values()], ops, lines };
}

/** Fetch the skeleton of an asset; resolves to {info} or {error}. */
async function loadSkeleton(asset) {
  try { return { info: await S.api.get(`/api/x/rig/${encodeURIComponent(asset.id)}/skeleton`) }; } catch (e) {
    const miss = e.status === 404 || /not found/i.test(e.message || '');
    return { error: miss ? 'The rig editor needs the newest core: restart Open Asset Studio so that /api/x/rig is available.' : (e.message || String(e)) };
  }
}

/** Mount the editor. `st` = {log, redo, sel} (kept by the viewer while it re-renders). Returns {destroy}. */
export function rigEditor({ asset, mv, host, overlayHost, scene, root, st, runJob, onClose }) {
  let info = null, alive = true, busy = false, raf = 0, live = null;     // live: {key, pos} while a dot is dragged
  const svgNS = 'http://www.w3.org/2000/svg';
  const body = h('div.rige-body', h('p.dim', ic('loader', 14, 'spin'), ' Reading the skeleton…'));
  const panel = h('div.rige', h('div.rige-head', h('b', 'Edit rig'), S.uiInfo('edit_rig'), h('span.dim.rige-asset.ellipsis', asset.name || asset.id), h('span.spacer'),
    btn('', { size: 'sm', icon: 'x', title: 'Close the rig editor', onClick: () => onClose() })), body);
  host.replaceChildren(panel); host.hidden = false;
  const svg = document.createElementNS(svgNS, 'svg'); svg.setAttribute('class', 'rige-lines');
  const layer = h('div.rige-layer', svg); overlayHost.append(layer);
  const dots = new Map(), lines = new Map();
  const cam = () => scene.getCamera?.() || scene.camera;
  const V3 = root.position.constructor, M4 = root.matrixWorld.constructor;
  const tv = new V3(), tw = new V3(), tn = new V3(), tinv = new M4();
  let cur = { joints: [], ops: [], lines: [] };

  const state = () => { cur = replay(info, st.log); if (st.sel && !cur.joints.find(j => j.key === st.sel)) st.sel = null; };
  const byKey = k => cur.joints.find(j => j.key === k);
  const posOf = j => (live && live.key === j.key ? live.pos : j.pos);
  const cm = v => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)} cm`;     // glTF units are metres (info.units is only a description)

  // ---------------------------------------------------------------- overlay: dots + lines, repositioned every frame
  const syncNodes = () => {
    const keys = new Set(cur.joints.map(j => j.key));
    for (const [k, el] of dots) if (!keys.has(k)) { el.remove(); dots.delete(k); }
    for (const [k, el] of lines) if (!keys.has(k)) { el.remove(); lines.delete(k); }
    for (const j of cur.joints) {
      let d = dots.get(j.key);
      if (!d) {
        d = h('button.rige-dot', { type: 'button', 'aria-label': j.name, on: { pointerdown: e => down(e, j.key), keydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(j.key); } } } });
        layer.append(d); dots.set(j.key, d);
      }
      d.title = j.name + (j.role ? ` (${j.role})` : '');
      d.classList.toggle('sel', j.key === st.sel); d.classList.toggle('new', j.isNew);
      if (j.parent && !lines.has(j.key)) { const l = document.createElementNS(svgNS, 'line'); svg.append(l); lines.set(j.key, l); }
      const l = lines.get(j.key); if (l) l.setAttribute('class', `${j.isNew ? 'new' : ''}${j.key === st.sel ? ' sel' : ''}`);
    }
  };
  const project = p => {
    tv.set(p[0], p[1], p[2]).applyMatrix4(root.matrixWorld).project(cam());
    return [(tv.x * 0.5 + 0.5) * layer.clientWidth, (-tv.y * 0.5 + 0.5) * layer.clientHeight, tv.z];
  };
  const frame = () => {
    if (!alive) return;
    if (!mv.isConnected) return destroy();
    raf = requestAnimationFrame(frame);
    const c = cam(); if (!c || !layer.clientWidth) return;
    c.updateMatrixWorld?.(); root.updateMatrixWorld?.();
    const P = new Map();
    for (const j of cur.joints) {
      const [x, y, z] = project(posOf(j)); P.set(j.key, [x, y]);
      const d = dots.get(j.key); if (d) { d.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`; d.hidden = z > 1; }
    }
    for (const j of cur.joints) {
      const l = lines.get(j.key), a = P.get(j.key), b = j.parent && P.get(j.parent); if (!l || !a || !b) continue;
      l.setAttribute('x1', b[0].toFixed(1)); l.setAttribute('y1', b[1].toFixed(1)); l.setAttribute('x2', a[0].toFixed(1)); l.setAttribute('y2', a[1].toFixed(1));
    }
  };

  // ---------------------------------------------------------------- drag a dot on the screen plane through the joint
  function down(e, key) {
    e.preventDefault(); e.stopPropagation();
    const j = byKey(key); if (!j || busy) return;
    select(key);
    const el = e.currentTarget, x0 = e.clientX, y0 = e.clientY, start = [...j.pos];
    let moved = false;
    el.setPointerCapture?.(e.pointerId);
    const hit = ev => {                                   // pointer ray x plane through the joint, facing the camera
      const c = cam(), r = layer.getBoundingClientRect();
      const nx = ((ev.clientX - r.left) / r.width) * 2 - 1, ny = -(((ev.clientY - r.top) / r.height) * 2 - 1);
      tv.set(start[0], start[1], start[2]).applyMatrix4(root.matrixWorld);          // joint, world
      c.getWorldPosition(tw); c.getWorldDirection(tn);                              // origin, plane normal
      const dir = new V3(nx, ny, 0.5).unproject(c).sub(tw).normalize();
      const den = dir.dot(tn); if (Math.abs(den) < 1e-6) return null;
      const t = tv.clone().sub(tw).dot(tn) / den;
      const w = tw.clone().addScaledVector(dir, t);
      tinv.copy(root.matrixWorld).invert(); w.applyMatrix4(tinv);
      return [w.x, w.y, w.z];
    };
    const mvE = ev => {
      if (!moved && Math.hypot(ev.clientX - x0, ev.clientY - y0) < 4) return;
      moved = true; const p = hit(ev); if (p) live = { key, pos: p };
    };
    const up = () => {
      el.removeEventListener('pointermove', mvE); el.removeEventListener('pointerup', up); el.removeEventListener('pointercancel', up);
      if (moved && live) push({ type: 'move', key, to: live.pos, follow: followOn() });
      live = null;
    };
    el.addEventListener('pointermove', mvE); el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
  }

  // ---------------------------------------------------------------- panel
  const nameIn = h('input.input.rige-name', { type: 'text', maxlength: 60, 'aria-label': 'Bone name', placeholder: 'Bone name',
    on: { keydown: e => { if (e.key === 'Enter') rename(); } } });
  const follow = h('input', { type: 'checkbox', id: `rige-f-${asset.id}` });
  const followOn = () => follow.checked;
  const stepSel = h('select.select.rige-step', { 'aria-label': 'Nudge step' });
  const joints = h('select.select.rige-pick', { 'aria-label': 'Select a bone', on: { change: e => select(e.target.value || null) } });
  const sel = h('div.rige-sel');
  const list = h('ol.rige-log');
  const status = h('div.rige-status.dim');
  const applyB = btn('Apply', { kind: 'primary', size: 'sm', icon: 'check', title: 'Write the changes as a new rig-edited copy of this mesh (the original is not changed)', onClick: () => apply() });
  const skinB = btn('Re-skin', { size: 'sm', icon: 'bot', title: 'Recompute the skin weights for the current skeleton (task reskin, GPU, about half a minute). Needed after adding bones.', onClick: () => reskin() });
  const undoB = btn('Undo', { size: 'sm', icon: 'rotate-ccw', title: 'Undo the last change', onClick: () => { const e = st.log.pop(); if (e) st.redo.push(e); refresh(); } });
  const redoB = btn('Redo', { size: 'sm', title: 'Redo', onClick: () => { const e = st.redo.pop(); if (e) st.log.push(e); refresh(); } });
  const nudges = DIRS.map(([label, v]) => btn(label, { size: 'sm', title: `Move ${label.toLowerCase()} (the character's ${label.toLowerCase()})`, onClick: () => nudge(v) }));
  const addB = btn('Add child', { size: 'sm', icon: 'plus', title: 'Add a bone below the selected one', onClick: () => addChild() });
  const delB = btn('Delete', { size: 'sm', icon: 'trash-2', kind: 'danger', title: 'Delete the selected bone (its skin weights go to its parent)', onClick: () => del() });
  const renB = btn('Rename', { size: 'sm', onClick: () => rename() });

  const select = k => { st.sel = k || null; refresh(); };
  function push(e) { st.log.push(e); st.redo.length = 0; refresh(); }
  function nudge(v) {
    const j = byKey(st.sel); if (!j) return;
    const step = +stepSel.value || 0.01;
    push({ type: 'move', key: j.key, to: add3(j.pos, v.map(c => c * step)), follow: followOn() });
  }
  function rename() {
    const j = byKey(st.sel); const n = nameIn.value.trim();
    if (!j || !n || n === j.name) return;
    if (cur.joints.some(o => o !== j && o.name === n)) { S.toast(`A bone called “${n}” exists already.`, { kind: 'warn' }); return; }
    push({ type: 'rename', key: j.key, name: n });
  }
  function addChild() {
    const j = byKey(st.sel); if (!j) return;
    const p = j.parent && byKey(j.parent);
    let dir = p ? sub3(j.pos, p.pos) : [0, 1, 0];
    const L = len3(dir); dir = L > 1e-6 ? dir.map(c => c / L) : [0, 1, 0];
    const dist = Math.max(0.02, (p ? L * 0.6 : 0) || (info.height || 1) * 0.05);
    let n = 1; const names = new Set(cur.joints.map(o => o.name));
    while (names.has(`bone_new_${n}`)) n++;
    const key = `n${Date.now().toString(36)}${n}`;
    st.log.push({ type: 'add', key, parent: j.key, name: `bone_new_${n}`, pos: add3(j.pos, dir.map(c => c * dist)) }); st.redo.length = 0;
    st.sel = key; refresh();
  }
  function del() {
    const j = byKey(st.sel); if (!j) return;
    if (!j.parent) { S.toast('The root bone cannot be deleted here.', { kind: 'warn' }); return; }
    st.log.push({ type: 'delete', key: j.key }); st.redo.length = 0; st.sel = j.parent; refresh();
  }
  const note = t => status.replaceChildren(t || '');

  function refresh() {
    if (!info) return;
    state();
    syncNodes();
    const j = byKey(st.sel);
    joints.replaceChildren(h('option', { value: '' }, 'Select a bone…'),
      ...cur.joints.map(o => h('option', { value: o.key, selected: o.key === st.sel }, `${o.isNew ? '+ ' : ''}${o.name}${o.role ? ` · ${o.role}` : ''}`)));
    sel.replaceChildren(j
      ? h('span', h('b', j.name), j.role ? h('span.dim', ` · ${j.role}`) : null,
        h('span.dim', ` · parent ${j.parent ? byKey(j.parent)?.name : 'none (root)'} · ${cur.joints.filter(o => o.parent === j.key).length} child(ren)`),
        j.isNew ? h('span.rige-tag', 'new, no skin weight') : null)
      : h('span.dim', 'Click a dot in the viewer (or pick a bone from the list).'));
    if (document.activeElement !== nameIn) nameIn.value = j ? j.name : '';
    for (const b of [nameIn, renB, addB, delB, ...nudges, stepSel, follow]) b.disabled = !j || busy;
    delB.disabled = !j || !j.parent || busy;
    list.replaceChildren(...cur.lines.filter(Boolean).map(t => h('li', t)));
    const n = cur.ops.length;
    applyB.disabled = !n || busy; undoB.disabled = !st.log.length || busy; redoB.disabled = !st.redo.length || busy;
    applyB.replaceChildren(ic('check', 15), h('span', n ? `Apply (${n})` : 'Apply'));
    const needs = !!(asset.info?.needs_reskin);
    skinB.disabled = busy || n > 0 || !reskinModel();
    skinB.title = !reskinModel() ? 'Task reskin is not available: restart the core to load the newest registry, or install the UniRig model.'
      : n ? 'Apply the pending changes first' : skinB.dataset.t || skinB.title;
    skinB.classList.toggle('btn-primary', needs && !n);
    note(needs && !n ? 'This rig has bones without skin weight: run Re-skin so the mesh follows them.' : '');
  }
  skinB.dataset.t = skinB.title;

  const reskinModel = () => {
    const ready = S.modelsFor('reskin').filter(m => m.status?.status === 'ready');
    const d = S.store.defaults?.reskin;
    return (ready.find(m => m.id === d) || ready[0])?.id || null;
  };

  async function apply() {
    if (busy || !cur.ops.length) return;
    busy = true; refresh(); note('Editing the rig…');
    try {
      const { assets } = await runJob({ tool: 'rig.edit', inputs: [{ asset: asset.id, role: 'mesh' }], params: { ops: cur.ops },
        project: S.store.project, title: `Edit rig · ${asset.name || 'mesh'}`.slice(0, 60), meta: { studio: '3d' } });
      st.log.length = 0; st.redo.length = 0; st.sel = null;
      const out = assets[0];
      S.toast(out?.info?.needs_reskin ? 'Rig edited: a new copy is open. Press Re-skin so the new bones get skin weights.' : 'Rig edited: the new copy is open.', { kind: 'ok' });
    } catch (e) { S.toastError(e); note(e.message || 'The edit failed.'); } finally { busy = false; if (alive) refresh(); }
  }
  async function reskin() {
    const model = reskinModel(); if (busy || !model) return;
    busy = true; refresh(); note('Re-skinning (GPU, about half a minute)…');
    try {
      await runJob({ task: 'reskin', model, params: {}, inputs: [{ asset: asset.id, role: 'mesh' }], project: S.store.project,
        title: `Re-skin · ${asset.name || 'mesh'}`.slice(0, 60), meta: { studio: '3d' } });
      S.toast('Re-skinned: the new copy is open.', { kind: 'ok' });
    } catch (e) { S.toastError(e); note(e.message || 'The re-skin failed.'); } finally { busy = false; if (alive) refresh(); }
  }

  function build() {
    const hgt = info.height || 1;
    stepSel.replaceChildren(...[0.005, 0.02, 0.05].map(f => { const v = r4(hgt * f); return h('option', { value: v, selected: f === 0.02 }, cm(v)); }));
    body.replaceChildren(
      h('p.dim.rige-help', 'Click a dot to select a bone, drag it (or use the buttons) to move it. Changes are written to a new copy; the mesh and its rest pose stay as they are.'),
      h('div.rige-row', joints, S.uiInfo('rige_select')), sel,
      h('div.rige-row', nameIn, renB),
      h('div.rige-row.rige-nudge', h('span.dim', 'Move'), stepSel, S.uiInfo('rige_step'), ...nudges),
      h('div.rige-row', h('label.rige-check', follow, ' children follow'), S.uiInfo('rige_follow')),
      h('div.rige-row', addB, delB, h('span.spacer'), undoB, redoB, S.uiInfo('rige_undo')),
      list, h('div.rige-row', applyB, S.uiInfo('rige_apply'), skinB, S.uiInfo('rige_reskin')), status);
    refresh();
    raf = requestAnimationFrame(frame);
  }

  function destroy() {
    if (!alive) return; alive = false; cancelAnimationFrame(raf); layer.remove(); host.replaceChildren(); host.hidden = true;
  }

  loadSkeleton(asset).then(r => {
    if (!alive) return;
    if (r.error) { body.replaceChildren(h('p.rige-err', ic('triangle-alert', 14), ' ', r.error)); return; }
    info = r.info; build();
  });
  return { destroy };
}
