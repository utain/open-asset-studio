// 3D studio: Image → 3D shape (+ optional PBR texture in the same job), Texture / retexture an existing mesh, and
// Export (clean-up, decimation, smoothing, GLB / OBJ / PLY / STL). Hunyuan3D 2.1 through the proven engines/hy3d engine:
// every engine knob is in the form (shape knobs on the Image → 3D tab, texture knobs on the Texture tab; a textured
// Image → 3D job uses the Texture tab's settings). Big model-viewer preview, automatic mesh thumbnails.
import * as S from '../core/sdk.js';
import { css, meshViewer, autoThumb, runJob } from './i3d/common.js';
const { h, ic, btn } = S;

const TASKS = [
  { id: 'shape', label: 'Image → 3D', result: 'mesh', prompt: false, receives: ['shape'],
    inputs: [{ key: 'image', label: 'Concept image', types: ['image'], role: 'image', required: true,
      hint: 'Best: one centred object, fully visible, transparent or plain background (Image › 3D prep).' }] },
  { id: 'text3d', label: 'Text → 3D', result: 'mesh', prompt: true, placeholder: 'A low-poly wooden treasure chest with iron bands, game prop… (cloud: Meshy / Tripo)' },
  { id: 'texture', label: 'Texture', result: 'mesh', prompt: false, receives: ['retexture'],
    inputs: [{ key: 'mesh', label: 'Mesh to texture', types: ['mesh'], role: 'mesh', required: true },
      { key: 'image', label: 'Reference image (optional)', types: ['image'], role: 'image', hint: 'Empty = the image the shape was made from. Set Reference image to “current” to paint from this one.' }] },
  { id: 'multiview', label: 'Multi-view', result: 'image', prompt: false, receives: ['multiview'],
    inputs: [{ key: 'image', label: 'Object image', types: ['image'], role: 'image', required: true,
      hint: 'One object, fully visible, plain or transparent background. Result: the same object from other sides (sheet + one image per view; SV3D also attaches an orbit video).' }] },
  { id: 'rig', label: 'Rig', result: 'mesh', prompt: false, receives: ['rig'],
    inputs: [{ key: 'mesh', label: 'Mesh to rig', types: ['mesh'], role: 'mesh', required: true,
      hint: 'A character or creature in a neutral pose (A- or T-pose) works best. Result: a GLB with a skeleton, skin weights and two preview animations (flex test, wiggle).' }] },
  { id: 'reskin', label: 'Re-skin', result: 'mesh', prompt: false, receives: ['reskin'],
    inputs: [{ key: 'mesh', label: 'Rigged mesh (edited skeleton)', types: ['mesh'], role: 'mesh', required: true,
      hint: 'A rigged GLB whose skeleton you edited (Edit rig in the viewer, or tool rig.edit). The skeleton is kept as it is; only the skin weights are recomputed, so new bones move the mesh.' }] },
  { id: 'motion', label: 'Motion', result: 'mesh', prompt: true, receives: ['motion'],
    placeholder: 'a person walks forward and waves…  (English, describe the body: walk, run, jump, wave the right hand, kick, dance, turn around)',
    inputs: [{ key: 'mesh', label: 'Rigged character', types: ['mesh'], role: 'mesh', required: true,
      hint: 'A rigged humanoid from the Rig tab, in a T- or A-pose (two legs, two arms). Result: a new mesh asset with the animation clip “motion”, retargeted onto its skeleton.' }] },
  { id: 'mesh_export', label: 'Export', result: 'mesh', prompt: false, receives: ['export'],
    inputs: [{ key: 'mesh', label: 'Meshes', types: ['mesh'], role: 'mesh', multiple: true, max: 12, required: true }] },
];

export default S.defineStudio({
  id: '3d', title: '3D', icon: 'cuboid', color: 'var(--c-3d)', order: 20, mobile: true,
  receives: [
    { action: 'shape', label: 'Make 3D', icon: 'cuboid', types: ['image'] },
    { action: 'retexture', label: 'Retexture', icon: 'palette', types: ['mesh'] },
    { action: 'export', label: 'Export mesh', icon: 'package', types: ['mesh'] },
    { action: 'rig', label: 'Auto-rig', icon: 'bot', types: ['mesh'] },
    { action: 'reskin', label: 'Re-skin (keep the skeleton)', icon: 'bot', types: ['mesh'] },
    { action: 'motion', label: 'Animate (text to motion)', icon: 'wand-sparkles', types: ['mesh'] },
    { action: 'multiview', label: 'Other views', icon: 'grid-2x2', types: ['image'] },
  ],
  mount(el) {
    css();
    el.style.height = '100%';
    const viewer = meshViewer({ height: 440, actions: a => [
      btn('Retexture', { size: 'sm', icon: 'palette', onClick: () => receive({ action: 'retexture', assets: [a] }) }),
      btn('Rig', { size: 'sm', icon: 'bot', onClick: () => receive({ action: 'rig', assets: [a] }) }),
      isRigged(a) ? btn('Motion', { size: 'sm', icon: 'wand-sparkles', title: 'Describe a movement in words and animate this rigged character (text to motion)', onClick: () => receive({ action: 'motion', assets: [a] }) }) : null,
      needsReskin(a) ? btn('Re-skin', { size: 'sm', icon: 'bot', kind: 'primary', title: 'This rig has bones without skin weight (it was edited). Recompute the weights for its skeleton.', onClick: () => receive({ action: 'reskin', assets: [a] }) }) : null,
      needsClips(a) ? btn('Animate preview', { size: 'sm', icon: 'play', title: 'Add a flex test and a wiggle loop to this rigged mesh (a new asset) to see how the skin moves', onClick: e => animatePreview(a, e.currentTarget) }) : null,
      btn('Export', { size: 'sm', icon: 'package', onClick: () => receive({ action: 'export', assets: [a] }) }),
    ] });
    const kit = new S.TaskStudio({
      studio: '3d',
      tasks: TASKS.filter(t => t.id !== 'reskin' || S.modelsFor('reskin').length),    // an older core has no task reskin: no tab
      extraPanel: (k, t) => extraPanel(k, t),
      beforeSubmit: (spec, k) => beforeSubmit(spec, k),
    });
    const top = h('div.m3d-top', viewer);
    kit.center.querySelector('.sc-body')?.prepend(top);
    el.append(kit.el);
    // the 3D viewer only belongs to mesh-result tasks: Multi-view results are images (sheet, views, orbit video in the drawer)
    const syncViewer = () => { top.hidden = kit.cur().result !== 'mesh'; };
    const onModel = kit._onModel.bind(kit);
    kit._onModel = (m, fromBuild) => { onModel(m, fromBuild); kit._m3dNote?.(); };
    const setTask = kit.setTask.bind(kit);
    kit.setTask = id => { setTask(id); syncViewer(); };
    syncViewer();

    const showLatest = () => {
      const meshes = kit.grid.assets().filter(a => a.type === 'mesh');
      autoThumb(meshes);
      const cur = viewer.current();
      if (!cur || !meshes.find(a => a.id === cur.id)) viewer.set(meshes[0] || null);
    };
    kit.grid.onChange(() => setTimeout(showLatest, 0));
    S.bus.on('asset', ev => {
      if (ev.action === 'created' && ev.asset.type === 'mesh' && ev.asset.project === S.store.project) { autoThumb(ev.asset); if (ev.asset.studio === '3d' || !viewer.current()) viewer.set(ev.asset); }
      if (ev.action === 'updated' && viewer.current()?.id === ev.asset.id && JSON.stringify(viewer.current().files) !== JSON.stringify(ev.asset.files)) viewer.set(ev.asset);
    });
    S.bus.on('drawer', a => { if (a?.type === 'mesh' && !el.hidden) viewer.set(a); });

    async function receive(p) {
      const assets = p.assets || [];
      const img = assets.find(a => a.type === 'image');
      const meshes = assets.filter(a => a.type === 'mesh');
      if (p.action === 'shape' && img) { kit.setTask('shape'); await kit.slots.image.set([img.id]); S.toast('Image loaded: check the settings and press Generate.', { kind: 'info' }); return; }
      if (p.action === 'retexture' && meshes[0]) { kit.setTask('texture'); await kit.slots.mesh.set([meshes[0].id]); if (img) await kit.slots.image.set([img.id]); viewer.set(meshes[0]); return; }
      if (p.action === 'multiview' && img) { kit.setTask('multiview'); await kit.slots.image.set([img.id]); S.toast('Image loaded: pick a model and press Generate.', { kind: 'info' }); return; }
      if (p.action === 'rig' && meshes[0]) { kit.setTask('rig'); await kit.slots.mesh.set([meshes[0].id]); viewer.set(meshes[0]); return; }
      if (p.action === 'reskin' && meshes[0]) { if (!kit.tasks.find(t => t.id === 'reskin')) { S.toast('Re-skin needs the newest core: restart Open Asset Studio.', { kind: 'warn' }); return; }
        kit.setTask('reskin'); await kit.slots.mesh.set([meshes[0].id]); viewer.set(meshes[0]); S.toast('Press Generate to recompute the skin weights for this skeleton.', { kind: 'info' }); return; }
      if (p.action === 'motion' && meshes[0]) { kit.setTask('motion'); await kit.slots.mesh.set([meshes[0].id]); viewer.set(meshes[0]); S.toast('Character loaded: describe the movement and press Generate.', { kind: 'info' }); return; }
      if (p.action === 'export' && meshes.length) { kit.setTask('mesh_export'); await kit.slots.mesh.set(meshes.map(a => a.id)); viewer.set(meshes[0]); return; }
      return kit.receive(p);
    }
    return { route: r => { kit.route(r); syncViewer(); }, receive: async p => { const r = await receive(p); syncViewer(); return r; }, show: () => { showLatest(); syncViewer(); } };
  },
});

// Rigged meshes made before the preview clips existed (or imported ones) get them from the CPU tool tool:mesh.flex_preview.
const isRigged = a => !!(a?.worker?.rigged || a?.info?.rigged);
const clipsOf = a => [...(a?.worker?.animations || []), ...(a?.info?.animations || [])];
const needsReskin = a => !!a?.info?.needs_reskin;
const needsClips = a => isRigged(a) && !clipsOf(a).length;
async function animatePreview(a, button) {
  if (button) button.disabled = true;
  try {
    S.toast('Adding the animation preview…', { kind: 'info', timeout: 2500 });
    await runJob({ tool: 'mesh.flex_preview', inputs: [{ asset: a.id, role: 'mesh' }], project: S.store.project, title: `Animate ${a.name || 'mesh'}`.slice(0, 60), meta: { studio: '3d' } });
    S.toast('Animated copy ready: it plays in the viewer (clip picker next to the play button).', { kind: 'ok' });
  } catch (e) { S.toastError(e); } finally { if (button) button.disabled = false; }
}

function extraPanel(kit, t) {
  if (t.id === 'shape') {
    // the hint follows the model: Hunyuan3D 2.1 can paint in the same job, the others differ (see features.texture)
    const note = h('div.field-help');
    const upd = () => {
      const m = kit.picker?.model;
      const tex = m?.features?.texture;
      const txt = tex === 'vertex_colors' ? ' This model paints vertex colours (no UV texture). For a PBR texture, send the mesh to the Texture tab (Hunyuan3D 2.1).'
        : tex === 'pbr' ? ' This model bakes a UV-unwrapped PBR texture in the same job (CPU baker).'
        : tex === 'none' ? ' Geometry only (white mesh). To paint it, send the mesh to the Texture tab (Hunyuan3D 2.1 uses the stored input image as the reference).'
        : ' “Paint texture too” runs the texture stage right after the shape with the settings of the Texture tab. Shape ~1 min, texture 3-5 min (Balanced).';
      note.replaceChildren(ic('info', 13), txt);
    };
    kit._m3dNote = upd;
    upd();
    return note;
  }
  if (t.id === 'multiview') {
    return h('div.field-help', ic('info', 13), ' Open an image result and use “Make 3D” on a view, or feed several views to a multi-view shape model. Orbit models (SV3D) attach an orbit.mp4 to the sheet.');
  }
  if (t.id === 'rig') {
    return h('div.field-help', ic('info', 13), ' The rigged GLB keeps the mesh and adds a skeleton with skin weights. Bones have generic names (bone_0, bone_1, …); there is no skeleton template or Mixamo / VRM retargeting. The result also plays two preview animations in the viewer: “flex” bends every joint in turn so you can check the skinning, “wiggle” moves all joints a little. “rigged_plain.glb” is the same file without them. Open it in Blender, Godot or Unity.');
  }
  if (t.id === 'reskin') {
    return h('div.field-help', ic('info', 13), ' Keeps the skeleton of the input GLB exactly as it is (names, hierarchy, positions) and only recomputes the skin weights with UniRig, so bones you added or moved start to move the mesh. About half a minute on the GPU. The result is a new asset with the flex / wiggle preview clips; the input is not changed. Edit the skeleton first: “Edit rig” in the viewer (Bones on), or the Assistant.');
  }
  if (t.id === 'motion') {
    return h('div.field-help', ic('info', 13), ' HY-Motion writes up to 12 s of human motion from the text and retargets it onto the skeleton of the rigged character (the original asset is not changed; the result plays in the viewer). The character must be a two-legged, two-armed humanoid standing upright; other skeletons are refused with a reason. Prompts in English work best (“a person …”). “Root motion: in place” keeps the character on the spot. Commercial use: Tencent’s licence excludes the EU, UK and South Korea.');
  }
  if (t.id === 'mesh_export') {
    return h('div.field-help', ic('info', 13), ' Presets: “mobile” 5k faces, “pc” 30k faces (untextured shape, then retexture), “textured_glb” keeps the PBR texture. OBJ exports a zip with .mtl and texture maps.');
  }
  return null;
}

function beforeSubmit(spec, kit) {
  const t = kit.cur();
  if (t.id === 'shape' && spec.params.textured) {
    const all = kit.form.allValues();
    for (const [k, v] of Object.entries(all)) if (k.startsWith('tex_') || ['super_res', 'sr_tile', 'bake_exp', 'bake_angle', 'vertex_inpaint', 'uv_padding', 'metallic_scale', 'roughness_scale'].includes(k)) spec.params[k] = v;
  }
  const first = spec.inputs[0];
  if (t.id === 'mesh_export') { spec.params.batch = spec.inputs.length; spec.title = `Export ${spec.params.export_type?.toUpperCase() || ''} · ${spec.inputs.length} mesh(es)`; }
  if (t.id === 'shape') spec.title = `3D from ${kit.slots.image?.metas.get(first?.asset)?.name || 'image'}`.slice(0, 60);
  if (t.id === 'texture') spec.title = `Texture ${kit.slots.mesh?.metas.get(spec.inputs.find(i => i.role === 'mesh')?.asset)?.name || 'mesh'}`.slice(0, 60);
  return spec;
}
