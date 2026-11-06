// 3D studio: Image → 3D shape (+ optional PBR texture in the same job), Texture / retexture an existing mesh, and
// Export (clean-up, decimation, smoothing, GLB / OBJ / PLY / STL). Hunyuan3D 2.1 through the proven engines/hy3d engine:
// every engine knob is in the form (shape knobs on the Image → 3D tab, texture knobs on the Texture tab; a textured
// Image → 3D job uses the Texture tab's settings). Big model-viewer preview, automatic mesh thumbnails.
import * as S from '../core/sdk.js';
import { css, meshViewer, autoThumb } from './i3d/common.js';
const { h, ic, btn } = S;

const TASKS = [
  { id: 'shape', label: 'Image → 3D', result: 'mesh', prompt: false, receives: ['shape'],
    inputs: [{ key: 'image', label: 'Concept image', types: ['image'], role: 'image', required: true,
      hint: 'Best: one centred object, fully visible, transparent or plain background (Image › 3D prep).' }] },
  { id: 'text3d', label: 'Text → 3D', result: 'mesh', prompt: true, placeholder: 'A low-poly wooden treasure chest with iron bands, game prop… (cloud: Meshy / Tripo)' },
  { id: 'texture', label: 'Texture', result: 'mesh', prompt: false, receives: ['retexture'],
    inputs: [{ key: 'mesh', label: 'Mesh to texture', types: ['mesh'], role: 'mesh', required: true },
      { key: 'image', label: 'Reference image (optional)', types: ['image'], role: 'image', hint: 'Empty = the image the shape was made from. Set Reference image to “current” to paint from this one.' }] },
  { id: 'mesh_export', label: 'Export', result: 'mesh', prompt: false, receives: ['export'],
    inputs: [{ key: 'mesh', label: 'Meshes', types: ['mesh'], role: 'mesh', multiple: true, max: 12, required: true }] },
];

export default S.defineStudio({
  id: '3d', title: '3D', icon: 'cuboid', color: 'var(--c-3d)', order: 20, mobile: true,
  receives: [
    { action: 'shape', label: 'Make 3D', icon: 'cuboid', types: ['image'] },
    { action: 'retexture', label: 'Retexture', icon: 'palette', types: ['mesh'] },
    { action: 'export', label: 'Export mesh', icon: 'package', types: ['mesh'] },
  ],
  mount(el) {
    css();
    el.style.height = '100%';
    const viewer = meshViewer({ height: 440, actions: a => [
      btn('Retexture', { size: 'sm', icon: 'palette', onClick: () => receive({ action: 'retexture', assets: [a] }) }),
      btn('Export', { size: 'sm', icon: 'package', onClick: () => receive({ action: 'export', assets: [a] }) }),
    ] });
    const kit = new S.TaskStudio({
      studio: '3d',
      tasks: TASKS,
      extraPanel: (k, t) => extraPanel(k, t),
      beforeSubmit: (spec, k) => beforeSubmit(spec, k),
    });
    kit.center.querySelector('.sc-body')?.prepend(h('div.m3d-top', viewer));
    el.append(kit.el);

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
      if (p.action === 'export' && meshes.length) { kit.setTask('mesh_export'); await kit.slots.mesh.set(meshes.map(a => a.id)); viewer.set(meshes[0]); return; }
      return kit.receive(p);
    }
    return { route: r => kit.route(r), receive, show: () => showLatest() };
  },
});

function extraPanel(kit, t) {
  if (t.id === 'shape') {
    return h('div.field-help', ic('info', 13), ' “Paint texture too” runs the texture stage right after the shape with the settings of the Texture tab. Shape ~1 min, texture 3-5 min (Balanced).');
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
