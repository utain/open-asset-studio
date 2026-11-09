// Materials studio: seamless PBR materials from a description, a photo or an existing material (base colour, normal, height, roughness,
// metallic, AO, ORM), plus a CPU "Make tileable" tool. Built on the shared TaskStudio kit. The drawer preview of a material (3D with live
// tiling, 3x3 tile check, map strip, export) lives in mat/viewer.js and is registered here at start-up (S.registerPreview).
// Send-to: Image results -> "Make material" / "Make tileable"; a material -> "Variations" (this studio, slot Base material, 4 takes).
import * as S from '../core/sdk.js';
import { css, materialPreview } from './mat/viewer.js';
const { h, ic } = S;

S.registerPreview('material', materialPreview);

const IMG = ['image'];
const IDEAS = 'Mossy cobblestone, wet after rain… (Thai or English). Describe the surface only, not an object.';
const TASKS = [
  { id: 'material', label: 'Material', result: 'material', prompt: true, promptRequired: false, receives: ['material', 'material_vary'], placeholder: IDEAS,
    inputs: [
      { key: 'image', label: 'Photo (optional)', types: IMG, role: 'image', hint: 'A flat, evenly lit photo of a surface. It is cropped square and made tileable.' },
      { key: 'base', label: 'Base material (optional)', types: ['material', 'image'], role: 'base', hint: 'Make variations of an existing material or image.' }] },
  { id: 'tileable', label: 'Make tileable', result: 'image', prompt: false, receives: ['tileable'],
    inputs: [{ key: 'images', label: 'Images', types: IMG, role: 'input', multiple: true, max: 16, required: true, hint: 'Blend the edges so the picture repeats without a seam. CPU, about a second each.' }] },
];

export default S.defineStudio({
  id: 'material', title: 'Materials', icon: 'layers', color: 'var(--c-image)', order: 15, mobile: false,
  receives: [
    { action: 'material', label: 'Make material', icon: 'layers', types: IMG },
    { action: 'tileable', label: 'Make tileable', icon: 'grid-2x2', types: IMG },
    { action: 'material_vary', label: 'Variations', icon: 'shuffle', types: ['material'] },
  ],
  mount(el) {
    css();
    el.style.height = '100%';
    const head = h('div.mat-headtools');
    const kit = new S.TaskStudio({
      studio: 'material', tasks: TASKS,
      centerHeader: () => head,
      extraPanel: (k, t) => extraPanel(k, t),
      beforeSubmit: (spec, k) => beforeSubmit(spec, k),
    });
    head.append(h('div.mat-tip', ic('info', 13), ' Describe the surface, not an object: “red brick wall, weathered”. Results tile in all directions; open one to check the seams in 3D or as a 3×3 grid.'));
    el.append(kit.el);

    async function receive(p) {
      const assets = p.assets || [];
      if (p.action === 'material_vary' && assets[0]) {
        kit.setTask('material');
        await kit.slots.base.set([assets[0].id]);
        const prompt = assets[0].prompt || assets[0].x_style?.user_prompt || '';
        if (prompt && kit.prompt.ta && !kit.prompt.value) kit.prompt.set(prompt);
        kit.form.setValues({ batch: 4, seed: -1 });
        S.toast('Four variations of this material: press Generate (Variation sets how far they drift).', { kind: 'info' });
        return;
      }
      if (p.action === 'material' && assets[0]) { kit.setTask('material'); await kit.slots.image.set([assets[0].id]); kit.prompt.focus(); return; }
      if (p.action === 'tileable' && assets.length) { kit.setTask('tileable'); await kit.slots.images.set(assets.map(a => a.id).slice(0, 16)); return; }
      return kit.receive(p);
    }
    return { route: r => kit.route(r), receive };
  },
});

function extraPanel(kit, t) {
  if (t.id === 'tileable') {
    return h('div.mat-tip', ic('info', 13), ' The blend method keeps your pixels in the middle of the picture. For a textured surface that still shows a seam, run it through Material with the photo slot and “Photo seams: regenerate”.');
  }
  return null;
}

function beforeSubmit(spec, kit) {
  const t = kit.cur();
  if (t.id !== 'material') return spec;
  const photo = kit.slots.image?.ids[0], base = kit.slots.base?.ids[0];
  const prompt = (spec.params.prompt || '').trim();
  if (!prompt && !photo && !base) { S.toast('Type a description or add a photo.', { kind: 'warn' }); kit.prompt.focus(); return null; }
  if (photo && base) {
    spec.inputs = spec.inputs.filter(i => i.role !== 'image');
    S.toast('A base material is set, so the photo is ignored.', { kind: 'info' });
  }
  if (base && !prompt) {
    const m = kit.slots.base.metas.get(base);
    const p = m?.x_style?.user_prompt || m?.prompt;
    if (p) spec.params.prompt = p;
  }
  const label = spec.params.prompt || (base ? `${kit.slots.base.metas.get(base)?.name || 'Material'} variation` : 'Material from a photo');
  spec.title = String(label).slice(0, 60);
  return spec;
}

