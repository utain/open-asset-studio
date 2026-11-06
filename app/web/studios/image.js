// Image studio (concept art, thumbnails, b-roll stills, product shots, storyboard frames, game sprites):
// Generate / Edit (multi-reference) / Transparent RGBA / Extract subject / Remove background / Upscale /
// 3D prep (crop + recenter + pad). The use-case profile sets the default aspect ratio (16:9 video, 9:16 social, 1:1 podcast
// cover) and puts the profile's style chips first. Built on the shared TaskStudio kit: the model picker switches models by task with
// feature + licence badges, the parameter form comes from the registry, ✨ Improve writes the selected model's prompt
// format with style chips (profile first). Results: batches, A/B compare (select two), alpha checkerboard zoom (drawer),
// send-to Edit / Extract / Upscale / 3D.
import * as S from '../core/sdk.js';
import { css, EDIT_PRESETS, alphaBgToggle } from './i3d/common.js';
const { h, ic, btn } = S;

const IMG = ['image'];
const IDEAS = {
  game: 'A stylized treasure chest with gold trim, game asset… (Thai or English)',
  video: 'Wide b-roll still: misty harbour at sunrise, soft light, shallow depth of field… (Thai or English)',
  podcast: 'Cover art for a late-night history podcast: lamp, old microphone, warm tones… (Thai or English)',
  social: 'Vertical product shot of a matte black bottle on wet stone, neon rim light… (Thai or English)',
  film: 'Storyboard frame: detective at a rain-soaked window, low-key light, wide lens… (Thai or English)',
  general: 'A lighthouse at dusk, painterly, calm sea… (Thai or English)',
};
const TASKS = [
  { id: 't2i', label: 'Generate', result: 'image', prompt: true, placeholder: p => IDEAS[p.id] || IDEAS.general },
  { id: 'edit', label: 'Edit', result: 'image', prompt: true, receives: ['edit'],
    placeholder: 'Change the wood to red lacquer, keep everything else identical. Refer to references as "image 2"…',
    inputs: [{ key: 'image', label: 'Image to edit (image 1)', types: IMG, role: 'image', required: true },
      { key: 'refs', label: 'References (image 2, 3 …)', types: IMG, role: 'ref', multiple: true, max: 9, hint: 'Qwen-Image: up to 9 extra references (style, outfit, object). Kontext: none.' }] },
  { id: 'rgba', label: 'Transparent', result: 'image', prompt: true, placeholder: 'A glowing health potion bottle, isolated item… (gets the RGBA trigger automatically)' },
  { id: 'extract', label: 'Extract', result: 'image', prompt: true, promptRequired: false, receives: ['extract'],
    placeholder: 'What to cut out, e.g. "the sword" or "the character in red" (empty = the main subject)',
    inputs: [{ key: 'image', label: 'Image', types: IMG, role: 'image', required: true, hint: 'Qwen-Image redraws the subject on a real transparent background (model-based, keeps thin parts).' }] },
  { id: 'rmbg', label: 'Remove BG', result: 'image', prompt: false, receives: ['rmbg'],
    inputs: [{ key: 'images', label: 'Images', types: IMG, role: 'input', multiple: true, max: 16, required: true, hint: 'Fast matting (CPU). Transparent inputs are flattened on white first.' }] },
  { id: 'upscale', label: 'Upscale', result: 'image', prompt: false, receives: ['upscale'],
    inputs: [{ key: 'images', label: 'Images', types: IMG, role: 'input', multiple: true, max: 16, required: true, hint: 'Real-ESRGAN x4plus; alpha is kept (edge colours are bled first).' }] },
  { id: 'recenter', label: '3D prep', result: 'image', prompt: false, receives: ['recenter'],
    inputs: [{ key: 'images', label: 'Images', types: IMG, role: 'input', multiple: true, max: 16, required: true, hint: 'Crop to the object, centre it on a square with a margin: the input Hunyuan3D likes.' }] },
];
const MULTI_TOOL = new Set(['rmbg', 'upscale', 'recenter']);

export default S.defineStudio({
  id: 'image', title: 'Image', icon: 'image', color: 'var(--c-image)', order: 10, mobile: true,
  receives: [
    { action: 'edit', label: 'Edit', icon: 'pencil', types: ['image'] },
    { action: 'extract', label: 'Cut out', icon: 'scissors', types: ['image'] },
    { action: 'rmbg', label: 'Remove BG', icon: 'eraser', types: ['image'] },
    { action: 'upscale', label: 'Upscale', icon: 'maximize-2', types: ['image'] },
    { action: 'variations', label: 'Variations', icon: 'shuffle', types: ['image'] },
  ],
  mount(el) {
    css();
    el.style.height = '100%';
    let kit;
    const headTools = h('div.img-headtools');
    kit = new S.TaskStudio({
      studio: 'image',
      tasks: TASKS,
      context: () => S.prefs.get('image.context', ''),
      centerHeader: () => headTools,
      extraPanel: (k, t) => extraPanel(k, t),
      beforeSubmit: (spec, k) => beforeSubmit(spec, k),
    });
    headTools.append(alphaBgToggle(kit.center));
    el.append(kit.el);

    async function receive(p) {
      const assets = (p.assets || []).filter(a => a.type === 'image');
      if (p.action === 'variations' && assets[0]) {
        await kit.receive({ action: 'regenerate', assets: [assets[0]] });
        const m = kit.picker.model;
        const maxB = (m?.params || []).find(x => x.name === 'batch')?.max || 1;
        kit.form.setValues({ seed: -1, batch: Math.min(4, maxB) });
        if (kit.cur().prompt !== false && !kit.prompt.value) { S.toast('Settings restored: add a prompt and generate.', { kind: 'info' }); return; }
        S.toast('Generating variations with the same settings and new seeds…', { kind: 'info' });
        setTimeout(() => kit.generate(), 300);
        return;
      }
      if (p.action === 'edit' && assets.length) {
        kit.setTask('edit');
        await kit.slots.image.set([assets[0].id]);
        if (assets.length > 1) await kit.slots.refs.set(assets.slice(1, 10).map(a => a.id));
        kit.prompt.focus();
        return;
      }
      return kit.receive(p);
    }
    return { route: r => kit.route(r), receive };
  },
});

function extraPanel(kit, t) {
  if (t.id === 'edit') {
    return h('div.field', h('div.field-label', ic('wand-sparkles', 14), 'Quick instructions'),
      h('div.img-presets', ...EDIT_PRESETS.map(([label, text]) => h('button.chip', { type: 'button', title: text, on: { click: () => { kit.prompt.set(text); kit.prompt.focus(); } } }, label))));
  }
  if (t.id === 'rgba') {
    return h('div.img-tip', ic('info', 13), ' Qwen-Image-2.1 is trained to output real transparency when the prompt starts with “This is an RGBA image with transparency.” The studio adds it for you; the faint alpha haze is cleaned (Advanced › Transparency clean-up).');
  }
  if (t.id === 'recenter') {
    return h('div.img-tip', ic('cuboid', 13), ' Result: square PNG, object centred with a margin. Then send it to 3D › Image → 3D (remove background is skipped automatically for transparent images).');
  }
  if (t.id === 't2i') {
    const ctxIn = h('input.input.input-sm', { placeholder: 'Project art direction for ✨ Improve, e.g. "cozy low-poly farming game" or "warm 35mm documentary look"', value: S.prefs.get('image.context', ''),
      on: { change: e => S.prefs.set('image.context', e.target.value) } });
    return h('label.field', h('span.field-label', ic('palette', 14), 'Art direction (optional)'), ctxIn,
      h('span.field-help', 'Sent to the prompt improver as context, so every prompt keeps the same style.'));
  }
  return null;
}

function beforeSubmit(spec, kit) {
  const m = kit.picker.model;
  const t = kit.cur();
  const nIn = spec.inputs.length;
  if (t.id === 'edit') {
    const max = Number(m?.features?.max_refs || 1);
    if (nIn > max) { S.toast(`${m.name} takes ${max} image${max > 1 ? 's' : ''}: remove ${nIn - max} reference(s) or pick Qwen-Image.`, { kind: 'warn' }); return null; }
  }
  if (MULTI_TOOL.has(t.id)) {
    spec.params.batch = Math.max(1, nIn);           // live placeholders: one per input
    spec.title = `${t.label} · ${nIn} image${nIn > 1 ? 's' : ''}`;
  }
  if (t.id === 'extract') spec.title = `Extract: ${kit.prompt.value || 'main subject'}`.slice(0, 60);
  if (t.id === 'rgba') spec.params.transparent = true;
  if (spec.params.batch && t.id === 'extract') spec.params.batch = 1;
  return spec;
}
