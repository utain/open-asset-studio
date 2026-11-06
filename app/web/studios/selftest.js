// Diagnostics: a complete studio page built with TaskStudio on the CPU reference worker (test-tone / test-swatch).
// It is also the reference implementation for the Image / 3D / Sound studios: copy this file and change the tasks.
import * as S from '../core/sdk.js';

export default S.defineStudio({
  id: 'selftest', title: 'Diagnostics', icon: 'flask-conical', order: 95, section: 'system', color: 'var(--c-text)',
  receives: [{ action: 'reference', label: 'Use in Diagnostics', icon: 'flask-conical', types: ['image'] }],
  mount(el) {
    const kit = new S.TaskStudio({
      studio: 'selftest',
      includeTest: true,
      tasks: [
        { id: 'sfx', label: 'Tone (audio)', result: 'audio', prompt: true, promptRequired: false, placeholder: 'Any text: it is stored as the prompt. Try ✨ Improve.' },
        { id: 't2i', label: 'Swatch (image)', result: 'image', prompt: true, promptRequired: false, receives: ['reference'],
          inputs: [{ key: 'ref', label: 'Reference image (optional)', types: ['image'], role: 'ref' }] },
      ],
      beforeSubmit(spec) { spec.title = spec.title || 'Diagnostics'; return spec; },
    });
    el.append(kit.el);
    el.style.height = '100%';
    return kit;
  },
});
