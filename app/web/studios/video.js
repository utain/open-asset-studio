// Video studio: Text → Video, Image → Video (first frame), First + last frame, and Smooth (RIFE frame interpolation).
// Built on the shared TaskStudio kit (model picker, schema-driven settings, ✨ Improve with video styles, live job cards,
// compare / export) plus a big player for the newest or opened clip and a small toolbar for what comes next:
//   · Add sound   → sends the clip to Sound › Video to sound (v2a, MMAudio / HunyuanVideo-Foley)
//   · Continue    → takes the LAST frame of the clip as the first frame of a new Image → Video job (long shots in 5 s steps)
//   · Smooth      → RIFE x2 / x4
//   · Tools ▾     → CPU tools (plugins/video.py): trim, loop, speed, reframe 9:16 / 1:1, convert, join, soundtrack, frame
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const IMG = ['image'];
const TASKS = [
  { id: 't2v', label: 'Text', short: 'Text', result: 'video', prompt: true,
    placeholder: 'A red fox trots through fresh snow at dawn, slow tracking shot, soft golden light… (Thai or English)' },
  { id: 'i2v', label: 'Image', short: 'Image', result: 'video', prompt: true, receives: ['i2v'],
    placeholder: 'Describe the MOTION: the fox turns its head and trots away, camera slowly pushes in…',
    inputs: [{ key: 'image', label: 'First frame', types: IMG, role: 'image', required: true, hint: 'A still from the Image studio, an upload, or the last frame of another clip (Continue).' }] },
  { id: 'flf2v', label: 'First+Last', short: 'F+L', result: 'video', prompt: true,
    placeholder: 'How does the first picture turn into the last one? Describe the movement between them…',
    inputs: [{ key: 'image', label: 'First frame', types: IMG, role: 'image', required: true },
      { key: 'last', label: 'Last frame', types: IMG, role: 'last_image', required: true, hint: 'The clip starts on the first image and lands on the last one.' }] },
  { id: 'interpolate', label: 'Smooth', short: 'Smooth', result: 'video', prompt: false, receives: ['interpolate'],
    inputs: [{ key: 'video', label: 'Video', types: ['video'], role: 'video', required: true, hint: 'RIFE doubles (x2) or quadruples (x4) the frame rate: 16 fps Wan clips become 32 / 64 fps.' }] },
];

const ASPECT_CHIPS = [['16:9', 'YouTube · 16:9'], ['9:16', 'Reels / Shorts · 9:16'], ['1:1', 'Feed · 1:1']];

let cssDone = false;
function css() {
  if (cssDone) return; cssDone = true;
  const st = document.createElement('style');
  st.textContent = `
.vid-stage { margin-bottom: 14px; }
.vid-stage .video-box video { max-height: 52vh; }
.vid-stage.empty-stage { display: none; }
.vid-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 8px 0 0; }
.vid-bar .vid-title { font-weight: 600; min-width: 0; flex: 1 1 160px; }
.vid-note { font-size: 12px; color: var(--text-2); display: flex; gap: 6px; align-items: flex-start; }
.vid-note svg { flex: none; margin-top: 2px; }
.vid-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.vid-tool-form { display: grid; gap: 12px; }
.vid-tool-form label { display: grid; gap: 4px; font-size: 12.5px; }
.vid-tool-form .row { align-items: center; }
.acard.mod-video .thumb { aspect-ratio: 16 / 9; }
@media (max-width: 820px) { .vid-stage .video-box video { max-height: 40vh; } }
`;
  document.head.append(st);
}

export default S.defineStudio({
  id: 'video', title: 'Video', icon: 'film', color: '#ff7a90', order: 15, mobile: true, section: 'studio',
  receives: [
    { action: 'i2v', label: 'Animate', icon: 'video', types: ['image'] },
    { action: 'continue', label: 'Continue', icon: 'arrow-right', types: ['video'] },
    { action: 'interpolate', label: 'Smooth', icon: 'zap', types: ['video'] },
    { action: 'tools', label: 'Video tools', icon: 'scissors', types: ['video'] },
  ],
  mount(el) {
    css();
    el.style.height = '100%';
    const stageHost = h('div.vid-stage.empty-stage');
    let kit; let stageAsset = null;
    kit = new S.TaskStudio({
      studio: 'video',
      tasks: TASKS,
      context: () => S.getSetting('prompt.context.video') || S.prefs.get('video.context', ''),
      extraPanel: (k, t) => extraPanel(k, t),
      beforeSubmit: (spec, k) => beforeSubmit(spec, k),
    });
    kit.center.querySelector('.sc-body')?.prepend(stageHost);
    el.append(kit.el);

    // ------------------------------------------------------------ player stage
    function setStage(a) {
      if (!a || a.type !== 'video') return;
      stageAsset = a;
      stageHost.classList.remove('empty-stage');
      const bar = h('div.vid-bar',
        h('div.vid-title.ellipsis', { title: a.name }, a.name || a.id),
        a.info?.has_audio ? h('span.badge.outline', ic('volume-2', 11), ' sound') : h('span.badge.outline', 'silent'),
        a.info?.width ? h('span.badge.outline', `${a.info.width}×${a.info.height} · ${a.info.fps ? Math.round(a.info.fps * 10) / 10 + ' fps' : ''}`) : null,
        btn('Add sound', { size: 'sm', icon: 'clapperboard', title: 'Sound › Video to sound: foley and ambience that follow the picture', onClick: () => S.sendTo('sound', a.id, { action: 'v2a' }) }),
        btn('Continue', { size: 'sm', icon: 'arrow-right', title: 'Use the last frame as the first frame of a new Image → Video job', onClick: () => continueFrom(a) }),
        btn('Smooth', { size: 'sm', icon: 'zap', title: 'RIFE frame interpolation', onClick: () => toSmooth(a) }),
        btn('Tools', { size: 'sm', icon: 'scissors', onClick: e => toolsMenu(e.currentTarget, a) }),
        btn('Details', { size: 'sm', kind: 'ghost', icon: 'info', onClick: () => S.openAsset(a, { list: kit.grid.assets().map(x => x.id) }) }));
      const box = S.videoBox(a, { compact: false });
      const v = box.querySelector('video');
      v.loop = true;
      stageHost.replaceChildren(box, bar);
      kit.grid.setActive?.(a.id);
    }
    const showLatest = () => {
      const vids = kit.grid.assets().filter(a => a.type === 'video');
      if (!stageAsset || !vids.find(a => a.id === stageAsset.id)) { if (vids[0]) setStage(vids[0]); else { stageHost.classList.add('empty-stage'); stageAsset = null; } }
    };
    kit.grid.onChange(() => setTimeout(showLatest, 0));
    S.bus.on('asset', ev => {
      if (ev.action === 'created' && ev.asset.type === 'video' && ev.asset.project === S.store.project && ev.asset.studio === 'video') setStage(ev.asset);
      if (ev.action === 'updated' && stageAsset?.id === ev.asset.id && ev.asset.type === 'video') stageAsset = ev.asset;
    });
    S.bus.on('drawer', a => { if (a?.type === 'video' && !el.hidden) setStage(a); });
    // clicking a card plays it on the stage as well as opening the details drawer
    kit.grid.el.addEventListener('click', e => {
      const card = e.target.closest?.('.acard'); const a = card?._asset;
      if (a?.type === 'video' && !e.target.closest('button')) setStage(a);
    });

    // ------------------------------------------------------------ actions
    function toSmooth(a) { kit.setTask('interpolate'); kit.slots.video.set([a.id]); S.toast('Clip loaded: choose x2 or x4 and press Generate.', { kind: 'info' }); }

    async function runTool(tool, inputs, params, title) {
      try {
        const job = await S.api.submit({ tool, inputs, params, project: S.store.project, title, meta: { studio: 'video' } });
        S.store.jobs.set(job.id, job);
        S.toast(`${title}…`, { kind: 'info', timeout: 1800 });
        const j = await S.waitJob(job.id);
        if (j.status === 'failed') { S.toast(j.error?.message || 'The tool failed', { kind: 'err', title }); return null; }
        if (j.status === 'done') S.toast(`${title}: done. The result is in the Library${tool === 'video.frame' ? '' : ' and on this page'}.`, { kind: 'ok' });
        return j;
      } catch (e) { S.toastError(e, title); return null; }
    }

    async function continueFrom(a) {
      const j = await runTool('video.frame', [{ asset: a.id, role: 'video' }], { at: 'last' }, 'Taking the last frame');
      const id = j?.outputs?.[0];
      if (!id) return;
      kit.setTask('i2v');
      await kit.slots.image.set([id]);
      kit.prompt.focus();
      S.toast('Last frame loaded as the first frame. Describe what happens next, then Generate.', { kind: 'ok' });
    }

    function toolsMenu(anchor, a) {
      const sel = kit.grid.selected();
      const one = (tool, params, title) => runTool(tool, [{ asset: a.id, role: 'video' }], params, title);
      S.menu(anchor, [
        { head: 'Make a derived copy' },
        { label: 'Trim…', icon: 'scissors', onClick: async () => { const v = await ask('Trim', [['start', 'Start (s)', 'number', 0], ['end', 'End (s, 0 = to the end)', 'number', 0]]); if (v) one('video.trim', v, 'Trim'); } },
        { label: 'Loop (ping-pong)', icon: 'repeat', sub: 'forward then backward, seamless', onClick: () => one('video.loop', { mode: 'pingpong' }, 'Ping-pong loop') },
        { label: 'Repeat x3', icon: 'repeat', onClick: () => one('video.loop', { mode: 'repeat', times: 3 }, 'Repeat x3') },
        { label: 'Speed…', icon: 'gauge', onClick: async () => { const v = await ask('Speed', [['speed', 'Speed (0.25 = slow motion, 2 = double)', 'number', 0.5]]); if (v) one('video.speed', v, 'Change speed'); } },
        { sep: true },
        { head: 'Social formats' },
        { label: '9:16 vertical (blurred sides)', icon: 'smartphone', onClick: () => one('video.reframe', { aspect: '9:16', fit: 'blur' }, 'Reframe 9:16') },
        { label: '9:16 vertical (crop)', icon: 'crop', onClick: () => one('video.reframe', { aspect: '9:16', fit: 'crop' }, 'Crop 9:16') },
        { label: '1:1 square (crop)', icon: 'crop', onClick: () => one('video.reframe', { aspect: '1:1', fit: 'crop' }, 'Crop 1:1') },
        { label: '16:9 (blurred sides)', icon: 'film', onClick: () => one('video.reframe', { aspect: '16:9', fit: 'blur' }, 'Reframe 16:9') },
        { label: 'WebM / smaller file…', icon: 'package', onClick: async () => { const v = await ask('Convert', [['format', 'Format', ['mp4', 'webm'], 'webm'], ['max_side', 'Longest side in px (0 = keep)', 'number', 1280], ['fps', 'Frame rate (0 = keep)', 'number', 0]]); if (v) one('video.convert', v, 'Convert'); } },
        { sep: true },
        { head: 'Combine' },
        { label: sel.length > 1 ? `Join ${sel.length} selected clips` : 'Join clips (select 2 or more first)', icon: 'layers', disabled: sel.length < 2,
          onClick: () => runTool('video.concat', sel.map(id => ({ asset: id, role: 'video' })), {}, `Join ${sel.length} clips`) },
        { label: 'Add a soundtrack from the library…', icon: 'music', onClick: async () => {
          const ids = await S.pickAssets({ types: ['audio', 'voice'], title: 'Pick the music, voice-over or foley' });
          if (ids?.length) runTool('video.mux', [{ asset: a.id, role: 'video' }, { asset: ids[0], role: 'audio' }], { keep_original: false }, 'Add soundtrack');
        } },
        { label: 'Save a frame as image…', icon: 'image', onClick: async () => { const v = await ask('Frame', [['at', 'Which frame', ['last', 'first', 'time'], 'last'], ['t', 'Time in seconds (when "time")', 'number', 1]]); if (v) one('video.frame', { at: v.at === 'time' ? String(v.t) : v.at }, 'Extract a frame'); } },
      ].filter(Boolean), { align: 'end' });
    }

    /** Tiny parameter dialog: fields = [[name, label, 'number' | [choices], default]] -> Promise<{name: value} | null> */
    function ask(title, fields) {
      return new Promise(resolve => {
        const vals = {}; const inputs = [];
        let done = false;
        const form = h('div.vid-tool-form', ...fields.map(([name, label, kind, def]) => {
          vals[name] = def;
          let input;
          if (Array.isArray(kind)) input = h('select.select', { on: { change: e => { vals[name] = e.target.value; } } }, ...kind.map(c => h('option', { value: c, selected: c === def }, c)));
          else input = h('input.input', { type: 'number', step: 'any', value: def, on: { input: e => { vals[name] = Number(e.target.value); } } });
          inputs.push(input);
          return h('label', h('span', label), input);
        }));
        const md = S.modal({ title, icon: 'scissors', width: 420, body: form, onClose: () => { if (!done) resolve(null); },
          foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn('Run', { kind: 'primary', onClick: () => { done = true; md.close(); resolve(vals); } })] });
        setTimeout(() => inputs[0]?.focus(), 50);
      });
    }

    // ------------------------------------------------------------ extra panel (per task)
    function extraPanel(k, t) {
      if (t.id === 'interpolate') return h('div.vid-note', ic('info', 13), 'Runs in a few seconds. Needs a clip from the Library (any mp4 / webm / mov).');
      const note = h('div.vid-note');
      const chips = h('div.vid-chips', ...ASPECT_CHIPS.map(([a, lab]) => h('button.chip', { type: 'button', title: `Set the aspect ratio to ${a}`, on: { click: () => k.form.setValues({ aspect: a }) } }, lab)));
      const ctxField = h('input.input.input-sm', { placeholder: 'Project style for ✨ Improve, e.g. "warm 35mm documentary look"', value: S.getSetting('prompt.context.video') || S.prefs.get('video.context', ''),
        on: { change: async e => {      // saved with the project (Settings > Prompts & text); the old browser-only value is dropped
          try { await S.setProjectSettings({ prompt: { context: { video: e.target.value.trim() || null } } }); S.prefs.set('video.context', ''); } catch (err) { S.toastError(err); }
        } } });
      S.bus.on('settings', () => { if (ctxField.isConnected && document.activeElement !== ctxField) ctxField.value = S.getSetting('prompt.context.video') || S.prefs.get('video.context', ''); });
      const ctxIn = t.id === 't2v' ? h('label.field', h('span.field-label', ic('palette', 14), 'Look / art direction (optional)'), ctxField,
        h('span.field-help', 'Sent to the prompt improver as context, so every shot keeps the same look.')) : null;
      const update = m => {
        const f = m?.features || {};
        note.replaceChildren(ic('info', 13), h('span',
          f.audio ? 'This model also makes the sound (ambience, effects, speech) together with the picture. ' : 'This model is silent: use Add sound to give the clip foley, or Tools › Add a soundtrack. ',
          m?.license?.commercial === false ? 'Non-commercial licence. ' : '',
          f.max_resolution ? `Up to ${f.max_resolution}p.` : ''));
      };
      update(k.picker.model);
      const orig = k._onModel.bind(k);
      k._onModel = (m, fromBuild) => { orig(m, fromBuild); update(m); };
      return h('div.col', { style: { gap: '10px' } }, h('div.field', h('div.field-label', ic('crop', 14), 'Format'), chips), ctxIn, note);
    }

    // ------------------------------------------------------------ submit tweaks
    function beforeSubmit(spec, k) {
      const t = k.cur();
      const m = k.picker.model;
      const nm = m?.name || 'Video';
      if (t.id === 'interpolate') spec.title = `Smooth x${spec.params.factor || 2} · ${k.slots.video?.metas.get(spec.inputs[0]?.asset)?.name || 'clip'}`.slice(0, 60);
      else if (t.id === 'i2v' || t.id === 'flf2v') spec.title = (k.prompt.value || `${nm} from ${k.slots.image?.metas.get(spec.inputs[0]?.asset)?.name || 'image'}`).slice(0, 60);
      return spec;
    }

    async function receive(p) {
      const assets = p.assets || [];
      const img = assets.find(a => a.type === 'image');
      const vid = assets.find(a => a.type === 'video');
      if (p.action === 'i2v' && img) { kit.setTask('i2v'); await kit.slots.image.set([img.id]); kit.prompt.focus(); S.toast('Image loaded as the first frame: describe the motion and press Generate.', { kind: 'info' }); return; }
      if (p.action === 'interpolate' && vid) { toSmooth(vid); return; }
      if (p.action === 'continue' && vid) { setStage(vid); await continueFrom(vid); return; }
      if (p.action === 'tools' && vid) { setStage(vid); setTimeout(() => toolsMenu(stageHost.querySelector('.vid-bar button:nth-last-child(2)') || stageHost, vid), 150); return; }
      return kit.receive(p);
    }
    return { route: r => kit.route(r), receive, show: () => showLatest() };
  },
});
