// Home: quick-start flows, an idea box with the prompt improver, recent assets, queue, memory, model readiness.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

// Quick-start flows. `profiles` tags decide which flows come first for the active use-case profile (untagged = all).
const FLOWS = [
  { id: 'concept3d', title: 'Concept → 3D prop', color: 'var(--c-3d)', icon: 'cuboid', steps: ['sparkles', 'image', 'scissors', 'cuboid', 'package'], profiles: ['game', 'film'],
    text: 'Idea → 4 fast concepts → refine → transparent cut-out → textured GLB.', go: () => S.navigate('pipelines', 'concept3d') },
  { id: 'character', title: 'Character kit', color: 'var(--accent)', icon: 'user', steps: ['image', 'cuboid', 'mic', 'audio-lines'], profiles: ['game', 'film'],
    text: 'Concept art → 3D model → a designed voice → barks and SFX for it.', go: () => S.navigate('pipelines', 'character') },
  { id: 'sfx', title: 'SFX pass', color: 'var(--c-sound)', icon: 'audio-lines', steps: ['list-checks', 'shuffle', 'star', 'scissors', 'package'],
    text: 'Paste an event list → N takes each → audition, rate, trim, loop → export for your editor or engine.', go: () => S.navigate('sound', 'sfx') },
  { id: 'music', title: 'Music & loops', color: 'var(--c-sound)', icon: 'music', steps: ['sparkles', 'music', 'repeat', 'package'],
    text: 'Mood + genre (+ lyrics) → takes → extend / seamless loop → export.', go: () => S.navigate('sound', 'music') },
  { id: 'thumb', title: 'Thumbnails & b-roll stills', color: 'var(--c-image)', icon: 'image', steps: ['sparkles', 'image', 'maximize-2', 'package'], profiles: ['video', 'social', 'podcast'],
    text: 'Describe the shot → 16:9 / 9:16 / square takes → upscale → export as JPEG or PNG at platform size.', go: () => S.navigate('image') },
  { id: 'sting', title: 'Stings, jingles & beds', color: 'var(--c-sound)', icon: 'music', steps: ['sparkles', 'music', 'gauge', 'package'], profiles: ['video', 'podcast', 'social'],
    text: 'Intro / outro sting, ad jingle or ambience bed → pick a take → export normalised to -14 / -16 LUFS.', go: () => S.navigate('sound', 'music') },
  { id: 'voiceover', title: 'Voice-over & narration', color: 'var(--c-sound)', icon: 'mic', steps: ['sparkles', 'mic', 'gauge', 'package'], profiles: ['video', 'podcast', 'social', 'film'],
    text: 'Write or paste the script → pick or design a voice → takes → loudness-matched export.', go: () => S.navigate('sound', 'voice') },
  { id: 'storyboard', title: 'Storyboard & concept art', color: 'var(--c-image)', icon: 'film', steps: ['sparkles', 'image', 'columns-2', 'package'], profiles: ['film', 'video'],
    text: 'Scene description → 16:9 frames → compare and pick → export the board as PNG.', go: () => S.navigate('image') },
  { id: 'iterate', title: 'Iterate on an asset', color: 'var(--c-image)', icon: 'history', steps: ['eye', 'sliders-horizontal', 'refresh-cw', 'columns-2'],
    text: 'Open any asset, see exactly how it was made, tweak one knob, regenerate, compare A/B.', go: () => S.navigate('library') },
];

const TARGETS = [
  { value: 'image', label: 'Image', style: 'qwen_image', studio: 'image', task: 't2i' },
  { value: 'image3d', label: '3D-ready', style: 'image3d', studio: 'image', task: 't2i' },
  { value: 'sfx', label: 'SFX', style: 'sfx', studio: 'sound', task: 'sfx' },
  { value: 'music', label: 'Music', style: 'sa3_music', studio: 'sound', task: 'music' },
  { value: 'song', label: 'Song', style: 'minimax_music', studio: 'sound', task: 'song' },
  { value: 'voice', label: 'Voice', style: 'voice', studio: 'sound', task: 'voice_design' },
];

const PLACEHOLDER = {
  game: 'Type an idea in Thai or English, e.g. “หีบสมบัติไม้เก่า ประดับทอง” or “boss fight in a lava cave”',
  video: 'Type an idea, e.g. “warm sunrise over a harbour, slow drone shot” or “punchy 5-second logo sting”',
  podcast: 'Type an idea, e.g. “cosy cover art for a late-night history podcast” or “warm 6-second intro sting”',
  social: 'Type an idea, e.g. “vertical product shot of a matte black bottle, neon rim light” or “upbeat 15-second ad jingle”',
  film: 'Type an idea, e.g. “storyboard frame: detective at a rain-soaked window, low key light” or “tense strings, sparse”',
  general: 'Type an idea in Thai or English, e.g. “a lighthouse at dusk, painterly” or “soft piano bed for a documentary”',
};

const rank = (profiles, prof) => (!profiles ? 1 : profiles.includes(prof) ? 0 : 2);

function greeting() { const hr = new Date().getHours(); return hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening'; }

export default S.defineStudio({
  id: 'home', title: 'Home', icon: 'home', order: 0, section: 'home', mobile: true,
  async mount(el) {
    const statsEl = h('div.stats');
    const heroTitle = h('h1');
    const hero = h('section.hero', h('div.row.wrap', { style: { alignItems: 'flex-start', gap: '20px' } },
      h('div.grow', heroTitle, h('p', 'Images, 3D, sound effects, music and voices for games, video, podcasts, social and film, from open-weight models running on this machine. Every result keeps its full recipe, and exports are normalised for where they will be used.')),
      statsEl));

    // idea box
    let target = S.prefs.get('home.target', 'image');
    const tgt = () => TARGETS.find(t => t.value === target) || TARGETS[0];
    const box = new S.PromptBox({ memoryKey: 'home', placeholder: PLACEHOLDER[S.store.profile] || PLACEHOLDER.general, style: tgt().style,
      onSubmit: () => box.run() });
    const segT = S.seg(TARGETS.map(t => ({ value: t.value, label: t.label })), target, v => { target = v; S.prefs.set('home.target', v); box.setStyle(tgt().style); });
    const sendBtn = btn('Open in studio', { icon: 'arrow-right', onClick: () => {
      const t = tgt(); if (!box.value) { box.focus(); return; }
      S.sendTo(t.studio, [], { action: 'prompt', sub: t.task, params: { prompt: box.value, task: t.task, ...(box.improved?.lyrics ? { lyrics: box.improved.lyrics } : {}), ...(box.improved?.line ? { text: box.improved.line } : {}) } });
    } });
    const idea = h('section.card.card-pad.col', { style: { gap: '12px' } },
      h('div.section-head', { style: { margin: 0 } }, h('h3', ic('sparkles', 17), 'Idea box'), h('span.spacer'), h('span.dim.hide-sm', { style: { fontSize: '12.5px' } }, 'Write it once, get a model-ready prompt')),
      h('div.row.wrap', h('span.field-label', 'For'), segT), box.el, h('div.row', h('span.spacer'), sendBtn));

    const flowsEl = h('div.flows');
    const drawFlows = () => {
      const prof = S.store.profile;
      const sorted = FLOWS.map((f, i) => [f, i]).filter(([f]) => prof === 'general' || rank(f.profiles, prof) < 2 || ['sfx', 'music', 'iterate'].includes(f.id))
        .sort((a, b) => (prof === 'general' ? 0 : rank(a[0].profiles, prof) - rank(b[0].profiles, prof)) || a[1] - b[1]).map(x => x[0]);
      flowsEl.replaceChildren(...sorted.map(f => h('div.flow', { tabindex: 0, role: 'button', style: { '--fc': f.color }, on: { click: f.go, keydown: e => e.key === 'Enter' && f.go() } },
        h('div.row', h('div.fi', ic(f.icon, 17)), h('h4.grow', f.title), ic('arrow-right', 16, 'dim')),
        h('p', f.text), h('div.steps', ...f.steps.flatMap((s2, i) => [i ? ic('chevron-right', 12) : null, ic(s2, 14)])))));
      box.ta.placeholder = PLACEHOLDER[prof] || PLACEHOLDER.general;
    };
    const profEl = h('div.profile-strip');
    const drawProfiles = () => {
      profEl.replaceChildren(...Object.values(S.store.profiles).map(p => h(`button.profile-card${p.id === S.store.profile ? '.on' : ''}`, { type: 'button', title: p.summary,
        on: { click: async () => { try { await S.setProfile(p.id); } catch (e) { S.toastError(e); } } } }, h('b', ic(p.icon || 'layers', 16), p.label), h('span', p.summary))));
    };
    S.bus.on('profile', () => { drawProfiles(); drawFlows(); });
    drawProfiles(); drawFlows();
    const flows = flowsEl;

    // recent
    const grid = new S.ResultsGrid({ selectable: false, onOpen: a => S.openAsset(a, { list: grid.assets().map(x => x.id) }),
      emptyState: S.empty('library', 'No assets yet', 'Generate something, upload files, or run a self-test from Queue & System.',
        btn('Run a self-test', { icon: 'flask-conical', onClick: () => S.navigate('selftest') })) });
    grid.gridEl.style.setProperty('--card-min', '150px');
    let lastTotal;
    const loadRecent = async () => { try { const r = await S.api.assets({ project: S.store.project, limit: 12 }); grid.set(r.items); lastTotal = r.total; renderStats(r.total); } catch (e) { S.toastError(e); } };
    S.bus.on('asset', ev => { if (ev.asset.project !== S.store.project) return; if (ev.action === 'created') { grid.add(ev.asset); lastTotal = (lastTotal || 0) + 1; renderStats(lastTotal); const n = grid.assets(); if (n.length > 12) grid.remove(n[n.length - 1].id); } });
    S.bus.on('project', loadRecent);

    // side: queue + memory + readiness
    const jobs = new S.JobList({ limit: 5, compact: true, emptyText: 'Jobs you start appear here with live progress.' });
    const memEl = h('div.col', { style: { gap: '10px' } });
    const drawMem = () => {
      const m = S.store.memory; if (!m) return;
      const used = m.total_gb - m.available_gb;
      memEl.replaceChildren(h('div.row', h('div.stat.grow', h('b', `${Math.round(m.available_gb)} GB`), h('span', `free of ${Math.round(m.total_gb)} GB unified memory`)),
        h('div.stat', { style: { textAlign: 'right' } }, h('b', `${m.ours_gb} GB`), h('span', 'studio models'))),
      h('div.mem-bar', { style: { height: '8px' } }, h('i.other', { style: { width: `${(used / m.total_gb) * 100}%` } }), h('i.ours', { style: { width: `${(Math.min(m.ours_gb, used) / m.total_gb) * 100}%` } })),
      m.loaded.length ? h('div.col', { style: { gap: '4px' } }, ...m.loaded.map(e => h('div.row', h('span.status-dot.ready'), h('span.ellipsis.grow', e.label), h('span.dim.mono', `${e.gb} GB`)))) : h('div.dim', { style: { fontSize: '12.5px' } }, 'No studio model is loaded. Models load on first use and unload when idle.'));
    };
    S.bus.on('system', drawMem); drawMem();
    const readyEl = h('div.col', { style: { gap: '8px' } });
    const drawReady = () => {
      const groups = [['image', 'Image', 'var(--c-image)', 'image'], ['mesh', '3D', 'var(--c-3d)', 'cuboid'], ['audio', 'Sound', 'var(--c-sound)', 'audio-lines'], ['text', 'Prompt LLM', 'var(--c-text)', 'sparkles']];
      readyEl.replaceChildren(...groups.map(([mod, label, color, i]) => {
        const ms = S.store.models.filter(m => m.modality === mod && !m.test);
        const ok = ms.filter(m => m.status.status === 'ready').length;
        const soon = ms.filter(m => m.status.code === 'no_worker').length;
        return h('div.row', { style: { color } }, ic(i, 16), h('span.grow', { style: { color: 'var(--text)' } }, label),
          h('span.badge.ok', `${ok} ready`), soon ? h('span.badge.outline', `${soon} coming`) : null, ms.length - ok - soon ? h('span.badge.warn', `${ms.length - ok - soon} missing`) : null);
      }), h('div', btn('Open models', { size: 'sm', kind: 'ghost', icon: 'arrow-right', onClick: () => S.navigate('models') })));
    };
    S.bus.on('models', drawReady); drawReady();

    const renderStats = total => {
      const p = S.store.projects.find(x => x.id === S.store.project);
      heroTitle.textContent = `${greeting()} · ${p?.title || S.store.project}`;
      const ready = S.store.models.filter(m => m.status.status === 'ready' && !m.test).length;
      statsEl.replaceChildren(h('div.stat', h('b', String(total ?? '–')), h('span', 'assets in project')), h('div.stat', h('b', String(ready)), h('span', 'models ready')),
        h('div.stat', h('b', String((S.store.counts?.running || 0) + (S.store.counts?.queued || 0))), h('span', 'jobs active')));
    };
    renderStats();

    el.append(h('div.page-pad.col', { style: { gap: '18px' } }, hero,
      h('div.home-grid',
        h('div.col', { style: { gap: '18px' } }, idea,
          h('section', h('div.section-head', h('h3.section-title', ic('layers', 17), 'What are you making?'), h('span.spacer'), h('span.dim.hide-sm', { style: { fontSize: '12.5px' } }, 'Sets style presets, aspect ratios and export defaults')), profEl),
          h('section', h('div.section-head', h('h3.section-title', ic('rocket', 17), 'Quick start')), flows),
          h('section', h('div.section-head', h('h3.section-title', ic('history', 17), 'Recent assets'), h('span.spacer'), btn('Library', { size: 'sm', kind: 'ghost', icon: 'arrow-right', onClick: () => S.navigate('library') })), grid.el)),
        h('div.col', { style: { gap: '16px' } },
          h('section.card.card-pad', h('div.section-head', h('h3', ic('list-checks', 16), 'Queue'), h('span.spacer'), btn('All jobs', { size: 'sm', kind: 'ghost', icon: 'arrow-right', onClick: () => S.navigate('system') })), jobs.el),
          h('section.card.card-pad', h('div.section-head', h('h3', ic('memory-stick', 16), 'Memory')), memEl),
          h('section.card.card-pad', h('div.section-head', h('h3', ic('cpu', 16), 'Models')), readyEl)))));
    loadRecent();
    return { show: () => renderStats(lastTotal) };
  },
});
