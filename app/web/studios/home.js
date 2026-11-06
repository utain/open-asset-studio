// Home: quick-start flows, an idea box with the prompt improver, recent assets, queue, memory, model readiness.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const FLOWS = [
  { id: 'concept3d', title: 'Concept → 3D prop', color: 'var(--c-3d)', icon: 'cuboid', steps: ['sparkles', 'image', 'scissors', 'cuboid', 'package'],
    text: 'Idea → 4 fast concepts → refine → transparent cut-out → textured GLB.', go: () => S.navigate('pipelines', 'concept3d') },
  { id: 'character', title: 'Character kit', color: 'var(--accent)', icon: 'user', steps: ['image', 'cuboid', 'mic', 'audio-lines'],
    text: 'Concept art → 3D model → a designed voice → barks and SFX for it.', go: () => S.navigate('pipelines', 'character') },
  { id: 'sfx', title: 'SFX pass', color: 'var(--c-sound)', icon: 'audio-lines', steps: ['list-checks', 'shuffle', 'star', 'scissors', 'package'],
    text: 'Paste an event list → N takes each → audition, rate, trim, loop → export for Godot/Unity.', go: () => S.navigate('sound', 'sfx') },
  { id: 'music', title: 'Music & loops', color: 'var(--c-sound)', icon: 'music', steps: ['sparkles', 'music', 'repeat', 'package'],
    text: 'Mood + genre (+ lyrics) → takes → extend / seamless loop → export.', go: () => S.navigate('sound', 'music') },
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

function greeting() { const hr = new Date().getHours(); return hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening'; }

export default S.defineStudio({
  id: 'home', title: 'Home', icon: 'home', order: 0, section: 'home', mobile: true,
  async mount(el) {
    const statsEl = h('div.stats');
    const heroTitle = h('h1');
    const hero = h('section.hero', h('div.row.wrap', { style: { alignItems: 'flex-start', gap: '20px' } },
      h('div.grow', heroTitle, h('p', 'Concept art, 3D props, sound effects, music and voices from open-weight models running on this machine. Every result keeps its full recipe.')),
      statsEl));

    // idea box
    let target = S.prefs.get('home.target', 'image');
    const tgt = () => TARGETS.find(t => t.value === target) || TARGETS[0];
    const box = new S.PromptBox({ memoryKey: 'home', placeholder: 'Type an idea in Thai or English, e.g. “หีบสมบัติไม้เก่า ประดับทอง” or “boss fight in a lava cave”', style: tgt().style,
      onSubmit: () => box.run() });
    const segT = S.seg(TARGETS.map(t => ({ value: t.value, label: t.label })), target, v => { target = v; S.prefs.set('home.target', v); box.setStyle(tgt().style); });
    const sendBtn = btn('Open in studio', { icon: 'arrow-right', onClick: () => {
      const t = tgt(); if (!box.value) { box.focus(); return; }
      S.sendTo(t.studio, [], { action: 'prompt', sub: t.task, params: { prompt: box.value, task: t.task, ...(box.improved?.lyrics ? { lyrics: box.improved.lyrics } : {}), ...(box.improved?.line ? { text: box.improved.line } : {}) } });
    } });
    const idea = h('section.card.card-pad.col', { style: { gap: '12px' } },
      h('div.section-head', { style: { margin: 0 } }, h('h3', ic('sparkles', 17), 'Idea box'), h('span.spacer'), h('span.dim.hide-sm', { style: { fontSize: '12.5px' } }, 'Write it once, get a model-ready prompt')),
      h('div.row.wrap', h('span.field-label', 'For'), segT), box.el, h('div.row', h('span.spacer'), sendBtn));

    const flows = h('div.flows', ...FLOWS.map(f => h('div.flow', { tabindex: 0, role: 'button', style: { '--fc': f.color }, on: { click: f.go, keydown: e => e.key === 'Enter' && f.go() } },
      h('div.row', h('div.fi', ic(f.icon, 17)), h('h4.grow', f.title), ic('arrow-right', 16, 'dim')),
      h('p', f.text), h('div.steps', ...f.steps.flatMap((s, i) => [i ? ic('chevron-right', 12) : null, ic(s, 14)])))));

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
