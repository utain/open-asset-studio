// Queue & System: jobs (cancel / retry), loaded models (unload), workers, prompt LLM, memory, logs, self-test.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

export default S.defineStudio({
  id: 'system', title: 'Queue & System', icon: 'list-checks', order: 90, section: 'system',
  async mount(el) {
    // ---------------- jobs
    let jobFilter = S.prefs.get('system.jobs', 'active');
    const FILTERS = { active: ['running', 'queued'], failed: ['failed'], done: ['done'], all: null };
    const jobsBox = h('div');
    let jl = null;
    const drawJobs = () => { jl?.destroy(); jl = new S.JobList({ statuses: FILTERS[jobFilter], limit: 60, emptyText: jobFilter === 'active' ? 'Nothing is running. Start a job in a studio or run a self-test.' : 'No jobs in this list.' }); jobsBox.replaceChildren(jl.el); };
    const jobSeg = S.seg([{ value: 'active', label: 'Active' }, { value: 'failed', label: 'Failed' }, { value: 'done', label: 'Done' }, { value: 'all', label: 'All' }], jobFilter, v => { jobFilter = v; S.prefs.set('system.jobs', v); drawJobs(); });
    drawJobs();
    const clearBtn = btn('Clear finished', { size: 'sm', kind: 'ghost', icon: 'trash-2', onClick: async () => {
      if (!(await S.confirmDialog('Remove all finished jobs from the history? Assets are kept.', { ok: 'Clear' }))) return;
      await S.api.clearJobs().catch(S.toastError); drawJobs();
    } });

    // ---------------- memory + loaded models + workers + llm
    const memBox = h('div.col', { style: { gap: '12px' } });
    const workersBox = h('div');
    const llmBox = h('div.col', { style: { gap: '10px' } });
    const drawMem = (sys) => {
      const m = sys?.memory || S.store.memory; if (!m) return;
      const used = m.total_gb - m.available_gb; const ours = Math.min(m.ours_gb, used);
      memBox.replaceChildren(
        h('div.row.wrap', { style: { gap: '22px' } }, h('div.stat', h('b', `${m.available_gb} GB`), h('span', 'available (MemAvailable)')), h('div.stat', h('b', `${m.total_gb} GB`), h('span', 'unified memory')),
          h('div.stat', h('b', `${m.ours_gb} GB`), h('span', 'studio models (estimate)')), h('div.stat', h('b', `${Math.max(0, used - ours).toFixed(1)} GB`), h('span', 'other programs + cache'))),
        h('div.mem-bar', { style: { height: '10px' } }, h('i.other', { style: { width: `${(used / m.total_gb) * 100}%` } }), h('i.ours', { style: { width: `${(ours / m.total_gb) * 100}%` } })),
        h('div.row.nowrap', { style: { fontSize: '12px', gap: '14px' } }, h('span.row', { style: { gap: '5px' } }, h('span', { style: { width: '10px', height: '10px', borderRadius: '3px', background: 'var(--accent)' } }), 'studio'),
          h('span.row', { style: { gap: '5px' } }, h('span', { style: { width: '10px', height: '10px', borderRadius: '3px', background: 'var(--text-3)', opacity: .55 } }), 'everything else'),
          h('span.spacer'), h('span.dim.ellipsis', { title: `Jobs keep ${m.margin_gb} GB free on top of a model's need` }, `+${m.margin_gb} GB safety margin`)),
        m.loaded.length ? h('div.tbl-wrap', h('table.tbl', h('thead', h('tr', ...['Loaded', 'Kind', 'Memory', ''].map(x => h('th', x)))),
          h('tbody', ...m.loaded.map(e => h('tr', h('td', h('b', e.label), h('div.dim', { style: { fontSize: '11.5px' } }, `${e.key} · used ${S.fmtAgo(new Date(e.last_used * 1000).toISOString())}`)), h('td', e.kind === 'llm' ? h('span.badge.info', 'LLM') : h('span.badge.outline', 'worker')),
            h('td.mono.nowrap', `${e.gb} GB`),
            h('td', e.busy ? h('span.badge.accent', 'busy') : btn('Unload', { size: 'sm', icon: 'download', onClick: async () => {
              const [backend, model] = e.key.split(':');
              try { await S.api.unload(backend === 'llm' ? { llm: true } : { backend, model }); S.toast(`${e.label} unloaded`, { kind: 'ok', timeout: 2000 }); refresh(); } catch (err) { S.toastError(err); }
            } }))))))) : h('div.dim', 'No studio model is loaded right now.'),
        m.recent_evictions?.length ? h('details', h('summary.dim', { style: { cursor: 'pointer', fontSize: '12.5px' } }, `Recent automatic unloads (${m.recent_evictions.length})`),
          h('div.col', { style: { gap: '4px', marginTop: '6px', fontSize: '12.5px' } }, ...m.recent_evictions.slice().reverse().map(ev => h('div.row', h('span.mono.dim', ev.at), h('span.grow', `${ev.label} (${ev.gb} GB)`), h('span.dim', ev.reason))))) : null);
    };
    const drawWorkers = sys => {
      const ws = sys?.workers || [];
      workersBox.replaceChildren(ws.length ? h('div.tbl-wrap', h('table.tbl', h('thead', h('tr', ...['Backend', 'Process', 'Loaded models', 'State', ''].map(x => h('th', x)))),
        h('tbody', ...ws.map(w => h('tr', h('td', h('b', w.backend)), h('td', w.running ? h('span.mono', `pid ${w.pid} · ${w.rss_gb} GB RSS`) : h('span.dim', `stopped${w.last_exit ? ` (${w.last_exit})` : ''}`)),
          h('td', Object.keys(w.loaded).join(', ') || h('span.dim', '–')), h('td', w.busy ? h('span.badge.accent', 'running a job') : w.running ? h('span.badge.ok', 'idle') : h('span.badge.outline', 'off')),
          h('td.nowrap', btn('Log', { size: 'sm', kind: 'ghost', icon: 'terminal', onClick: () => { logSel.value = w.log.replace(/\.log$/, ''); loadLog(); logCard.scrollIntoView({ behavior: 'smooth' }); } }),
            w.running && !w.busy ? btn('Stop', { size: 'sm', icon: 'square', onClick: async () => { try { await S.api.unload({ backend: w.backend }); refresh(); } catch (e) { S.toastError(e); } } }) : null))))))
        : h('div.dim', 'No worker process has been started yet. Workers start on the first job of their backend and exit when idle.'));
    };
    const drawLlm = l => {
      l = l || S.store.llm || {};
      const st = { ready: 'ok', sleeping: 'info', starting: 'warn', failed: 'err', external: 'info', stopped: 'outline' }[l.state] || 'outline';
      llmBox.replaceChildren(h('div.row.wrap', h(`span.badge.${st}`, l.state || 'unknown'), h('b.grow', l.model || l.primary || 'Prompt LLM'), l.vision ? h('span.badge.outline', 'vision') : null),
        h('div.dim', { style: { fontSize: '12.5px' } }, l.state === 'sleeping' ? 'Weights released after idle (0.2 GB); wakes in ~10 s on the next request.' : l.state === 'ready' ? `Serving on ${l.url}${l.idle_s != null ? ` · idle ${S.fmtDur(l.idle_s)}` : ''}` : l.state === 'external' ? `Using a llama-server started by another studio instance at ${l.url}.` : l.error || `Starts on the first “Improve” click (primary: ${l.primary}; fallback: ${l.fallback}).`),
        h('div.row', l.state === 'stopped' || l.state === 'failed' ? btn('Start now', { size: 'sm', icon: 'play', onClick: async e => { e.currentTarget.disabled = true; try { drawLlm(await S.api.llmWarmup()); } catch (err) { S.toastError(err); } } }) : null,
          ['ready', 'sleeping', 'starting'].includes(l.state) ? btn('Stop', { size: 'sm', icon: 'square', onClick: async () => { try { drawLlm(await S.api.llmRelease()); } catch (err) { S.toastError(err); } } }) : null));
    };
    const refresh = async () => { try { const s = await S.api.system(); drawMem(s); drawWorkers(s); drawLlm(s.llm); } catch (e) { /* offline */ } };
    S.bus.on('system', sys => { if (el.isConnected && !el.hidden) { drawMem(sys); drawLlm(sys.llm); } });
    S.bus.on('llm', l => drawLlm(l));
    let wt = null;

    // ---------------- logs
    const logSel = h('select.select', { style: { width: 'auto' }, 'aria-label': 'Log file', on: { change: () => loadLog() } });
    const logBox = h('pre.logbox', 'Loading…');
    let follow = S.prefs.get('system.follow', true);
    const followT = S.toggle(follow, v => { follow = v; S.prefs.set('system.follow', v); }, 'Follow');
    const loadLog = async () => {
      try {
        let r = await S.api.logs(logSel.value || 'studio', 400);
        if (!r.available.includes(r.name) && r.available.length) r = await S.api.logs(r.available.includes('studio') ? 'studio' : r.available[0], 400);
        const cur = r.name;
        logSel.replaceChildren(...(r.available.length ? r.available : ['studio']).map(n => h('option', { value: n, selected: n === cur }, n)));
        const atBottom = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 30;
        logBox.textContent = r.lines.length ? r.lines.join('\n') : '(empty)';
        if (atBottom || follow) logBox.scrollTop = logBox.scrollHeight;
      } catch (e) { logBox.textContent = e.message; }
    };
    const logCard = h('section.card.card-pad', h('div.section-head', h('h3', ic('terminal', 16), 'Logs'), h('span.spacer'), logSel, h('label.row', { style: { gap: '6px', fontSize: '12.5px', color: 'var(--text-2)' } }, followT, 'Follow'),
      btn('', { size: 'sm', kind: 'ghost', icon: 'refresh-cw', title: 'Reload', onClick: () => loadLog() })), logBox);

    // ---------------- self-test
    const selftest = h('section.card.card-pad.col', { style: { gap: '10px' } },
      h('div.section-head', { style: { margin: 0 } }, h('h3', ic('flask-conical', 16), 'Self-test')),
      h('div.dim', { style: { fontSize: '12.5px' } }, 'Runs the CPU reference worker through the real queue → worker → SSE → library path. Outputs are tagged “self-test”.'),
      h('div.row.wrap', btn('Audio self-test', { icon: 'audio-lines', onClick: () => run('audio') }), btn('Image self-test', { icon: 'image', onClick: () => run('image') }),
        btn('Diagnostics studio', { kind: 'ghost', icon: 'arrow-right', onClick: () => S.navigate('selftest') })));
    async function run(kind) {
      try { const j = await S.api.selftest(kind, S.store.project); S.toast(`Self-test (${kind}) queued`, { kind: 'info', timeout: 1800 });
        S.waitJob(j.id).then(jj => S.toast(jj.status === 'done' ? `${jj.outputs.length} asset(s) in the library` : (jj.error?.message || jj.status), { kind: jj.status === 'done' ? 'ok' : 'err', title: `Self-test ${jj.status}`,
          action: jj.status === 'done' ? { label: 'Open', onClick: () => S.openAsset(jj.outputs[0], { list: jj.outputs }) } : null }));
      } catch (e) { S.toastError(e); }
    }

    // ---------------- studio settings: use-case profile, export default, commercial-safe filter
    const settingsBox = h('div.col', { style: { gap: '14px' } });
    const drawSettings = () => {
      const prof = S.getProfile();
      const profSel = h('select.select', { 'aria-label': 'Profile', on: { change: async e => { try { await S.setProfile(e.target.value); S.toast(`Profile: ${S.getProfile().label}`, { kind: 'ok', timeout: 1800 }); } catch (err) { S.toastError(err); } } } },
        ...Object.values(S.store.profiles).map(p => h('option', { value: p.id, selected: p.id === S.store.profile }, p.label)));
      const presetIds = Object.keys(S.store.exportPresets);
      const expSel = h('select.select', { 'aria-label': 'Default export preset', on: { change: async e => { try { await S.patchProfileSettings({ export: { preset: e.target.value || null } }); } catch (err) { S.toastError(err); } } } },
        h('option', { value: '', selected: !S.store.exportOverride }, `Profile default (${S.store.exportPresets[prof.export]?.label || prof.export})`),
        ...presetIds.map(id => h('option', { value: id, selected: S.store.exportOverride === id }, S.store.exportPresets[id].label)));
      const comm = S.toggle(S.store.license.commercial_only, async v => { try { await S.setCommercialOnly(v); } catch (err) { S.toastError(err); comm.input.checked = !v; } }, 'Commercial-safe models only');
      const strict = S.toggle(S.store.license.strict, async v => { try { await S.setCommercialOnly(S.store.license.commercial_only, v); } catch (err) { S.toastError(err); strict.input.checked = !v; } }, 'Also refuse models whose licence must be checked');
      settingsBox.replaceChildren(
        h('div.set-grid',
          h('label.field', h('span.field-label', 'Profile'), profSel, h('span.field-help', prof.summary)),
          h('label.field', h('span.field-label', 'Default export'), expSel, h('span.field-help', 'Used by plain Export / Export zip. Originals are never modified; conversions are derived files.'))),
        h('div.col', { style: { gap: '10px' } },
          h('label.set-row', h('span', h('b', 'Commercial-safe models only'), h('div.dim', { style: { fontSize: '12.5px' } }, 'Hide non-commercial models in every picker and make the API refuse them (HTTP 403, kind “license”).')), comm),
          h('label.set-row', h('span', h('b', 'Strict'), h('div.dim', { style: { fontSize: '12.5px' } }, 'Also refuse models whose registry entry says the licence must be verified.')), strict)));
    };
    for (const ev of ['profile', 'license', 'export']) S.bus.on(ev, drawSettings);
    drawSettings();

    el.append(h('div.page-pad.col', { style: { gap: '16px' } },
      h('div.section-head', { style: { margin: 0 } }, h('div.grow', h('h1.page-title', 'Queue & System'), h('p.page-sub', 'One GPU job runs at a time; CPU tools run alongside. Models unload when idle or when memory is needed.'))),
      h('div.home-grid',
        h('div.col', { style: { gap: '16px' } },
          h('section.card.card-pad', h('div.section-head', h('h3', ic('list-checks', 16), 'Jobs'), h('span.spacer'), jobSeg, clearBtn), jobsBox),
          h('section.card.card-pad', h('div.section-head', h('h3', ic('download', 16), 'Downloads')), S.downloadsList()),
          logCard),
        h('div.col', { style: { gap: '16px' } },
          h('section.card.card-pad', h('div.section-head', h('h3', ic('sliders-horizontal', 16), 'Studio settings')), settingsBox),
          h('section.card.card-pad', h('div.section-head', h('h3', ic('memory-stick', 16), 'Memory'), h('span.spacer'),
            btn('Unload all', { size: 'sm', kind: 'ghost', icon: 'download', onClick: async () => { try { await S.api.unload({ all: true }); S.toast('Idle models unloaded', { kind: 'ok', timeout: 2000 }); refresh(); } catch (e) { S.toastError(e); } } })), memBox),
          h('section.card.card-pad', h('div.section-head', h('h3', ic('sparkles', 16), 'Prompt improver LLM')), llmBox),
          h('section.card.card-pad', h('div.section-head', h('h3', ic('server', 16), 'Workers')), workersBox),
          selftest))));
    await refresh(); loadLog();
    return {
      show() { refresh(); loadLog(); clearInterval(wt); wt = setInterval(() => { refresh(); if (follow) loadLog(); }, 4000); },
      hide() { clearInterval(wt); },
    };
  },
});
