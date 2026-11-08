// Models: capability matrix of the registry, status + fix, license, memory, speed, default per task, backends.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const MODS = [{ value: '', label: 'All' }, { value: 'image', label: 'Image' }, { value: 'mesh', label: '3D' }, { value: 'audio', label: 'Sound' }, { value: 'text', label: 'LLM' }];

export default S.defineStudio({
  id: 'models', title: 'Models', icon: 'cpu', order: 80, section: 'system',
  async mount(el) {
    const f = { mod: '', q: '', lic: '', status: '', ...S.prefs.get('models.filters', {}) };
    if (f.commercial) { f.lic = f.lic || 'commercial'; } delete f.commercial;
    let backends = [];
    const body = h('div.col', { style: { gap: '14px' } });
    const summary = h('div.row.wrap');
    const q = h('input.input', { type: 'search', placeholder: 'Search models, tasks, features…', value: f.q, style: { maxWidth: '280px' }, on: { input: S.debounce(() => { f.q = q.value; render(); }, 150) } });
    const modSeg = S.seg(MODS, f.mod, v => { f.mod = v; render(); });
    const statusSel = h('select.select', { style: { width: 'auto' }, 'aria-label': 'Status', on: { change: e => { f.status = e.target.value; render(); } } },
      ...[['', 'Any status'], ['ready', 'Ready'], ['missing', 'Missing weights / env'], ['no_worker', 'Coming soon'], ['broken', 'Broken'], ['cloud', 'Cloud (API)'], ['local', 'Local only']].map(([v, l]) => h('option', { value: v, selected: f.status === v }, l)));
    const licSel = h('select.select', { style: { width: 'auto' }, 'aria-label': 'Licence', on: { change: e => { f.lic = e.target.value; render(); } } },
      ...[['', 'Any licence'], ['commercial', 'Commercial OK'], ['check', 'Check licence'], ['noncommercial', 'Non-commercial'], ['usable', 'Commercial OK + check']].map(([v, l]) => h('option', { value: v, selected: f.lic === v }, l)));
    const comm = S.toggle(S.store.license.commercial_only, async v => {
      try { await S.setCommercialOnly(v); S.toast(v ? 'Non-commercial models are now hidden in the pickers and refused by the API.' : 'All models are available again.', { kind: 'ok', timeout: 2600 }); } catch (e) { S.toastError(e); comm.input.checked = !v; }
    }, 'Commercial-safe models only');
    S.bus.on('license', () => { comm.input.checked = S.store.license.commercial_only; });
    const refresh = btn('Re-check', { icon: 'refresh-cw', onClick: async () => {
      refresh.disabled = true;
      try { const r = await S.api.refreshModels(); backends = r.backends; await S.loadModels(); S.toast('Model status re-checked', { kind: 'ok', timeout: 1800 }); } catch (e) { S.toastError(e); }
      refresh.disabled = false;
    } });

    const match = m => {
      if (m.test && !S.store.ui.show_test_models && f.mod !== 'test') { if (!f.q.includes('test')) return false; }
      if (f.mod && m.modality !== f.mod) return false;
      if (f.lic === 'usable' ? S.licenseState(m.license) === 'noncommercial' : f.lic && S.licenseState(m.license) !== f.lic) return false;
      const st = m.status.status, code = m.status.code;
      if (f.status === 'ready' && st !== 'ready') return false;
      if (f.status === 'missing' && !['missing_weights', 'missing_env'].includes(st)) return false;
      if (f.status === 'no_worker' && code !== 'no_worker') return false;
      if (f.status === 'broken' && (st !== 'broken' || code === 'no_worker')) return false;
      if (f.status === 'cloud' && !m.cloud) return false;
      if (f.status === 'local' && m.cloud) return false;
      if (f.q) {
        const hay = [m.name, m.id, m.summary, m.tasks.join(' '), (m.tags || []).join(' '), m.license?.name, Object.keys(m.features || {}).join(' ')].join(' ').toLowerCase();
        if (!f.q.toLowerCase().split(/\s+/).every(t => hay.includes(t))) return false;
      }
      return true;
    };
    const defStar = (m, t) => {
      const on = S.store.defaults[t] === m.id;
      const proj = S.isOverridden(`defaults.${t}`);      // this project has its own default for the task: edit that one
      const b = h(`button.def-star${on ? '.on' : ''}`, { type: 'button', title: (on ? `Default for ${t}` : `Make default for ${t}`) + (proj ? ' (set by this project)' : ''), on: { click: async e => {
        e.stopPropagation();
        try { await S.api.setDefault(t, on ? null : m.id, proj ? S.store.project : null); await Promise.all([S.loadModels(), S.refreshEffective()]); render(); S.toast(on ? `Default for ${t} reset` : `${m.name} is now the default for ${S.store.tasks[t]?.label || t}`, { kind: 'ok', timeout: 2200 }); } catch (err) { S.toastError(err); }
      } } }, ic('star', 13));
      return b;
    };
    const taskChips = m => h('div.chips', ...m.tasks.filter(t => t !== 'test').map(t => h('span.chip', { style: { height: '24px', cursor: 'default', gap: '3px', paddingRight: '4px' } }, S.store.tasks[t]?.label || t, defStar(m, t))));
    const statusCell = m => {
      const st = m.status;
      if (m.cloud && st.status !== 'ready') return h('div', S.statusBadge(st), h('div.dim', { style: { fontSize: '12px', marginTop: '4px' } }, st.reason),
        btn('Providers', { size: 'sm', icon: 'key', onClick: () => S.navigate('providers') }));
      const fix = st.fix && st.code !== 'no_worker' ? h('div.fixwrap', h('code.fixcode', st.fix), btn('', { size: 'sm', icon: 'copy', title: 'Copy fix command', onClick: () => S.copyText(st.fix) })) : null;
      const dl = S.canDownload(m) || S.store.downloads.has(m.id) ? S.downloadWidget(m) : null;
      return h('div', S.statusBadge(st), st.reason && st.status !== 'ready' ? h('div.dim', { style: { fontSize: '12px', marginTop: '4px', overflowWrap: 'anywhere' } }, st.reason) : null, dl, fix);
    };

    function render() {
      S.prefs.set('models.filters', { ...f, q: '' });
      const all = S.store.models;
      const list = all.filter(match);
      const real = all.filter(m => !m.test);
      const n = s => real.filter(s).length;
      summary.replaceChildren(
        h('span.badge.ok', `${n(m => m.status.status === 'ready')} ready`),
        h('span.badge.outline', `${n(m => m.status.code === 'no_worker')} coming soon`),
        h('span.badge.warn', `${n(m => ['missing_weights', 'missing_env'].includes(m.status.status))} missing`),
        h('span.badge.err', `${n(m => m.status.status === 'broken' && m.status.code !== 'no_worker')} broken`),
        h('span.badge.ok', { title: 'Licence allows commercial use' }, ic('shield-check', 12), `${n(m => S.licenseState(m.license) === 'commercial')} commercial OK`),
        h('span.badge.warn', { title: 'Commercial use looks allowed but the terms must be verified' }, ic('triangle-alert', 12), `${n(m => S.licenseState(m.license) === 'check')} check licence`),
        h('span.badge.err', { title: 'Research / non-commercial licence' }, ic('ban', 12), `${n(m => S.licenseState(m.license) === 'noncommercial')} non-commercial`),
        h('span.badge.cloud', { title: 'Cloud models: enable and add keys on the Providers page' }, ic('cloud', 12), `${n(m => m.cloud && m.status.status === 'ready')}/${n(m => m.cloud)} cloud ready`));
      const rows = list.map(m => h(`tr.mod-${m.modality}${m.status.code === 'no_worker' ? '.dimrow' : ''}${S.isLicenseBlocked(m) ? '.lic-blocked' : ''}`,
        h('td', h('div.row', { style: { alignItems: 'flex-start' } }, h('div.mp-icon', ic(S.MOD_ICON[m.modality] || 'cpu', 17)),
          h('div', { style: { minWidth: '160px', maxWidth: '300px' } }, h('div.mname', m.name), h('div.dim.mono', { style: { fontSize: '11px' } }, m.id), h('div.muted', { style: { fontSize: '12px', marginTop: '3px' } }, m.summary || ''),
            m.cloud ? h('div.row.wrap', { style: { gap: '4px', marginTop: '4px' } }, S.cloudBadge(m), S.priceBadge(m)) : null))),
        h('td', taskChips(m)),
        h('td', h('div.feat', ...S.featureBadges(m))),
        h('td', S.licenseBadge(m.license), h('div.dim', { style: { fontSize: '11.5px', marginTop: '4px', maxWidth: '170px' } }, m.license?.name), S.isLicenseBlocked(m) ? h('div.err', { style: { fontSize: '11.5px', marginTop: '3px', maxWidth: '170px' } }, 'Disabled by Commercial-safe filter') : null, m.license?.note ? h('div.dim', { style: { fontSize: '11px', maxWidth: '170px' }, title: m.license.note }, ic('info', 11), ' terms') : null),
        h('td.nowrap', m.cloud ? (m.cloud.local ? 'endpoint' : 'cloud') : m.memory_gb ? `${m.memory_gb} GB` : '–', h('div.dim', { style: { fontSize: '11.5px' } }, m.cloud ? 'no GPU' : m.device === 'cpu' ? 'CPU' : 'GPU')),
        h('td', h('div.muted', { style: { fontSize: '12px', maxWidth: '200px' } }, m.speed || '')),
        h('td', statusCell(m))));
      const table = h('div.card.matrix-wrap', h('div.tbl-wrap', h('table.tbl.matrix', h('thead', h('tr', ...[['Model'], ['Tasks (★ = default)', 'models_default'], ['Features'], ['License', 'models_license'], ['Memory', 'models_memory'], ['Speed'], ['Status']].map(([x, k]) => h('th', k ? { style: { whiteSpace: 'nowrap' } } : {}, x, k ? S.uiInfo(k) : null)))), h('tbody', ...rows))));
      const cards = h('div.mcards', ...list.map(m => h(`div.card.card-pad.col.mod-${m.modality}`, { style: { gap: '8px' } },
        h('div.row', h('div.mp-icon', ic(S.MOD_ICON[m.modality] || 'cpu', 17)), h('div.grow', h('div.mname', m.name), h('div.dim', { style: { fontSize: '12px' } }, m.speed || '')), S.statusBadge(m.status)),
        h('div.muted', { style: { fontSize: '12.5px' } }, m.summary), taskChips(m), h('div.feat.row.wrap', S.cloudBadge(m), S.priceBadge(m), S.licenseBadge(m.license), S.memBadge(m), ...S.featureBadges(m)),
        m.status.status !== 'ready' && m.status.code !== 'no_worker' ? statusCell(m) : null)));
      const be = h('section.card.card-pad', h('div.section-head', h('h3', ic('server', 16), 'Backends (one worker process per venv)', S.uiInfo('models_deep_check', { title: 'Backends and Deep check' }))),
        h('div.tbl-wrap', h('table.tbl', h('thead', h('tr', ...['Backend', 'Environment', 'Worker', 'Models', ''].map(x => h('th', x)))),
          h('tbody', ...backends.map(b => h('tr',
            h('td', h('b', b.name), h('div.dim', { style: { fontSize: '12px', maxWidth: '340px' } }, b.label)),
            h('td', b.env_ok ? h('span.badge.ok', 'found') : h('span.badge.warn', 'missing'), h('div.dim.mono', { style: { fontSize: '11px', maxWidth: '320px', overflowWrap: 'anywhere' } }, b.python)),
            h('td', b.kind === 'service' ? h('span.badge.outline', 'service') : b.worker_ok ? h('span.badge.ok', b.script) : h('span.badge.outline', 'not implemented'),
              b.probed ? h('div', { style: { marginTop: '4px' } }, b.probe_ok ? h('span.badge.ok', 'import ok') : h('span.badge.err', 'import failed')) : null),
            h('td.dim', { style: { fontSize: '12px' } }, `${b.models.length}`),
            h('td', b.worker_ok && b.env_ok && b.kind !== 'service' ? btn('Deep check', { size: 'sm', icon: 'flask-conical', title: 'Import the backend in its venv and ask it about each model (slow)', onClick: async e => {
              const bb = e.currentTarget; bb.disabled = true; bb.querySelector('span').textContent = 'Checking…';
              try { const r = await S.api.probe(b.name); const bad = Object.entries(r.results).filter(([, v]) => !v.ok); S.toast(bad.length ? `${bad.length} model(s) failed: ${bad[0][1].reason}` : 'All models import fine', { kind: bad.length ? 'warn' : 'ok', title: b.name }); const rr = await S.api.models({ include_test: true }); backends = rr.backends; await S.loadModels(); render(); }
              catch (err) { S.toastError(err); bb.disabled = false; }
            } }) : null)))))));
      body.replaceChildren(table, cards, list.length ? null : S.empty('search', 'No model matches', 'Clear the filters.'), be);
    }
    S.bus.on('models', render);
    el.append(h('div.page-pad.col', { style: { gap: '14px', maxWidth: '1600px' } },
      h('div.section-head', { style: { margin: 0, flexWrap: 'wrap' } }, h('div.grow', h('h1.page-title', 'Models'), h('p.page-sub', 'What each model can do, whether it runs here, and whether its output may be used commercially. ★ sets the default model of a task.')), S.uiInfo('models_recheck'), refresh),
      summary,
      h('div.row.wrap', { style: { gap: '8px' } }, q, modSeg, statusSel, S.uiInfo('models_status'), licSel, h('label.row', { style: { gap: '6px', fontSize: '13px', color: 'var(--text-2)' }, title: 'Hide non-commercial models in every studio picker and make the API refuse them' }, comm, ic('shield-check', 14), 'Commercial-safe models only', S.uiInfo('commercial_only'))),
      body));
    try { const r = await S.api.models({ include_test: true }); backends = r.backends; } catch (e) { S.toastError(e); }
    render();
  },
});
