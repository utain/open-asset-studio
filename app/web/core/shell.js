// App shell: sidebar (bottom bar on phones), top bar (project switcher, search, queue indicator, memory meter,
// theme), page containers per studio, routing and send-to delivery.
import { api } from './api.js';
import { store, bus, prefs, setProject, getProfile, setProfile, setCommercialOnly, isOverridden } from './store.js';
import { h, ic, btn, menu, popover, promptDialog, toast, toastError, fmtAgo } from './ui.js';
import { allStudios, getStudio, parseHash, takePayload, navigate, _setCurrent } from './studio.js';
import { JobList } from './components/jobs.js';
import { closeAsset } from './components/drawer.js';
import * as sdk from './sdk.js';

const pages = new Map();   // studio id -> {el, inst}
let mainEl, titleEl, navEls = new Map(), bottomEls = new Map();

export function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  prefs.set('theme', t);
  bus.emit('theme', t);
}

export function buildShell(root) {
  const sidebar = h('nav.sidebar', { 'aria-label': 'Main' });
  sidebar.append(h('div.brand', h('div.brand-mark', ic('boxes', 18)), h('div.brand-name', 'Open Asset Studio', h('small', 'Media assets · local'))));
  const sections = [['', null], ['Create', 'studio'], ['Workspace', 'workspace'], ['System', 'system']];
  const list = allStudios().filter(s => s.nav !== false);
  for (const [label, sec] of sections) {
    const items = list.filter(s => (sec === null ? s.section === 'home' : s.section === sec));
    if (!items.length) continue;
    if (label) sidebar.append(h('div.nav-sec', label));
    for (const s of items) {
      const a = h('a.nav-item', { href: `#/${s.id}`, style: s.color ? { '--nav-c': s.color } : null, title: s.title, 'aria-label': s.title },
        ic(s.icon || 'circle-dot', 18), h('span.lbl', s.title), s.soon ? h('span.soon', 'soon') : null, h('span.count'));
      navEls.set(s.id, a);
      sidebar.append(a);
    }
  }
  const themeBtn = h('button.nav-item', { type: 'button', title: 'Light / dark', 'aria-label': 'Light / dark', on: { click: () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light') } }, ic('sun', 18), h('span.lbl', 'Light / dark'));
  const sse = h('span.dim.lbl', { style: { fontSize: '11px', paddingLeft: '10px' } });
  const drawSse = () => { sse.textContent = `v${store.version} · ${store.sse === 'live' ? 'connected' : 'reconnecting…'}`; };
  bus.on('sse', drawSse); drawSse();
  // collapse / expand toggle at the bottom (icon rail <-> full sidebar)
  const sbToggle = h('button.nav-item.sb-toggle', { type: 'button', on: { click: () => setSidebar(!sidebarCollapsed(), true) } });
  sidebar.append(h('div.sidebar-foot', themeBtn, sbToggle, sse));
  const appEl = root;
  const sbMq = matchMedia('(max-width: 1280px)');
  const sidebarCollapsed = () => appEl.classList.contains('sb-collapsed');
  // preference: 'collapsed' | 'expanded' (explicit, remembered) | null (automatic: icon rail up to 1280 px wide)
  const sbWanted = () => { const p = prefs.get('sidebar', null); return p === 'collapsed' ? true : p === 'expanded' ? false : sbMq.matches; };
  const drawSb = () => {
    const c = sidebarCollapsed();
    sbToggle.replaceChildren(ic(c ? 'chevrons-right' : 'chevrons-left', 18), h('span.lbl', 'Collapse'));
    sbToggle.title = c ? 'Expand the sidebar' : 'Collapse the sidebar to icons';
    sbToggle.setAttribute('aria-label', sbToggle.title);
    sbToggle.setAttribute('aria-expanded', String(!c));
  };
  function setSidebar(collapsed, explicit) {
    if (explicit) prefs.set('sidebar', collapsed ? 'collapsed' : 'expanded');
    appEl.classList.toggle('sb-collapsed', collapsed);
    drawSb();
  }
  setSidebar(sbWanted(), false);
  sbMq.addEventListener('change', () => setSidebar(sbWanted(), false));
  requestAnimationFrame(() => requestAnimationFrame(() => appEl.classList.add('sb-anim')));   // no width animation on first paint

  // ----- top bar
  titleEl = h('div.title');
  const projBtn = h('button.pill-btn', { type: 'button', title: 'Switch project', on: { click: () => projectMenu(projBtn) } });
  const drawProj = () => { const p = store.projects.find(x => x.id === store.project); projBtn.replaceChildren(ic('folder', 16), h('span.lbl.ellipsis', { style: { maxWidth: '160px' } }, p?.title || store.project), ic('chevron-down', 14)); };
  bus.on('project', drawProj); drawProj();
  const search = h('input', { placeholder: 'Search assets…', type: 'search', 'aria-label': 'Search assets',
    on: { keydown: e => { if (e.key === 'Enter') { navigate('library', '', search.value ? { q: search.value } : null); } } } });
  const searchBox = h('label.search', ic('search', 16), search, h('kbd', '/'));
  addEventListener('keydown', e => { if (e.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) { e.preventDefault(); search.focus(); } });
  const queueBtn = h('button.pill-btn.queue-btn', { type: 'button', title: 'Queue', on: { click: () => queuePopover(queueBtn) } });
  const drawQueue = () => {
    const c = store.counts || {}; const busy = (c.running || 0) + (c.queued || 0);
    queueBtn.replaceChildren(h(`span.dot${c.running ? '.live' : ''}`), ic('list-checks', 16), h('span.lbl', busy ? `${c.running || 0} running${c.queued ? ` · ${c.queued} queued` : ''}` : 'Queue idle'));
    const qn = navEls.get('system')?.querySelector('.count'); if (qn) qn.textContent = busy ? String(busy) : '';
  };
  bus.on('system', drawQueue); bus.on('job', () => { recount(); drawQueue(); }); drawQueue();
  const mem = h('button.mem-meter.pill-btn', { type: 'button', title: 'Memory (unified CPU + GPU)', on: { click: () => navigate('system') } });
  const drawMem = () => {
    const m = store.memory; if (!m) { mem.replaceChildren(ic('memory-stick', 16), h('span.lbl', '…')); return; }
    const used = m.total_gb - m.available_gb; const ours = Math.min(m.ours_gb, used);
    mem.replaceChildren(ic('memory-stick', 16), h('div.mem-bar', h('i.other', { style: { width: `${(used / m.total_gb) * 100}%` } }), h('i.ours', { style: { width: `${(ours / m.total_gb) * 100}%` } })),
      h('span.lbl', `${Math.round(m.available_gb)} GB free`));
    mem.title = `Unified memory: ${used.toFixed(0)} / ${m.total_gb.toFixed(0)} GB used (studio models ${m.ours_gb} GB). Click for details.`;
  };
  bus.on('system', drawMem); drawMem();
  const themeTop = btn('', { icon: 'moon', kind: 'ghost', title: 'Light / dark', onClick: () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light') });
  themeTop.classList.add('hide-sm');
  const drawTheme = t => themeTop.replaceChildren(ic(t === 'light' ? 'moon' : 'sun', 17));
  bus.on('theme', drawTheme); drawTheme(document.documentElement.dataset.theme);
  // use-case profile (game | video | podcast | social | film | general) and the "Commercial-safe models only" switch
  const profBtn = h('button.pill-btn.prof-btn', { type: 'button', title: 'Use-case profile: style presets, aspect ratios and export defaults', on: { click: () => profileMenu(profBtn) } });
  // a small dot marks a setting that the current project (not settings.json) decides
  const projTitle = () => store.projects.find(x => x.id === store.project)?.title || store.project;
  const overrideDot = key => (isOverridden(key) ? h('span.pill-dot', { title: `Set by project ${projTitle()}` }) : null);
  const drawProf = () => { const p = getProfile(); profBtn.replaceChildren(ic(p.icon || 'layers', 16), h('span.lbl.hide-sm', p.label || p.id), overrideDot('profile'), ic('chevron-down', 14)); };
  bus.on('profile', drawProf); bus.on('settings', drawProf); drawProf();
  const licBtn = h('button.pill-btn.lic-toggle', { type: 'button', on: { click: async () => {
    try { await setCommercialOnly(!store.license.commercial_only); toast(store.license.commercial_only ? 'Commercial-safe models only: non-commercial models are hidden and refused.' : 'All models are available again.', { kind: 'ok', timeout: 2600 }); } catch (e) { toastError(e); }
  } } });
  const drawLic = () => {
    const on = store.license.commercial_only;
    licBtn.classList.toggle('on', on);
    licBtn.title = on ? 'Commercial-safe models only: ON. Non-commercial models are hidden in the pickers and refused by the API. Click to turn off.' : 'Commercial-safe models only: off. Click to hide / refuse non-commercial models.';
    licBtn.setAttribute('aria-pressed', String(on));
    licBtn.replaceChildren(ic(on ? 'shield-check' : 'shield-off', 16), h('span.lbl.hide-sm', on ? 'Commercial-safe' : 'All models'), overrideDot('license'));
    if (isOverridden('license')) licBtn.title += ` (set by project ${projTitle()})`;
  };
  bus.on('license', drawLic); bus.on('settings', drawLic); bus.on('project', () => { drawProf(); drawLic(); }); drawLic();
  const top = h('header.topbar', titleEl, projBtn, searchBox, h('span.spacer'), profBtn, licBtn, queueBtn, mem, themeTop);

  mainEl = h('main.main', { id: 'main' });
  // ----- bottom bar (phones)
  const bottom = h('nav.bottombar', { 'aria-label': 'Main (mobile)' });
  const mob = allStudios().filter(s => s.mobile).slice(0, 4);
  for (const s of mob) {
    const b = h('button', { type: 'button', style: s.color ? { '--nav-c': s.color } : null, on: { click: () => navigate(s.id) } }, ic(s.icon, 20), h('span', s.short || s.title));
    bottomEls.set(s.id, b); bottom.append(b);
  }
  const more = h('button', { type: 'button', on: { click: () => menu(more, [
    ...allStudios().filter(s => !s.mobile && s.nav !== false).map(s => ({ label: s.title, icon: s.icon, onClick: () => navigate(s.id) })),
    { sep: true }, { label: 'Light / dark', icon: 'sun', onClick: () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light') },
  ]) } }, ic('menu', 20), h('span', 'More'));
  bottom.append(more);
  root.replaceChildren(sidebar, top, mainEl, bottom);
  addEventListener('hashchange', route);
  bus.on('route', r => show(r));
}

function profileMenu(anchor) {
  const cur = store.profile;
  menu(anchor, [{ head: 'Profile: what are you making?' },
    ...Object.values(store.profiles).map(p => ({ label: p.label, sub: p.summary, icon: p.icon, on: p.id === cur,
      onClick: async () => { try { await setProfile(p.id); toast(`Profile: ${p.label}`, { kind: 'ok', timeout: 1800 }); } catch (e) { toastError(e); } } }))]);
}

function recount() {
  let running = 0, queued = 0;
  for (const j of store.jobs.values()) { if (j.status === 'running') running++; else if (j.status === 'queued') queued++; }
  store.counts = { ...store.counts, running, queued };
}

async function projectMenu(anchor) {
  try { store.projects = (await api.projects()).projects; } catch { /* keep */ }
  menu(anchor, [{ head: 'Projects' },
    ...store.projects.map(p => ({ label: p.title, sub: `${p.assets} assets · ${p.id}`, icon: 'folder', on: p.id === store.project, onClick: () => { setProject(p.id); toast(`Project: ${p.title}`, { kind: 'ok', timeout: 1800 }); } })),
    { sep: true },
    { label: 'New project…', icon: 'plus', onClick: async () => {
      const title = await promptDialog('Project name', { title: 'New project', placeholder: 'e.g. Dungeon Crawler' }); if (!title) return;
      const id = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'project';
      try { await api.createProject(id, title); store.projects = (await api.projects()).projects; setProject(id); toast(`Created ${title}`, { kind: 'ok' }); } catch (e) { toastError(e); }
    } }]);
}

function queuePopover(anchor) {
  const jl = new JobList({ statuses: ['running', 'queued', 'failed', 'done', 'cancelled'], limit: 8, compact: true, emptyText: 'Nothing has run yet.' });
  popover(anchor, h('div', { style: { width: 'min(400px, calc(100vw - 24px))', padding: '6px' } },
    h('div.row', { style: { marginBottom: '8px' } }, h('b.grow', 'Recent jobs'), btn('Open queue', { size: 'sm', icon: 'arrow-right', onClick: () => { navigate('system'); } })), jl.el), { align: 'end', onClose: () => jl.destroy() });
}

function route() { show(parseHash()); }

async function show(r) {
  let def = getStudio(r.studio);
  if (!def) { def = getStudio('home'); r = { ...r, studio: 'home' }; }
  _setCurrent(r);
  closeAsset();
  for (const [id, a] of navEls) a.classList.toggle('active', id === def.id);
  for (const [id, b] of bottomEls) b.classList.toggle('active', id === def.id);
  titleEl.textContent = def.title;
  document.title = `${def.title} · Open Asset Studio`;
  let page = pages.get(def.id);
  for (const [id, p] of pages) if (id !== def.id && !p.el.hidden) { p.el.hidden = true; p.inst?.hide?.(); }
  if (!page) {
    const el = h('section.page', { dataset: { studio: def.id } });
    mainEl.append(el);
    page = { el, inst: null };
    pages.set(def.id, page);
    try { page.inst = (await def.mount(el, sdk)) || {}; } catch (e) { console.error(e); el.append(h('div.page-pad', h('div.card.card-pad', h('b', `The ${def.title} page failed to load`), h('pre.mono', String(e.stack || e))))); page.inst = {}; }
  }
  page.el.hidden = false;
  page.inst?.show?.();
  try { page.inst?.route?.(r); } catch (e) { console.error(e); }
  if (r.query.send) {
    const payload = takePayload(r.query.send);
    history.replaceState(null, '', `#/${r.studio}${r.sub ? '/' + r.sub : ''}`);
    if (payload) {
      try {
        const assets = payload.assetIds.length ? (await api.assets({ ids: payload.assetIds.join(','), limit: 50 })).items : [];
        const ordered = payload.assetIds.map(id => assets.find(a => a.id === id)).filter(Boolean);
        if (page.inst?.receive) page.inst.receive({ ...payload, assets: ordered });
        else toast(`${def.title} cannot receive assets yet.`, { kind: 'warn' });
      } catch (e) { toastError(e); }
    }
  }
}
export { route as startRouter };
export function lastSeen(iso) { return fmtAgo(iso); }
