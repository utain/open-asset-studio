// Boot: bootstrap data, theme, models, studio modules (web/studios/*.js, discovered by the server), shell, SSE.
import { api } from './core/api.js';
import { store, prefs, loadModels, connectEvents } from './core/store.js';
import { registerStudio } from './core/studio.js';
import { buildShell, applyTheme, startRouter } from './core/shell.js';
import { h } from './core/ui.js';

async function boot() {
  applyTheme(prefs.get('theme', matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'));
  const root = document.getElementById('app');
  let b;
  try { b = await api.bootstrap(); } catch (e) {
    root.replaceChildren(h('div.empty', { style: { gridColumn: '1 / -1', margin: 'auto' } }, h('h4', 'The studio server is not reachable'), h('p', e.message)));
    return;
  }
  Object.assign(store, { projects: b.projects, tasks: b.tasks, defaults: b.settings.defaults, ui: b.settings.ui || {}, memory: b.memory,
    llm: b.llm, counts: b.jobs, version: b.version, tools: b.tools });
  if (!store.projects.find(p => p.id === store.project)) store.project = store.projects[0]?.id || 'default';
  await loadModels();
  try { for (const d of (await api.downloads()).downloads) store.downloads.set(d.model, d); } catch { /* older core */ }
  const mods = await Promise.all(b.modules.map(async f => {
    try { const m = await import(`./studios/${f}`); return m.default; } catch (e) { console.error(`studio module ${f} failed to load`, e); return null; }
  }));
  for (const def of mods) if (def && def.id) registerStudio(def);
  buildShell(root);
  connectEvents();
  startRouter();
  if (b.plugin_errors && Object.keys(b.plugin_errors).length) console.warn('plugin errors', b.plugin_errors);
}
boot();
