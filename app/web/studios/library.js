// Library: every asset of the current project. Filters (type, model, tag, rating, favourites), search, sort,
// multi-select (compare, export, tag, delete), drag-and-drop upload, detail drawer.
import * as S from '../core/sdk.js';
const { h, ic, btn } = S;

const PAGE = 120;

export default S.defineStudio({
  id: 'library', title: 'Library', icon: 'library', order: 60, section: 'workspace', mobile: true,
  receives: [],
  async mount(el) {
    const f = { type: '', model: '', tag: '', rating_min: '', favorite: false, q: '', sort: 'new', ...S.prefs.get('library.filters', {}) };
    let offset = 0, total = 0;
    const grid = new S.ResultsGrid({ selectable: true, onOpen: a => S.openAsset(a, { list: grid.assets().map(x => x.id) }),
      emptyState: S.empty('library', 'No assets match', 'Change the filters, drop files here to upload, or generate something in a studio.') });
    const countEl = h('span.dim');
    const q = h('input.input', { type: 'search', placeholder: 'Search name, prompt, tag, model…', value: f.q, style: { maxWidth: '320px' }, 'aria-label': 'Search' });
    q.addEventListener('input', S.debounce(() => { f.q = q.value; reload(); }, 250));
    const typeSeg = S.seg([{ value: '', label: 'All' }, { value: 'image', label: 'Images' }, { value: 'material', label: 'Materials' }, { value: 'mesh', label: '3D' }, { value: 'audio', label: 'Audio' }, { value: 'voice', label: 'Voices' }, { value: 'video', label: 'Video' }], f.type, v => { f.type = v; reload(); });
    const modelSel = h('select.select', { style: { width: 'auto', maxWidth: '220px' }, 'aria-label': 'Model', on: { change: e => { f.model = e.target.value; reload(); } } });
    const tagSel = h('select.select', { style: { width: 'auto', maxWidth: '180px' }, 'aria-label': 'Tag', on: { change: e => { f.tag = e.target.value; reload(); } } });
    const ratingSel = h('select.select', { style: { width: 'auto' }, 'aria-label': 'Rating', on: { change: e => { f.rating_min = e.target.value; reload(); } } },
      ...[['', 'Any rating'], ['1', '★ 1+'], ['3', '★ 3+'], ['4', '★ 4+'], ['5', '★ 5']].map(([v, l]) => h('option', { value: v, selected: String(f.rating_min) === v }, l)));
    const sortSel = h('select.select', { style: { width: 'auto' }, 'aria-label': 'Sort', on: { change: e => { f.sort = e.target.value; reload(); } } },
      ...[['new', 'Newest'], ['old', 'Oldest'], ['rating', 'Best rated'], ['name', 'Name']].map(([v, l]) => h('option', { value: v, selected: f.sort === v }, l)));
    const favBtn = h(`button.chip${f.favorite ? '.on' : ''}`, { type: 'button', on: { click: () => { f.favorite = !f.favorite; favBtn.classList.toggle('on', f.favorite); reload(); } } }, ic('heart', 13), 'Favourites');
    const more = btn('Load more', { onClick: () => load(true) });
    const fileIn = h('input', { type: 'file', multiple: true, class: 'hidden', accept: '.png,.jpg,.jpeg,.webp,.wav,.flac,.ogg,.mp3,.mp4,.mov,.webm,.mkv,.glb', on: { change: () => uploadFiles(fileIn.files) } });
    const selBar = h('div.row.wrap.hidden', { style: { padding: '8px 10px', borderRadius: '12px', background: 'var(--accent-soft)', border: '1px solid var(--accent)' } });

    async function uploadFiles(files) {
      for (const file of files) {
        const t = S.toast(`Uploading ${file.name}…`, { timeout: 0 });
        try { await S.api.upload(file, { project: S.store.project }); t.remove(); S.toast(file.name, { kind: 'ok', title: 'Uploaded', timeout: 2000 }); }
        catch (e) { t.remove(); S.toastError(e, `Upload of ${file.name} failed`); }
      }
      fileIn.value = '';
    }
    grid.onSelectionChange(sel => {
      selBar.classList.toggle('hidden', !sel.length);
      selBar.replaceChildren(h('b', `${sel.length} selected`), h('span.spacer'),
        sel.length === 2 ? btn('Compare', { size: 'sm', icon: 'columns-2', onClick: () => S.compare(sel[0], sel[1]) }) : null,
        btn('Tag', { size: 'sm', icon: 'tag', onClick: async () => {
          const t = await S.promptDialog('Tag to add', { title: `Tag ${sel.length} asset(s)` }); if (!t) return;
          for (const id of sel) { const a = grid.assets().find(x => x.id === id); await S.api.patchAsset(id, { tags: [...new Set([...(a?.tags || []), t])] }).catch(S.toastError); }
        } }),
        btn('Export zip', { size: 'sm', icon: 'package', onClick: () => S.exportAssets(sel) }),
        btn('Export…', { size: 'sm', icon: 'sliders-horizontal', title: 'Choose an export preset: loudness, sample rate, format, image size and aspect', onClick: () => S.exportAssets(sel, { options: true }) }),
        ...S.selectionActions(sel.map(id => grid.assets().find(x => x.id === id)).filter(Boolean)).map(a =>
          btn(a.label, { size: 'sm', icon: a.icon, title: a.title, onClick: () => a.run(sel, sel.map(id => grid.assets().find(x => x.id === id)).filter(Boolean)) })),
        btn('Delete', { size: 'sm', icon: 'trash-2', kind: 'danger', onClick: async () => {
          if (!(await S.confirmDialog(`Delete ${sel.length} asset(s)? They are moved to the project's .trash folder.`, { ok: 'Delete', danger: true }))) return;
          for (const id of sel) await S.api.deleteAsset(id).catch(S.toastError);
          grid.clearSelection();
        } }),
        btn('Select all', { size: 'sm', kind: 'ghost', onClick: () => grid.selectAll() }),
        btn('', { size: 'sm', kind: 'ghost', icon: 'x', title: 'Clear selection', onClick: () => grid.clearSelection() }));
    });

    async function load(append = false) {
      S.prefs.set('library.filters', { ...f, q: '' });
      try {
        const r = await S.api.assets({ project: S.store.project, type: f.type || undefined, model: f.model || undefined, tag: f.tag || undefined,
          rating_min: f.rating_min || undefined, favorite: f.favorite || undefined, q: f.q || undefined, sort: f.sort, limit: PAGE, offset: append ? offset : 0 });
        if (!append) { offset = 0; grid.set(r.items); } else r.items.forEach(a => grid.add(a, { prepend: false }));
        offset += r.items.length; total = r.total;
        countEl.textContent = `${total} asset${total === 1 ? '' : 's'}`;
        more.classList.toggle('hidden', offset >= total);
        const fill = (sel, opts, cur, all) => sel.replaceChildren(h('option', { value: '' }, all), ...opts.map(o => h('option', { value: o.value, selected: o.value === cur }, o.label)));
        if (!append) {
          fill(modelSel, r.models.map(m => ({ value: m, label: S.store.modelsById.get(m)?.name || m })), f.model, 'All models');
          fill(tagSel, r.tags.map(t => ({ value: t, label: t })), f.tag, 'All tags');
        }
      } catch (e) { S.toastError(e); }
    }
    const reload = () => load(false);
    S.bus.on('project', reload);
    S.bus.on('asset', ev => {
      if (ev.asset.project !== S.store.project) return;
      if (ev.action === 'created' && !grid.cards.has(ev.asset.id) && (!f.type || f.type === ev.asset.type) && !f.q && !f.tag && !f.model && !f.favorite && !f.rating_min && f.sort === 'new') {
        grid.add(ev.asset); total++; countEl.textContent = `${total} asset${total === 1 ? '' : 's'}`;
      }
    });

    const page = h('div.page-pad.col', { style: { gap: '14px' } },
      h('div.section-head', { style: { margin: 0, flexWrap: 'wrap' } }, h('div.grow', h('h1.page-title', 'Library'), h('p.page-sub', countEl)),
        btn('Upload', { icon: 'upload', onClick: () => fileIn.click() }), fileIn),
      h('div.row.wrap', { style: { gap: '8px' } }, q, typeSeg, modelSel, tagSel, ratingSel, favBtn, h('span.spacer'), sortSel),
      selBar, grid.el, h('div.row', { style: { justifyContent: 'center' } }, more));
    // drop anywhere on the page to upload
    el.addEventListener('dragover', e => { if ([...e.dataTransfer.types].includes('Files')) { e.preventDefault(); page.classList.add('drag'); } });
    el.addEventListener('dragleave', e => { if (e.target === el) page.classList.remove('drag'); });
    el.addEventListener('drop', e => { if (e.dataTransfer.files.length) { e.preventDefault(); uploadFiles(e.dataTransfer.files); } });
    el.append(page);
    reload();
    return {
      route(r) { if (r.query.q !== undefined && r.query.q !== f.q) { f.q = r.query.q; q.value = r.query.q; reload(); } },
    };
  },
});
