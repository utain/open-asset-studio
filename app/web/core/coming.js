// "Coming soon" studio page used by the stub modules (image / 3d / sound / pipelines) until a studio is built.
// It lists the planned features and the models already registered for its tasks (live status), and accepts
// send-to payloads so the actions in the asset drawer already work end to end.
import * as S from './sdk.js';
const { h, ic, btn } = S;

export function comingStudio({ id, title, icon, color, order, tasks = [], features = [], receives = [], blurb, mobile = false, short }) {
  return S.defineStudio({
    id, title, icon, color, order, section: 'studio', soon: true, receives, mobile, short,
    mount(el) {
      const models = h('div.col', { style: { gap: '8px' } });
      const inbox = h('div');
      const draw = () => {
        const ms = S.store.models.filter(m => !m.test && m.tasks.some(t => tasks.includes(t)));
        models.replaceChildren(...ms.map(m => h(`div.row.wrap.mod-${m.modality}`, { style: { padding: '8px 10px', border: '1px solid var(--line-soft)', borderRadius: '10px', background: 'var(--panel)' } },
          h('div.mp-icon', ic(S.MOD_ICON[m.modality] || 'cpu', 16)), h('div.grow', { style: { minWidth: '160px' } }, h('div', { style: { fontWeight: 600 } }, m.name), h('div.dim', { style: { fontSize: '12px' } }, m.tasks.filter(t => tasks.includes(t)).map(t => S.store.tasks[t]?.label || t).join(' · '))),
          h('div.row', S.licenseBadge(m.license), S.statusBadge(m.status)))));
        if (!ms.length) models.append(h('div.dim', 'No model registered yet.'));
      };
      S.bus.on('models', draw); draw();
      el.append(h('div.coming', { style: { '--cc': color } },
        h('div.coming-hero', h('div.ci', ic(icon, 26)), h('h1.page-title', title), h('p.page-sub', blurb),
          h('div.row.wrap', { style: { marginTop: '14px' } }, h('span.badge.outline', ic('hammer', 12), 'Studio under construction'), h('span.spacer'),
            btn('Library', { size: 'sm', icon: 'library', onClick: () => S.navigate('library') }), btn('Diagnostics', { size: 'sm', icon: 'flask-conical', onClick: () => S.navigate('selftest') })),
          h('div.feature-list', ...features.map(f => h('div', ic('check', 15), h('span', f))))),
        inbox,
        h('section', { style: { marginTop: '22px' } }, h('div.section-head', h('h3.section-title', ic('cpu', 16), 'Models registered for this studio')), models)));
      return {
        receive(p) {
          inbox.replaceChildren(h('div.card.card-pad', { style: { marginTop: '16px' } }, h('div.row', ic('send', 16), h('b.grow', `Received “${p.action}” with ${p.assets.length} asset(s)`)),
            h('div.dim', { style: { fontSize: '12.5px', marginTop: '6px' } }, 'The studio will pick these up as soon as it is built.'),
            h('div.row.wrap', { style: { marginTop: '8px' } }, ...p.assets.map(a => btn(a.name || a.id, { size: 'sm', icon: S.MOD_ICON[a.type], onClick: () => S.openAsset(a) })))));
        },
      };
    },
  });
}
