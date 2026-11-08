// The (i) info button with a beginner explanation balloon, shared by every settings surface (Settings page, run forms, prompt box,
// export dialog, Providers, 3D tools). S.infoBtn({title, text, facts: [[label, value, mono?], ...]}) -> <button>, or null when there is no text.
//
// One balloon at a time (S.popover closes the previous one). Mouse: opens on hover after 250 ms and stays while the pointer is over the
// icon or the balloon; a click pins it (click outside / Esc / second click closes). Touch and keyboard: click / Enter / Space toggles.
// role=tooltip while hovering, role=dialog once pinned; the icon keeps aria-expanded / aria-describedby in sync. Esc closes only the
// balloon, not a modal or drawer underneath it.
import { h, ic, popover, closePopover } from '../ui.js';

const CSS = `
.tip-info { display: inline-flex; align-items: center; justify-content: center; flex: none; width: 22px; height: 22px; padding: 0; margin: 0 -2px; border: 0; border-radius: 50%; background: none; color: var(--accent); opacity: .75; cursor: pointer; vertical-align: middle; font-weight: 400; text-transform: none; letter-spacing: 0; transition: color .12s, background .12s; }
.tip-info:hover, .tip-info[aria-expanded="true"] { opacity: 1; color: var(--accent); background: var(--accent-soft); }
.tip-info:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
@media (pointer: coarse) { .tip-info { width: 30px; height: 30px; margin: -4px -4px; } }
.field-label > .tip-info { margin-left: -2px; }
.popover.tip-bal { padding: 0; min-width: 0; z-index: 85; }
.tip-bal-body { padding: 11px 13px 12px; font-size: 13px; line-height: 1.5; color: var(--text); font-weight: 400; text-transform: none; letter-spacing: 0; }
.tip-bal-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.tip-bal-text { color: var(--text-2); overflow-wrap: anywhere; }
.tip-bal-facts { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 3px 10px; margin: 10px 0 0; padding-top: 9px; border-top: 1px solid var(--line-soft); font-size: 12px; }
.tip-bal-facts dt { color: var(--text-3); }
.tip-bal-facts dd { margin: 0; color: var(--text-2); overflow-wrap: anywhere; }
.tip-bal-facts code { font-family: var(--mono); font-size: 11.5px; color: var(--text); user-select: all; }
`;
let styled = false;
function ensureStyle() { if (!styled) { styled = true; document.head.append(h('style', CSS)); } }

let bal = null, hoverT = 0, leaveT = 0, balSeq = 0;

/** Close the open balloon (if any), e.g. when the page that owns it is hidden or redrawn. */
export function closeInfo() { clearTimeout(hoverT); clearTimeout(leaveT); if (bal) closePopover(); }
/** True while a balloon is open for a button that was removed from the page (the caller redrew): close it then. */
export function closeOrphanInfo() { if (bal && !bal.btn.isConnected) closeInfo(); }

function balloon(spec) {
  const body = h('div.tip-bal-body', h('div.tip-bal-title', spec.title), h('div.tip-bal-text', spec.text));
  if (spec.facts?.length) body.append(h('dl.tip-bal-facts', ...spec.facts.filter(f => f && f[1] != null && f[1] !== '').flatMap(([k, v, mono]) => [h('dt', k), h('dd', mono ? h('code', String(v)) : String(v))])));
  return body;
}

function openInfo(btn, spec, pinned) {
  if (bal && bal.btn === btn) return;
  if (bal?.pinned && !pinned) return;                      // a pinned balloon is not replaced by a hover
  clearTimeout(leaveT);
  const id = `tip-bal-${++balSeq}`;
  const onScroll = e => { if (!bal || !bal.pop.el.contains(e.target)) closeInfo(); };
  const onKey = e => { if (e.key === 'Escape' && bal === cur) { e.stopPropagation(); closeInfo(); } };
  const cur = { btn, pinned, pop: null };
  const pop = popover(btn, balloon(spec), { role: pinned ? 'dialog' : 'tooltip', width: 'min(320px, calc(100vw - 24px))', onClose: () => {
    btn.setAttribute('aria-expanded', 'false'); btn.removeAttribute('aria-describedby');
    removeEventListener('scroll', onScroll, true); removeEventListener('keydown', onKey, true);
    if (bal === cur) bal = null;
  } });
  cur.pop = pop; bal = cur;
  pop.el.id = id; pop.el.classList.add('tip-bal');
  pop.place();
  if (pinned) pop.el.setAttribute('aria-label', `About ${spec.title}`);
  btn.setAttribute('aria-expanded', 'true'); btn.setAttribute('aria-describedby', id);
  pop.el.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') clearTimeout(leaveT); });
  pop.el.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse' && bal === cur && !cur.pinned) leaveT = setTimeout(closeInfo, 200); });
  addEventListener('scroll', onScroll, true);
  addEventListener('keydown', onKey, true);
}
function pinInfo() {
  if (!bal) return;
  bal.pinned = true; bal.pop.el.setAttribute('role', 'dialog'); bal.pop.el.setAttribute('aria-label', `About ${bal.btn.dataset.title || 'this setting'}`);
}

/** The (i) button for a spec {title, text, facts: [[label, value, mono?], ...]}; null when there is nothing to say. */
export function infoBtn(spec) {
  if (!spec?.text) return null;
  ensureStyle();
  const b = h('button.tip-info', { type: 'button', 'aria-label': `About ${spec.title}`, 'aria-expanded': 'false', 'aria-haspopup': 'dialog', dataset: { title: spec.title } }, ic('info', 17));
  b.addEventListener('pointerenter', e => {
    if (e.pointerType !== 'mouse') return;
    clearTimeout(leaveT);
    if (!bal || bal.btn !== b) { clearTimeout(hoverT); hoverT = setTimeout(() => openInfo(b, spec, false), 250); }
  });
  b.addEventListener('pointerleave', e => {
    if (e.pointerType !== 'mouse') return;
    clearTimeout(hoverT);
    if (bal && bal.btn === b && !bal.pinned) leaveT = setTimeout(closeInfo, 200);
  });
  b.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation(); clearTimeout(hoverT);
    if (bal && bal.btn === b) { if (bal.pinned) closeInfo(); else pinInfo(); } else openInfo(b, spec, true);
  });
  b.addEventListener('keydown', e => { if (e.key === 'Tab' && bal && bal.btn === b) closeInfo(); });
  return b;
}

/** Beginner (i) text for a registry param: {title, text, facts}. `task` picks info_by_task[task] over `info` over the one-line `help`. */
export function paramSpec(p, task) {
  const text = p.info_by_task?.[task] || p.info || p.help || '';
  const facts = [];
  const num = x => `${x}${p.unit ? ' ' + p.unit : ''}`;
  const lab = c => (c && typeof c === 'object' ? (c.label ?? c.value) : c);
  if (p.type === 'seed') facts.push(['Default', 'random (-1)']);
  else if (p.default !== undefined && p.default !== null && p.default !== '') facts.push(['Default', p.type === 'bool' ? (p.default ? 'on' : 'off') : p.type === 'choice' ? String(lab((p.choices || []).find(c => (c && typeof c === 'object' ? c.value : c) === p.default) ?? p.default)) : num(p.default)]);
  if ((p.type === 'int' || p.type === 'float') && p.min != null && p.max != null) facts.push(['Range', `${num(`${p.min} to ${p.max}`)}${p.step ? `, step ${p.step}` : ''}`]);
  else if (p.type === 'choice' && p.choices?.length) facts.push(['Options', p.choices.map(lab).join(', ')]);
  else if (p.max_length) facts.push(['Length', `up to ${p.max_length} characters`]);
  facts.push(['Key', p.name, true]);
  return { title: p.label || p.name, text, facts };
}
