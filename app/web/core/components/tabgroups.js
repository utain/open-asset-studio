// TabGroups: the task tab row of a studio, optionally grouped into workflow steps.
//   flat    (no item has a `group`):  one row of tabs, exactly like the classic task tabs
//   grouped (some item has `group`):  a row of steps (1 Create · 2 Refine …); the active step shows its tasks as small pills
//                                     underneath (no pill row when the step holds a single task)
// Used by TaskStudio and by studios with their own page (Sound). Nothing here knows about tasks: items are {id, label, group?, title?}.
//
//   new TabGroups({items, groups: [{id, label, title?}], value, onSelect(id), numbered, accent, label, remember: {get(gid), set(gid, id)}})
//     .el  .set(id)  .selectGroup(gid)  .hasGroup(gid)  .value  .setItems(items)
//
// Items whose group is not listed in `groups` get a step of their own, after the listed ones, in first-seen order; items
// without a `group` in a grouped studio become one-task steps. Empty groups (a task filtered out) are not drawn.
// Keyboard: role=tablist / tab; ArrowLeft / ArrowRight / Home / End move the focus inside a row (roving tabindex), Enter / Space picks.
import { h, clear } from '../ui.js';

const cap = s => (s ? s[0].toUpperCase() + s.slice(1) : s);

export class TabGroups {
  constructor(o) {
    this.o = o;
    this.last = {};     // group id -> task last used in it (this session; o.remember persists it)
    this.el = h('div.task-tabs', o.accent ? { style: { '--tg-accent': o.accent } } : null);
    this.setItems(o.items, o.value);
  }
  get value() { return this._value; }
  setItems(items, value = this._value) {
    this.items = items;
    this.grouped = items.some(i => i.group);
    this.groups = this.grouped ? this._groupsOf(items) : [];
    this._value = items.find(i => i.id === value) ? value : items[0]?.id;
    this.el.classList.toggle('hidden', items.length < 2);
    this.el.classList.toggle('grouped', this.grouped);
    this._build();
  }
  _groupsOf(items) {
    const map = new Map();
    for (const g of this.o.groups || []) map.set(g.id, { id: g.id, label: g.label || cap(g.id), title: g.title, items: [] });
    for (const it of items) {
      const gid = it.group || it.id;
      if (!map.has(gid)) map.set(gid, { id: gid, label: it.group ? cap(it.group) : (it.short || it.label), items: [] });
      map.get(gid).items.push(it);
    }
    return [...map.values()].filter(g => g.items.length);
  }
  groupOf(id) { return this.groups.find(g => g.items.some(i => i.id === id)); }
  hasGroup(gid) { return !!this.groups.find(g => g.id === gid); }
  set(id) {
    if (!this.items.find(i => i.id === id)) return;
    this._value = id;
    this._paint();
  }
  /** Open a step: its last-used task, else its first. Returns the task id (the caller decides what to do with it). */
  selectGroup(gid) {
    const g = this.groups.find(x => x.id === gid); if (!g) return null;
    const rem = this.last[gid] || this.o.remember?.get?.(gid);
    return (g.items.find(i => i.id === rem) || g.items[0]).id;
  }
  _pick(id) { if (id === this._value) return; this.o.onSelect?.(id); }

  _build() {
    clear(this.el);
    this.stepBtns = new Map(); this.pillsEl = null; this.pillsG = undefined; this.pills = new Map();
    if (!this.grouped) {
      this.el.setAttribute('role', 'tablist'); if (this.o.label) this.el.setAttribute('aria-label', this.o.label);
      this.flat = new Map();
      for (const it of this.items) {
        const b = h('button', { type: 'button', role: 'tab', title: it.title || null, on: { click: () => this._pick(it.id) } }, it.short || it.label);
        this.flat.set(it.id, b); this.el.append(b);
      }
      this._keys(this.el);
    } else {
      this.el.removeAttribute('role'); this.el.removeAttribute('aria-label');
      this.stepsEl = h('div.tg-steps', { role: 'tablist', 'aria-label': this.o.label || 'Steps' });
      this.groups.forEach((g, n) => {
        const b = h('button', { type: 'button', role: 'tab', title: g.title || `${g.label}: ${g.items.map(i => i.label).join(', ')}`, on: { click: () => this._pick(this.selectGroup(g.id)) } },
          this.o.numbered ? h('span.tg-num', { 'aria-hidden': 'true' }, String(n + 1)) : null, h('span.tg-lbl', g.label));
        this.stepBtns.set(g.id, b); this.stepsEl.append(b);
      });
      this.pillsEl = h('div.tg-pills', { role: 'tablist' });
      this.el.append(this.stepsEl, this.pillsEl);
      this._keys(this.stepsEl); this._keys(this.pillsEl);
    }
    this._paint();
  }
  _paint() {
    const v = this._value;
    if (!this.grouped) {
      for (const [id, b] of this.flat) this._mark(b, id === v);
      return;
    }
    const cur = this.groupOf(v) || this.groups[0];
    if (cur) this.last[cur.id] = v, this.o.remember?.set?.(cur.id, v);
    for (const [gid, b] of this.stepBtns) this._mark(b, gid === cur?.id);
    if (this.pillsG !== cur?.id) {   // rebuild the pills only when the step changes, so a focused pill keeps focus
      this.pillsG = cur?.id;
      clear(this.pillsEl);
      this.pills = new Map();
      if (cur && cur.items.length > 1) {
        this.pillsEl.setAttribute('aria-label', `${cur.label} tasks`);
        for (const it of cur.items) {
          const b = h('button.tg-pill', { type: 'button', role: 'tab', title: it.title || null, on: { click: () => this._pick(it.id) } }, it.short || it.label);
          this.pills.set(it.id, b); this.pillsEl.append(b);
        }
      }
      this.pillsEl.hidden = !(cur && cur.items.length > 1);
    }
    for (const [id, b] of this.pills) this._mark(b, id === v);
  }
  _mark(b, on) { b.classList.toggle('on', on); b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1; }
  _keys(row) {
    row.addEventListener('keydown', e => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) return;
      const tabs = [...row.querySelectorAll('[role=tab]')]; const i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      const n = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1) + tabs.length) % tabs.length;
      e.preventDefault(); tabs[n].focus();
    });
  }
}
