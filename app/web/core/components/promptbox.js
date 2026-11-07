// Prompt box with the ✨ Improve panel (local LLM by default; a backend picker in the panel chooses API / CLI LLMs): Thai or English in, a model-specific prompt out, with
// variations, notes and style chips. Insert / replace / keep original. Extra outputs (lyrics, the line to speak)
// are handed to the studio through onApply({field: value}) using the style's `apply` map.
import { api } from '../api.js';
import { store, bus, promptStyleOf, prefs } from '../store.js';
import { h, ic, btn, toastError, isThai, copyText, seg } from '../ui.js';
import { backendPicker, generatedBy } from './writer.js';

/** Order a style's chips for the active use-case profile (chips carry `profiles: [...]` in prompts/*.yaml): chips for
 *  this profile first, untagged chips next, chips tagged only for other profiles last. `general` keeps the file order;
 *  `game` treats untagged chips like game chips, so it keeps today's order. */
export function profileChips(chips, profile = store.profile) {
  if (!profile || profile === 'general') return chips;
  const rank = c => (c.profiles?.includes(profile) ? 0 : !c.profiles ? (profile === 'game' ? 0 : 1) : 2);
  return chips.map((c, i) => [c, i]).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(x => x[0]);
}

let stylesP = null;
export function improveStyles() { if (!stylesP) stylesP = api.improveStyles().catch(e => { stylesP = null; throw e; }); return stylesP; }

/**
 * new PromptBox({label, placeholder, model, task, style, negative, onChange, onApply, imageAsset: () => assetId|null,
 *                context: () => string, rows})
 *   .el  .value  .negative  .set(text)  .setNegative(text)  .setModel(model, task)  .setStyle(styleId)  .focus()
 *   .improved  -> the improve result last applied (studios store it in job meta: {prompt, improved_prompt})
 */
export class PromptBox {
  constructor(o = {}) {
    this.o = o; this.model = o.model || null; this.task = o.task || null; this.style = o.style || null; this.improved = null;
    this.chips = new Set(prefs.get(`chips.${o.memoryKey || 'default'}`, []));
    this.n = prefs.get('improve.n', 3);
    this.backend = prefs.get('improve.backend', '');   // '' = the global default (settings textgen.improve, local unless changed)
    this.ta = h('textarea', { placeholder: o.placeholder || 'Describe what you want… (Thai or English)', rows: o.rows || 3, 'aria-label': o.label || 'Prompt',
      on: { input: () => { this._grow(); this._count(); this.improved = this.improved && this.ta.value === this.improved.applied ? this.improved : null; o.onChange?.(this.value); },
        keydown: e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); o.onSubmit?.(); } } } });
    this.cnt = h('span.cnt');
    this.thai = h('span.badge.info.hidden', 'TH');
    this.impBtn = h('button.btn.btn-sm.btn-improve', { type: 'button', title: 'Improve this prompt for the selected model (local LLM by default; API / CLI backends in the panel)', on: { click: () => this.togglePanel() } }, ic('sparkles', 15), 'Improve');
    this.negWrap = h('div.hidden');
    this.neg = h('input.input.input-sm', { placeholder: 'Negative prompt (what to avoid)', 'aria-label': 'Negative prompt', on: { input: () => o.onChange?.(this.value) } });
    this.negWrap.append(h('label.field', h('span.field-label', ic('ban', 13), 'Negative'), this.neg));
    this.panel = h('div.imp.hidden');
    this.el = h('div.pb',
      o.label ? h('div.field-label', o.label, h('span.spacer'), h('span.dim', { style: { fontWeight: 400, fontSize: '11.5px' } }, '⌘/Ctrl + Enter to generate')) : null,
      h('div.pb-box', this.ta, h('div.pb-tools', this.thai, this.cnt, h('span.spacer'),
        btn('', { icon: 'copy', size: 'sm', kind: 'ghost', title: 'Copy prompt', onClick: () => this.value && copyText(this.value) }),
        btn('', { icon: 'eraser', size: 'sm', kind: 'ghost', title: 'Clear', onClick: () => { this.set(''); this.ta.focus(); } }),
        this.impBtn)),
      this.panel, this.negWrap);
    this.setModel(this.model, this.task);
    this._count();
    bus.on('profile', () => { if (!this.panel.classList.contains('hidden')) this._renderPanel(this.lastResult); });
  }
  get value() { return this.ta.value.trim(); }
  get negative() { return this.negWrap.classList.contains('hidden') ? '' : this.neg.value.trim(); }
  set(text) { this.ta.value = text || ''; this._grow(); this._count(); this.o.onChange?.(this.value); }
  setNegative(text) { this.neg.value = text || ''; }
  focus() { this.ta.focus(); }
  setStyle(style) { this.style = style; if (!this.panel.classList.contains('hidden')) this._renderPanel(); }
  setModel(model, task = this.task) {
    this.model = model; this.task = task;
    const f = model?.features || {};
    const neg = f.negative === true || f.negative === 'cfg';
    this.negWrap.classList.toggle('hidden', !neg || this.o.negative === false);
    if (!this.panel.classList.contains('hidden')) this._renderPanel();
  }
  targetStyle() { return this.style || promptStyleOf(this.model, this.task); }
  _grow() { this.ta.style.height = 'auto'; this.ta.style.height = Math.min(320, Math.max(92, this.ta.scrollHeight)) + 'px'; }
  _count() { const n = this.ta.value.length; this.cnt.textContent = n ? `${n} chars` : ''; this.thai.classList.toggle('hidden', !isThai(this.ta.value)); }
  togglePanel(force) {
    const show = force ?? this.panel.classList.contains('hidden');
    this.panel.classList.toggle('hidden', !show);
    if (show) this._renderPanel();
  }
  /** Chips with `default_for: [profile]` start selected the first time a studio box is used under that profile. */
  _seed(c) {
    if (!c.default_for?.includes(store.profile)) return false;
    const key = `chips.seeded.${this.o.memoryKey || 'default'}.${store.profile}.${c.id}`;
    if (prefs.get(key, false)) return false;
    prefs.set(key, true);
    return true;
  }
  async _renderPanel(result = this.lastResult, state = null) {
    const p = this.panel; p.replaceChildren();
    const styleId = this.targetStyle();
    let styles = [];
    try { styles = (await improveStyles()).styles; } catch { /* shown below */ }
    const style = styles.find(s => s.id === styleId);
    const llm = store.llm;
    p.append(h('div.imp-head', ic('sparkles', 16), h('span.nowrap', 'Improve'), h('span.badge.outline.ellipsis', { title: 'Prompt format of the target model' }, style?.label || styleId || 'pick a model'),
      h('span.spacer'), btn('', { icon: 'x', size: 'sm', kind: 'ghost', title: 'Close', onClick: () => this.togglePanel(false) })));
    if (!styleId) { p.append(h('div.imp-notes', 'Pick a model first: the improver writes the prompt format that model understands.')); return; }
    const chipsEl = h('div.chips');
    for (const c of profileChips(style?.chips || [])) {
      if (this._seed(c)) this.chips.add(c.id);
      chipsEl.append(h(`button.chip${this.chips.has(c.id) ? '.on' : ''}${c.profiles && !c.profiles.includes(store.profile) && store.profile !== 'general' ? '.chip-off' : ''}`, { type: 'button', title: c.hint, on: { click: e => {
        if (this.chips.has(c.id)) this.chips.delete(c.id); else this.chips.add(c.id);
        e.currentTarget.classList.toggle('on'); prefs.set(`chips.${this.o.memoryKey || 'default'}`, [...this.chips]);
      } } }, c.label));
    }
    const go = btn(state === 'busy' ? 'Improving…' : (result ? 'Again' : 'Improve'), { icon: state === 'busy' ? 'loader' : 'wand-sparkles', kind: 'primary', size: 'sm', disabled: state === 'busy', onClick: () => this.run() });
    if (state === 'busy') go.querySelector('.ic')?.classList.add('spin');
    const img = this.o.imageAsset?.();
    if (!this.bpick) this.bpick = backendPicker({ purpose: 'improve', value: this.backend, small: true, label: 'Improve with', onChange: v => { this.backend = v; prefs.set('improve.backend', v); } });
    p.append(chipsEl.childElementCount ? chipsEl : null,
      h('div.row.wrap', h('span.field-label', 'Variations'), seg([0, 2, 3, 4].map(v => ({ value: v, label: String(v) })), this.n, v => { this.n = v; prefs.set('improve.n', v); }),
        img ? h('span.badge.info', ic('image', 12), 'uses the input image') : null, h('span.spacer'), go),
      h('div.imp-backend', h('span.field-label.nowrap', ic('bot', 13), 'With'), this.bpick));
    if (state === 'busy') p.append(h('div.imp-notes', h('span', ic('loader', 14, 'spin'), ' '), this.backend && this.backend !== 'local' ? 'Writing… (external backend, 5-30 s)' : llm?.state === 'ready' || llm?.state === 'external' ? 'Writing…' : 'Loading the prompt model (5-20 s the first time)…'));
    if (state?.error) p.append(h('div.imp-notes', { style: { color: 'var(--err)' } }, state.error, state.fix ? h('div.dim', state.fix) : null));
    if (!result) return;
    const apply = result.apply || { prompt: 'prompt' };
    const useMain = (text, mode) => {
      if (mode === 'insert') this.set(this.value ? `${this.value}\n${text}` : text); else this.set(text);
      if (apply.negative && result.negative && !this.negWrap.classList.contains('hidden')) this.setNegative(result.negative);
      const extras = {};
      for (const [k, field] of Object.entries(apply)) if (k !== 'prompt' && k !== 'negative' && result[k]) extras[field] = result[k];
      if (apply.prompt && apply.prompt !== 'prompt') { extras[apply.prompt] = text; }
      if (Object.keys(extras).length) this.o.onApply?.(extras, result);
      this.improved = { ...result, applied: this.ta.value };
      bus.emit('improve:applied', { box: this, result });
    };
    const promptGoesToBox = !apply.prompt || apply.prompt === 'prompt';
    const sugg = (text, main) => h(`div.imp-sugg${main ? '.main' : ''}`, h('div', text),
      h('div.acts', btn(promptGoesToBox ? 'Replace' : 'Use', { size: 'sm', kind: main ? 'primary' : '', icon: 'check', onClick: () => useMain(text, 'replace') }),
        promptGoesToBox ? btn('Insert', { size: 'sm', icon: 'plus', onClick: () => useMain(text, 'insert') }) : null,
        btn('', { size: 'sm', kind: 'ghost', icon: 'copy', title: 'Copy', onClick: () => copyText(text) })));
    p.append(h('div.row.wrap', { style: { minWidth: 0 } }, result.translated ? h('span.badge.info', ic('type', 12), 'Translated from Thai') : null,
      result.backend && result.backend.kind !== 'local' ? generatedBy(result) : h('span.dim.ellipsis', { style: { fontSize: '11.5px' }, title: result.llm }, `${result.seconds}s · ${String(result.llm).split(' (')[0]}`)));
    p.append(sugg(result.prompt, true));
    if (result.negative) p.append(h('div.imp-sugg', h('div.field-label', ic('ban', 13), 'Negative'), h('div', result.negative)));
    if (result.lyrics) p.append(h('div.imp-sugg', h('div.field-label', ic('music', 13), 'Lyrics'), h('pre', result.lyrics)));
    if (result.line) p.append(h('div.imp-sugg', h('div.field-label', ic('mic', 13), 'Line'), h('div', result.line)));
    if (result.variations?.length) {
      p.append(h('div.field-label', `Variations (${result.variations.length})`));
      for (const v of result.variations) p.append(sugg(v, false));
    }
    if (result.notes) p.append(h('div.imp-notes', ic('info', 13), ' ', result.notes));
    if (result.warnings?.length) p.append(h('div.imp-notes', { style: { color: 'var(--warn)' } }, ic('triangle-alert', 13), ' Checks: ', result.warnings.join(', ')));
    p.append(h('div.row', btn('Keep original', { size: 'sm', kind: 'ghost', onClick: () => { this.set(result.original); this.togglePanel(false); } })));
  }
  async run() {
    const text = this.value;
    const img = this.o.imageAsset?.() || null;
    if (!text && !img) { this.ta.focus(); this._renderPanel(this.lastResult, { error: 'Write an idea first (Thai or English).' }); return; }
    this._renderPanel(this.lastResult, 'busy');
    try {
      const body = { text: this.improved && text === this.improved.applied ? this.improved.original : text, target_style: this.style || undefined,
        model_id: this.model?.id, task: this.task, n: this.n, chips: [...this.chips], context: [store.profiles[store.profile]?.context, this.o.context?.()].filter(Boolean).join(' '), image_asset: img || undefined, backend: this.backend || undefined, project: store.project };
      const r = await api.improve(body);
      this.lastResult = r;
      this._renderPanel(r);
    } catch (e) {
      this._renderPanel(this.lastResult, { error: e.message, fix: e.fix });
      if (e.kind === 'network') toastError(e);
    }
  }
}

/** Job meta for a prompt box: what the user typed vs what was sent (DESIGN 4.4 prompt / improved_prompt). */
export function promptMeta(box) {
  const imp = box.improved;
  if (imp && box.value === imp.applied.trim()) return { prompt: imp.original, improved_prompt: box.value, improve_style: imp.style };
  return { prompt: box.value };
}
