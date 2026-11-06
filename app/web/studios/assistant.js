// Assistant: describe what you want ("10 medieval village props with matching ambience and footsteps") and the
// assistant proposes a PLAN using only the studio's read-only tools (list models, search assets, read meta) and
// propose_plan. It never runs anything itself: the plan appears as an editable card, you approve all or some steps,
// and the studio runs them through the normal job API (cost guard included). Progress, results and continue /
// refine stay in the chat. Backends per chat: Claude Code CLI (default when installed), other CLIs, API LLMs, local.
// Server: app/plugins/assistant.py. Also hosts the ✍ Write tools and the text-backend settings.
import * as S from '../core/sdk.js';

const { h, ic, btn, toast, toastError, store, bus, api } = S;
const API = '/api/x/assistant';
const PROMPT_TASKS = ['t2i', 'edit', 'multi_ref', 'inpaint', 'rgba', 'sfx', 'ambience', 'music', 'song', 'tts', 'voice_design', 'voice_clone'];
const STARTERS = [
  'A set of 10 medieval village props (barrel, cart, well, market stall…) as 3D models, with a matching village ambience loop and footsteps on cobblestone.',
  'Make 4 UI sound effects for a cozy farming game: click, confirm, cancel, coin pickup.',
  'Concept art for a crystal cave level: 4 variations, then the best one as a transparent prop.',
  'What can I make here for free commercial use?',
];

const CSS = `
.as { display: grid; grid-template-columns: 270px minmax(0, 1fr); height: 100%; min-height: 0; }
.as-side { border-right: 1px solid var(--line-soft); background: var(--bg-2); display: flex; flex-direction: column; min-height: 0; padding: 14px 12px; gap: 10px; }
.as-chats { display: flex; flex-direction: column; gap: 4px; overflow: auto; min-height: 0; flex: 1; }
.as-chat { display: block; text-align: left; border: 1px solid transparent; background: none; border-radius: 9px; padding: 8px 10px; cursor: pointer; color: var(--text); min-width: 0; }
.as-chat:hover { background: var(--panel-2); }
.as-chat.on { background: var(--panel); border-color: var(--line); }
.as-chat .t { font-size: 13px; font-weight: 600; } .as-chat .s { font-size: 11.5px; color: var(--text-3); }
.as-main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.as-head { display: flex; align-items: center; gap: 10px; padding: 10px 18px; border-bottom: 1px solid var(--line-soft); flex-wrap: wrap; min-width: 0; }
.as-head .tg-pick { width: min(330px, 100%); }
.as-thread { flex: 1; overflow: auto; padding: 18px; display: flex; flex-direction: column; gap: 14px; min-height: 0; }
.as-inner { max-width: 900px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }
.as-msg { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.as-msg.user { align-items: flex-end; }
.as-bub { padding: 10px 13px; border-radius: 13px; font-size: 14px; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; max-width: min(720px, 100%); }
.as-msg.user .as-bub { background: var(--accent-soft); border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent); }
.as-msg.assistant .as-bub { background: var(--panel); border: 1px solid var(--line-soft); }
.as-meta { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 11.5px; color: var(--text-3); min-width: 0; }
.as-err { color: var(--err); font-size: 13px; } .as-err .dim { color: var(--text-3); }
.as-comp { border-top: 1px solid var(--line-soft); padding: 12px 18px 14px; background: var(--bg-2); }
.as-comp-in { max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; }
.as-comp textarea { width: 100%; min-height: 64px; max-height: 220px; }
.as-starters { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.as-starter { text-align: left; padding: 12px; border-radius: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--text); cursor: pointer; font-size: 13px; line-height: 1.45; }
.as-starter:hover { border-color: var(--accent); }
.as-plan { border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--line)); border-radius: 14px; background: var(--panel); padding: 14px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.as-plan-head { display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; min-width: 0; }
.as-tot { display: flex; gap: 6px; flex-wrap: wrap; }
.as-step { border: 1px solid var(--line-soft); border-radius: 11px; padding: 10px; display: flex; flex-direction: column; gap: 8px; background: var(--panel-2); min-width: 0; }
.as-step.off { opacity: .55; }
.as-step.bad { border-color: color-mix(in srgb, var(--err) 55%, var(--line)); }
.as-step-top { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; min-width: 0; }
.as-step-top .input { flex: 1 1 180px; min-width: 0; }
.as-step-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 8px; min-width: 0; }
.as-step-grid .select { width: 100%; min-width: 0; }
.as-step textarea { width: 100%; min-height: 40px; }
.as-step details summary { cursor: pointer; font-size: 12.5px; color: var(--text-2); }
.as-step .pf { margin-top: 8px; }
.as-ins { display: flex; gap: 6px; flex-wrap: wrap; font-size: 12px; }
.as-outs { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
.as-outs .acard.audio { grid-column: 1 / -1; }
.as-steperr { color: var(--err); font-size: 12px; }
.as-plan-foot { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.as-be { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--line-soft); font-size: 13px; }
.as-be .dim { font-size: 11.5px; }
.as-mob { display: none; }
@media (max-width: 820px) {
  .as { grid-template-columns: minmax(0, 1fr); }
  .as-side { display: none; }
  .as-mob { display: flex; gap: 8px; align-items: center; min-width: 0; flex: 1 1 100%; }
  .as-mob .select { flex: 1; min-width: 0; }
  .as-head, .as-thread, .as-comp { padding-left: 12px; padding-right: 12px; }
  .as-step-grid { grid-template-columns: minmax(0, 1fr); }
  .as-head .tg-pick { width: 100%; }
  .as-head .as-title, .as-head .as-del { display: none; }
  .as-head > .row { flex: 1 1 100%; }
}
`;

export default S.defineStudio({
  id: 'assistant', title: 'Assistant', icon: 'bot', color: 'var(--c-text)', order: 45, section: 'workspace', short: 'Assistant',
  mount(el) {
    if (!document.getElementById('as-css')) document.head.append(h('style#as-css', CSS));
    el.style.height = '100%';
    const P = new Page(el);
    return { show: () => P.show(), route: r => P.route(r) };
  },
});

class Page {
  constructor(el) {
    this.chat = null; this.chats = []; this.cards = new Map();
    this.side = h('aside.as-side');
    this.head = h('div.as-head');
    this.thread = h('div.as-thread');
    this.inner = h('div.as-inner');
    this.thread.append(this.inner);
    this.comp = h('div.as-comp');
    el.append(h('div.as', this.side, h('div.as-main', this.head, this.thread, this.comp)));
    this.buildSide(); this.buildComposer();
    bus.on('assistant', ev => this.onEvent(ev));
    bus.on('project', () => { this.chat = null; this.want = null; this.loadChats(true); });
    this.loadChats(true);
  }
  show() { this.loadChats(false); }
  route(r) { if (r.sub && r.sub !== this.chat?.id) { this.want = r.sub; this.open(r.sub); } }

  // ---------------------------------------------------------------- side
  buildSide() {
    this.chatList = h('div.as-chats');
    this.side.append(btn('New chat', { icon: 'plus', kind: 'primary', onClick: () => this.newChat() }), this.chatList,
      h('div.col', { style: { gap: '6px' } }, btn('Write tools', { icon: 'pen-line', onClick: e => this.writeMenu(e.currentTarget) }),
        btn('Text backends', { icon: 'settings-2', onClick: () => this.settings() })));
  }
  async loadChats(openFirst) {
    try { this.chats = (await api.get(`${API}/chats`, { project: store.project })).chats; } catch (e) { this.chats = []; }
    this.drawChats();
    if (this.want) { this.want = null; return; }  // a #/assistant/<chat id> route is opening that chat
    if (openFirst || !this.chat) { if (this.chats[0]) this.open(this.chats[0].id); else this.renderEmpty(); }
  }
  drawChats() {
    this.chatList.replaceChildren(...(this.chats.length ? this.chats.map(c => h(`button.as-chat${this.chat?.id === c.id ? '.on' : ''}`, { type: 'button', on: { click: () => this.open(c.id) } },
      h('div.t.ellipsis', c.title || 'New chat'), h('div.s.ellipsis', c.busy ? 'thinking…' : (c.last || `${c.messages} messages`)))) : [h('div.dim', { style: { fontSize: '12.5px', padding: '4px' } }, 'No chats yet.')]));
    if (this.mobSel) this.drawMob();
  }
  async newChat(text) {
    try {
      const c = await api.post(`${API}/chats`, { project: store.project, backend: this.pick?.value || null });
      this.chats.unshift({ id: c.id, title: c.title, messages: 0 });
      this.chat = c; this.render();
      this.drawChats();
      if (text) { this.input.value = text; this.send(); }
    } catch (e) { toastError(e); }
  }
  async open(id) {
    try { this.chat = await api.get(`${API}/chats/${id}`, { project: store.project }); } catch (e) { toastError(e); return; }
    this.render(); this.drawChats();
  }

  // ---------------------------------------------------------------- header + composer
  drawHead() {
    const c = this.chat;
    const delChat = async () => {
      if (!await S.confirmDialog('Delete this chat? Generated assets stay in the library.', { danger: true, ok: 'Delete' })) return;
      await api.del(`${API}/chats/${c.id}?project=${encodeURIComponent(store.project)}`).catch(toastError); this.chat = null; this.loadChats(true);
    };
    const del = c ? btn('', { icon: 'trash-2', kind: 'ghost', size: 'sm', title: 'Delete chat', onClick: delChat }) : null;
    this.mobSel = h('select.select', { 'aria-label': 'Chat', on: { change: e => e.target.value === '__new' ? this.newChat() : this.open(e.target.value) } });
    this.drawMob();
    const mob = h('div.as-mob', this.mobSel, btn('', { icon: 'pen-line', title: 'Write tools', onClick: e => this.writeMenu(e.currentTarget) }), btn('', { icon: 'settings-2', title: 'Text backends', onClick: () => this.settings() }),
      c ? btn('', { icon: 'trash-2', kind: 'ghost', title: 'Delete chat', onClick: delChat }) : null);
    this.pick = S.backendPicker({ purpose: 'assistant', value: c?.backend || '', label: 'Assistant backend', onChange: async v => {
      if (!this.chat) return;
      try { this.chat = await api.patch(`${API}/chats/${this.chat.id}?project=${encodeURIComponent(store.project)}`, { backend: v || null }); } catch (e) { toastError(e); }
    } });
    const title = c ? h('input.input.input-sm.as-title', { value: c.title || '', 'aria-label': 'Chat title', style: { flex: '1 1 160px', minWidth: 0, fontWeight: 600 },
      on: { change: async e => { try { await api.patch(`${API}/chats/${c.id}?project=${encodeURIComponent(store.project)}`, { title: e.target.value }); this.loadChats(false); } catch (err) { toastError(err); } } } }) : h('b.grow', 'Assistant');
    this.head.replaceChildren(mob, title, h('div.row', { style: { minWidth: 0, flex: '0 1 auto' } }, h('span.field-label.nowrap', ic('bot', 14), 'Backend'), this.pick),
      del ? Object.assign(del, { className: del.className + ' as-del' }) : null);
  }
  drawMob() {
    this.mobSel.replaceChildren(h('option', { value: '__new' }, '+ New chat'), ...this.chats.map(c => h('option', { value: c.id, selected: c.id === this.chat?.id }, c.title || 'New chat')));
    if (!this.chat) this.mobSel.value = '__new';
  }
  buildComposer() {
    this.input = h('textarea.textarea', { rows: 2, placeholder: 'Describe what you want to make… (Thai or English). Enter to send, Shift+Enter for a new line.', 'aria-label': 'Message',
      on: { keydown: e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); } } } });
    this.sendBtn = btn('Send', { kind: 'primary', icon: 'send', onClick: () => this.send() });
    this.stopBtn = btn('Stop', { icon: 'square', onClick: () => this.stop() });
    this.stopBtn.classList.add('hidden');
    this.tos = h('div.dim', { style: { fontSize: '11.5px' } }, ic('shield-check', 12), ' The assistant can only read models and assets and propose plans; nothing runs until you approve it.');
    this.comp.append(h('div.as-comp-in', this.input, h('div.row.wrap', this.tos, h('span.spacer'), this.stopBtn, this.sendBtn)));
  }
  busy() { return !!this.chat?.messages?.some(m => m.pending); }
  drawBusy() { const b = this.busy(); this.sendBtn.disabled = b; this.stopBtn.classList.toggle('hidden', !b); }
  async send(textArg) {
    const text = (textArg ?? this.input.value).trim();
    if (!text) { this.input.focus(); return; }
    if (!this.chat) return this.newChat(text);
    if (this.busy()) return toast('The assistant is still answering.', { kind: 'warn' });
    this.input.value = '';
    try {
      const r = await api.post(`${API}/chats/${this.chat.id}/message`, { project: store.project, text, backend: this.pick?.value || null });
      this.upsert(r.message);
    } catch (e) { this.input.value = text; toastError(e); }
  }
  async stop() { if (this.chat) await api.post(`${API}/chats/${this.chat.id}/cancel`).catch(toastError); }

  // ---------------------------------------------------------------- thread
  renderEmpty() {
    this.chat = null; this.drawHead(); this.drawBusy();
    this.inner.replaceChildren(this.starters());
  }
  starters() {
    return h('div.col', { style: { gap: '14px', paddingTop: '6vh' } },
      h('div', { style: { textAlign: 'center' } }, ic('bot', 34), h('h2', { style: { margin: '8px 0 4px' } }, 'What should we make?'),
        h('div.dim', 'The assistant plans with the models installed here; you review, edit and approve before anything runs.')),
      h('div.as-starters', ...STARTERS.map(t => h('button.as-starter', { type: 'button', on: { click: () => this.chat ? this.send(t) : this.newChat(t) } }, t))));
  }
  render() {
    this.drawHead(); this.cards.clear();
    const msgs = this.chat?.messages || [];
    this.inner.replaceChildren(...(msgs.length ? msgs.map(m => this.msgEl(m)) : [this.starters()]));
    this.drawBusy(); this.scroll();
  }
  scroll() { requestAnimationFrame(() => { this.thread.scrollTop = this.thread.scrollHeight; }); }
  upsert(m) {
    if (!this.chat) return;
    const i = this.chat.messages.findIndex(x => x.id === m.id);
    if (i >= 0) this.chat.messages[i] = { ...this.chat.messages[i], ...m }; else this.chat.messages.push(m);
    if (m.pending === undefined && i >= 0) delete this.chat.messages[i].pending;
    const old = this.inner.querySelector(`[data-mid="${m.id}"]`);
    const el = this.msgEl(this.chat.messages.find(x => x.id === m.id));
    if (old) old.replaceWith(el); else { if (this.chat.messages.length === 1) this.inner.replaceChildren(); this.inner.append(el); this.scroll(); }
    this.drawBusy();
  }
  onEvent(ev) {
    if (ev.project !== store.project) return;
    if (ev.kind === 'message' && ev.message?.role === 'assistant' && !ev.message.pending) this.loadChats(false);
    if (!this.chat || ev.chat !== this.chat.id) return;
    if (ev.kind === 'message') { this.upsert(ev.message); if (!ev.message.pending) this.scroll(); }
    else if (ev.kind === 'status') { const el = this.inner.querySelector(`[data-mid="${ev.message_id}"] .as-act`); if (el) el.textContent = `using ${ev.activity.replace('_', ' ')}…`; }
    else if (ev.kind === 'run') {
      const m = this.chat.messages.find(x => x.id === ev.message_id); if (!m) return;
      m.run = ev.run;
      this.cards.get(m.id)?.setRun(ev.run);
    }
  }
  msgEl(m) {
    if (m.role === 'user') return h('div.as-msg.user', { dataset: { mid: m.id } }, h('div.as-bub', m.text));
    const meta = h('div.as-meta', ic('bot', 13), h('span', 'Assistant'),
      m.backend ? h('span', `· ${m.backend.label}`) : null,
      m.backend?.subscription ? h('span.badge.warn', { title: 'Counts against your subscription plan limits (personal use).' }, 'plan quota') : null,
      m.backend?.tested === false ? h('span.badge.outline', 'untested backend') : null,
      m.seconds != null ? h('span', `· ${m.seconds}s`) : null,
      m.tools?.length ? h('span', { title: m.tools.join(', ') }, `· looked up: ${summarizeTools(m.tools)}`) : null);
    const kids = [meta];
    if (m.pending) kids.push(h('div.as-bub', h('span.row', ic('loader', 14, 'spin'), h('span.as-act', 'thinking…'))));
    else if (m.text) kids.push(h('div.as-bub', m.text));
    if (m.error) kids.push(h('div.as-err', ic('circle-alert', 13), ' ', m.error.message, m.error.fix ? h('div.dim', m.error.fix) : null,
      h('div.row', { style: { marginTop: '6px' } }, btn('Retry', { size: 'sm', icon: 'rotate-ccw', onClick: () => { const last = [...this.chat.messages].reverse().find(x => x.role === 'user'); if (last) this.send(last.text); } }))));
    if (m.warnings?.length) kids.push(h('div.dim', { style: { fontSize: '11.5px', color: 'var(--warn)' } }, m.warnings.join('; ')));
    if (m.plan) { const card = new PlanCard(this, m); this.cards.set(m.id, card); kids.push(card.el); }
    return h('div.as-msg.assistant', { dataset: { mid: m.id } }, ...kids);
  }

  // ---------------------------------------------------------------- write tools
  writeMenu(anchor) {
    S.menu(anchor, [{ head: 'Write with a text backend' },
      { label: 'Character lines (barks)', icon: 'mic', sub: 'from a description + personality → Sound line sheet', onClick: () => S.writeHelper('barks', { insertLabel: 'Send to a new line sheet',
        onInsert: rows => S.sendTo('sound', [], { action: 'sheet_rows', sub: 'sheets', params: { kind: 'lines', rows, name: `${rows[0]?.character || 'Character'} lines` } }) }) },
      { label: 'Design doc → SFX event list', icon: 'zap', sub: 'events with Stable Audio prompts → Sound SFX list', onClick: () => S.writeHelper('sfx_events', { insertLabel: 'Send to a new SFX list',
        onInsert: rows => S.sendTo('sound', [], { action: 'sheet_rows', sub: 'sheets', params: { kind: 'sfx', rows, name: 'SFX from design doc' } }) }) },
      { label: 'Design doc → asset list', icon: 'list', sub: 'images / 3D / sound with prompts → a plan here', onClick: () => S.writeHelper('asset_list', { insertLabel: 'Plan these',
        onInsert: rows => this.send(`Make a plan for these assets (use the ready models, sensible takes):\n${rows.map(r => `- ${r.name} [${r.kind}]: ${r.prompt}`).join('\n')}`) }) },
      { label: 'Name, tag and describe assets', icon: 'tag', sub: 'pick assets → names, tags, descriptions', onClick: () => S.writeHelper('asset_meta', {}) }]);
  }

  // ---------------------------------------------------------------- settings
  async settings() {
    let d;
    try { d = await api.textBackends(true); } catch (e) { return toastError(e); }
    const body = h('div.col', { style: { gap: '14px' } });
    const md = S.modal({ title: 'Text backends', icon: 'bot', width: 720, body, foot: [btn('Close', { kind: 'ghost', onClick: () => md.close() })] });
    const draw = () => {
      const defs = h('div.col', { style: { gap: '10px' } });
      for (const [p, lab, hint] of [['improve', '✨ Improve', 'Default for every Improve button (the panel can override per click).'], ['write', '✍ Write tools', 'Lines, SFX lists, asset lists, names & tags.'], ['assistant', 'Assistant', 'New chats (each chat can pick its own).']]) {
        const pk = S.backendPicker({ purpose: p, value: d.settings?.[p] || d.defaults[p] || 'local', allowDefault: false, onChange: async v => { try { d = await api.setTextBackends({ [p]: v }); S.textBackends(true); toast('Saved', { kind: 'ok', timeout: 1200 }); } catch (e) { toastError(e); } } });
        defs.append(h('div.field', h('span.field-label', lab, h('span.dim', { style: { fontWeight: 400, marginLeft: '6px' } }, hint)), pk));
      }
      const rows = h('div');
      for (const b of d.backends) {
        const st = b.available ? h('span.badge.ok', 'available') : h('span.badge.warn', { title: b.reason }, b.kind === 'cli' && !b.installed ? 'not installed' : 'unavailable');
        rows.append(h('div.as-be', h('div', { style: { minWidth: 0 } }, h('div.row.wrap', h('b', b.label), h('span.dim', b.detail || ''), st,
          b.kind === 'cli' && !b.tested ? h('span.badge.outline', 'untested') : null, b.kind === 'cli' && b.logged_in ? h('span.badge.ok', 'logged in') : null),
          h('div.dim', [b.cost, b.vision ? 'vision' : null, b.tools === 'mcp' ? 'tools via MCP' : b.tools === 'openai' ? 'tool calling' : 'JSON plan', b.reason].filter(Boolean).join(' · '))),
        btn('Test', { size: 'sm', disabled: !b.available, title: 'One tiny call (CLI: counts against your plan)', onClick: async e => {
          const t = e.currentTarget; t.disabled = true;
          try { const r = await api.post(`${API}/backends/test`, { backend: b.id }); toast(`${r.backend.label}: “${r.data?.greeting}” in ${r.seconds}s`, { kind: 'ok' }); } catch (err) { toastError(err, b.label); } finally { t.disabled = false; }
        } })));
      }
      const prov = h('select.select', ...['openrouter', 'openai', 'gemini', ...Object.keys(d.settings?.endpoints || {}).map(n => `openai_compat:${n}`)].map(p => h('option', { value: p }, p)));
      const model = h('input.input', { placeholder: 'model id, e.g. anthropic/claude-haiku-4.5' });
      const epName = h('input.input', { placeholder: 'name, e.g. lab' }); const epUrl = h('input.input', { placeholder: 'http://127.0.0.1:7880/v1' });
      const add = h('div.col', { style: { gap: '8px' } },
        h('div.field-label', 'Add an API model (keys live on the Providers page)'),
        h('div.row.wrap', h('div', { style: { flex: '0 1 200px', minWidth: 0 } }, prov), h('div.grow', { style: { flexBasis: '200px' } }, model),
          btn('Add', { size: 'sm', icon: 'plus', onClick: async () => { if (!model.value.trim()) return; try { d = await api.setTextBackends({ api_models: [...(d.settings?.api_models || []), { provider: prov.value, model: model.value.trim() }] }); S.textBackends(true); draw(); } catch (e) { toastError(e); } } })),
        h('div.field-label', 'OpenAI-compatible endpoint (llama.cpp, vLLM, LM Studio…)'),
        h('div.row.wrap', h('div', { style: { flex: '0 1 160px', minWidth: 0 } }, epName), h('div.grow', { style: { flexBasis: '200px' } }, epUrl),
          btn('Save', { size: 'sm', icon: 'check', onClick: async () => { try { d = await api.setTextBackends({ endpoints: { ...(d.settings?.endpoints || {}), [epName.value.trim()]: epUrl.value.trim() } }); S.textBackends(true); draw(); } catch (e) { toastError(e); } } })),
        (d.settings?.api_models || []).length ? h('div.chips', ...(d.settings.api_models).map((m, i) => h('span.chip', `${m.provider}/${m.model}`, h('button.btn.btn-ghost.btn-sm.btn-icon', { type: 'button', title: 'Remove', on: { click: async () => {
          d = await api.setTextBackends({ api_models: d.settings.api_models.filter((_, j) => j !== i) }); S.textBackends(true); draw(); } } }, ic('x', 12))))) : null);
      body.replaceChildren(h('div.card.card-pad', { style: { borderColor: 'var(--warn)' } }, h('div.row', ic('triangle-alert', 16), h('b', 'Subscription CLIs')), h('div.dim', { style: { fontSize: '12.5px', marginTop: '4px' } }, d.tos)),
        defs, h('div', h('div.field-label', 'Backends'), rows), add);
    };
    draw();
  }
}

function summarizeTools(list) {
  const c = {}; for (const t of list) c[t] = (c[t] || 0) + 1;
  return Object.entries(c).map(([k, n]) => `${k.replace('_', ' ')}${n > 1 ? ` ×${n}` : ''}`).join(', ');
}

// ------------------------------------------------------------------ plan card (editable, approve, live run)
class PlanCard {
  constructor(page, msg) {
    this.page = page; this.msg = msg;
    this.v = msg.plan;                                     // {plan, ok, errors, warnings, totals}
    this.plan = structuredClone(msg.plan.plan || { steps: [] });
    this.run = msg.run || null;
    this.sel = new Set(this.plan.steps.filter(s => s.enabled !== false).map(s => s.id));
    this.el = h('div.as-plan');
    this.dirty = false;
    this.render();
  }
  setRun(run) { this.run = run; this.render(); }
  running() { return this.run?.status === 'running'; }
  render() {
    const v = this.v, t = v.totals || {};
    const errs = v.errors || {};
    const head = h('div.as-plan-head', h('div.grow', { style: { minWidth: '200px' } },
      h('div.row', ic('list-checks', 16), h('b', this.plan.title || 'Plan'), this.dirty ? h('span.badge.info', 'edited') : null),
      this.plan.summary ? h('div.dim', { style: { fontSize: '13px', marginTop: '4px' } }, this.plan.summary) : null),
      h('div.as-tot', h('span.badge.outline', `${this.plan.steps.length} steps`), t.est_s ? h('span.badge.outline', ic('clock', 12), `~${S.fmtDur(t.est_s)}`) : null,
        h(`span.badge${t.cloud_steps ? '.warn' : '.ok'}`, { title: 'Estimated cloud cost (local models are free)' }, t.cloud_steps ? `cloud ≈ $${(t.est_usd || 0).toFixed(3)}${t.usd_unknown ? '+' : ''}` : 'local · $0')));
    const kids = [head];
    if (errs.plan) kids.push(h('div.as-steperr', errs.plan.join(' ')));
    this.plan.steps.forEach((s, i) => kids.push(this.stepEl(s, i, errs[s.id] || [])));
    if (v.warnings?.length) kids.push(h('div.dim', { style: { fontSize: '12px', color: 'var(--warn)' } }, ic('triangle-alert', 12), ' ', v.warnings.join(' · ')));
    const r = this.run;
    const foot = h('div.as-plan-foot');
    if (this.running()) foot.append(h('span.badge.accent', ic('loader', 12, 'spin'), 'running'), btn('Cancel run', { size: 'sm', icon: 'square', onClick: () => api.post(`${API}/runs/${r.id}/cancel`).catch(toastError) }));
    else {
      foot.append(btn(this.sel.size === this.plan.steps.length ? (r ? 'Run all again' : 'Approve & run all') : `Run ${this.sel.size} selected`, { kind: 'primary', icon: 'play', disabled: !this.sel.size, onClick: () => this.approve() }),
        btn('Check', { size: 'sm', icon: 'badge-check', title: 'Validate the edited plan', onClick: () => this.validate() }));
      if (r) foot.append(h('span', { class: `badge ${r.status === 'done' ? 'ok' : r.status === 'failed' ? 'err' : 'warn'}` }, `last run: ${r.status}`),
        btn('Continue', { size: 'sm', icon: 'arrow-right', title: 'Ask the assistant what to do next with these results', onClick: () => this.page.send('The plan has run (results are in the chat context). Suggest the next step, or propose a follow-up plan.') }));
      foot.append(btn('Refine…', { size: 'sm', kind: 'ghost', icon: 'pencil', onClick: () => { this.page.input.value = 'Refine the plan: '; this.page.input.focus(); } }));
    }
    kids.push(foot);
    this.el.replaceChildren(...kids);
  }
  stepEl(s, i, errs) {
    const rs = this.run?.steps?.[s.id];
    const on = this.sel.has(s.id);
    const lock = this.running();
    const cb = h('input', { type: 'checkbox', checked: on, disabled: lock, 'aria-label': `Include step ${s.id}`, on: { change: e => { e.target.checked ? this.sel.add(s.id) : this.sel.delete(s.id); this.render(); } } });
    const title = h('input.input.input-sm', { value: s.title || '', disabled: lock, 'aria-label': 'Step title', on: { change: e => { s.title = e.target.value; this.touch(); } } });
    const stBadge = rs ? h(`span.badge.${{ done: 'ok', failed: 'err', blocked: 'err', running: 'accent', waiting: 'outline', skipped: 'warn', cancelled: 'warn' }[rs.status] || 'outline'}`,
      rs.status === 'running' ? ic('loader', 11, 'spin') : null, rs.status) : null;
    const top = h('div.as-step-top', cb, h('span.badge.outline.mono', s.id), title, stBadge,
      s.est_s ? h('span.dim', { style: { fontSize: '11.5px' } }, `~${S.fmtDur(s.est_s)}`) : null,
      s.cloud ? h('span.badge.warn', `$${(s.est_usd ?? 0).toFixed(3)}`) : null,
      s.license && !s.license.commercial ? h('span.badge.err', { title: s.license.name }, 'NC') : null);
    const kids = [top];
    if (s.tool) kids.push(h('div.dim', { style: { fontSize: '12.5px' } }, ic('wand', 12), ` CPU tool: ${s.tool} ${Object.entries(s.params || {}).map(([k, v]) => `${k}=${v}`).join(' ')}`));
    else {
      const tasks = Object.entries(store.tasks || {}).filter(([k]) => !['improve', 'test', 'qa_clap', 'qa_asr'].includes(k) || k === s.task);
      const taskSel = h('select.select', { disabled: lock, 'aria-label': 'Task', on: { change: e => { s.task = e.target.value; s.model = (S.modelsFor(s.task).find(m => m.status.status === 'ready') || {}).id || null; this.touch(true); } } },
        ...tasks.map(([k, t]) => h('option', { value: k, selected: k === s.task }, t.label || k)));
      const models = S.modelsFor(s.task, { includeTest: true }).filter(m => !m.test || m.id === s.model || store.ui?.show_test_models);
      const modelSel = h('select.select', { disabled: lock, 'aria-label': 'Model', on: { change: e => { s.model = e.target.value; this.touch(true); } } },
        ...models.map(m => h('option', { value: m.id, selected: m.id === s.model, disabled: m.status.status !== 'ready' }, `${m.name}${m.license?.commercial === false ? ' (NC)' : ''}${m.status.status !== 'ready' ? ' — not ready' : ''}`)));
      kids.push(h('div.as-step-grid', taskSel, modelSel));
      const pkey = 'text' in (s.params || {}) && !('prompt' in (s.params || {})) ? 'text' : 'prompt';
      const wantsText = (s.params && (s.params.prompt != null || s.params.text != null)) || PROMPT_TASKS.includes(s.task);
      const tx = h('textarea.textarea', { rows: 2, disabled: lock, placeholder: pkey === 'prompt' ? 'prompt' : 'text to speak', 'aria-label': pkey });
      tx.value = s.params?.[pkey] || '';
      tx.addEventListener('change', () => { s.params = { ...s.params, [pkey]: tx.value }; this.touch(); });
      if (wantsText) kids.push(tx);
      const m = store.modelsById.get(s.model);
      if (m && (m.params || []).some(p => !['asset', 'assets'].includes(p.type))) {
        const det = h('details', h('summary', `Settings: ${Object.entries(s.params || {}).filter(([k]) => k !== 'prompt' && k !== 'text').map(([k, v]) => `${k}=${typeof v === 'object' ? '…' : v}`).join(', ') || 'defaults'}`));
        let pf = null;
        det.addEventListener('toggle', () => {
          if (!det.open || pf) return;
          pf = new S.ParamForm({ model: m, task: s.task, memoryKey: `assistant.${s.id}`, exclude: ['prompt', 'negative', 'text'],
            onChange: () => { if (!pf) return; s.params = { ...s.params, ...pf.values() }; this.dirty = true; } });
          pf.setValues(s.params || {});
          det.append(pf.el);
        });
        kids.push(det);
      }
    }
    if (s.inputs?.length) kids.push(h('div.as-ins', ic('arrow-left', 12), ...s.inputs.map(x => h('span.chip', x.step ? `from ${x.step} (${x.use || 'first'})` : `asset ${String(x.asset).slice(-6)}`))));
    if (errs.length) kids.push(h('div.as-steperr', ic('circle-alert', 12), ' ', errs.join(' ')));
    if (rs?.error) kids.push(h('div.as-steperr', rs.error, rs.fix ? h('div.dim', rs.fix) : null));
    if (rs?.jobs?.length && rs.status !== 'done') {
      const box = h('div.col', { style: { gap: '6px' } });
      for (const jid of rs.jobs.slice(0, 6)) {
        const j = store.jobs.get(jid);
        if (j) box.append(S.jobCard(j, { compact: true }));
        else api.job(jid).then(jj => { store.jobs.set(jj.id, jj); box.append(S.jobCard(jj, { compact: true })); }).catch(() => {});
      }
      kids.push(box);
    }
    if (rs?.outputs?.length) {
      const outs = h('div.as-outs');
      api.assets({ ids: rs.outputs.slice(0, 24).join(','), limit: 50 }).then(r => {
        outs.replaceChildren(...r.items.map(a => S.assetCard(a, { onOpen: x => S.openAsset(x, { list: rs.outputs }) })));
      }).catch(() => {});
      kids.push(outs);
    }
    return h(`div.as-step${on ? '' : '.off'}${errs.length ? '.bad' : ''}`, ...kids);
  }
  touch(rerender = false) { this.dirty = true; if (rerender) this.validate(); }
  payload() { return { ...this.plan, steps: this.plan.steps.map(s => ({ ...s })) }; }
  async validate() {
    try {
      this.v = await api.post(`${API}/chats/${this.page.chat.id}/validate`, { project: store.project, plan: this.payload() });
      this.plan = structuredClone(this.v.plan);
      this.render();
      if (this.v.ok) toast('Plan looks good.', { kind: 'ok', timeout: 1500 });
    } catch (e) { toastError(e); }
  }
  async approve() {
    const steps = [...this.sel];
    const cloud = this.plan.steps.filter(s => this.sel.has(s.id) && s.cloud);
    if (cloud.length && !await S.confirmDialog(`${cloud.length} step(s) use cloud models (≈ $${cloud.reduce((a, s) => a + (s.est_usd || 0), 0).toFixed(3)}). The cost guard still checks each job. Run?`, { title: 'Cloud cost', ok: 'Run' })) return;
    try {
      const r = await api.post(`${API}/chats/${this.page.chat.id}/run`, { project: store.project, plan: this.payload(), message: this.msg.id, steps });
      if (!r.ok) { this.v = r; this.plan = structuredClone(r.plan); this.render(); toast('Fix the marked steps first.', { kind: 'warn' }); return; }
      this.dirty = false;
      toast('Plan approved: jobs are queued.', { kind: 'ok' });
    } catch (e) { toastError(e); }
  }
}
