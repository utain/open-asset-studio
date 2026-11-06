// Job progress UI: a live job card (progress, stage, ETA, cancel / retry, error + fix) and a live job list.
import { api } from '../api.js';
import { bus, store } from '../store.js';
import { h, ic, btn, progressBar, fmtDur, fmtAgo, toastError, empty } from '../ui.js';
import { openAsset } from './drawer.js';
import { jobCost } from '../cost.js';

const TERMINAL = ['done', 'failed', 'cancelled'];

export function jobCard(job, { compact = false, onOutputs } = {}) {
  const el = h(`div.job.${job.status}`);
  const t0 = Date.now();
  let firstFrac = null;
  const render = j => {
    el.className = `job ${j.status}`;
    const p = j.progress || {};
    const running = j.status === 'running';
    let eta = '';
    if (running && p.frac > 0.03 && p.stage !== 'loading') {
      if (firstFrac == null) firstFrac = { f: p.frac, t: Date.now() };
      const df = p.frac - firstFrac.f, dt = (Date.now() - firstFrac.t) / 1000;
      if (df > 0.02 && dt > 1) eta = ` · ~${fmtDur((1 - p.frac) * dt / df)} left`;
    }
    const title = j.title || j.model_name || j.tool || j.task;
    el.replaceChildren(
      h('div.job-top', h(`span.st.${j.status}`, j.status), h('span.job-title.ellipsis.grow', title),
        !compact ? h('span.dim.nowrap', { style: { fontSize: '12px' } }, fmtAgo(j.created)) : null,
        !TERMINAL.includes(j.status) ? btn('', { size: 'sm', kind: 'ghost', icon: 'square', title: 'Cancel', onClick: () => api.cancel(j.id).catch(toastError) }) : null,
        j.status === 'failed' || j.status === 'cancelled' ? btn('', { size: 'sm', kind: 'ghost', icon: 'rotate-ccw', title: 'Retry', onClick: () => api.retry(j.id).catch(toastError) }) : null),
      !TERMINAL.includes(j.status) ? progressBarFor(j) : null,
      h('div.row', h('div.job-msg.ellipsis.grow', j.status === 'queued' ? (j.queue_position ? `Waiting for the GPU · position ${j.queue_position}` : 'Waiting') : (p.msg || p.stage || '')),
        jobCost(j),
        running ? h('span.dim.nowrap', { style: { fontSize: '11.5px' } }, `${Math.round((p.frac || 0) * 100)}%${eta}`) : null,
        j.status === 'done' ? h('span.dim.nowrap', { style: { fontSize: '11.5px' } }, fmtDur(j.timings?.run_s)) : null),
      running && p.interruptible === false ? h('div.fix', ic('info', 12), ' This stage cannot be interrupted; cancel takes effect after it.') : null,
      j.status === 'failed' && j.error ? h('div.err', j.error.message) : null,
      j.status === 'failed' && j.error?.fix ? h('div.fix', j.error.fix) : null,
      j.status === 'done' && j.outputs?.length && !compact ? h('div.row', btn('Open result', { size: 'sm', icon: 'eye', onClick: () => openAsset(j.outputs[0], { list: j.outputs }) })) : null);
    if (j.status === 'done') onOutputs?.(j.outputs);
  };
  render(job);
  const off = bus.on('job:' + job.id, j => { if (!el.isConnected && Date.now() - t0 > 2000) { off(); return; } render(j); });
  el.update = render;
  return el;
}
function progressBarFor(j) {
  const p = j.progress || {};
  return progressBar(p.frac || 0, j.status === 'queued' || p.stage === 'loading' || p.stage === 'memory' || !p.frac);
}

/** Live list of jobs (filtered), newest first. */
export class JobList {
  constructor({ statuses = null, limit = 30, project = null, compact = false, emptyText = 'No jobs yet.' } = {}) {
    this.statuses = statuses; this.limit = limit; this.project = project; this.compact = compact;
    this.el = h('div.col', { style: { gap: '8px' } });
    this.emptyEl = empty('list-checks', 'Queue is empty', emptyText);
    this.cards = new Map();
    this._off = bus.on('job', j => this._upsert(j));
    this.reload();
  }
  _match(j) { return (!this.statuses || this.statuses.includes(j.status)) && (!this.project || j.project === this.project); }
  async reload() {
    try {
      const r = await api.jobs({ limit: this.limit, status: this.statuses?.join(','), project: this.project || undefined });
      for (const j of r.jobs) store.jobs.set(j.id, j);
      this.el.replaceChildren(); this.cards.clear();
      for (const j of r.jobs) { const c = jobCard(j, { compact: this.compact }); this.cards.set(j.id, c); this.el.append(c); }
      this._empty();
    } catch (e) { this.el.replaceChildren(h('div.dim', e.message)); }
  }
  _upsert(j) {
    const c = this.cards.get(j.id);
    if (c && !this._match(j)) { setTimeout(() => { if (!this._match(store.jobs.get(j.id) || j)) { c.remove(); this.cards.delete(j.id); this._empty(); } }, 2500); c.update(j); return; }
    if (!c && this._match(j)) { const nc = jobCard(j, { compact: this.compact }); this.cards.set(j.id, nc); this.el.prepend(nc); while (this.cards.size > this.limit) { const last = [...this.cards.keys()].pop(); this.cards.get(last).remove(); this.cards.delete(last); } this._empty(); }
  }
  _empty() { if (!this.cards.size) this.el.replaceChildren(this.emptyEl); else this.emptyEl.remove(); }
  destroy() { this._off(); }
}
