// Sound studio: SFX / Ambience & loops / Music / Songs (+ score singing) / Voice (TTS, design, clone, conversion, voice reply, voice library) / Video to sound / Sheets.
// Built from the shared kit (ModelPicker, ParamForm, PromptBox, AssetSlot, jobCard, player) with sound-specific
// pieces: take cards (waveform, play/stop, loop, LUFS, peak, QA, rating), batches per prompt, post tools,
// CLAP / Whisper QA, voice library, SFX event lists and voice line sheets, engine-ready export.
import * as S from '../core/sdk.js';

const { h, ic, btn, toast, toastError, store, bus, api, prefs } = S;

// ------------------------------------------------------------------ static config
const VOICE_MODES = [
  { value: 'tts', label: 'Speak', icon: 'volume-2' },
  { value: 'voice_design', label: 'Design', icon: 'wand-sparkles' },
  { value: 'voice_clone', label: 'Clone', icon: 'mic' },
  { value: 'voice_convert', label: 'Convert', icon: 'arrow-left-right' },
  { value: 's2s', label: 'Reply', icon: 'message-circle' },
];
const SONG_MODES = [
  { value: 'song', label: 'Song', icon: 'drum' },
  { value: 'sing', label: 'Sing a score', icon: 'mic-vocal' },
];
const MEDIA_MODES = [
  { value: 'subtitles', label: 'Subtitles', icon: 'type' },
  { value: 'separate', label: 'Stems', icon: 'layers' },
  { value: 'dub', label: 'Dub', icon: 'film' },
];
// tabs with `modes` hold several tasks behind a mode switch (pref key per tab); the others have one task
const TABS = [
  { id: 'sfx', label: 'SFX', icon: 'zap', tasks: ['sfx'] },
  { id: 'ambience', label: 'Ambience & loops', short: 'Ambience', icon: 'trees', tasks: ['ambience'] },
  { id: 'music', label: 'Music', icon: 'music', tasks: ['music'] },
  { id: 'song', label: 'Song', icon: 'drum', tasks: ['song', 'sing'], modes: SONG_MODES, pref: 'sound.songMode' },
  { id: 'voice', label: 'Voice', icon: 'mic', tasks: ['tts', 'voice_design', 'voice_clone', 'voice_convert', 's2s'], modes: VOICE_MODES, pref: 'sound.voiceMode' },
  { id: 'v2a', label: 'Video to sound', short: 'Video', icon: 'clapperboard', tasks: ['v2a'] },
  { id: 'media', label: 'Subtitles, stems & dub', short: 'Media', icon: 'film', tasks: ['subtitles', 'separate', 'dub'], modes: MEDIA_MODES, pref: 'sound.mediaMode' },
  { id: 'sheets', label: 'Sheets', icon: 'list-checks', tasks: [] },
];
const SING_EXAMPLE = '你 C4 0.5\n好 E4 0.5\nSP rest 0.2\n歌 G4,E4 0.4,0.4';
// qa: 'clap' | 'asr' | null (no automatic check); noPrompt: no prompt box; param: the model param the prompt box fills;
// needsParam: hide the prompt box for models without that param; promptOptional: Generate does not need text
const TASK_UI = {
  sfx: { label: 'Describe the sound', ph: 'e.g. ดาบฟันโดนเกราะเหล็ก · heavy sword hits iron armor · UI confirm click', qa: 'clap', empty: 'One-shots, UI sounds, foley, creatures. Pick a few takes, rate the best, normalize, export.' },
  ambience: { label: 'Describe the place', ph: 'e.g. haunted forest at night, wind and distant owls, loopable', qa: 'clap', empty: 'Loopable beds for levels. Use “Seamless loop” on a take to crossfade its seam.' },
  music: { label: 'Describe the music', ph: 'e.g. boss battle, orchestral, driving drums, 150 BPM, D minor, loopable', qa: 'clap', empty: 'Instrumental tracks with BPM and key. Extend a take or make it loop. MusicGen Melody also follows a hummed or played tune.' },
  song: { label: 'Describe the song', ph: 'e.g. เพลงป๊อปอะคูสติก นักร้องหญิง เสียงนุ่ม เรื่องเช้าวันใหม่ในหมู่บ้าน', qa: 'clap', empty: 'Songs with lyrics and vocals (MiniMax-Music3, ACE-Step, DiffRhythm, YuE …). Use ✨ Improve to draft the caption and tagged lyrics.' },
  sing: { label: 'Score (one syllable per line)', ph: `<word> <note> <seconds>, one line per syllable, e.g.\n${SING_EXAMPLE}`, qa: null, param: 'score', mono: true, example: SING_EXAMPLE,
    empty: 'Write a score and a Mandarin singing voice (DiffSinger) sings it: one syllable per line as “<word> <note> <seconds>”, “SP rest 0.2” for a pause, “G4,E4 0.4,0.4” for a slur. ✨ Improve writes a score from an idea. Run a take through Voice → Convert to change the singer.' },
  tts: { label: 'Line to speak', ph: 'e.g. "Hold the line! Reinforcements are on the way." (angry knight)', qa: 'asr', param: 'text', empty: 'Preset voices with a delivery direction (emotion, tone, pace). ✨ Improve writes the direction.' },
  voice_design: { label: 'Describe the voice', ph: 'e.g. old gruff dwarf blacksmith, deep raspy voice, slow', qa: 'asr', param: 'design', empty: 'New voices from a description. Designed takes are saved as voices: use one as the reference to clone lines.' },
  voice_clone: { label: 'Line to speak', ph: 'The line, in the voice of the reference clip. Thai works with OmniVoice.', qa: 'asr', param: 'text', empty: 'Speak lines in a voice from your library (or any clip). Pick a reference, write the line, generate takes.' },
  voice_convert: { label: 'Voice conversion', noPrompt: true, qa: null,
    empty: 'Change who is speaking or singing in a recording. Pick the source clip and (for most models) a reference voice, then generate. Use only voices you have the right to use.' },
  s2s: { label: 'Message (optional)', ph: 'Type a message to answer, or leave empty and just send the clip', qa: null, param: 'text', needsParam: true, promptOptional: true,
    empty: 'Speak to the model and it answers in speech, with its text. Add a question clip and/or type a message, plus an optional instruction (persona, language, length). The reply text shows on each take.' },
  v2a: { label: 'Describe the sound (optional)', ph: 'e.g. footsteps on gravel, a wooden door creaks, distant thunder', qa: 'clap', promptOptional: true,
    empty: 'Drop in a video clip (mp4, mov, webm, mkv) and get matching foley and ambience. Each take also comes as a copy of the video with the new sound muxed in.' },
  subtitles: { label: 'Subtitles', noPrompt: true, qa: null,
    empty: 'Drop in a recording or a video and get SRT / VTT subtitles (plus word timings as JSON). Language is detected, or pick it; Whisper large-v3 can translate any language into English subtitles.' },
  separate: { label: 'Stems', noPrompt: true, qa: null,
    empty: 'Split a song, a mix or a video soundtrack into vocals and instrumental (and drums / bass / other with Demucs). Mel-Band RoFormer separates dense mixes best.' },
  dub: { label: 'Dub', noPrompt: true, qa: null,
    empty: 'Video or audio in, dubbed audio and MP4 out: the speech is separated from the background, transcribed, translated, spoken again in a clone of the original voice (fitted to the timing) and mixed back. Run “Transcript and translation only” first to edit the lines.' },
};
const EXTEND_TASKS = ['music', 'ambience', 'sfx'];
const DUB_FIELDS = ['target_language', 'source_language', 'tts_model', 'translate_backend', 'segments'];   // drawn by the Dub box, not the generic form
// hard-coded input slots, used when the model declares no `input_slots` for the task (docs/CONTRACTS.md section 1)
const REF_SLOT = { role: 'ref', label: 'Reference voice', types: ['voice', 'audio'], required: true, hint: 'A voice from the library or any clean speech clip (5-15 s). Its transcript fills in automatically.' };
const EXTEND_SLOT = { role: 'source', label: 'Clip to extend', types: ['audio'], required: false, hint: 'Set a clip to generate its continuation (Advanced → Extend by). The prompt should describe the clip.' };
const LANG_NAMES = { en: 'English', th: 'Thai', zh: 'Chinese', ja: 'Japanese', ko: 'Korean', fr: 'French', de: 'German', es: 'Spanish', it: 'Italian', ru: 'Russian', pt: 'Portuguese', vi: 'Vietnamese', id: 'Indonesian' };
const isVoiceTask = t => ['tts', 'voice_design', 'voice_clone'].includes(t);
const isSpeechTask = t => isVoiceTask(t) || ['voice_convert', 's2s'].includes(t);   // speech out (no 'quiet tail' hint, can be saved as a voice)
const qaOf = t => (t && TASK_UI[t] && 'qa' in TASK_UI[t] ? TASK_UI[t].qa : (isVoiceTask(t) ? 'asr' : 'clap'));
const isThaiText = s => /[฀-๿]/.test(s || '');
const fmtDb = v => (v == null ? '–' : `${v.toFixed(1)}`);

const CSS = `
.snd .studio-panel .sp-body { gap: 14px; }
.snd-tabs { display: flex; gap: 2px; padding: 0 10px; border-bottom: 1px solid var(--line-soft); overflow-x: auto; background: var(--bg-2); position: sticky; top: 0; z-index: 3; scrollbar-width: none; }
.snd-tabs button { border: 0; background: none; padding: 12px 7px 10px; color: var(--text-2); cursor: pointer; font-weight: 600; border-bottom: 2px solid transparent; white-space: nowrap; font-size: 13px; flex: 1 0 auto; }
.snd-tabs button.on { color: var(--text); border-bottom-color: var(--c-sound); }
.snd > .studio-panel, .snd > .studio-center { min-width: 0; }
.snd-mode { display: grid; grid-template-columns: repeat(auto-fit, minmax(72px, 1fr)); gap: 6px; }
.snd-mode button { border: 1px solid var(--line); background: var(--panel); border-radius: 10px; padding: 8px 6px; cursor: pointer; display: flex; flex-direction: column; align-items: center; gap: 4px; font-weight: 600; font-size: 12.5px; color: var(--text-2); }
.snd-mode button.on { border-color: var(--c-sound); color: var(--text); background: color-mix(in srgb, var(--c-sound) 10%, var(--panel)); }
.snd-mode button .ic { color: var(--c-sound); }
.snd-voicechips { display: flex; flex-wrap: wrap; gap: 6px; }
.snd-group { border: 1px solid var(--line-soft); border-radius: 14px; background: color-mix(in srgb, var(--panel) 55%, transparent); margin-bottom: 14px; min-width: 0; }
.snd-ghead { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-bottom: 1px solid var(--line-soft); flex-wrap: wrap; min-width: 0; }
.snd-ghead .gp { flex: 1 1 260px; min-width: 0; }
.snd-ghead .gp .t { font-weight: 600; font-size: 13px; overflow-wrap: anywhere; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.snd-ghead .gp .s { font-size: 11.5px; color: var(--text-3); display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 3px; }
.snd-ghead .ga { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
.snd-takes { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 10px; padding: 10px; }
.snd-card { border: 1px solid var(--line-soft); border-radius: 12px; background: var(--panel); padding: 9px 10px 8px; display: flex; flex-direction: column; gap: 6px; min-width: 0; position: relative; transition: border-color .15s; }
.snd-card:hover { border-color: var(--line); }
.snd-card.sel { border-color: var(--c-sound); box-shadow: 0 0 0 2px color-mix(in srgb, var(--c-sound) 22%, transparent); }
.snd-card.pending { opacity: .8; }
.snd-card.best { border-color: color-mix(in srgb, #ffc44d 55%, var(--line)); }
.snd-card .r1 { display: flex; align-items: center; gap: 8px; min-width: 0; }
.snd-card .r1 canvas.wave { height: 44px; }
.snd-card .nm { font-weight: 600; font-size: 12.5px; cursor: pointer; min-width: 0; }
.snd-card .nm:hover { color: var(--c-sound); }
.snd-card .r2 { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; font-size: 11.5px; color: var(--text-2); min-width: 0; }
.snd-card .r3 { display: flex; align-items: center; gap: 4px; min-width: 0; }
.snd-m { font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: 11px; padding: 1px 6px; border-radius: 6px; background: var(--panel-2); color: var(--text-2); white-space: nowrap; }
.snd-m.bad { color: var(--err); background: var(--err-soft); }
.snd-m.good { color: var(--ok); background: var(--ok-soft); }
.snd-m.warn { color: var(--warn); background: var(--warn-soft); }
.snd-loop { width: 30px; height: 30px; border-radius: 8px; border: 1px solid var(--line); background: none; color: var(--text-3); display: grid; place-items: center; cursor: pointer; flex: none; }
.snd-loop.on { color: var(--c-sound); border-color: var(--c-sound); background: color-mix(in srgb, var(--c-sound) 12%, transparent); }
.snd-sel { width: 22px; height: 22px; border-radius: 6px; border: 1.5px solid var(--line); background: none; display: grid; place-items: center; color: transparent; cursor: pointer; flex: none; }
.snd-card.sel .snd-sel { background: var(--c-sound); border-color: var(--c-sound); color: #04130d; }
.snd-post { font-size: 11px; color: var(--c-sound); display: flex; gap: 4px; align-items: center; min-width: 0; }
.snd-qa { font-size: 11.5px; color: var(--text-2); overflow-wrap: anywhere; }
.snd-qa.flag { color: var(--warn); }
.snd-lib { border: 1px solid var(--line-soft); border-radius: 14px; padding: 10px 12px; margin-bottom: 14px; background: var(--panel); }
.snd-lib .vl { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 8px; margin-top: 8px; }
.snd-voice { display: flex; gap: 8px; align-items: center; padding: 7px; border: 1px solid var(--line-soft); border-radius: 10px; min-width: 0; }
.snd-voice.on { border-color: var(--c-sound); background: color-mix(in srgb, var(--c-sound) 8%, transparent); }
.snd-voice .vt { flex: 1; min-width: 0; }
.snd-voice .vt b { display: block; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.snd-voice .vt span { display: block; font-size: 11px; color: var(--text-3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.snd-ops { display: flex; flex-direction: column; gap: 10px; }
.snd-op { border: 1px solid var(--line-soft); border-radius: 10px; padding: 9px 10px; display: flex; flex-direction: column; gap: 8px; }
.snd-op .oh { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.snd-op .ob { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
.snd-op .ob label { display: flex; flex-direction: column; gap: 3px; font-size: 11.5px; color: var(--text-2); }
.snd-op .ob input.input, .snd-op .ob select.select { width: 120px; }
.snd-sheet-rows { display: flex; flex-direction: column; gap: 8px; }
.snd-row { border: 1px solid var(--line-soft); border-radius: 12px; background: var(--panel); padding: 8px 10px; display: grid; gap: 8px; align-items: start; min-width: 0; }
.snd-row.sfx { grid-template-columns: 28px minmax(110px, 160px) minmax(0, 1fr) 70px 64px auto; }
.snd-row.lines { grid-template-columns: 28px minmax(90px, 120px) minmax(0, 1fr) minmax(90px, 140px) 64px 64px auto; }
.snd-row .rn { color: var(--text-3); font-family: var(--mono); font-size: 11.5px; padding-top: 8px; }
.snd-row .input, .snd-row .textarea, .snd-row .select { width: 100%; min-width: 0; }
.snd-row textarea.textarea { min-height: 36px; resize: vertical; }
.snd-row .acts { display: flex; gap: 2px; justify-content: flex-end; }
.snd-row .takes { grid-column: 2 / -1; display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
.snd-row .stat { grid-column: 2 / -1; font-size: 11.5px; color: var(--text-3); display: flex; gap: 8px; align-items: center; min-width: 0; }
.snd-mini { display: flex; align-items: center; gap: 6px; border: 1px solid var(--line-soft); border-radius: 10px; padding: 3px 6px 3px 3px; min-width: 0; max-width: 100%; }
.snd-mini.pick { border-color: color-mix(in srgb, #ffc44d 60%, var(--line)); }
.snd-mini .play-btn { width: 28px; height: 28px; }
.snd-mini canvas.wave { width: 90px; height: 26px; }
.snd-cast { display: flex; flex-direction: column; gap: 8px; }
.snd-cast .cr { border: 1px solid var(--line-soft); border-radius: 10px; padding: 8px; display: flex; flex-direction: column; gap: 6px; }
.snd-head-sel { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.snd-filter { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.snd-tip { font-size: 12px; color: var(--text-3); }
.snd-err { color: var(--err); font-size: 12px; padding: 0 12px 10px; overflow-wrap: anywhere; }
.snd-reply { display: flex; gap: 6px; align-items: flex-start; font-size: 12px; color: var(--text-2); background: var(--panel-2); border-radius: 8px; padding: 6px 8px; cursor: pointer; min-width: 0; }
.snd-reply .ri { color: var(--c-sound); flex: none; padding-top: 1px; }
.snd-reply-t { flex: 1; min-width: 0; overflow-wrap: anywhere; white-space: pre-wrap; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.snd-reply.open .snd-reply-t { -webkit-line-clamp: unset; display: block; }
.snd-reply-dl { color: var(--text-3); flex: none; }
.snd-card .video-box { margin-top: 2px; }
.snd-pre { font-family: var(--mono); font-size: 11.5px; line-height: 1.45; color: var(--text-2); background: var(--panel-2); border-radius: 8px; padding: 6px 8px; margin: 0; max-height: 96px; overflow: hidden; white-space: pre-wrap; overflow-wrap: anywhere; }
.snd-dub { display: flex; flex-direction: column; gap: 10px; }
.snd-dub .select { width: 100%; }
.snd-lines { display: flex; flex-direction: column; gap: 6px; max-height: 56vh; overflow: auto; padding-right: 2px; }
.snd-ln { display: grid; grid-template-columns: 24px 74px minmax(0, 1fr) minmax(0, 1.2fr); gap: 8px; align-items: start; border: 1px solid var(--line-soft); border-radius: 10px; padding: 6px 8px; }
.snd-ln.skip { opacity: .5; }
.snd-ln .tm { font-family: var(--mono); font-size: 11px; color: var(--text-3); padding-top: 7px; }
.snd-ln .src { font-size: 12.5px; color: var(--text-2); padding-top: 5px; overflow-wrap: anywhere; }
.snd-ln textarea.textarea { min-height: 34px; width: 100%; resize: vertical; }
@media (max-width: 900px) {
  .snd-row.sfx, .snd-row.lines { grid-template-columns: 28px minmax(0, 1fr) minmax(0, 1fr); }
  .snd-row .rn { grid-row: span 3; }
  .snd-row .f-prompt, .snd-row .f-line { grid-column: 2 / -1; }
  .snd-row .acts { grid-column: 2 / -1; justify-content: flex-start; flex-wrap: wrap; }
}
@media (max-width: 760px) {
  .snd-ln { grid-template-columns: 24px minmax(0, 1fr); }
  .snd-ln .src, .snd-ln textarea.textarea { grid-column: 2; }
  .snd-takes { grid-template-columns: 1fr; }
  .snd-tabs button { padding: 11px 6px 9px; }
}
`;

// ------------------------------------------------------------------ small helpers
function paramOf(m, name) { return (m?.params || []).find(p => p.name === name); }
/** Input slots for a model + task: the model's declared `input_slots` for the task, else the hard-coded defaults. */
function slotDefsFor(task, m) {
  const decl = (Array.isArray(m?.input_slots) ? m.input_slots : []).filter(d => !d.tasks || d.tasks.includes(task));
  if (decl.length) return decl;
  const out = [];
  if (task === 'voice_clone') out.push(REF_SLOT);
  if (m?.features?.extend && EXTEND_TASKS.includes(task)) out.push(EXTEND_SLOT);
  return out;
}
function showPromptFor(task, m) {
  const ui = TASK_UI[task] || {};
  if (ui.noPrompt) return false;
  if (ui.needsParam && !paramOf(m, ui.param)) return false;
  return true;
}
function promptParamFor(task, m) {
  const ui = TASK_UI[task] || {};
  if (ui.param === 'design') return paramOf(m, 'instruction') ? 'instruction' : 'instruct';
  return ui.param || 'prompt';
}
function langFor(m, code) {
  const p = paramOf(m, 'language'); if (!p) return undefined;
  const ch = (p.choices || []).map(c => (typeof c === 'object' ? c.value : c));
  if (!code) return p.default;
  const want = [code, LANG_NAMES[code]].filter(Boolean).map(s => String(s).toLowerCase());
  return ch.find(c => want.includes(String(c).toLowerCase())) ?? (ch.find(c => /^auto$/i.test(c)) ?? p.default);
}
function ready(id) { return store.modelsById.get(id)?.status?.status === 'ready'; }
function thaiOk(m) { const t = m?.features?.thai; return t === true || t === 'weak'; }
function voiceRefText(a) { return a?.x_voice?.ref_text || a?.params?.text || a?.x_qa?.asr?.transcript || ''; }
function scoreOf(a) { return a.x_qa?.clap?.clap ?? a.x_qa?.asr?.recall ?? null; }
function parseCSV(text) {
  const rows = []; let row = [], cur = '', q = false;
  const sep = (text.split('\n')[0].match(/\t/g) || []).length > (text.split('\n')[0].match(/,/g) || []).length ? '\t' : ',';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; continue; }
    if (c === '"') q = true; else if (c === sep) { row.push(cur.trim()); cur = ''; } else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cur.trim()); rows.push(row); row = []; cur = ''; } else cur += c;
  }
  if (cur || row.length) { row.push(cur.trim()); rows.push(row); }
  return rows.filter(r => r.some(x => x));
}
const slug = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40);
const uid = () => Math.random().toString(36).slice(2, 9);
/** <textarea> with a value (h() would set the value attribute, which textareas ignore). */
function textarea(sel, attrs, value) { const t = h(sel, attrs); t.value = value ?? ''; return t; }

/** ParamForm renderers for textarea / lyrics (the core form sets `value` as an attribute, which a textarea ignores). */
const LYRIC_TAGS = ['[intro]', '[verse]', '[pre-chorus]', '[chorus]', '[bridge]', '[instrumental]', '[outro]'];
function textRenderer(p, v, set, form) {
  const lyr = p.type === 'lyrics';
  const t = textarea('textarea.textarea', { rows: lyr ? 9 : 3, placeholder: p.placeholder || (lyr ? '[verse]\nfirst line\n[chorus]\n…' : ''), style: lyr ? { fontFamily: 'var(--mono)', fontSize: '12.5px' } : null,
    on: { input: e => { form.vals[p.name] = e.target.value; form.mem[p.name] = e.target.value; }, change: e => set(e.target.value) } }, v);
  if (!lyr) return t;
  const chips = h('div.chips', ...LYRIC_TAGS.map(tag => h('button.chip', { type: 'button', on: { click: () => {
    const pos = t.selectionStart ?? t.value.length;
    t.value = t.value.slice(0, pos) + (pos && t.value[pos - 1] !== '\n' ? '\n' : '') + tag + '\n' + t.value.slice(pos); set(t.value); t.focus();
  } } }, tag)));
  return h('div.col', { style: { gap: '6px' } }, t, chips);
}
const RENDERERS = { textarea: textRenderer, lyrics: textRenderer };

// ------------------------------------------------------------------ take card
const replyCache = new Map();
/** The spoken reply of a Voice-reply take: worker.reply_text, else the attached take_N.txt (fetched once). */
function replyBlock(x) {
  const att = (x.files || []).find(f => f.role !== 'main' && /\.txt$/i.test(f.name));
  const known = x.worker?.reply_text;
  if (!known && !att) return null;
  const body = h('div.snd-reply-t', known || replyCache.get(`${x.id}/${att.name}`) || 'Loading the reply text…');
  if (!known && !replyCache.has(`${x.id}/${att.name}`)) {
    fetch(S.fileUrl(x, att.name)).then(r => r.text()).then(t => { replyCache.set(`${x.id}/${att.name}`, t.trim()); body.textContent = t.trim() || '(empty reply)'; })
      .catch(() => { body.textContent = '(the reply text could not be loaded)'; });
  }
  return h('div.snd-reply', { title: 'Reply text. Click to expand.', on: { click: e => e.currentTarget.classList.toggle('open') } }, h('span.ri', ic('message-circle', 13)), body,
    att ? h('a.snd-reply-dl', { href: S.fileUrl(x, att.name, true), title: 'Download the reply text', on: { click: e => e.stopPropagation() } }, ic('download', 12)) : null);
}
/** A take: play/stop, loop, waveform, duration, LUFS, peak, QA, rating, select, menu. */
function soundCard(a, ctx) {
  const el = h('div.snd-card', { dataset: { id: a.id } });
  let loop = false;
  const render = x => {
    a = x; el._asset = x;
    const I = x.info || {};
    const playB = h('button.play-btn', { type: 'button', title: 'Play / stop (space)', on: { click: e => { e.stopPropagation(); if (S.player.current === x.id && S.player.playing) S.player.stop(); else S.player.play(x, { loop, at: S.player.current === x.id ? null : 0 }); } } });
    const drawPlay = () => { const on = S.player.current === x.id && S.player.playing; playB.classList.toggle('on', on); playB.replaceChildren(ic(on ? 'square' : 'play', 16)); };
    drawPlay();
    el._drawPlay = drawPlay;
    const loopB = h(`button.snd-loop${loop ? '.on' : ''}`, { type: 'button', title: 'Loop playback', on: { click: e => { e.stopPropagation(); loop = !loop; loopB.classList.toggle('on', loop); if (S.player.current === x.id) S.player.setLoop(loop); } } }, ic('repeat', 15));
    const selB = h('button.snd-sel', { type: 'button', title: 'Select', on: { click: e => { e.stopPropagation(); ctx.toggleSelect(x.id); } } }, ic('check', 13));
    const q = x.x_qa || {};
    const qa = [];
    if (q.clap?.clap != null) qa.push(h(`span.snd-m.${q.clap.flag ? 'warn' : 'good'}`, { title: `CLAP prompt match ${q.clap.clap} (silence ${q.clap.clap_silence}, speech ${q.clap.clap_speech})` }, `CLAP ${q.clap.clap.toFixed(2)}`));
    if (q.asr?.recall != null) qa.push(h(`span.snd-m.${q.asr.flag ? 'bad' : 'good'}`, { title: `Whisper: “${q.asr.transcript}”` }, `ASR ${Math.round(q.asr.recall * 100)}%`));
    const nc = x.license && x.license.commercial === false;
    const post = x.x_post?.labels?.length ? h('div.snd-post', ic('git-fork', 12), h('span.ellipsis', x.x_post.labels.join(' · '))) : null;
    el.replaceChildren(
      h('div.r1', playB, h('div.grow', { style: { minWidth: 0 } }, S.waveCanvas(x, { height: 44 })), loopB),
      h('div.r3', selB, h('span.nm.ellipsis.grow', { title: `${x.name}\nClick for details`, on: { click: () => ctx.open(x) } }, x.name || x.id),
        S.stars(x.rating || 0, v => api.patchAsset(x.id, { rating: v }).catch(toastError), 13),
        btn('', { size: 'sm', kind: 'ghost', icon: 'more-vertical', title: 'Actions', onClick: e => ctx.menu(e.currentTarget, x) })),
      h('div.r2', h('span.snd-m', { title: 'Duration' }, S.fmtClock(I.duration || 0)),
        h(`span.snd-m${I.lufs != null && (I.lufs > -8 || I.lufs < -35) ? '.warn' : ''}`, { title: 'Integrated loudness (LUFS)' }, `${fmtDb(I.lufs)} LUFS`),
        h(`span.snd-m${I.clipped ? '.bad' : ''}`, { title: 'Peak (dBFS)' }, `pk ${fmtDb(I.peak_db)}`),
        I.silence_fraction > 0.5 && !isSpeechTask(x.task) && !['separate', 'dub'].includes(x.task) && x.type !== 'voice' && !x.x_post?.ops?.trim ? h('span.snd-m.warn', { title: 'Mostly near-silence (tail / padding): ⋮ → Audio tools → Trim silence' }, `${Math.round(I.silence_fraction * 100)}% quiet`) : null,
        x.type === 'voice' ? h('span.badge.info', ic('mic', 11), 'voice') : null,
        nc ? S.licenseBadge(x.license) : null, ...qa),
      post,
      replyBlock(x),
      S.videoAttachments(x)[0] ? S.videoBox(x, { name: S.videoAttachments(x)[0].name, compact: true }) : null,
      q.asr?.transcript && q.asr.flag ? h('div.snd-qa.flag', ic('triangle-alert', 12), ` heard: “${q.asr.transcript}”`) : null);
    el.classList.toggle('sel', ctx.isSelected(x.id));
    el.classList.toggle('best', ctx.isBest?.(x.id) || false);
  };
  render(a);
  el.update = render;
  el.addEventListener('dblclick', () => ctx.open(a));
  const off = bus.on('player', () => { if (!el.isConnected) { off(); return; } el._drawPlay?.(); });
  return el;
}

// ------------------------------------------------------------------ subtitles / stems / dub: cards for text and video outputs
const textCache = new Map();
function fetchText(x, name) {
  const k = `${x.id}/${name}`;
  if (textCache.has(k)) return Promise.resolve(textCache.get(k));
  return fetch(S.fileUrl(x, name)).then(r => r.text()).then(t => { textCache.set(k, t); return t; });
}
/** A subtitle / dub-plan text asset or a dubbed video: preview, downloads, and "Edit lines & render" for a draft. */
function mediaCard(a, ctx) {
  const el = h('div.snd-card.snd-media', { dataset: { id: a.id } });
  const render = x => {
    a = x; el._asset = x;
    const draft = !!x.x_dub && (x.tags || []).includes('draft');
    const selB = h('button.snd-sel', { type: 'button', title: 'Select', on: { click: e => { e.stopPropagation(); ctx.toggleSelect(x.id); } } }, ic('check', 13));
    const downloads = (x.files || []).filter(f => /\.(srt|vtt|json|txt)$/i.test(f.name)).map(f =>
      h('a.btn.btn-sm.btn-ghost', { href: S.fileUrl(x, f.name, true), title: `Download ${f.name}`, on: { click: e => e.stopPropagation() } }, ic('download', 12), ' ', f.name.split('.').pop().toUpperCase()));
    const pre = x.type === 'text' ? h('pre.snd-pre', 'Loading…') : null;
    if (pre) {
      fetchText(x, x.main).then(t => {
        if (draft) { const n = (x.x_dub?.segments || []).length; pre.textContent = (x.x_dub?.segments || []).slice(0, 4).map(g => `${S.fmtClock(g.start)}  ${g.translation || '(not translated yet)'}`).join('\n') + (n > 4 ? `\n… ${n} lines` : ''); return; }
        pre.textContent = t.replace(/\r/g, '').split(/\n\n+/).slice(0, 4).map(b => b.split('\n').filter(l => !/^\d+$/.test(l.trim()) && !/-->/.test(l)).join('  ')).join('\n') || '(empty)';
      }).catch(() => { pre.textContent = '(preview unavailable)'; });
    }
    const lic = x.license && x.license.commercial === false;
    el.replaceChildren(
      h('div.r3', selB, h('span.nm.ellipsis.grow', { title: `${x.name}\nClick for details`, on: { click: () => ctx.open(x) } }, x.name || x.id),
        btn('', { size: 'sm', kind: 'ghost', icon: 'more-vertical', title: 'Actions', onClick: e => ctx.menu(e.currentTarget, x) })),
      h('div.r2', h(`span.badge.${draft ? 'warn' : 'info'}`, ic(x.type === 'video' ? 'film' : 'type', 11), draft ? 'dub draft' : x.type === 'video' ? 'dubbed video' : 'text'),
        x.info?.duration ? h('span.snd-m', S.fmtClock(x.info.duration)) : null, lic ? S.licenseBadge(x.license) : null,
        x.x_dub?.warnings?.length ? h('span.snd-m.warn', { title: x.x_dub.warnings.join('\n') }, `${x.x_dub.warnings.length} warning(s)`) : null),
      pre,
      x.type === 'video' ? S.videoBox(x, { compact: true }) : null,
      h('div.r2', ...downloads, draft ? btn('Edit lines & render', { size: 'sm', kind: 'primary', icon: 'pencil', onClick: () => ctx.editDraft?.(x) }) : null));
    el.classList.toggle('sel', ctx.isSelected(x.id));
  };
  render(a);
  el.update = render;
  el.addEventListener('dblclick', () => ctx.open(a));
  return el;
}
const isMediaOut = a => a.type === 'text' || a.type === 'video';

/** Modal: edit the translated lines of a dub draft, translate the empty ones, render the dub. */
async function dubDraftModal(a, st) {
  const plan = a.x_dub || {};
  let lines = (plan.segments || []).map(g => ({ ...g }));
  if (!lines.length) return toast('This draft has no lines.', { kind: 'warn' });
  const tgt = plan.target_language || 'en';
  const list = h('div.snd-lines');
  const status = h('div.snd-tip');
  const draw = () => list.replaceChildren(...lines.map((g, i) => {
    const tr = textarea('textarea.textarea', { rows: 2, placeholder: 'Translation', on: { input: e => { g.translation = e.target.value; } } }, g.translation || '');
    const sk = h('input', { type: 'checkbox', checked: !g.skip, title: 'Speak this line', on: { change: e => { g.skip = !e.target.checked; row.classList.toggle('skip', g.skip); } } });
    const row = h(`div.snd-ln${g.skip ? '.skip' : ''}`, sk, h('div.tm', `${S.fmtClock(g.start)}–${S.fmtClock(g.end)}`), h('div.src', g.text), tr);
    return row;
  }));
  draw();
  const back = h('select.select', h('option', { value: '' }, 'Translation backend: default'));
  api.get('/api/x/media_audio/dub/options').then(o => (o.translate_backends || []).forEach(b => back.append(h('option', { value: b.id, disabled: b.available === false }, `${b.label || b.id}${b.available === false ? ' (unavailable)' : ''}${b.cloud ? ' · cloud' : ''}`)))).catch(() => {});
  const trBtn = btn('Translate empty lines', { size: 'sm', icon: 'type', onClick: async () => {
    trBtn.disabled = true; status.textContent = 'Translating…';
    try {
      const r = await api.post('/api/x/media_audio/dub/translate', { segments: lines, source_language: plan.source_language, target_language: tgt, backend: back.value || null, only_missing: true });
      lines = r.segments.map((g, i) => ({ ...lines[i], ...g })); draw(); status.textContent = `Translated with ${r.backend || 'the backend'}.`;
    } catch (e) { status.textContent = ''; toastError(e, 'Translation failed'); }
    trBtn.disabled = false;
  } });
  const md = S.modal({ title: `Dub lines → ${tgt}`, icon: 'film', width: 860, body: h('div.col', { style: { gap: '10px' } },
    h('div.snd-tip', 'Edit the translation of each line (untick a line to leave it out). Keep lines about as long as the original: a longer line is sped up (up to the limit in Advanced) and then reported as overlapping.'),
    h('div.row.wrap', back, trBtn, status), list),
  foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn('Render dub', { kind: 'primary', icon: 'sparkles', onClick: async () => {
    const todo = lines.filter(g => !g.skip && !String(g.translation || '').trim());
    if (todo.length) return toast(`${todo.length} line(s) have no translation: translate them or untick them.`, { kind: 'warn' });
    const inputs = [{ asset: plan.media, role: 'media' }];
    if (plan.stems?.vocals) inputs.push({ asset: plan.stems.vocals, role: 'vocals' });
    if (plan.stems?.instrumental) inputs.push({ asset: plan.stems.instrumental, role: 'bg' });
    try {
      const params = { ...st.formValues(), target_language: tgt, source_language: plan.source_language || 'auto', tts_model: plan.tts_model || '', stage: 'all',
        separate_model: plan.separate_model || '', segments: JSON.stringify(lines) };
      const spec = { task: 'dub', model: 'dub-pipeline', params, inputs, project: store.project, title: `Dub → ${tgt}: ${a.name}`.slice(0, 70), meta: { studio: 'sound' } };
      md.close();
      await st.generate(spec);
    } catch (e) { toastError(e); }
  } })] });
}

function pendingTake(i) {
  return h('div.snd-card.pending', h('div.r1', h('div.play-btn', ic('loader', 16, 'spin')), h('div.grow.skeleton', { style: { height: '44px', borderRadius: '8px', background: 'var(--panel-2)' } })),
    h('div.r2', h('span.dim', `take ${i + 1}…`)));
}

/** Compact take for sheet rows: play, tiny wave, stars. */
function miniTake(a, onOpen) {
  const el = h(`div.snd-mini${(a.rating || 0) > 0 || a.favorite ? '.pick' : ''}`, { title: `${a.name} · ${S.fmtClock(a.info?.duration || 0)} · ${fmtDb(a.info?.lufs)} LUFS` },
    S.playButton(a, 28), S.waveCanvas(a, { height: 26 }),
    a.x_qa?.asr?.recall != null ? h(`span.snd-m.${a.x_qa.asr.flag ? 'bad' : 'good'}`, `${Math.round(a.x_qa.asr.recall * 100)}%`) : null,
    a.x_qa?.clap?.clap != null ? h('span.snd-m', a.x_qa.clap.clap.toFixed(2)) : null,
    S.stars(a.rating || 0, v => api.patchAsset(a.id, { rating: v }).catch(toastError), 12),
    btn('', { size: 'sm', kind: 'ghost', icon: 'panel-right', title: 'Details', onClick: () => onOpen(a) }));
  return el;
}

// ------------------------------------------------------------------ modals: post tools / export / save voice
const OPS = [
  { id: 'trim', label: 'Trim silence', icon: 'scissors', fields: [{ k: 'threshold_db', label: 'Threshold dB', v: -50 }, { k: 'pad_ms', label: 'Keep (ms)', v: 30 }] },
  { id: 'loop', label: 'Seamless loop (crossfade the seam)', icon: 'repeat', fields: [{ k: 'xfade_ms', label: 'Crossfade (ms)', v: 500 }] },
  { id: 'fade', label: 'Fade in / out', icon: 'audio-waveform', fields: [{ k: 'in_ms', label: 'In (ms)', v: 10 }, { k: 'out_ms', label: 'Out (ms)', v: 200 }] },
  { id: 'normalize', label: 'Normalize loudness', icon: 'gauge', fields: [{ k: 'lufs', label: 'Target LUFS', v: -16 }, { k: 'ceiling_db', label: 'Peak ceiling dB', v: -1 }] },
];
function postToolsModal(ids, { preset = null, title = 'Audio tools' } = {}) {
  if (!ids.length) return toast('Select takes first.', { kind: 'warn' });
  const saved = prefs.get('sound.post', { trim: true, normalize: true });
  const on = { ...saved, ...(preset ? Object.fromEntries(OPS.map(o => [o.id, !!preset[o.id]])) : {}) };
  const vals = prefs.get('sound.postVals', {});
  const box = h('div.snd-ops');
  const inputs = {};
  for (const o of OPS) {
    const fields = h('div.ob', ...o.fields.map(f => {
      const i = h('input.input.input-sm', { type: 'number', value: vals[`${o.id}.${f.k}`] ?? f.v, step: 'any' });
      inputs[`${o.id}.${f.k}`] = i;
      return h('label', f.label, i);
    }));
    const tg = S.toggle(on[o.id], v => { on[o.id] = v; fields.style.display = v ? '' : 'none'; if (o.id === 'loop' && v) toast('Loop replaces the fade: the crossfade already makes the seam seamless.', { kind: 'info' }); }, o.label);
    fields.style.display = on[o.id] ? '' : 'none';
    box.append(h('div.snd-op', h('div.oh', ic(o.icon, 16), h('span.grow', o.label), tg), fields));
  }
  const sr = h('select.select', ...[['', 'keep'], ['22050', '22.05 kHz'], ['24000', '24 kHz'], ['32000', '32 kHz'], ['44100', '44.1 kHz'], ['48000', '48 kHz']].map(([v, l]) => h('option', { value: v, selected: String(vals.sr || '') === v }, l)));
  const chn = h('select.select', ...[['', 'keep'], ['mono', 'mono'], ['stereo', 'stereo']].map(([v, l]) => h('option', { value: v, selected: (vals.ch || '') === v }, l)));
  const fmt = h('select.select', ...['wav', 'ogg', 'mp3', 'flac'].map(v => h('option', { value: v, selected: (vals.fmt || 'wav') === v }, v.toUpperCase())));
  box.append(h('div.snd-op', h('div.oh', ic('settings-2', 16), h('span.grow', 'Format')), h('div.ob', h('label', 'Sample rate', sr), h('label', 'Channels', chn), h('label', 'File', fmt))));
  const go = btn(`Process ${ids.length} take${ids.length > 1 ? 's' : ''}`, { kind: 'primary', icon: 'wand', onClick: async () => {
    const ops = {};
    const nv = { sr: sr.value, ch: chn.value, fmt: fmt.value };
    for (const o of OPS) {
      for (const f of o.fields) nv[`${o.id}.${f.k}`] = Number(inputs[`${o.id}.${f.k}`].value);
      if (on[o.id]) ops[o.id] = Object.fromEntries(o.fields.map(f => [f.k, Number(inputs[`${o.id}.${f.k}`].value)]));
    }
    if (sr.value) ops.sample_rate = Number(sr.value);
    if (chn.value) ops.channels = chn.value;
    if (!Object.keys(ops).length && fmt.value === 'wav') return toast('Turn on at least one tool.', { kind: 'warn' });
    prefs.set('sound.post', Object.fromEntries(OPS.map(o => [o.id, !!on[o.id]]))); prefs.set('sound.postVals', nv);
    try {
      const j = await api.submit({ tool: 'audio.process', params: { ops, format: fmt.value }, inputs: ids.map(id => ({ asset: id, role: 'input' })), project: store.project,
        title: `Post: ${Object.keys(ops).join(' + ') || fmt.value} (${ids.length})`, meta: { studio: 'sound' } });
      md.close();
      toast(`Processing ${ids.length} take(s). New takes appear next to the originals, with lineage.`, { kind: 'ok', title: 'Audio tools' });
      S.waitJob(j.id).then(jj => { if (jj.status === 'failed') toast(jj.error?.message || 'failed', { kind: 'err', title: 'Audio tools failed' }); });
    } catch (e) { toastError(e, 'Cannot process'); }
  } });
  const md = S.modal({ title, icon: 'wand', width: 520, body: h('div.col', { style: { gap: '10px' } }, h('div.snd-tip', 'Runs on the CPU next to GPU jobs. Each take becomes a new derived take (the original is kept). Order: trim → loop → fade → resample → channels → normalize.'), box),
    foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), go] });
}

async function exportModal(ids, { title = 'Export for the game engine', folder = '' } = {}) {
  if (!ids.length) return toast('Nothing to export.', { kind: 'warn' });
  const v = prefs.get('sound.export', { format: 'ogg', sr: '', ch: '', lufs: '' });
  const fmt = S.seg(['ogg', 'wav', 'mp3'].map(x => ({ value: x, label: x.toUpperCase() })), v.format, x => { v.format = x; });
  const sr = h('select.select', ...[['', 'keep'], ['22050', '22.05 kHz'], ['44100', '44.1 kHz'], ['48000', '48 kHz']].map(([a, l]) => h('option', { value: a, selected: v.sr === a }, l)));
  const ch = h('select.select', ...[['', 'keep'], ['mono', 'mono'], ['stereo', 'stereo']].map(([a, l]) => h('option', { value: a, selected: v.ch === a }, l)));
  const lufs = h('select.select', ...[['', 'off'], ['-14', '-14 LUFS (music, loud)'], ['-16', '-16 LUFS (SFX / game mix)'], ['-18', '-18 LUFS'], ['-20', '-20 LUFS (dialogue)'], ['-23', '-23 LUFS (broadcast)']].map(([a, l]) => h('option', { value: a, selected: v.lufs === a }, l)));
  const fold = h('input.input', { value: folder || `${store.project}_sound`, placeholder: 'folder name' });
  const res = h('div');
  const go = btn(`Export ${ids.length}`, { kind: 'primary', icon: 'package', onClick: async () => {
    go.disabled = true; res.replaceChildren(h('div.dim', ic('loader', 14, 'spin'), ' Converting…'));
    prefs.set('sound.export', { ...v, sr: sr.value, ch: ch.value, lufs: lufs.value });
    try {
      const r = await api.post('/api/x/sound/export', { ids, format: v.format, folder: fold.value, sample_rate: sr.value ? Number(sr.value) : null,
        channels: ch.value || null, normalize_lufs: lufs.value ? Number(lufs.value) : null });
      res.replaceChildren(h('div.snd-tip', `${r.count} files in ${r.folder}/ · ${S.fmtBytes(r.bytes)}`), r.warning ? h('div', { style: { color: 'var(--err)', fontSize: '12.5px' } }, ic('ban', 13), ' ', r.warning) : null,
        h('pre.mono', { style: { fontSize: '11.5px', maxHeight: '160px', overflow: 'auto', whiteSpace: 'pre-wrap' } }, r.files.join('\n') + (r.count > r.files.length ? `\n… +${r.count - r.files.length}` : '')));
      const a = document.createElement('a'); a.href = r.url; a.download = r.name; document.body.append(a); a.click(); a.remove();
    } catch (e) { res.replaceChildren(h('div', { style: { color: 'var(--err)' } }, e.message)); }
    go.disabled = false;
  } });
  const md = S.modal({ title, icon: 'package', width: 520, body: h('div.col', { style: { gap: '12px' } },
    h('div.snd-tip', 'A zip with one folder per category: sfx/, ambience/, music/, voice/<character>/, voices/ — files named <event>_<take>.ext (best rated first) plus manifest.json with prompts, models, licences, loudness and QA.'),
    h('div.field', h('span.field-label', 'Format'), fmt),
    h('div.row.wrap', h('label.field.grow', h('span.field-label', 'Sample rate'), sr), h('label.field.grow', h('span.field-label', 'Channels'), ch)),
    h('label.field', h('span.field-label', 'Normalize'), lufs), h('label.field', h('span.field-label', 'Folder'), fold), res),
    foot: [btn('Close', { kind: 'ghost', onClick: () => md.close() }), go] });
}

async function saveVoiceModal(a) {
  const name = h('input.input', { value: a.x_voice?.character || (a.type === 'voice' ? a.name : '') || '', placeholder: 'e.g. Old blacksmith' });
  const txt = textarea('textarea.textarea', { rows: 3, placeholder: 'Exact words spoken in the clip (improves cloning)' }, voiceRefText(a));
  const lang = h('select.select', ...Object.entries({ '': 'auto', ...LANG_NAMES }).map(([k, l]) => h('option', { value: k, selected: (a.x_voice?.language || a.params?.language || '') === k }, l)));
  const tr = btn('Transcribe with Whisper', { size: 'sm', icon: 'type', onClick: async () => {
    tr.disabled = true;
    try {
      const j = await api.submit({ task: 'qa_asr', model: 'whisper-v3-turbo', inputs: [{ asset: a.id }], params: {}, project: store.project, title: `Transcribe ${a.name}`, meta: { studio: 'sound' } });
      const jj = await S.waitJob(j.id);
      if (jj.status !== 'done') throw new Error(jj.error?.message || 'Transcription failed');
      const m = await api.asset(a.id); txt.value = m.x_qa?.asr?.transcript || '';
    } catch (e) { toastError(e); }
    tr.disabled = false;
  } });
  const md = S.modal({ title: 'Save as voice', icon: 'mic', width: 480, body: h('div.col', { style: { gap: '12px' } },
    h('div.snd-tip', 'Adds a cleaned copy (trimmed, mono, -20 LUFS, up to 30 s) to the voice library. Clone models use it as the reference.'),
    h('label.field', h('span.field-label', 'Voice / character name'), name),
    h('label.field', h('span.field-label', 'Transcript of the clip'), txt), h('div.row', tr),
    h('label.field', h('span.field-label', 'Language'), lang)),
  foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn('Save voice', { kind: 'primary', icon: 'check', onClick: async () => {
    try {
      const j = await api.submit({ tool: 'audio.make_voice', params: { name: name.value, ref_text: txt.value, language: lang.value || null, character: name.value || null }, inputs: [{ asset: a.id }], project: store.project, title: `Voice: ${name.value || a.name}`, meta: { studio: 'sound' } });
      md.close();
      const jj = await S.waitJob(j.id);
      if (jj.status === 'done') toast(`“${name.value || 'Voice'}” is in the voice library.`, { kind: 'ok', title: 'Voice saved', action: { label: 'Clone with it', onClick: () => S.sendTo('sound', jj.outputs[0], { action: 'voice_ref' }) } });
      else toast(jj.error?.message || 'failed', { kind: 'err' });
    } catch (e) { toastError(e); }
  } })] });
}

// ------------------------------------------------------------------ QA
async function runQA(assets, { kind = null, quiet = false } = {}) {
  const list = assets.filter(a => a.type === 'audio' || a.type === 'voice');
  if (!list.length) return null;
  const voice = kind ? kind === 'asr' : qaOf(list[0].task) === 'asr';
  if (!voice) {   // CLAP compares audio with the prompt: takes made without one (video to sound, no description) cannot be scored
    const keep = list.filter(a => a.final_prompt || a.params?.prompt || a.prompt);
    if (!keep.length) { if (!quiet) toast('These takes have no prompt to compare with.', { kind: 'warn' }); return null; }
    list.splice(0, list.length, ...keep);
  }
  const spec = voice
    ? { task: 'qa_asr', model: 'whisper-v3-turbo', params: { texts: Object.fromEntries(list.map(a => [a.id, { text: a.x_sheet?.line || a.params?.text || a.x_voice?.ref_text || '', language: a.params?.language && String(a.params.language).length <= 3 && a.params.language !== 'auto' ? a.params.language : (isThaiText(a.params?.text) ? 'th' : null) }])) } }
    : { task: 'qa_clap', model: 'clap-htsat', params: { prompts: Object.fromEntries(list.map(a => [a.id, a.final_prompt || a.params?.prompt || a.prompt || ''])) } };
  try {
    const j = await api.submit({ ...spec, inputs: list.map(a => ({ asset: a.id })), project: store.project, title: `${voice ? 'Whisper check' : 'CLAP rank'} (${list.length})`, meta: { studio: 'sound' } });
    if (!quiet) toast(`${voice ? 'Transcribing' : 'Scoring'} ${list.length} take(s)…`, { kind: 'info', title: voice ? 'Whisper transcript match' : 'CLAP prompt match' });
    const jj = await S.waitJob(j.id);
    if (jj.status === 'failed') toast(jj.error?.message || 'QA failed', { kind: 'err', title: 'QA failed' });
    else if (!quiet && jj.result_meta) toast(`${Object.keys(jj.result_meta.scores || {}).length} scored · ${jj.result_meta.flagged || 0} flagged`, { kind: jj.result_meta.flagged ? 'warn' : 'ok', title: 'QA done' });
    return jj;
  } catch (e) { toastError(e, 'QA could not start'); return null; }
}

// ------------------------------------------------------------------ the generator (left panel + batch view)
class SoundStudio {
  constructor(el) {
    this.el = el;
    this.tab = prefs.get('sound.tab', 'sfx');
    this.modes = Object.fromEntries(TABS.filter(t => t.modes).map(t => [t.id, prefs.get(t.pref, t.modes[0].value)]));   // tab -> current task
    this.slots = [];                // [{def, slot: AssetSlot}] of the current model + task
    this.sel = new Set();
    this.groups = new Map();        // batch id -> {el, cards: Map, takesEl, head, pend: []}
    this.cards = new Map();         // asset id -> card
    this.assets = new Map();
    this.sort = prefs.get('sound.sort', 'new');
    this.dub = { target: 'en', source: 'auto', tts: '', translate: '', ...prefs.get('sound.dub', {}) };
    this.autoQA = prefs.get('sound.autoqa', false);
    this.root = h('div.studio.snd');
    this.tabsEl = h('div.snd-tabs', { role: 'tablist' });
    this.body = h('div.sp-body');
    this.genBtn = h('button.btn.btn-primary.gen-btn', { type: 'button', on: { click: () => this.generate() } });
    this.jobsEl = h('div.col', { style: { gap: '8px' } });
    this.foot = h('div.sp-foot', this.jobsEl, h('div', { style: { height: '8px' } }), this.genBtn);
    this.panel = h('div.studio-panel', this.tabsEl, this.body, this.foot);
    this.headTitle = h('div.grow', { style: { minWidth: '0' } });
    this.headSel = h('div.snd-head-sel');
    this.headTools = h('div.snd-filter');
    this.centerBody = h('div.sc-body');
    this.center = h('div.studio-center', h('div.sc-head', this.headTitle, this.headTools, this.headSel), this.centerBody);
    this.root.append(this.panel, this.center);
    el.append(this.root);
    this.picker = new S.ModelPicker({ task: this.task(), onChange: m => this._onModel(m) });
    this.form = new S.ParamForm({ memoryKey: 'sound', onChange: () => this._eta(), renderers: RENDERERS });
    this.prompt = new S.PromptBox({ memoryKey: 'sound', label: 'Prompt', onSubmit: () => this.generate(), onApply: (x, r) => this._onApply(x, r),
      context: () => S.getSetting('prompt.context.sound') || prefs.get('sound.context', '') });
    this.prompt.ta?.addEventListener('input', () => { clearTimeout(this._etaT); this._etaT = setTimeout(() => this._eta(), 350); });
    this.sheets = new Sheets(this);
    bus.on('asset', ev => this._onAssetEvent(ev));
    bus.on('project', () => { this.reload(); this.sheets.load(); });
    bus.on('models', () => this._eta());
    this.render();
  }
  get refSlot() { return this.slotOf('ref'); }
  get srcSlot() { return this.slotOf('source'); }
  slotOf(role) { return this.slots.find(x => x.def.role === role)?.slot || null; }
  task() {
    const t = TABS.find(x => x.id === this.tab); if (!t || !t.tasks.length) return null;
    if (!t.modes) return t.tasks[0];
    const m = this.modes[t.id];
    return t.tasks.includes(m) ? m : t.tasks[0];
  }

  // ---------------------------------------------------------------- layout
  render() {
    this.tabsEl.replaceChildren(...TABS.map(t => h(`button${t.id === this.tab ? '.on' : ''}`, { type: 'button', role: 'tab', title: t.label, 'aria-selected': t.id === this.tab, on: { click: () => this.setTab(t.id) } },
      t.short || t.label)));
    if (this.tab === 'sheets') { this.sheets.mount(); return; }
    this.foot.style.display = '';
    const task = this.task();
    if (this.picker.task !== task) this.picker.setTask(task);
    this._buildPanel();
    this.reload();
  }
  setTab(id, { mode } = {}) {
    const tab = TABS.find(t => t.id === id);
    if (mode && tab?.modes?.some(v => v.value === mode)) { this.modes[id] = mode; prefs.set(tab.pref, mode); }
    if (id === this.tab && !mode) return;
    this.tab = id; prefs.set('sound.tab', id);
    this.clearSel();
    this.render();
  }
  setTask(task) {
    const tab = TABS.find(t => t.tasks.includes(task));
    if (!tab) return;
    if (tab.modes) this.setTab(tab.id, { mode: task }); else this.setTab(tab.id);
  }
  _layoutSig(task, m) { return `${task}|${slotDefsFor(task, m).map(d => `${d.role}${d.required ? '!' : '?'}${d.types.join('+')}`).join(',')}|${showPromptFor(task, m) ? 'p' : '-'}`; }
  _buildPanel() {
    const task = this.task(); const ui = TASK_UI[task]; const m = this.picker.model;
    const tab = TABS.find(t => t.id === this.tab);
    const kids = [];
    if (tab?.modes) {
      kids.push(h('div.snd-mode', ...tab.modes.map(v => h(`button${v.value === task ? '.on' : ''}`, { type: 'button', title: store.tasks[v.value]?.label || v.label, on: { click: () => this.setTab(tab.id, { mode: v.value }) } }, ic(v.icon, 17), v.label))));
    }
    kids.push(this.picker.el);
    // input slots: the model's declared ones for this task, else the hard-coded reference / extend slots; inputs survive a model switch inside a task
    const keep = this._slotTask === task ? new Map(this.slots.map(x => [x.def.role, x.slot.ids])) : new Map();
    this.slots.forEach(x => x.slot.destroy()); this.slots = [];
    for (const def of slotDefsFor(task, m)) {
      const slot = new S.AssetSlot({ label: def.label + (def.required ? '' : ' (optional)'), types: def.types, role: def.role, hint: def.hint || '', onChange: ids => this._onSlot(def, slot, ids) });
      if (def.required) slot.el.querySelector('.field-label')?.append(h('span', { style: { color: 'var(--err)' }, title: 'Required' }, ' *'));
      this.slots.push({ def, slot });
      if (keep.get(def.role)?.length) slot.set(keep.get(def.role));
      kids.push(slot.el);
      if (def === REF_SLOT) kids.push(this._voiceChips());
    }
    this._slotTask = task;
    this.showPrompt = showPromptFor(task, m);
    if (this.showPrompt) {
      this.prompt.ta.placeholder = ui.ph;
      this.prompt.ta.style.fontFamily = ui.mono ? 'var(--mono)' : '';
      this.prompt.el.querySelector('.field-label')?.firstChild && (this.prompt.el.querySelector('.field-label').firstChild.textContent = ui.label);
      kids.push(this.prompt.el);
      if (ui.example) kids.push(h('div.row.wrap', btn('Insert example', { size: 'sm', icon: 'plus', onClick: () => { this.prompt.set(ui.example); this.prompt.focus(); } }),
        h('span.snd-tip', 'One syllable per line. “SP rest 0.2” = a pause, “G4,E4 0.4,0.4” = a slur.')));
    }
    if (task === 'dub') kids.push(this._dubBox());
    kids.push(h('div.field', h('div.field-label', ic('settings-2', 14), 'Settings'), this.form.el));
    if (ui.qa) {
      const qaT = S.toggle(this.autoQA, v => { this.autoQA = v; prefs.set('sound.autoqa', v); }, 'Auto QA');
      kids.push(h('div.row', h('span.field-label.grow', ic('badge-check', 14), ui.qa === 'asr' ? 'Check takes with Whisper when done' : 'Rank takes with CLAP when done'), qaT));
    }
    this.body.replaceChildren(...kids);
    this._onModel(m, true);
  }
  _onSlot(def, slot, ids) {
    const audioish = t => t === 'audio' || t === 'voice';
    const bad = ids.filter(id => { const t = slot.metas.get(id)?.type; return t && !def.types.some(d => d === t || (audioish(d) && audioish(t))); });
    if (bad.length) {
      toast(`${def.label} needs ${def.types.join(' or ')}, not ${slot.metas.get(bad[0]).type}.`, { kind: 'warn' });
      slot.set(ids.filter(id => !bad.includes(id)));
      return;
    }
    if (def.role === 'ref') this._onRef(slot, ids);
    this._eta();
  }
  _voiceChips() {
    const box = h('div.snd-voicechips');
    api.get('/api/x/sound/voices', { project: store.project }).then(r => {
      if (!r.items.length) { box.append(h('span.snd-tip', 'No voices yet: design one (Voice → Design) or save any take as a voice.')); return; }
      box.append(h('span.snd-tip', { style: { width: '100%' } }, 'Your voices:'), ...r.items.slice(0, 12).map(v => h('button.chip', { type: 'button', title: voiceRefText(v), on: { click: () => this.refSlot?.set([v.id]) } }, ic('mic', 12), v.name)));
    }).catch(() => {});
    return box;
  }
  async _onRef(slot, ids) {
    if (!ids.length) return;
    try {
      const a = slot.metas.get(ids[0]) || await api.asset(ids[0]);
      const t = voiceRefText(a);
      if (t && paramOf(this.picker.model, 'ref_text')) this.form.setValues({ ref_text: t });
      const lang = a.x_voice?.language;
      if (lang) { const l = langFor(this.picker.model, lang); if (l) this.form.setValues({ language: l }); }
    } catch { /* deleted */ }
  }
  _onModel(m, fromBuild = false) {
    const task = this.task();
    const pp = promptParamFor(task, m);
    this.promptParam = pp;
    this.form.exclude = new Set([...(pp === 'prompt' ? [] : [pp]), ...(task === 'dub' ? DUB_FIELDS : [])]);
    this.form.setModel(m, task);
    this.prompt.setModel(m, task);
    if (!fromBuild && this._layoutSig(task, m) !== this._sig) { this._buildPanel(); return; }
    this._sig = this._layoutSig(task, m);
    // ✨ Improve only where the model has a prompt template for the task (e.g. not for voice replies)
    const hasStyle = !!S.promptStyleOf(m, task);
    this.prompt.impBtn.classList.toggle('hidden', !hasStyle);
    if (!hasStyle) this.prompt.togglePanel(false);
    this._eta();
  }
  _onApply(extras, result) {
    const chosen = this.prompt.ta.value;
    const isVariation = chosen.trim() !== String(result.prompt || '').trim();
    const formVals = {};
    for (const [field, value] of Object.entries(extras)) {
      if (field === this.promptParam) {
        if (field === 'text' && isVariation && !result.apply?.prompt) continue;   // a chosen variation IS the line
        this.prompt.set(value);
      } else formVals[field] = value;
    }
    if (Object.keys(formVals).length) this.form.setValues(formVals);
    if (this.prompt.improved) this.prompt.improved.applied = this.prompt.ta.value;
  }
  _eta() {
    const m = this.picker.model;
    const loaded = (store.memory?.loaded || []).some(e => e.key?.endsWith(':' + m?.id));
    const s = S.estimateSeconds(m, this.form.allValues(), loaded);
    const ok = m?.status?.status === 'ready';
    const n = Number(this.form.allValues().batch || 1);
    const notReady = m?.status?.status === 'cloud_disabled' ? 'Cloud disabled' : m?.status?.status === 'missing_key' ? 'Needs API key' : 'Model not ready';
    this.genBtn.replaceChildren(ic('sparkles', 18), h('span', !m ? 'No model' : ok ? `Generate${n > 1 ? ` ${n} takes` : ''}` : notReady), s ? h('span.eta', `· ~${S.fmtDur(s)}`) : null,
      ok ? S.costSpan(m, { ...this.form.allValues(), prompt: this.prompt.value, text: this.prompt.value }, this.slots.reduce((n, x) => n + x.slot.ids.length, 0)) : null);
    this.genBtn.disabled = !m || !ok;
    this.genBtn.title = m && !ok ? m.status.reason : 'Generate (Ctrl/⌘ + Enter in the prompt)';
  }

  formValues() { return { ...this.form.values(), translate_backend: this.dub.translate || '' }; }
  /** Dub setup: voice-clone model, languages it can speak, translation backend (from /api/x/media_audio/dub/options). */
  _dubBox() {
    const box = h('div.snd-dub', h('div.snd-tip', ic('loader', 13, 'spin'), ' Loading the dub options…'));
    const save = () => prefs.set('sound.dub', this.dub);
    api.get('/api/x/media_audio/dub/options').then(o => {
      const names = o.language_names || LANG_NAMES;
      const models = o.tts_models || [];
      if (!models.some(m => m.id === this.dub.tts)) this.dub.tts = (models.find(m => m.id === o.default_tts && m.status === 'ready') || models.find(m => m.status === 'ready') || models[0] || {}).id || '';
      const draw = () => {
        const tm = models.find(m => m.id === this.dub.tts);
        const langs = tm?.languages?.length ? tm.languages : Object.keys(names);
        if (!langs.includes(this.dub.target)) this.dub.target = langs.includes('en') ? 'en' : langs[0];
        save();
        const sel = (items, cur, on) => { const e = h('select.select', { on: { change: ev => on(ev.target.value) } }, ...items.map(([v, l, dis]) => h('option', { value: v, selected: v === cur, disabled: !!dis }, l))); return e; };
        const note = tm ? [tm.status !== 'ready' ? h('div.snd-tip', { style: { color: 'var(--warn)' } }, ic('triangle-alert', 12), ` ${tm.name} is not ready: ${tm.reason || tm.status}`) : null,
          tm.license && tm.license.commercial === false ? h('div.snd-tip', S.licenseBadge(tm.license), ' The dub inherits this licence.') : null,
          !tm.languages?.length ? h('div.snd-tip', 'This model lists no languages: it may speak any language it was trained on.') : null] : [];
        box.replaceChildren(
          h('label.field', h('span.field-label', 'Voice-clone model (speaks the dub)'), sel(models.map(m => [m.id, `${m.name}${m.status === 'ready' ? '' : ' (not ready)'}`]), this.dub.tts, v => { this.dub.tts = v; draw(); })), ...note,
          h('div.row.wrap', h('label.field.grow', h('span.field-label', 'Dub language'), sel(langs.map(c => [c, `${names[c] || c} (${c})`]), this.dub.target, v => { this.dub.target = v; save(); })),
            h('label.field.grow', h('span.field-label', 'Spoken language'), sel([['auto', 'Detect'], ...Object.entries(names).map(([c, n]) => [c, `${n} (${c})`])], this.dub.source, v => { this.dub.source = v; save(); }))),
          h('label.field', h('span.field-label', 'Translate with'), sel([['', 'Local prompt LLM (default)'], ...(o.translate_backends || []).map(b => [b.id, `${b.label || b.id}${b.available === false ? ' (unavailable)' : ''}${b.cloud ? ' · cloud' : ''}`, b.available === false])], this.dub.translate, v => { this.dub.translate = v; save(); })),
          h('div.snd-tip', 'One speaker works best: the voice is cloned from the first clean seconds. Without a working translation backend the job stops as an editable draft.'));
      };
      draw();
    }).catch(e => box.replaceChildren(h('div.snd-err', `Could not load the dub options: ${e.message}`)));
    return box;
  }

  // ---------------------------------------------------------------- submit
  spec() {
    const task = this.task(); const m = this.picker.model;
    const params = { ...this.form.values() };
    const text = this.showPrompt ? this.prompt.value : '';
    if (this.showPrompt) {
      params[this.promptParam] = text;
      if (this.promptParam !== 'prompt') params.prompt = text;
      if (this.prompt.negative) params.negative = this.prompt.negative;
    }
    const inputs = this.slots.flatMap(x => x.slot.inputs());
    if (EXTEND_TASKS.includes(task) && this.slotOf('source')?.ids.length) params.mode = 'extend';
    const meta = { studio: 'sound', ...(this.showPrompt ? S.promptMeta(this.prompt) : {}) };
    const nameOf = role => { const sl = this.slotOf(role); const id = sl?.ids[0]; return id ? (sl.metas.get(id)?.name || '') : ''; };
    let label = text;
    if (task === 'voice_design') label = params.text || text;
    else if (task === 'voice_convert') label = `Convert: ${nameOf('source') || 'clip'}`;
    else if (task === 's2s') label = text || params.instruction || `Reply to ${nameOf('audio') || 'the clip'}`;
    else if (task === 'v2a') label = text || `Sound for ${nameOf('video') || 'the video'}`;
    else if (task === 'sing') label = `Sing: ${text.split('\n')[0] || ''}`;
    else if (task === 'subtitles') label = `Subtitles: ${nameOf('media') || 'clip'}`;
    else if (task === 'separate') label = `Stems: ${nameOf('media') || 'clip'}`;
    else if (task === 'dub') { label = `Dub → ${this.dub.target}: ${nameOf('media') || 'clip'}`; Object.assign(params, { target_language: this.dub.target, source_language: this.dub.source || 'auto', tts_model: this.dub.tts || '', translate_backend: this.dub.translate || '' }); }
    meta.asset_extra = { x_inputs: inputs };   // lets Regenerate / More takes restore the input slots by role
    if (!text) meta.asset_extra.x_title = (label || '').slice(0, 120);   // tasks without a prompt: the batch header shows this
    if (task === 'voice_design') meta.asset_extra.x_voice = { ref_text: params.text || '', design_prompt: text, language: params.language && params.language !== 'auto' ? params.language : (isThaiText(params.text) ? 'th' : null) };
    return { task, model: m?.id, params, inputs, project: store.project, meta, title: (label || m?.name || task).slice(0, 70) };
  }
  /** Generate needs text unless the task makes it optional (voice reply, video to sound without a text-only model) or a slot replaces it. */
  _promptRequired() {
    const task = this.task(); const m = this.picker.model;
    if (!this.showPrompt) return false;
    if (task === 'v2a') return !!m?.features?.needs_prompt;
    if (TASK_UI[task]?.promptOptional) return false;
    return !this.slots.some(x => x.def.replaces_prompt && x.slot.ids.length);
  }
  async generate(specOverride = null) {
    const task = this.task(); const m = this.picker.model;
    if (!m) return toast('Pick a model first.', { kind: 'warn' });
    const spec = specOverride || this.spec();
    if (!specOverride) {
      const missing = this.slots.find(x => x.def.required && !x.slot.ids.length);
      if (missing) return toast(`${missing.def.label} is required: pick one first.`, { kind: 'warn' });
      if (task === 's2s' && !this.slotOf('audio')?.ids.length && !spec.params.text && !spec.params.instruction) return toast('Add a question clip or type a message first.', { kind: 'warn' });
      if (this._promptRequired() && !this.prompt.value) { this.prompt.focus(); return toast(`${TASK_UI[task].label} first.`, { kind: 'warn' }); }
      if (task === 'voice_design' && !spec.params.text) return toast('Write a short line for the new voice to say (Settings → Line).', { kind: 'warn' });
      if (isThaiText(spec.params.text || spec.params.prompt) && isVoiceTask(task) && !thaiOk(m)) {
        toast(`${m.name} has no Thai. Use OmniVoice / OmniVoice Thai (or MOSS-TTS) for Thai lines.`, { kind: 'warn', title: 'Thai line' });
        return;
      }
    }
    try {
      const job = await api.submit(spec);
      store.jobs.set(job.id, job);
      this._trackJob(job, Number(spec.params.batch || 1));
      const card = S.jobCard(job, { compact: true });
      this.jobsEl.prepend(card);
      const off = bus.on('job:' + job.id, j => {
        if (['done', 'cancelled'].includes(j.status)) { setTimeout(() => card.remove(), 1800); off(); }
        if (j.status === 'failed') off();
        if (j.status === 'done' && this.autoQA && j.outputs?.length && !['qa_clap', 'qa_asr'].includes(j.task)) {
          api.assets({ ids: j.outputs.join(','), limit: 50 }).then(r => runQA(r.items, { quiet: true }));
        }
      });
      while (this.jobsEl.children.length > 3) this.jobsEl.lastChild.remove();
    } catch (e) { toastError(e, 'Cannot start the job'); }
  }

  // ---------------------------------------------------------------- center: batches
  async reload() {
    const task = this.task(); if (!task) return;
    this._renderHead();
    try {
      const r = await api.assets({ project: store.project, studio: 'sound', task, limit: 300 });
      if (this.task() !== task) return;
      this.groups.clear(); this.cards.clear(); this.assets.clear();
      this.centerBody.replaceChildren();
      if (this.tab === 'voice') this.centerBody.append(this._voiceLibrary());
      this.listEl = h('div');
      this.centerBody.append(this.listEl);
      for (const a of r.items) this._add(a, false);
      this._sortGroups();
      this._refreshEmpty();
    } catch (e) { toastError(e); }
  }
  _groupKey(a) { return a.batch?.id || a.job || a.id; }
  _group(key, sample, { prepend = true } = {}) {
    let g = this.groups.get(key);
    if (g) return g;
    const takesEl = h('div.snd-takes');
    const head = h('div.snd-ghead');
    const el = h('div.snd-group', head, takesEl);
    g = { key, el, head, takesEl, ids: [], pend: [], sample, job: null };
    this.groups.set(key, g);
    if (prepend) this.listEl.prepend(el); else this.listEl.append(el);
    this._renderGroupHead(g);
    return g;
  }
  _renderGroupHead(g) {
    const a = g.sample || {};
    const job = g.job;
    const m = store.modelsById.get(a.model || job?.model);
    const isPost = !!a.x_post;
    const title = isPost ? `Processed: ${a.x_post.labels?.join(' · ') || ''}` : (a.improved_prompt || a.final_prompt || a.prompt || a.x_title || job?.title || a.name || '');
    const ids = () => g.ids.filter(id => this.assets.has(id));
    const assets = () => ids().map(id => this.assets.get(id));
    const qk = qaOf(a.task || job?.task);
    const voice = qk === 'asr';
    g.head.replaceChildren(
      h('div.gp', h('div.t', { title }, isPost ? h('span', ic('git-fork', 13), ' ', title) : title || '—'),
        h('div.s', h('span', m?.name || a.model_name || job?.model_name || ''), m && m.license?.commercial === false ? S.licenseBadge(m.license) : null,
          a.params?.speaker ? h('span.snd-m', a.params.speaker) : null, a.params?.instruct ? h('span.ellipsis', { style: { maxWidth: '260px' }, title: a.params.instruct }, `“${a.params.instruct}”`) : null,
          h('span', `${ids().length} take${ids().length === 1 ? '' : 's'}`), a.created ? h('span', S.fmtAgo(a.created)) : null)),
      h('div.ga',
        job && !['done', 'failed', 'cancelled'].includes(job.status) ? h('span.badge.accent', ic('loader', 12, 'spin'), job.status === 'queued' ? (job.queue_position ? `queued #${job.queue_position}` : 'queued') : `${Math.round((job.progress?.frac || 0) * 100)}%`) : null,
        job && !['done', 'failed', 'cancelled'].includes(job.status) ? btn('', { size: 'sm', kind: 'ghost', icon: 'square', title: 'Cancel', onClick: () => api.cancel(job.id).catch(toastError) }) : null,
        ids().length && qk ? btn(voice ? 'Check' : 'Rank', { size: 'sm', icon: 'badge-check', title: voice ? 'Whisper: transcript match % per take (flags mismatches)' : 'CLAP: prompt-match score per take (best first)', onClick: async () => { await runQA(assets()); this.sort = 'qa'; prefs.set('sound.sort', 'qa'); this._renderHead(); this._sortGroup(g); } }) : null,
        ids().length && !isPost && a.model ? btn('More', { size: 'sm', icon: 'plus', title: 'More takes of this prompt (new seeds)', onClick: () => this.variations(a) }) : null,
        ids().length ? btn('', { size: 'sm', icon: 'wand', title: 'Audio tools for all takes in this batch', onClick: () => postToolsModal(ids()) }) : null,
        ids().length ? btn('', { size: 'sm', kind: 'ghost', icon: 'check', title: 'Select all takes of this batch', onClick: () => { ids().forEach(id => this.sel.add(id)); this._selChanged(); } }) : null),
      job?.status === 'failed' ? h('div.snd-err', { style: { width: '100%', padding: 0 } }, ic('circle-alert', 13), ' ', job.error?.message || 'failed', job.error?.fix ? h('div.dim', job.error.fix) : null,
        h('div', { style: { marginTop: '6px' } }, btn('Retry', { size: 'sm', icon: 'rotate-ccw', onClick: () => api.retry(job.id).then(j => this._trackJob(j, job.params?.batch || 1)).catch(toastError) }))) : null);
  }
  _ctx() {
    return { open: a => S.openAsset(a, { list: [...this.cards.keys()] }), toggleSelect: id => { this.sel.has(id) ? this.sel.delete(id) : this.sel.add(id); this._selChanged(); },
      isSelected: id => this.sel.has(id), menu: (anchor, a) => this.cardMenu(anchor, a), editDraft: a => dubDraftModal(a, this),
      isBest: id => { const a = this.assets.get(id); const g = this.groups.get(this._groupKey(a)); if (!g || g.ids.length < 2) return false; const sc = g.ids.map(i => scoreOf(this.assets.get(i) || {})); const s = scoreOf(a); return s != null && s === Math.max(...sc.filter(x => x != null)); } };
  }
  _add(a, live = true) {
    if (this.cards.has(a.id)) { this._update(a); return; }
    this.assets.set(a.id, a);
    const key = this._groupKey(a);
    const g = this._group(key, a, { prepend: live });
    if (!g.sample?.created || g.sample.id === undefined) g.sample = a;
    if (!g.ids.includes(a.id)) g.ids.push(a.id);
    const c = (isMediaOut(a) ? mediaCard : soundCard)(a, this._ctx());
    this.cards.set(a.id, c);
    const ph = g.pend.shift();
    if (ph) ph.replaceWith(c); else g.takesEl.append(c);
    if (live) this._sortGroup(g);
    this._renderGroupHead(g);
    this._refreshEmpty();
  }
  _update(a) {
    this.assets.set(a.id, a);
    this.cards.get(a.id)?.update(a);
    const g = this.groups.get(this._groupKey(a));
    if (g) { if (this.sort === 'qa' || this.sort === 'rating') this._sortGroup(g); for (const id of g.ids) if (id !== a.id) this.cards.get(id)?.update(this.assets.get(id)); }
  }
  _remove(id) {
    const a = this.assets.get(id); if (!a) return;
    this.cards.get(id)?.remove(); this.cards.delete(id); this.assets.delete(id); this.sel.delete(id);
    const g = this.groups.get(this._groupKey(a));
    if (g) { g.ids = g.ids.filter(x => x !== id); if (!g.ids.length && !g.pend.length) { g.el.remove(); this.groups.delete(g.key); } else this._renderGroupHead(g); }
    this._selChanged(); this._refreshEmpty();
  }
  _sortGroup(g) {
    const key = id => { const a = this.assets.get(id) || {}; if (this.sort === 'qa') return -(scoreOf(a) ?? -9); if (this.sort === 'rating') return -((a.rating || 0) * 10 + (a.favorite ? 5 : 0)); return a.batch?.index ?? 0; };
    g.ids.sort((x, y) => key(x) - key(y));
    for (const id of g.ids) { const c = this.cards.get(id); if (c) g.takesEl.append(c); }
    for (const p of g.pend) g.takesEl.append(p);
  }
  _sortGroups() { for (const g of this.groups.values()) this._sortGroup(g); }
  _trackJob(job, n) {
    const g = this._group(job.id, null);
    g.job = job;
    for (let i = 0; i < Math.max(1, Math.min(n, 8)); i++) { const p = pendingTake(i); g.pend.push(p); g.takesEl.append(p); }
    this._renderGroupHead(g); this._refreshEmpty();
    this.listEl.prepend(g.el);
    const off = bus.on('job:' + job.id, j => {
      g.job = j;
      if (['done', 'failed', 'cancelled'].includes(j.status)) {
        off(); g.pend.forEach(p => p.remove()); g.pend = [];
        if (!g.ids.length && j.status !== 'failed') { g.el.remove(); this.groups.delete(g.key); }
        if (j.status === 'failed') toast(j.error?.message || 'Job failed', { kind: 'err', title: 'Generation failed' });
        // outputs that arrived before the asset events (or after a reconnect)
        const missing = (j.outputs || []).filter(id => !this.cards.has(id));
        if (missing.length) api.assets({ ids: missing.join(','), limit: 50 }).then(r => r.items.forEach(a => { if (a.task === this.task()) this._add(a); })).catch(() => {});
      }
      this._renderGroupHead(g);
      this._refreshEmpty();
    });
  }
  _onAssetEvent(ev) {
    const a = ev.asset;
    if (ev.action === 'deleted') { this._remove(a.id); this.sheets.onAsset(ev); return; }
    if (a.project !== store.project) return;
    this.sheets.onAsset(ev);
    if (this.tab === 'voice' && a.type === 'voice') this._refreshLibrary();
    if (ev.action === 'created' && a.studio === 'sound' && a.task === this.task() && this.tab !== 'sheets') this._add(a);
    else if (ev.action === 'updated' && this.cards.has(a.id)) this._update(a);
  }
  _refreshEmpty() {
    if (!this.listEl) return;
    const emptyEl = this.centerBody.querySelector(':scope > .empty');
    const none = !this.groups.size;
    if (none && !emptyEl) { const t = this.task(); this.centerBody.append(S.empty(TABS.find(x => x.id === this.tab)?.icon || 'audio-lines', 'No takes yet', TASK_UI[t]?.empty)); }
    if (!none && emptyEl) emptyEl.remove();
    this._renderHeadTitle();
  }
  _renderHeadTitle() {
    const t = this.task();
    this.headTitle.replaceChildren(h('div', { style: { fontWeight: 650 } }, store.tasks[t]?.label || t), h('div.dim', { style: { fontSize: '12px' } }, `${this.cards.size} take(s) · ${this.groups.size} batch(es) in ${store.project}`));
  }
  _renderHead() {
    this._renderHeadTitle();
    this.headTools.replaceChildren(h('span.dim', { style: { fontSize: '12px' } }, 'Sort'), S.seg([{ value: 'new', label: 'Take #' }, { value: 'qa', label: 'QA' }, { value: 'rating', label: '★' }], this.sort, v => { this.sort = v; prefs.set('sound.sort', v); this._sortGroups(); }));
    this._selChanged();
  }
  clearSel() { this.sel.clear(); this._selChanged(); }
  _selChanged() {
    for (const [id, c] of this.cards) c.classList.toggle('sel', this.sel.has(id));
    const ids = [...this.sel].filter(id => this.assets.has(id));
    this.headSel.replaceChildren();
    if (!ids.length) return;
    this.headSel.append(h('span.badge.accent', `${ids.length} selected`),
      ids.length === 2 ? btn('A/B', { size: 'sm', icon: 'columns-2', onClick: () => S.compare(ids[0], ids[1]) }) : null,
      btn('Tools', { size: 'sm', icon: 'wand', onClick: () => postToolsModal(ids) }),
      btn('QA', { size: 'sm', icon: 'badge-check', onClick: () => runQA(ids.map(i => this.assets.get(i))) }),
      btn('Export', { size: 'sm', icon: 'package', onClick: () => exportModal(ids) }),
      btn('', { size: 'sm', kind: 'ghost', icon: 'x', title: 'Clear selection', onClick: () => this.clearSel() }));
  }

  // ---------------------------------------------------------------- voice library
  _voiceLibrary() {
    this.libEl = h('div.snd-lib');
    this._refreshLibrary();
    return this.libEl;
  }
  async _refreshLibrary() {
    if (!this.libEl) return;
    let items = [];
    try { items = (await api.get('/api/x/sound/voices', { project: store.project })).items; } catch { /* offline */ }
    const file = h('input', { type: 'file', class: 'hidden', accept: '.wav,.flac,.ogg,.mp3', on: { change: async () => {
      for (const f of file.files) {
        try { const a = await api.upload(f, { project: store.project, type: 'voice', name: f.name.replace(/\.[^.]+$/, '') }); toast('Voice uploaded. Add its transcript for better clones.', { kind: 'ok' }); saveVoiceModal(a); } catch (e) { toastError(e, 'Upload failed'); }
      }
    } } });
    const open = prefs.get('sound.libOpen', true);
    const list = h('div.vl', ...items.map(v => h(`div.snd-voice${this.refSlot?.ids.includes(v.id) ? '.on' : ''}`, S.playButton(v, 32),
      h('div.vt', h('b', { title: v.name }, v.name), h('span', { title: voiceRefText(v) }, voiceRefText(v) || 'no transcript'), h('span', [v.x_voice?.language, S.fmtClock(v.info?.duration || 0), v.model_name].filter(Boolean).join(' · '))),
      btn('', { size: 'sm', icon: 'mic', title: 'Clone lines with this voice', onClick: () => this.useVoice(v) }),
      btn('', { size: 'sm', kind: 'ghost', icon: 'more-vertical', title: 'More', onClick: e => S.menu(e.currentTarget, [
        { label: 'Details', icon: 'panel-right', onClick: () => S.openAsset(v) },
        { label: 'Edit transcript / name…', icon: 'pencil', onClick: async () => { const t = await S.promptDialog('Transcript of the clip', { value: voiceRefText(v), title: v.name }); if (t != null) api.patchAsset(v.id, { x_voice: { ...(v.x_voice || {}), ref_text: t } }).then(() => this._refreshLibrary()).catch(toastError); } },
        { label: 'Delete', icon: 'trash-2', danger: true, onClick: async () => { if (await S.confirmDialog(`Delete the voice “${v.name}”?`, { danger: true, ok: 'Delete' })) api.deleteAsset(v.id).catch(toastError); } },
      ], { align: 'end' }) }))));
    if (!items.length) list.append(h('div.snd-tip', { style: { gridColumn: '1 / -1' } }, 'Design a voice (Design), or save any take with ⋮ → Save as voice. Upload a clean 5-15 s clip to clone a real voice you have rights to.'));
    const det = h('details', { open, on: { toggle: e => prefs.set('sound.libOpen', e.currentTarget.open) } },
      h('summary', { style: { cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', listStyle: 'none' } }, ic('mic', 16), h('b.grow', `Voice library (${items.length})`),
        btn('Upload', { size: 'sm', icon: 'upload', onClick: e => { e.preventDefault(); file.click(); } }), file), list);
    this.libEl.replaceChildren(det);
  }
  useVoice(v) {
    this.setTab('voice', { mode: 'voice_clone' });
    this.refSlot?.set([v.id]);
    toast(`Cloning with “${v.name}”: write the line and generate.`, { kind: 'info' });
    this.prompt.focus();
  }

  // ---------------------------------------------------------------- card actions / send-to
  cardMenu(anchor, a) {
    if (isMediaOut(a)) {   // subtitles, dub plan, dubbed video
      const draft = !!a.x_dub && (a.tags || []).includes('draft');
      return S.menu(anchor, [
        { label: 'Details & lineage', icon: 'panel-right', onClick: () => S.openAsset(a, { list: [...this.cards.keys()] }) },
        draft ? { label: 'Edit lines & render…', icon: 'pencil', onClick: () => dubDraftModal(a, this) } : null,
        { label: 'Regenerate with these settings', icon: 'refresh-cw', onClick: () => this.regenerate(a) },
        { label: 'Download', icon: 'download', onClick: () => { location.href = S.fileUrl(a, a.main, true); } },
        { sep: true },
        { label: 'Delete', icon: 'trash-2', danger: true, onClick: async () => { if (await S.confirmDialog(`Delete “${a.name}”?`, { danger: true, ok: 'Delete' })) api.deleteAsset(a.id).catch(toastError); } },
      ], { align: 'end' });
    }
    const voiceish = a.type === 'voice' || isSpeechTask(a.task);
    const ext = ['music', 'ambience', 'sfx'].includes(a.task) && ready('sa3-medium');
    S.menu(anchor, [
      { label: 'Details & lineage', icon: 'panel-right', onClick: () => S.openAsset(a, { list: [...this.cards.keys()] }) },
      { label: 'Audio tools…', icon: 'wand', sub: 'trim, fade, normalize, loop, convert', onClick: () => postToolsModal([a.id]) },
      { label: 'Make seamless loop', icon: 'repeat', onClick: () => postToolsModal([a.id], { preset: { loop: true, normalize: true }, title: 'Seamless loop' }) },
      { label: 'More takes like this', icon: 'shuffle', onClick: () => this.variations(a) },
      ext ? { label: 'Extend (continue it)', icon: 'arrow-right', onClick: () => this.extend(a) } : null,
      { label: 'Regenerate with these settings', icon: 'refresh-cw', onClick: () => this.regenerate(a) },
      { sep: true },
      voiceish && a.type !== 'voice' ? { label: 'Save as voice…', icon: 'mic', onClick: () => saveVoiceModal(a) } : null,
      voiceish || a.type === 'audio' ? { label: 'Use as voice reference', icon: 'mic', onClick: () => this.receive({ action: 'voice_ref', assets: [a] }) } : null,
      voiceish || a.type === 'audio' ? { label: 'Convert the voice…', icon: 'arrow-left-right', sub: 'Voice → Convert', onClick: () => this.receive({ action: 'convert', assets: [a] }) } : null,
      qaOf(a.task) || a.type === 'voice' ? { label: voiceishLabel(a), icon: 'badge-check', onClick: () => runQA([a]) } : null,
      { label: 'Export…', icon: 'package', onClick: () => exportModal([a.id]) },
      { label: 'Download', icon: 'download', onClick: () => { location.href = S.fileUrl(a, a.main, true); } },
      { sep: true },
      { label: 'Delete', icon: 'trash-2', danger: true, onClick: async () => { if (await S.confirmDialog(`Delete “${a.name}”?`, { danger: true, ok: 'Delete' })) api.deleteAsset(a.id).catch(toastError); } },
    ], { align: 'end' });
    function voiceishLabel(x) { return x.type === 'voice' || qaOf(x.task) === 'asr' ? 'Check transcript (Whisper)' : 'Score prompt match (CLAP)'; }
  }
  async _origin(a) {   // processed takes regenerate from their source take
    let cur = a;
    for (let k = 0; k < 6 && cur.x_post?.from; k++) { try { cur = await api.asset(cur.x_post.from); } catch { break; } }
    return cur;
  }
  async regenerate(a, { autoRun = false, newSeed = false } = {}) {
    a = await this._origin(a);
    if (a.x_sheet) { this.setTab('sheets'); this.sheets.focus(a.x_sheet); return; }
    if (!a.task || !TASK_UI[a.task]) return toast('This asset was not made in the Sound studio.', { kind: 'warn' });
    this.setTask(a.task);
    if (a.model && store.modelsById.get(a.model)?.tasks.includes(a.task)) { this.picker.set(a.model, true); this._onModel(this.picker.model); }
    const p = { ...(a.params || {}) };
    if (a.task === 'dub') { this.dub = { ...this.dub, target: p.target_language || this.dub.target, source: p.source_language || 'auto', tts: p.tts_model || this.dub.tts, translate: p.translate_backend || '' }; prefs.set('sound.dub', this.dub); this._buildPanel(); }
    this.form.setValues({ ...p, seed: newSeed ? -1 : (a.seed ?? p.seed) });
    if (this.showPrompt) { this.prompt.set(p[this.promptParam] || a.improved_prompt || a.prompt || ''); this.prompt.setNegative(a.negative || p.negative || ''); }
    if (Array.isArray(a.x_inputs)) {   // saved with the take: restore every slot by role
      const byRole = new Map();
      for (const i of a.x_inputs) byRole.set(i.role, [...(byRole.get(i.role) || []), i.asset]);
      for (const [role, ids] of byRole) this.slotOf(role)?.set(ids);
    } else {
      const ref = a.worker?.ref_asset || (a.task === 'voice_clone' ? (a.parents || [])[0] : null);
      if (ref && this.refSlot) this.refSlot.set([ref]);
      if (p.mode === 'extend' && this.srcSlot && a.parents?.[0]) this.srcSlot.set([a.parents[0]]);
    }
    if (autoRun) { setTimeout(() => this.generate(), 50); return; }
    toast('Settings restored from the take: tweak one knob and generate.', { kind: 'info' });
  }
  variations(a) { return this.regenerate(a, { autoRun: true, newSeed: true }); }
  async extend(a) {
    a = await this._origin(a);
    this.setTask(['music', 'ambience', 'sfx'].includes(a.task) ? a.task : 'music');
    if (ready('sa3-medium')) { this.picker.set('sa3-medium', true); this._onModel(this.picker.model); this._buildPanel(); }
    this.srcSlot?.set([a.id]);
    this.prompt.set(a.final_prompt || a.prompt || '');
    toast('Clip set. Choose how much to add (Advanced → Extend by) and generate.', { kind: 'info' });
  }

  route(r) {
    if (!r.sub) return;
    if (r.sub === 'sheets') return this.setTab('sheets');
    if (TABS.find(t => t.id === r.sub)) { if (r.sub === 'voice' && r.query?.mode) this.setTab('voice', { mode: r.query.mode }); else this.setTab(r.sub); return; }
    if (TASK_UI[r.sub]) this.setTask(r.sub);
  }
  async receive(p) {
    const assets = p.assets || [];
    const a = assets[0];
    switch (p.action) {
      case 'prompt': {
        const task = p.params?.task;
        if (task && TASK_UI[task]) this.setTask(task);
        if (p.params?.prompt) this.prompt.set(p.params.prompt);
        const extra = { ...p.params }; delete extra.prompt; delete extra.task;
        if (Object.keys(extra).length) this.form.setValues(extra);
        return;
      }
      case 'regenerate': return a && this.regenerate(a);
      case 'variations': return a && this.variations(a);
      case 'extend': return a && this.extend(a);
      case 'voice_ref':
        if (!a) return;
        this.setTab('voice', { mode: 'voice_clone' });
        this.refSlot?.set([a.id]);
        if (a.type !== 'voice') toast('Using this take as the reference. Save it as a voice to reuse it in line sheets.', { kind: 'info', action: { label: 'Save as voice', onClick: () => saveVoiceModal(a) } });
        return;
      case 'convert':
        if (!a) return;
        this.setTab('voice', { mode: 'voice_convert' });
        this.slotOf('source')?.set([a.id]);
        toast('Source clip set. Pick the reference voice (most models) and generate.', { kind: 'info' });
        return;
      case 'reply':
        if (!a) return;
        this.setTab('voice', { mode: 's2s' });
        this.slotOf('audio')?.set([a.id]);
        return;
      case 'subtitles': case 'separate': case 'dub':
        if (!a) return;
        this.setTab('media', { mode: p.action });
        this.slotOf('media')?.set([a.id]);
        return;
      case 'v2a':
        if (!a) return;
        this.setTab('v2a');
        this.slotOf('video')?.set([a.id]);
        return;
      case 'loop': return postToolsModal(assets.map(x => x.id), { preset: { loop: true, normalize: true }, title: 'Seamless loop' });
      case 'normalize': return postToolsModal(assets.map(x => x.id), { preset: { normalize: true }, title: 'Normalize loudness' });
      case 'post': return postToolsModal(assets.map(x => x.id));
      case 'sheet_rows': this.setTab('sheets'); return this.sheets.importRows(p.params || {});
      default:
        if (a) this.regenerate(a);
    }
  }
  show() { this._eta(); }
}

// ------------------------------------------------------------------ sheets (SFX event lists + voice line sheets)
class Sheets {
  constructor(st) {
    this.st = st;
    this.data = { sheets: {} };
    this.cur = prefs.get('sound.sheet', null);
    this.takes = new Map();          // sheet id -> row id -> [assets]
    this.rowJobs = new Map();        // row id -> job
    this.loaded = false;
    this.voices = [];
    this._save = S.debounce(() => this.save(), 700);
  }
  async load() {
    try { this.data = await api.get('/api/x/sound/sheets', { project: store.project }); } catch { this.data = { sheets: {} }; }
    this.data.sheets = this.data.sheets || {};
    if (!this.data.sheets[this.cur]) this.cur = Object.keys(this.data.sheets)[0] || null;
    this.loaded = true;
    await this.loadTakes();
    if (this.st.tab === 'sheets') this.mount();
  }
  async loadTakes() {
    this.takes.clear();
    try {
      const r = await api.assets({ project: store.project, studio: 'sound', q: '', limit: 1000 });
      for (const a of r.items) this._takeIn(a);
      this.voices = (await api.get('/api/x/sound/voices', { project: store.project })).items;
    } catch { /* offline */ }
  }
  _takeIn(a) {
    const x = a.x_sheet; if (!x?.sheet || !x.row) return;
    if (!this.takes.has(x.sheet)) this.takes.set(x.sheet, new Map());
    const m = this.takes.get(x.sheet);
    const arr = (m.get(x.row) || []).filter(t => t.id !== a.id);
    arr.push(a); m.set(x.row, arr);
  }
  onAsset(ev) {
    const a = ev.asset;
    if (ev.action === 'deleted') { for (const m of this.takes.values()) for (const [k, arr] of m) m.set(k, arr.filter(t => t.id !== a.id)); }
    else if (a.x_sheet) this._takeIn(a);
    else if (a.type === 'voice') this.voices = [a, ...this.voices.filter(v => v.id !== a.id)];
    else return;
    if (this.st.tab === 'sheets' && this.cur && (a.x_sheet?.sheet === this.cur || ev.action === 'deleted')) this._rowsDirty();
  }
  _rowsDirty() { clearTimeout(this._rt); this._rt = setTimeout(() => this.renderRows(), 120); }
  async save() {
    try { await api.put('/api/x/sound/sheets', { project: store.project, sheets: this.data.sheets }); } catch (e) { toastError(e, 'Sheet not saved'); }
  }
  sheet() { return this.data.sheets[this.cur] || null; }
  newSheet(kind) {
    const id = uid();
    const n = Object.values(this.data.sheets).filter(s => s.kind === kind).length + 1;
    this.data.sheets[id] = kind === 'lines'
      ? { id, kind, name: `Line sheet ${n}`, rows: [], cast: {}, takes: 2, created: new Date().toISOString() }
      : { id, kind, name: `SFX list ${n}`, rows: [], task: 'sfx', model: null, takes: 3, seconds: 2, created: new Date().toISOString() };
    this.cur = id; prefs.set('sound.sheet', id); this._save(); this.mount();
  }
  focus(x) { if (this.data.sheets[x.sheet]) { this.cur = x.sheet; prefs.set('sound.sheet', x.sheet); this.mount(); } }
  // ------------------------------------------------------------ layout
  mount() {
    const st = this.st;
    st.foot.style.display = 'none';
    st.headTools.replaceChildren(); st.headSel.replaceChildren();
    if (!this.loaded) { st.body.replaceChildren(h('div.dim', ic('loader', 14, 'spin'), ' Loading sheets…')); this.load(); return; }
    const sh = this.sheet();
    const pick = h('select.select', { 'aria-label': 'Sheet', on: { change: e => { this.cur = e.target.value; prefs.set('sound.sheet', this.cur); this.mount(); } } },
      ...Object.values(this.data.sheets).map(s => h('option', { value: s.id, selected: s.id === this.cur }, `${s.kind === 'lines' ? '🎙' : '🔊'} ${s.name}`)));
    const kids = [
      h('div.field', h('div.field-label', ic('list-checks', 14), 'Sheet'), Object.keys(this.data.sheets).length ? pick : h('div.snd-tip', 'No sheets yet.'),
        h('div.row.wrap', btn('New SFX list', { size: 'sm', icon: 'zap', onClick: () => this.newSheet('sfx') }), btn('New line sheet', { size: 'sm', icon: 'mic', onClick: () => this.newSheet('lines') }))),
    ];
    if (sh) {
      const nm = h('input.input', { value: sh.name, on: { change: e => { sh.name = e.target.value || sh.name; this._save(); this.mount(); } } });
      kids.push(h('label.field', h('span.field-label', 'Name'), nm));
      kids.push(sh.kind === 'lines' ? this._linesPanel(sh) : this._sfxPanel(sh));
      kids.push(h('div.row.wrap', btn('Paste CSV / list', { icon: 'list', size: 'sm', onClick: () => this.pasteModal(sh) }),
        btn(sh.kind === 'lines' ? 'Write lines' : 'From design doc', { icon: 'pen-line', size: 'sm', title: sh.kind === 'lines' ? 'Write barks / lines from a character description (text backend)' : 'Turn design-doc text into SFX events (text backend)', onClick: () => this.writeRows(sh) }),
        btn('Add row', { icon: 'plus', size: 'sm', onClick: () => { this.addRow(sh, {}); } }),
        btn('', { icon: 'trash-2', size: 'sm', kind: 'ghost', title: 'Delete this sheet (takes stay in the library)', onClick: async () => { if (await S.confirmDialog(`Delete the sheet “${sh.name}”? Its takes stay in the library.`, { danger: true, ok: 'Delete' })) { delete this.data.sheets[sh.id]; this.cur = Object.keys(this.data.sheets)[0] || null; this._save(); this.mount(); } } })));
      const go = btn('Generate missing', { kind: 'primary', icon: 'sparkles', onClick: () => this.generateAll(sh, false) });
      kids.push(h('div.col', { style: { gap: '8px' } }, go, h('div.row.wrap', btn('Regenerate all', { size: 'sm', icon: 'refresh-cw', onClick: () => this.generateAll(sh, true) }),
        btn('QA all', { size: 'sm', icon: 'badge-check', onClick: () => runQA(this.allTakes(sh), { kind: sh.kind === 'lines' ? 'asr' : 'clap' }) }),
        btn('Export', { size: 'sm', icon: 'package', onClick: () => this.exportSheet(sh) }))));
    }
    st.body.replaceChildren(...kids);
    this.renderRows();
  }
  _sfxPanel(sh) {
    const box = h('div.col', { style: { gap: '12px' } });
    const picker = new S.ModelPicker({ task: sh.task || 'sfx', value: sh.model && store.modelsById.get(sh.model)?.tasks.includes(sh.task) ? sh.model : null,
      onChange: m => { sh.model = m?.id || null; this._save(); } });
    if (!sh.model) sh.model = picker.value;
    box.append(h('div.field', h('span.field-label', 'Kind'), S.seg([{ value: 'sfx', label: 'SFX' }, { value: 'ambience', label: 'Ambience' }], sh.task || 'sfx', v => { sh.task = v; sh.model = null; this._save(); this.mount(); })), picker.el,
      h('div.row', h('label.field.grow', h('span.field-label', 'Takes per event'), h('input.input.input-sm', { type: 'number', min: 1, max: 8, value: sh.takes || 3, on: { change: e => { sh.takes = Math.max(1, Math.min(8, Number(e.target.value) || 3)); this._save(); } } })),
        h('label.field.grow', h('span.field-label', 'Default length (s)'), h('input.input.input-sm', { type: 'number', min: 0.5, max: 60, step: 0.5, value: sh.seconds || 2, on: { change: e => { sh.seconds = Number(e.target.value) || 2; this._save(); } } }))),
      h('div.snd-tip', 'One event per row: a name (becomes the file name) and a prompt. ✨ writes the Stable Audio prompt for each row.'));
    return box;
  }
  _linesPanel(sh) {
    const chars = [...new Set(sh.rows.map(r => (r.character || '').trim()).filter(Boolean))];
    const tts = S.modelsFor('tts').filter(m => m.status.status === 'ready');
    const clone = S.modelsFor('voice_clone').filter(m => m.status.status === 'ready');
    const qwen = store.modelsById.get('qwen3-tts-customvoice');
    const box = h('div.snd-cast', h('div.field-label', ic('user', 14), `Cast (${chars.length})`));
    if (!chars.length) box.append(h('div.snd-tip', 'Add rows with a character name; each character gets a voice here.'));
    for (const c of chars) {
      const cast = sh.cast[c] || (sh.cast[c] = { kind: this.voices.length ? 'voice' : 'preset', voice: this.voices[0]?.id || null, model: null, speaker: 'Ryan' });
      const row = h('div.cr');
      const draw = () => {
        const kind = S.seg([{ value: 'voice', label: 'Library voice' }, { value: 'preset', label: 'Preset voice' }], cast.kind, v => { cast.kind = v; this._save(); draw(); });
        const parts = [h('div.row', h('b.grow.ellipsis', c), h('span.dim', `${sh.rows.filter(r => r.character === c).length} lines`)), kind];
        if (cast.kind === 'voice') {
          parts.push(h('select.select', { 'aria-label': `Voice of ${c}`, on: { change: e => { cast.voice = e.target.value || null; this._save(); } } }, h('option', { value: '' }, this.voices.length ? '— pick a voice —' : 'no voices: design one first'),
            ...this.voices.map(v => h('option', { value: v.id, selected: v.id === cast.voice }, v.name))));
          parts.push(h('select.select', { 'aria-label': 'Clone model', on: { change: e => { cast.model = e.target.value || null; this._save(); } } }, h('option', { value: '' }, 'Clone model: default'),
            ...clone.map(m => h('option', { value: m.id, selected: m.id === cast.model }, `${m.name}${m.license?.commercial === false ? ' (NC)' : ''}`))));
        } else {
          const model = store.modelsById.get(cast.model) || qwen;
          parts.push(h('select.select', { 'aria-label': 'TTS model', on: { change: e => { cast.model = e.target.value || null; this._save(); draw(); } } }, h('option', { value: '' }, `TTS model: ${qwen?.name || 'default'}`),
            ...tts.map(m => h('option', { value: m.id, selected: m.id === cast.model }, `${m.name}${m.license?.commercial === false ? ' (NC)' : ''}`))));
          const sp = paramOf(model, 'speaker') || paramOf(model, 'voice');
          if (sp) parts.push(h('select.select', { 'aria-label': 'Preset', on: { change: e => { cast.speaker = e.target.value; this._save(); } } }, ...(sp.choices || []).map(v => h('option', { value: v, selected: v === cast.speaker }, v))));
        }
        row.replaceChildren(...parts);
      };
      draw();
      box.append(row);
    }
    box.append(h('label.field', h('span.field-label', 'Takes per line'), h('input.input.input-sm', { type: 'number', min: 1, max: 6, value: sh.takes || 2, on: { change: e => { sh.takes = Math.max(1, Math.min(6, Number(e.target.value) || 2)); this._save(); } } })),
      h('div.snd-tip', 'Emotion goes to the model’s direction (Qwen3-TTS instruct). Thai lines are routed to OmniVoice when the cast model has no Thai.'));
    return box;
  }
  // ------------------------------------------------------------ rows
  addRow(sh, r, render = true) {
    const row = sh.kind === 'lines' ? { id: uid(), event: '', character: '', line: '', emotion: '', language: '', ...r } : { id: uid(), event: '', prompt: '', seconds: null, ...r };
    sh.rows.push(row); this._save();
    if (render) this.mount();
    return row;
  }
  rowEvent(sh, r, i) {
    if (r.event) return slug(r.event);
    if (sh.kind === 'lines') return `${slug(r.character || 'line')}_${String(i + 1).padStart(3, '0')}`;
    return slug(r.prompt).split('_').slice(0, 4).join('_') || `event_${i + 1}`;
  }
  rowTakes(sh, r) { return (this.takes.get(sh.id)?.get(r.id) || []).slice().sort((a, b) => (b.rating || 0) - (a.rating || 0) || (a.created < b.created ? -1 : 1)); }
  allTakes(sh) { return sh.rows.flatMap(r => this.rowTakes(sh, r)); }
  renderRows() {
    const st = this.st; const sh = this.sheet();
    st.centerBody.replaceChildren();
    if (!sh) {
      st.headTitle.replaceChildren(h('div', { style: { fontWeight: 650 } }, 'Sheets'), h('div.dim', { style: { fontSize: '12px' } }, 'Batches of many assets'));
      st.centerBody.append(S.empty('list-checks', 'Batch many sounds at once', 'SFX event list: paste one event per line → N takes each → audition, rate, export <event>_<take>.ogg. Line sheet: character, line, emotion, language (CSV) → per-line takes in each character’s voice.',
        h('div.row.wrap', { style: { justifyContent: 'center' } }, btn('New SFX list', { icon: 'zap', kind: 'primary', onClick: () => this.newSheet('sfx') }), btn('New line sheet', { icon: 'mic', onClick: () => this.newSheet('lines') }))));
      return;
    }
    const all = this.allTakes(sh);
    const done = sh.rows.filter(r => this.rowTakes(sh, r).length).length;
    st.headTitle.replaceChildren(h('div', { style: { fontWeight: 650 } }, sh.name), h('div.dim', { style: { fontSize: '12px' } }, `${sh.rows.length} rows · ${done} with takes · ${all.length} takes · ${all.filter(a => (a.rating || 0) > 0 || a.favorite).length} picked`));
    st.headTools.replaceChildren(sh.kind === 'sfx' ? btn('✨ Improve all', { size: 'sm', onClick: () => this.improveAll(sh) }) : btn('✨ Polish lines', { size: 'sm', onClick: () => this.improveAll(sh) }));
    const rows = h('div.snd-sheet-rows');
    sh.rows.forEach((r, i) => rows.append(this.rowEl(sh, r, i)));
    if (!sh.rows.length) rows.append(S.empty(sh.kind === 'lines' ? 'mic' : 'zap', 'Empty sheet', sh.kind === 'lines' ? 'Paste CSV: character, line, emotion, language (a header row is optional).' : 'Paste one event per line: “door_open, heavy wooden door creaks open” or just the prompt.',
      h('div.row.wrap', { style: { justifyContent: 'center' } }, btn('Paste CSV / list', { icon: 'list', kind: 'primary', onClick: () => this.pasteModal(sh) }),
        btn(sh.kind === 'lines' ? 'Write lines for a character' : 'From a design doc', { icon: 'pen-line', onClick: () => this.writeRows(sh) }))));
    st.centerBody.append(rows);
  }
  rowEl(sh, r, i) {
    const set = (k, v) => { r[k] = v; this._save(); };
    const inp = (k, ph, cls = '') => h(`input.input.input-sm${cls}`, { value: r[k] ?? '', placeholder: ph, 'aria-label': ph, on: { input: e => set(k, e.target.value) } });
    const takes = this.rowTakes(sh, r);
    const job = this.rowJobs.get(r.id);
    const live = job && !['done', 'failed', 'cancelled'].includes(job.status);
    const el = h(`div.snd-row.${sh.kind}`);
    const acts = h('div.acts',
      btn('', { size: 'sm', kind: 'ghost', icon: 'wand-sparkles', title: '✨ Improve this row', onClick: () => this.improveRow(sh, r).then(() => this.renderRows()) }),
      btn('', { size: 'sm', icon: live ? 'loader' : 'sparkles', title: 'Generate takes for this row', onClick: () => this.generateRow(sh, r, i) }),
      btn('', { size: 'sm', kind: 'ghost', icon: 'trash-2', title: 'Remove row (takes stay in the library)', onClick: () => { sh.rows = sh.rows.filter(x => x.id !== r.id); this._save(); this.renderRows(); } }));
    if (sh.kind === 'sfx') {
      el.append(h('div.rn', i + 1), inp('event', 'event name'), textarea('textarea.textarea.f-prompt', { rows: 1, placeholder: 'prompt (Thai ok, ✨ to improve)', 'aria-label': 'prompt', on: { input: e => set('prompt', e.target.value) } }, r.prompt),
        h('input.input.input-sm', { type: 'number', step: 0.5, min: 0.5, max: 60, value: r.seconds ?? '', placeholder: `${sh.seconds || 2}s`, title: 'Length (s)', 'aria-label': 'seconds', on: { input: e => set('seconds', e.target.value ? Number(e.target.value) : null) } }),
        h('input.input.input-sm', { type: 'number', min: 1, max: 8, value: r.takes ?? '', placeholder: `${sh.takes || 3}×`, title: 'Takes', 'aria-label': 'takes', on: { input: e => set('takes', e.target.value ? Number(e.target.value) : null) } }), acts);
    } else {
      const langs = ['', ...Object.keys(LANG_NAMES)];
      el.append(h('div.rn', i + 1), h('input.input.input-sm', { value: r.character || '', placeholder: 'character', 'aria-label': 'character', list: 'snd-chars', on: { input: e => set('character', e.target.value), change: () => this.mount() } }),
        textarea('textarea.textarea.f-line', { rows: 1, placeholder: 'line to speak', 'aria-label': 'line', on: { input: e => set('line', e.target.value) } }, r.line),
        inp('emotion', 'emotion / direction'),
        h('select.select', { 'aria-label': 'language', on: { change: e => set('language', e.target.value) } }, ...langs.map(l => h('option', { value: l, selected: (r.language || '') === l }, l || 'auto'))),
        h('input.input.input-sm', { type: 'number', min: 1, max: 6, value: r.takes ?? '', placeholder: `${sh.takes || 2}×`, title: 'Takes', 'aria-label': 'takes', on: { input: e => set('takes', e.target.value ? Number(e.target.value) : null) } }), acts);
    }
    const stat = h('div.stat', h('span.mono', this.rowEvent(sh, r, i)),
      live ? h('span.badge.accent', ic('loader', 11, 'spin'), job.status === 'queued' ? 'queued' : `${Math.round((job.progress?.frac || 0) * 100)}% ${job.progress?.msg || ''}`) : null,
      job?.status === 'failed' ? h('span', { style: { color: 'var(--err)' }, title: job.error?.fix || '' }, ic('circle-alert', 12), ' ', job.error?.message) : null,
      r.warn ? h('span', { style: { color: 'var(--warn)' } }, ic('triangle-alert', 12), ' ', r.warn) : null);
    el.append(stat);
    if (takes.length) el.append(h('div.takes', ...takes.map(a => miniTake(a, x => S.openAsset(x, { list: takes.map(t => t.id) })))));
    return el;
  }
  // ------------------------------------------------------------ jobs
  sfxSpec(sh, r, i) {
    const model = store.modelsById.get(sh.model) || S.modelsFor(sh.task || 'sfx').find(m => m.status.status === 'ready');
    if (!model) throw new Error('No ready model for this sheet.');
    if (!r.prompt?.trim()) throw new Error('This row has no prompt.');
    const ev = this.rowEvent(sh, r, i);
    const params = { prompt: r.prompt.trim(), batch: r.takes || sh.takes || 3, seed: -1 };
    if (paramOf(model, 'seconds')) params.seconds = r.seconds || sh.seconds || 2;
    if (r.negative) params.negative = r.negative;
    return { task: sh.task || 'sfx', model: model.id, params, project: store.project, title: ev,
      meta: { studio: 'sound', prompt: r.original || r.prompt, improved_prompt: r.original ? r.prompt : null, tags: ['sheet'], asset_extra: { x_event: ev, x_sheet: { sheet: sh.id, kind: 'sfx', row: r.id, event: ev, name: sh.name } } } };
  }
  lineSpec(sh, r, i) {
    const text = (r.line || '').trim();
    if (!text) throw new Error('This row has no line.');
    const thai = isThaiText(text) || r.language === 'th';
    const cast = sh.cast[(r.character || '').trim()] || { kind: 'preset' };
    let task, model, inputs = [], params = { text, batch: r.takes || sh.takes || 2, seed: -1 };
    r.warn = '';
    if (cast.kind === 'voice' && cast.voice) {
      task = 'voice_clone';
      model = store.modelsById.get(cast.model) || store.modelsById.get(store.defaults.voice_clone) || store.modelsById.get('omnivoice');
      if (thai && !thaiOk(model)) { r.warn = `${model.name} has no Thai: using OmniVoice`; model = store.modelsById.get('omnivoice'); }
      const v = this.voices.find(x => x.id === cast.voice);
      inputs = [{ asset: cast.voice, role: 'ref' }];
      if (paramOf(model, 'ref_text')) params.ref_text = voiceRefText(v);
      if (r.emotion && paramOf(model, 'instruct')) params.instruct = r.emotion;
    } else {
      task = 'tts';
      model = store.modelsById.get(cast.model) || store.modelsById.get(store.defaults.tts) || S.modelsFor('tts').find(m => m.status.status === 'ready');
      if (thai && !thaiOk(model)) { r.warn = `${model.name} has no Thai: using OmniVoice (auto voice)`; model = store.modelsById.get('omnivoice'); }
      if (paramOf(model, 'speaker') && cast.speaker) params.speaker = cast.speaker;
      if (paramOf(model, 'voice') && cast.speaker && (paramOf(model, 'voice').choices || []).includes(cast.speaker)) params.voice = cast.speaker;
      if (r.emotion && paramOf(model, 'instruct')) params.instruct = r.emotion;
    }
    if (!model || model.status.status !== 'ready') throw new Error(`${model?.name || 'The voice model'} is not ready.`);
    const lang = langFor(model, r.language || (thai ? 'th' : null));
    if (lang !== undefined) params.language = lang;
    const ev = this.rowEvent(sh, r, i);
    return { task, model: model.id, params, inputs, project: store.project, title: `${r.character || 'line'}: ${text.slice(0, 50)}`,
      meta: { studio: 'sound', prompt: text, tags: ['sheet', r.character].filter(Boolean),
        asset_extra: { x_event: ev, x_sheet: { sheet: sh.id, kind: 'lines', row: r.id, event: ev, character: r.character, line: text, emotion: r.emotion, language: r.language, name: sh.name } } } };
  }
  async generateRow(sh, r, i, quiet = false) {
    let spec;
    try { spec = sh.kind === 'lines' ? this.lineSpec(sh, r, i) : this.sfxSpec(sh, r, i); } catch (e) { if (!quiet) toast(e.message, { kind: 'warn' }); r.warn = e.message; this._rowsDirty(); return null; }
    try {
      const j = await api.submit(spec);
      this.rowJobs.set(r.id, j);
      const off = bus.on('job:' + j.id, jj => { this.rowJobs.set(r.id, jj); if (['done', 'failed', 'cancelled'].includes(jj.status)) off(); this._rowsDirty(); });
      this._rowsDirty();
      return j;
    } catch (e) { r.warn = e.message; this._rowsDirty(); if (!quiet) toastError(e); return null; }
  }
  async generateAll(sh, all) {
    const todo = sh.rows.map((r, i) => [r, i]).filter(([r]) => all || !this.rowTakes(sh, r).length).filter(([r]) => !['queued', 'running'].includes(this.rowJobs.get(r.id)?.status));
    if (!todo.length) return toast('Every row already has takes. Use “Regenerate all” for new ones.', { kind: 'info' });
    let n = 0;
    for (const [r, i] of todo) if (await this.generateRow(sh, r, i, true)) n++;
    this._save();
    toast(`${n} job(s) queued${n < todo.length ? `, ${todo.length - n} skipped (see the row warnings)` : ''}. They run one after another on the GPU.`, { kind: n ? 'ok' : 'warn', title: sh.name });
  }
  async improveRow(sh, r) {
    try {
      if (sh.kind === 'sfx') {
        const m = store.modelsById.get(sh.model);
        const res = await api.improve({ text: r.original || r.prompt || r.event, model_id: m?.id, task: sh.task || 'sfx', n: 0, chips: prefs.get('chips.sound', []), project: S.store.project });
        if (!r.original) r.original = r.prompt || r.event;
        r.prompt = res.prompt; if (res.negative) r.negative = res.negative;
      } else {
        const res = await api.improve({ text: `${r.character ? r.character + ': ' : ''}${r.line}${r.emotion ? ` (${r.emotion})` : ''}`, target_style: 'qwen_tts', n: 0, project: S.store.project });
        if (res.line) r.line = res.line;
        if (res.prompt && !r.emotion) r.emotion = res.prompt;
      }
      this._save();
    } catch (e) { toastError(e, 'Improve failed'); }
  }
  async improveAll(sh) {
    const t = toast(`Improving ${sh.rows.length} rows…`, { kind: 'info', timeout: 0 });
    for (const r of sh.rows) { await this.improveRow(sh, r); this.renderRows(); }
    t.remove(); toast('Rows improved. Check them, then Generate.', { kind: 'ok' });
  }
  // ✍ Write helper (app/web/core/components/writer.js): rows written by a text backend, inserted into this sheet
  writeRows(sh) {
    const done = n => { this._save(); this.mount(); toast(`${n} row(s) added to ${sh.name}.`, { kind: 'ok' }); };
    if (sh.kind === 'lines') {
      const chars = [...new Set(sh.rows.map(r => (r.character || '').trim()).filter(Boolean))];
      S.writeHelper('barks', { input: chars.length === 1 ? { character: chars[0] } : {}, onInsert: items => {
        for (const r of items) this.addRow(sh, { character: r.character || '', line: r.line, emotion: r.emotion || '', language: LANG_NAMES[r.language] ? r.language : '' }, false);
        done(items.length);
      } });
    } else {
      S.writeHelper('sfx_events', { onInsert: items => {
        for (const e of items) this.addRow(sh, { event: slug(e.event || e.prompt), prompt: e.prompt, seconds: Number(e.seconds) > 0 ? Math.min(60, Number(e.seconds)) : null }, false);
        done(items.length);
      } });
    }
  }
  /** Rows sent from another page (Assistant → "Send to Sound"): {kind: 'lines'|'sfx', rows, name}. */
  async importRows({ kind, rows, name }) {
    if (!this.loaded) await this.load();
    kind = kind === 'lines' ? 'lines' : 'sfx';
    const id = uid();
    this.data.sheets[id] = kind === 'lines'
      ? { id, kind, name: name || 'Written lines', rows: [], cast: {}, takes: 2, created: new Date().toISOString() }
      : { id, kind, name: name || 'SFX from design doc', rows: [], task: 'sfx', model: null, takes: 3, seconds: 2, created: new Date().toISOString() };
    const sh = this.data.sheets[id];
    for (const r of rows || []) {
      if (kind === 'lines') this.addRow(sh, { character: r.character || '', line: r.line || '', emotion: r.emotion || '', language: LANG_NAMES[r.language] ? r.language : '' }, false);
      else this.addRow(sh, { event: slug(r.event || r.name || r.prompt || 'event'), prompt: r.prompt || '', seconds: Number(r.seconds) > 0 ? Math.min(60, Number(r.seconds)) : null }, false);
    }
    this.cur = id; prefs.set('sound.sheet', id); this._save(); this.mount();
    toast(`${sh.rows.length} row(s) in the new sheet “${sh.name}”.`, { kind: 'ok' });
  }
  pasteModal(sh) {
    const ta = h('textarea.textarea', { rows: 10, placeholder: sh.kind === 'lines' ? 'character,line,emotion,language\nKnight,"Hold the line!",determined,en\nElder,ยินดีต้อนรับ นักเดินทาง,warm,th' : 'door_open, heavy wooden door creaks open slowly\nui_click, short soft UI click\nsword hits iron armor', style: { fontFamily: 'var(--mono)', fontSize: '12.5px' } });
    const replace = S.toggle(false, () => {}, 'Replace rows');
    const md = S.modal({ title: sh.kind === 'lines' ? 'Paste a line sheet (CSV / TSV)' : 'Paste an event list', icon: 'list', width: 620,
      body: h('div.col', { style: { gap: '10px' } }, h('div.snd-tip', sh.kind === 'lines' ? 'Columns: character, line, emotion, language (optional: id). Copy straight from a spreadsheet (tab-separated works too). A header row is detected.' : 'One event per line: “name, prompt[, seconds]” or just the prompt (a name is derived).'), ta,
        h('div.row', h('span.grow', 'Replace existing rows'), replace)),
      foot: [btn('Cancel', { kind: 'ghost', onClick: () => md.close() }), btn('Add rows', { kind: 'primary', icon: 'plus', onClick: () => {
        const rows = parseCSV(ta.value);
        if (!rows.length) return;
        const head = rows[0].map(x => x.toLowerCase());
        const hasHead = sh.kind === 'lines' ? head.includes('line') || head.includes('character') : head.includes('prompt') || head.includes('event');
        const idx = name => head.indexOf(name);
        const body = hasHead ? rows.slice(1) : rows;
        if (replace.input.checked) sh.rows = [];
        for (const c of body) {
          if (sh.kind === 'lines') {
            const g = (k, d) => (hasHead && idx(k) >= 0 ? c[idx(k)] : c[d]) || '';
            if (!g('line', 1) && c.length === 1) { this.addRow(sh, { line: c[0] }, false); continue; }
            this.addRow(sh, { character: g('character', 0), line: g('line', 1), emotion: g('emotion', 2), language: (g('language', 3) || '').toLowerCase().slice(0, 2), event: hasHead && idx('id') >= 0 ? c[idx('id')] : '' }, false);
          } else if (c.length === 1) this.addRow(sh, { prompt: c[0], event: slug(c[0]).split('_').slice(0, 4).join('_') }, false);
          else {
            const g = (k, d) => (hasHead && idx(k) >= 0 ? c[idx(k)] : c[d]) || '';
            const sec = Number(g('seconds', 2));
            this.addRow(sh, { event: g('event', 0), prompt: g('prompt', 1), seconds: sec > 0 ? sec : null }, false);
          }
        }
        this._save(); md.close(); this.mount();
        toast(`${body.length} row(s) added.`, { kind: 'ok' });
      } })] });
    setTimeout(() => ta.focus(), 40);
  }
  exportSheet(sh) {
    const all = this.allTakes(sh);
    const picked = sh.rows.flatMap(r => { const t = this.rowTakes(sh, r); const p = t.filter(a => (a.rating || 0) > 0 || a.favorite); return p; });
    if (!all.length) return toast('No takes yet: generate first.', { kind: 'warn' });
    const md = S.modal({ title: `Export ${sh.name}`, icon: 'package', width: 440, body: h('div.col', { style: { gap: '10px' } },
      h('p', { style: { margin: 0 } }, `${picked.length} picked take(s) (rated or favourite) of ${all.length}.`),
      h('div.snd-tip', 'Rate the takes you want (★) — the best rated becomes take 01 of each event.')),
    foot: [btn(`All takes (${all.length})`, { onClick: () => { md.close(); exportModal(all.map(a => a.id), { folder: slug(sh.name) }); } }),
      btn(`Picked (${picked.length})`, { kind: 'primary', disabled: !picked.length, onClick: () => { md.close(); exportModal(picked.map(a => a.id), { folder: slug(sh.name) }); } })] });
  }
}

// ------------------------------------------------------------------ module
export default S.defineStudio({
  id: 'sound', title: 'Sound', icon: 'audio-lines', color: 'var(--c-sound)', order: 30, mobile: true, short: 'Sound',
  receives: [
    { action: 'variations', label: 'Variations', icon: 'shuffle', types: ['audio'] },
    { action: 'voice_ref', label: 'Use as voice', icon: 'mic', types: ['audio', 'voice'] },
    { action: 'extend', label: 'Extend', icon: 'arrow-right', types: ['audio'] },
    { action: 'convert', label: 'Convert voice', icon: 'arrow-left-right', types: ['audio', 'voice'] },
    { action: 'reply', label: 'Voice reply', icon: 'message-circle', types: ['audio', 'voice'] },
    { action: 'v2a', label: 'Add sound', icon: 'clapperboard', types: ['video'] },
    { action: 'subtitles', label: 'Subtitles', icon: 'type', types: ['audio', 'voice', 'video'] },
    { action: 'separate', label: 'Split stems', icon: 'layers', types: ['audio', 'voice', 'video'] },
    { action: 'dub', label: 'Dub', icon: 'film', types: ['audio', 'voice', 'video'] },
    { action: 'loop', label: 'Loop', icon: 'repeat', types: ['audio'] },
    { action: 'normalize', label: 'Normalize', icon: 'gauge', types: ['audio', 'voice'] },
    { action: 'post', label: 'Audio tools', icon: 'wand', types: ['audio', 'voice'] },
  ],
  mount(el) {
    if (!document.getElementById('snd-css')) document.head.append(h('style#snd-css', CSS));
    el.style.height = '100%';
    const st = new SoundStudio(el);
    st.sheets.load();
    return { route: r => st.route(r), receive: p => st.receive(p), show: () => st.show() };
  },
});
