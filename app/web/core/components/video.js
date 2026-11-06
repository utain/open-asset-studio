// <video> players for video assets and for the muxed `take_N_video.mp4` attachments of video-to-sound takes.
import { fileUrl, thumbUrl } from '../api.js';
import { h, ic, fmtBytes } from '../ui.js';
import { player } from './audio.js';

const VIDEO_NAME = /\.(mp4|webm|mov|mkv)$/i;

/** The attached video files of an asset (role attachment), e.g. [{name: 'take_1_video.mp4', bytes, mime}]. */
export function videoAttachments(asset) {
  return (asset?.files || []).filter(f => f.role !== 'main' && VIDEO_NAME.test(f.name));
}

/**
 * videoBox(asset, {name, poster, compact}) -> element: a <video controls> of the asset's main file (type video) or of
 * the attachment `name`, with a download link. Starting it stops the shared audio player.
 */
export function videoBox(asset, { name = null, poster = null, compact = false } = {}) {
  const file = name || asset.main;
  const meta = (asset.files || []).find(f => f.name === file);
  const v = h('video.snd-video', { src: fileUrl(asset, file), controls: true, preload: 'metadata', playsInline: true, 'aria-label': `Video ${file}` });
  const p = poster ?? (asset.type === 'video' ? thumbUrl(asset) : null);
  if (p) v.poster = p;
  v.addEventListener('play', () => player.stop());
  return h(`div.video-box${compact ? '.compact' : ''}`, v,
    h('div.video-bar', h('span.dim.ellipsis', { title: file }, ic('video', 12), ' ', file, meta?.bytes ? ` · ${fmtBytes(meta.bytes)}` : ''),
      h('a.btn.btn-sm.btn-ghost', { href: fileUrl(asset, file, true), title: 'Download the video' }, ic('download', 13), ' Download')));
}
