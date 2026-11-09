"""Use-case profiles and export presets (docs/CONTRACTS.md "Profiles and export presets").

A profile is a bundle of defaults for one kind of production: which style chips come first (`profiles: [...]` tags in
prompts/*.yaml), the default aspect ratio of image / video models, and the default export preset applied when assets
are downloaded or zipped. Profiles never change what a model can do and never touch stored assets; export presets only
ever produce *derived* files (see media.export_audio / media.export_image and AssetStore.render / export).

`general` is the neutral default (no aspect override, original files on export); `game` keeps the behaviour the studio
always had (RGBA sprites, seamless loops, GLB) and also exports the original files.
"""
from __future__ import annotations

PROFILE_IDS = ("game", "video", "podcast", "social", "film", "general")
DEFAULT_PROFILE = "general"

# Audio options understood by media.export_audio:
#   format: wav | mp3 | ogg | flac        sample_rate: Hz | None (keep)     bit_depth: 16 | 24 | 32 (wav / flac) | None
#   lufs: integrated loudness target (two-pass ffmpeg loudnorm) | None        true_peak: dBTP ceiling (default -1.0)
#   channels: 1 | 2 | None (keep)         bitrate_kbps: mp3 (default 192)      music_sample_rate: used instead of
#   sample_rate for music / song assets (CD-style 44.1 kHz in a 48 kHz video preset)
# Image options understood by media.export_image:
#   format: png | jpeg | webp             width / height: exact output size (with `aspect` or both given, cropped)
#   max_side: longest side after crop     aspect: "16:9" | "9:16" | "1:1" ...  fit: crop | pad
#   quality: 1..100 (jpeg / webp)         background: "#rrggbb" used to flatten alpha for JPEG
EXPORT_PRESETS: dict[str, dict] = {
    "original": {"label": "Original files", "summary": "No conversion: the files exactly as stored.", "audio": None, "image": None},
    "video_online": {
        "label": "Online video (-14 LUFS)", "summary": "YouTube / web video: 48 kHz 24-bit WAV at -14 LUFS, 1920x1080 PNG.",
        "audio": {"format": "wav", "sample_rate": 48000, "bit_depth": 24, "lufs": -14.0, "true_peak": -1.0, "music_sample_rate": 44100},
        "image": {"format": "png", "aspect": "16:9", "width": 1920, "height": 1080, "fit": "crop"}},
    "broadcast": {
        "label": "Broadcast (-23 LUFS, EBU R128)", "summary": "TV / broadcast: 48 kHz 24-bit WAV at -23 LUFS, 1920x1080 PNG.",
        "audio": {"format": "wav", "sample_rate": 48000, "bit_depth": 24, "lufs": -23.0, "true_peak": -1.0},
        "image": {"format": "png", "aspect": "16:9", "width": 1920, "height": 1080, "fit": "crop"}},
    "film_master": {
        "label": "Film / animation master (-23 LUFS)", "summary": "Post-production: 48 kHz 24-bit WAV at -23 LUFS, source-size PNG.",
        "audio": {"format": "wav", "sample_rate": 48000, "bit_depth": 24, "lufs": -23.0, "true_peak": -1.0},
        "image": {"format": "png"}},
    "podcast": {
        "label": "Podcast (-16 LUFS, MP3)", "summary": "Spotify / Apple style: 44.1 kHz MP3 at -16 LUFS, 3000x3000 JPEG cover art.",
        "audio": {"format": "mp3", "sample_rate": 44100, "lufs": -16.0, "true_peak": -1.0, "bitrate_kbps": 192},
        "image": {"format": "jpeg", "aspect": "1:1", "width": 3000, "height": 3000, "fit": "crop", "quality": 92}},
    "social_vertical": {
        "label": "Social vertical (-14 LUFS, 9:16)", "summary": "Reels / Shorts / TikTok: 48 kHz MP3 at -14 LUFS, 1080x1920 JPEG.",
        "audio": {"format": "mp3", "sample_rate": 48000, "lufs": -14.0, "true_peak": -1.0, "bitrate_kbps": 192},
        "image": {"format": "jpeg", "aspect": "9:16", "width": 1080, "height": 1920, "fit": "crop", "quality": 90}},
    "social_square": {
        "label": "Social square (-14 LUFS, 1:1)", "summary": "Feed posts: 48 kHz MP3 at -14 LUFS, 1080x1080 JPEG.",
        "audio": {"format": "mp3", "sample_rate": 48000, "lufs": -14.0, "true_peak": -1.0, "bitrate_kbps": 192},
        "image": {"format": "jpeg", "aspect": "1:1", "width": 1080, "height": 1080, "fit": "crop", "quality": 90}},
    "thumbnail": {
        "label": "Thumbnail (16:9 JPEG)", "summary": "Video thumbnail: 1280x720 JPEG; audio is left unchanged.",
        "audio": None, "image": {"format": "jpeg", "aspect": "16:9", "width": 1280, "height": 720, "fit": "crop", "quality": 90}},
    "music_cd": {
        "label": "Music (44.1 kHz FLAC)", "summary": "Lossless music delivery: 44.1 kHz 16-bit FLAC, loudness untouched.",
        "audio": {"format": "flac", "sample_rate": 44100, "bit_depth": 16}, "image": None},
    "web_images": {
        "label": "Web images (WebP)", "summary": "Images as 2048 px WebP; audio is left unchanged.",
        "audio": None, "image": {"format": "webp", "max_side": 2048, "quality": 88}},
}

PROFILES: dict[str, dict] = {
    "general": {
        "label": "General", "icon": "layers",
        "summary": "Everything on, nothing assumed: pick any studio and any format.",
        "aspect": None, "aspects": ["1:1", "16:9", "9:16", "4:3"], "export": "original",
        "export_presets": ["original", "video_online", "podcast", "social_vertical", "thumbnail", "music_cd", "web_images"],
        "hints": {}},
    "game": {
        "label": "Game", "icon": "gamepad-2",
        "summary": "Game assets: transparent RGBA sprites, seamless loops, GLB props. Exports the original files.",
        "aspect": None, "aspects": ["1:1", "4:3", "3:4", "16:9"], "export": "original",
        "export_presets": ["original", "music_cd", "web_images", "video_online"],
        "hints": {"rgba": True, "loop": True, "mesh_format": "glb"}},
    "video": {
        "context": "Use case: online video production (b-roll stills, thumbnails, stings, beds, voice-over).",
        "label": "Video", "icon": "clapperboard",
        "summary": "Online video: 16:9 stills and thumbnails, b-roll, stings and beds, loudness -14 LUFS at 48 kHz.",
        "aspect": "16:9", "aspects": ["16:9", "1:1", "9:16"], "export": "video_online",
        "export_presets": ["video_online", "broadcast", "thumbnail", "original"],
        "hints": {"video_aspect": "16:9"}},
    "podcast": {
        "context": "Use case: podcast production (cover art, intro and outro stings, music beds, voices).",
        "label": "Podcast", "icon": "mic",
        "summary": "Podcasts: square cover art, intro / outro stings and beds, voices; -16 LUFS MP3 at 44.1 kHz.",
        "aspect": "1:1", "aspects": ["1:1", "16:9"], "export": "podcast",
        "export_presets": ["podcast", "music_cd", "original"],
        "hints": {"video_aspect": "16:9"}},
    "social": {
        "context": "Use case: short-form social video and ads (vertical 9:16 first, hooks, jingles).",
        "label": "Social", "icon": "smartphone",
        "summary": "Short-form and ads: vertical 9:16 first, jingles and hooks; -14 LUFS MP3, 1080x1920 JPEG.",
        "aspect": "9:16", "aspects": ["9:16", "1:1", "16:9"], "export": "social_vertical",
        "export_presets": ["social_vertical", "social_square", "thumbnail", "original"],
        "hints": {"video_aspect": "9:16"}},
    "film": {
        "context": "Use case: film and animation pre-production and post (storyboards, concept art, temp score, foley).",
        "label": "Film & animation", "icon": "film",
        "summary": "Storyboards, concept art, temp score and foley; 16:9, -23 LUFS 48 kHz 24-bit WAV masters.",
        "aspect": "16:9", "aspects": ["16:9", "4:3", "1:1"], "export": "film_master",
        "export_presets": ["film_master", "broadcast", "video_online", "original"],
        "hints": {"video_aspect": "16:9"}},
}

MUSIC_TASKS = ("music", "song")


def profile_id(settings) -> str:
    p = settings.get("profile", default=DEFAULT_PROFILE) if settings is not None else DEFAULT_PROFILE
    return p if p in PROFILES else DEFAULT_PROFILE


def default_export_preset(settings) -> str:
    over = settings.get("export", "preset") if settings is not None else None
    if over in EXPORT_PRESETS:
        return over
    return PROFILES[profile_id(settings)]["export"]


def public(settings=None) -> dict:
    """JSON for GET /api/profiles."""
    return {"current": profile_id(settings), "default_export": default_export_preset(settings),
            "profiles": [{"id": k, **v} for k, v in PROFILES.items()],
            "export_presets": [{"id": k, **v} for k, v in EXPORT_PRESETS.items()]}


def resolve_export(spec: dict | None, settings=None) -> dict:
    """Normalise an export request body to {preset, audio, image}: a named preset and/or explicit option dicts
    (explicit keys win over the preset's). Unknown preset -> ValueError. 'original' -> both None."""
    spec = dict(spec or {})
    name = spec.get("preset") or (default_export_preset(settings) if "audio" not in spec and "image" not in spec else None)
    base = {"audio": None, "image": None}
    if name:
        if name not in EXPORT_PRESETS:
            raise ValueError(f"Unknown export preset '{name}'.")
        base = {"audio": EXPORT_PRESETS[name].get("audio"), "image": EXPORT_PRESETS[name].get("image")}
    out = {"preset": name}
    for kind in ("audio", "image"):
        o = spec.get(kind)
        if o is None and kind not in spec:
            out[kind] = dict(base[kind]) if base[kind] else None
        elif o is None or o is False or o == {}:
            out[kind] = None            # explicit "do not convert this kind"
        else:
            out[kind] = {**(base[kind] or {}), **o}
    mat = spec.get("material")          # per-type options for material assets (layout / normal / size / glb); never part of a preset
    out["material"] = dict(mat) if isinstance(mat, dict) and mat else None
    return out
