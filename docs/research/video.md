# Video studio: models, install, measurements

Group "video" of the media expansion: a new **Video** studio (`app/web/studios/video.js`) for open-weight video generation.
Registry `models.d/video.yaml`, backend `video` (`envs/video`, `workers/video_worker.py`), CPU tools in `app/plugins/video.py`,
prompt styles `prompts/video.yaml`, tests `tests/test_video.py`, install recipe `install/video.yaml`, lock `env-lock/video.txt`.

Tasks: `t2v` (text to video), `i2v` (image to video), `flf2v` (first + last frame), `interpolate` (frame doubling of an existing clip).
All outputs are h264 / yuv420p / `+faststart` mp4 (AAC audio when the model makes sound), with a poster from the core.
Measured on an NVIDIA GB10 (128 GB unified memory, aarch64, torch 2.9.0 cu130, diffusers 0.42.1, SDPA attention). Every number
below comes from a real run through the worker protocol (`workers/video_worker.py`), not from model cards.

## 1. Models

| model id | what | licence | commercial | download | status |
|---|---|---|---|---|---|
| `wan21-t2v-1.3b` | Wan 2.1 T2V 1.3B, fast preview (480p, 16 fps, silent) | Apache-2.0 | yes | 5.8 GB (+ the 5B repo's text encoder) | ready, tested |
| `wan22-ti2v-5b` | Wan 2.2 TI2V 5B, quality t2v + i2v (up to 720p / 24 fps, silent) | Apache-2.0 | yes | 32 GB | ready, tested (t2v and i2v) |
| `ltx23-distilled` | LTX-2.3 distilled, t2v + i2v + flf2v, **with sound** | LTX Video 2 Open Source License (free below USD 10 M revenue, `verify: true`) | conditional | 89 GB | ready, tested (t2v, flf2v) |
| `kandinsky6-lite-distill` | Kandinsky 6.0 Lite distill, t2v + i2v, **with sound**, 5 s | MIT | yes | 27 GB | ready, tested (t2v + audio) |
| `hunyuan15-480p-i2v-distilled` | HunyuanVideo 1.5 480p image to video, step-distilled (12 steps), silent | Tencent HunyuanVideo 1.5 Community (not valid in EU / UK / South Korea, `verify: true`) | conditional | 33 GB | ready, tested (i2v) |
| `rife-v4` | RIFE v4 frame interpolation (x2, any clip), pure PyTorch port | MIT | yes | 12 MB | ready, tested |

None of these repositories is gated: no Hugging Face login is needed. Total disk for everything above: 183 GiB of weights (192 GB) plus the 4.6 GB venv.

Defaults per task: `t2v` Wan 2.1 1.3B (fast), `i2v` Wan 2.2 5B, `flf2v` LTX-2.3, `interpolate` RIFE.

### Measurements (one run each, GB10, shared with other jobs at times)

| model / task | clip | steps | load | generate | peak memory | notes |
|---|---|---|---|---|---|---|
| wan21-t2v-1.3b t2v | 3 s, 848x480, 16 fps (49 frames) | 30 | 18 s | 197 s | 22 GB | hummingbird prompt, picture clean |
| wan22-ti2v-5b t2v | 2 s, 864x480, 24 fps (49 frames) | 10 | 42 s | 60 s | 31.6 GB | VAE decode ~25 s of that |
| wan22-ti2v-5b i2v | 2 s, 1280x704, 24 fps (49 frames) | 10 | (loaded) | 142 s | 42.1 GB | last frame of the t2v clip as first frame; motion follows the prompt |
| kandinsky6-lite-distill t2v + audio | 2 s, 848x480, 24 fps (49 frames) | 16 | 124 s (first load from disk) | 136 s | 39.8 GB | tiled VAE decode; 44.1 kHz AAC, mean volume -24 dB |
| hunyuan15-480p-i2v-distilled i2v | 2 s, 848x480, 24 fps (49 frames) | 12 | 190 s (first load) | 268 s | 57 GB | tiled VAE decode |
| ltx23-distilled t2v + audio | 2 s, 1248x704, 24 fps (49 frames) | 8 | 264 s (first load; 66 GB resident) | 212 s | 72.9 GB | tiled decode; 48 kHz AAC, kitchen sound. At 848x480 the picture had dark patterned side bars, so 480p is not offered |
| ltx23-distilled flf2v + audio | 2 s, 1024x576, 24 fps (49 frames) | 8 | 262 s | 161 s | 72.9 GB | first and last frame (fox clip) are matched exactly |
| rife-v4 interpolate | 2 s, 864x480: 49 -> 97 frames | - | ~3 s | seconds (not timed precisely) | < 4 GB | 24 -> 48 fps, h264 yuv420p |

Registry `est` / `memory_gb` / `speed` were updated from these numbers. Longer clips cost more than linearly (attention):
a 5 s Wan 2.2 clip at 704p takes several times the 2 s time.

## 2. Findings that matter

- **Tiled VAE decode is mandatory for Kandinsky 6 and HunyuanVideo 1.5.** With full-frame decode the worker process was
  killed (SIGKILL, exit code -9, no Python traceback) right after the last denoising step, on a 128 GB machine with ~110 GB free.
  The registry now defaults `vae_tiling: true` for those two models. Wan decodes fine without tiling up to 704p.
- **Wan 2.1 1.3B has no text encoder of its own in the repo we download**: the worker loads UMT5 + tokenizer from the
  Wan 2.2 5B repo (the same weights), which saves 21 GB. Both repos must therefore be downloaded (`./install.sh model wan21-t2v-1.3b`
  fetches the 5B repo's `text_encoder/*` and `tokenizer/*` through the registry `weights:` list).
- Components are loaded straight to the GPU (`device_map="cuda"`, no mmap); the Wan text encoder uses the safetensors-to-cuda
  fast path from `workers/qwen_worker.py`. The first load of a 20+ GB model is bound by disk read speed (2-3 minutes), later ones are cached.
- Progress: diffusers video pipelines have no common callback, so the worker replaces `pipe.progress_bar` with a bar that calls
  `ctx.check()` (cancel) and `ctx.progress()` per step. Cancelling works between steps; the VAE decode and the mp4 write are not interruptible.
- Frame counts follow each model's rule (4k+1 for Wan / Kandinsky / Hunyuan, 8k+1 for LTX) and sizes are rounded to the model's grid
  (16 px, LTX / Wan 2.2 32 px). `match_image` makes image-to-video keep the aspect of the input.
- Memory options: `offload` none / model / sequential (diffusers CPU offload) and `vae_tiling`. On this 128 GB unified-memory machine
  offload does not help and is off by default; it is there for 24-48 GB cards.
- Hunyuan 1.5 and Kandinsky: output frames come out at a multiple of 16 (864x480 input image -> 848x480 video).

## 3. Tools and studio

`app/plugins/video.py` (CPU, ffmpeg only, run beside GPU jobs): `video.frame` (first / last / at seconds -> a PNG image asset),
`video.trim`, `video.loop` (ping-pong / repeat), `video.speed`, `video.reframe` (crop / blur / pad to 9:16, 1:1, 16:9, 4:5, 4:3, 3:4),
`video.convert` (mp4 / webm), `video.concat`, `video.mux` (video + an audio asset, with volume and fade), and
`GET /api/x/video/formats`. All tested end to end through `POST /api/jobs`.

Studio: four tabs (Text, Image, First+Last, Smooth), a player stage, actions **Add sound** (`sendTo('sound', id, {action:'v2a'})`,
lands on the Sound studio's "Video to sound" with the clip in the video slot), **Continue** (takes the last frame and opens
Image-to-video), **Smooth** (RIFE), **Tools**, **Details**. A real headless Chrome run on port 7864 generated a 2 s clip
with the Fast preset through the UI, showed it in the player with poster, and "Add sound" opened the Sound studio with the clip
loaded in the slot, with no console errors.

## 4. LTX-2.3 and flf2v

`ltx23-distilled` is the only registered flf2v model (the worker passes first and last frame as
`LTX2VideoCondition(index=0 / -1)` to `LTX2ConditionPipeline`). It is also the only fast model with synchronised speech and sound.
Tested: t2v at 704p and 480p, and flf2v at 576p with the first and last frame of a Wan clip. The output reproduces both
frames and the motion in between, with generated sound. Needs `av` (PyAV) in the venv: `LTX2ConditionPipeline` re-compresses
condition images with H.264 CRF (as in the reference implementation) and raises an ImportError without it. LTX-2.3 is the
heaviest model here: 66 GB resident after load and 73 GB peak, so on a 24-48 GB card it only runs with `offload` and a lot of patience.
Its non-commercial limit is a revenue threshold (USD 10 M); the registry marks it `commercial: true` with `verify: true`.

## 5. Install

```bash
./install.sh video                      # envs/video from install/video.yaml (torch 2.9.0 cu130 + env-lock/video.txt)
./install.sh model wan21-t2v-1.3b       # one repo at a time; add wan22-ti2v-5b, kandinsky6-lite-distill, ...
```

The steps behind that, by hand:

```bash
uv venv --python 3.12 envs/video
VIRTUAL_ENV=envs/video uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.0 torchvision==0.24.0
VIRTUAL_ENV=envs/video uv pip install -r env-lock/video.txt
envs/video/bin/python -c "from diffusers import WanPipeline, LTX2ConditionPipeline, Kandinsky6TI2VAPipeline, HunyuanVideo15ImageToVideoPipeline"
```

`ffmpeg` / `ffprobe` must be on `PATH`. Weights are downloaded by the user, one repository at a time, never at runtime
(`HF_HUB_OFFLINE=1` in the worker). RIFE code is a re-implementation (`workers/video_rife.py`, MIT, see `THIRD_PARTY_NOTICES.md`);
no repository is cloned into `vendor/`.

## 6. What was left out, and why

- **Wan 2.2 A14B (T2V / I2V)**: two 14B experts, 117 GB per variant in the Diffusers repos (fp32), far beyond the budget; the 5B model covers the same tasks.
- **CogVideoX, Mochi 1, SkyReels, Allegro**: older or heavier for the quality they give; Apache-2.0 alternatives already cover the same slots.
- **HunyuanVideo 1.5 text-to-video and 720p variants**: the loader supports them (`HunyuanVideo15Pipeline`), the repos are ~35 GB each and were not downloaded; add a registry entry with `family: hunyuan15` to use them.
- **LTX-2.3 non-distilled (dev) weights**: 8-step distilled is the practical one; the worker also handles the dev pipeline (`features.distilled: false`).
- **Licences to read before shipping**: LTX (revenue threshold), Tencent (regions and 100 M MAU), Kandinsky's Qwen2.5-VL text encoder. The registry flags LTX and Hunyuan with `verify: true`.

## 7. Uncertainties

- Registry licence details for LTX-2.3 and HunyuanVideo 1.5 were taken from the model cards and the published licence summaries; verify the licence text for your use.
- Timings are single runs and the GPU was shared with other agents' jobs for part of the time; treat them as +-30 %.
- i2v was tested on Wan 2.2 and HunyuanVideo 1.5 only; Kandinsky 6 and LTX-2.3 were tested on t2v (LTX also on flf2v), Wan 2.1 on t2v. Their i2v code paths use the documented pipeline arguments but have not been run.
- 5 s / 720p runs were not measured for every model (only the 2-3 s clips above), the `est` values extrapolate.
