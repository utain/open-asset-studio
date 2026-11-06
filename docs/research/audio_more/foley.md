# Foley group: video-to-audio (`v2a`) and text SFX

Backend `foley` = `envs/foley` + `workers/foley_worker.py` + `models.d/foley.yaml` (one venv for all three models).
All three models ran end to end through the worker protocol (JSON lines) under the GPU lock. Test clip:
MMAudio's `examples/sora_beach.mp4` (5.0 s, 1080p-ish) plus our own ffmpeg `testsrc2` clip
(`docs/research/audio_more/foley_media/foley_testsrc.mp4`, 640x360, 24 fps, 6 s, no audio).

## Status

| model id | repo | status | licence | peak GPU | load / run | sanity |
|---|---|---|---|---|---|---|
| `mmaudio-large-44k-v2` (v2a + sfx) | hkchengrex/MMAudio (+ nvidia/bigvgan_v2_44khz_128band_512x, apple/DFN5B-CLIP-ViT-H-14-384) | working | weights CC-BY-NC 4.0 (commercial: false), code MIT | 5.8 GB (v2a), 4.7 GB (sfx) | load 31 s; 2 takes of 5 s in 9.6 s; sfx 4 s in 3.0 s | see below |
| `hunyuan-foley` (v2a only) | tencent/HunyuanVideo-Foley (XXL, + google/siglip2-base-patch16-512, laion/larger_clap_general) | working | Tencent community licence: not valid in EU / UK / South Korea; >100 M MAU needs a licence (commercial: true, verify: true) | 15.3 GB (13.8 GB resident) | load 31 s; 5 s clip, 50 steps: 21.5 s | see below |
| `thinksound` (v2a only) | FunAudioLLM/ThinkSound `thinksound_light.ckpt` (+ facebook/metaclip-h14-fullcc2.5b, google/t5-v1_1-xl) | working (light checkpoint only) | Apache-2.0 code, README says research / non-commercial only (commercial: false) | 10.5 GB | load 21 s; 5 s clip, 24 steps: 20 s incl. features | see below |

Sanity checks (no Whisper: these are not speech). Every take: no NaN, correct duration, not silent (silence fraction
below 2 %, except the sfx door creak 22 % and the synthetic testsrc video 73 %, both plausible). CLAP logits
(laion/clap-htsat-unfused) against the true prompt vs a wrong prompt on the beach clip, prompt "waves crashing on a
beach, seagulls" vs "a person typing on a keyboard": MMAudio 12.2 / -2.8 and 12.6 / -3.3, HunyuanVideo-Foley
5.1 / 1.0, ThinkSound 5.3 / -1.8. MMAudio text-only "a wooden door creaks open slowly": 7.7 vs 1.6 for "an explosion".
Muxed mp4s: h264 + aac, video and audio of equal length (ffprobe). No listening test was done by a human.

Not tested: cancel mid-run (MMAudio checks between takes only; Hunyuan and ThinkSound run a whole take
uninterruptibly, marked `interruptible: false`), videos longer than 15 s, non-mp4 containers (code path uses PyAV and ffmpeg, so
webm/mov/mkv should work), portrait video, and `mask_video`.

## What the UI needs for `v2a`

Task `v2a` is declared in `models.d/foley.yaml`: label "Video to sound (Foley)", modality audio, studio `sound`.

* **Video input (required).** One video file, sent as `inputs: [{"asset": ..., "role": "video"}]`. The worker looks for
  role `video`, then type `video`, then any input whose path has a video extension (mp4 mov mkv webm avi m4v gif mpg
  mpeg ogv flv wmv). **The core has no video asset type or upload route today** (`/api/upload` validates image, audio
  and GLB only; the asset types are image/audio/voice/mesh/text/other). The lead needs to add a video upload
  (accept mp4/mov/webm/mkv, ffprobe validation, a poster thumbnail and `<video>` player in the library), or let the sound
  studio accept a local path and register it as an asset with `type: video` / `role: video`.
  The worker only needs the absolute file path in `inputs[].path`.
* **Accepted formats.** Anything PyAV/ffmpeg can decode: mp4, mov, mkv, webm, avi, gif. Audio track in the file is ignored (and replaced
  in the muxed output). Needs a video stream of at least 1 s (ThinkSound at least 2 s).
* **Max duration.** MMAudio 30 s (trained on 8 s; longer drifts), HunyuanVideo-Foley 15 s (hard cap), ThinkSound 20 s,
  whole seconds (trained on 9 s). The `seconds` param is cut to the video length, so a 5 s clip with seconds=8 gives 5 s.
  Suggest the UI trim a longer video to the first N seconds or show a clip-range picker (not implemented in the worker; it always starts at 0).
  Memory is flat in video length at these sizes; the video is read at 8 fps (CLIP) and 24-25 fps (Synchformer) in RAM.
* **Prompt.** Text box (`prompt`, style `sfx`). Optional for MMAudio, required by HunyuanVideo-Foley and ThinkSound (the worker raises
  an `input` error asking for it). `negative` (the standard negative box) is used by MMAudio and Hunyuan. Feature flags in the registry:
  `video_input`, `text_only`, `needs_prompt`, `cot`.
* **ThinkSound extra.** `cot` (textarea "Chain-of-thought description"): a step-by-step description of the sounds, fed to T5. Empty falls back
  to the caption. The original MLLM (VideoLLaMA2 / Qwen2.5-Omni) that writes this text is not part of the open weights; the studio's prompt
  LLM could draft it, but that is not wired.
* **Params** (all in the registry; same names across the three models):
  `seconds`, `batch` (takes), `seed`, `mux_video` (bool, default on), advanced: `cfg_scale`, `steps`;
  MMAudio also `mask_video` (advanced: keep timing, ignore picture content). Presets fast / balanced / quality exist for each model.
* **Outputs.** Per take one **audio asset** (24-bit wav: MMAudio 44.1 kHz mono, Hunyuan 48 kHz mono, ThinkSound 44.1 kHz stereo) named
  `<prompt> #N`, seed reported per take. When `mux_video` is on and ffmpeg exists, the same take muxed into the original video
  (h264 stream copied when possible, else re-encoded; aac 192k) is written next to it as `take_N_video.mp4` and listed in
  `attach` of the file meta, so the core stores it as an attachment of the audio asset (`meta.files`, role attachment). The UI should
  show a `<video>` player for that attachment (served by `/files/<project>/<asset>/take_N_video.mp4`) and a download button. Muxing failures
  only log a message; the audio is still delivered.
* **Text-only `sfx`.** Only `mmaudio-large-44k-v2` is registered for `sfx` (no video; `seconds` 1 to 30). HunyuanVideo-Foley and ThinkSound
  cannot run without a video (their registry features say `text_only: false`).
* **Defaults.** `mmaudio-large-44k-v2` is `default_for: [v2a]` (fastest, lowest memory, also does sfx, but non-commercial).
  For commercial work the picker should surface the licence badges: only HunyuanVideo-Foley is commercial (outside EU/UK/KR).
* **Cancel / timing.** `est` in the registry: MMAudio ~31 s load + 5 s/take, Hunyuan 31 s + 22 s/take, ThinkSound 21 s + 20 s/take (5 s clips).
  Hunyuan and ThinkSound cannot be cancelled inside a take.

## Gotchas

* Shared HF downloads were very slow on this machine (about 150 KB/s per connection). The weights were fetched with a
  small parallel range downloader (32 connections). It produced files with **silent holes** (the signed CDN URL expired,
  and truncated range bodies were counted as done). Every blob was verified by sha256 against its LFS name (all OK now). If you re-download, use
  `hf download` with `HF_HUB_DISABLE_XET=1` and verify with `sha256sum`; `from_pretrained` does not notice holes, and
  HunyuanVideo-Foley loads with `strict=False`, so a damaged `hunyuanvideo_foley.pth` still "works" and yields garbage.
* `torch.cuda` warns that GB10 is capability 12.1 (torch 2.9.1 supports up to 12.0). Harmless here.
* `numpy==1.26.4` pinned (Hunyuan's requirement; MMAudio needs <2.1). `transformers<5` (4.57.6): Hunyuan's SigLIP2
  `get_image_features` and the ThinkSound CLIP patch rely on the 4.x API.
* ThinkSound: the vendored `data_utils` read video with `torio` (removed in torchaudio 2.9) and its CLIP text patch passes `return_dict` to
  `CLIPTextTransformer` (not accepted by transformers 4.57). The worker therefore re-implements frame reading with PyAV and patches
  `get_text_features` itself. The DiT is built per duration (sequence lengths are baked into the config). flash-attn is absent (not needed), `natten`/`hiera` etc. are training-only.
  The full `thinksound.ckpt` (21 GB) was not downloaded; the light one (5.7 GB) is what the repo's `defaults.ini` points to.
  The newer PrismAudio (same repo, branch `prismaudio`) was not tried.
* MMAudio: 44k mode needs the BigVGAN v2 repo (`bigvgan.py` is vendored in the pip package, only weights come from the HF repo) and
  `apple/DFN5B-CLIP-ViT-H-14-384` (redirects to `-378` on the hub; the cache dir uses the 384 name, which is what open_clip asks for).
  The 16k variants need `best_netG.pt`, not downloaded. Do not `uv pip install tensordict`: it pulls a different torch.
* HunyuanVideo-Foley reads the whole file: the worker trims the input to `seconds` with ffmpeg first (`-an`, libx264 crf 18) in the job's out_dir
  and deletes it afterwards. Default negative prompt of the library is "noisy, harsh".
* MMAudio output is trimmed to `duration * 44100` samples (the net generates a few frames more).

## Reproduce

```bash
cd <repo>
uv venv --python 3.12 envs/foley
export VIRTUAL_ENV=$PWD/envs/foley
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1"
mkdir -p vendor && cd vendor
git clone https://github.com/hkchengrex/MMAudio.git                    # 974010a
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley.git    # df7b005
git clone https://github.com/FunAudioLLM/ThinkSound.git                # 4e2cfd5 (branch master)
cd ..
uv pip install "numpy==1.26.4" scipy librosa soundfile einops hydra-core torchdiffeq open_clip_torch av timm tqdm Pillow \
  opencv-python-headless requests huggingface_hub safetensors "transformers>=4.57,<5" sentencepiece accelerate omegaconf easydict \
  loguru pyyaml pandas diffusers colorlog python-dotenv rich tensorboard matplotlib tabulate protobuf \
  scikit-image dctorch clean-fid jsonmerge kornia torchsde wandb frozendict
uv pip install --no-deps -e vendor/MMAudio
uv pip install --no-deps descript-audiotools descript-audio-codec flatten-dict julius ffmpy randomname argbind importlib_resources \
  docstring_parser pyloudnorm markdown2 ftfy alias-free-torch k-diffusion lightning pytorch-lightning lightning-utilities torchmetrics \
  vector-quantize-pytorch x-transformers prefigure ema-pytorch local-attention einx clip-anytorch torch-einops-utils
# do NOT install tensordict / nitrous-ema (pull another torch). Frozen versions: env-lock/foley.txt (the editable mmaudio line is left out)

export HF_HUB_DISABLE_XET=1
H=envs/audio/bin/hf     # any hf CLI
$H download hkchengrex/MMAudio weights/mmaudio_large_44k_v2.pth ext_weights/v1-44.pth ext_weights/synchformer_state_dict.pth   # 6.3 GB
$H download nvidia/bigvgan_v2_44khz_128band_512x --include config.json bigvgan_generator.pt "*.py"                               # 0.5 GB
$H download apple/DFN5B-CLIP-ViT-H-14-384 --include "*.json" "*.txt" open_clip_pytorch_model.bin                               # 4 GB (not the duplicate pytorch_model.bin)
$H download tencent/HunyuanVideo-Foley hunyuanvideo_foley.pth vae_128d_48k.pth synchformer_state_dict.pth config.yaml LICENSE NOTICE  # 12.7 GB
$H download google/siglip2-base-patch16-512 ; $H download laion/larger_clap_general --include "*.json" "*.txt" pytorch_model.bin
$H download FunAudioLLM/ThinkSound thinksound_light.ckpt vae.ckpt synchformer_state_dict.pth LICENSE.md                         # 9.2 GB
$H download facebook/metaclip-h14-fullcc2.5b --include "*.json" "*.txt" model.safetensors                                      # 3.9 GB
$H download google/t5-v1_1-xl --include "*.json" spiece.model pytorch_model.bin                                                # 11.4 GB (only the encoder is used)
# verify: sha256sum of each blob must equal its file name in ~/.cache/huggingface/hub/models--*/blobs/
```

Hand test (any model): `flock <gpu.lock> envs/foley/bin/python` a small driver that pipes
`{"op":"run","id":"t","model":{"id":"mmaudio-large-44k-v2","name":"x","weights":[{"hf":"hkchengrex/MMAudio"}]},"task":"v2a","params":{"prompt":"waves","seconds":8,"batch":1,"seed":1},"inputs":[{"role":"video","type":"video","name":"clip","path":"/abs/clip.mp4"}],"out_dir":"/abs/out"}`
into `PYTHONPATH=workers HF_HUB_OFFLINE=1 envs/foley/bin/python -u workers/foley_worker.py`.

Files: `workers/foley_worker.py`, `models.d/foley.yaml`, `envs/foley`, `env-lock/foley.txt`, `vendor/{MMAudio,HunyuanVideo-Foley,ThinkSound}`,
`tests/test_foley.py` (registry-level, no GPU), this report, `docs/research/audio_more/foley_media/`.
