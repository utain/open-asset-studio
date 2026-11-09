# Third-party notices

The code in this repository is Apache-2.0 (see [LICENSE](LICENSE)). It works with, but does not include, the
third-party software and models below. Each keeps its own licence, and **you must follow that licence when you
install and use it.** Several are copyleft (GPL-3.0) or non-commercial. This list is a convenience, not legal
advice: always check the upstream licence at the pinned commit.

## Included in this repository

Front-end libraries in `app/web/vendor/` (versions: `app/web/vendor/VERSIONS.txt`):

| file | project | licence | licence text |
|---|---|---|---|
| `model-viewer.min.js` | [@google/model-viewer](https://github.com/google/model-viewer) 3.1.1 | BSD-3-Clause header, Apache-2.0 package licence | `app/web/vendor/model-viewer-LICENSE.txt` |
| `wavesurfer.esm.js`, `wavesurfer-regions.esm.js` | [wavesurfer.js](https://github.com/katspaugh/wavesurfer.js) 7.10.1 | BSD-3-Clause | `app/web/vendor/wavesurfer-LICENSE.txt` |
| `fonts/inter-latin-wght.woff2` | [Inter](https://github.com/rsms/inter) via @fontsource-variable/inter 5.1.1 | SIL OFL 1.1 | `app/web/vendor/fonts/inter-OFL-LICENSE.txt` |
| `app/web/core/icons.js` (inlined SVG paths) | [Lucide](https://github.com/lucide-icons/lucide) 0.468.0 | ISC | `app/web/vendor/lucide-LICENSE.txt` |

## Not included: fetched by `scripts/fetch_vendor.sh`

Upstream source code that the workers import, cloned at the pinned commit from `vendor/vendor.lock`:

| path | repository | commit | licence (as found at that commit) |
|---|---|---|---|
| `vendor/ACE-Step-1.5` | [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) | `ca1e85fe9430` | MIT |
| `vendor/Amphion` | [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) | `26f688311018` | MIT |
| `vendor/audiocraft` | [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | `896ec7c47f5e` | MIT (code; the MusicGen/AudioGen/MAGNeT weights are CC-BY-NC-4.0) |
| `vendor/BigVGAN-awei6` | [awei-6/BigVGAN](https://github.com/awei-6/BigVGAN) | `ac17a79ca9ab` | MIT |
| `vendor/delayed-streams-modeling` | [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) | `4c4f65e147df` | Apache-2.0 / MIT (dual) |
| `vendor/demucs` | [adefossez/demucs](https://github.com/adefossez/demucs) | `829d21890023` | MIT (code and the HTDemucs weights). Also pip-installed without vendoring: `bs-roformer` 0.4.1 (MIT) for the Mel-Band RoFormer vocal model, whose weights ([KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer)) are MIT per the model card |
| `vendor/dia2` | [nari-labs/dia2](https://github.com/nari-labs/dia2) | `8687268f4ed3` | Apache-2.0 |
| `vendor/DiffRhythm` | [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) | `28ad63c0f096` | Apache-2.0 (code) |
| `vendor/diffsinger` | [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) | `cb7869b8733a` | Apache-2.0 |
| `vendor/diffsinger-legacy` | [MoonInTheRiver/DiffSinger](https://github.com/MoonInTheRiver/DiffSinger) | `4662c53a27a5` | MIT (code; the Opencpop voice data is non-commercial) |
| `vendor/dinov2` | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | `7764ea0f912e` | Apache-2.0 (code; the DINOv2 weights are Apache-2.0 as well) |
| `vendor/FlexiCubes` | [MaxtirError/FlexiCubes](https://github.com/MaxtirError/FlexiCubes) | `815e075a2a40` | Apache-2.0 (git submodule of TRELLIS; used for its mesh decoder) |
| `vendor/generative-models` | [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) | `e8cd657656fa` | MIT (code; the SV3D weights are under the Stability AI Community License) |
| `vendor/GLM-4-Voice` | [zai-org/GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice) | `eb00ce9142e8` | Apache-2.0 (code; model licence separate) |
| `vendor/heartlib` | [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) | `5858ca8f1ffe` | Apache-2.0 |
| `vendor/HY-Motion-1.0` | [Tencent-Hunyuan/HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) | `4e426f5a1021` | Tencent HY-Motion 1.0 Community License (not licensed in the EU, UK and South Korea; more than 1 million monthly active users needs a separate licence). Used by `workers/motion_hy_worker.py`; text encoders Qwen3-8B (Apache-2.0) and CLIP ViT-L/14 (MIT) |
| `vendor/Hunyuan3D-2` | [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | `f8db63096c82` | Tencent Hunyuan 3D 2.0 Community License (not valid in the EU, UK and South Korea) |
| `vendor/HunyuanVideo-Foley` | [Tencent-Hunyuan/HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) | `df7b005b5023` | Tencent Hunyuan Community License (not valid in the EU, UK and South Korea) |
| `vendor/index-tts` | [index-tts/index-tts](https://github.com/index-tts/index-tts) | `d9e41aac89fd` | bilibili Model Use License Agreement |
| `vendor/Kimi-Audio` | [MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) | `349251e1d8f4` | Apache-2.0 (Qwen-derived parts) + MIT (README; no LICENSE file) |
| `vendor/LLaMA-Omni2` | [ictnlp/LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2) | `c8afa9061a9c` | Apache-2.0 code (README); models academic use only |
| `vendor/magenta-realtime` | [magenta/magenta-realtime](https://github.com/magenta/magenta-realtime) | `694a545e4ba0` | Apache-2.0 |
| `vendor/Make-An-Audio-2` | [bytedance/Make-An-Audio-2](https://github.com/bytedance/Make-An-Audio-2) | `f02ec0a3d874` | MIT |
| `vendor/Matcha-TTS` | [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | `bd4d90d93214` | MIT |
| `vendor/MeloTTS` | [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) | `209145371cff` | MIT |
| `vendor/meshoptimizer` | [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) | `f0a684f11ead` | MIT (c) Arseny Kapoulkine. Built into `envs/retopo/lib/libmeshoptimizer.so` by `install/engine.yaml` and called through ctypes (`workers/meshopt_ctypes.py`) for the LOD chain of the game-engine export |
| `vendor/MiniCPM-o` | [OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) | `47e1fa75b45b` | Apache-2.0 (code; model licence separate) |
| `vendor/MMAudio` | [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) | `974010a026c7` | MIT (code; weights CC-BY-NC-4.0) |
| `vendor/OpenVoice` | [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) | `74a1d147b17a` | MIT |
| `vendor/parler-tts` | [huggingface/parler-tts](https://github.com/huggingface/parler-tts) | `d108732cd577` | Apache-2.0 |
| `vendor/PyNanoInstantMeshes` | [vork/PyNanoInstantMeshes](https://github.com/vork/PyNanoInstantMeshes) (Python binding of [Instant Meshes](https://github.com/wjakob/instant-meshes)) | `d02a725f93a1` | BSD-3-Clause per the package classifier and `native/LICENSE.txt` (Instant Meshes, (c) Wenzel Jakob, Daniele Panozzo, Marco Tarini, Olga Sorkine-Hornung); **the repository root has no LICENSE file: verify: true**. Built from source with `install/files/pynim_pure_quad.patch` (adds a `pure_quad` switch; BSD-3, same terms). Bundled in `native/ext`: TBB (Apache-2.0), Eigen (MPL-2.0, headers, unmodified), rply (MIT), pcg32 (Apache-2.0), dset (zlib-style), pss (Intel BSD-style), half (MIT) |
| `vendor/rvc` | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | `81eed5e8f68b` | MIT |
| `vendor/seed-vc` | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | `51383efd9210` | GPL-3.0 |
| `vendor/Spark-TTS` | [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) | `2f1ea9082400` | Apache-2.0 (code; weights CC-BY-NC-SA-4.0) |
| `vendor/stable-fast-3d` | [Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d) | `ff21fc491b4d` | Stability AI Community License (free for research and commercial use below USD 1M annual revenue; the weights are gated on Hugging Face) |
| `vendor/stable-point-aware-3d` | [Stability-AI/stable-point-aware-3d](https://github.com/Stability-AI/stable-point-aware-3d) | `fdc311b16809` | Stability AI Community License (same terms; weights gated) |
| `vendor/Step-Audio2` | [stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) | `76e272b56c39` | Apache-2.0 |
| `vendor/Step1X-3D` | [stepfun-ai/Step1X-3D](https://github.com/stepfun-ai/Step1X-3D) | `cb5ac944709c` | Apache-2.0 |
| `vendor/StyleTTS2` | [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) | `5cedc71c333f` | MIT |
| `vendor/tango` | [declare-lab/tango](https://github.com/declare-lab/tango) | `34ecd3889850` | CC-BY-NC-SA-4.0 |
| `vendor/TangoFlux` | [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) | `164420faa2b3` | TangoFlux non-commercial community licence |
| `vendor/ThinkSound` | [FunAudioLLM/ThinkSound](https://github.com/FunAudioLLM/ThinkSound) | `4e2cfd57efaa` | Apache-2.0 (README; no LICENSE file) |
| `vendor/TRELLIS` | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | `193e0502b5cb` | MIT |
| `vendor/TripoSG` | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) | `fc5c40990181` | MIT |
| `vendor/TripoSR` | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) | `107cefdc244c` | MIT |
| `vendor/UniRig` | [VAST-AI-Research/UniRig](https://github.com/VAST-AI-Research/UniRig) | `6793c6640ff0` | MIT |
| `vendor/YuE` | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | `1116bea76032` | Apache-2.0 |
| `vendor/YuE-v1` | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | `6d4f0b1f8ce6` | Apache-2.0 |
| `vendor/Zonos` | [Zyphra/Zonos](https://github.com/Zyphra/Zonos) | `bc40d98e1e1a` | Apache-2.0 |
| `engines/hy3d/vendor/Hunyuan3D-2.1` | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | `82920d643c0d` | Tencent Hunyuan 3D 2.1 Community License (not valid in the EU, UK and South Korea) |
| `vendor/MOSS-TTS` | [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) | `934d6826b084` | Apache-2.0 |
| `vendor/MOSS-Audio` | [OpenMOSS/MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio) | `66326e6e0db3` | Apache-2.0 (README; no LICENSE file) |
| `vendor/MOSS-Transcribe-Diarize` | [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) | `79c675f0670c` | Apache-2.0 |
| `vendor/magenta-realtime/magenta_rt/_vendor/sequence-layers` | [magenta/sequence-layers](https://github.com/magenta/sequence-layers) | `0773d337495a` | Apache-2.0 |
| `vendor/vllm-omni` | [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) | `109942465fa4` | Apache-2.0 |

**GPL-3.0 note:** `workers/vc_worker.py` imports Seed-VC (GPL-3.0) at runtime in its own process. If you
distribute a build that bundles Seed-VC, the GPL applies to that distribution.

## Derived code

Parts of this repository are adapted from third-party code. The adapted files remain subject to the upstream licence
in addition to this repository's Apache-2.0 licence.

- **vLLM-Omni deploy configs and examples** (Apache-2.0, [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)).
  The YAML files in `docs/research/audio_snippets/vllm_omni_deploy/` are low-memory copies of the stock vLLM-Omni deploy
  configs (`cosyvoice3`, `fish_qwen3_omni`, `higgs_multimodal_qwen3_low_latency`, `voxtral_tts`) with only the GPU
  memory and concurrency settings changed. The offline-inference recipes in `workers/vllm_omni_worker.py` and
  `docs/research/audio_snippets/vllm_omni_tts.sh` follow vLLM-Omni's
  `examples/offline_inference/text_to_speech/*/end2end.py`.
- **StyleTTS 2 inference code** (MIT, [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)). `workers/ttslight_styletts2.py`
  is ported from the upstream `Demo/Inference_LibriTTS.ipynb` notebook and loads the upstream code from
  `vendor/StyleTTS2` (pinned in `vendor/vendor.lock`; licence listed in the table above).
- **RIFE v4 IFNet** (MIT, [hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE) / Practical-RIFE). `workers/video_rife.py`
  is a pure-PyTorch re-implementation of the v4 `IFNet` (frame interpolation) that loads the `TensorForger/RIFE-safetensors`
  weights. No upstream repository is cloned.

- **Retopo + bake dependencies** (`install/retopo.yaml`, `env-lock/retopo.txt`; pip-installed into `envs/retopo`, no weights):
  [xatlas](https://github.com/mworchel/xatlas-python) 0.0.11 (MIT; wraps [xatlas](https://github.com/jpcy/xatlas), MIT) for UV unwrapping,
  [Open3D](https://github.com/isl-org/Open3D) 0.20.0 (MIT; its Embree 4 is Apache-2.0) for ray casting, closest point and distance queries,
  [pymikktspace](https://github.com/bhouston/pymikktspace) 0.3.0 (MIT wrapper around MikkTSpace by Morten S. Mikkelsen, zlib-style licence)
  for tangents, [trimesh](https://github.com/mikedh/trimesh) (MIT), fast-simplification (MIT), SciPy and NumPy (BSD), Pillow (MIT-CMU).
  Instant Meshes is listed in the table above. The `retopo-instant-meshes` registry entry is marked commercial because every
  component is BSD / MIT / Apache; re-check the Instant Meshes repository licence (see `verify: true` above) before shipping a product.

- **Game-engine export** (`models.d/engine.yaml`, `workers/engine_worker.py`, `app/plugins/engine.py`; commercial: yes). LODs use
  [meshoptimizer](https://github.com/zeux/meshoptimizer) (MIT, listed in the table above). Collision uses
  [CoACD](https://github.com/SarahWeiii/CoACD) 1.0.14 (MIT, (c) Xinyue Wei et al.; the `coacd` PyPI wheel bundles OpenVDB (MPL-2.0), oneTBB (Apache-2.0),
  Boost (BSL-1.0), Eigen (MPL-2.0), spdlog (MIT), zlib and other permissive libraries; installed into `envs/retopo` and pinned in `env-lock/retopo.txt`).
  The binary FBX 7.4 writer (`workers/fbx_write.py`) is original code written from the public file-format description and checked with ufbx;
  it contains no Autodesk code and uses no FBX SDK. The Unity editor script `workers/engine_files/OpenAssetStudioPostprocessor.cs` is part of this project (it calls only the public Unity API).

## Not included: model weights

Weights are downloaded by you from their publishers on Hugging Face and are never part of this repository. Their
licences vary widely (Apache-2.0, MIT, CC-BY-NC, CPML, research-only, community licences with regional limits,
gated terms). Each model's licence, with a commercial yes/no flag, is in its registry entry (`models.yaml`,
`models.d/*.yaml`), shown on the Models page and stored on every asset. Audio models are indexed in
[docs/research/audio_more/README.md](docs/research/audio_more/README.md).
3D models (image to mesh, rigging, multi-view; Stability AI Community License weights are gated) are indexed in
[docs/research/3d_more.md](docs/research/3d_more.md).

## Seamless PBR materials (`workers/span_lite.py`, `scripts/fetch_pbrify.py`)

- **PBRify_Remix** by Kim2091: the four SPAN checkpoints that estimate normal, height and roughness maps (`PBRify_NormalV3`, `PBRify_Height`, `PBRify_Roughness`, `PBRify_UpscalerSPAN_Small_x4`). Licence **CC0-1.0**
  (public domain dedication, [Kim2091/PBRify_Remix](https://github.com/Kim2091/PBRify_Remix)). They are not in this repository: `scripts/fetch_pbrify.py` downloads the release
  [v1.7.2_ComfyOnly](https://github.com/Kim2091/PBRify_Remix/releases/tag/v1.7.2_ComfyOnly) (`PBRify_Remix_1.7.2_ComfyUI_ONLY.zip`) into `~/.cache/pbrify/1.7.2` and checks the sha256 of the archive and of each file.
- **SPAN architecture** (`workers/span_lite.py`): a short re-implementation of the network, written after [spandrel](https://github.com/chaiNNer-org/spandrel) (MIT) and the original SPAN code by
  hongyuanyu ([hongyuanyu/SPAN](https://github.com/hongyuanyu/SPAN), Apache-2.0; paper: Wan et al., "Swift Parameter-free Attention Network for Efficient Super-Resolution", CVPR-W 2024).
  The `Conv3XC` re-parametrisation is rebuilt from the stored `conv1/conv2/conv3` weights, because the checkpoints carry stale `eval_conv` tensors (`docs/research/materials.md`).
- **Z-Image Turbo** (Apache-2.0) generates the base colour; it is the model already listed on the Models page.

Some outputs carry watermarks added by upstream code (for example Chatterbox's Perth watermark).

## Optional cloud providers

The optional cloud adapters (`app/cloud/`, `models.d/cloud.yaml`) call third-party APIs with your own keys, under
those providers' terms. They are off by default.
