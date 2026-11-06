#!/usr/bin/env bash
# Higgs TTS 3 / Voxtral TTS / Fish S2 Pro / CosyVoice3 through vLLM-Omni's offline engine (Omni class),
# using the EXISTING venv envs/vllm_omni (vllm 0.25.0, vllm-omni 0.25.0rc2.dev41+g109942465, torch 2.11 cu130)
# read-only. Fish needs fish_speech DAC, CosyVoice3 needs s3tokenizer: PYTHONPATH overlay envs/vllm_omni_overlay (built with
#   uv pip install --python envs/vllm_omni/bin/python --target envs/vllm_omni_overlay --no-deps \
#     "fish-speech @ git+https://github.com/fishaudio/fish-speech@0f8a12e" descript-audio-codec descript-audiotools \
#     flatten-dict julius ffmpy importlib_resources tensorboard absl-py randomname argbind s3tokenizer onnx soxr )
# Deploy configs: low-memory copies in ./vllm_omni_deploy/*_studio.yaml (stock ones reserve 40-90 % of the GPU).
# usage: vllm_omni_tts.sh higgs|voxtral|fish|cosyvoice OUTDIR "text" [REF.wav "ref text"]
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
PY=envs/vllm_omni/bin/python
export PYTHONPATH=$(cd "$(dirname "$0")/../../../envs/vllm_omni_overlay" && pwd)  # fish_speech DAC + s3tokenizer (CosyVoice3), no-deps overlay
EX=<vllm-omni checkout>/vllm-omni/examples/offline_inference/text_to_speech
HERE=$(cd "$(dirname "$0")" && pwd); DC=$HERE/vllm_omni_deploy
m=$1 out=$2 text=$3 ref=${4:-} reftext=${5:-}
case $m in
  higgs)   $PY $EX/higgs_audio_v3/end2end.py --model bosonai/higgs-tts-3-4b --texts "$text" --output-dir "$out" \
             --deploy-config $DC/higgs_multimodal_qwen3_low_latency_studio.yaml ${ref:+--ref-audio "$ref" --ref-text "$reftext"} ;;
  # voxtral: pass the local snapshot dir; with a repo id + HF_HUB_OFFLINE=1 model_type detection (params.json) fails
  voxtral) $PY $EX/voxtral_tts/end2end.py --model $(ls -d ~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/*) --text "$text" --output-dir "$out" --write-audio \
             --deploy-config $DC/voxtral_tts_studio.yaml $( [ -n "$ref" ] && echo --ref-audio "$ref" || echo --voice casual_male ) ;;  # 20 presets in voice_embedding/
  fish)    $PY $EX/fish_speech/end2end.py --model fishaudio/s2-pro \
             --text "$text" --output-dir "$out" --deploy-config $DC/fish_qwen3_omni_studio.yaml ${ref:+--ref-audio "$ref" --ref-text "$reftext"} ;;
  cosyvoice) C=$(ls -d ~/.cache/huggingface/hub/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/*)
           # cached config.json is "{}"; vllm-omni needs model_type -> symlinked copy with a real config.json
           S=$(realpath -m "$out")/.cosyvoice3_model; mkdir -p "$S"; for f in "$C"/*; do [ "$(basename "$f")" = config.json ] || ln -sfn "$f" "$S/"; done
           echo '{"model_type": "cosyvoice3", "architectures": ["CosyVoice3Model"]}' > "$S/config.json"
           # s3tokenizer.load_model("speech_tokenizer_v3_25hz") would download 925 MB from modelscope into ~/.cache/s3tokenizer;
           # it is byte-identical (sha256 23236a74...) to the cached speech_tokenizer_v3.onnx -> symlink it instead
           mkdir -p ~/.cache/s3tokenizer; ln -sfn "$(readlink -f "$C/speech_tokenizer_v3.onnx")" ~/.cache/s3tokenizer/speech_tokenizer_v3_25hz.onnx
           mkdir -p "$out" && cd "$out"   # example has no --output-dir: writes ./output_0.wav (labelled 22050 Hz, see notes)
           $PY $EX/cosyvoice3/end2end.py --model $S --tokenizer $S/CosyVoice-BlankEN --text "$text" \
             --deploy-config $DC/cosyvoice3_studio.yaml --ref-audio "$ref" --prompt-text "You are a helpful assistant.<|endofprompt|>$reftext" ;;
esac
