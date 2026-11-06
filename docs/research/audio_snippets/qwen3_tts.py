"""Qwen3-TTS 12Hz 1.7B: CustomVoice (9 preset speakers + `instruct` emotion/style) and Base (zero-shot clone).
venv: envs/qwen (qwen-tts==0.1.1 pins transformers==4.57.3; torch 2.9.1 cu130). 24 kHz mono.
usage: python qwen3_tts.py OUT.wav custom|clone [REF.wav "ref transcript"]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import importlib.util
import torch
from qwen_tts import Qwen3TTSModel

out, mode = sys.argv[1], sys.argv[2]
repo = "Qwen/Qwen3-TTS-12Hz-1.7B-" + ("CustomVoice" if mode == "custom" else "Base")
attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

b = Bench(repo)
# transformers 4.57.x calls model_info() (is_base_mistral) for hub ids even with HF_HUB_OFFLINE=1 -> pass local path
from huggingface_hub import snapshot_download
local = snapshot_download(repo, local_files_only=True)
model = Qwen3TTSModel.from_pretrained(local, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn)
b.loaded()
torch.manual_seed(42)
text = "Stand behind me. Nothing gets through this shield while I still draw breath."
if mode == "custom":
    # speakers: Vivian Serena Uncle_Fu Dylan Eric Ryan Aiden Ono_Anna Sohee ; language "Auto" or one of 10
    wavs, sr = model.generate_custom_voice(text=text, language="English", speaker="Ryan",
                                           instruct="Deep, determined and heroic, speaking slowly.")
else:
    ref, ref_text = sys.argv[3], sys.argv[4]
    prompt = model.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text, x_vector_only_mode=False)
    wavs, sr = model.generate_voice_clone(text="Hold the line! Reinforcements are on the way.",
                                          language="English", voice_clone_prompt=prompt)
b.done(wavs[0], sr, out)
