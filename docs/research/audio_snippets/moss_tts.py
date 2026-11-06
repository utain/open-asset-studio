"""MOSS-TTS-v1.5 (clone from ref clip, 20+ langs) and MOSS-VoiceGenerator (voice from text description).
Both trust_remote_code; audio codec = OpenMOSS-Team/MOSS-Audio-Tokenizer (cached). 24 kHz mono.
Adapted from the earlier standalone TTS worker. venv: envs/moss (transformers==5.0.0 exactly; tested). NOT envs/audio: transformers 5.22 -> "Unexpected keyword argument revision".
usage: python moss_tts.py OUT.wav design "voice description" "text"
       python moss_tts.py OUT.wav clone REF.wav "text" [language]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import importlib.util
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

torch.backends.cuda.enable_cudnn_sdp(False)   # model card: cuDNN SDPA backend is broken
out, mode = sys.argv[1], sys.argv[2]
repo = "OpenMOSS-Team/MOSS-VoiceGenerator" if mode == "design" else "OpenMOSS-Team/MOSS-TTS-v1.5"
SAMPLING = (dict(audio_temperature=1.5, audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1) if mode == "design"
            else dict(audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25, audio_repetition_penalty=1.0))
attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

b = Bench(repo)
proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True, **({"normalize_inputs": True} if mode == "design" else {}))
proc.audio_tokenizer = proc.audio_tokenizer.to("cuda")
model = AutoModel.from_pretrained(repo, trust_remote_code=True, attn_implementation=attn, torch_dtype=torch.bfloat16).to("cuda").eval()
sr = proc.model_config.sampling_rate
b.loaded()

if mode == "design":
    msg = proc.build_user_message(text=sys.argv[4], instruction=sys.argv[3])
    max_new = 1000
else:
    wav, ref_sr = sf.read(sys.argv[3], dtype="float32", always_2d=True)
    ref = proc.encode_audios_from_wav([torch.from_numpy(wav.T.copy())], ref_sr)[0]  # avoids torchaudio.load/torchcodec
    msg = proc.build_user_message(text=sys.argv[4], reference=[ref], language=sys.argv[5] if len(sys.argv) > 5 else None)
    max_new = int(12.5 * 20)                     # 12.5 tokens/s -> cap at 20 s
torch.manual_seed(42)
batch = proc([[msg]], mode="generation")
with torch.no_grad():
    o = model.generate(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"),
                       max_new_tokens=max_new, **SAMPLING)
audio = proc.decode(o)[0].audio_codes_list[0].float().cpu().numpy().reshape(-1)
b.done(audio, sr, out)
