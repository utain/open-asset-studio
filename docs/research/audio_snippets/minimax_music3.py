"""MiniMax-Music3: full songs with lyrics + vocals (44.1 kHz stereo), up to ~5 min.
venv: envs/audio (diffusers @ git main a91eccd, transformers 5.22.1, torch 2.9.1 cu130)
usage: python minimax_music3.py OUT.wav [SECONDS] [SEED]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import torch
from diffusers import ModularPipeline

out = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7

b = Bench("MiniMaxAI/MiniMax-Music3")
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")
# local_files_only=True: diffusers calls model_info() for sharded repos unless set -> fails with HF_HUB_OFFLINE=1
# device_map="cuda": load straight to GPU; CPU-load + .to("cuda") doubles RAM on unified memory (OOM seen)
pipe.load_components(dtype=torch.bfloat16, local_files_only=True, device_map="cuda")
pipe.to("cuda")  # no-op for already-placed modules
b.loaded()

# Structure tags must be alone on their own line; text after a tag on the same line is dropped.
lyrics = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""
prompt = ("Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
          "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
          "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus.")

audio = pipe(prompt=prompt, lyrics=lyrics, audio_duration=seconds,  # upper bound; LM may stop earlier
             generator=torch.Generator("cuda").manual_seed(seed), output="audios")[0]
wav = audio.T if hasattr(audio, "T") else audio
wav = wav.float().cpu().numpy() if torch.is_tensor(wav) else wav
b.done(wav, pipe.sampling_rate, out)
