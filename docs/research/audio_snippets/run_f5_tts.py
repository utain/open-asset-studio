"""F5-TTS v1 Base: zero-shot voice clone (EN/ZH), flow matching, 24 kHz mono. Ref audio <=12 s + transcript.
venv: envs/audio (f5-tts==1.1.22 --no-deps + deps, vocos cached: charactr/vocos-mel-24khz)
usage: python run_f5_tts.py OUT.wav REF.wav "ref transcript" "text to speak" """
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import soundfile as sf, torch, torchaudio

def _sf_load(path, *a, **k):  # torchaudio 2.9 load() needs torchcodec (no aarch64 wheel) -> soundfile
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(w.T.copy()), sr
torchaudio.load = _sf_load
from f5_tts.api import F5TTS

out, ref, ref_text, text = sys.argv[1:5]
b = Bench("SWivid/F5-TTS")
tts = F5TTS(model="F5TTS_v1_Base")   # resolves ckpt from HF cache (SWivid/F5-TTS/F5TTS_v1_Base)
b.loaded()
wav, sr, _ = tts.infer(ref_file=ref, ref_text=ref_text, gen_text=text,
                       nfe_step=32, cfg_strength=2, sway_sampling_coef=-1, speed=1.0, seed=42)
b.done(wav, sr, out)
