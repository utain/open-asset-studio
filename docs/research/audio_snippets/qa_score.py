"""QA proxies: Whisper-large-v3-turbo transcript (+WER-ish word overlap) for TTS, LAION-CLAP text-audio
cosine for SFX/music. venv: envs/audio. Both small (~1.6 GB / 0.6 GB GPU).
usage: python qa_score.py JOBS.json  ->  [{"path", "kind": "tts"|"sfx", "text"/"prompt", "lang"?}] ; prints JSON lines"""
import json, os, re, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, librosa, torch
from transformers import pipeline, ClapModel, ClapProcessor

jobs = json.load(open(sys.argv[1]))
asr = clap = None
def words(s): return re.findall(r"\w+", s.lower())
for j in jobs:
    y, _ = librosa.load(j["path"], sr=None, mono=True)
    r = {"path": os.path.basename(j["path"])}
    if j["kind"] == "tts":
        if asr is None:
            asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
        y16 = librosa.resample(y, orig_sr=librosa.get_samplerate(j["path"]), target_sr=16000)
        kw = {"language": j["lang"]} if j.get("lang") else {}
        r["asr"] = asr({"raw": y16, "sampling_rate": 16000}, generate_kwargs=kw)["text"].strip()
        ref, hyp = words(j["text"]), words(r["asr"])
        if j.get("lang") in ("th", "zh", "ja"):   # no spaces: char overlap
            ref, hyp = list("".join(ref)), list("".join(hyp))
        r["word_recall"] = round(sum(1 for w in ref if w in hyp) / max(len(ref), 1), 2)
    else:
        if clap is None:
            clap = (ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda").eval(), ClapProcessor.from_pretrained("laion/clap-htsat-unfused"))
        m, p = clap
        y48 = librosa.resample(y, orig_sr=librosa.get_samplerate(j["path"]), target_sr=48000)
        inp = p(text=[j["prompt"], "silence", "speech"], audio=[y48], sampling_rate=48000, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = m(**inp)
        sims = torch.nn.functional.cosine_similarity(o.audio_embeds, o.text_embeds).tolist()
        r["clap_prompt"], r["clap_silence"], r["clap_speech"] = [round(s, 3) for s in sims]
    print(json.dumps(r, ensure_ascii=False), flush=True)
