"""Orpheus 3B 0.1 ft (unsloth copy of canopylabs/orpheus-3b-0.1-ft): English TTS, 8 preset voices,
inline emotive tags <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>. SNAC 24 kHz decode.
Plain transformers generate (orpheus-speech pkg needs vllm). venv: envs/audio (+snac; hubertsiuzdak/snac_24khz cached)
usage: python orpheus.py OUT.wav [voice] "text" """
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

out = sys.argv[1]
voice = sys.argv[2] if len(sys.argv) > 2 else "leo"   # tara leah jess leo dan mia zac zoe
text = sys.argv[3] if len(sys.argv) > 3 else "Stand behind me. <sigh> Nothing gets through this shield while I still draw breath."
repo = "unsloth/orpheus-3b-0.1-ft"
b = Bench(repo)
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
b.loaded()

# prompt: SOH(128259) + "voice: text" + EOT(128009) EOH(128260); model emits SOS(128257) codes ... EOS(128258)
ids = tok(f"{voice}: {text}", return_tensors="pt").input_ids
ids = torch.cat([torch.tensor([[128259]]), ids, torch.tensor([[128009, 128260]])], dim=1).to("cuda")
torch.manual_seed(42)
with torch.no_grad():
    gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=1200, do_sample=True,
                         temperature=0.6, top_p=0.95, repetition_penalty=1.1, eos_token_id=128258)
g = gen[0, ids.shape[1]:].tolist()
if 128257 in g:
    g = g[len(g) - g[::-1].index(128257):]
codes = [t - 128266 for t in g if t != 128258 and t >= 128266]
codes = codes[: len(codes) // 7 * 7]
l1, l2, l3 = [], [], []
for i in range(len(codes) // 7):          # 7 tokens per frame -> 3 SNAC layers (1 + 2 + 4)
    c = codes[7 * i: 7 * i + 7]
    l1.append(c[0]); l2 += [c[1] - 4096, c[4] - 4 * 4096]
    l3 += [c[2] - 2 * 4096, c[3] - 3 * 4096, c[5] - 5 * 4096, c[6] - 6 * 4096]
with torch.no_grad():
    audio = snac.decode([torch.tensor(l, device="cuda").unsqueeze(0) for l in (l1, l2, l3)])
b.done(audio[0, 0].float().cpu().numpy(), 24000, out)
