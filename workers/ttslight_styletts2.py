"""StyleTTS 2 (yl4579/StyleTTS2-LibriTTS) inference, ported from Demo/Inference_LibriTTS.ipynb.

Code: vendor/StyleTTS2 (git clone, MIT). Weights: Models/LibriTTS/{config.yml,epochs_2nd_00020.pth} from the HF repo.
The repo's own helper nets (ASR, JDC pitch, PLBERT) are plain files of the clone (Utils/*). The clone uses relative
paths, so loading chdir()s into it. `utils.py` of the clone (monotonic_align, training only) is never imported.

LibriTTS is a multispeaker model: the diffusion sampler is conditioned on a reference style, so "tts" without a
reference clip uses one of the demo clips shipped in the HF repo (reference_audio.zip) as the voice.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "StyleTTS2"

# built-in voices for the plain tts task: label -> file name inside reference_audio.zip
BUILTIN_VOICES = {
    "libritts-1221": "1221-135767-0014.wav",
    "libritts-908": "908-157963-0027.wav",
    "libritts-4077": "4077-13754-0000.wav",
    "libritts-5639": "5639-40744-0020.wav",
    "libritts-696": "696_92939_000016_000006.wav",
    "libritts-1789": "1789_142896_000022_000005.wav",
}


class StyleTTS2Infer:
    def __init__(self, snapshot: str, device: str):
        import librosa  # noqa: F401
        import torch
        import torchaudio
        import yaml
        from munch import Munch  # noqa: F401

        self.torch, self.device = torch, device
        self.snapshot = Path(snapshot)
        # torch >= 2.6 defaults weights_only=True: the upstream checkpoints (and the helper nets) need False
        _orig_load = torch.load

        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)
        torch.load = _load

        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        cwd = os.getcwd()
        os.chdir(VENDOR)                       # config.yml refers to Utils/... relative to the repo
        try:
            self._set_espeak()
            import phonemizer
            from models import build_model, load_ASR_models, load_F0_models
            from text_utils import TextCleaner
            from Utils.PLBERT.util import load_plbert
            from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule

            cfg_path = self.snapshot / "Models" / "LibriTTS" / "config.yml"
            config = yaml.safe_load(open(cfg_path))
            text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
            pitch_extractor = load_F0_models(config["F0_path"])
            plbert = load_plbert(config["PLBERT_dir"])
            self.params = self._munch(config["model_params"])
            model = build_model(self.params, text_aligner, pitch_extractor, plbert)
            for k in model:
                model[k].eval().to(device)
            ck = torch.load(str(self.snapshot / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"), map_location="cpu")["net"]
            for key in model:
                if key in ck:
                    try:
                        model[key].load_state_dict(ck[key])
                    except Exception:  # noqa: BLE001  (DataParallel checkpoints carry a "module." prefix)
                        sd = OrderedDict((k[7:], v) for k, v in ck[key].items())
                        model[key].load_state_dict(sd, strict=False)
            for k in model:
                model[k].eval()
            self.model = model
            self.sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
                                            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
            self.cleaner = TextCleaner()
            self.phon = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        finally:
            os.chdir(cwd)
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self._builtin: dict[str, object] = {}

    @staticmethod
    def _munch(d):
        from munch import Munch

        if isinstance(d, dict):
            return Munch({k: StyleTTS2Infer._munch(v) for k, v in d.items()})
        if isinstance(d, list):
            return [StyleTTS2Infer._munch(v) for v in d]
        return d

    @staticmethod
    def _set_espeak():
        try:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        except Exception:  # noqa: BLE001  (a system espeak-ng, if there is one, still works)
            pass

    # ------------------------------------------------------------------ style
    def _mel(self, wave: np.ndarray):
        torch = self.torch
        m = self.to_mel(torch.from_numpy(wave).float())
        return (torch.log(1e-5 + m.unsqueeze(0)) - (-4)) / 4

    def style_of(self, wave: np.ndarray, sr: int):
        """Reference style vector (1, 256) = [acoustic 128 | prosodic 128] of a mono clip."""
        import librosa

        torch = self.torch
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        wave, _ = librosa.effects.trim(wave, top_db=30)
        if len(wave) < 24000 * 0.5:
            raise ValueError("the reference clip is too short (need at least about 1 s of speech)")
        mel = self._mel(wave.astype(np.float32)).to(self.device)
        with torch.no_grad():
            s = self.model.style_encoder(mel.unsqueeze(1))
            p = self.model.predictor_encoder(mel.unsqueeze(1))
        return torch.cat([s, p], dim=1)

    def builtin_style(self, name: str):
        import soundfile as sf

        if name not in self._builtin:
            with zipfile.ZipFile(self.snapshot / "reference_audio.zip") as z:
                data = z.read("reference_audio/" + BUILTIN_VOICES[name])
            y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            self._builtin[name] = self.style_of(y.mean(axis=1), sr)
        return self._builtin[name]

    # --------------------------------------------------------------- synthesis
    def synth(self, text: str, ref_wav=None, ref_sr=None, voice: str | None = None, alpha=0.3, beta=0.7,
              diffusion_steps=10, embedding_scale=1.0, cancel=lambda: None) -> np.ndarray:
        from nltk.tokenize import NLTKWordTokenizer

        torch = self.torch
        dev = self.device
        m = self.model
        if ref_wav is not None:
            ref_s = self.style_of(ref_wav, ref_sr)
        else:
            ref_s = self.builtin_style(voice if voice in BUILTIN_VOICES else next(iter(BUILTIN_VOICES)))
        out = []
        # long text: one sentence-ish chunk at a time (the model is trained on short utterances)
        import re

        chunks = [c.strip() for c in re.split(r"(?<=[.!?;])\s+|\n+", text.strip()) if c.strip()]
        for chunk in chunks:
            cancel()
            ps = self.phon.phonemize([chunk])[0]
            ps = " ".join(NLTKWordTokenizer().tokenize(ps))
            tokens = self.cleaner(ps)
            tokens.insert(0, 0)
            tokens = torch.LongTensor(tokens).to(dev).unsqueeze(0)
            with torch.no_grad():
                lens = torch.LongTensor([tokens.shape[-1]]).to(dev)
                mask = torch.arange(lens.max()).unsqueeze(0).expand(1, -1).type_as(lens)
                mask = torch.gt(mask + 1, lens.unsqueeze(1))
                t_en = m.text_encoder(tokens, lens, mask)
                bert_dur = m.bert(tokens, attention_mask=(~mask).int())
                d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
                s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(dev), embedding=bert_dur,
                                      embedding_scale=embedding_scale, features=ref_s, num_steps=diffusion_steps).squeeze(1)
                s = s_pred[:, 128:]
                ref = s_pred[:, :128]
                ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
                s = beta * s + (1 - beta) * ref_s[:, 128:]
                d = m.predictor.text_encoder(d_en, s, lens, mask)
                x, _ = m.predictor.lstm(d)
                duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze()).clamp(min=1)
                aln = torch.zeros(lens, int(pred_dur.sum().data))
                c = 0
                for i in range(aln.size(0)):
                    aln[i, c:c + int(pred_dur[i].data)] = 1
                    c += int(pred_dur[i].data)
                en = d.transpose(-1, -2) @ aln.unsqueeze(0).to(dev)
                if self.params.decoder.type == "hifigan":
                    shifted = torch.zeros_like(en)
                    shifted[:, :, 0] = en[:, :, 0]
                    shifted[:, :, 1:] = en[:, :, 0:-1]
                    en = shifted
                F0_pred, N_pred = m.predictor.F0Ntrain(en, s)
                asr = t_en @ aln.unsqueeze(0).to(dev)
                if self.params.decoder.type == "hifigan":
                    shifted = torch.zeros_like(asr)
                    shifted[:, :, 0] = asr[:, :, 0]
                    shifted[:, :, 1:] = asr[:, :, 0:-1]
                    asr = shifted
                wav = m.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            out.append(wav.squeeze().cpu().numpy()[..., :-50])
            out.append(np.zeros(int(0.12 * 24000), dtype=np.float32))
        if not out:
            raise ValueError("empty text")
        return np.concatenate(out[:-1]).astype(np.float32)
