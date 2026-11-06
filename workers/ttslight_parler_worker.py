"""Parler-TTS + MeloTTS backend (envs/ttslight_parler: transformers 4.46.1). Same code as ttslight_worker.py."""
from ttslight_worker import TTSLightBackend, serve


class TTSLightParlerBackend(TTSLightBackend):
    name = "ttslight_parler"
    max_loaded = 2


if __name__ == "__main__":
    serve(TTSLightParlerBackend())
