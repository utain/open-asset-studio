"""Music backend `music_levo`: placeholder for Tencent SongGeneration (LeVo). NOT FUNCTIONAL.

The official code repository (github.com/tencent-ailab/SongGeneration) returns 404 and tencent/SongGeneration is not public on
the Hub; only third-party weight uploads remain and their licence cannot be verified. The registry entry is marked
`features.blocked`; this stub exists so the backend resolves and reports a clear error. See docs/research/audio_more/music.md.
"""
from __future__ import annotations

from studio_worker import Backend, serve

REASON = "SongGeneration/LeVo is blocked: the upstream code repository is unavailable (404) and its licence cannot be verified."


class MusicLevo(Backend):
    name = "music_levo"
    max_loaded = 1

    def probe(self, model):
        return False, REASON

    def load(self, model, ctx):
        raise RuntimeError(REASON)

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        raise RuntimeError(REASON)


if __name__ == "__main__":
    serve(MusicLevo())
