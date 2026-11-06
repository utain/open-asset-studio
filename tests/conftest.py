"""Tests run against a throw-away data/settings/logs folder (set before any app module is imported)."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="studio-test-"))
os.environ["STUDIO_DATA"] = str(_TMP / "data")
os.environ["STUDIO_LOGS"] = str(_TMP / "logs")
os.environ["STUDIO_SETTINGS"] = str(_TMP / "settings.json")
(_TMP / "logs").mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path
