"""Repository root for all analysis scripts (portable clone / GitHub checkout)."""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MIGRAINE_REPO_ROOT", _REPO)).expanduser().resolve()