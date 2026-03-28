from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"

DATA_DIR = (
    Path(os.environ["MM_DATA_DIR"]).expanduser().resolve()
    if "MM_DATA_DIR" in os.environ and os.environ["MM_DATA_DIR"].strip()
    else _DEFAULT_DATA_DIR
)


def get_data_dir() -> Path:
    return DATA_DIR
