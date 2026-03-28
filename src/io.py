# from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import DATA_DIR


def read_csv(filename: str, *, data_dir: Path | None = None) -> pd.DataFrame:
    base = DATA_DIR if data_dir is None else data_dir
    return pd.read_csv(base / filename)

# from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import DATA_DIR


def require_files(names: list[str]) -> None:
    missing: list[Path] = []

    for name in names:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(path)

    if missing:
        missing_list = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            f"Missing required data file(s):\n{missing_list}\n\nDATA_DIR={DATA_DIR}"
        )


def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n\nDATA_DIR={DATA_DIR}\n"
            "Tip: set MM_DATA_DIR to override the default data directory."
        )
    return pd.read_csv(path)

