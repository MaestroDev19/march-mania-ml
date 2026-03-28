"""
Emit web/public/data/teams.json from Kaggle MTeams.csv and WTeams.csv.

Run from repo root: python scripts/generate_teams_json.py
Requires: data/MTeams.csv, data/WTeams.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "web" / "public" / "data" / "teams.json"


def _read_teams(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tid = int(r["TeamID"])
            name = str(r["TeamName"]).strip()
            rows.append({"id": tid, "name": name})
    rows.sort(key=lambda x: (str(x["name"]).lower(), int(x["id"])))
    return rows


def main() -> None:
    m_path = DATA / "MTeams.csv"
    w_path = DATA / "WTeams.csv"
    if not m_path.is_file() or not w_path.is_file():
        raise FileNotFoundError(f"Need {m_path} and {w_path}")

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sources": {"M": "MTeams.csv", "W": "WTeams.csv"},
        "M": _read_teams(m_path),
        "W": _read_teams(w_path),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(payload['M'])} men, {len(payload['W'])} women teams)")


if __name__ == "__main__":
    main()
