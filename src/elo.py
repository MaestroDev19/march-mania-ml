"""Elo rating computation for March Machine Learning Mania (2026).

This module computes end-of-season Elo snapshots for NCAA men's and women's
basketball teams using the rules specified in `00_prompts.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


ELO_INIT: float = 1500.0
ELO_HCA: float = 100.0
K_FACTOR: float = 20.0

SEASON_REGRESSION_PREV: float = 0.75
SEASON_REGRESSION_MEAN: float = 0.25


_COLUMNS = [
    "Season",
    "DayNum",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
    "WLoc",
    "NumOT",
]


def _load_compact_results(path: Path) -> pd.DataFrame:
    """Load a compact results CSV and coerce required columns to numeric types."""
    df = pd.read_csv(path)
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[_COLUMNS].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["DayNum"] = pd.to_numeric(df["DayNum"], errors="raise").astype(int)
    df["WTeamID"] = pd.to_numeric(df["WTeamID"], errors="raise").astype(int)
    df["LTeamID"] = pd.to_numeric(df["LTeamID"], errors="raise").astype(int)

    # Scores/NumOT are not required for Elo updates, but we coerce for robustness.
    df["WScore"] = pd.to_numeric(df["WScore"], errors="raise").astype(float)
    df["LScore"] = pd.to_numeric(df["LScore"], errors="raise").astype(float)
    df["NumOT"] = pd.to_numeric(df["NumOT"], errors="raise").astype(int)
    return df


def _expected_win_probability(w_rating: float, l_rating: float) -> float:
    """Expected probability that winner beats loser.

    Uses:
      E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((l_rating - w_rating) / 400.0))


def _compute_elo_for_gender(
    regular_df: pd.DataFrame,
    tourney_df: pd.DataFrame,
) -> Dict[Tuple[int, int], float]:
    """Compute (season, team_id) -> Elo snapshot for one gender."""
    elo: Dict[int, float] = {}
    season_elos: Dict[Tuple[int, int], float] = {}

    all_games = pd.concat([regular_df, tourney_df], ignore_index=True)
    if all_games.empty:
        return season_elos

    all_games = all_games.sort_values(["Season", "DayNum"], kind="mergesort")

    prev_season: int | None = None
    for row in all_games.itertuples(index=False, name="Game"):
        season = int(row.Season)

        if prev_season is None:
            prev_season = season
        elif season != prev_season:
            # Snapshot at end of the just-finished season.
            for tid, r in elo.items():
                season_elos[(prev_season, tid)] = float(r)

            # Regression toward the mean before starting the next season.
            elo = {
                tid: (SEASON_REGRESSION_PREV * r + SEASON_REGRESSION_MEAN * ELO_INIT)
                for tid, r in elo.items()
            }
            prev_season = season

        w_id = int(row.WTeamID)
        l_id = int(row.LTeamID)

        w_elo = elo.get(w_id, ELO_INIT)
        l_elo = elo.get(l_id, ELO_INIT)

        # Home court adjustment affects expected-score only.
        w_loc = row.WLoc
        if w_loc == "H":
            w_adj = w_elo + ELO_HCA
        elif w_loc == "A":
            w_adj = w_elo - ELO_HCA
        else:
            w_adj = w_elo

        exp_w = _expected_win_probability(w_adj, l_elo)

        # Update:
        #  R_A_new = R_A + K * (1 - E_A)
        #  R_B_new = R_B + K * (0 - (1 - E_A))
        elo[w_id] = w_elo + K_FACTOR * (1.0 - exp_w)
        elo[l_id] = l_elo + K_FACTOR * (0.0 - (1.0 - exp_w))

    # Snapshot for the last processed season.
    if prev_season is not None:
        for tid, r in elo.items():
            season_elos[(prev_season, tid)] = float(r)

    return season_elos


def compute_elo_men(data_dir: str) -> Dict[Tuple[int, int], float]:
    """Compute end-of-season Elo snapshots for NCAA men's teams.

    Args:
        data_dir: Directory containing the required CSVs.

    Returns:
        Dict mapping (season, team_id) -> Elo at end of that season.
    """
    base = Path(data_dir)
    regular_path = base / "MRegularSeasonCompactResults.csv"
    tourney_path = base / "MNCAATourneyCompactResults.csv"

    regular_df = _load_compact_results(regular_path)
    tourney_df = _load_compact_results(tourney_path)
    return _compute_elo_for_gender(regular_df, tourney_df)


def compute_elo_women(data_dir: str) -> Dict[Tuple[int, int], float]:
    """Compute end-of-season Elo snapshots for NCAA women's teams.

    Args:
        data_dir: Directory containing the required CSVs.

    Returns:
        Dict mapping (season, team_id) -> Elo at end of that season.
    """
    base = Path(data_dir)
    regular_path = base / "WRegularSeasonCompactResults.csv"
    tourney_path = base / "WNCAATourneyCompactResults.csv"

    regular_df = _load_compact_results(regular_path)
    tourney_df = _load_compact_results(tourney_path)
    return _compute_elo_for_gender(regular_df, tourney_df)

