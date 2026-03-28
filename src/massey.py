from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


_MASSEY_FILENAME: Final[str] = "MMasseyOrdinals.csv"
_TOURNEY_FILENAME: Final[str] = "MNCAATourneyCompactResults.csv"
_RANKING_DAYNUM_FINAL: Final[int] = 133


def _load_massey_ordinals(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _MASSEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["RankingDayNum"] = pd.to_numeric(df["RankingDayNum"], errors="raise").astype(int)
    df["SystemName"] = df["SystemName"].astype(str)
    df["TeamID"] = pd.to_numeric(df["TeamID"], errors="raise").astype(int)
    df["OrdinalRank"] = pd.to_numeric(df["OrdinalRank"], errors="raise").astype(float)
    return df


def _load_tourney_compact(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _TOURNEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "WTeamID", "LTeamID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["WTeamID"] = pd.to_numeric(df["WTeamID"], errors="raise").astype(int)
    df["LTeamID"] = pd.to_numeric(df["LTeamID"], errors="raise").astype(int)
    return df


def list_stable_systems(data_dir: str, min_coverage: float = 0.8) -> list[str]:
    """Return SystemName values that survive the season-coverage filter."""
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must be in [0, 1]")

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()

    total_seasons = int(df["Season"].nunique())
    if total_seasons == 0:
        return []

    season_counts = df.groupby("SystemName")["Season"].nunique()
    keep = season_counts >= (min_coverage * total_seasons)
    return sorted(season_counts[keep].index.tolist())


def _zscore_by_season_system(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["Season", "SystemName"], sort=False)["OrdinalRank"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)

    # Lower rank is better; negate z so better teams have higher scores.
    z = -((df["OrdinalRank"] - mean) / std)
    out = df.copy()
    out["z"] = z.fillna(0.0)
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_massey_features(data_dir: str, min_coverage: float = 0.8) -> pd.DataFrame:
    """Return per-(Season, TeamID) Massey features (men)."""
    stable_systems = list_stable_systems(data_dir, min_coverage=min_coverage)
    if not stable_systems:
        return pd.DataFrame(
            columns=["Season", "TeamID", "massey_consensus", "massey_n_systems"]
        )

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()
    df = df[df["SystemName"].isin(stable_systems)].copy()
    df = _zscore_by_season_system(df)

    mat = (
        df.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="z",
            aggfunc="mean",
        )
        .fillna(0.0)
        .reindex(columns=stable_systems, fill_value=0.0)
    )

    out = pd.DataFrame(index=mat.index)
    out["massey_consensus"] = mat.mean(axis=1)
    out["massey_n_systems"] = len(stable_systems)

    tourney = _load_tourney_compact(data_dir)
    t1 = np.minimum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    t2 = np.maximum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    y = (tourney["WTeamID"].to_numpy() == t1).astype(float)
    seasons = tourney["Season"].to_numpy().astype(int)

    system_scores: dict[str, float] = {}
    for sys in stable_systems:
        z_sys = mat[sys]
        z1 = np.array([float(z_sys.get((s, int(a)), 0.0)) for s, a in zip(seasons, t1)])
        z2 = np.array([float(z_sys.get((s, int(b)), 0.0)) for s, b in zip(seasons, t2)])
        system_scores[sys] = _pearson_corr(z1 - z2, y)

    top3 = sorted(stable_systems, key=lambda s: system_scores.get(s, 0.0), reverse=True)[:3]
    for sys in top3:
        out[f"massey_top3_{sys}"] = mat[sys]

    out = out.reset_index()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out

# from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


_MASSEY_FILENAME: Final[str] = "MMasseyOrdinals.csv"
_TOURNEY_FILENAME: Final[str] = "MNCAATourneyCompactResults.csv"
_RANKING_DAYNUM_FINAL: Final[int] = 133


def _load_massey_ordinals(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _MASSEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["RankingDayNum"] = pd.to_numeric(df["RankingDayNum"], errors="raise").astype(int)
    df["SystemName"] = df["SystemName"].astype(str)
    df["TeamID"] = pd.to_numeric(df["TeamID"], errors="raise").astype(int)
    df["OrdinalRank"] = pd.to_numeric(df["OrdinalRank"], errors="raise").astype(float)
    return df


def _load_tourney_compact(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _TOURNEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "WTeamID", "LTeamID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["WTeamID"] = pd.to_numeric(df["WTeamID"], errors="raise").astype(int)
    df["LTeamID"] = pd.to_numeric(df["LTeamID"], errors="raise").astype(int)
    return df


def list_stable_systems(data_dir: str, min_coverage: float = 0.8) -> list[str]:
    """Return SystemName values that survive the season-coverage filter."""
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must be in [0, 1]")

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()

    total_seasons = int(df["Season"].nunique())
    if total_seasons == 0:
        return []

    season_counts = df.groupby("SystemName")["Season"].nunique()
    keep = season_counts >= (min_coverage * total_seasons)
    return sorted(season_counts[keep].index.tolist())


def _zscore_by_season_system(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["Season", "SystemName"], sort=False)["OrdinalRank"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)

    # Lower rank is better; negate z so better teams have higher scores.
    z = -((df["OrdinalRank"] - mean) / std)
    out = df.copy()
    out["z"] = z.fillna(0.0)
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_massey_features(data_dir: str, min_coverage: float = 0.8) -> pd.DataFrame:
    """Return per-(Season, TeamID) Massey features (men).

    Output columns:
    - Season, TeamID
    - massey_consensus: mean z-score across kept systems
    - massey_n_systems: number of kept systems
    - massey_top3_<SystemName>: z-score for up to 3 most predictive systems
    """
    stable_systems = list_stable_systems(data_dir, min_coverage=min_coverage)
    if not stable_systems:
        return pd.DataFrame(
            columns=["Season", "TeamID", "massey_consensus", "massey_n_systems"]
        )

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()
    df = df[df["SystemName"].isin(stable_systems)].copy()
    df = _zscore_by_season_system(df)

    mat = (
        df.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="z",
            aggfunc="mean",
        )
        .fillna(0.0)
        .reindex(columns=stable_systems, fill_value=0.0)
    )

    out = pd.DataFrame(index=mat.index)
    out["massey_consensus"] = mat.mean(axis=1)
    out["massey_n_systems"] = len(stable_systems)

    tourney = _load_tourney_compact(data_dir)
    t1 = np.minimum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    t2 = np.maximum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    y = (tourney["WTeamID"].to_numpy() == t1).astype(float)
    seasons = tourney["Season"].to_numpy().astype(int)

    system_scores: dict[str, float] = {}
    for sys in stable_systems:
        z_sys = mat[sys]
        z1 = np.array([float(z_sys.get((s, int(a)), 0.0)) for s, a in zip(seasons, t1)])
        z2 = np.array([float(z_sys.get((s, int(b)), 0.0)) for s, b in zip(seasons, t2)])
        system_scores[sys] = _pearson_corr(z1 - z2, y)

    top3 = sorted(stable_systems, key=lambda s: system_scores.get(s, 0.0), reverse=True)[:3]
    for sys in top3:
        out[f"massey_top3_{sys}"] = mat[sys]

    out = out.reset_index()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out


_MASSEY_FILENAME: Final[str] = "MMasseyOrdinals.csv"
_TOURNEY_FILENAME: Final[str] = "MNCAATourneyCompactResults.csv"
_RANKING_DAYNUM_FINAL: Final[int] = 133


def _load_massey_ordinals(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _MASSEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["RankingDayNum"] = pd.to_numeric(df["RankingDayNum"], errors="raise").astype(int)
    df["SystemName"] = df["SystemName"].astype(str)
    df["TeamID"] = pd.to_numeric(df["TeamID"], errors="raise").astype(int)
    df["OrdinalRank"] = pd.to_numeric(df["OrdinalRank"], errors="raise").astype(float)
    return df


def _load_tourney_compact(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _TOURNEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "WTeamID", "LTeamID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["WTeamID"] = pd.to_numeric(df["WTeamID"], errors="raise").astype(int)
    df["LTeamID"] = pd.to_numeric(df["LTeamID"], errors="raise").astype(int)
    return df


def list_stable_systems(data_dir: str, min_coverage: float = 0.8) -> list[str]:
    """Return SystemName values that survive the season-coverage filter."""
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must be in [0, 1]")

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()

    seasons = sorted(df["Season"].unique().tolist())
    total_seasons = len(seasons)
    if total_seasons == 0:
        return []

    season_counts = df.groupby("SystemName")["Season"].nunique()
    keep = season_counts >= (min_coverage * total_seasons)
    return sorted(season_counts[keep].index.tolist())


def _zscore_by_season_system(df: pd.DataFrame) -> pd.DataFrame:
    """Add column `z` as sign-flipped z-score within each (Season, SystemName)."""
    grouped = df.groupby(["Season", "SystemName"], sort=False)["OrdinalRank"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)

    # Lower rank is better; negate z so better teams have higher scores.
    z = -((df["OrdinalRank"] - mean) / std)
    out = df.copy()
    out["z"] = z.fillna(0.0)
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_massey_features(data_dir: str, min_coverage: float = 0.8) -> pd.DataFrame:
    """Load per-(Season, TeamID) Massey features for men.

    Returns columns:
      Season, TeamID,
      massey_consensus, massey_n_systems,
      massey_top3_<SystemName> for up to 3 systems.
    """
    stable_systems = list_stable_systems(data_dir, min_coverage=min_coverage)
    if not stable_systems:
        return pd.DataFrame(
            columns=["Season", "TeamID", "massey_consensus", "massey_n_systems"]
        )

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()
    df = df[df["SystemName"].isin(stable_systems)].copy()
    df = _zscore_by_season_system(df)

    mat = (
        df.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="z",
            aggfunc="mean",
        )
        .fillna(0.0)
        .reindex(columns=stable_systems, fill_value=0.0)
    )

    out = pd.DataFrame(index=mat.index)
    out["massey_consensus"] = mat.mean(axis=1)
    out["massey_n_systems"] = len(stable_systems)

    tourney = _load_tourney_compact(data_dir)
    t1 = np.minimum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    t2 = np.maximum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    y = (tourney["WTeamID"].to_numpy() == t1).astype(float)
    seasons = tourney["Season"].to_numpy().astype(int)

    system_scores: dict[str, float] = {}
    for sys in stable_systems:
        z_sys = mat[sys]
        z1 = np.array([float(z_sys.get((s, int(a)), 0.0)) for s, a in zip(seasons, t1)])
        z2 = np.array([float(z_sys.get((s, int(b)), 0.0)) for s, b in zip(seasons, t2)])
        diff = z1 - z2
        system_scores[sys] = _pearson_corr(diff, y)

    top3 = sorted(stable_systems, key=lambda s: system_scores.get(s, 0.0), reverse=True)[:3]
    for sys in top3:
        out[f"massey_top3_{sys}"] = mat[sys]

    out = out.reset_index()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out

# from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


_MASSEY_FILENAME: Final[str] = "MMasseyOrdinals.csv"
_TOURNEY_FILENAME: Final[str] = "MNCAATourneyCompactResults.csv"
_RANKING_DAYNUM_FINAL: Final[int] = 133


def _load_massey_ordinals(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _MASSEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["RankingDayNum"] = pd.to_numeric(df["RankingDayNum"], errors="raise").astype(int)
    df["SystemName"] = df["SystemName"].astype(str)
    df["TeamID"] = pd.to_numeric(df["TeamID"], errors="raise").astype(int)
    df["OrdinalRank"] = pd.to_numeric(df["OrdinalRank"], errors="raise").astype(float)
    return df


def _load_tourney_compact(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _TOURNEY_FILENAME
    df = pd.read_csv(path)
    required = ["Season", "WTeamID", "LTeamID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["WTeamID"] = pd.to_numeric(df["WTeamID"], errors="raise").astype(int)
    df["LTeamID"] = pd.to_numeric(df["LTeamID"], errors="raise").astype(int)
    return df


def list_stable_systems(data_dir: str, min_coverage: float = 0.8) -> list[str]:
    """List Massey systems that survive the season-coverage filter."""
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must be in [0, 1]")

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()

    seasons = sorted(df["Season"].unique().tolist())
    total_seasons = len(seasons)
    if total_seasons == 0:
        return []

    season_counts = df.groupby("SystemName")["Season"].nunique()
    keep = season_counts >= (min_coverage * total_seasons)
    stable = sorted(season_counts[keep].index.tolist())
    return stable


def _zscore_by_season_system(df: pd.DataFrame) -> pd.DataFrame:
    """Add column `z` as sign-flipped z-score within each (Season, SystemName)."""
    grouped = df.groupby(["Season", "SystemName"], sort=False)["OrdinalRank"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)

    # Lower rank is better; negate z so better teams get higher scores.
    z = -((df["OrdinalRank"] - mean) / std)
    df = df.copy()
    df["z"] = z.fillna(0.0)
    return df


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_massey_features(data_dir: str, min_coverage: float = 0.8) -> pd.DataFrame:
    """Load per-team Massey features (men) with consensus and top-3 systems."""
    stable_systems = list_stable_systems(data_dir, min_coverage=min_coverage)
    if not stable_systems:
        return pd.DataFrame(
            columns=["Season", "TeamID", "massey_consensus", "massey_n_systems"]
        )

    df = _load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _RANKING_DAYNUM_FINAL].copy()
    df = df[df["SystemName"].isin(stable_systems)].copy()
    df = _zscore_by_season_system(df)

    # Matrix: index (Season, TeamID) and columns SystemName with z-scores.
    mat = df.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="z",
        aggfunc="mean",
    ).fillna(0.0)

    # Consensus mean across kept systems (missing treated as 0).
    mat = mat.reindex(columns=stable_systems, fill_value=0.0)
    consensus = mat.mean(axis=1)

    out = consensus.rename("massey_consensus").to_frame()
    out["massey_n_systems"] = len(stable_systems)

    # Compute top-3 predictive systems using tourney outcomes.
    tourney = _load_tourney_compact(data_dir)
    # Build labels in Kaggle convention: A = lower TeamID.
    t1 = np.minimum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    t2 = np.maximum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    y = (tourney["WTeamID"].to_numpy() == t1).astype(float)
    seasons = tourney["Season"].to_numpy().astype(int)

    system_scores: dict[str, float] = {}
    for sys in stable_systems:
        # Lookup z for t1/t2; missing => 0.
        z_sys = mat[sys]
        z1 = np.array([float(z_sys.get((s, int(a)), 0.0)) for s, a in zip(seasons, t1)])
        z2 = np.array([float(z_sys.get((s, int(b)), 0.0)) for s, b in zip(seasons, t2)])
        diff = z1 - z2
        system_scores[sys] = _pearson_corr(diff, y)

    top3 = sorted(stable_systems, key=lambda s: system_scores.get(s, 0.0), reverse=True)[:3]
    for sys in top3:
        out[f"massey_top3_{sys}"] = mat[sys]

    out = out.reset_index()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out

# from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .io import read_csv
from .paths import DATA_DIR


_SNAPSHOT_DAY_NUM = 133


def _massey_filename(sex: str) -> str:
    sex_norm = sex.strip().upper()
    if sex_norm not in {"M", "W"}:
        raise ValueError(f"sex must be 'M' or 'W' (got {sex!r})")
    return f"{sex_norm}MasseyOrdinals.csv"

def load_massey_strength(sex: str) -> dict[tuple[int, int], float]:
    """
    Return per-(Season, TeamID) Massey snapshot strength from day 133.

    Strength is a season-wise z-score of (average ordinal rank across systems):
        strength = -(rank - mean_rank) / std_rank
    where higher is better.
    """

    filename = _massey_filename(sex)
    path = DATA_DIR / filename

    if not path.exists():
        if filename.startswith("W"):
            return {}
        raise FileNotFoundError(f"Massey ordinals CSV not found: {path}")

    df = read_csv(filename)
    required_cols = {"Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"{filename} missing required column(s): {missing_list}")

    snap = df.loc[df["RankingDayNum"] == _SNAPSHOT_DAY_NUM, ["Season", "TeamID", "OrdinalRank"]]
    if snap.empty:
        return {}

    snap = snap.copy()
    snap["OrdinalRank"] = pd.to_numeric(snap["OrdinalRank"], errors="coerce")
    snap = snap.dropna(subset=["OrdinalRank"])
    if snap.empty:
        return {}

    team_rank = (
        snap.groupby(["Season", "TeamID"], as_index=False)["OrdinalRank"]
        .mean()
        .rename(columns={"OrdinalRank": "MeanOrdinalRank"})
    )

    season_stats = (
        team_rank.groupby("Season", as_index=False)["MeanOrdinalRank"]
        .agg(mean_rank="mean", std_rank=lambda s: float(s.std(ddof=0)))
        .set_index("Season")
    )

    def _strength_row(row: Mapping[str, object]) -> float:
        season = int(row["Season"])
        rank = float(row["MeanOrdinalRank"])
        mean_rank = float(season_stats.loc[season, "mean_rank"])
        std_rank = float(season_stats.loc[season, "std_rank"])
        if std_rank == 0.0 or pd.isna(std_rank):
            return 0.0
        return -((rank - mean_rank) / std_rank)

    team_rank["Strength"] = team_rank.apply(_strength_row, axis=1)

    strengths: dict[tuple[int, int], float] = {}
    for season, team_id, strength in team_rank[["Season", "TeamID", "Strength"]].itertuples(
        index=False, name=None
    ):
        strengths[(int(season), int(team_id))] = float(strength)

    return strengths


# ---------------------------------------------------------------------------
# Canonical implementations (override any accidental duplicates above)
# ---------------------------------------------------------------------------

from pathlib import Path as _Path
from typing import Final as _Final

import numpy as _np
import pandas as _pd

_CANON_MASSEY_FILENAME: _Final[str] = "MMasseyOrdinals.csv"
_CANON_TOURNEY_FILENAME: _Final[str] = "MNCAATourneyCompactResults.csv"
_CANON_RANKING_DAYNUM_FINAL: _Final[int] = 133


def _canon_load_massey_ordinals(data_dir: str) -> _pd.DataFrame:
    path = _Path(data_dir) / _CANON_MASSEY_FILENAME
    df = _pd.read_csv(path)
    required = ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = _pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["RankingDayNum"] = _pd.to_numeric(df["RankingDayNum"], errors="raise").astype(int)
    df["SystemName"] = df["SystemName"].astype(str)
    df["TeamID"] = _pd.to_numeric(df["TeamID"], errors="raise").astype(int)
    df["OrdinalRank"] = _pd.to_numeric(df["OrdinalRank"], errors="raise").astype(float)
    return df


def _canon_load_tourney_compact(data_dir: str) -> _pd.DataFrame:
    path = _Path(data_dir) / _CANON_TOURNEY_FILENAME
    df = _pd.read_csv(path)
    required = ["Season", "WTeamID", "LTeamID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[required].copy()
    df["Season"] = _pd.to_numeric(df["Season"], errors="raise").astype(int)
    df["WTeamID"] = _pd.to_numeric(df["WTeamID"], errors="raise").astype(int)
    df["LTeamID"] = _pd.to_numeric(df["LTeamID"], errors="raise").astype(int)
    return df


def list_stable_systems(data_dir: str, min_coverage: float = 0.8) -> list[str]:
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must be in [0, 1]")

    df = _canon_load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _CANON_RANKING_DAYNUM_FINAL].copy()

    total_seasons = int(df["Season"].nunique())
    if total_seasons == 0:
        return []

    season_counts = df.groupby("SystemName")["Season"].nunique()
    keep = season_counts >= (min_coverage * total_seasons)
    return sorted(season_counts[keep].index.tolist())


def _canon_zscore_by_season_system(df: _pd.DataFrame) -> _pd.DataFrame:
    grouped = df.groupby(["Season", "SystemName"], sort=False)["OrdinalRank"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, _np.nan)

    z = -((df["OrdinalRank"] - mean) / std)
    out = df.copy()
    out["z"] = z.fillna(0.0)
    return out


def _canon_pearson_corr(x: _np.ndarray, y: _np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if _np.allclose(x, x[0]) or _np.allclose(y, y[0]):
        return 0.0
    return float(_np.corrcoef(x, y)[0, 1])


def load_massey_features(data_dir: str, min_coverage: float = 0.8) -> _pd.DataFrame:
    stable_systems = list_stable_systems(data_dir, min_coverage=min_coverage)
    if not stable_systems:
        return _pd.DataFrame(
            columns=["Season", "TeamID", "massey_consensus", "massey_n_systems"]
        )

    df = _canon_load_massey_ordinals(data_dir)
    df = df[df["RankingDayNum"] == _CANON_RANKING_DAYNUM_FINAL].copy()
    df = df[df["SystemName"].isin(stable_systems)].copy()
    df = _canon_zscore_by_season_system(df)

    mat = (
        df.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="z",
            aggfunc="mean",
        )
        .fillna(0.0)
        .reindex(columns=stable_systems, fill_value=0.0)
    )

    out = _pd.DataFrame(index=mat.index)
    out["massey_consensus"] = mat.mean(axis=1)
    out["massey_n_systems"] = len(stable_systems)

    tourney = _canon_load_tourney_compact(data_dir)
    t1 = _np.minimum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    t2 = _np.maximum(tourney["WTeamID"].to_numpy(), tourney["LTeamID"].to_numpy())
    y = (tourney["WTeamID"].to_numpy() == t1).astype(float)
    seasons = tourney["Season"].to_numpy().astype(int)

    system_scores: dict[str, float] = {}
    for sys in stable_systems:
        z_sys = mat[sys]
        z1 = _np.array([float(z_sys.get((s, int(a)), 0.0)) for s, a in zip(seasons, t1)])
        z2 = _np.array([float(z_sys.get((s, int(b)), 0.0)) for s, b in zip(seasons, t2)])
        system_scores[sys] = _canon_pearson_corr(z1 - z2, y)

    top3 = sorted(stable_systems, key=lambda s: system_scores.get(s, 0.0), reverse=True)[:3]
    for sys in top3:
        out[f"massey_top3_{sys}"] = mat[sys]

    out = out.reset_index()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out

