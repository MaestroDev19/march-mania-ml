from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from .elo import compute_elo_men, compute_elo_women
from .features import compute_efficiency, compute_four_factors, encode_matchups
from .massey import load_massey_features
from .paths import get_data_dir

_logger = logging.getLogger(__name__)

_PRED_MIN = 0.05
_PRED_MAX = 0.975


def _seed_num(seed_str: str) -> int | None:
    if not isinstance(seed_str, str):
        return None
    digits = "".join([c for c in seed_str if c.isdigit()])
    if not digits:
        return None
    return int(digits[:2]) if len(digits) >= 2 else int(digits)


def build_feature_store(data_dir: Path | str | None = None) -> dict[str, Any]:
    """Rebuild the same feature_store used for Stage-2 submission generation."""
    root = Path(data_dir) if data_dir is not None else get_data_dir()
    elo_m = compute_elo_men(str(root))
    elo_w = compute_elo_women(str(root))
    elo_all = dict(elo_m)
    elo_all.update(elo_w)

    m_det = pd.read_csv(root / "MRegularSeasonDetailedResults.csv")
    w_det = pd.read_csv(root / "WRegularSeasonDetailedResults.csv")
    efficiency = pd.concat([compute_efficiency(m_det), compute_efficiency(w_det)], ignore_index=True)
    four_factors = pd.concat([compute_four_factors(m_det), compute_four_factors(w_det)], ignore_index=True)
    massey = load_massey_features(str(root), min_coverage=0.8)

    seeds_m_raw = pd.read_csv(root / "MNCAATourneySeeds.csv")[["Season", "TeamID", "Seed"]].copy()
    seeds_w_raw = pd.read_csv(root / "WNCAATourneySeeds.csv")[["Season", "TeamID", "Seed"]].copy()
    seeds_m = (
        seeds_m_raw.assign(
            Season=seeds_m_raw["Season"].astype(int),
            TeamID=seeds_m_raw["TeamID"].astype(int),
            Seed=seeds_m_raw["Seed"].astype(str).map(_seed_num),
        )
        .dropna(subset=["Seed"])
        .reset_index(drop=True)
    )
    seeds_w = (
        seeds_w_raw.assign(
            Season=seeds_w_raw["Season"].astype(int),
            TeamID=seeds_w_raw["TeamID"].astype(int),
            Seed=seeds_w_raw["Seed"].astype(str).map(_seed_num),
        )
        .dropna(subset=["Seed"])
        .reset_index(drop=True)
    )

    return {
        "elo": elo_all,
        "efficiency": efficiency,
        "four_factors": four_factors,
        "massey": massey,
        "seeds_m": seeds_m[["Season", "TeamID", "Seed"]].copy(),
        "seeds_w": seeds_w[["Season", "TeamID", "Seed"]].copy(),
    }


def _select_feature_columns(encoded_df: pd.DataFrame) -> list[str]:
    cols = [c for c in encoded_df.columns if c != "label"]
    if not cols:
        raise ValueError("Encoded matchup DataFrame has no feature columns")
    return cols


def _predict_raw_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("model must implement predict_proba(X) -> array[n,2]")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"predict_proba returned unexpected shape: {proba.shape}")
    return proba[:, 1].astype(float, copy=False)


def _transform_calibrated(calibrator: Any, gender: str, raw_scores: np.ndarray) -> np.ndarray:
    if gender == "M":
        if not hasattr(calibrator, "transform_men"):
            raise TypeError("calibrator_men must implement transform_men(raw_scores)")
        return np.asarray(calibrator.transform_men(raw_scores), dtype=float)
    if not hasattr(calibrator, "transform_women"):
        raise TypeError("calibrator_women must implement transform_women(raw_scores)")
    return np.asarray(calibrator.transform_women(raw_scores), dtype=float)


@dataclass
class LoadedModels:
    model_men: Any
    model_women: Any
    calibrator_men: Any
    calibrator_women: Any
    feature_store: dict[str, Any]
    teams_m: pd.DataFrame
    teams_w: pd.DataFrame


def load_models_for_inference(
    *,
    models_dir: Path | None = None,
    data_dir: Path | None = None,
) -> LoadedModels:
    """Load tuned XGB + calibrators and rebuild feature_store from disk data."""
    src_dir = Path(__file__).resolve().parent
    mdir = models_dir if models_dir is not None else src_dir / "models"
    ddir = Path(data_dir) if data_dir is not None else get_data_dir()

    paths = {
        "men": mdir / "best_men.pkl",
        "women": mdir / "best_women.pkl",
        "cal_m": mdir / "best_calibrator_men.joblib",
        "cal_w": mdir / "best_calibrator_women.joblib",
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Missing model artifacts:\n{msg}")

    feature_store = build_feature_store(ddir)
    teams_m = pd.read_csv(ddir / "MTeams.csv")
    teams_w = pd.read_csv(ddir / "WTeams.csv")

    return LoadedModels(
        model_men=joblib_load(paths["men"]),
        model_women=joblib_load(paths["women"]),
        calibrator_men=joblib_load(paths["cal_m"]),
        calibrator_women=joblib_load(paths["cal_w"]),
        feature_store=feature_store,
        teams_m=teams_m,
        teams_w=teams_w,
    )


def predict_matchup(
    loaded: LoadedModels,
    *,
    season: int,
    team_lower_id: int,
    team_higher_id: int,
) -> dict[str, Any]:
    """
    P(team with lower TeamID wins), matching Kaggle ID convention SSSS_lower_higher.
    """
    if team_lower_id >= team_higher_id:
        raise ValueError("team_lower_id must be strictly less than team_higher_id")

    gender = "M" if team_lower_id < 3000 else "W"
    matchups_df = pd.DataFrame(
        {"Season": [int(season)], "Team1ID": [int(team_lower_id)], "Team2ID": [int(team_higher_id)]}
    )
    seeds_df = loaded.feature_store["seeds_m"] if gender == "M" else loaded.feature_store["seeds_w"]
    encoded = encode_matchups(
        matchups_df=matchups_df,
        elo_dict=loaded.feature_store["elo"],
        efficiency_df=loaded.feature_store["efficiency"],
        four_factors_df=loaded.feature_store["four_factors"],
        massey_df=loaded.feature_store["massey"],
        seeds_df=seeds_df,
        label_col=None,
    )
    feature_cols = _select_feature_columns(encoded)
    X = encoded[feature_cols].copy()
    if X.isna().any().any():
        X = X.fillna(0.0)
        _logger.warning("Some encoded feature values were missing; filled with 0.")

    model = loaded.model_men if gender == "M" else loaded.model_women
    calibrator = loaded.calibrator_men if gender == "M" else loaded.calibrator_women

    raw = _predict_raw_scores(model, X)
    calibrated = _transform_calibrated(calibrator, gender, raw)
    pred = float(np.clip(calibrated[0], _PRED_MIN, _PRED_MAX))

    name_m = dict(zip(loaded.teams_m["TeamID"].astype(int), loaded.teams_m["TeamName"].astype(str)))
    name_w = dict(zip(loaded.teams_w["TeamID"].astype(int), loaded.teams_w["TeamName"].astype(str)))
    names = name_m if gender == "M" else name_w
    t1 = names.get(int(team_lower_id), "UNKNOWN")
    t2 = names.get(int(team_higher_id), "UNKNOWN")

    return {
        "id": f"{int(season)}_{int(team_lower_id)}_{int(team_higher_id)}",
        "season": int(season),
        "team1_id": int(team_lower_id),
        "team2_id": int(team_higher_id),
        "team1_name": t1,
        "team2_name": t2,
        "gender": gender,
        "pred": pred,
        "prob_lower_team_wins": pred,
        "elo_diff": float(encoded["elo_diff"].iloc[0]),
        "adjEM_diff": float(encoded["adjEM_diff"].iloc[0]),
        "seed_diff": float(encoded["seed_diff"].iloc[0]),
        "massey_diff": float(encoded["massey_diff"].iloc[0]),
    }


def normalize_team_pair(team_a_id: int, team_b_id: int) -> tuple[int, int]:
    """Return (lower_id, higher_id) for Kaggle matchup IDs."""
    a, b = int(team_a_id), int(team_b_id)
    if a == b:
        raise ValueError("team_a_id and team_b_id must differ")
    return (a, b) if a < b else (b, a)
