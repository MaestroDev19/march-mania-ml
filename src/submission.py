 # from __future__ import annotations  # duplicate/legacy block (kept to avoid SyntaxError)

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import pandas as pd

from joblib import load as joblib_load
from sklearn.linear_model import LogisticRegression

from .features import encode_matchups, encode_matchups_symmetric
from .io import read_csv
from .nn_blend import predict_nn_mlp_proba

_logger = logging.getLogger(__name__)

_PRED_MIN: float = 0.05
_PRED_MAX: float = 0.975

_COLS_LR: list[str] = ["elo_diff", "seed_diff"]
_TRAIN_SEASONS = np.arange(2010, 2023)
_RECENCY_ANCHOR_YEAR: int = 2024


def _maybe_load_blend_artifacts() -> tuple[bool, dict[str, Any]]:
    """
    Returns:
      (available, artifacts)
    """
    models_dir = Path(__file__).resolve().parent / "models"

    weights_path = models_dir / "best_blend_weights_lr_xgb_nn.joblib"
    nn_men_path = models_dir / "best_nn_men.pkl"
    nn_women_path = models_dir / "best_nn_women.pkl"
    xgb_men_path = models_dir / "best_men.pkl"
    xgb_women_path = models_dir / "best_women.pkl"

    required = [
        weights_path,
        nn_men_path,
        nn_women_path,
        xgb_men_path,
        xgb_women_path,
    ]
    if not all(p.exists() for p in required):
        return False, {}

    weights = joblib_load(weights_path)
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False
    return True, {
        "models_dir": models_dir,
        "weights": weights,
        "nn_men_path": nn_men_path,
        "nn_women_path": nn_women_path,
        "torch_available": torch_available,
    }


def _build_lr_for_gender(*, feature_store: dict[str, Any], gender: str) -> LogisticRegression:
    tourney_csv = "MNCAATourneyCompactResults.csv" if gender == "M" else "WNCAATourneyCompactResults.csv"
    seeds_df = feature_store["seeds_m"] if gender == "M" else feature_store["seeds_w"]

    tourney_df = read_csv(tourney_csv)
    d = tourney_df[tourney_df["Season"].astype(int).isin(_TRAIN_SEASONS)].copy()

    season = d["Season"].astype(int).to_numpy()
    w_team = d["WTeamID"].astype(int).to_numpy()
    l_team = d["LTeamID"].astype(int).to_numpy()

    team1 = np.minimum(w_team, l_team)
    team2 = np.maximum(w_team, l_team)
    y = (w_team == team1).astype(int)

    matchups_df = pd.DataFrame({"Season": season, "Team1ID": team1, "Team2ID": team2})

    # Same recency weighting scheme used in the training notebooks.
    w = np.exp(-0.1 * (_RECENCY_ANCHOR_YEAR - season)).astype(float)

    encoded = encode_matchups_symmetric(
        matchups_df=matchups_df,
        elo_dict=feature_store["elo"],
        efficiency_df=feature_store["efficiency"],
        four_factors_df=feature_store["four_factors"],
        massey_df=feature_store["massey"],
        seeds_df=seeds_df,
        label_col=pd.Series(y, name="label"),
    )

    X_lr_train = encoded[_COLS_LR].copy()
    y_train = encoded["label"].astype(int).to_numpy()
    w_train = np.asarray(w, dtype=float).repeat(2)

    if not (len(X_lr_train) == len(y_train) == len(w_train)):
        raise ValueError(
            "LR training length mismatch: "
            f"X={len(X_lr_train)} y={len(y_train)} w={len(w_train)}"
        )

    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_lr_train, y_train, sample_weight=w_train)
    return lr


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_submission_id(sample_sub: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in sample_sub.columns:
        raise ValueError("sample_sub must contain column 'ID'")

    parts = sample_sub["ID"].astype(str).str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError("ID format must be SSSS_LOWERID_HIGHERID")

    out = sample_sub.copy()
    out["Season"] = parts[0].astype(int)
    out["Team1ID"] = parts[1].astype(int)
    out["Team2ID"] = parts[2].astype(int)
    out["gender"] = np.where(out["Team1ID"].to_numpy() < 3000, "M", "W")
    out["Pred"] = np.nan

    bad_order = (out["Team1ID"] >= out["Team2ID"]).to_numpy()
    if np.any(bad_order):
        examples = out.loc[bad_order, ["ID", "Team1ID", "Team2ID"]].head(5).to_dict("records")
        raise ValueError(f"Expected Team1ID < Team2ID. Examples: {examples}")

    return out


def _ensure_feature_store_keys(feature_store: dict[str, Any]) -> None:
    required = {"elo", "efficiency", "four_factors", "massey", "seeds_m", "seeds_w"}
    missing = required.difference(feature_store.keys())
    if missing:
        raise ValueError(f"feature_store missing required keys: {sorted(missing)}")


def _predict_raw_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("model must implement predict_proba(X)")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"predict_proba returned unexpected shape: {proba.shape}")
    return proba[:, 1]


def _transform_calibrated(calibrator: Any, gender: str, raw_scores: np.ndarray) -> np.ndarray:
    if gender == "M":
        if not hasattr(calibrator, "transform_men"):
            raise TypeError("calibrator_men must implement transform_men(raw_scores)")
        return np.asarray(calibrator.transform_men(raw_scores), dtype=float)
    if not hasattr(calibrator, "transform_women"):
        raise TypeError("calibrator_women must implement transform_women(raw_scores)")
    return np.asarray(calibrator.transform_women(raw_scores), dtype=float)


def _select_feature_columns(encoded_df: pd.DataFrame) -> list[str]:
    feature_cols = [c for c in encoded_df.columns if c != "label"]
    if not feature_cols:
        raise ValueError("Encoded matchup DataFrame has no feature columns")
    return feature_cols


def generate_submissions(
    sample_sub_path: str,
    feature_store: dict[str, Any],
    model_men: Any,
    model_women: Any,
    calibrator_men: Any,
    calibrator_women: Any,
    teams_m_df: pd.DataFrame,
    teams_w_df: pd.DataFrame,
    output_dir: str = "submissions/",
) -> Tuple[str, str]:
    """Generate minimal and enriched submissions per `00_prompts.md`."""
    _ensure_feature_store_keys(feature_store)

    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"sample_sub_path not found: {sample_sub_path}")

    # If tuned 3-way blend artifacts exist, we can calibrate-after-blend end-to-end.
    blend_available, blend_artifacts = _maybe_load_blend_artifacts()
    blend_weights = blend_artifacts.get("weights")
    torch_available = bool(blend_artifacts.get("torch_available", False))
    # Unit tests pass tiny dummy `feature_store` fixtures; avoid triggering the NN
    # inference path unless we have real feature data to train LR and run NN.
    blend_available = bool(blend_available) and torch_available and bool(feature_store.get("elo")) and bool(
        len(feature_store.get("efficiency", []))
    ) and bool(len(feature_store.get("four_factors", []))) and bool(
        len(feature_store.get("seeds_m", []))
    ) and bool(len(feature_store.get("seeds_w", [])))

    # If the user passed blend-trained TournamentCalibrators but torch isn't available,
    # fail loudly instead of silently using XGB-only raw scores.
    if bool(blend_available) is False and bool(blend_artifacts):
        w_nn = float(blend_weights.get("w_nn", 0.0)) if isinstance(blend_weights, dict) else 0.0
        looks_like_tournament_cal = hasattr(calibrator_men, "calibrator_men") or hasattr(
            calibrator_women, "calibrator_women"
        )
        if not torch_available and looks_like_tournament_cal and w_nn > 0:
            raise RuntimeError(
                "NN-based 3-way blend requires PyTorch (`torch`) for inference. "
                "Install torch or regenerate calibrators for the XGB-only pipeline."
            )

    trained_lr_models: dict[str, LogisticRegression] = {}
    loaded_nn_models: dict[str, Any] = {}

    os.makedirs(output_dir, exist_ok=True)

    sample_sub = pd.read_csv(sample_sub_path)
    base = _parse_submission_id(sample_sub)
    n_rows = len(sample_sub)

    # Columns required by the enriched submission.
    for col in ["elo_diff", "adjEM_diff", "seed_diff", "massey_diff"]:
        base[col] = np.nan

    def _process_gender(gender: str) -> None:
        subset = base[base["gender"] == gender].copy()
        if subset.empty:
            return

        matchups_df = subset[["Season", "Team1ID", "Team2ID"]].copy()

        seeds_df = feature_store["seeds_m"] if gender == "M" else feature_store["seeds_w"]
        encoded = encode_matchups(
            matchups_df=matchups_df,
            elo_dict=feature_store["elo"],
            efficiency_df=feature_store["efficiency"],
            four_factors_df=feature_store["four_factors"],
            massey_df=feature_store["massey"],
            seeds_df=seeds_df,
            label_col=None,
        )

        feature_cols = _select_feature_columns(encoded)
        X = encoded[feature_cols].copy()
        if X.isna().any().any():
            X = X.fillna(0.0)
            _logger.warning("Some encoded feature values were missing; filled with 0.")

        model = model_men if gender == "M" else model_women

        if blend_available and blend_weights is not None:
            # Calibrator was trained on blended raw probabilities, not a single XGB voice.
            # Compute LR/XGB/NN raw probabilities and blend them with tuned weights.
            if gender not in trained_lr_models:
                trained_lr_models[gender] = _build_lr_for_gender(
                    feature_store=feature_store, gender=gender
                )
            lr_model = trained_lr_models[gender]

            if gender not in loaded_nn_models:
                nn_path = (
                    blend_artifacts["nn_men_path"] if gender == "M" else blend_artifacts["nn_women_path"]
                )
                loaded_nn_models[gender] = joblib_load(nn_path)
            nn_model = loaded_nn_models[gender]

            xgb_prob = _predict_raw_scores(model, X)
            lr_prob = lr_model.predict_proba(X[_COLS_LR])[:, 1]
            nn_prob = predict_nn_mlp_proba(nn_model, X)

            w_lr = float(blend_weights["w_lr"])
            w_xgb = float(blend_weights["w_xgb"])
            w_nn = float(blend_weights["w_nn"])

            raw_scores = np.clip(w_lr * lr_prob + w_xgb * xgb_prob + w_nn * nn_prob, 0.0, 1.0)
        else:
            raw_scores = _predict_raw_scores(model, X)

        calibrator = calibrator_men if gender == "M" else calibrator_women
        calibrated = _transform_calibrated(calibrator, gender, raw_scores)
        pred = np.clip(calibrated, _PRED_MIN, _PRED_MAX)

        base.loc[subset.index, "Pred"] = pred
        base.loc[subset.index, "elo_diff"] = encoded["elo_diff"].to_numpy(dtype=float, copy=False)
        base.loc[subset.index, "adjEM_diff"] = encoded["adjEM_diff"].to_numpy(dtype=float, copy=False)
        base.loc[subset.index, "seed_diff"] = encoded["seed_diff"].to_numpy(dtype=float, copy=False)
        base.loc[subset.index, "massey_diff"] = encoded["massey_diff"].to_numpy(dtype=float, copy=False)

    _process_gender("M")
    _process_gender("W")

    if base["Pred"].isna().any():
        missing = int(base["Pred"].isna().sum())
        raise RuntimeError(f"Predictions missing for {missing} rows")

    submission_min = base[["ID", "Pred"]].copy()
    submission_min["Pred"] = submission_min["Pred"].astype(float)
    assert len(submission_min) == n_rows, "Row count mismatch vs sample submission"

    ts = _timestamp()
    path_min = os.path.join(output_dir, f"submission_minimal_v{ts}.csv")
    submission_min.to_csv(path_min, index=False)

    # Names for enriched submission.
    name_m = dict(zip(teams_m_df["TeamID"].astype(int), teams_m_df["TeamName"].astype(str)))
    name_w = dict(zip(teams_w_df["TeamID"].astype(int), teams_w_df["TeamName"].astype(str)))

    team1_name = np.where(
        base["gender"].to_numpy() == "M",
        base["Team1ID"].astype(int).map(name_m),
        base["Team1ID"].astype(int).map(name_w),
    )
    team2_name = np.where(
        base["gender"].to_numpy() == "M",
        base["Team2ID"].astype(int).map(name_m),
        base["Team2ID"].astype(int).map(name_w),
    )
    # Replace possible NaNs from missing lookups.
    team1_name = pd.Series(team1_name).fillna("UNKNOWN").astype(str).to_numpy()
    team2_name = pd.Series(team2_name).fillna("UNKNOWN").astype(str).to_numpy()

    submission_enriched = base.copy()
    submission_enriched["Team1Name"] = team1_name
    submission_enriched["Team2Name"] = team2_name
    submission_enriched = submission_enriched[
        [
            "ID",
            "Season",
            "Team1ID",
            "Team1Name",
            "Team2ID",
            "Team2Name",
            "Pred",
            "elo_diff",
            "adjEM_diff",
            "seed_diff",
            "massey_diff",
            "gender",
        ]
    ].copy()

    assert len(submission_enriched) == n_rows, "Row count mismatch vs sample submission"
    assert not submission_enriched["Pred"].isna().any(), "NaNs found in Pred column"
    submission_enriched["Pred"] = submission_enriched["Pred"].astype(float)

    path_enriched = os.path.join(output_dir, f"submission_enriched_v{ts}.csv")
    submission_enriched.to_csv(path_enriched, index=False)

    # Summary prints (no strict testing).
    preds = submission_enriched["Pred"].to_numpy(dtype=float)
    print(f"Wrote {len(preds)} rows")
    print(f"Pred mean={preds.mean():.6f} std={preds.std(ddof=0):.6f}")

    bins = [0.0, 0.4, 0.6, 1.0]
    labels = ["[0,0.4)", "[0.4,0.6)", "[0.6,1.0)"]
    dist = pd.cut(preds, bins=bins, right=False, labels=labels).value_counts()
    for lab in labels:
        count = int(dist.loc[lab]) if lab in dist.index else 0
        print(f"Preds {lab}: {count}")

    conf = np.abs(preds - 0.5)
    order = np.argsort(-conf)[: min(10, len(preds))]
    top = submission_enriched.iloc[order][["ID", "Team1Name", "Team2Name", "gender", "Pred"]]
    print("Top confident predictions:")
    print(top.to_string(index=False))

    return path_min, path_enriched


if __name__ == "__main__":
    class _DummyModel:
        def __init__(self, p: float) -> None:
            self._p = float(p)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            n = int(X.shape[0])
            p = self._p
            return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])

    class _DummyCal:
        def __init__(self, scale: float) -> None:
            self._scale = float(scale)

        def transform_men(self, raw_scores: np.ndarray) -> np.ndarray:
            return np.asarray(raw_scores) * self._scale

        def transform_women(self, raw_scores: np.ndarray) -> np.ndarray:
            return np.asarray(raw_scores) * self._scale

    sample_path = os.path.join("data", "SampleSubmissionStage2.csv")
    teams_m_path = os.path.join("data", "MTeams.csv")
    teams_w_path = os.path.join("data", "WTeams.csv")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing {sample_path}")
    if not os.path.exists(teams_m_path):
        raise FileNotFoundError(f"Missing {teams_m_path}")
    if not os.path.exists(teams_w_path):
        raise FileNotFoundError(f"Missing {teams_w_path}")

    teams_m_df = pd.read_csv(teams_m_path)
    teams_w_df = pd.read_csv(teams_w_path)

    feature_store = {
        "elo": {},
        "efficiency": pd.DataFrame(columns=["Season", "TeamID", "AdjEM"]),
        "four_factors": pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "eFG_pct",
                "TO_pct",
                "OR_pct",
                "FTRate",
                "opp_eFG_pct",
                "opp_TO_pct",
                "opp_OR_pct",
                "opp_FTRate",
            ]
        ),
        "massey": pd.DataFrame(columns=["Season", "TeamID", "massey_consensus"]),
        "seeds_m": pd.DataFrame(columns=["Season", "TeamID", "Seed"]),
        "seeds_w": pd.DataFrame(columns=["Season", "TeamID", "Seed"]),
    }

    print("Running dummy generate_submissions()...")
    generate_submissions(
        sample_sub_path=sample_path,
        feature_store=feature_store,
        model_men=_DummyModel(p=0.5),
        model_women=_DummyModel(p=0.5),
        calibrator_men=_DummyCal(scale=1.0),
        calibrator_women=_DummyCal(scale=1.0),
        teams_m_df=teams_m_df,
        teams_w_df=teams_w_df,
        output_dir="submissions",
    )

# from __future__ import annotations  # duplicate/legacy block (kept to avoid SyntaxError)

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .features import encode_matchups

_logger = logging.getLogger(__name__)

_PRED_MIN: float = 0.05
_PRED_MAX: float = 0.975


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_submission_id(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise ValueError("sample_sub must contain column 'ID'")

    parts = df["ID"].astype(str).str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError("ID format must be SSSS_LOWERID_HIGHERID")

    out = df.copy()
    out["Season"] = parts[0].astype(int)
    out["Team1ID"] = parts[1].astype(int)
    out["Team2ID"] = parts[2].astype(int)
    out["Pred"] = np.nan

    bad_order = (out["Team1ID"] >= out["Team2ID"]).to_numpy()
    if np.any(bad_order):
        bad_rows = out.loc[bad_order, ["ID", "Team1ID", "Team2ID"]].head(5).to_dict("records")
        raise ValueError(f"Expected Team1ID < Team2ID always. Examples: {bad_rows}")

    out["gender"] = np.where(out["Team1ID"].to_numpy() < 3000, "M", "W")
    return out


def _ensure_feature_store_keys(feature_store: dict[str, Any]) -> None:
    required = {"elo", "efficiency", "four_factors", "massey", "seeds_m", "seeds_w"}
    missing = required.difference(feature_store.keys())
    if missing:
        raise ValueError(f"feature_store missing required keys: {sorted(missing)}")


def _predict_raw_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("model must implement predict_proba(X) -> array[n,2]")

    proba = model.predict_proba(X)
    proba_arr = np.asarray(proba, dtype=float)
    if proba_arr.ndim != 2 or proba_arr.shape[1] < 2:
        raise ValueError(f"predict_proba returned unexpected shape: {proba_arr.shape}")
    return proba_arr[:, 1].astype(float, copy=False)


def _transform_calibrated(calibrator: Any, gender: str, raw_scores: np.ndarray) -> np.ndarray:
    if gender == "M":
        if not hasattr(calibrator, "transform_men"):
            raise TypeError("calibrator_men must implement transform_men(raw_scores)")
        return np.asarray(calibrator.transform_men(raw_scores), dtype=float)
    if not hasattr(calibrator, "transform_women"):
        raise TypeError("calibrator_women must implement transform_women(raw_scores)")
    return np.asarray(calibrator.transform_women(raw_scores), dtype=float)


def _select_feature_columns(encoded_df: pd.DataFrame) -> list[str]:
    # encode_matchups always returns a `label` column when label_col is None.
    cols = [c for c in encoded_df.columns if c != "label"]
    if not cols:
        raise ValueError("Encoded matchup DataFrame has no feature columns")
    return cols


def generate_submissions(
    sample_sub_path: str,
    feature_store: dict[str, Any],
    model_men: Any,
    model_women: Any,
    calibrator_men: Any,
    calibrator_women: Any,
    teams_m_df: pd.DataFrame,
    teams_w_df: pd.DataFrame,
    output_dir: str = "submissions/",
) -> Tuple[str, str]:
    """Generate competition submission files (minimal and enriched)."""
    _ensure_feature_store_keys(feature_store)

    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"sample_sub_path not found: {sample_sub_path}")

    os.makedirs(output_dir, exist_ok=True)

    sample_sub = pd.read_csv(sample_sub_path)
    if "ID" not in sample_sub.columns:
        raise ValueError("sample_sub must have an 'ID' column")

    base = _parse_submission_id(sample_sub)

    # Store per-row enriched diff features.
    for col in ["elo_diff", "adjEM_diff", "seed_diff", "massey_diff"]:
        base[col] = np.nan

    def _process_gender(gender: str) -> None:
        nonlocal base
        subset = base[base["gender"] == gender].copy()
        if subset.empty:
            return

        matchups_df = subset[["Season", "Team1ID", "Team2ID"]].copy()

        seeds_df = feature_store["seeds_m"] if gender == "M" else feature_store["seeds_w"]
        encoded = encode_matchups(
            matchups_df=matchups_df,
            elo_dict=feature_store["elo"],
            efficiency_df=feature_store["efficiency"],
            four_factors_df=feature_store["four_factors"],
            massey_df=feature_store["massey"],
            seeds_df=seeds_df,
            label_col=None,
        )

        feature_cols = _select_feature_columns(encoded)
        X = encoded[feature_cols].copy()
        if X.isna().any().any():
            X = X.fillna(0.0)
            _logger.warning("Some encoded feature values were missing; filled with 0.")

        model = model_men if gender == "M" else model_women
        raw_scores = _predict_raw_scores(model, X)

        calibrator = calibrator_men if gender == "M" else calibrator_women
        calibrated = _transform_calibrated(calibrator, gender, raw_scores)

        pred = np.clip(calibrated, _PRED_MIN, _PRED_MAX)

        # Assign predictions back to base by index alignment.
        base.loc[subset.index, "Pred"] = pred

        # Attach only the enriched columns required by the spec.
        base.loc[subset.index, "elo_diff"] = encoded["elo_diff"].to_numpy(dtype=float, copy=False)
        base.loc[subset.index, "adjEM_diff"] = encoded["adjEM_diff"].to_numpy(dtype=float, copy=False)
        base.loc[subset.index, "seed_diff"] = encoded["seed_diff"].to_numpy(dtype=float, copy=False)
        base.loc[subset.index, "massey_diff"] = encoded["massey_diff"].to_numpy(dtype=float, copy=False)

    _process_gender("M")
    _process_gender("W")

    if base["Pred"].isna().any():
        missing_count = int(base["Pred"].isna().sum())
        raise RuntimeError(f"Predictions missing for {missing_count} rows")

    # Minimal submission: exactly ID and Pred.
    submission_min = base[["ID", "Pred"]].copy()
    assert len(submission_min) == len(sample_sub), "Row count mismatch vs sample submission"
    submission_min = submission_min.assign(Pred=submission_min["Pred"].astype(float))

    ts = _timestamp()
    path_min = os.path.join(output_dir, f"submission_minimal_v{ts}.csv")
    submission_min.to_csv(path_min, index=False)

    # Enriched submission.
    name_m = dict(zip(teams_m_df["TeamID"].astype(int), teams_m_df["TeamName"].astype(str)))
    name_w = dict(zip(teams_w_df["TeamID"].astype(int), teams_w_df["TeamName"].astype(str)))

    team1_name = []
    team2_name = []
    for _, row in base.iterrows():
        if row["gender"] == "M":
            team1_name.append(name_m.get(int(row["Team1ID"]), "UNKNOWN"))
            team2_name.append(name_m.get(int(row["Team2ID"]), "UNKNOWN"))
        else:
            team1_name.append(name_w.get(int(row["Team1ID"]), "UNKNOWN"))
            team2_name.append(name_w.get(int(row["Team2ID"]), "UNKNOWN"))

    submission_enriched = base.copy()
    submission_enriched["Team1Name"] = team1_name
    submission_enriched["Team2Name"] = team2_name

    # Ensure correct output columns per prompt.
    submission_enriched = submission_enriched[
        [
            "ID",
            "Season",
            "Team1ID",
            "Team1Name",
            "Team2ID",
            "Team2Name",
            "Pred",
            "elo_diff",
            "adjEM_diff",
            "seed_diff",
            "massey_diff",
            "gender",
        ]
    ].copy()

    assert len(submission_enriched) == len(sample_sub), "Row count mismatch vs sample submission"
    assert not submission_enriched["Pred"].isna().any(), "NaNs found in Pred column"

    path_enriched = os.path.join(output_dir, f"submission_enriched_v{ts}.csv")
    submission_enriched.to_csv(path_enriched, index=False)

    # Summary prints.
    preds = submission_enriched["Pred"].to_numpy(dtype=float)
    mean_pred = float(preds.mean())
    std_pred = float(preds.std(ddof=0))
    print(f"Wrote {len(preds)} rows")
    print(f"Pred mean={mean_pred:.6f} std={std_pred:.6f}")

    bins = [0.0, 0.4, 0.6, 1.0]
    labels = ["[0,0.4)", "[0.4,0.6)", "[0.6,1.0)"]
    dist = pd.cut(preds, bins=bins, right=False, labels=labels).value_counts()
    for lab in labels:
        count = int(dist.loc[lab]) if lab in dist.index else 0
        print(f"Preds {lab}: {count}")

    conf = np.abs(preds - 0.5)
    order = np.argsort(-conf)[: min(10, len(preds))]
    top = submission_enriched.iloc[order].copy()
    top = top[["ID", "Team1Name", "Team2Name", "gender", "Pred"]]
    print("Top confident predictions:")
    print(top.to_string(index=False))

    return path_min, path_enriched


if __name__ == "__main__":
    # Dummy end-to-end run. This expects the default `data/` files to exist.
    class _DummyModel:
        def __init__(self, p: float) -> None:
            self._p = float(p)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            n = int(X.shape[0])
            p = self._p
            return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])

    class _DummyCal:
        def __init__(self, scale: float) -> None:
            self._scale = float(scale)

        def transform_men(self, raw_scores: np.ndarray) -> np.ndarray:
            return np.asarray(raw_scores) * self._scale

        def transform_women(self, raw_scores: np.ndarray) -> np.ndarray:
            return np.asarray(raw_scores) * self._scale

    data_dir = "data"
    sample_path = os.path.join(data_dir, "SampleSubmissionStage2.csv")
    teams_m_path = os.path.join(data_dir, "MTeams.csv")
    teams_w_path = os.path.join(data_dir, "WTeams.csv")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing {sample_path}")
    if not os.path.exists(teams_m_path):
        raise FileNotFoundError(f"Missing {teams_m_path}")
    if not os.path.exists(teams_w_path):
        raise FileNotFoundError(f"Missing {teams_w_path}")

    teams_m_df = pd.read_csv(teams_m_path)
    teams_w_df = pd.read_csv(teams_w_path)

    # For a dummy run, we intentionally use minimal empty feature stores.
    # The real pipeline should supply precomputed feature_store (elo/eff/four_factors/massey/seeds).
    feature_store = {
        "elo": {},
        "efficiency": pd.DataFrame(columns=["Season", "TeamID", "AdjEM"]),
        "four_factors": pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "eFG_pct",
                "TO_pct",
                "OR_pct",
                "FTRate",
                "opp_eFG_pct",
                "opp_TO_pct",
                "opp_OR_pct",
                "opp_FTRate",
            ]
        ),
        "massey": pd.DataFrame(columns=["Season", "TeamID", "massey_consensus"]),
        "seeds_m": pd.DataFrame(columns=["Season", "TeamID", "Seed"]),
        "seeds_w": pd.DataFrame(columns=["Season", "TeamID", "Seed"]),
    }

    output_dir = "submissions"
    print("Running dummy generate_submissions()...")
    generate_submissions(
        sample_sub_path=sample_path,
        feature_store=feature_store,
        model_men=_DummyModel(p=0.5),
        model_women=_DummyModel(p=0.5),
        calibrator_men=_DummyCal(scale=1.0),
        calibrator_women=_DummyCal(scale=1.0),
        teams_m_df=teams_m_df,
        teams_w_df=teams_w_df,
        output_dir=output_dir,
    )

# from __future__ import annotations  # duplicate/legacy block (kept to avoid SyntaxError)

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier

from .calibration import clip_probs, fit_platt, predict_platt
from .io import read_csv, require_files

# Optional legacy imports kept for older helper functions in this file. These
# paths are not required for the tuned-disk Stage2 path.
try:  # pragma: no cover
    from .benchmark import benchmark_models, instantiate_benchmark_model  # type: ignore
except Exception:  # pragma: no cover
    benchmark_models = None  # type: ignore[assignment]
    instantiate_benchmark_model = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .elo_legacy import compute_elo_by_season_team, load_mens_games, load_womens_games  # type: ignore
except Exception:  # pragma: no cover
    compute_elo_by_season_team = None  # type: ignore[assignment]
    load_mens_games = None  # type: ignore[assignment]
    load_womens_games = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .massey_legacy import load_massey_strength  # type: ignore
except Exception:  # pragma: no cover
    load_massey_strength = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .matchups import build_features_for_pairs, make_game_rows_compact  # type: ignore
except Exception:  # pragma: no cover
    build_features_for_pairs = None  # type: ignore[assignment]
    make_game_rows_compact = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .modeling import train_mens_model, train_womens_model  # type: ignore
except Exception:  # pragma: no cover
    train_mens_model = None  # type: ignore[assignment]
    train_womens_model = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .seed import load_seed_map  # type: ignore
except Exception:  # pragma: no cover
    load_seed_map = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .team_stats import build_team_season_stats  # type: ignore
except Exception:  # pragma: no cover
    build_team_season_stats = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Stage2SubmissionResult:
    path: str
    n_rows: int
    pred_min: float
    pred_max: float
    pred_mean: float


def _logit_from_proba(p: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p1 = np.asarray(p, dtype="float64")
    p1 = np.clip(p1, eps, 1.0 - eps)
    return np.log(p1 / (1.0 - p1))


def _scores_for_platt(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype="float64")
    proba = np.asarray(model.predict_proba(X)[:, 1], dtype="float64")
    return _logit_from_proba(proba)


def _train_best_benchmarked_model(
    sex: str,
    *,
    selector_mode: str = "cv_mean",
    cv_folds: int = 5,
):
    """
    Fit the best model (by rolling-CV mean Brier) and return it.
    This intentionally mirrors the benchmark selection rule.
    """
    r = benchmark_models(sex=sex, selector_mode=selector_mode, cv_folds=cv_folds)
    best = r.best_by_cv

    # Local re-fit on the same train split as benchmark holdout training uses (<=2022).
    base = train_mens_model() if sex == "M" else train_womens_model()
    X_train = _rebuild_training_features_for_sex(sex=sex, feature_names=base.feature_names, max_season=2022)
    y_train = _rebuild_training_labels_for_sex(sex=sex, max_season=2022)

    if best == "voting_soft":
        estimators = [(n, instantiate_benchmark_model(n, random_state=7)) for n in r.voting_member_names]
        vote = VotingClassifier(estimators=estimators, voting="soft")
        vote.fit(X_train, y_train)
        return vote

    m = instantiate_benchmark_model(best, random_state=7)
    m.fit(X_train, y_train)
    return m


def _rebuild_training_labels_for_sex(*, sex: str, max_season: int) -> np.ndarray:
    sex_norm = str(sex).strip().upper()
    tourney_csv = "MNCAATourneyCompactResults.csv" if sex_norm == "M" else "WNCAATourneyCompactResults.csv"
    tourney = read_csv(tourney_csv)
    rows = pd.DataFrame(
        {
            "Season": tourney["Season"],
            "WTeamID": tourney["WTeamID"],
            "LTeamID": tourney["LTeamID"],
        }
    )
    from .matchups import make_game_rows_compact

    pairs = make_game_rows_compact(rows)
    train_rows = pairs.loc[pairs["Season"] <= int(max_season)].reset_index(drop=True)
    return train_rows["y"].to_numpy(dtype="int64", copy=False)


def _rebuild_training_features_for_sex(
    *,
    sex: str,
    feature_names: list[str],
    max_season: int,
) -> np.ndarray:
    sex_norm = str(sex).strip().upper()
    tourney_csv = "MNCAATourneyCompactResults.csv" if sex_norm == "M" else "WNCAATourneyCompactResults.csv"
    tourney = read_csv(tourney_csv)
    from .matchups import make_game_rows_compact

    pairs = make_game_rows_compact(tourney)
    train_pairs = pairs.loc[pairs["Season"] <= int(max_season)].reset_index(drop=True)

    elo = compute_elo_by_season_team(load_mens_games() if sex_norm == "M" else load_womens_games())
    seeds = load_seed_map(sex_norm)
    massey = load_massey_strength("M") if sex_norm == "M" else None
    stats = build_team_season_stats(sex_norm)

    X, names = build_features_for_pairs(
        train_pairs,
        elo_by_season_team=elo,
        seed_map=seeds,
        massey_strength=massey,
        team_season_stats=stats,
    )
    if names != feature_names:
        raise ValueError("Feature mismatch when rebuilding benchmark training features.")
    return X


def _parse_stage2_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise ValueError("Sample submission missing required column: ID")

    parts = df["ID"].astype(str).str.split("_", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("Unexpected ID format; expected 'Season_TeamID1_TeamID2'.")

    out = pd.DataFrame(
        {
            "ID": df["ID"].astype(str),
            "Season": pd.to_numeric(parts[0], errors="coerce"),
            "TeamID1": pd.to_numeric(parts[1], errors="coerce"),
            "TeamID2": pd.to_numeric(parts[2], errors="coerce"),
        }
    ).dropna(subset=["Season", "TeamID1", "TeamID2"])

    out["Season"] = out["Season"].astype("int64")
    out["TeamID1"] = out["TeamID1"].astype("int64")
    out["TeamID2"] = out["TeamID2"].astype("int64")

    return out.reset_index(drop=True)


def write_stage2_2026_submission(
    *,
    output_path: str = "submissions/submission_stage2_2026.csv",
    model_strategy: str = "tuned_disk",
    benchmark_selector_mode: str = "cv_mean",
    benchmark_cv_folds: int = 5,
    lo: float = 0.05,
    hi: float = 0.975,
) -> Stage2SubmissionResult:
    """
    Generate a Kaggle-valid Stage2 submission for season 2026 potential matchups.

    Output format:
      ID,Pred
      2026_1101_1102,0.5234
      ...
    Pred is P(lower TeamID wins).
    """
    require_files(["SampleSubmissionStage2.csv"])

    strategy = str(model_strategy).strip().lower()
    if strategy not in {"tuned_disk"}:
        raise ValueError("model_strategy must be 'tuned_disk' for this repository.")

    # Load tuned artifacts from src/models (produced by hypetyune.ipynb).
    models_dir = Path(__file__).resolve().parent / "models"
    model_m_path = models_dir / "best_men.pkl"
    model_w_path = models_dir / "best_women.pkl"
    cal_m_path = models_dir / "best_calibrator_men.joblib"
    cal_w_path = models_dir / "best_calibrator_women.joblib"

    missing = [p for p in [model_m_path, model_w_path, cal_m_path, cal_w_path] if not p.exists()]
    if missing:
        missing_list = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Missing tuned model/calibrator artifact(s). Run `src/hypetyune.ipynb` first:\n"
            f"{missing_list}"
        )

    model_m = joblib_load(model_m_path)
    model_w = joblib_load(model_w_path)
    cal_m = joblib_load(cal_m_path)
    cal_w = joblib_load(cal_w_path)

    # Build feature_store (same sources as eda.ipynb fallback path).
    from .elo import compute_elo_men, compute_elo_women
    from .features import compute_efficiency, compute_four_factors
    from .massey import load_massey_features
    from .paths import get_data_dir

    data_dir = Path(get_data_dir())

    elo_m = compute_elo_men(str(data_dir))
    elo_w = compute_elo_women(str(data_dir))
    elo_all = dict(elo_m)
    elo_all.update(elo_w)

    m_det = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    w_det = pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv")
    efficiency = pd.concat([compute_efficiency(m_det), compute_efficiency(w_det)], ignore_index=True)
    four_factors = pd.concat([compute_four_factors(m_det), compute_four_factors(w_det)], ignore_index=True)
    massey = load_massey_features(str(data_dir), min_coverage=0.8)

    def _seed_num(seed_str: str) -> int | None:
        if not isinstance(seed_str, str):
            return None
        digits = "".join([c for c in seed_str if c.isdigit()])
        if not digits:
            return None
        return int(digits[:2]) if len(digits) >= 2 else int(digits)

    seeds_m_raw = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")[["Season", "TeamID", "Seed"]].copy()
    seeds_w_raw = pd.read_csv(data_dir / "WNCAATourneySeeds.csv")[["Season", "TeamID", "Seed"]].copy()
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

    feature_store = {
        "elo": elo_all,
        "efficiency": efficiency,
        "four_factors": four_factors,
        "massey": massey,
        "seeds_m": seeds_m[["Season", "TeamID", "Seed"]].copy(),
        "seeds_w": seeds_w[["Season", "TeamID", "Seed"]].copy(),
    }

    # Run the shared submission generator, then copy minimal stage2 to output_path.
    sample_path = str(data_dir / "SampleSubmissionStage2.csv")
    teams_m_df = pd.read_csv(data_dir / "MTeams.csv")
    teams_w_df = pd.read_csv(data_dir / "WTeams.csv")

    out_dir = str(Path(output_path).parent)
    path_min, _ = generate_submissions(
        sample_sub_path=sample_path,
        feature_store=feature_store,
        model_men=model_m,
        model_women=model_w,
        calibrator_men=cal_m,
        calibrator_women=cal_w,
        teams_m_df=teams_m_df,
        teams_w_df=teams_w_df,
        output_dir=out_dir,
    )

    # Keep only 2026 rows and write to the requested output path.
    stage2 = pd.read_csv(path_min)
    stage2 = stage2.loc[stage2["ID"].astype(str).str.startswith("2026_"), ["ID", "Pred"]].copy()
    stage2["Pred"] = np.clip(stage2["Pred"].to_numpy(dtype=float), float(lo), float(hi))
    stage2.to_csv(output_path, index=False)

    preds = stage2["Pred"].to_numpy(dtype=float)

    return Stage2SubmissionResult(
        path=output_path,
        n_rows=int(len(stage2)),
        pred_min=float(np.min(preds)),
        pred_max=float(np.max(preds)),
        pred_mean=float(np.mean(preds)),
    )


if __name__ == "__main__":
    r = write_stage2_2026_submission()
    print(r)

