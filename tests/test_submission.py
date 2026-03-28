import pandas as pd
import numpy as np


class DummyModel:
    def __init__(self, p: float) -> None:
        self._p = float(p)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Return probabilities for class 0/1. Ignore X content; caller routes by gender.
        n = int(X.shape[0])
        p = self._p
        return np.column_stack([np.full(n, 1.0 - p, dtype=float), np.full(n, p, dtype=float)])


class DummyCalibrator:
    def __init__(self, scale: float) -> None:
        self._scale = float(scale)

    def transform_men(self, raw_scores: np.ndarray) -> np.ndarray:
        return np.asarray(raw_scores, dtype=float) * self._scale

    def transform_women(self, raw_scores: np.ndarray) -> np.ndarray:
        return np.asarray(raw_scores, dtype=float) * self._scale


def _write_sample_submission(path: str, rows: list[tuple[int, int, int]]) -> None:
    # ID format: SSSS_LOWERID_HIGHERID
    out = []
    for season, t1, t2 in rows:
        out.append({"ID": f"{season:04d}_{t1}_{t2}"})
    df = pd.DataFrame(out)
    df["Pred"] = 0.5  # placeholder column; generate_submissions should overwrite it.
    df.to_csv(path, index=False)


def test_generate_submissions_writes_minimal_and_enriched_and_clips_preds(tmp_path) -> None:
    from src.submission import generate_submissions

    sample_path = tmp_path / "SampleSubmissionStage2.csv"
    _write_sample_submission(
        str(sample_path),
        rows=[
            # Men row (t1<3000)
            (2026, 1101, 1501),
            # Women row (t1>=3000)
            (2026, 3101, 3501),
            # Men row
            (2026, 1202, 1802),
        ],
    )

    teams_m_df = pd.DataFrame(
        {"TeamID": [1101, 1501, 1202, 1802], "TeamName": ["A", "B", "C", "D"]}
    )
    teams_w_df = pd.DataFrame(
        {"TeamID": [3101, 3501], "TeamName": ["W_A", "W_B"]}
    )

    # Minimal feature inputs needed by src.features.encode_matchups.
    # encode_matchups uses pre-tournament Elo at (Season-1, TeamID).
    elo = {
        (2025, 1101): 1600.0,
        (2025, 1501): 1500.0,
        (2025, 1202): 1500.0,
        (2025, 1802): 1400.0,
        (2025, 3101): 1500.0,
        (2025, 3501): 1400.0,
    }

    efficiency = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1101, "AdjEM": 5.0},
            {"Season": 2026, "TeamID": 1501, "AdjEM": 1.0},
            {"Season": 2026, "TeamID": 1202, "AdjEM": 2.0},
            {"Season": 2026, "TeamID": 1802, "AdjEM": -1.0},
            {"Season": 2026, "TeamID": 3101, "AdjEM": 3.0},
            {"Season": 2026, "TeamID": 3501, "AdjEM": 0.0},
        ]
    )

    four_factors = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "eFG_pct": 0.50,
                "TO_pct": 0.18,
                "OR_pct": 0.30,
                "FTRate": 0.30,
                "opp_eFG_pct": 0.49,
                "opp_TO_pct": 0.19,
                "opp_OR_pct": 0.31,
                "opp_FTRate": 0.29,
            },
            {
                "Season": 2026,
                "TeamID": 1501,
                "eFG_pct": 0.48,
                "TO_pct": 0.20,
                "OR_pct": 0.29,
                "FTRate": 0.28,
                "opp_eFG_pct": 0.51,
                "opp_TO_pct": 0.18,
                "opp_OR_pct": 0.33,
                "opp_FTRate": 0.30,
            },
            {
                "Season": 2026,
                "TeamID": 1202,
                "eFG_pct": 0.49,
                "TO_pct": 0.19,
                "OR_pct": 0.28,
                "FTRate": 0.27,
                "opp_eFG_pct": 0.52,
                "opp_TO_pct": 0.17,
                "opp_OR_pct": 0.34,
                "opp_FTRate": 0.31,
            },
            {
                "Season": 2026,
                "TeamID": 1802,
                "eFG_pct": 0.47,
                "TO_pct": 0.21,
                "OR_pct": 0.30,
                "FTRate": 0.26,
                "opp_eFG_pct": 0.50,
                "opp_TO_pct": 0.20,
                "opp_OR_pct": 0.32,
                "opp_FTRate": 0.29,
            },
            {
                "Season": 2026,
                "TeamID": 3101,
                "eFG_pct": 0.51,
                "TO_pct": 0.19,
                "OR_pct": 0.31,
                "FTRate": 0.29,
                "opp_eFG_pct": 0.48,
                "opp_TO_pct": 0.20,
                "opp_OR_pct": 0.30,
                "opp_FTRate": 0.30,
            },
            {
                "Season": 2026,
                "TeamID": 3501,
                "eFG_pct": 0.49,
                "TO_pct": 0.18,
                "OR_pct": 0.29,
                "FTRate": 0.28,
                "opp_eFG_pct": 0.52,
                "opp_TO_pct": 0.17,
                "opp_OR_pct": 0.33,
                "opp_FTRate": 0.27,
            },
        ]
    )

    massey = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1101, "massey_consensus": 1.0},
            {"Season": 2026, "TeamID": 1501, "massey_consensus": 0.0},
            {"Season": 2026, "TeamID": 1202, "massey_consensus": 0.5},
            {"Season": 2026, "TeamID": 1802, "massey_consensus": -0.5},
            {"Season": 2026, "TeamID": 3101, "massey_consensus": 0.2},
            {"Season": 2026, "TeamID": 3501, "massey_consensus": -0.2},
        ]
    )

    seeds_m = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1101, "Seed": 3.0},
            {"Season": 2026, "TeamID": 1501, "Seed": 6.0},
            {"Season": 2026, "TeamID": 1202, "Seed": 4.0},
            {"Season": 2026, "TeamID": 1802, "Seed": 10.0},
        ]
    )
    seeds_w = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 3101, "Seed": 5.0},
            {"Season": 2026, "TeamID": 3501, "Seed": 8.0},
        ]
    )

    feature_store = {
        "elo": elo,
        "efficiency": efficiency,
        "four_factors": four_factors,
        "massey": massey,
        "seeds_m": seeds_m,
        "seeds_w": seeds_w,
    }

    # Men should be routed to model_men; women routed to model_women.
    model_men = DummyModel(p=0.9)  # raw -> 0.9
    model_women = DummyModel(p=0.01)  # raw -> 0.01
    calibrator_men = DummyCalibrator(scale=1.2)  # -> 1.08 then clipped to 0.975
    calibrator_women = DummyCalibrator(scale=0.1)  # -> 0.001 then clipped to 0.05

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    path_min, path_enriched = generate_submissions(
        sample_sub_path=str(sample_path),
        feature_store=feature_store,
        model_men=model_men,
        model_women=model_women,
        calibrator_men=calibrator_men,
        calibrator_women=calibrator_women,
        teams_m_df=teams_m_df,
        teams_w_df=teams_w_df,
        output_dir=str(output_dir),
    )

    minimal = pd.read_csv(path_min)
    assert minimal.columns.tolist() == ["ID", "Pred"]
    assert len(minimal) == 3

    enriched = pd.read_csv(path_enriched)
    assert len(enriched) == 3
    assert enriched["Pred"].notna().all()
    assert enriched["Pred"].between(0.05, 0.975, inclusive="both").all()

    assert set(enriched["gender"].tolist()) == {"M", "W"}
    assert set(enriched.columns) >= {
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
    }

