import numpy as np
import pandas as pd


def test_tournament_calibrator_clips_to_bounds_after_transform() -> None:
    from src.calibration import TournamentCalibrator

    rng = np.random.default_rng(42)
    n = 500
    raw_scores = rng.random(n)
    # Create a deliberately miscalibrated mapping: true win prob is non-linear in raw.
    true_probs = np.clip(0.05 + 0.9 * (raw_scores**0.5), 0.0, 1.0)
    y_true = rng.binomial(1, true_probs).astype(int)

    cal = TournamentCalibrator()
    cal.fit_men(raw_scores, y_true)
    probs = cal.transform_men(raw_scores)

    assert np.all(probs >= 0.05)
    assert np.all(probs <= 0.975)


def test_platt_calibration_improves_brier_on_synthetic_miscalibration() -> None:
    from src.calibration import TournamentCalibrator

    rng = np.random.default_rng(42)
    n = 2000
    raw_scores = rng.random(n)
    true_probs = np.clip(0.05 + 0.9 * (raw_scores**0.5), 0.0, 1.0)
    y_true = rng.binomial(1, true_probs).astype(int)

    brier_raw = float(np.mean((raw_scores - y_true) ** 2))

    cal = TournamentCalibrator()
    cal.fit_men(raw_scores, y_true)
    probs = cal.transform_men(raw_scores)
    brier_cal = cal.brier_score(y_true, probs)

    assert brier_cal <= brier_raw


def test_compare_calibration_methods_returns_expected_dataframe() -> None:
    from src.calibration import compare_calibration_methods

    rng = np.random.default_rng(42)
    n = 800
    raw_scores = rng.random(n)
    true_probs = np.clip(0.05 + 0.9 * (raw_scores**0.5), 0.0, 1.0)
    y_true = rng.binomial(1, true_probs).astype(int)

    df = compare_calibration_methods(raw_scores, y_true)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"method", "brier_score", "log_loss"}
    assert len(df) == 3
    assert set(df["method"].tolist()) == {"raw", "platt", "isotonic"}

