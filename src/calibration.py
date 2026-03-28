from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

_CLIP_MIN: float = 0.05
_CLIP_MAX: float = 0.975


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs.astype(float), _CLIP_MIN, _CLIP_MAX)


@dataclass
class _ConstantCalibrator:
    p: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = int(x.shape[0])
        p = float(self.p)
        return np.column_stack([1.0 - p * np.ones(n), p * np.ones(n)])


class TournamentCalibrator:
    """Calibrate tournament probabilities using Platt scaling (logistic)."""

    def __init__(self) -> None:
        # Each sport gets its own calibrator instance.
        self.calibrator_men: LogisticRegression | _ConstantCalibrator = LogisticRegression(
            C=1e10,
            max_iter=2000,
        )
        self.calibrator_women: LogisticRegression | _ConstantCalibrator = LogisticRegression(
            C=1e10,
            max_iter=2000,
        )

    # Sport-specific API.
    def fit_men(
        self, raw_scores: np.ndarray, y_true: np.ndarray
    ) -> "TournamentCalibrator":
        self.calibrator_men = self._fit_platt(raw_scores, y_true)
        return self

    def fit_women(
        self, raw_scores: np.ndarray, y_true: np.ndarray
    ) -> "TournamentCalibrator":
        self.calibrator_women = self._fit_platt(raw_scores, y_true)
        return self

    def transform_men(self, raw_scores: np.ndarray) -> np.ndarray:
        probs = self._predict_platt(self.calibrator_men, raw_scores)
        probs = _clip_probs(probs)
        assert np.all(probs >= _CLIP_MIN) and np.all(probs <= _CLIP_MAX)
        return probs

    def transform_women(self, raw_scores: np.ndarray) -> np.ndarray:
        probs = self._predict_platt(self.calibrator_women, raw_scores)
        probs = _clip_probs(probs)
        assert np.all(probs >= _CLIP_MIN) and np.all(probs <= _CLIP_MAX)
        return probs

    def fit_transform_men(
        self, raw_scores: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        self.fit_men(raw_scores, y_true)
        return self.transform_men(raw_scores)

    def fit_transform_women(
        self, raw_scores: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        self.fit_women(raw_scores, y_true)
        return self.transform_women(raw_scores)

    # Generic API (defaults to men calibrator).
    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "TournamentCalibrator":
        return self.fit_men(raw_scores, y_true)

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.transform_men(raw_scores)

    def fit_transform(
        self, raw_scores: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        self.fit_men(raw_scores, y_true)
        return self.transform_men(raw_scores)

    @staticmethod
    def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(probs, dtype=float)
        return float(np.mean((p - y) ** 2))

    @staticmethod
    def _fit_platt(
        raw_scores: np.ndarray, y_true: np.ndarray
    ) -> LogisticRegression | _ConstantCalibrator:
        x = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int).reshape(-1)

        unique = np.unique(y)
        if unique.size < 2:
            # Degenerate case: can't learn calibration; use empirical rate.
            p = float(np.mean(y))
            return _ConstantCalibrator(p=_clip_probs(np.array([p]))[0])

        model = LogisticRegression(C=1e10, max_iter=2000)
        model.fit(x, y)
        return model

    @staticmethod
    def _predict_platt(
        calibrator: LogisticRegression | _ConstantCalibrator, raw_scores: np.ndarray
    ) -> np.ndarray:
        x = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        proba = calibrator.predict_proba(x)
        # sklearn returns [P(class0), P(class1)].
        return proba[:, 1].astype(float, copy=False)

    def plot_reliability_curve(
        self,
        y_true: np.ndarray,
        raw_scores: np.ndarray,
        calibrated_scores: np.ndarray,
        ax: Optional[object] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        y = np.asarray(y_true, dtype=float)
        raw = np.asarray(raw_scores, dtype=float)
        cal = np.asarray(calibrated_scores, dtype=float)

        bins = np.linspace(0.0, 1.0, 11)  # 10 bins

        def _bin_curve(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            idx = np.digitize(pred, bins, right=False) - 1
            xs = np.full(10, np.nan, dtype=float)
            ys = np.full(10, np.nan, dtype=float)
            for b in range(10):
                mask = idx == b
                if np.any(mask):
                    xs[b] = float(np.mean(pred[mask]))
                    ys[b] = float(np.mean(y[mask]))
            return xs, ys

        raw_x, raw_y = _bin_curve(raw)
        cal_x, cal_y = _bin_curve(cal)

        if ax is None:
            _, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.plot(raw_x, raw_y, "o--", color="orange", label="raw")
        ax.plot(cal_x, cal_y, "o-", color="blue", label="calibrated")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual win rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        return None


def compare_calibration_methods(
    raw_scores: np.ndarray, y_true: np.ndarray
) -> pd.DataFrame:
    """Compare raw vs Platt vs isotonic calibration."""
    raw = np.asarray(raw_scores, dtype=float)
    y = np.asarray(y_true, dtype=int)

    raw_clipped = np.clip(raw, 0.0, 1.0)
    probs_raw = _clip_probs(raw_clipped)

    cal = TournamentCalibrator().fit_men(raw_clipped, y)
    probs_platt = cal.transform_men(raw_clipped)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_clipped.reshape(-1, 1), y)
    probs_iso = _clip_probs(iso.predict(raw_clipped.reshape(-1, 1)))

    eps = 1e-15
    rows: list[dict[str, float | str]] = []
    for name, probs in [
        ("raw", probs_raw),
        ("platt", probs_platt),
        ("isotonic", probs_iso),
    ]:
        brier = TournamentCalibrator.brier_score(y, probs)
        ll = float(log_loss(y, np.clip(probs, eps, 1.0 - eps), labels=[0, 1]))
        rows.append({"method": name, "brier_score": brier, "log_loss": ll})

    return pd.DataFrame(rows)


def clip_probs(probs: np.ndarray, *, lo: float = _CLIP_MIN, hi: float = _CLIP_MAX) -> np.ndarray:
    """Compatibility helper: clip probabilities to tournament-safe bounds."""
    return np.clip(np.asarray(probs, dtype=float), float(lo), float(hi))


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Compatibility helper: fit Platt scaling (logistic regression) for binary outcomes."""
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=int).reshape(-1)

    unique = np.unique(y_arr)
    if unique.size < 2:
        # Degenerate case: logistic regression can't learn. Return a model that
        # predicts a constant probability via a fitted single-class intercept hack.
        # We mimic constant behavior using scikit-learn's logistic regression by
        # fitting on two pseudo-samples.
        p = float(np.mean(y_arr))
        dummy_x = np.array([[0.0], [1.0]], dtype=float)
        dummy_y = np.array([0, 1], dtype=int)
        lr = LogisticRegression(C=1e10, max_iter=2000)
        lr.fit(dummy_x, dummy_y)
        # Store desired constant rate on the estimator for predict_platt().
        setattr(lr, "_mm_constant_rate", clip_probs(np.array([p]))[0])
        return lr

    lr = LogisticRegression(C=1e10, max_iter=2000)
    lr.fit(x, y_arr)
    return lr


def predict_platt(platt: Any, scores: np.ndarray) -> np.ndarray:
    """Compatibility helper: map Platt-scaled logistic outputs to probabilities."""
    scores_arr = np.asarray(scores, dtype=float).reshape(-1, 1)
    constant_rate = getattr(platt, "_mm_constant_rate", None)
    if constant_rate is not None:
        return clip_probs(np.full((scores_arr.shape[0],), float(constant_rate)))

    proba = platt.predict_proba(scores_arr)[:, 1]
    return clip_probs(proba)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    raw_scores = rng.random(n)

    # Non-linear ground truth to induce miscalibration.
    true_probs = np.clip(0.05 + 0.9 * (raw_scores**0.5), 0.0, 1.0)
    y_true = rng.binomial(1, true_probs).astype(int)

    cal = TournamentCalibrator()
    cal.fit_men(raw_scores, y_true)
    calibrated_scores = cal.transform_men(raw_scores)

    brier_raw = cal.brier_score(y_true, _clip_probs(raw_scores))
    brier_cal = cal.brier_score(y_true, calibrated_scores)
    print(f"Brier raw: {brier_raw:.6f}")
    print(f"Brier calibrated: {brier_cal:.6f}")

    cmp = compare_calibration_methods(raw_scores, y_true)
    print(cmp.sort_values("brier_score"))

