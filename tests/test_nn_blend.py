import numpy as np
import pandas as pd

from src.nn_blend import tune_blend_weights_lr_xgb_nn


def test_tune_blend_weights_prefers_best_component() -> None:
    y_men = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    y_women = np.array([0, 1, 0, 1, 0, 1], dtype=int)

    # lr is intentionally strongest, xgb is weaker, nn is weakest
    lr_men = np.array([0.92, 0.08, 0.87, 0.12, 0.90, 0.10], dtype=float)
    xgb_men = np.array([0.70, 0.30, 0.65, 0.35, 0.68, 0.32], dtype=float)
    nn_men = np.array([0.58, 0.42, 0.56, 0.44, 0.59, 0.41], dtype=float)

    lr_women = np.array([0.12, 0.88, 0.10, 0.90, 0.09, 0.91], dtype=float)
    xgb_women = np.array([0.31, 0.69, 0.35, 0.65, 0.32, 0.68], dtype=float)
    nn_women = np.array([0.45, 0.55, 0.48, 0.52, 0.44, 0.56], dtype=float)

    result = tune_blend_weights_lr_xgb_nn(
        y_men=y_men,
        y_women=y_women,
        lr_prob_men=lr_men,
        xgb_prob_men=xgb_men,
        nn_prob_men=nn_men,
        lr_prob_women=lr_women,
        xgb_prob_women=xgb_women,
        nn_prob_women=nn_women,
        step=0.1,
    )

    assert result["w_lr"] >= result["w_xgb"]
    assert result["w_lr"] >= result["w_nn"]
    assert 0.0 <= result["w_lr"] <= 1.0
    assert 0.0 <= result["w_xgb"] <= 1.0
    assert 0.0 <= result["w_nn"] <= 1.0
    assert np.isclose(result["w_lr"] + result["w_xgb"] + result["w_nn"], 1.0)
    assert result["brier_mean"] >= 0.0
    assert isinstance(result["grid"], pd.DataFrame)
    assert set(["w_lr", "w_xgb", "w_nn", "brier_men", "brier_women", "brier_mean"]).issubset(
        set(result["grid"].columns)
    )
