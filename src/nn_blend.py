from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class TrainedNNBlendModel:
    model: Any
    feature_columns: list[str]
    mu: np.ndarray
    std: np.ndarray


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch as _torch
        import torch.nn as _nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for NN blend training. Install torch to use train_nn_mlp()."
        ) from exc
    return _torch, _nn, DataLoader, TensorDataset


if nn is not None:
    class ResidualMatchupMLP(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(n_features, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()

        def forward(self, xb: Any) -> Any:
            h1 = self.dropout(self.relu(self.bn1(self.fc1(xb))))
            h2 = self.dropout(self.relu(self.bn2(self.fc2(h1))))
            h = h1 + h2  # residual skip in the 64-dim block
            h = self.relu(self.fc3(h))
            logits = self.fc4(h).squeeze(1)
            return logits
else:  # pragma: no cover
    ResidualMatchupMLP = None  # type: ignore[assignment]


def _as_dataframe(x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x)


def _weighted_bce_loss(losses: Any, weights: Any, eps: float = 1e-12) -> Any:
    weighted = losses * weights
    return weighted.sum() / (weights.sum() + eps)


def train_nn_mlp(
    *,
    x: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int = 42,
    n_epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    val_frac: float = 0.15,
    min_improvement: float = 1e-5,
) -> TrainedNNBlendModel:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    if ResidualMatchupMLP is None:  # pragma: no cover
        raise ImportError("PyTorch is required for NN blend training.")

    x_df = _as_dataframe(x)
    feature_columns = list(x_df.columns.astype(str))

    x_arr = x_df.to_numpy(dtype=np.float32, copy=True)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    w_arr = np.asarray(sample_weight, dtype=np.float32).reshape(-1)

    if not (len(x_arr) == len(y_arr) == len(w_arr)):
        raise ValueError("x, y, and sample_weight must have the same length.")
    if len(x_arr) < 10:
        raise ValueError("Need at least 10 rows to train NN reliably.")

    torch.manual_seed(int(random_state))

    n_val = max(1, int(len(y_arr) * float(val_frac)))
    x_tr, x_vl = x_arr[:-n_val], x_arr[-n_val:]
    y_tr, y_vl = y_arr[:-n_val], y_arr[-n_val:]
    w_tr = w_arr[:-n_val]

    mu = x_tr.mean(axis=0)
    std = x_tr.std(axis=0) + 1e-8
    x_tr = (x_tr - mu) / std
    x_vl = (x_vl - mu) / std

    ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(w_tr, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True)

    model = ResidualMatchupMLP(n_features=int(x_tr.shape[1]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_brier = float("inf")
    best_state: dict[str, Any] | None = None
    wait = 0

    for epoch in range(int(n_epochs)):
        model.train()
        for xb, yb, wb in dl:
            optimizer.zero_grad()
            logits = model(xb)
            losses = criterion(logits, yb)
            loss = _weighted_bce_loss(losses, wb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl_logits = model(torch.tensor(x_vl, dtype=torch.float32))
            vl_pred = torch.sigmoid(vl_logits).cpu().numpy()
            val_brier = float(np.mean((vl_pred - y_vl) ** 2))

        if val_brier < best_val_brier - float(min_improvement):
            best_val_brier = val_brier
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if epoch > 50 and val_brier > 0.25:
                break
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedNNBlendModel(
        model=model,
        feature_columns=feature_columns,
        mu=mu.astype(np.float32, copy=False),
        std=std.astype(np.float32, copy=False),
    )


def predict_nn_mlp_proba(
    trained: TrainedNNBlendModel,
    x: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    torch, _, _, _ = _require_torch()

    x_df = _as_dataframe(x)
    if x_df.shape[1] != len(trained.feature_columns):
        raise ValueError("Feature count mismatch for NN inference.")

    x_df.columns = [str(c) for c in x_df.columns]
    x_in = x_df[trained.feature_columns].to_numpy(dtype=np.float32, copy=True)
    x_in = (x_in - trained.mu) / trained.std

    trained.model.eval()
    with torch.no_grad():
        logits = trained.model(torch.tensor(x_in, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.astype(float, copy=False)


def tune_blend_weights_lr_xgb_nn(
    *,
    y_men: np.ndarray,
    y_women: np.ndarray,
    lr_prob_men: np.ndarray,
    xgb_prob_men: np.ndarray,
    nn_prob_men: np.ndarray,
    lr_prob_women: np.ndarray,
    xgb_prob_women: np.ndarray,
    nn_prob_women: np.ndarray,
    step: float = 0.05,
) -> dict[str, Any]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")

    y_m = np.asarray(y_men, dtype=int).reshape(-1)
    y_w = np.asarray(y_women, dtype=int).reshape(-1)
    lr_m = np.asarray(lr_prob_men, dtype=float).reshape(-1)
    xgb_m = np.asarray(xgb_prob_men, dtype=float).reshape(-1)
    nn_m = np.asarray(nn_prob_men, dtype=float).reshape(-1)
    lr_w = np.asarray(lr_prob_women, dtype=float).reshape(-1)
    xgb_w = np.asarray(xgb_prob_women, dtype=float).reshape(-1)
    nn_w = np.asarray(nn_prob_women, dtype=float).reshape(-1)

    if not (len(y_m) == len(lr_m) == len(xgb_m) == len(nn_m)):
        raise ValueError("Men arrays must all have equal length.")
    if not (len(y_w) == len(lr_w) == len(xgb_w) == len(nn_w)):
        raise ValueError("Women arrays must all have equal length.")

    w_vals = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    rows: list[dict[str, float]] = []

    for w_lr, w_xgb in product(w_vals, w_vals):
        w_nn = 1.0 - float(w_lr) - float(w_xgb)
        if w_nn < -1e-12:
            continue
        if w_nn < 0:
            w_nn = 0.0
        if not np.isclose(w_lr + w_xgb + w_nn, 1.0, atol=1e-8):
            continue

        p_m = w_lr * lr_m + w_xgb * xgb_m + w_nn * nn_m
        p_w = w_lr * lr_w + w_xgb * xgb_w + w_nn * nn_w
        b_m = float(brier_score_loss(y_m, p_m))
        b_w = float(brier_score_loss(y_w, p_w))
        rows.append(
            {
                "w_lr": float(w_lr),
                "w_xgb": float(w_xgb),
                "w_nn": float(w_nn),
                "brier_men": b_m,
                "brier_women": b_w,
                "brier_mean": float((b_m + b_w) / 2.0),
            }
        )

    if not rows:
        raise ValueError("No valid weight combinations generated.")

    grid = pd.DataFrame(rows).sort_values(["brier_mean", "w_lr"], ascending=[True, False]).reset_index(
        drop=True
    )
    best = grid.iloc[0].to_dict()

    return {
        "w_lr": float(best["w_lr"]),
        "w_xgb": float(best["w_xgb"]),
        "w_nn": float(best["w_nn"]),
        "brier_men": float(best["brier_men"]),
        "brier_women": float(best["brier_women"]),
        "brier_mean": float(best["brier_mean"]),
        "grid": grid,
    }
