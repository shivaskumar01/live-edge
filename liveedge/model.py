"""Model, calibration, metrics, and persistence.

A win-probability model is judged on *calibration*, whether the games it calls "70%" win
about 70% of the time, not on classification accuracy. So this module provides the metrics
that matter (log loss, Brier, a reliability table, ECE) and a temperature scaler to calibrate
the network's logits on a held-out split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class WinProbNet(nn.Module):
    """A small MLP mapping a feature vector to a single win-probability logit."""

    def __init__(
        self, n_features: int, hidden: tuple[int, ...] = (64, 64), dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # final logit (no activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StandardScaler:
    """Standardize features to zero mean / unit variance, saved alongside the model so
    live inference standardizes exactly as training did."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero on constant columns
        self.std = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x) - self.mean) / self.std


class TemperatureScaler:
    """Single-parameter calibration: divide the logits by a learned temperature T.

    T > 1 softens overconfident probabilities; T < 1 sharpens underconfident ones. It is fit
    on a held-out calibration split, never on the training data.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(
        self, logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, max_iter: int = 200
    ) -> "TemperatureScaler":
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        T = nn.Parameter(torch.ones(1))
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
        loss_fn = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = loss_fn(logits_t / T.clamp(min=1e-3), labels_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(T.detach().clamp(min=1e-3))
        return self

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


# --------------------------------------------------------------------------------------
# Metrics (numpy)
# --------------------------------------------------------------------------------------


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy (a.k.a. log loss). Lower is better; rewards calibration."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def reliability_table(
    probs: np.ndarray, labels: np.ndarray, bins: int = 10
) -> list[tuple[float, float, int, float, float]]:
    """Bin predictions and compare predicted vs. actual win rate per bin.

    Returns rows of (bin_low, bin_high, n, predicted_mean, actual_rate). The final bin is
    inclusive on its upper edge so probs of exactly 1.0 land somewhere. Empty bins report
    n=0 with NaN means.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[tuple[float, float, int, float, float]] = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        n = int(mask.sum())
        if n == 0:
            rows.append((float(low), float(high), 0, float("nan"), float("nan")))
        else:
            rows.append(
                (
                    float(low),
                    float(high),
                    n,
                    float(probs[mask].mean()),
                    float(labels[mask].mean()),
                )
            )
    return rows


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Weighted mean over non-empty bins of |predicted_mean - actual_rate|."""
    rows = reliability_table(probs, labels, bins)
    total = len(np.asarray(probs))
    if total == 0:
        return float("nan")
    ece = 0.0
    for _, _, n, pred, actual in rows:
        if n == 0:
            continue
        ece += (n / total) * abs(pred - actual)
    return float(ece)


# --------------------------------------------------------------------------------------
# Inference + persistence
# --------------------------------------------------------------------------------------


def predict_prob(
    model: WinProbNet,
    scaler: StandardScaler,
    features: np.ndarray,
    calibrator: TemperatureScaler | None = None,
) -> np.ndarray:
    """Run features (2D array-like) through scaler -> model -> (optional) calibrator -> sigmoid.

    GOTCHA: a *loaded* scaler's mean/std come back as float64 and would promote the input to
    double, crashing the float32 model with "mat1 and mat2 must have the same dtype". So we
    force float32 both before and after scaling.
    """
    model.eval()
    x = scaler.transform(np.asarray(features, dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        if calibrator is not None:
            logits = calibrator.apply(logits)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def _hidden_from_model(model: WinProbNet) -> tuple[int, ...]:
    """Recover the hidden layer sizes from the Linear layers, excluding the output layer."""
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    return tuple(layer.out_features for layer in linears[:-1])


def save_bundle(
    path: str,
    model: WinProbNet,
    scaler: StandardScaler,
    calibrator: TemperatureScaler | None,
    feature_names: list[str],
    sport: str,
    metrics: dict | None = None,
    reliability: list | None = None,
) -> None:
    """Persist a model bundle: weights to `<path>.pt` and metadata to `<path>.json`.

    Optionally embeds the validation `metrics` and `reliability` table so downstream tools (e.g.
    the dashboard) can show the model's calibration without recomputing it. NaNs (empty bins)
    are stored as null for valid JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(p) + ".pt")
    meta = {
        "sport": sport,
        "feature_names": list(feature_names),
        "scaler_mean": np.asarray(scaler.mean, dtype=np.float64).tolist(),
        "scaler_std": np.asarray(scaler.std, dtype=np.float64).tolist(),
        "temperature": float(calibrator.temperature) if calibrator is not None else 1.0,
        "hidden": list(_hidden_from_model(model)),
    }
    if metrics is not None:
        meta["metrics"] = {
            k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()
        }
    if reliability is not None:
        meta["reliability"] = [
            [lo, hi, n, None if math.isnan(pred) else pred, None if math.isnan(actual) else actual]
            for lo, hi, n, pred, actual in reliability
        ]
    with open(str(p) + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_bundle(
    path: str,
) -> tuple[WinProbNet, StandardScaler, TemperatureScaler, dict]:
    """Inverse of save_bundle: rebuild model + scaler + calibrator from `<path>.{pt,json}`."""
    p = Path(path)
    with open(str(p) + ".json") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    hidden = tuple(meta.get("hidden", (64, 64)))
    model = WinProbNet(len(feature_names), hidden)
    model.load_state_dict(torch.load(str(p) + ".pt", map_location="cpu"))
    model.eval()

    scaler = StandardScaler()
    scaler.mean = np.asarray(meta["scaler_mean"], dtype=np.float64)
    scaler.std = np.asarray(meta["scaler_std"], dtype=np.float64)

    calibrator = TemperatureScaler()
    calibrator.temperature = float(meta.get("temperature", 1.0))

    return model, scaler, calibrator, meta
