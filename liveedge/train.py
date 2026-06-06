"""Training pipeline + CLI.

Trains a WinProbNet on a feature frame, calibrates it with temperature scaling on a held-out
split, and reports the metrics that matter for a probability model (log loss, Brier, ECE,
plus a reliability table and a base-rate baseline).

Usage
-----
Zero-setup synthetic path (no network, no API keys — proves the machinery only):

    python -m liveedge.train --sport nba --synthetic 30000 --epochs 15

Real historical data (needs the per-sport source packages):

    python -m liveedge.train --sport nfl --seasons 2016 2017 2018 2019 2020 2021 2022 2023
    python -m liveedge.train --sport nba --seasons 2021 2022
    python -m liveedge.train --sport mlb --seasons 2022 2023
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from liveedge.features import get_spec
from liveedge.model import (
    StandardScaler,
    TemperatureScaler,
    WinProbNet,
    brier_score,
    expected_calibration_error,
    log_loss,
    predict_prob,
    reliability_table,
    save_bundle,
)

_SPLIT_SEED = 0


def _print_reliability(rows: list[tuple[float, float, int, float, float]]) -> None:
    print(f"  {'bin':>11}  {'n':>7}  {'pred':>6}  {'actual':>6}")
    for low, high, n, pred, actual in rows:
        if n == 0:
            print(f"  {low:.1f}-{high:.1f}  {n:>7}  {'--':>6}  {'--':>6}")
        else:
            print(f"  {low:.1f}-{high:.1f}  {n:>7}  {pred:6.3f}  {actual:6.3f}")


def train_from_frame(
    df,
    sport: str,
    *,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden: tuple[int, ...] = (64, 64),
    dropout: float = 0.2,
    out_path: str | None = None,
    verbose: bool = True,
) -> tuple[WinProbNet, StandardScaler, TemperatureScaler, dict]:
    """Train + calibrate a win-prob model on `df`. Returns (model, scaler, calibrator, metrics)."""
    spec = get_spec(sport)
    needed = spec.features + [spec.label]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")

    df = df.dropna(subset=needed)
    x = df[spec.features].to_numpy(dtype=np.float32)
    y = df[spec.label].to_numpy(dtype=np.float32)
    n = len(x)
    if n < 10:
        raise ValueError(f"not enough rows to train: {n}")

    # Seeded 70/15/15 train / val / cal split. Train fits the net, val tracks the best
    # epoch, cal is held out purely for temperature scaling.
    torch.manual_seed(_SPLIT_SEED)
    perm = np.random.default_rng(_SPLIT_SEED).permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    tr, va, ca = perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]

    scaler = StandardScaler().fit(x[tr])
    xt = scaler.transform(x[tr]).astype(np.float32)
    xv = scaler.transform(x[va]).astype(np.float32)
    xc = scaler.transform(x[ca]).astype(np.float32)

    model = WinProbNet(x.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xt), torch.from_numpy(y[tr])),
        batch_size=batch_size,
        shuffle=True,
    )

    xv_t = torch.from_numpy(xv)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(xv_t)).numpy()
        val_ll = log_loss(val_probs, y[va])
        if val_ll < best_val:
            best_val = val_ll
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"epoch {epoch:3d}/{epochs}  val_logloss {val_ll:.4f}")

    model.load_state_dict(best_state)

    # Calibrate on the held-out cal split, then evaluate (post-calibration) on val.
    model.eval()
    with torch.no_grad():
        cal_logits = model(torch.from_numpy(xc))
    calibrator = TemperatureScaler().fit(cal_logits, y[ca])

    val_probs = predict_prob(model, scaler, x[va], calibrator)
    base_rate = float(y[va].mean())
    baseline_probs = np.full(len(va), base_rate, dtype=np.float64)

    metrics = {
        "log_loss": log_loss(val_probs, y[va]),
        "brier": brier_score(val_probs, y[va]),
        "ece": expected_calibration_error(val_probs, y[va]),
        "baseline_log_loss": log_loss(baseline_probs, y[va]),
        "temperature": calibrator.temperature,
        "n_rows": int(n),
    }

    if verbose:
        print(
            "\nfinal (post-calibration, on val split):"
            f"\n  n_rows           {metrics['n_rows']}"
            f"\n  log_loss         {metrics['log_loss']:.4f}"
            f"\n  baseline_logloss {metrics['baseline_log_loss']:.4f}  (predict base rate {base_rate:.3f})"
            f"\n  brier            {metrics['brier']:.4f}"
            f"\n  ece              {metrics['ece']:.4f}"
            f"\n  temperature      {metrics['temperature']:.3f}"
        )
        print("\nreliability table (val):")
        _print_reliability(reliability_table(val_probs, y[va]))

    if out_path:
        save_bundle(out_path, model, scaler, calibrator, spec.features, sport)
        if verbose:
            print(f"\nsaved bundle to {out_path}.pt / {out_path}.json")

    return model, scaler, calibrator, metrics


def _load_real_frame(sport: str, seasons: list[int], max_games: int = 200):
    """Lazily import and call the matching historical loader (keeps torch-only paths light)."""
    if sport == "nfl":
        from liveedge.data_nfl import load_nfl_frame

        return load_nfl_frame(seasons)
    if sport == "nba":
        from liveedge.data_nba import load_nba_frame

        return load_nba_frame(seasons, max_games=max_games)
    if sport == "mlb":
        from liveedge.data_mlb import load_mlb_frame

        return load_mlb_frame(seasons)
    raise ValueError(f"unknown sport {sport!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a per-sport win-probability model.")
    parser.add_argument("--sport", choices=["nfl", "nba", "mlb"], required=True)
    parser.add_argument("--seasons", type=int, nargs="+", help="seasons to train on (real data)")
    parser.add_argument(
        "--synthetic", type=int, metavar="N", help="train on N synthetic games instead of real data"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default=None, help="output bundle path (default models/<sport>)")
    parser.add_argument(
        "--max-games", type=int, default=200, dest="max_games",
        help="NBA only: cap games pulled per season (use e.g. 2000 for a full season)",
    )
    args = parser.parse_args(argv)

    out_path = args.out or f"models/{args.sport}"

    if args.synthetic:
        from tools.simulate import synthetic_frame

        print(
            f"=== SYNTHETIC data ({args.synthetic} games) — proves the train/calibrate/eval "
            "machinery ONLY, not real-world performance ===\n"
        )
        df = synthetic_frame(args.sport, n_games=args.synthetic)
    else:
        if not args.seasons:
            parser.error("--seasons is required unless --synthetic N is given")
        print(f"=== REAL data: {args.sport} seasons {args.seasons} ===\n")
        df = _load_real_frame(args.sport, args.seasons, max_games=args.max_games)

    train_from_frame(df, args.sport, epochs=args.epochs, out_path=out_path)


if __name__ == "__main__":
    main()
