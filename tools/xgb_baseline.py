"""XGBoost baseline — the documented "recommended next step" for this project.

Gradient-boosted trees usually beat an MLP on tabular win-probability (it's what nflfastR
uses), so this trains an XGBoost model on the *same* feature contract (`features.py`) and the
*same* metrics (`model.py`) as liveedge and prints an apples-to-apples comparison against the
MLP. The meaningful test is out-of-sample BY SEASON — a random split flatters both models — so
prefer --seasons (train) + --test-season (held out).

TESTING / EXPERIMENT tool: lives in tools/, never imported by the live monitor.

    python -m tools.xgb_baseline --sport nfl --seasons 2010 2011 2012 2013 2014 2015 2016 \
        2017 2018 2019 2020 2021 2022 --test-season 2023
    python -m tools.xgb_baseline --sport nba --synthetic 30000
"""

from __future__ import annotations

# XGBoost and PyTorch each bundle an OpenMP runtime (libomp); loading both in one process
# aborts/segfaults on macOS unless the duplicate is allowed. This MUST run before importing
# numpy / xgboost / torch (torch arrives via the liveedge imports). tools/ experiment only.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-thread OpenMP: avoids the torch/xgb pool clash

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from liveedge.features import get_spec  # noqa: E402
from liveedge.model import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    log_loss,
    predict_prob,
    reliability_table,
)
from liveedge.train import _load_real_frame, _print_reliability, train_from_frame  # noqa: E402

# Reasonable win-prob defaults; documented as tunable.
_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": 0,
    "nthread": 1,
}


def _xy(df, spec) -> tuple[np.ndarray, np.ndarray]:
    return df[spec.features].to_numpy(np.float32), df[spec.label].to_numpy(np.float32)


def train_xgb(train_df, sport: str, *, params: dict | None = None, num_round: int = 350,
              early_stop: int = 30, seed: int = 0):
    """Train XGBoost with a 70/15/15 train/val/cal split (val = early stopping, cal = held out
    for probability calibration). Returns (booster, x_cal, y_cal)."""
    spec = get_spec(sport)
    df = train_df.dropna(subset=spec.features + [spec.label])
    x, y = _xy(df, spec)
    perm = np.random.default_rng(seed).permutation(len(x))
    n_tr, n_va = int(0.70 * len(x)), int(0.15 * len(x))
    tr, va, ca = perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]
    p = dict(_PARAMS, **(params or {}))
    dtr = xgb.DMatrix(x[tr], label=y[tr], feature_names=spec.features)
    dva = xgb.DMatrix(x[va], label=y[va], feature_names=spec.features)
    bst = xgb.train(p, dtr, num_boost_round=num_round, evals=[(dva, "val")],
                    early_stopping_rounds=early_stop, verbose_eval=False)
    return bst, x[ca], y[ca]


def _best_range(bst) -> tuple[int, int]:
    return (0, bst.best_iteration + 1)


def calibrate_xgb(bst, x_cal: np.ndarray, y_cal: np.ndarray, sport: str):
    """Temperature-scale XGBoost margins on the held-out cal split — the SAME calibration
    method the MLP gets, so the comparison isolates the model family from the calibration."""
    from liveedge.model import TemperatureScaler

    spec = get_spec(sport)
    dcal = xgb.DMatrix(x_cal, feature_names=spec.features)
    margins = bst.predict(dcal, output_margin=True, iteration_range=_best_range(bst))
    return TemperatureScaler().fit(margins, y_cal)


def xgb_predict(bst, df, sport: str, calibrator=None) -> np.ndarray:
    spec = get_spec(sport)
    x, _ = _xy(df, spec)
    d = xgb.DMatrix(x, feature_names=spec.features)
    if calibrator is None:
        return bst.predict(d, iteration_range=_best_range(bst))
    margins = bst.predict(d, output_margin=True, iteration_range=_best_range(bst))
    z = calibrator.apply(margins)  # TemperatureScaler.apply works on numpy too (logits / T)
    return 1.0 / (1.0 + np.exp(-z))


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(p, y),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y),
    }


def compare(train_df, test_df, sport: str) -> None:
    """Calibrated MLP vs. XGBoost (untuned / tuned / tuned+temperature-calibrated), trained on
    the same data and scored on the same held-out test set."""
    import torch

    torch.set_num_threads(1)  # keep torch single-threaded too (see the OpenMP note at top)
    spec = get_spec(sport)
    yte = test_df[spec.label].to_numpy(np.float32)

    # MLP (liveedge): trains + temperature-calibrates internally.
    mlp, scaler, cal, _ = train_from_frame(train_df, sport, epochs=25, verbose=False)
    mlp_p = predict_prob(mlp, scaler, test_df[spec.features].to_numpy(np.float32), cal)

    # XGBoost: untuned (depth 4) and light-tuned (depth 6); temperature-calibrate the tuned one.
    bst4, _, _ = train_xgb(train_df, sport)
    bst6, xc6, yc6 = train_xgb(train_df, sport, params={"max_depth": 6})
    tcal = calibrate_xgb(bst6, xc6, yc6, sport)

    rows = [
        ("MLP (calibrated)", mlp_p),
        ("XGB d4 (raw)", xgb_predict(bst4, test_df, sport)),
        ("XGB d6 (raw)", xgb_predict(bst6, test_df, sport)),
        (f"XGB d6 +temp(T={tcal.temperature:.2f})", xgb_predict(bst6, test_df, sport, tcal)),
    ]

    base = float(yte.mean())
    print(f"\nTest set: n={len(yte)}  base_rate={base:.3f}  "
          f"baseline_logloss={log_loss(np.full(len(yte), base), yte):.4f}")
    print(f"{'model':<24}{'log_loss':>10}{'brier':>9}{'ece':>9}")
    best_xgb_p = rows[-1][1]
    for name, p in rows:
        m = _metrics(p, yte)
        print(f"{name:<24}{m['log_loss']:>10.4f}{m['brier']:>9.4f}{m['ece']:>9.4f}")

    print("\nXGB d6 +temp reliability (test):")
    _print_reliability(reliability_table(best_xgb_p, yte))

    gain = bst6.get_score(importance_type="gain")
    if gain:
        print("\nXGBoost (d6) feature importance (gain):")
        for name, g in sorted(gain.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:<20}{g:10.1f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="XGBoost baseline vs. the liveedge MLP.")
    p.add_argument("--sport", choices=["nfl", "nba", "mlb"], required=True)
    p.add_argument("--seasons", type=int, nargs="+", help="training seasons (real data)")
    p.add_argument("--test-season", type=int, dest="test_season", help="held-out season")
    p.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic games instead")
    args = p.parse_args(argv)

    if args.synthetic:
        from tools.simulate import synthetic_frame

        df = synthetic_frame(args.sport, n_games=args.synthetic, seed=0)
        # Rows are emitted game-by-game, so a positional split keeps games disjoint.
        cut = int(0.8 * len(df))
        train_df, test_df = df.iloc[:cut], df.iloc[cut:]
        print(f"=== SYNTHETIC {args.sport} ({args.synthetic} games) — machinery comparison only ===")
    else:
        if not args.seasons or not args.test_season:
            p.error("give --seasons (train) and --test-season (held out), or --synthetic N")
        print(f"=== REAL {args.sport}: train {args.seasons}, test {args.test_season} ===")
        train_df = _load_real_frame(args.sport, args.seasons)
        test_df = _load_real_frame(args.sport, [args.test_season])

    compare(train_df, test_df, args.sport)


if __name__ == "__main__":
    main()
