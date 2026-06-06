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


def train_xgb(train_df, sport: str, *, num_round: int = 600, early_stop: int = 40,
              val_frac: float = 0.15, seed: int = 0) -> "xgb.Booster":
    """Train XGBoost on train_df, carving a random val split for early stopping."""
    spec = get_spec(sport)
    df = train_df.dropna(subset=spec.features + [spec.label])
    x, y = _xy(df, spec)
    perm = np.random.default_rng(seed).permutation(len(x))
    cut = int((1 - val_frac) * len(x))
    tr, va = perm[:cut], perm[cut:]
    dtr = xgb.DMatrix(x[tr], label=y[tr], feature_names=spec.features)
    dva = xgb.DMatrix(x[va], label=y[va], feature_names=spec.features)
    return xgb.train(
        _PARAMS, dtr, num_boost_round=num_round,
        evals=[(dva, "val")], early_stopping_rounds=early_stop, verbose_eval=False,
    )


def xgb_predict(bst: "xgb.Booster", df, sport: str) -> np.ndarray:
    spec = get_spec(sport)
    x, _ = _xy(df, spec)
    d = xgb.DMatrix(x, feature_names=spec.features)
    return bst.predict(d, iteration_range=(0, bst.best_iteration + 1))


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(p, y),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y),
    }


def compare(train_df, test_df, sport: str) -> None:
    """Train the MLP and XGBoost on the same data and compare on the same test set."""
    import torch

    torch.set_num_threads(1)  # keep torch single-threaded too (see the OpenMP note at top)
    spec = get_spec(sport)
    yte = test_df[spec.label].to_numpy(np.float32)

    # MLP (liveedge): trains + temperature-calibrates internally on train_df.
    mlp, scaler, cal, _ = train_from_frame(train_df, sport, epochs=25, verbose=False)
    mlp_p = predict_prob(mlp, scaler, test_df[spec.features].to_numpy(np.float32), cal)
    mlp_m = _metrics(mlp_p, yte)

    # XGBoost baseline.
    bst = train_xgb(train_df, sport)
    xgb_m = _metrics(xgb_predict(bst, test_df, sport), yte)

    base = float(yte.mean())
    print(
        f"\nTest set: n={len(yte)}  base_rate={base:.3f}  "
        f"baseline_logloss={log_loss(np.full(len(yte), base), yte):.4f}"
    )
    print(f"{'model':<10}{'log_loss':>10}{'brier':>9}{'ece':>9}")
    print(f"{'MLP':<10}{mlp_m['log_loss']:>10.4f}{mlp_m['brier']:>9.4f}{mlp_m['ece']:>9.4f}")
    print(f"{'XGBoost':<10}{xgb_m['log_loss']:>10.4f}{xgb_m['brier']:>9.4f}{xgb_m['ece']:>9.4f}")

    print("\nXGBoost reliability (test):")
    _print_reliability(reliability_table(xgb_predict(bst, test_df, sport), yte))

    gain = bst.get_score(importance_type="gain")
    if gain:
        print("\nXGBoost feature importance (gain):")
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
