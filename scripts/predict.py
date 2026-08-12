"""
predict.py
==========
Minimal, focused usage script: train a PGNN soft sensor on a chosen
number of labelled campaigns from the pool, then predict CB (with a 90%
conformal interval) for the held-out test campaigns, printing a small
report. This is deliberately NOT a web service / API / dashboard --
per the project's scope, the deliverable is the scientific artifact
(a soft sensor + its validation), not a product shell around it.

Usage:
    python scripts/predict.py --n-train 60 --n-show 10
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config as cfg
from src import data_pipeline as dp
from src import model_selection as ms
from src import conformal as cp
from src.metrics import mae, rmse, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    parser = argparse.ArgumentParser(description="Train and use the physics-guided soft sensor.")
    parser.add_argument("--n-train", type=int, default=60,
                         help="Number of labelled campaigns to train on (default: 60).")
    parser.add_argument("--n-show", type=int, default=10,
                         help="Number of test predictions to print (default: 10).")
    parser.add_argument("--lambda-value", type=float, default=None,
                         help="Fix the physics-loss weight instead of selecting it by "
                              "cross-validation (default: select automatically).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(DATA_DIR, "pool_campaigns.csv")):
        print("No data found. Run 'python scripts/generate_data.py' first.")
        sys.exit(1)

    (X_pool, y_pool, phys_pool), (X_test, y_test, phys_test) = dp.get_feature_arrays(DATA_DIR)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X_pool), size=args.n_train, replace=False)
    X_tr, y_tr = X_pool[idx], y_pool[idx]

    if args.lambda_value is None:
        lam = ms.select_lambda_cv(X_tr, y_tr, X_pool, phys_pool, cfg.LAMBDA_GRID, seed=args.seed)
        print(f"Selected physics-loss weight (via cross-validation): lambda = {lam:.3f}")
    else:
        lam = args.lambda_value
        print(f"Using fixed physics-loss weight: lambda = {lam:.3f}")

    model = ms.fit_final_pgnn(X_tr, y_tr, X_pool, phys_pool, lam, seed=args.seed)

    # calibrate a 90% conformal interval on a fresh calibration split from the pool
    remaining = np.setdiff1d(np.arange(len(X_pool)), idx)
    cal_idx = rng.choice(remaining, size=min(60, len(remaining)), replace=False)
    cal_resid = np.abs(y_pool[cal_idx] - model.predict(X_pool[cal_idx]))
    q_hat = cp.calibrate(cal_resid, alpha=0.10)

    pred = model.predict(X_test)
    lower, upper = cp.predict_interval(pred, q_hat)

    print(f"\nTrained on {args.n_train} labelled campaigns.")
    print(f"Test set performance ({len(X_test)} held-out campaigns):")
    print(f"  MAE  = {mae(y_test, pred):.4f} mol/L")
    print(f"  RMSE = {rmse(y_test, pred):.4f} mol/L")
    print(f"  R2   = {r2_score(y_test, pred):.4f}")
    print(f"  90% conformal interval half-width = {q_hat:.4f} mol/L")
    print(f"  Empirical coverage on test set     = "
          f"{cp.empirical_coverage(y_test, lower, upper):.3f}")

    print(f"\nExample predictions (first {args.n_show} test campaigns):")
    print(f"{'true CB':>10} {'predicted':>10} {'90% interval':>22} {'physics-only':>14}")
    for i in range(min(args.n_show, len(X_test))):
        print(f"{y_test[i]:10.4f} {pred[i]:10.4f} "
              f"[{lower[i]:7.4f}, {upper[i]:7.4f}]      {phys_test[i]:10.4f}")


if __name__ == "__main__":
    main()
