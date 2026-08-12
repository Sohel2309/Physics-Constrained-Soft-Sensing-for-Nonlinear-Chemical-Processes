"""
run_conformal_experiment.py
============================
Split conformal prediction on top of the PGNN (and, for comparison, the
Standard MLP), evaluated at a representative training-set size.

Because every campaign is an i.i.d. draw of a random operating setpoint
(see process.sample_setpoint), the calibration and test sets ARE
exchangeable, so the standard split-conformal MARGINAL coverage
guarantee legitimately applies here (see src/conformal.py docstring for
the precise statement and its limits -- marginal, not conditional on X).

For each of several nominal confidence levels (config.CONFORMAL_ALPHA_LEVELS),
and repeated over config.N_SEEDS random train/calibration splits:
  - Train the point-prediction model on a training subset
  - Calibrate the conformal interval on a separate calibration subset
  - Evaluate empirical coverage AND mean interval width on the held-out
    test set

Output:
  results/tables/conformal_results.csv
  results/plots/conformal_coverage.png   (empirical vs. nominal coverage
                                            -- the "reliability diagram")
  results/plots/conformal_example_intervals.png (a handful of example
                                            predictions with their
                                            intervals, sorted by true CB)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src import data_pipeline as dp
from src import model_selection as ms
from src import conformal as cp
from src.metrics import rmse

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CONFORMAL_TRAIN_SIZE = 100   # a representative, moderate training size
N_CONFORMAL_SEEDS = 15
FIXED_LAMBDA = 0.1           # representative lambda (mid-range from the sweep), fixed here
                              # so this experiment isolates conformal calibration behaviour
                              # rather than re-doing lambda selection every seed


def main():
    (X_pool, y_pool, phys_pool), (X_test, y_test, phys_test) = dp.get_feature_arrays(DATA_DIR)

    rows = []
    example_record = None

    for seed in range(N_CONFORMAL_SEEDS):
        rng = np.random.default_rng(9000 + seed)
        # split the pool into a train subset and a separate calibration subset
        perm = rng.permutation(len(X_pool))
        train_idx = perm[:CONFORMAL_TRAIN_SIZE]
        n_cal = int(np.ceil(CONFORMAL_TRAIN_SIZE * cfg.CONFORMAL_CALIBRATION_FRACTION /
                             (1 - cfg.CONFORMAL_CALIBRATION_FRACTION)))
        cal_idx = perm[CONFORMAL_TRAIN_SIZE:CONFORMAL_TRAIN_SIZE + n_cal]

        X_tr, y_tr = X_pool[train_idx], y_pool[train_idx]
        X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]

        for model_name in ["Standard_MLP", "PGNN"]:
            if model_name == "Standard_MLP":
                model = ms.fit_standard_mlp(X_tr, y_tr, seed=seed)
            else:
                model = ms.fit_final_pgnn(X_tr, y_tr, X_pool, phys_pool, FIXED_LAMBDA, seed=seed)

            cal_resid = np.abs(y_cal - model.predict(X_cal))
            test_pred = model.predict(X_test)

            for alpha in cfg.CONFORMAL_ALPHA_LEVELS:
                q_hat = cp.calibrate(cal_resid, alpha)
                lower, upper = cp.predict_interval(test_pred, q_hat)
                coverage = cp.empirical_coverage(y_test, lower, upper)
                width = cp.mean_interval_width(lower, upper)
                rows.append({
                    "seed": seed, "model": model_name,
                    "nominal_confidence": 1 - alpha,
                    "empirical_coverage": coverage,
                    "mean_interval_width": width,
                    "q_hat": q_hat,
                    "test_RMSE": rmse(y_test, test_pred),
                })

            if model_name == "PGNN" and seed == 0:
                q90 = cp.calibrate(cal_resid, 0.10)
                lower90, upper90 = cp.predict_interval(test_pred, q90)
                example_record = (y_test.copy(), test_pred.copy(), lower90.copy(), upper90.copy())

        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "tables", "conformal_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    make_plots(df, example_record)


def make_plots(df, example_record):
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    # --- Reliability diagram: empirical vs nominal coverage ---
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for model_name, color in [("Standard_MLP", "#ff7f0e"), ("PGNN", "#2ca02c")]:
        sub = df[df.model == model_name]
        grouped = sub.groupby("nominal_confidence")["empirical_coverage"].agg(["mean", "std"])
        plt.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                      marker="o", label=model_name, color=color, capsize=4)
    plt.xlabel("Nominal confidence level")
    plt.ylabel("Empirical coverage on held-out test set")
    plt.title(f"Split conformal calibration (N_train={CONFORMAL_TRAIN_SIZE}, "
              f"{N_CONFORMAL_SEEDS} seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "plots", "conformal_coverage.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved plot: {out1}")

    # --- Example intervals plot ---
    if example_record is not None:
        y_true, pred, lower, upper = example_record
        order = np.argsort(y_true)
        n_show = min(60, len(y_true))
        show_idx = order[np.linspace(0, len(order) - 1, n_show).astype(int)]

        plt.figure(figsize=(9, 5))
        xs = np.arange(n_show)
        plt.fill_between(xs, lower[show_idx], upper[show_idx], alpha=0.3,
                          color="#2ca02c", label="90% conformal interval (PGNN)")
        plt.plot(xs, pred[show_idx], "o", color="#2ca02c", markersize=3, label="Point prediction")
        plt.plot(xs, y_true[show_idx], "x", color="black", markersize=5, label="True CB")
        plt.xlabel("Test example (sorted by true CB)")
        plt.ylabel("CB (mol/L)")
        plt.title("Example 90% conformal prediction intervals (PGNN)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(RESULTS_DIR, "plots", "conformal_example_intervals.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"Saved plot: {out2}")


if __name__ == "__main__":
    main()
