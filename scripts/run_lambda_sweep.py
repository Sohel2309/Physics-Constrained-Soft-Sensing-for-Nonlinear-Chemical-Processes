"""
run_lambda_sweep.py
====================
Dedicated experiment for the physics-loss weight lambda in
    L_total = L_data + lambda * L_physics

At two representative training-set sizes (one very small, one moderate --
config.LAMBDA_SWEEP_SIZES), train the PGNN at EVERY lambda in
config.LAMBDA_GRID (no CV selection here -- the whole point of this
script is to show the sweep itself), repeated over config.N_SEEDS seeds.

Reports, for every lambda:
  - test MAE / RMSE (does the physics term help or hurt prediction?)
  - "physics adherence": mean absolute difference between the network's
    prediction and the physics-based estimate on the test set (does the
    prediction actually move toward the physics estimate as lambda grows?
    this is a direct, mechanistic check that the loss term is doing what
    it is mathematically supposed to do, independent of whether that is
    good for accuracy)

Output:
  results/tables/lambda_sweep_results.csv
  results/plots/lambda_sweep_N<small>.png
  results/plots/lambda_sweep_N<moderate>.png
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
from src.metrics import mae, rmse

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

LAMBDA_SWEEP_SIZES = [15, 100]
N_SWEEP_SEEDS = 8


def main():
    (X_pool, y_pool, phys_pool), (X_test, y_test, phys_test) = dp.get_feature_arrays(DATA_DIR)

    rows = []
    for N in LAMBDA_SWEEP_SIZES:
        for seed in range(N_SWEEP_SEEDS):
            rng = np.random.default_rng(5000 * N + seed)
            idx = rng.choice(len(X_pool), size=N, replace=False)
            X_tr, y_tr = X_pool[idx], y_pool[idx]

            for lam in cfg.LAMBDA_GRID:
                net = ms.fit_final_pgnn(X_tr, y_tr, X_pool, phys_pool, lam, seed=seed)
                pred = net.predict(X_test)
                adherence = mae(pred, phys_test)  # distance from physics estimate
                rows.append({
                    "N_train": N, "seed": seed, "lambda": lam,
                    "MAE": mae(y_test, pred), "RMSE": rmse(y_test, pred),
                    "physics_adherence_MAE": adherence,
                })
            print(f"N={N} seed={seed} done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "tables", "lambda_sweep_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    for N in LAMBDA_SWEEP_SIZES:
        sub = df[df.N_train == N]
        grouped = sub.groupby("lambda").agg(
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            adherence_mean=("physics_adherence_MAE", "mean"),
        )

        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.errorbar(grouped.index, grouped["RMSE_mean"], yerr=grouped["RMSE_std"],
                      marker="o", color="#2ca02c", label="Test RMSE")
        ax1.set_xlabel("Physics-loss weight (lambda)")
        ax1.set_ylabel("Test RMSE (mol/L)", color="#2ca02c")
        ax1.tick_params(axis="y", labelcolor="#2ca02c")
        ax1.set_xscale("symlog", linthresh=0.02)

        ax2 = ax1.twinx()
        ax2.plot(grouped.index, grouped["adherence_mean"], marker="s", color="#d62728",
                  linestyle="--", label="Physics adherence (MAE to physics estimate)")
        ax2.set_ylabel("Mean |prediction - physics estimate| (mol/L)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        plt.title(f"Effect of physics-loss weight lambda (N_train={N}, {N_SWEEP_SEEDS} seeds)")
        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "plots", f"lambda_sweep_N{N}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
