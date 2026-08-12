"""
run_main_experiment.py
=======================
THE central experiment of the project.

Research question: can physical process constraints improve ML quality
prediction when labelled training data is scarce?

For every training-set size N in config.TRAIN_SIZES, and for
config.N_SEEDS independent random training subsets of that size (drawn
from the fixed data/pool_campaigns.csv pool):

  - Fit a PLS baseline
  - Fit a Standard MLP (lambda = 0)
  - Fit a Physics-Guided Neural Network (lambda selected by cross-
    validation WITHIN the training subset only -- see
    src/model_selection.py)

All three are evaluated on the SAME fixed, never-touched-during-training
held-out test set (data/test_campaigns.csv). The physics-only estimator
(zero trainable parameters, needs no labelled data at all) is also
evaluated on the same test set, as a reference line, not as a fourth
competing "model".

Output:
  results/tables/main_experiment_results.csv   (one row per N, seed, model)
  results/plots/main_experiment_rmse.png
  results/plots/main_experiment_mae.png
  results/plots/main_experiment_r2.png
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src import data_pipeline as dp
from src import model_selection as ms
from src.pls_model import PLSSoftSensor
from src.metrics import mae, rmse, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main(train_sizes=None, append=False, make_plots_flag=True):
    train_sizes = train_sizes if train_sizes is not None else cfg.TRAIN_SIZES
    (X_pool, y_pool, phys_pool), (X_test, y_test, phys_test) = dp.get_feature_arrays(DATA_DIR)

    phys_only_mae = mae(y_test, phys_test)
    phys_only_rmse = rmse(y_test, phys_test)
    phys_only_r2 = r2_score(y_test, phys_test)
    print(f"[reference] physics-only on full test set: "
          f"MAE={phys_only_mae:.4f} RMSE={phys_only_rmse:.4f} R2={phys_only_r2:.4f}")

    rows = []
    total_conditions = len(train_sizes) * cfg.N_SEEDS
    done = 0
    t_start = time.time()

    for N in train_sizes:
        for seed in range(cfg.N_SEEDS):
            rng = np.random.default_rng(1000 * N + seed)
            idx = rng.choice(len(X_pool), size=N, replace=False)
            X_tr, y_tr = X_pool[idx], y_pool[idx]

            # ---- PLS ----
            pls = PLSSoftSensor(max_components=8).fit(X_tr, y_tr)
            pred_pls = pls.predict(X_test)

            # ---- Standard MLP (lambda = 0) ----
            mlp = ms.fit_standard_mlp(X_tr, y_tr, seed=seed)
            pred_mlp = mlp.predict(X_test)

            # ---- Physics-Guided Neural Network ----
            lam = ms.select_lambda_cv(X_tr, y_tr, X_pool, phys_pool, cfg.LAMBDA_GRID, seed=seed)
            pgnn = ms.fit_final_pgnn(X_tr, y_tr, X_pool, phys_pool, lam, seed=seed)
            pred_pgnn = pgnn.predict(X_test)

            for model_name, pred in [
                ("PLS", pred_pls),
                ("Standard_MLP", pred_mlp),
                ("PGNN", pred_pgnn),
            ]:
                rows.append({
                    "N_train": N,
                    "seed": seed,
                    "model": model_name,
                    "lambda_selected": lam if model_name == "PGNN" else np.nan,
                    "MAE": mae(y_test, pred),
                    "RMSE": rmse(y_test, pred),
                    "R2": r2_score(y_test, pred),
                })

            done += 1
            elapsed = time.time() - t_start
            print(f"[{done}/{total_conditions}] N={N:4d} seed={seed:2d}  "
                  f"PLS_RMSE={rmse(y_test,pred_pls):.4f}  MLP_RMSE={rmse(y_test,pred_mlp):.4f}  "
                  f"PGNN_RMSE={rmse(y_test,pred_pgnn):.4f} (lambda={lam:.3f})  "
                  f"[{elapsed:.0f}s elapsed]")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "tables", "main_experiment_results.csv")
    if append and os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results to {out_csv} ({len(df)} rows total)")

    # Also save the physics-only reference numbers
    with open(os.path.join(RESULTS_DIR, "tables", "physics_only_reference.txt"), "w") as f:
        f.write(f"physics_only_MAE={phys_only_mae:.6f}\n")
        f.write(f"physics_only_RMSE={phys_only_rmse:.6f}\n")
        f.write(f"physics_only_R2={phys_only_r2:.6f}\n")

    if make_plots_flag:
        make_plots(df, phys_only_mae, phys_only_rmse, phys_only_r2)


def make_plots(df, phys_mae, phys_rmse, phys_r2):
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    colors = {"PLS": "#1f77b4", "Standard_MLP": "#ff7f0e", "PGNN": "#2ca02c"}
    labels = {"PLS": "PLS", "Standard_MLP": "Standard MLP", "PGNN": "PGNN (physics-guided)"}

    for metric, phys_ref, ylabel in [
        ("MAE", phys_mae, "Test MAE (mol/L)"),
        ("RMSE", phys_rmse, "Test RMSE (mol/L)"),
        ("R2", phys_r2, "Test R2"),
    ]:
        plt.figure(figsize=(7, 5))
        for model_name in ["PLS", "Standard_MLP", "PGNN"]:
            sub = df[df.model == model_name]
            grouped = sub.groupby("N_train")[metric].agg(["mean", "std"])
            plt.plot(grouped.index, grouped["mean"], marker="o", label=labels[model_name],
                      color=colors[model_name])
            plt.fill_between(grouped.index,
                              grouped["mean"] - grouped["std"],
                              grouped["mean"] + grouped["std"],
                              alpha=0.15, color=colors[model_name])
        plt.axhline(phys_ref, color="gray", linestyle="--", linewidth=1.5,
                    label="Physics-only (0 trained parameters)")
        plt.xscale("log")
        plt.xlabel("Training-set size N (log scale)")
        plt.ylabel(ylabel)
        plt.title(f"{metric} vs. training-set size (mean +/- std over {cfg.N_SEEDS} seeds)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "plots", f"main_experiment_{metric.lower()}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sizes", type=int, nargs="+", default=None,
                         help="Subset of training sizes to run this invocation "
                              "(default: all of config.TRAIN_SIZES). Useful for "
                              "running the experiment in smaller chunks.")
    parser.add_argument("--append", action="store_true",
                         help="Append to (rather than overwrite) the existing results CSV.")
    parser.add_argument("--no-plots", action="store_true",
                         help="Skip plot generation (useful for intermediate chunks).")
    args = parser.parse_args()
    main(train_sizes=args.train_sizes, append=args.append, make_plots_flag=not args.no_plots)
