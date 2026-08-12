"""
model_selection.py
===================
Helper for selecting the physics-loss weight lambda for the PGNN using
K-FOLD CROSS-VALIDATION WITHIN THE TRAINING SET ONLY -- the held-out
test set is never touched during hyperparameter selection, for any
model. This is what makes the final reported test-set comparison fair:
lambda is not chosen to make the PGNN look good on the numbers we
report.

For very small training sets, the fold count is reduced automatically
(down to leave-one-out) so that every fold still has at least 2 points
to train on.

To keep runtime reasonable, cross-validation candidate models are
trained with a reduced epoch budget (CV_N_EPOCHS); the final model
(after lambda is chosen) is retrained on the full training set with the
full epoch budget (config.NN_N_EPOCHS).
"""

import numpy as np

from . import config as cfg
from .nn_model import PhysicsGuidedMLP
from .metrics import rmse

CV_N_EPOCHS = 300


def select_lambda_cv(X_train, y_train, X_extra_pool, phys_extra_pool, lambda_grid, seed):
    """
    Returns the lambda in lambda_grid with the lowest mean cross-validated
    RMSE, using K-fold CV restricted to (X_train, y_train). The physics
    loss during CV candidate training uses the FULL extra pool (this is
    legitimate: the extra pool's physics estimates require no labels, so
    using them during model selection does not leak any test-set label
    information).
    """
    n = len(X_train)
    k = min(5, n)  # leave-one-out if n < 5
    if k < 2:
        # cannot cross-validate with fewer than 2 points; fall back to a
        # small default in the middle of the grid
        return float(np.median(lambda_grid))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    mean_errs = []
    for lam in lambda_grid:
        fold_errs = []
        for fi in range(k):
            val_idx = folds[fi]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fi])
            if len(train_idx) < 2:
                continue
            net = PhysicsGuidedMLP(
                n_inputs=X_train.shape[1], hidden_sizes=cfg.NN_HIDDEN_SIZES,
                lam=lam, learning_rate=cfg.NN_LEARNING_RATE,
                n_epochs=CV_N_EPOCHS, l2_reg=cfg.NN_L2_REG, seed=seed,
            )
            if lam > 0:
                net.fit(X_train[train_idx], y_train[train_idx],
                        X_extra=X_extra_pool, phys_extra=phys_extra_pool)
            else:
                net.fit(X_train[train_idx], y_train[train_idx])
            pred = net.predict(X_train[val_idx])
            fold_errs.append(rmse(y_train[val_idx], pred))
        mean_errs.append(np.mean(fold_errs) if fold_errs else np.inf)

    best_idx = int(np.argmin(mean_errs))
    return float(lambda_grid[best_idx])


def fit_final_pgnn(X_train, y_train, X_extra_pool, phys_extra_pool, lam, seed):
    net = PhysicsGuidedMLP(
        n_inputs=X_train.shape[1], hidden_sizes=cfg.NN_HIDDEN_SIZES,
        lam=lam, learning_rate=cfg.NN_LEARNING_RATE,
        n_epochs=cfg.NN_N_EPOCHS, l2_reg=cfg.NN_L2_REG, seed=seed,
    )
    if lam > 0:
        net.fit(X_train, y_train, X_extra=X_extra_pool, phys_extra=phys_extra_pool)
    else:
        net.fit(X_train, y_train)
    return net


def fit_standard_mlp(X_train, y_train, seed):
    net = PhysicsGuidedMLP(
        n_inputs=X_train.shape[1], hidden_sizes=cfg.NN_HIDDEN_SIZES,
        lam=0.0, learning_rate=cfg.NN_LEARNING_RATE,
        n_epochs=cfg.NN_N_EPOCHS, l2_reg=cfg.NN_L2_REG, seed=seed,
    )
    net.fit(X_train, y_train)
    return net
