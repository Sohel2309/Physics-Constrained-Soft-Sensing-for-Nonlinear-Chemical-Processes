"""
features.py
===========
Turns a raw campaign record (noisy sensor time series) into the fixed-
length feature vector used by all THREE models (PLS, MLP, PGNN).

All three models receive EXACTLY the same features -- this is a
deliberate experimental control: the only thing that differs between
the "Standard MLP" and the "Physics-Guided Neural Network" conditions
is the training loss (see nn_model.py), not the information available
to them. This isolates the causal effect of the physics-guidance term.

Feature set (10 features, see config.FEATURE_NAMES):
  - D_mean, CAf_mean, T_mean       : mean of the noisy sensor readings
  - D_std,  CAf_std,  T_std        : within-campaign variability (a
                                      cheap, always-available signal of
                                      how much the operating condition
                                      wandered during the run)
  - invT_mean                      : mean of 1/T -- the natural variable
                                      in the Arrhenius law, included
                                      because the true rate constants are
                                      exponential in 1/T, not in T itself
  - D_sq_mean                      : mean of D**2 -- the Van de Vusse
                                      steady state is a quadratic in D
                                      (via the CA quadratic), so this is
                                      a physically motivated engineered
                                      feature, not an arbitrary one
  - CAf_D_mean, D_T_mean           : interaction terms motivated by the
                                      product terms appearing in the
                                      governing ODEs (D*CAf, and D
                                      interacting with the temperature-
                                      dependent rate constants)

Note: these are all pre-computed from the RAW measured signals -- they
are legitimate "off-the-shelf feature engineering", not physics-loss
guidance (that only happens inside nn_model.py's training loss). PLS
and the Standard MLP get exactly the same engineered features as the
PGNN; nothing about the physics is smuggled in through the inputs.
"""

import numpy as np

from . import config as cfg


def record_to_features(record):
    D = record["measured_D"]
    CAf = record["measured_CAf"]
    T = record["measured_T"]

    D_mean, CAf_mean, T_mean = D.mean(), CAf.mean(), T.mean()
    D_std, CAf_std, T_std = D.std(), CAf.std(), T.std()
    invT_mean = np.mean(1.0 / T)
    D_sq_mean = np.mean(D ** 2)
    CAf_D_mean = np.mean(CAf * D)
    D_T_mean = np.mean(D * T)

    return np.array([
        D_mean, CAf_mean, T_mean,
        D_std, CAf_std, T_std,
        invT_mean, D_sq_mean, CAf_D_mean, D_T_mean,
    ], dtype=float)


def records_to_dataset(records):
    """
    Returns:
      X: (n, 10) feature matrix
      y: (n,) true CB labels (the expensive "assay" result)
      phys: (n,) physics-based CB estimate (zero-label-cost reference)
    """
    from .physics_model import physics_estimate_from_record

    X = np.stack([record_to_features(r) for r in records])
    y = np.array([r["true_CB_final"] for r in records], dtype=float)
    phys = np.array([physics_estimate_from_record(r) for r in records], dtype=float)
    return X, y, phys
