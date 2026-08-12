"""
physics_model.py
=================
The first-principles physics-based estimator for CB.

Given the (noisy, measured) mean operating conditions of a campaign, this
computes the CLOSED-FORM steady-state solution of the Van de Vusse CSTR
using the ASSUMED (imperfect) kinetic parameters -- i.e. what an engineer
with a process model but no product-quality lab data would predict.

Derivation
----------
At steady state, dCA/dt = 0:
    D*(CAf - CA) - k1*CA - k3*CA**2 = 0
    => k3*CA**2 + (k1 + D)*CA - D*CAf = 0        (quadratic in CA)
    => CA = [-(k1+D) + sqrt((k1+D)**2 + 4*k3*D*CAf)] / (2*k3)   (positive root)

At steady state, dCB/dt = 0:
    -D*CB + k1*CA - k2*CB = 0
    => CB = k1*CA / (D + k2)

This closed form was numerically verified against long-horizon ODE
integration in scripts/verify_physics_model.py (agreement to machine
precision in the noiseless case).

Crucially, this estimator requires NO labelled CB data at all -- it only
needs the (cheaply, routinely measured) operating conditions plus the
ASSUMED kinetic parameters. That is what makes it valuable when labelled
(assayed) data is scarce: it is a "free" source of information.
"""

import numpy as np

from . import config as cfg
from .process import _rate_constants


def assumed_rate_constants(T):
    return _rate_constants(
        T, cfg.ASSUMED_K1_REF, cfg.ASSUMED_K2_REF, cfg.ASSUMED_K3_REF,
        cfg.ASSUMED_EA1, cfg.ASSUMED_EA2, cfg.ASSUMED_EA3,
    )


def steady_state_CB(D, CAf, T):
    """
    Vectorised closed-form steady-state CB using the ASSUMED kinetics.
    D, CAf, T may be scalars or numpy arrays of the same shape.
    """
    D = np.asarray(D, dtype=float)
    CAf = np.asarray(CAf, dtype=float)
    T = np.asarray(T, dtype=float)

    k1, k2, k3 = assumed_rate_constants(T)

    a = k3
    b = k1 + D
    c = -D * CAf
    disc = b ** 2 - 4 * a * c
    disc = np.clip(disc, 0.0, None)  # guard against tiny negative rounding
    CA_ss = (-b + np.sqrt(disc)) / (2 * a)
    CA_ss = np.clip(CA_ss, 0.0, None)
    CB_ss = k1 * CA_ss / (D + k2)
    return CB_ss


def physics_estimate_from_record(record):
    """Physics estimate using the MEAN of the noisy measured signals."""
    D_mean = float(np.mean(record["measured_D"]))
    CAf_mean = float(np.mean(record["measured_CAf"]))
    T_mean = float(np.mean(record["measured_T"]))
    return steady_state_CB(D_mean, CAf_mean, T_mean)
