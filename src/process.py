"""
process.py
==========
The data-generating "true plant": a Van de Vusse CSTR simulated with
realistic process noise (fluctuating operating conditions during a
campaign) and measurement noise (sensor noise on top of the true signal).

Each "campaign" = one operating run at a target setpoint (D, CAf, T),
during which the actual inputs wander stochastically around that
setpoint (piecewise-constant noise, refreshed every
CAMPAIGN_DURATION / N_NOISE_STEPS minutes). The reactor's true CB
concentration at the end of the campaign is treated as the quantity a
lab assay would measure -- this is the expensive/slow-to-obtain label
that a soft sensor is meant to avoid needing routinely.

This module deliberately knows nothing about the "assumed" kinetic
parameters used by the physics-based estimator (see physics_model.py):
this file is the plant, physics_model.py is the (imperfect) model of
the plant.
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import config as cfg


def _rate_constants(T, k1_ref, k2_ref, k3_ref, ea1, ea2, ea3):
    """Arrhenius rate constants at temperature T (K), referenced to T_REF."""
    inv_term = (1.0 / T - 1.0 / cfg.T_REF)
    k1 = k1_ref * np.exp(-ea1 / cfg.R_GAS * inv_term)
    k2 = k2_ref * np.exp(-ea2 / cfg.R_GAS * inv_term)
    k3 = k3_ref * np.exp(-ea3 / cfg.R_GAS * inv_term)
    return k1, k2, k3


def true_rate_constants(T):
    return _rate_constants(
        T, cfg.TRUE_K1_REF, cfg.TRUE_K2_REF, cfg.TRUE_K3_REF,
        cfg.TRUE_EA1, cfg.TRUE_EA2, cfg.TRUE_EA3,
    )


def _vdv_rhs(t, y, D_of_t, CAf_of_t, T_of_t):
    """Van de Vusse ODE right-hand side with time-varying (noisy) inputs."""
    CA, CB = y
    D = D_of_t(t)
    CAf = CAf_of_t(t)
    T = T_of_t(t)
    k1, k2, k3 = true_rate_constants(T)
    dCA = D * (CAf - CA) - k1 * CA - k3 * CA ** 2
    dCB = -D * CB + k1 * CA - k2 * CB
    return [dCA, dCB]


def _make_noisy_signal(setpoint, process_std, n_steps, duration, rng, lower_clip=None):
    """
    Build a piecewise-constant random-walk-like signal around `setpoint`:
    each of n_steps segments gets an independent Gaussian perturbation
    (this is the "fluctuating operating conditions" process noise).
    Returns (times, values, callable(t) -> value).
    """
    times = np.linspace(0.0, duration, n_steps + 1)
    perturbations = rng.normal(0.0, process_std, size=n_steps)
    values = setpoint + perturbations
    if lower_clip is not None:
        values = np.clip(values, lower_clip, None)

    def signal_fn(t):
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, n_steps - 1)
        return values[idx]

    return times, values, signal_fn


def simulate_campaign(D_setpoint, CAf_setpoint, T_setpoint, rng):
    """
    Simulate ONE operating campaign and return a dict with:
      - true_CB_final: the quantity a lab assay would report (the label)
      - measured_D, measured_CAf, measured_T: noisy sensor time series
        (process noise + measurement noise), each an array of N_NOISE_STEPS
        samples (one per noise segment) -- these are the ONLY things a
        soft sensor is allowed to see.
    """
    duration = cfg.CAMPAIGN_DURATION
    n_steps = cfg.N_NOISE_STEPS

    _, D_true_vals, D_fn = _make_noisy_signal(
        D_setpoint, cfg.PROCESS_NOISE_STD["D_rate"], n_steps, duration, rng, lower_clip=0.02
    )
    _, CAf_true_vals, CAf_fn = _make_noisy_signal(
        CAf_setpoint, cfg.PROCESS_NOISE_STD["CAf"], n_steps, duration, rng, lower_clip=0.5
    )
    _, T_true_vals, T_fn = _make_noisy_signal(
        T_setpoint, cfg.PROCESS_NOISE_STD["T"], n_steps, duration, rng, lower_clip=280.0
    )

    CA_init = cfg.CA0_INIT_FRACTION * CAf_setpoint
    y0 = [CA_init, cfg.CB0_INIT]

    sol = solve_ivp(
        _vdv_rhs, [0.0, duration], y0,
        args=(D_fn, CAf_fn, T_fn),
        method="LSODA", rtol=1e-8, atol=1e-10, dense_output=False,
    )
    true_CB_final = float(sol.y[1, -1])
    true_CA_final = float(sol.y[0, -1])

    # Sensor measurements = true (process-noisy) signal + independent
    # measurement noise, sampled once per noise segment (i.e. what an
    # operator's historian would log).
    measured_D = D_true_vals + rng.normal(0.0, cfg.SENSOR_NOISE_STD["D_rate"], n_steps)
    measured_CAf = CAf_true_vals + rng.normal(0.0, cfg.SENSOR_NOISE_STD["CAf"], n_steps)
    measured_T = T_true_vals + rng.normal(0.0, cfg.SENSOR_NOISE_STD["T"], n_steps)

    return {
        "true_CB_final": true_CB_final,
        "true_CA_final": true_CA_final,
        "measured_D": measured_D,
        "measured_CAf": measured_CAf,
        "measured_T": measured_T,
        "D_setpoint": D_setpoint,
        "CAf_setpoint": CAf_setpoint,
        "T_setpoint": T_setpoint,
    }


def sample_setpoint(rng):
    D = rng.uniform(*cfg.D_RANGE)
    CAf = rng.uniform(*cfg.CAF_RANGE)
    T = rng.uniform(*cfg.T_RANGE)
    return D, CAf, T


def generate_dataset(n_samples, seed):
    """Generate n_samples independent campaigns (i.i.d. draws of setpoints)."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_samples):
        D, CAf, T = sample_setpoint(rng)
        rec = simulate_campaign(D, CAf, T, rng)
        records.append(rec)
    return records
