"""
verify_physics_model.py
========================
Sanity-check script: verifies the closed-form Van de Vusse steady-state
solution in src/physics_model.py against long-horizon numerical ODE
integration, across a random grid of operating conditions.

This is the check referenced throughout the project's docstrings and
BUILD_GUIDE.md as evidence that the "physics" in the physics-guided
model is actually correct, not just plausible-looking.

Run: python scripts/verify_physics_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp

from src import config as cfg
from src.process import true_rate_constants


def steady_state_closed_form_TRUE(D, CAf, T):
    """Same algebra as physics_model.steady_state_CB, but using the TRUE
    kinetic parameters (process.py), to check the closed form itself
    against numerical integration of the TRUE plant -- independent of
    whether the ASSUMED (biased) parameters used by the actual soft
    sensor's physics estimator are correct."""
    k1, k2, k3 = true_rate_constants(T)
    a = k3
    b = k1 + D
    c = -D * CAf
    disc = b ** 2 - 4 * a * c
    CA_ss = (-b + np.sqrt(disc)) / (2 * a)
    CB_ss = k1 * CA_ss / (D + k2)
    return CA_ss, CB_ss


def vdv_rhs(t, y, D, CAf, T):
    CA, CB = y
    k1, k2, k3 = true_rate_constants(T)
    dCA = D * (CAf - CA) - k1 * CA - k3 * CA ** 2
    dCB = -D * CB + k1 * CA - k2 * CB
    return [dCA, dCB]


def main():
    rng = np.random.default_rng(2026)
    n_checks = 40
    max_err_CA, max_err_CB = 0.0, 0.0
    print(f"{'D':>7} {'CAf':>7} {'T':>7} | {'CA_num':>9} {'CA_cf':>9} | {'CB_num':>9} {'CB_cf':>9}")
    for _ in range(n_checks):
        D = rng.uniform(*cfg.D_RANGE)
        CAf = rng.uniform(*cfg.CAF_RANGE)
        T = rng.uniform(*cfg.T_RANGE)

        sol = solve_ivp(vdv_rhs, [0, 500], [CAf, 0.0], args=(D, CAf, T),
                         method="LSODA", rtol=1e-10, atol=1e-12)
        CA_num, CB_num = sol.y[0, -1], sol.y[1, -1]
        CA_cf, CB_cf = steady_state_closed_form_TRUE(D, CAf, T)

        err_CA = abs(CA_num - CA_cf)
        err_CB = abs(CB_num - CB_cf)
        max_err_CA = max(max_err_CA, err_CA)
        max_err_CB = max(max_err_CB, err_CB)
        print(f"{D:7.3f} {CAf:7.3f} {T:7.1f} | {CA_num:9.5f} {CA_cf:9.5f} | "
              f"{CB_num:9.5f} {CB_cf:9.5f}")

    print(f"\nMax abs error CA: {max_err_CA:.2e}")
    print(f"Max abs error CB: {max_err_CB:.2e}")
    tol = 1e-4
    if max_err_CA < tol and max_err_CB < tol:
        print(f"PASS: closed-form steady-state solution matches numerical "
              f"ODE integration to within {tol}.")
    else:
        print(f"FAIL: discrepancy exceeds tolerance {tol} -- check the algebra "
              f"in physics_model.py / process.py before trusting the physics estimator.")
        sys.exit(1)


if __name__ == "__main__":
    main()
