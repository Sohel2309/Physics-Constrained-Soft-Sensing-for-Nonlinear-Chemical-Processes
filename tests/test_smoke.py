"""
test_smoke.py
==============
Fast smoke tests (run in well under a minute) that verify the
installation is working correctly and the core scientific claims of the
codebase (closed-form physics solution, gradient correctness, basic
model fitting) hold -- WITHOUT re-running the full multi-minute
experiments.

Run from the project root:
    python -m pytest tests/test_smoke.py -v
or, if pytest is not installed:
    python tests/test_smoke.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_physics_closed_form_matches_ode():
    from scipy.integrate import solve_ivp
    from src.process import true_rate_constants
    from src import config as cfg

    def rhs(t, y, D, CAf, T):
        CA, CB = y
        k1, k2, k3 = true_rate_constants(T)
        return [D * (CAf - CA) - k1 * CA - k3 * CA ** 2, -D * CB + k1 * CA - k2 * CB]

    D, CAf, T = 0.9, 10.0, 350.0
    sol = solve_ivp(rhs, [0, 400], [CAf, 0.0], args=(D, CAf, T), method="LSODA",
                     rtol=1e-10, atol=1e-12)
    CA_num, CB_num = sol.y[0, -1], sol.y[1, -1]

    k1, k2, k3 = true_rate_constants(T)
    a, b, c = k3, k1 + D, -D * CAf
    CA_cf = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    CB_cf = k1 * CA_cf / (D + k2)

    assert abs(CA_num - CA_cf) < 1e-4, f"CA mismatch: {CA_num} vs {CA_cf}"
    assert abs(CB_num - CB_cf) < 1e-4, f"CB mismatch: {CB_num} vs {CB_cf}"
    print("PASS: test_physics_closed_form_matches_ode")


def test_nn_gradient_correctness():
    """Numerical gradient check on a small random network/dataset."""
    from src.nn_model import PhysicsGuidedMLP

    rng = np.random.default_rng(0)
    n, d = 6, 4
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    Xe = rng.normal(size=(9, d))
    phys = rng.normal(size=9)

    net = PhysicsGuidedMLP(n_inputs=d, hidden_sizes=[5, 5], lam=0.7, n_epochs=1,
                            learning_rate=0.0, seed=1)
    net._x_mean = X.mean(axis=0)
    net._x_std = X.std(axis=0)
    net._x_std[net._x_std < 1e-8] = 1.0
    net._y_mean = y.mean()
    net._y_std = y.std() if y.std() > 1e-8 else 1.0

    Xs = (X - net._x_mean) / net._x_std
    ys = (y - net._y_mean) / net._y_std
    Xse = (Xe - net._x_mean) / net._x_std
    physs = (phys - net._y_mean) / net._y_std

    def total_loss():
        act, _ = net._forward(Xs)
        pred = act[-1].ravel()
        data_loss = np.mean((pred - ys) ** 2)
        act2, _ = net._forward(Xse)
        pred2 = act2[-1].ravel()
        phys_loss = np.mean((pred2 - physs) ** 2)
        return data_loss + net.lam * phys_loss

    act, pre = net._forward(Xs)
    pred = act[-1].ravel()
    grad_out_data = (2.0 / len(ys)) * (pred - ys).reshape(-1, 1)
    dW_data, db_data = net._backward(act, pre, grad_out_data)

    act2, pre2 = net._forward(Xse)
    pred2 = act2[-1].ravel()
    grad_out_phys = (2.0 / len(physs)) * (pred2 - physs).reshape(-1, 1)
    dW_phys, db_phys = net._backward(act2, pre2, grad_out_phys)

    dW = [dW_data[i] + net.lam * dW_phys[i] for i in range(net.n_layers)]

    eps = 1e-5
    max_err = 0.0
    li = 0
    for idx in [(0, 0), (1, 1), (2, 0)]:
        if idx[0] >= net.W[li].shape[0] or idx[1] >= net.W[li].shape[1]:
            continue
        orig = net.W[li][idx]
        net.W[li][idx] = orig + eps
        lp = total_loss()
        net.W[li][idx] = orig - eps
        lm = total_loss()
        net.W[li][idx] = orig
        numgrad = (lp - lm) / (2 * eps)
        analytic = dW[li][idx] - net.l2_reg * net.W[li][idx]
        max_err = max(max_err, abs(numgrad - analytic))

    assert max_err < 1e-3, f"Gradient check failed, max error {max_err}"
    print(f"PASS: test_nn_gradient_correctness (max error {max_err:.2e})")


def test_end_to_end_small_pipeline():
    """Generate a tiny dataset in-memory and run all three models end to end."""
    from src import process, features as feat, model_selection as ms
    from src.pls_model import PLSSoftSensor
    from src.metrics import rmse

    pool_recs = process.generate_dataset(60, seed=42)
    test_recs = process.generate_dataset(30, seed=43)
    X_pool, y_pool, phys_pool = feat.records_to_dataset(pool_recs)
    X_test, y_test, _ = feat.records_to_dataset(test_recs)

    idx = np.arange(15)
    X_tr, y_tr = X_pool[idx], y_pool[idx]

    pls = PLSSoftSensor(max_components=8).fit(X_tr, y_tr)
    pred_pls = pls.predict(X_test)
    assert np.all(np.isfinite(pred_pls))

    mlp = ms.fit_standard_mlp(X_tr, y_tr, seed=0)
    pred_mlp = mlp.predict(X_test)
    assert np.all(np.isfinite(pred_mlp))

    lam = ms.select_lambda_cv(X_tr, y_tr, X_pool, phys_pool, [0.0, 0.1, 0.5], seed=0)
    pgnn = ms.fit_final_pgnn(X_tr, y_tr, X_pool, phys_pool, lam, seed=0)
    pred_pgnn = pgnn.predict(X_test)
    assert np.all(np.isfinite(pred_pgnn))

    for name, pred in [("PLS", pred_pls), ("MLP", pred_mlp), ("PGNN", pred_pgnn)]:
        r = rmse(y_test, pred)
        assert r < 5.0, f"{name} RMSE implausibly large: {r}"  # sanity bound, not a tuned threshold

    print("PASS: test_end_to_end_small_pipeline "
          f"(RMSE: PLS={rmse(y_test,pred_pls):.3f} MLP={rmse(y_test,pred_mlp):.3f} "
          f"PGNN={rmse(y_test,pred_pgnn):.3f})")


def test_conformal_basic_properties():
    from src import conformal as cp

    rng = np.random.default_rng(0)
    cal_resid = np.abs(rng.normal(0, 1, size=200))
    q_hat = cp.calibrate(cal_resid, alpha=0.10)
    assert q_hat > 0

    test_true = rng.normal(0, 1, size=1000)
    test_pred = np.zeros_like(test_true)
    lower, upper = cp.predict_interval(test_pred, q_hat)
    coverage = cp.empirical_coverage(test_true, lower, upper)
    # with matched calibration/test distributions, coverage should be
    # reasonably close to the nominal 90% (allow a generous tolerance
    # since this is a single random draw, not an average over many)
    assert 0.80 < coverage < 0.99, f"Coverage {coverage} outside plausible range"
    print(f"PASS: test_conformal_basic_properties (coverage={coverage:.3f})")


if __name__ == "__main__":
    test_physics_closed_form_matches_ode()
    test_nn_gradient_correctness()
    test_end_to_end_small_pipeline()
    test_conformal_basic_properties()
    print("\nAll smoke tests passed.")
