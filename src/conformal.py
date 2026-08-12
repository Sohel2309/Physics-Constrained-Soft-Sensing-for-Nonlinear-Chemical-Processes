"""
conformal.py
============
Split conformal prediction (Papadopoulos 2002; Lei et al. 2018;
Vovk/Gammerman/Shafer's conformal prediction framework generally).

Given a trained point-prediction model, a held-out CALIBRATION set (not
used for training), and a target miscoverage level alpha:

  1. Compute nonconformity scores on the calibration set:
         s_i = | y_i - pred(x_i) |
  2. Take q_hat = the finite-sample-corrected (1 - alpha) empirical
     quantile of {s_i}:
         q_hat = quantile({s_i}, ceil((n_cal + 1) * (1 - alpha)) / n_cal)
  3. For a new point x, the prediction interval is
         [ pred(x) - q_hat , pred(x) + q_hat ]

Why this gives a real, honest guarantee here:
--------------------------------------------
Split conformal's marginal coverage guarantee
    P( y_test in interval ) >= 1 - alpha
relies on EXCHANGEABILITY of the calibration and test points (a slightly
weaker condition than i.i.d., but i.i.d. implies it). In this project,
every campaign is generated as an INDEPENDENT draw of a random operating
setpoint (see process.sample_setpoint), so calibration and test points
ARE i.i.d. -- exchangeability holds, and the standard finite-sample
marginal coverage guarantee is legitimately applicable here.

This would NOT automatically hold if calibration/test data came from a
single evolving time series (sequential campaigns on the same physical
reactor, where conditions drift over time and consecutive points are
correlated) -- that is explicitly flagged as a limitation in
BUILD_GUIDE.md rather than glossed over.

The guarantee is also MARGINAL (averaged over the randomness of the
calibration set and test point), not conditional on X -- i.e. it does not
promise 90% coverage within every local region of operating-condition
space, only on average across the whole operating envelope tested here.
"""

import numpy as np


def calibrate(residual_abs_calibration, alpha):
    """
    residual_abs_calibration: array of |y - pred| on the calibration set.
    alpha: miscoverage level (e.g. 0.1 for a 90% interval).
    Returns q_hat (half-width of the symmetric interval).
    """
    s = np.sort(np.asarray(residual_abs_calibration, dtype=float))
    n = len(s)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = np.quantile(s, level, method="higher")
    return float(q_hat)


def predict_interval(point_predictions, q_hat):
    point_predictions = np.asarray(point_predictions, dtype=float)
    lower = point_predictions - q_hat
    upper = point_predictions + q_hat
    return lower, upper


def empirical_coverage(y_true, lower, upper):
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mean_interval_width(lower, upper):
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
