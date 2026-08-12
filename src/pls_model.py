"""
pls_model.py
============
Thin wrapper around scikit-learn's PLSRegression, with feature
standardisation (PLS is scale-sensitive). This is the classical,
industry-standard linear soft-sensor baseline that any physics-guided
or black-box nonlinear model has to beat to be worth using.

Number of latent components is chosen by leave-one-out-friendly
cross-validation when there is enough data, and capped sensibly for
very small training sets (a PLS model cannot have more components than
min(n_samples-1, n_features)).
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression


class PLSSoftSensor:
    def __init__(self, max_components=8):
        self.max_components = max_components
        self.model = None
        self.n_components_ = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        self._x_mean, self._x_std = X.mean(axis=0), X.std(axis=0)
        self._x_std[self._x_std < 1e-8] = 1.0
        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 1e-8 else 1.0

        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std

        max_possible = max(1, min(self.max_components, d, n - 1))
        if n < 6:
            # too few points to cross-validate component count reliably
            n_comp = min(2, max_possible)
        else:
            # simple k-fold CV over candidate component counts
            from sklearn.model_selection import KFold
            k = min(5, n)
            best_n, best_err = 1, np.inf
            for n_comp in range(1, max_possible + 1):
                kf = KFold(n_splits=k, shuffle=True, random_state=0)
                errs = []
                for tr_idx, val_idx in kf.split(Xs):
                    if len(tr_idx) <= n_comp:
                        continue
                    m = PLSRegression(n_components=n_comp)
                    m.fit(Xs[tr_idx], ys[tr_idx])
                    pred = m.predict(Xs[val_idx]).ravel()
                    errs.append(np.mean((pred - ys[val_idx]) ** 2))
                if errs:
                    mean_err = np.mean(errs)
                    if mean_err < best_err:
                        best_err, best_n = mean_err, n_comp
            n_comp = best_n

        self.n_components_ = n_comp
        self.model = PLSRegression(n_components=n_comp)
        self.model.fit(Xs, ys)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self._x_mean) / self._x_std
        pred_s = self.model.predict(Xs).ravel()
        return pred_s * self._y_std + self._y_mean
