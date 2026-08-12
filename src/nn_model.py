"""
nn_model.py
===========
A small fully-connected neural network implemented from scratch in NumPy
(forward pass, manual backpropagation, Adam optimizer -- no autodiff
framework). It is used for BOTH of the two neural conditions compared in
this project:

  - "Standard MLP"                : lambda = 0   (pure data loss)
  - "Physics-Guided Neural Network (PGNN)" : lambda > 0

Using ONE implementation for both conditions, with identical
architecture/initialisation/optimiser/training budget, is a deliberate
experimental control: the ONLY difference between the two conditions is
the value of lambda in the loss below. This isolates the causal effect
of physics guidance from any confound due to different libraries,
optimisers, or default hyperparameters (which would happen if, e.g., the
"Standard MLP" used scikit-learn's MLPRegressor and the PGNN used a
different custom implementation).

Why this is a NumPy implementation rather than PyTorch/TensorFlow:
this network is trained with a CUSTOM composite loss (see below), and
implementing the forward/backward pass by hand -- rather than relying on
an autodiff framework -- means every gradient in this file can be
derived and defended from first principles in an interview. It also
keeps the project's dependency footprint to numpy/scipy/sklearn/pandas/
matplotlib, all of which install reliably on any laptop with no GPU/CUDA
requirements.

Loss function
-------------
    L_total = L_data + lambda * L_physics

    L_data     = mean squared error between the network's prediction and
                 the TRUE (assayed) CB, computed ONLY on the small labelled
                 training set (size N_train).

    L_physics  = mean squared error between the network's prediction and
                 the physics-based closed-form CB estimate (see
                 physics_model.py), computed over a LARGER pool that
                 includes the labelled points PLUS additional "unlabelled"
                 operating conditions for which no assay result is used --
                 only the (free, always-computable) physics estimate.

    lambda = 0 recovers a standard, purely data-driven MLP exactly (the
    physics term contributes zero gradient), so "Standard MLP" and "PGNN"
    are literally the same class with one hyperparameter set to zero.

Because the physics term does not use automatic differentiation of a
governing differential equation residual with respect to the network's
own input (which is what defines a true Physics-Informed Neural Network,
PINN, in the Raissi/Perdikaris/Karniadakis 2019 sense), and does not
hard-enforce a constraint by construction, this is NOT a PINN. See
BUILD_GUIDE.md, Section 4, for the full terminology discussion and the
reasoning for calling this a Physics-Guided Neural Network (PGNN).
"""

import numpy as np


def _xavier_init(n_in, n_out, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


class PhysicsGuidedMLP:
    """
    Fully-connected network: input -> [hidden layers, tanh] -> linear output.
    Trained by full-batch gradient descent with the Adam optimiser and a
    composite (data + lambda * physics) MSE loss, plus L2 weight decay.
    """

    def __init__(self, n_inputs, hidden_sizes, lam=0.0, learning_rate=0.01,
                 n_epochs=800, l2_reg=1e-4, seed=0):
        self.n_inputs = n_inputs
        self.hidden_sizes = list(hidden_sizes)
        self.lam = float(lam)
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.rng = np.random.default_rng(seed)

        sizes = [n_inputs] + self.hidden_sizes + [1]
        self.n_layers = len(sizes) - 1
        self.W = [_xavier_init(sizes[i], sizes[i + 1], self.rng) for i in range(self.n_layers)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(self.n_layers)]

        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(bb) for bb in self.b]
        self.vb = [np.zeros_like(bb) for bb in self.b]
        self._t = 0

        # Feature standardisation stats (fit during .fit())
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

        self.loss_history = []

    # ---------------------------------------------------------- forward ---
    def _forward(self, Xs):
        """Xs: standardised input, shape (n, n_inputs). Returns (activations, pre-activations)."""
        activations = [Xs]
        pre_acts = []
        a = Xs
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            pre_acts.append(z)
            if i < self.n_layers - 1:
                a = np.tanh(z)
            else:
                a = z  # linear output layer
            activations.append(a)
        return activations, pre_acts

    def _predict_standardised(self, Xs):
        activations, _ = self._forward(Xs)
        return activations[-1].ravel()

    # --------------------------------------------------------- backward ---
    def _backward(self, activations, pre_acts, grad_output):
        """
        grad_output: dL/d(output activation), shape (n, 1) -- already
        includes the standardised-target scaling and the (1/n) averaging
        from whichever loss term(s) it came from.
        Returns gradient lists dW, db (same shapes as self.W, self.b).
        """
        n = grad_output.shape[0]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        delta = grad_output  # dL/dz for the output layer (linear, so dz=da)
        for i in reversed(range(self.n_layers)):
            a_prev = activations[i]
            dW[i] = a_prev.T @ delta + self.l2_reg * self.W[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                da_prev = delta @ self.W[i].T
                z_prev = pre_acts[i - 1]
                delta = da_prev * (1.0 - np.tanh(z_prev) ** 2)  # tanh'(z) = 1 - tanh(z)^2
        return dW, db

    def _adam_step(self, dW, db):
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(self.n_layers):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self._t)
            vW_hat = self.vW[i] / (1 - beta2 ** self._t)
            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self._t)
            vb_hat = self.vb[i] / (1 - beta2 ** self._t)
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # -------------------------------------------------------------- fit ---
    def fit(self, X_labeled, y_labeled, X_extra=None, phys_extra=None, verbose=False):
        """
        X_labeled: (n_lab, n_inputs), y_labeled: (n_lab,)  -- true assayed CB
        X_extra:   (n_extra, n_inputs) additional operating points used ONLY
                   via their physics estimate (no true label used), may
                   overlap in distribution with X_labeled but represents the
                   "cheap, unlabeled" pool. If None, physics loss uses only
                   the labelled points' own physics estimates (phys_extra
                   must then be provided for X_labeled itself).
        phys_extra: (n_extra,) physics-based CB estimate for X_extra.

        If lam == 0, X_extra/phys_extra are ignored entirely (pure data fit).
        """
        X_labeled = np.asarray(X_labeled, dtype=float)
        y_labeled = np.asarray(y_labeled, dtype=float)

        self._x_mean = X_labeled.mean(axis=0)
        self._x_std = X_labeled.std(axis=0)
        self._x_std[self._x_std < 1e-8] = 1.0
        self._y_mean = y_labeled.mean()
        self._y_std = y_labeled.std() if y_labeled.std() > 1e-8 else 1.0

        Xs_lab = (X_labeled - self._x_mean) / self._x_std
        ys_lab = (y_labeled - self._y_mean) / self._y_std
        n_lab = Xs_lab.shape[0]

        use_physics = self.lam > 0.0 and X_extra is not None and phys_extra is not None
        if use_physics:
            X_extra = np.asarray(X_extra, dtype=float)
            phys_extra = np.asarray(phys_extra, dtype=float)
            Xs_phys = (X_extra - self._x_mean) / self._x_std
            phys_s = (phys_extra - self._y_mean) / self._y_std
            n_phys = Xs_phys.shape[0]

        for epoch in range(self.n_epochs):
            # ---- data loss gradient (on labelled set) ----
            activations, pre_acts = self._forward(Xs_lab)
            pred = activations[-1].ravel()
            resid = (pred - ys_lab)
            grad_out_data = (2.0 / n_lab) * resid.reshape(-1, 1)
            dW_data, db_data = self._backward(activations, pre_acts, grad_out_data)

            if use_physics:
                # ---- physics loss gradient (on labelled + extra pool) ----
                act_p, pre_p = self._forward(Xs_phys)
                pred_p = act_p[-1].ravel()
                resid_p = (pred_p - phys_s)
                grad_out_phys = (2.0 / n_phys) * resid_p.reshape(-1, 1)
                dW_phys, db_phys = self._backward(act_p, pre_p, grad_out_phys)

                dW = [dW_data[i] + self.lam * dW_phys[i] for i in range(self.n_layers)]
                db = [db_data[i] + self.lam * db_phys[i] for i in range(self.n_layers)]
            else:
                dW, db = dW_data, db_data

            self._adam_step(dW, db)

            if verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                data_loss = float(np.mean(resid ** 2))
                phys_loss = float(np.mean(resid_p ** 2)) if use_physics else 0.0
                total = data_loss + self.lam * phys_loss
                self.loss_history.append((epoch, data_loss, phys_loss, total))
                if verbose:
                    print(f"epoch {epoch:4d}  data_mse={data_loss:.5f}  "
                          f"phys_mse={phys_loss:.5f}  total={total:.5f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self._x_mean) / self._x_std
        pred_s = self._predict_standardised(Xs)
        return pred_s * self._y_std + self._y_mean
