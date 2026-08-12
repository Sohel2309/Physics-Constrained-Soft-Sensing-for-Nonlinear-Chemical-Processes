"""
config.py
=========
Single source of truth for every physical, simulation, and experiment
constant used in the project. Keeping every number here (instead of
scattered through the code) makes the project auditable: anyone
reviewing the project can see exactly what was assumed, and change
one number here to see how it propagates.

All chemistry: the Van de Vusse CSTR reaction scheme
    A --k1--> B --k2--> C      (series, first order)
    2A --k3--> D                (parallel, second order in A)

State variables: CA (mol/L), CB (mol/L) -- C and D are not tracked,
consistent with the classic Van de Vusse formulation, which only
follows A and B.

Governing ODEs (constant-volume, well-mixed CSTR, dilution rate
D_rate = F/V):
    dCA/dt = D_rate*(CAf - CA) - k1*CA - k3*CA**2
    dCB/dt = -D_rate*CB + k1*CA - k2*CB

Rate constants follow the Arrhenius law relative to a reference
temperature T_REF (this form avoids the usual awkward pre-exponential
factor and is numerically identical to the standard Arrhenius law):
    k_i(T) = k_i_ref * exp( -(Ea_i/R) * (1/T - 1/T_REF) )
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol*K)
T_REF = 350.0  # K, reference temperature for the Arrhenius parameterisation

# ---------------------------------------------------------------------------
# TRUE kinetic parameters (used only by the data-generating simulator --
# these represent the "real plant", which is never directly known to the
# physics model or to any of the ML models).
#
# Magnitudes (k1 ~ 50 /h, k2 ~ 100 /h, k3 ~ 10 L/mol/h, converted to
# per-minute units) are chosen to be representative of the well-known
# Van de Vusse benchmark used throughout the nonlinear process-control
# and soft-sensor literature -- they are illustrative, standard-order-of-
# magnitude values, not a citation of one specific paper's exact numbers.
# ---------------------------------------------------------------------------
TRUE_K1_REF = 0.8333   # 1/min   at T_REF
TRUE_K2_REF = 1.6667   # 1/min   at T_REF
TRUE_K3_REF = 0.16667  # L/(mol*min) at T_REF

TRUE_EA1 = 50_000.0  # J/mol
TRUE_EA2 = 55_000.0  # J/mol
TRUE_EA3 = 60_000.0  # J/mol

# ---------------------------------------------------------------------------
# ASSUMED kinetic parameters (used ONLY by the physics-based estimator).
# These are deliberately offset from the TRUE values by a fixed,
# unknown-to-the-model bias, representing realistic uncertainty in
# lab-estimated/literature kinetic parameters (typical Arrhenius-fit
# parameter uncertainty is commonly in the 5-15% range for liquid-phase
# reaction systems). This bias is FIXED (not resampled per run) because
# it represents a persistent gap between the plant and the design-basis
# kinetics, not random noise.
# ---------------------------------------------------------------------------
ASSUMED_K1_REF = TRUE_K1_REF * 1.08   # +8% bias
ASSUMED_K2_REF = TRUE_K2_REF * 0.94   # -6% bias
ASSUMED_K3_REF = TRUE_K3_REF * 1.12   # +12% bias

ASSUMED_EA1 = TRUE_EA1 * 1.04   # +4% bias
ASSUMED_EA2 = TRUE_EA2 * 0.97   # -3% bias
ASSUMED_EA3 = TRUE_EA3 * 1.06   # +6% bias

# ---------------------------------------------------------------------------
# Operating envelope (the ranges from which per-sample setpoints are drawn).
# Verified (see scripts/verify_physics_model.py) to keep the process in a
# well-behaved, non-degenerate region that includes the characteristic
# non-monotonic Van de Vusse yield curve (CB has an interior maximum in D).
# ---------------------------------------------------------------------------
D_RANGE = (0.15, 1.60)    # dilution rate, 1/min
CAF_RANGE = (8.0, 12.0)   # feed concentration, mol/L
T_RANGE = (330.0, 365.0)  # reactor temperature, K

# ---------------------------------------------------------------------------
# Process noise (fluctuating operating conditions during a "campaign")
# and measurement noise (sensor noise on top of the true fluctuating
# signal). Both are additive Gaussian, expressed as standard deviations.
# ---------------------------------------------------------------------------
CAMPAIGN_DURATION = 60.0      # minutes, length of one operating campaign
N_NOISE_STEPS = 30            # number of piecewise-constant noise segments per campaign

PROCESS_NOISE_STD = {
    "D_rate": 0.04,   # 1/min
    "CAf": 0.30,      # mol/L
    "T": 2.5,         # K
}

SENSOR_NOISE_STD = {
    "D_rate": 0.01,   # 1/min
    "CAf": 0.10,      # mol/L
    "T": 0.5,          # K
}

# Initial condition for each campaign's ODE integration (reactor assumed
# to start near the feed concentration with no product, i.e. a freshly
# started or purged reactor -- a conservative, physically reasonable choice)
CA0_INIT_FRACTION = 1.0   # CA(0) = CA0_INIT_FRACTION * CAf_setpoint
CB0_INIT = 0.0

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_SEED_DATA = 20260812   # seed for the master dataset generation
POOL_SIZE = 500                # size of the pool from which small training subsets are drawn
TEST_SET_SIZE = 400           # held-out test set, fixed across all experiments
N_SEEDS = 12                  # number of repeats (different training subsets) per condition

TRAIN_SIZES = [8, 15, 25, 40, 65, 100, 160, 250]

LAMBDA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]

# Neural network architecture shared by the "Standard MLP" (lambda=0) and
# the "Physics-Guided Neural Network" (lambda>0) -- identical architecture,
# identical optimiser, identical training budget. The ONLY difference
# between the two conditions is the value of lambda in the loss function.
NN_HIDDEN_SIZES = [16, 16]
NN_LEARNING_RATE = 0.01
NN_N_EPOCHS = 800
NN_L2_REG = 1e-4

# Conformal prediction
CONFORMAL_ALPHA_LEVELS = [0.20, 0.10, 0.05]  # -> 80%, 90%, 95% intervals
CONFORMAL_CALIBRATION_FRACTION = 0.3

FEATURE_NAMES = [
    "D_mean", "CAf_mean", "T_mean",
    "D_std", "CAf_std", "T_std",
    "invT_mean", "D_sq_mean", "CAf_D_mean", "D_T_mean",
]
