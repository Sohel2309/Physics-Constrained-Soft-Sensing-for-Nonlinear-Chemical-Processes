"""
data_pipeline.py
=================
Generates and persists the two fixed datasets every experiment in this
project is built on top of:

  - POOL  : a large set of campaigns (config.POOL_SIZE) from which small
            labelled training subsets are drawn for the low-data
            experiments. The full POOL also serves as the "cheap,
            unlabelled" feature+physics-estimate source for the PGNN's
            physics loss (see nn_model.py) -- using its features and
            physics estimates is free; using its true CB labels beyond
            the sampled training subset is deliberately NOT done, to
            honestly emulate a label-scarce setting.

  - TEST  : a separate, fixed, held-out set (config.TEST_SET_SIZE),
            generated with a different seed, used identically by every
            experiment and NEVER used for training or model selection.

Both are generated once and saved to data/ so that every experiment
script re-loads the exact same numbers -- this is what makes the
reported results reproducible byte-for-byte given the same config.py.
"""

import numpy as np
import pandas as pd

from . import config as cfg
from . import process, features as feat

POOL_SIZE = cfg.POOL_SIZE


def _records_to_dataframe(records):
    rows = []
    for r in records:
        rows.append({
            "D_setpoint": r["D_setpoint"],
            "CAf_setpoint": r["CAf_setpoint"],
            "T_setpoint": r["T_setpoint"],
            "true_CB_final": r["true_CB_final"],
            "true_CA_final": r["true_CA_final"],
            "measured_D": ";".join(f"{v:.6f}" for v in r["measured_D"]),
            "measured_CAf": ";".join(f"{v:.6f}" for v in r["measured_CAf"]),
            "measured_T": ";".join(f"{v:.6f}" for v in r["measured_T"]),
        })
    return pd.DataFrame(rows)


def _dataframe_to_records(df):
    records = []
    for _, row in df.iterrows():
        records.append({
            "D_setpoint": row["D_setpoint"],
            "CAf_setpoint": row["CAf_setpoint"],
            "T_setpoint": row["T_setpoint"],
            "true_CB_final": row["true_CB_final"],
            "true_CA_final": row["true_CA_final"],
            "measured_D": np.array([float(v) for v in row["measured_D"].split(";")]),
            "measured_CAf": np.array([float(v) for v in row["measured_CAf"].split(";")]),
            "measured_T": np.array([float(v) for v in row["measured_T"].split(";")]),
        })
    return records


def generate_and_save(data_dir):
    pool_records = process.generate_dataset(POOL_SIZE, seed=cfg.RANDOM_SEED_DATA)
    test_records = process.generate_dataset(cfg.TEST_SET_SIZE, seed=cfg.RANDOM_SEED_DATA + 1)

    pool_df = _records_to_dataframe(pool_records)
    test_df = _records_to_dataframe(test_records)

    pool_df.to_csv(f"{data_dir}/pool_campaigns.csv", index=False)
    test_df.to_csv(f"{data_dir}/test_campaigns.csv", index=False)

    return pool_records, test_records


def load(data_dir):
    pool_df = pd.read_csv(f"{data_dir}/pool_campaigns.csv")
    test_df = pd.read_csv(f"{data_dir}/test_campaigns.csv")
    pool_records = _dataframe_to_records(pool_df)
    test_records = _dataframe_to_records(test_df)
    return pool_records, test_records


def get_feature_arrays(data_dir):
    """Convenience: load raw records and immediately convert to (X, y, phys)."""
    pool_records, test_records = load(data_dir)
    X_pool, y_pool, phys_pool = feat.records_to_dataset(pool_records)
    X_test, y_test, phys_test = feat.records_to_dataset(test_records)
    return (X_pool, y_pool, phys_pool), (X_test, y_test, phys_test)
