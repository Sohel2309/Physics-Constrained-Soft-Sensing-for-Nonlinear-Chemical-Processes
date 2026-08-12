"""
generate_data.py
=================
Generates the fixed POOL and TEST datasets (see src/data_pipeline.py)
and saves them to data/. Run this once before any experiment script.

Usage:
    python scripts/generate_data.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_pipeline as dp

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    pool_records, test_records = dp.generate_and_save(DATA_DIR)
    print(f"Generated and saved:")
    print(f"  {len(pool_records)} pool campaigns  -> data/pool_campaigns.csv")
    print(f"  {len(test_records)} test campaigns   -> data/test_campaigns.csv")
