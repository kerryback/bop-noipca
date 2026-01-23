"""
SDF Utilities

Shared utilities for loading and working with pre-computed SDF conditional moments.
Used by Fama, DKKM, and IPCA portfolio statistics computation.
"""

import pickle
import os
from typing import Tuple, Dict

from config import TEMP_DIR


def load_precomputed_moments(panel_id: str) -> Tuple[Dict, int, int, int]:
    """
    Load pre-computed SDF conditional moments from pickle file.

    Args:
        panel_id: Panel identifier (e.g., 'kp14_0')

    Returns:
        Tuple of (moments dict, N, start_month, end_month)
        moments dict has structure: {month: {'rp', 'cond_var', 'second_moment', 'second_moment_inv', ...}}
    """
    # Load the moments pickle file
    moments_file = os.path.join(TEMP_DIR, f'{panel_id}_moments.pkl')

    if not os.path.exists(moments_file):
        raise FileNotFoundError(
            f"Moments file not found: {moments_file}\n"
            f"Please run: python calculate_moments.py {panel_id}"
        )

    with open(moments_file, 'rb') as f:
        moments_data = pickle.load(f)

    moments = moments_data['moments']
    N = moments_data['N']
    start_month = moments_data['start_month']
    end_month = moments_data['end_month']

    return moments, N, start_month, end_month
