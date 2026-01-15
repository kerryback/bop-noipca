"""
Shared utilities for factor computation scripts.

This module contains common functionality used by run_fama.py, run_dkkm.py,
and other factor computation scripts to reduce code duplication.

Functions:
    parse_panel_arguments: Parse panel_id from command line arguments
    load_model_config: Load appropriate model configuration
    load_panel_data: Load panel data from pickle file
    prepare_panel: Clean and prepare panel data for factor computation
    save_factor_results: Save results to pickle file with summary
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Import configurations
try:
    from config import DATA_DIR
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Back-Ober-Pruitt directory")
    sys.exit(1)


def parse_panel_arguments(script_name: str = "script",
                         additional_args: Optional[Dict[str, str]] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse panel_id and other arguments from command line.

    Args:
        script_name: Name of the calling script (for error messages)
        additional_args: Dictionary of additional arguments to parse
                        e.g., {'nfeatures': 'int'} for type validation

    Returns:
        tuple: (panel_id, model_name, parsed_args_dict)

    Examples:
        >>> panel_id, model_name, args = parse_panel_arguments('run_dkkm', {'nfeatures': 'int'})
        >>> # From: python run_dkkm.py bgn_0 1000
        >>> # Returns: ('bgn_0', 'bgn', {'nfeatures': 1000})
    """
    if len(sys.argv) < 2:
        print(f"ERROR: Panel ID required")
        print(f"\nUsage: python {script_name}.py [panel_id] [additional_args...]")
        print(f"  Example: python {script_name}.py bgn_0")
        print(f"           python {script_name}.py kp14_5")
        sys.exit(1)

    panel_id = sys.argv[1]

    # Extract model from panel_id (e.g., "kp14_0" -> "kp14")
    model_name = panel_id.split('_')[0].lower()

    # Parse additional arguments
    parsed_args = {}
    if additional_args:
        arg_idx = 2
        for arg_name, arg_type in additional_args.items():
            if len(sys.argv) > arg_idx:
                try:
                    if arg_type == 'int':
                        parsed_args[arg_name] = int(sys.argv[arg_idx])
                    elif arg_type == 'float':
                        parsed_args[arg_name] = float(sys.argv[arg_idx])
                    elif arg_type == 'str':
                        parsed_args[arg_name] = sys.argv[arg_idx]
                    else:
                        parsed_args[arg_name] = sys.argv[arg_idx]
                except (ValueError, IndexError) as e:
                    print(f"ERROR: Invalid value for {arg_name}: {sys.argv[arg_idx]}")
                    print(f"Expected type: {arg_type}")
                    sys.exit(1)
                arg_idx += 1

    return panel_id, model_name, parsed_args


def load_model_config(model_name: str):
    """
    Load appropriate model configuration.

    Args:
        model_name: Model identifier ('bgn', 'kp14', or 'gs21')

    Returns:
        Dictionary with model configuration

    Raises:
        SystemExit: If model_name is not recognized
    """
    from config import get_model_config
    return get_model_config(model_name)


def load_panel_data(panel_id: str, model_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load panel data from pickle file in DATA_DIR.

    Args:
        panel_id: Panel identifier (e.g., 'bgn_0', 'kp14_5')
        model_name: Model name for error messages

    Returns:
        tuple: (panel DataFrame, arrays_data dictionary)

    Raises:
        SystemExit: If file not found
    """
    panel_path = os.path.join(DATA_DIR, f"{panel_id}_panel.pkl")

    if not os.path.exists(panel_path):
        print(f"ERROR: Panel file not found at: {panel_path}")
        print(f"\nPlease run generate_panel.py to create the panel file.")
        print(f"\nUsage examples:")
        print(f"  python run_fama.py {panel_id}")
        print(f"  python run_dkkm.py {panel_id} [nfeatures]")
        sys.exit(1)

    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        arrays_data = pickle.load(f)

    panel = arrays_data['panel']
    print(f"Loaded panel: shape={panel.shape}")

    return panel, arrays_data


def prepare_panel(panel: pd.DataFrame, chars: list) -> Tuple[pd.DataFrame, int, int]:
    """
    Clean and prepare panel data for factor computation.

    Applies standard cleaning steps:
    1. Reset index
    2. Add 'size' column (log of mve)
    3. Filter to month >= 2
    4. Replace infinite values with NaN
    5. Set multi-index (month, firmid)
    6. Remove rows with NaN in characteristics or returns

    Args:
        panel: Raw panel DataFrame
        chars: List of characteristic column names

    Returns:
        tuple: (cleaned_panel, start_month, end_month)
    """
    # Reset index and add size
    panel = panel.reset_index()
    panel["size"] = np.log(panel.mve)

    # Filter early months
    panel = panel[panel.month >= 2]

    # Replace infinite values
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Set multi-index
    panel.set_index(["month", "firmid"], inplace=True)

    # Remove NaN rows
    nans = panel[chars + ["mve", "xret"]].isnull().any(axis=1)
    keep = nans[~nans].index
    panel = panel.loc[keep]

    # Get time range
    start = panel.index.unique("month").min()
    end = panel.index.unique("month").max()

    print(f"\nPanel after cleaning:")
    print(f"  Start month: {start}, End month: {end}")
    print(f"  Total observations: {len(panel)}")

    return panel, start, end


def save_factor_results(results: Dict[str, Any],
                        output_file: str,
                        verbose: bool = True) -> None:
    """
    Save factor computation results to pickle file and print summary.

    Args:
        results: Dictionary of results to save
        output_file: Path to output pickle file
        verbose: Whether to print detailed summary
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save as pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f"\n[OK] Results saved to: {output_file}")
        print(f"     Dictionary keys: {list(results.keys())}")
        print(f"\n     Contents:")
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                print(f"       {key}: DataFrame {value.shape}")
            elif isinstance(value, np.ndarray):
                print(f"       {key}: ndarray {value.shape}")
            elif isinstance(value, list):
                print(f"       {key}: list (length {len(value)})")
            elif value is None:
                print(f"       {key}: None")
            else:
                print(f"       {key}: {type(value).__name__} = {value}")


def print_script_header(title: str,
                       model: str,
                       panel_id: str,
                       config,
                       additional_info: Optional[Dict[str, str]] = None) -> None:
    """
    Print standardized script header.

    Args:
        title: Script title (e.g., "FAMA-FRENCH & FAMA-MACBETH FACTORS")
        model: Model name
        panel_id: Panel identifier
        config: Model configuration object
        additional_info: Optional dictionary of additional info to display
    """
    from datetime import datetime

    print("="*70)
    print(title)
    print("="*70)
    print(f"Model: {model}")
    print(f"Panel ID: {panel_id}")
    print(f"Configuration: N={config['N']}, T={config['T']}, n_jobs={config['n_jobs']}")

    if additional_info:
        for key, value in additional_info.items():
            print(f"{key}: {value}")

    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print("="*70)


def print_script_footer(panel_id: str = None,
                       usage_examples: Optional[list] = None) -> None:
    """
    Print standardized script footer with usage examples.

    Args:
        panel_id: Panel identifier for examples
        usage_examples: List of usage example strings
    """
    from datetime import datetime

    print(f"\n{'='*70}")
    print("COMPUTATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

    if usage_examples:
        print(f"\nUsage examples:")
        for example in usage_examples:
            print(f"  {example}")

    print(f"{'='*70}")


def rank_standardize(arr: np.ndarray) -> np.ndarray:
    """
    Rank-based standardization (cross-sectional).

    Maps values to [-0.5, 0.5] based on their rank.
    Uses average ranking for ties (matches pandas .rank() behavior).

    Args:
        arr: (N,) or (N, P) array

    Returns:
        Standardized array with same shape
    """
    from scipy.stats import rankdata

    # Handle both 1D and 2D arrays
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    N, P = arr.shape
    result = np.zeros_like(arr)

    for j in range(P):
        # Rank using average method for ties (1-based, like pandas)
        ranks = rankdata(arr[:, j], method='average')
        # Map to [-0.5, 0.5]
        result[:, j] = (ranks - 0.5) / N - 0.5

    return result.squeeze() if squeeze else result


def standardize_columns(X: np.ndarray, robust: bool = False) -> np.ndarray:
    """
    Standardize columns of matrix.

    Args:
        X: (N, P) matrix
        robust: Use median/MAD instead of mean/std

    Returns:
        Standardized matrix
    """
    if robust:
        # Robust standardization using median and MAD
        medians = np.median(X, axis=0)
        mad = np.median(np.abs(X - medians), axis=0)
        mad[mad < 1e-10] = 1.0  # Avoid division by zero
        return (X - medians) / (1.4826 * mad)  # Scale MAD to match std
    else:
        # Standard standardization
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds < 1e-10] = 1.0  # Avoid division by zero
        return (X - means) / stds
