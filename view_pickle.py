"""
View contents of pickle files.

Displays the structure, info, and preview of data in pickle files.

Usage:
    python view_pickle.py <pickle_file>
    python view_pickle.py outputs/bgn_0_panel.pkl
    python view_pickle.py outputs/bgn_0_moments.pkl
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def print_header(text, char='='):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"{text}")
    print(f"{char * 70}")


def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n{'-' * 70}")
    print(f"{text}")
    print(f"{'-' * 70}")


def view_array(arr, name, indent=""):
    """View numpy array details."""
    print(f"{indent}Type: {type(arr).__name__}")
    print(f"{indent}Shape: {arr.shape}")
    print(f"{indent}Dtype: {arr.dtype}")

    # Check for NaNs and Infs
    if np.issubdtype(arr.dtype, np.floating):
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            print(f"{indent}NaN count: {nan_count} ({100*nan_count/arr.size:.2f}%)")
        if inf_count > 0:
            print(f"{indent}Inf count: {inf_count} ({100*inf_count/arr.size:.2f}%)")

    # Statistics
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        try:
            print(f"{indent}Min: {np.nanmin(arr):.6f}")
            print(f"{indent}Max: {np.nanmax(arr):.6f}")
            print(f"{indent}Mean: {np.nanmean(arr):.6f}")
        except:
            pass

    # Preview
    if arr.size > 0:
        print(f"{indent}Preview (first 5):")
        if arr.ndim == 1:
            print(f"{indent}  {arr[:5]}")
        elif arr.ndim == 2:
            print(f"{indent}  {arr[:5, :min(5, arr.shape[1])]}")
        else:
            print(f"{indent}  (Multi-dimensional array, shape: {arr.shape})")


def view_dataframe(df, name, indent=""):
    """View pandas DataFrame details."""
    print(f"{indent}Type: DataFrame")
    print(f"{indent}Shape: {df.shape}")
    print()

    # Info
    print(f"{indent}DataFrame Info:")
    df.info()
    print()

    # Head
    print(f"{indent}Head (first 5 rows):")
    print(df.head())
    print()

    # Tail
    print(f"{indent}Tail (last 5 rows):")
    print(df.tail())
    print()

    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"{indent}Columns with NaNs:")
        print(nan_counts[nan_counts > 0])


def view_series(series, name, indent=""):
    """View pandas Series details."""
    print(f"{indent}Type: Series")
    print(f"{indent}Length: {len(series)}")
    print(f"{indent}Dtype: {series.dtype}")
    print()

    # Head
    print(f"{indent}Head (first 5):")
    print(series.head())
    print()

    # Tail
    print(f"{indent}Tail (last 5):")
    print(series.tail())
    print()

    # Statistics
    if np.issubdtype(series.dtype, np.number):
        print(f"{indent}Statistics:")
        print(series.describe())


def view_value(value, name, indent=""):
    """View a single value."""
    print(f"{indent}Type: {type(value).__name__}")
    print(f"{indent}Value: {value}")


def view_dict(data, name="Dictionary", indent=""):
    """View dictionary contents recursively."""
    print(f"{indent}Type: dict")
    print(f"{indent}Keys: {list(data.keys())}")
    print()

    for key, value in data.items():
        print_subheader(f"Key: {key}")
        view_object(value, str(key), indent + "  ")


def view_list_or_tuple(data, name, indent=""):
    """View list or tuple contents."""
    print(f"{indent}Type: {type(data).__name__}")
    print(f"{indent}Length: {len(data)}")
    print()

    for i, item in enumerate(data):
        print_subheader(f"Index {i}")
        view_object(item, f"[{i}]", indent + "  ")
        if i >= 9:  # Limit to first 10 items
            print(f"{indent}... ({len(data) - 10} more items)")
            break


def view_object(obj, name="Object", indent=""):
    """Route to appropriate viewer based on object type."""
    if isinstance(obj, pd.DataFrame):
        view_dataframe(obj, name, indent)
    elif isinstance(obj, pd.Series):
        view_series(obj, name, indent)
    elif isinstance(obj, np.ndarray):
        view_array(obj, name, indent)
    elif isinstance(obj, dict):
        view_dict(obj, name, indent)
    elif isinstance(obj, (list, tuple)):
        view_list_or_tuple(obj, name, indent)
    else:
        view_value(obj, name, indent)


def view_pickle_file(filepath):
    """Main function to view pickle file contents."""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return

    print_header(f"PICKLE FILE VIEWER: {filepath.name}")
    print(f"File path: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")

    # Load pickle file
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"\nERROR loading pickle file: {e}")
        return

    print_header("CONTENTS")
    view_object(data, filepath.stem)

    print_header("END OF FILE")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("ERROR: Pickle file path required")
        print()
        print("Usage: python view_pickle.py <pickle_file>")
        print()
        print("Examples:")
        print("  python view_pickle.py outputs/bgn_0_panel.pkl")
        print("  python view_pickle.py outputs/kp14_0_moments.pkl")
        print("  python view_pickle.py outputs/bgn_0_fama.pkl")
        print("  python view_pickle.py outputs/bgn_0_dkkm_360.pkl")
        sys.exit(1)

    pickle_file = sys.argv[1]
    view_pickle_file(pickle_file)


if __name__ == "__main__":
    main()
