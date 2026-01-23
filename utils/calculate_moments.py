"""
Calculate and save SDF conditional moments for a panel.

This script computes the expensive SDF outputs (rp, cond_var, second_moment,
second_moment_inv) once and saves them to a pickle file. This avoids redundant
computation across run_fama.py and run_dkkm.py.

Usage:
    python calculate_moments.py [panel_id] [--config CONFIG_MODULE]

Arguments:
    panel_id: Panel identifier (e.g., "bgn_0", "kp14_0", "gs21_5")
              Reads from {panel_id}_panel.pkl
              Output: {panel_id}_moments.pkl
    --config: Optional config module name (default: 'config')

Examples:
    python calculate_moments.py kp14_0
    python calculate_moments.py bgn_5 --config temp_config_xyz
"""

import sys
import os
import numpy as np
import pickle
from datetime import datetime
import time
import gc
from joblib import Parallel, delayed
import importlib

# Add current directory and parent directory to path for imports
# Parent directory is needed for temp_config files created by main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_month_moments(sdf_loop, month):
    """
    Compute SDF moments for a single month.

    Args:
        sdf_loop: SDF computation function
        month: Month number (arrays are indexed 0 to T+burnin-1, matching month numbers)

    Returns:
        Tuple of (month, moments_dict)
    """
    # Call sdf_loop with month-1 to compute statistics at time month
    # (sdf_loop(t) computes returns from time t to t+1, using data at t+1)
    sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1, 0)

    # Compute second moment matrix and its inverse
    rp_vec = rp.reshape(-1, 1)
    second_moment = cond_var + (rp_vec @ rp_vec.T)
    second_moment_inv = np.linalg.inv(second_moment)

    moments_dict = {
        'rp': rp,
        'cond_var': cond_var,
        'second_moment': second_moment,
        'second_moment_inv': second_moment_inv,
        'sdf_ret': sdf_ret,
        'max_sr': max_sr
    }

    return month, moments_dict


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse optional --config argument (defaults to 'config' if not provided)
    config_module_name = 'config'
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        config_module_name = sys.argv[config_idx + 1]
        # Remove --config and its value from sys.argv
        sys.argv.pop(config_idx)
        sys.argv.pop(config_idx)

    # Import config module dynamically and inject into sys.modules
    # This ensures utility modules that do "from config import" get the correct config
    config = importlib.import_module(config_module_name)
    sys.modules['config'] = config

    # Parse command-line argument for panel identifier
    if len(sys.argv) > 1:
        panel_id = sys.argv[1]
        # Extract model from panel_id (e.g., "kp14_0" -> "kp14")
        model_name = panel_id.split('_')[0].lower()
    else:
        print("ERROR: Panel ID required")
        print("\nUsage: python calculate_moments.py [panel_id] [--config CONFIG_MODULE]")
        print("  Example: python calculate_moments.py kp14_0")
        print("           python calculate_moments.py bgn_5")
        sys.exit(1)

    # Validate model
    valid_models = ['bgn', 'kp14', 'gs21']
    if model_name not in valid_models:
        print(f"ERROR: Unknown model '{model_name}' in panel_id '{panel_id}'")
        print(f"Valid models: {', '.join(valid_models)}")
        sys.exit(1)

    print("="*70)
    print("SDF CONDITIONAL MOMENTS CALCULATION")
    print("="*70)
    print(f"Panel ID: {panel_id}")
    print(f"Model: {model_name}")
    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print("="*70)

    # Load panel data
    panel_path = os.path.join(config.TEMP_DIR, f"{panel_id}_panel.pkl")

    if not os.path.exists(panel_path):
        print(f"ERROR: Panel file not found at: {panel_path}")
        print(f"\nPlease run generate_panel.py first to create the panel file.")
        sys.exit(1)

    print(f"\nLoading panel from {panel_path}...")
    with open(panel_path, 'rb') as f:
        arrays_data = pickle.load(f)

    arr_tuple = arrays_data['arr_tuple']
    N = arrays_data['N']
    T = arrays_data['T']

    print(f"Loaded arrays: N={N}, T={T}")

    # Import appropriate SDF compute module and get burnin
    if model_name == 'bgn':
        from utils_bgn import sdf_compute_bgn as sdf_module
        burnin = config.BGN_BURNIN
    elif model_name == 'kp14':
        from utils_kp14 import sdf_compute_kp14 as sdf_module
        burnin = config.KP14_BURNIN
    elif model_name == 'gs21':
        from utils_gs21 import sdf_compute_gs21 as sdf_module
        burnin = config.GS21_BURNIN
    else:
        print(f"ERROR: Unknown model: {model_name}")
        sys.exit(1)

    print(f"\n{'-'*70}")
    print("Computing SDF conditional moments...")
    print(f"{'-'*70}")

    t0 = time.time()

    # Create SDF compute function
    sdf_loop = sdf_module.sdf_compute(N, T + burnin, arr_tuple)

    # Compute SDF outputs for months that will be used in portfolio stats
    # Portfolio stats use months >= burnin + 360 (needs 360 months of history)
    start_month = burnin + 360
    end_month = T + burnin - 1  # Last available month

    n_months = end_month - start_month + 1
    months_list = list(range(start_month, end_month + 1))

    print(f"Computing moments for months {start_month} to {end_month}")
    print(f"  Total: {n_months} months")
    print(f"  Using parallel processing with n_jobs={config.N_JOBS}")

    # Process in chunks to avoid memory exhaustion
    # Chunking is the key optimization - prevents accumulating all 13+ GB of results
    chunk_size = 50  # Process 50 months at a time (reduced from 100 for lower memory usage)
    n_chunks = (n_months + chunk_size - 1) // chunk_size  # Ceiling division
    chunk_files = []

    print(f"  Processing in {n_chunks} chunks of {chunk_size} months each")
    print(f"{'-'*70}")

    for chunk_idx in range(n_chunks):
        chunk_start_idx = chunk_idx * chunk_size
        chunk_end_idx = min((chunk_idx + 1) * chunk_size, n_months)
        chunk_months = months_list[chunk_start_idx:chunk_end_idx]

        print(f"\nChunk {chunk_idx + 1}/{n_chunks}: months {chunk_months[0]} to {chunk_months[-1]} ({len(chunk_months)} months)")

        # Use context manager to ensure workers are cleaned up after each chunk
        # This prevents memory accumulation across chunks
        with Parallel(n_jobs=config.N_JOBS, verbose=5) as parallel:
            chunk_results = parallel(
                delayed(compute_month_moments)(sdf_loop, month)
                for month in chunk_months
            )

        # Convert chunk results to dictionary
        chunk_moments = {month: moments_dict for month, moments_dict in chunk_results}

        # Save chunk to temporary file
        chunk_file = os.path.join(config.TEMP_DIR, f"{panel_id}_moments_chunk{chunk_idx}.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk_moments, f)
        chunk_files.append(chunk_file)

        print(f"  [OK] Chunk {chunk_idx + 1} saved to {os.path.basename(chunk_file)}")

        # Free memory - workers are already terminated by context manager
        del chunk_results, chunk_moments
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'-'*70}")
    print(f"[OK] All chunks computed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    # Consolidate chunks into final moments dictionary
    print(f"\n{'-'*70}")
    print("Consolidating chunks into final moments dictionary...")
    print(f"{'-'*70}")

    moments = {}
    for chunk_idx, chunk_file in enumerate(chunk_files):
        print(f"  Loading chunk {chunk_idx + 1}/{n_chunks}...")
        with open(chunk_file, 'rb') as f:
            chunk_moments = pickle.load(f)
        moments.update(chunk_moments)
        del chunk_moments
        gc.collect()

    # Sort moments by month key
    print(f"  Sorting moments by month...")
    moments = dict(sorted(moments.items()))

    print(f"[OK] Consolidated {len(moments)} months")

    # Save moments to file
    output_file = os.path.join(config.TEMP_DIR, f"{panel_id}_moments.pkl")

    print(f"\nSaving consolidated moments to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'moments': moments,
            'panel_id': panel_id,
            'model': model_name,
            'N': N,
            'T': T,
            'burnin': burnin,
            'start_month': start_month,
            'end_month': end_month
        }, f)

    print(f"[OK] Moments saved successfully")

    # Clean up temporary chunk files
    print(f"\nCleaning up temporary chunk files...")
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
            print(f"  Deleted {os.path.basename(chunk_file)}")
        except Exception as e:
            print(f"  Warning: Could not delete {os.path.basename(chunk_file)}: {e}")

    print(f"[OK] Cleanup complete")

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("COMPUTATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nOutput file: {output_file}")
    print(f"  Months: {start_month} to {end_month} ({len(moments)} total)")
    print(f"  Each month contains:")
    print(f"    - rp: ({N},) expected returns vector")
    print(f"    - cond_var: ({N}, {N}) conditional variance matrix")
    print(f"    - second_moment: ({N}, {N}) second moment matrix")
    print(f"    - second_moment_inv: ({N}, {N}) inverse second moment matrix")
    print(f"    - sdf_ret: scalar SDF return")
    print(f"    - max_sr: scalar maximum Sharpe ratio")
    print(f"\nTo load moments:")
    print(f"  import pickle")
    print(f"  with open('{output_file}', 'rb') as f:")
    print(f"      data = pickle.load(f)")
    print(f"  moments = data['moments']")
    print(f"  # Access specific month:")
    print(f"  rp = moments[{start_month}]['rp']")
    print(f"  cond_var = moments[{start_month}]['cond_var']")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
