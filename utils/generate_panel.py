"""
Generate and save panel data for BGN, KP14, or GS21 models.

This script orchestrates panel generation across all three supported models,
extracting common workflow logic while delegating model-specific operations
to the appropriate panel_functions module.

Usage:
    python generate_panel.py <model> [identifier] [--config CONFIG_MODULE]

Arguments:
    model: Model name ('bgn', 'kp14', or 'gs21')
    identifier: Optional integer identifier (default: 0)
    --config: Optional config module name (default: 'config')

Examples:
    python generate_panel.py bgn 0
    python generate_panel.py kp14 5 --config temp_config_xyz
    python generate_panel.py gs21 3

Output:
    Creates {model}_{identifier}_arrays.pkl in the data directory
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
import importlib

# Add current directory and parent directory to path for imports
# Parent directory is needed for temp_config files created by main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("ERROR: Model name required")
        print("\nUsage: python generate_panel.py <model> [identifier] [--config CONFIG_MODULE]")
        print("  Models: bgn, kp14, gs21")
        print("  Example: python generate_panel.py bgn 0")
        print("           python generate_panel.py kp14 5")
        sys.exit(1)

    model_name = sys.argv[1].lower()

    # Parse optional identifier
    if len(sys.argv) > 2:
        identifier = int(sys.argv[2])
    else:
        identifier = 0

    # Validate model
    valid_models = ['bgn', 'kp14', 'gs21']
    if model_name not in valid_models:
        print(f"ERROR: Unknown model '{model_name}'")
        print(f"Valid models: {', '.join(valid_models)}")
        sys.exit(1)

    print("="*70)
    print("PANEL GENERATION")
    print("="*70)
    print(f"Model: {model_name.upper()}")
    print(f"Identifier: {identifier}")
    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print("="*70)

    # Import appropriate panel generation module
    if model_name == 'bgn':
        from utils_bgn import panel_functions_bgn as panel_module
        burnin = config.BGN_BURNIN
    elif model_name == 'kp14':
        from utils_kp14 import panel_functions_kp14 as panel_module
        burnin = config.KP14_BURNIN
    elif model_name == 'gs21':
        from utils_gs21 import panel_functions_gs21 as panel_module
        burnin = config.GS21_BURNIN
    else:
        print(f"ERROR: Unknown model: {model_name}")
        sys.exit(1)

    # Get model configuration
    model_config = config.get_model_config(model_name)

    # Parameters for panel generation
    N = config.N
    T = config.T

    chars = model_config['chars']  # Characteristics defined in config

    print(f"\nParameters:")
    print(f"  N (firms): {N}")
    print(f"  T (periods): {T}")
    print(f"  Burnin: {burnin}")
    print(f"  Characteristics: {chars}")

    print(f"\n{'-'*70}")
    print(f"Generating panel for {model_name.upper()} model...")
    print(f"{'-'*70}")

    # Create arrays and panel
    print("Creating arrays...")
    arr_tuple = panel_module.create_arrays(N, T + burnin)

    print("Creating panel...")
    panel = panel_module.create_panel(N, T + burnin, arr_tuple)

    # Add size characteristic (if mve column exists)
    if 'mve' in panel.columns:
        panel["size"] = np.log(panel.mve)
    else:
        print("Warning: 'mve' column not found, skipping size calculation")

    # Count firm-months with zero book-to-market (after burnin period)
    if 'bm' in panel.columns:
        # Filter for months >= burnin (month is a column, not index level)
        panel_post_burnin = panel[panel['month'] >= burnin]
        zero_bm_count = (panel_post_burnin['bm'] == 0).sum()
        print(f"\nFirm-months with bm=0 (after burnin): {zero_bm_count:,}")
    else:
        zero_bm_count = None
        print("\nWarning: 'bm' column not found, cannot count zero book-to-market")

    # Save panel and arrays to temporary directory
    panel_filename = os.path.join(config.TEMP_DIR, f'{model_name}_{identifier}_panel.pkl')

    print(f"\n{'-'*70}")
    print(f"Saving panel and arrays to {panel_filename}...")
    print(f"{'-'*70}")

    with open(panel_filename, 'wb') as f:
        pickle.dump({
            'panel': panel,
            'arr_tuple': arr_tuple,
            'N': N,
            'T': T,
            'model': model_name,
            'chars': chars,
            'identifier': identifier,
            'zero_bm_count': zero_bm_count,
            'burnin': burnin
        }, f)

    print(f"[OK] Saved successfully")

    # Print summary statistics
    print(f"\n{'-'*70}")
    print(f"PANEL SUMMARY")
    print(f"{'-'*70}")
    print(f"Model: {model_name.upper()} (identifier={identifier})")
    print(f"  Shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)}")
    print(f"  Months: {panel.month.min()} to {panel.month.max()}")
    print(f"  Unique months: {panel.month.nunique()}")
    print(f"  Firms: {panel.firmid.nunique()}")
    print(f"  Characteristics: {chars}")

    if 'xret' in panel.columns:
        print(f"  Mean excess return: {panel.xret.mean():.4f}")
        print(f"  Std excess return: {panel.xret.std():.4f}")

    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print(f"Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"\nOutput file: {panel_filename}")
    print(f"\nTo compute SDF moments:")
    print(f"  python calculate_moments.py {model_name}_{identifier}")
    print(f"\nTo extract factors:")
    print(f"  python run_fama.py {model_name}_{identifier}")
    print(f"  python run_dkkm.py {model_name}_{identifier} 1000")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
