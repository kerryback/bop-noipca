"""
DKKM (Random Fourier Features) factor computation.

Computes DKKM factors with specified number of features.

Usage:
    python run_dkkm.py [panel_id] [nfeatures]

Arguments:
    panel_id: Identifier for panel data (e.g., "bgn_0", "kp14_5")
              Reads from {panel_id}_panel.pkl
    nfeatures: Number of DKKM RFF features (e.g., 6, 36, 360)

Output:
    output/{panel_id}_dkkm_{nfeatures}.pkl

Examples:
    python run_dkkm.py bgn_0 360
    python run_dkkm.py kp14_5 1000
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
import time

# Import refactored modules
try:
    from config import DATA_DIR, GAMMA_GRID as gamma_grid
    from utils_factors import dkkm_functions as dkkm
    from utils_factors import portfolio_stats
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Back-Ober-Pruitt directory:")
    print("  cd Back-Ober-Pruitt")
    print("  python run_dkkm.py")
    sys.exit(1)


def compute_dkkm_factors(panel: pd.DataFrame, start: int, end: int, nfeatures: int, CONFIG, MODEL, CHARS):
    """
    Compute DKKM (Random Fourier Features) factors.

    Args:
        panel: Panel data
        start: Start month
        end: End month
        nfeatures: Number of RFF features (D)
        CONFIG: Model configuration
        MODEL: Model name
        CHARS: List of characteristics

    Returns:
        dkkm_lst: List of (W, f) tuples
    """
    rank_standardize = CONFIG['dkkm_rank_standardize']

    print(f"\n{'-'*70}")
    print(f"Computing DKKM factors...")
    print(f"  Features: {nfeatures}, Matrices: {CONFIG['nmat']}")
    print(f"  Rank standardize: {rank_standardize}")

    t0 = time.time()

    def generate_rff_panel(i):
        """Generate RFF features for one weight matrix."""
        W = np.random.normal(
            size=(int(nfeatures/2),
                  len(CHARS) + (MODEL == 'bgn'))
        )
        gamma = np.random.choice(gamma_grid,
                                size=(int(nfeatures/2), 1))
        W = gamma * W

        print(f"  Weight matrix {i+1}/{CONFIG['nmat']}")

        res_rs, res_nors = dkkm.factors(
            panel=panel, W=W, n_jobs=CONFIG['n_jobs'],
            start=start, end=end, model=MODEL, chars=CHARS
        )

        # Return only the requested standardization method
        res = res_rs if rank_standardize else res_nors
        return W, res

    # Generate DKKM features
    dkkm_lst = [generate_rff_panel(i) for i in range(CONFIG['nmat'])]

    elapsed = time.time() - t0
    print(f"[OK] DKKM factors computed in {elapsed:.1f}s at "
          f"{datetime.now().strftime('%I:%M%p')}")

    return dkkm_lst


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse command-line arguments
    panel_id, model_name, parsed_args = factor_utils.parse_panel_arguments(
        script_name='run_dkkm',
        additional_args={'nfeatures': 'int'}
    )

    # Load model configuration
    CONFIG = factor_utils.load_model_config(model_name)

    # Get nfeatures from command line (required)
    if 'nfeatures' not in parsed_args:
        print("ERROR: nfeatures argument is required")
        print("\nUsage: python run_dkkm.py [panel_id] [nfeatures]")
        print("  Example: python run_dkkm.py bgn_0 360")
        sys.exit(1)

    nfeatures = parsed_args['nfeatures']

    # Model selection
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']

    # Load panel data
    panel, arrays_data = factor_utils.load_panel_data(panel_id, model_name)

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Extract actual N and T from loaded data
    CONFIG['T'] = end - start + 1  # Actual number of periods
    CONFIG['N'] = panel.groupby(level='month').size().max()  # Max firms per month

    # Print header
    factor_utils.print_script_header(
        title="DKKM (RANDOM FOURIER FEATURES) FACTORS",
        model=MODEL,
        panel_id=panel_id,
        config=CONFIG,
        additional_info={'Features': nfeatures}
    )

    # Compute DKKM factors
    dkkm_lst = compute_dkkm_factors(panel, start, end, nfeatures, CONFIG, MODEL, CHARS)

    # Prepare DKKM factors
    dkkm_factors = None
    weights = None

    if len(dkkm_lst) > 0:
        W, f = dkkm_lst[0]
        dkkm_factors = f
        weights = W

    # Compute portfolio statistics
    print(f"\n{'-'*70}")
    print("Computing portfolio statistics...")
    print(f"  Alpha grid: {CONFIG['alpha_lst']}")
    print(f"  Include market: {CONFIG['include_mkt']}")

    t0 = time.time()

    dkkm_stats = None
    if dkkm_factors is not None:
        # Get market returns if needed
        mkt_returns = None
        if CONFIG['include_mkt'] and 'arrays' in arrays_data:
            # Extract market returns from arrays (if available)
            arrays = arrays_data['arrays']
            if 'mkt_rf' in arrays:
                mkt_returns = pd.DataFrame(
                    {'mkt_rf': arrays['mkt_rf']},
                    index=range(start, end + 1)
                )
                mkt_returns.index.name = 'month'

        # Get the weight matrix for the first iteration
        W = dkkm_lst[0][0]

        dkkm_stats = portfolio_stats.compute_dkkm_portfolio_stats(
            dkkm_factors,
            panel,
            panel_id=panel_id,
            model=model_name,
            W=W,
            chars=CHARS,
            start_month=start,
            end_month=end,
            alpha_lst=CONFIG['alpha_lst'],
            include_mkt=CONFIG['include_mkt'],
            mkt_returns=mkt_returns,
            matrix_idx=0,
            burnin=CONFIG['burnin']
        )
        print(f"  [OK] DKKM stats: {len(dkkm_stats)} observations")

    elapsed = time.time() - t0
    print(f"[OK] Portfolio statistics computed in {elapsed:.1f}s")

    # Summary
    print(f"\nFactor Returns Summary:")
    print(f"  DKKM: {len(dkkm_lst)} weight matrices")
    print(f"  Features: {nfeatures}")
    print(f"  Rank standardize: {CONFIG['dkkm_rank_standardize']}")

    # Create results dictionary
    results = {
        'dkkm_factors': dkkm_factors,
        'dkkm_stats': dkkm_stats,
        'weights': weights,
        'nfeatures': nfeatures,
        'nmat': CONFIG['nmat'],
        'rank_standardize': CONFIG['dkkm_rank_standardize'],
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }

    # Save results
    output_file = os.path.join(DATA_DIR, f"{panel_id}_dkkm_{nfeatures}.pkl")
    factor_utils.save_factor_results(results, output_file, verbose=True)

    # Print runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Print footer with usage examples
    factor_utils.print_script_footer(
        panel_id=panel_id,
        usage_examples=[
            "import pickle",
            f"with open('{output_file}', 'rb') as f:",
            "    results = pickle.load(f)",
            "dkkm = results['dkkm_factors']",
        ]
    )


if __name__ == "__main__":
    main()
