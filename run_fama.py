"""
Fama-French and Fama-MacBeth factor computation.

Computes only:
1. Fama-French factors (characteristic-sorted portfolios)
2. Fama-MacBeth factors (two-stage cross-sectional regression)

Usage:
    python run_fama.py [panel_id]

Arguments:
    panel_id: Identifier for panel data (e.g., "bgn_0", "kp14_5")
              Reads from {panel_id}_panel.pkl
              Output: output/{panel_id}_fama.pkl

Examples:
    python run_fama.py bgn_0
    python run_fama.py kp14_5
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
    from config import LOADING_KEYS, FACTOR_KEYS, DATA_DIR
    from utils_factors import fama_functions as fama
    from utils_factors import portfolio_stats
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Back-Ober-Pruitt directory:")
    print("  cd Back-Ober-Pruitt")
    print("  python run_fama.py")
    sys.exit(1)


def compute_fama_factors(panel: pd.DataFrame, start: int, end: int, CONFIG, CHARS):
    """Compute Fama-French and Fama-MacBeth factors."""
    print(f"\n{'-'*70}")
    print("Computing Fama-French and Fama-MacBeth factors...")

    t0 = time.time()

    ff_rets = fama.factors(fama.fama_french, panel,
                           n_jobs=CONFIG['n_jobs'], start=start, end=end, chars=CHARS)
    fm_rets = fama.factors(fama.fama_macbeth, panel,
                           n_jobs=CONFIG['n_jobs'], start=start, end=end, chars=CHARS,
                           stdz_fm=CONFIG['stdz_fm'])

    elapsed = time.time() - t0
    print(f"[OK] Fama factors computed in {elapsed:.1f}s at "
          f"{datetime.now().strftime('%I:%M%p')}")

    return ff_rets, fm_rets


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse command-line arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(script_name='run_fama')

    # Load model configuration
    CONFIG = factor_utils.load_model_config(model_name)

    # Model selection
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    NAMES = CONFIG['factor_names']

    # Load panel data
    panel, arrays_data = factor_utils.load_panel_data(panel_id, model_name)

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Extract actual N and T from loaded data
    CONFIG['T'] = end - start + 1  # Actual number of periods
    CONFIG['N'] = panel.groupby(level='month').size().max()  # Max firms per month

    # Print header
    factor_utils.print_script_header(
        title="FAMA-FRENCH & FAMA-MACBETH FACTORS",
        model=MODEL,
        panel_id=panel_id,
        config=CONFIG,
        additional_info={'Methods': 'Fama-French, Fama-MacBeth'}
    )

    # Compute Fama factors
    ff_rets, fm_rets = compute_fama_factors(panel, start, end, CONFIG, CHARS)

    # Extract model factors from arr_tuple
    print(f"\n{'-'*70}")
    print("Extracting model factors from array data...")

    model_premia = {}

    # Model factors are computed from arr_tuple (not stored in panel to save memory)
    # arr_tuple contains: r, mu, xi, sigmaj, chi, beta, corr_zr, eret, ret, P, corr_zr, ...
    if 'arr_tuple' in arrays_data and MODEL.lower() == 'bgn':
        arr_tuple = arrays_data['arr_tuple']

        # Extract needed arrays: r[0], mu[1], xi[2], corr_zr[10]
        r_arr = arr_tuple[0]
        mu = arr_tuple[1]
        xi = arr_tuple[2]
        corr_zr = arr_tuple[10]

        # Import sigma_z from BGN vasicek module
        from utils_bgn.vasicek import sigma_z

        # Compute factor returns using original formula
        # f_mu_true = sigma_z * sqrt(1 - corr_zr^2) + mu
        # f_xi_true = sigma_z * corr_zr + xi
        f_mu_true = sigma_z * np.sqrt(1 - corr_zr**2) + mu[1:]  # mu[1:] to match r[1:-1]
        f_xi_true = sigma_z * corr_zr + xi[1:]

        # Create DataFrame with factor returns
        # Note: months go from 1 to T (matching panel.month values after burnin removal)
        T = len(f_mu_true)
        month_range = range(1, T + 1)

        factor_data = pd.DataFrame({
            'mu': f_mu_true,
            'xi': f_xi_true
        }, index=month_range)
        factor_data.index.name = 'month'

        # Filter to current date range (after burnin removal in prepare_panel)
        factor_data = factor_data.loc[start:end]

        # Both Taylor and Projection use the same true factors
        model_premia['taylor'] = factor_data.copy()
        model_premia['proj'] = factor_data.copy()

        print(f"  [OK] Taylor factors: {model_premia['taylor'].shape}")
        print(f"  [OK] Projection factors: {model_premia['proj'].shape}")
    else:
        print(f"  [INFO] No model factors computed (only available for BGN model)")

    # Compute portfolio statistics
    print(f"\n{'-'*70}")
    print("Computing portfolio statistics...")
    print(f"  Alpha grids: Fama={CONFIG['alpha_lst_fama']}")

    t0 = time.time()

    # Model factors statistics (if available)
    model_stats = None
    if model_premia:
        model_stats = portfolio_stats.compute_model_portfolio_stats(
            model_premia, panel, start, end
        )
        print(f"  [OK] Model stats: {len(model_stats)} observations")

    # Fama factors statistics
    fama_stats = portfolio_stats.compute_fama_portfolio_stats(
        ff_rets, fm_rets, panel,
        panel_id=panel_id,
        model=model_name,
        chars=CHARS,
        start_month=start,
        end_month=end,
        alpha_lst=CONFIG['alpha_lst_fama'],
        burnin=CONFIG['burnin']
    )
    print(f"  [OK] Fama stats: {len(fama_stats)} observations")

    elapsed = time.time() - t0
    print(f"[OK] Portfolio statistics computed in {elapsed:.1f}s")

    # Summary
    print(f"\nFactor Returns Summary:")
    print(f"  FF returns shape: {ff_rets.shape}")
    print(f"  FM returns shape: {fm_rets.shape}")
    if model_premia:
        for method, rets in model_premia.items():
            print(f"  Model {method} returns shape: {rets.shape}")

    # Create results dictionary
    results = {
        'ff_returns': ff_rets,
        'fm_returns': fm_rets,
        'fama_stats': fama_stats,
        'model_premia_taylor': model_premia.get('taylor'),
        'model_premia_proj': model_premia.get('proj'),
        'model_stats': model_stats,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }

    # Save results
    output_file = os.path.join(DATA_DIR, f"{panel_id}_fama.pkl")
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
            "ff = results['ff_returns']",
            "fm = results['fm_returns']",
        ]
    )


if __name__ == "__main__":
    main()
