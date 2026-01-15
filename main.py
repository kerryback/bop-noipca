"""
Back-Ober-Pruitt - Master Orchestration Script

Runs the complete workflow for a range of panel identifiers:
1. Generate panel data
2. Calculate SDF moments
3. Compute Fama factors
4. Compute DKKM factors for multiple feature counts

Usage:
    python main.py [model] [start] [end]

Arguments:
    model: Model name (bgn, kp14, gs21) - case insensitive
    start: Starting index (optional, default: 0)
    end: Ending index (optional, default: 1)

    Runs workflow for panel_id in range(start, end)

Examples:
    python main.py bgn           # Runs for index 0 only
    python main.py bgn 0 5       # Runs for indices 0, 1, 2, 3, 4
    python main.py kp14 10 15    # Runs for indices 10, 11, 12, 13, 14

Output:
    All output is logged to: logs/{model}_{start}_{end}.log
"""

import sys
import os
import subprocess
import time
import pickle
from datetime import datetime
from config import (DATA_DIR, N_DKKM_FEATURES_LIST,
                    KEEP_PANEL, KEEP_MOMENTS, KEEP_FACTOR_DETAILS,
                    N, T, BGN_BURNIN, KP14_BURNIN, GS21_BURNIN)


def run_script(script_name, args, description):
    """Run a Python script and handle errors. Returns elapsed time in seconds."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, script_name] + args
    start_time = time.time()
    # Explicitly pass redirected stdout/stderr so subprocess output goes to log file
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        sys.exit(1)

    return elapsed


def cleanup_factor_file(filepath, stats_key):
    """
    Remove all keys from factor pickle file except stats_key.

    Args:
        filepath: Path to pickle file
        stats_key: Key to keep (e.g., 'fama_stats', 'dkkm_stats', 'ipca_stats')
    """
    if not os.path.exists(filepath):
        return

    # Read full pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Keep only stats key
    if stats_key in data:
        original_size = os.path.getsize(filepath) / (1024**2)  # MB
        cleaned_data = {stats_key: data[stats_key]}

        # Overwrite with cleaned data
        with open(filepath, 'wb') as f:
            pickle.dump(cleaned_data, f)

        new_size = os.path.getsize(filepath) / (1024**2)  # MB
        print(f"[CLEANUP] Reduced {os.path.basename(filepath)}: {original_size:.2f} MB -> {new_size:.2f} MB")


def run_workflow_for_index(model, panel_id):
    """Run the complete workflow for a single panel index."""
    full_panel_id = f"{model}_{panel_id}"

    print(f"\n{'='*70}")
    print(f"RUNNING WORKFLOW FOR {full_panel_id.upper()}")
    print(f"{'='*70}")

    # Track timing for each step
    timings = {}

    # Step 1: Generate panel data
    timings['generate_panel'] = run_script(
        "generate_panel.py",
        [model, str(panel_id)],
        f"STEP 1: Generating {model.upper()} panel data (index={panel_id})"
    )

    # Step 2: Calculate SDF conditional moments
    timings['calculate_moments'] = run_script(
        "calculate_moments.py",
        [full_panel_id],
        "STEP 2: Calculating SDF conditional moments (rp, cond_var, etc.)"
    )

    # Step 3: Compute Fama factors
    timings['run_fama'] = run_script(
        "run_fama.py",
        [full_panel_id],
        "STEP 3: Computing Fama-French and Fama-MacBeth factors"
    )

    # Clean up Fama file if requested
    if not KEEP_FACTOR_DETAILS:
        fama_file = os.path.join(DATA_DIR, f"{full_panel_id}_fama.pkl")
        cleanup_factor_file(fama_file, 'fama_stats')

    # Step 4: Compute DKKM factors for each feature count
    timings['run_dkkm'] = {}
    for i, nfeatures in enumerate(N_DKKM_FEATURES_LIST, 1):
        timings['run_dkkm'][nfeatures] = run_script(
            "run_dkkm.py",
            [full_panel_id, str(nfeatures)],
            f"STEP 4.{i}: Computing DKKM factors (nfeatures={nfeatures})"
        )

        # Clean up DKKM file if requested
        if not KEEP_FACTOR_DETAILS:
            dkkm_file = os.path.join(DATA_DIR, f"{full_panel_id}_dkkm_{nfeatures}.pkl")
            cleanup_factor_file(dkkm_file, 'dkkm_stats')

    # Step 5: Clean up moments and panel files to save disk space
    if not KEEP_MOMENTS:
        moments_file = os.path.join(DATA_DIR, f"{full_panel_id}_moments.pkl")
        if os.path.exists(moments_file):
            file_size = os.path.getsize(moments_file) / (1024**3)  # Size in GB
            os.remove(moments_file)
            print(f"\n[CLEANUP] Deleted moments file ({file_size:.2f} GB): {moments_file}")
    else:
        print(f"\n[KEEP] Keeping moments file (KEEP_MOMENTS=True)")

    if not KEEP_PANEL:
        panel_file = os.path.join(DATA_DIR, f"{full_panel_id}_panel.pkl")
        if os.path.exists(panel_file):
            file_size = os.path.getsize(panel_file) / (1024**3)  # Size in GB
            os.remove(panel_file)
            print(f"[CLEANUP] Deleted panel file ({file_size:.2f} GB): {panel_file}")
    else:
        print(f"[KEEP] Keeping panel file (KEEP_PANEL=True)")

    # Print summary for this index
    total_time = sum([timings['generate_panel'], timings['calculate_moments'], timings['run_fama']] +
                     list(timings['run_dkkm'].values()))

    print(f"\n{'='*70}")
    print(f"WORKFLOW COMPLETE FOR {full_panel_id.upper()}")
    print(f"{'='*70}")
    print(f"Execution Times:")
    print(f"  1. Generate panel:      {timings['generate_panel']:7.1f}s ({timings['generate_panel']/60:5.1f}min)")
    print(f"  2. Calculate moments:   {timings['calculate_moments']:7.1f}s ({timings['calculate_moments']/60:5.1f}min)")
    print(f"  3. Fama factors:        {timings['run_fama']:7.1f}s ({timings['run_fama']/60:5.1f}min)")
    for i, nfeatures in enumerate(N_DKKM_FEATURES_LIST, 1):
        dkkm_time = timings['run_dkkm'][nfeatures]
        print(f"  4.{i} DKKM (n={nfeatures:4d}):      {dkkm_time:7.1f}s ({dkkm_time/60:5.1f}min)")
    print(f"  {'-'*50}")
    print(f"  Total for {full_panel_id}:  {total_time:7.1f}s ({total_time/60:5.1f}min)")
    print(f"{'='*70}\n")

    return timings


def main():
    """Main orchestration function."""
    overall_start = time.time()

    # Print global parameters FIRST so user can cancel if incorrect
    print("="*70)
    print("BACK-OBER-PRUITT - COMPLETE WORKFLOW")
    print("="*70)
    print(f"Global Parameters:")
    print(f"  N (firms):        {N}")
    print(f"  T (periods):      {T}")
    print(f"  BGN burnin:       {BGN_BURNIN}")
    print(f"  KP14 burnin:      {KP14_BURNIN}")
    print(f"  GS21 burnin:      {GS21_BURNIN}")
    print("="*70)
    print()

    # Parse arguments
    if len(sys.argv) < 2:
        print("ERROR: Model name required")
        print("\nUsage: python main.py [model] [start] [end]")
        print("  model: bgn, kp14, or gs21 (case insensitive)")
        print("  start: starting index (optional, default: 0)")
        print("  end: ending index (optional, default: 1)")
        print("\nExamples:")
        print("  python main.py bgn           # Runs for index 0")
        print("  python main.py bgn 0 5       # Runs for indices 0-4")
        print("  python main.py kp14 10 15    # Runs for indices 10-14")
        sys.exit(1)

    # Get model name (case insensitive)
    model = sys.argv[1].lower()
    valid_models = ['bgn', 'kp14', 'gs21']

    if model not in valid_models:
        print(f"ERROR: Invalid model '{sys.argv[1]}'")
        print(f"Valid models: {', '.join(valid_models)} (case insensitive)")
        sys.exit(1)

    # Get start and end indices
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # Validate range
    if start >= end:
        print(f"ERROR: start ({start}) must be less than end ({end})")
        sys.exit(1)

    # Setup logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, f"{model}_{start}_{end}.log")

    # Redirect stdout and stderr to log file
    print(f"Redirecting output to log file: {log_file}")
    print("(All subsequent output will be written to the log file)")
    print()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
    sys.stdout = log_handle
    sys.stderr = log_handle

    try:
        print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
        print()
        print(f"Configuration:")
        print(f"  Model: {model}")
        print(f"  Index range: {start} to {end-1} (inclusive)")
        print(f"  Total runs: {end - start}")
        print(f"  DKKM features: {N_DKKM_FEATURES_LIST}")
        print(f"  Log file: {log_file}")
        print()

        # Run workflow for each index in range
        all_timings = {}
        failed_indices = []
        for i in range(start, end):
            try:
                all_timings[i] = run_workflow_for_index(model, i)
            except Exception as e:
                failed_indices.append(i)

                # Clean up moments and panel files for failed panel (if they exist)
                if not KEEP_MOMENTS:
                    moments_file = os.path.join(DATA_DIR, f"{model}_{i}_moments.pkl")
                    if os.path.exists(moments_file):
                        file_size = os.path.getsize(moments_file) / (1024**3)  # Size in GB
                        os.remove(moments_file)
                        print(f"\n[CLEANUP] Deleted moments file for failed panel ({file_size:.2f} GB): {moments_file}")

                if not KEEP_PANEL:
                    panel_file = os.path.join(DATA_DIR, f"{model}_{i}_panel.pkl")
                    if os.path.exists(panel_file):
                        file_size = os.path.getsize(panel_file) / (1024**3)  # Size in GB
                        os.remove(panel_file)
                        print(f"[CLEANUP] Deleted panel file for failed panel ({file_size:.2f} GB): {panel_file}")

                print(f"\n{'='*70}")
                print(f"ERROR: Workflow failed for {model}_{i}")
                print(f"{'='*70}")
                print(f"Error: {str(e)}")
                print(f"Continuing to next panel...")
                print(f"{'='*70}\n")

        # Final summary
        overall_elapsed = time.time() - overall_start

        print(f"\n{'='*70}")
        print("ALL WORKFLOWS COMPLETE")
        print(f"{'='*70}")
        print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
        print()
        print(f"Summary for {model.upper()} indices {start} to {end-1}:")
        print(f"  Total runs: {end - start}")
        print(f"  Successful: {len(all_timings)}")
        print(f"  Failed: {len(failed_indices)}")
        if failed_indices:
            print(f"  Failed indices: {failed_indices}")
        print(f"  Total time: {overall_elapsed:7.1f}s ({overall_elapsed/60:5.1f}min)")
        if all_timings:
            print(f"  Average time per successful run: {overall_elapsed/len(all_timings):7.1f}s ({overall_elapsed/len(all_timings)/60:5.1f}min)")
        print()
        print(f"Output files created in: {DATA_DIR}")
        for i in range(start, end):
            if i in failed_indices:
                print(f"\n  Index {i} ({model}_{i}): FAILED - No output files")
            else:
                full_panel_id = f"{model}_{i}"
                print(f"\n  Index {i} ({full_panel_id}):")
                panel_status = "kept" if KEEP_PANEL else "deleted after use"
                moments_status = "kept" if KEEP_MOMENTS else "deleted after use"
                print(f"    - Panel data: {full_panel_id}_panel.pkl ({panel_status})")
                print(f"    - SDF moments: {full_panel_id}_moments.pkl ({moments_status})")
                print(f"    - Fama factors: {full_panel_id}_fama.pkl")
                for nfeatures in N_DKKM_FEATURES_LIST:
                    print(f"    - DKKM (n={nfeatures}): {full_panel_id}_dkkm_{nfeatures}.pkl")
        print()
        print(f"Log file: {log_file}")
        print(f"{'='*70}")

    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()

    # Print completion message to console
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Model: {model.upper()}")
    print(f"Indices: {start} to {end-1}")
    print(f"Successful: {end - start - len(failed_indices)}/{end - start}")
    if failed_indices:
        print(f"Failed indices: {failed_indices}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    print(f"Log file: {log_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
