"""
Run All Tests - Comprehensive Test Suite Runner

This script runs all regression tests organized by model:
For each model (BGN, KP14, GS21):
  1. Panel generation
  2. Moment calculation
  3. Fama factor computation
  4. DKKM factor computation (nfeatures=360)
  5. IPCA factor computation (K=1)

Usage:
    cd tests
    python run_all_tests.py

Requirements:
    - N = 50, T = 400 (configured in config.py)
    - All dependencies installed
    - Sufficient disk space for temporary files
"""

import sys
import os
import subprocess
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import N, T

# Test configuration
DKKM_NFEATURES = 36
IPCA_K = 2


def check_configuration():
    """Check if N and T are set to test values."""
    print("=" * 70)
    print("CONFIGURATION CHECK")
    print("=" * 70)
    print(f"Current configuration:")
    print(f"  N = {N}")
    print(f"  T = {T}")
    print()

    if N != 50 or T != 400:
        print("WARNING: Tests are designed for N=50 and T=400")
        print("Running with different values may cause tests to fail or take much longer.")
        print()
        response = input("Do you want to continue anyway? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nTests cancelled. Please update config.py to set N=50 and T=400.")
            return False
        print()

    return True


def run_test(test_file, args, description):
    """
    Run a single test file with arguments and return success status and error info.

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    print("\n" + "-" * 70)
    print(f"RUNNING: {description}")
    print("-" * 70)

    # Build command
    cmd = [sys.executable, test_file] + args

    # Run test with real-time output
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True
    )

    # Show output during execution
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"[FAIL] {description}")
        # Extract detailed failure information
        full_output = result.stdout + "\n" + result.stderr
        error_summary = extract_failure_details(full_output)
        return False, error_summary
    else:
        print(f"[PASS] {description}")
        return True, None


def extract_failure_details(output):
    """
    Extract comprehensive failure information from test output.

    Looks for:
    - Lines containing [FAIL]
    - Assertion errors
    - Traceback information
    - Error messages

    Returns:
        str: Formatted error summary with all relevant failure details
    """
    if not output:
        return "Test failed with no output"

    lines = output.split('\n')

    # Collect all failure-related lines
    failure_lines = []
    in_traceback = False
    processed_line_indices = set()  # Track which lines we've already added

    for i, line in enumerate(lines):
        # Skip if we've already processed this line
        if i in processed_line_indices:
            continue

        # Check for failure indicators
        if '[FAIL]' in line:
            # Add the failure line and some context
            failure_lines.append(line)
            processed_line_indices.add(i)

            # Look ahead for additional error details
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith('    ['):
                    failure_lines.append(next_line)
                    processed_line_indices.add(j)
                elif '[PASS]' in next_line or next_line.startswith('==='):
                    break

        # Check for traceback
        elif 'Traceback (most recent call last)' in line:
            in_traceback = True
            failure_lines.append('\n--- Traceback ---')
            failure_lines.append(line)
            processed_line_indices.add(i)

        elif in_traceback:
            failure_lines.append(line)
            processed_line_indices.add(i)
            # End of traceback is typically an error line without leading spaces
            # or an empty line followed by non-error content
            if line and not line.startswith(' ') and 'Error:' in line:
                in_traceback = False

        # Check for assertion errors not already captured
        elif 'AssertionError' in line or ('assert' in line.lower() and 'Assertion' in line):
            # Add context around assertion
            start = max(0, i-2)
            end = min(len(lines), i+5)

            # Only add if we haven't processed these lines yet
            if i not in processed_line_indices:
                failure_lines.append('\n--- Assertion Error ---')
                for idx in range(start, end):
                    if idx not in processed_line_indices:
                        failure_lines.append(lines[idx])
                        processed_line_indices.add(idx)

    if failure_lines:
        return '\n'.join(failure_lines)
    else:
        # If no specific failures found, return last 30 lines for more context
        return '\n'.join(lines[-30:]) if len(lines) > 30 else output


def main():
    """Run all tests organized by model."""
    if not check_configuration():
        sys.exit(1)

    print("=" * 70)
    print("STARTING FULL TEST SUITE")
    print("=" * 70)
    print("\nThis will run 5 tests for each of 3 models (15 tests total):")
    print(f"  - Panel generation")
    print(f"  - Moment calculation")
    print(f"  - Fama factor computation")
    print(f"  - DKKM factor computation (nfeatures={DKKM_NFEATURES})")
    print(f"  - IPCA factor computation (K={IPCA_K})")
    print()

    # Track results by model and test type
    # Structure: results[model][test_type] = (success, error_msg)
    results = defaultdict(dict)

    models = ['bgn', 'kp14', 'gs21']
    test_types = ['panel', 'moments', 'fama', 'dkkm', 'ipca']

    # Run tests for each model
    for model in models:
        print("\n" + "=" * 70)
        print(f"TESTING MODEL: {model.upper()}")
        print("=" * 70)

        # 1. Panel generation
        test_file = f'test_panel_{model}.py'
        description = f"{model.upper()} - Panel Generation"
        success, error = run_test(test_file, [], description)
        results[model]['panel'] = (success, error)

        # 2. Moment calculation
        test_file = f'test_moments_{model}.py'
        description = f"{model.upper()} - Moment Calculation"
        success, error = run_test(test_file, [], description)
        results[model]['moments'] = (success, error)

        # 3. Fama factors
        test_file = 'test_fama.py'
        description = f"{model.upper()} - Fama Factors"
        success, error = run_test(test_file, [model], description)
        results[model]['fama'] = (success, error)

        # 4. DKKM factors
        test_file = 'test_dkkm.py'
        description = f"{model.upper()} - DKKM Factors (nfeatures={DKKM_NFEATURES})"
        success, error = run_test(test_file, [model, str(DKKM_NFEATURES)], description)
        results[model]['dkkm'] = (success, error)

        # 5. IPCA factors
        test_file = 'test_ipca.py'
        description = f"{model.upper()} - IPCA Factors (K={IPCA_K})"
        success, error = run_test(test_file, [model, str(IPCA_K)], description)
        results[model]['ipca'] = (success, error)

    # Generate comprehensive summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_tests = len(models) * len(test_types)
    total_passed = 0
    total_failed = 0

    for model in models:
        for test_type in test_types:
            success, _ = results[model][test_type]
            if success:
                total_passed += 1
            else:
                total_failed += 1

    print(f"\nTotal tests run: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print()

    # Summary table
    print("Results by model and test type:")
    print()
    print(f"{'Model':<10} {'Panel':<8} {'Moments':<8} {'Fama':<8} {'DKKM':<8} {'IPCA':<8}")
    print("-" * 58)

    for model in models:
        status_symbols = []
        for test_type in test_types:
            success, _ = results[model][test_type]
            status_symbols.append('PASS' if success else 'FAIL')

        print(f"{model.upper():<10} {status_symbols[0]:<8} {status_symbols[1]:<8} "
              f"{status_symbols[2]:<8} {status_symbols[3]:<8} {status_symbols[4]:<8}")

    # Detailed failure information
    if total_failed > 0:
        print("\n" + "=" * 70)
        print("DETAILED FAILURE INFORMATION")
        print("=" * 70)

        for model in models:
            model_failures = []
            for test_type in test_types:
                success, error = results[model][test_type]
                if not success:
                    model_failures.append((test_type, error))

            if model_failures:
                print(f"\n{model.upper()} Failures:")
                print("-" * 70)
                for test_type, error in model_failures:
                    print(f"\n  Test: {test_type}")
                    if error:
                        # Indent error message
                        indented_error = '\n'.join('    ' + line for line in error.split('\n'))
                        print(f"  Error:\n{indented_error}")
                    else:
                        print(f"  Error: Unknown error (test returned non-zero exit code)")

    print("\n" + "=" * 70)
    if total_failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"TESTS FAILED: {total_failed}/{total_tests}")
        print()
        print("Failed tests by model:")
        for model in models:
            failed_tests = [test_type for test_type in test_types
                          if not results[model][test_type][0]]
            if failed_tests:
                print(f"  {model.upper()}: {', '.join(failed_tests)}")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
