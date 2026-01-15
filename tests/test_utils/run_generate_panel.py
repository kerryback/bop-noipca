"""
Wrapper to run panel generation with a fixed seed for testing.

This script calls the panel generation functions directly with TEST_SEED
to ensure reproducibility for regression testing.

Usage:
    python run_generate_panel.py <model> <identifier>

Examples:
    python run_generate_panel.py bgn 0
    python run_generate_panel.py kp14 999
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add root directory to path (two levels up: test_utils -> tests -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Fixed seed for reproducibility in testing
TEST_SEED = 12345


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_generate_panel.py <model> <identifier>")
        print("  Models: bgn, kp14, gs21")
        print("  Example: python run_generate_panel.py bgn 0")
        sys.exit(1)

    model = sys.argv[1].lower()
    identifier = int(sys.argv[2])

    # Set seed for reproducibility
    np.random.seed(TEST_SEED)

    # Import and run generate_panel.main()
    # This will use the seed we just set
    sys.argv = ['generate_panel.py', model, str(identifier)]

    # Change to root directory so generate_panel.py runs from correct location
    os.chdir(str(Path(__file__).parent.parent.parent))

    # Import and run
    from generate_panel import main as generate_main
    generate_main()


if __name__ == "__main__":
    main()
