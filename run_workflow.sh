#!/bin/bash
# Entry point script to run different workflows based on WORKFLOW_TYPE env variable

WORKFLOW_TYPE=${WORKFLOW_TYPE:-kp14}

case "$WORKFLOW_TYPE" in
    bgn)
        echo "Starting BGN workflow..."
        python run_bgn.py
        ;;
    kp14)
        echo "Starting KP14 workflow..."
        python run_kp14.py
        ;;
    gs21)
        echo "Starting GS21 workflow..."
        python run_gs21.py
        ;;
    *)
        echo "Unknown WORKFLOW_TYPE: $WORKFLOW_TYPE"
        echo "Valid options: bgn, kp14, gs21"
        exit 1
        ;;
esac
