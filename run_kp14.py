"""Wrapper to run kp14 workflow with periodic progress monitoring."""
import sys
import subprocess
import time
import os
from datetime import datetime

print("="*70)
print("Starting KP14 workflow: indices 0-9")
print("="*70)
print()

# Run the main workflow
cmd = [sys.executable, "main.py", "kp14", "0", "10"]
print(f"Executing: {' '.join(cmd)}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

log_file = "/app/logs/kp14_0_10.log"
last_position = 0
last_update = time.time()

# Start the workflow and stream output in real-time
print("Starting workflow...")
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print(f"Process started (PID: {process.pid})")
print("="*70)
print()

try:
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())

    # Wait for process to complete
    return_code = process.wait()

    print("\n" + "="*70)
    print(f"WORKFLOW COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {return_code}")
    print("="*70)

    if return_code == 0:
        print("\n✓ WORKFLOW COMPLETED SUCCESSFULLY")
    else:
        print(f"\n✗ WORKFLOW FAILED WITH EXIT CODE: {return_code}")

except KeyboardInterrupt:
    print("\n\nInterrupted! Terminating workflow...")
    process.terminate()
    process.wait(timeout=10)
    print("Workflow terminated.")

except Exception as e:
    print("\n" + "="*70)
    print(f"ERROR: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()

    # Try to terminate process
    try:
        process.terminate()
        process.wait(timeout=10)
    except:
        pass

# Keep container alive after workflow completes
print("\n" + "="*70)
print("Workflow finished. Keeping container alive...")
print(f"Full logs available at: {log_file}")
print("="*70)
print()

# Sleep forever to keep container running
while True:
    time.sleep(3600)
