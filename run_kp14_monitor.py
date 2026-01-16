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

# Start the workflow in background
print("Starting workflow in background...")
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print(f"Process started (PID: {process.pid})")
print("Monitoring progress every 5 minutes...")
print("="*70)
print()

try:
    while True:
        # Check if process is still running
        return_code = process.poll()

        if return_code is not None:
            # Process completed
            print("\n" + "="*70)
            print(f"WORKFLOW COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Exit code: {return_code}")
            print("="*70)

            # Print final log contents
            if os.path.exists(log_file):
                print("\n" + "="*70)
                print("FINAL LOG CONTENTS:")
                print("="*70)
                with open(log_file, 'r') as f:
                    content = f.read()
                    if len(content) > 20000:
                        print(content[:10000])
                        print("\n... (truncated middle) ...\n")
                        print(content[-10000:])
                    else:
                        print(content)

            if return_code == 0:
                print("\n✓ WORKFLOW COMPLETED SUCCESSFULLY")
            else:
                print(f"\n✗ WORKFLOW FAILED WITH EXIT CODE: {return_code}")
            break

        # Check for new log content every 5 minutes
        current_time = time.time()
        if current_time - last_update >= 300:  # 5 minutes
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_content = f.read()

                        if new_content:
                            print(f"\n--- Progress Update [{datetime.now().strftime('%H:%M:%S')}] ---")
                            # Print last 2000 chars of new content
                            if len(new_content) > 2000:
                                print("...")
                                print(new_content[-2000:])
                            else:
                                print(new_content)
                            print("-" * 70)

                            last_position = f.tell()
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Workflow still running, no new log output...")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Error reading log: {e}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for log file to be created...")

            last_update = current_time

        # Sleep for 30 seconds before next check
        time.sleep(30)

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
