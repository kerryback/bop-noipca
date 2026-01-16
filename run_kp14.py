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

# Upload results to S3 (if configured) or warn about data loss
print("\n" + "="*70)
print("Workflow finished. Uploading results to S3...")
print("="*70)
print()

# Check if S3 is configured
s3_configured = os.environ.get('S3_BUCKET') and os.environ.get('AWS_ACCESS_KEY_ID')

if s3_configured:
    try:
        # Run S3 upload script
        upload_result = subprocess.run(
            [sys.executable, "upload_to_s3.py"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        if upload_result.returncode == 0:
            print("\n✓ Results uploaded successfully to S3")
        else:
            print("\n✗ S3 upload failed (see errors above)")
            print("WARNING: Results will be lost when container exits")

    except Exception as e:
        print(f"\n✗ Error uploading to S3: {e}")
        print("WARNING: Results will be lost when container exits")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  S3 NOT CONFIGURED - Results will be LOST when container exits!")
    print()
    print("To preserve results, configure S3 environment variables:")
    print("  - S3_BUCKET")
    print("  - AWS_ACCESS_KEY_ID")
    print("  - AWS_SECRET_ACCESS_KEY")
    print()
    print("See S3_SETUP.md for configuration instructions.")
    print()
    print("Container will exit in 30 seconds to stop billing...")
    time.sleep(30)

# Exit cleanly to stop instance billing
print("\n" + "="*70)
print("Container exiting...")
print("="*70)
sys.exit(return_code if return_code else 0)
