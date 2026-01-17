"""Wrapper to run gs21 workflow with periodic progress monitoring."""
import sys
import subprocess
import time
import os
from datetime import datetime

print("="*70)
print("Starting GS21 workflow: indices 0-9")
print("="*70)
print()

# Run the main workflow
cmd = [sys.executable, "main.py", "gs21", "0", "10"]
print(f"Executing: {' '.join(cmd)}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

log_file = "/app/logs/gs21_0_10.log"
runtime_log_file = "/app/logs/gs21_0_10_runtime.log"

# Open runtime log file to capture all stdout/stderr
runtime_log = open(runtime_log_file, 'w', buffering=1)

# Start the workflow and stream output in real-time
print("Starting workflow...")
runtime_log.write("="*70 + "\n")
runtime_log.write(f"Starting GS21 workflow: indices 0-9\n")
runtime_log.write("="*70 + "\n\n")
runtime_log.write(f"Executing: {' '.join(cmd)}\n")
runtime_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print(f"Process started (PID: {process.pid})")
runtime_log.write(f"Process started (PID: {process.pid})\n")
runtime_log.write("="*70 + "\n\n")
print("="*70)
print()

try:
    # Stream output in real-time and save to runtime log
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            runtime_log.write(line)

    # Wait for process to complete
    return_code = process.wait()
    runtime_log.flush()

    completion_msg = "\n" + "="*70 + "\n"
    completion_msg += f"WORKFLOW COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    completion_msg += f"Exit code: {return_code}\n"
    completion_msg += "="*70 + "\n"

    print(completion_msg)
    runtime_log.write(completion_msg)

    if return_code == 0:
        success_msg = "\n✓ WORKFLOW COMPLETED SUCCESSFULLY\n"
        print(success_msg)
        runtime_log.write(success_msg)
    else:
        fail_msg = f"\n✗ WORKFLOW FAILED WITH EXIT CODE: {return_code}\n"
        print(fail_msg)
        runtime_log.write(fail_msg)

except KeyboardInterrupt:
    interrupt_msg = "\n\nInterrupted! Terminating workflow...\n"
    print(interrupt_msg)
    runtime_log.write(interrupt_msg)
    process.terminate()
    process.wait(timeout=10)
    terminate_msg = "Workflow terminated.\n"
    print(terminate_msg)
    runtime_log.write(terminate_msg)

except Exception as e:
    error_msg = "\n" + "="*70 + "\n"
    error_msg += f"ERROR: {e}\n"
    error_msg += "="*70 + "\n"
    print(error_msg)
    runtime_log.write(error_msg)

    import traceback
    traceback_str = traceback.format_exc()
    print(traceback_str)
    runtime_log.write(traceback_str)

    # Try to terminate process
    try:
        process.terminate()
        process.wait(timeout=10)
    except:
        pass

finally:
    # Close runtime log file
    runtime_log.close()

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
