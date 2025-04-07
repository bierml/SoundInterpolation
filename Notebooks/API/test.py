import subprocess
import sys
'''command = [
    "python",
    "inference_batch_prediction.py",
    "model_checkpoint.keras",
    "1c.wav",
    "output_last.wav",
    "0.95"
]
result = subprocess.run(command, capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)'''

command = [
    "python",
    "script.py",
    "1.rf64",
    "1c.rf64",
    "32"
]
# Open the process with line-buffered output.
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True
)

# Read and print output dynamically as it is produced.
i = 0 
for line in process.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
process.stdout.close()
return_code = process.wait()
