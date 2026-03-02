## Uber (german) Python script to run a set of scripts for training and evaluating latent action models 

import subprocess
import time

files = ["warp_movingmnist.py", "wm_movingmnist.py", "warp_weather.py", "wm_weather.py"]
# files = ["warp_movingmnist.py", "warp_weather.py"]

for i, file in enumerate(files):
    print(f"Running {file}...", flush=True)
    with open("nohup.log", "w") as f:
        result = subprocess.run(["python", "-u", file], stdout=f, stderr=f)
    print(f"Finished {file} with return code {result.returncode}", flush=True)

    if i < len(files) - 1:
        print("Waiting 60 seconds before next run...", flush=True)
        time.sleep(60)

print("All done!", flush=True)
