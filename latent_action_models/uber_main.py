# ## Uber (german) Python script to run a set of scripts for training and evaluating latent action models 

# import subprocess
# import time

# # files = ["warp_movingmnist.py", "wm_movingmnist.py", "warp_weather.py", "wm_weather.py"]
# files = ["warp_movingmnist.py"]

# for i, file in enumerate(files):
#     print(f"Running {file}...", flush=True)
#     # with open("nohup.log", "w") as f:
#     with open("nohup_new.log", "w") as f:
#         result = subprocess.run(["python", "-u", file], stdout=f, stderr=f)
#     print(f"Finished {file} with return code {result.returncode}", flush=True)

#     if i < len(files) - 1:
#         print("Waiting 60 seconds before next run...", flush=True)
#         time.sleep(60)

# print("All done!", flush=True)



import os
import subprocess
import time

target_pid = 4112807
print(f"Waiting for process {target_pid} to finish...", flush=True)
# Loop and wait as long as the process directory exists in /proc
while os.path.exists(f"/proc/{target_pid}"):
    time.sleep(60*15)  # Check every 30 seconds to save CPU cycles
print(f"Process {target_pid} has finished. Initiating the über script runs...", flush=True)

# files = ["warp_movingmnist.py", "wm_movingmnist.py", "warp_weather.py", "wm_weather.py"]
files = ["warp_movingmnist.py"]

for i, file in enumerate(files):
    print(f"Running {file}...", flush=True)
    with open("nohup.log", "w") as f:
        result = subprocess.run(["python", "-u", file], stdout=f, stderr=f)
    print(f"Finished {file} with return code {result.returncode}", flush=True)

    if i < len(files) - 1:
        print("Waiting 60 seconds before next run...", flush=True)
        time.sleep(60)

print("All done!", flush=True)
