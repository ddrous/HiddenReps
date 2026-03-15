#%% Cell 1: Imports, Utilities, and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
from jax.flatten_util import ravel_pytree
from typing import Optional

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Sampler

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

import h5py
import io
import tempfile
import cv2
from PIL import Image
from IPython.display import display, Image as IPyImage

#%% Load the data in "data/PhyWorld/uniform_motion_30K.hdf5", and visualise 

# data_path = "data/PhyWorld/uniform_motion_30K.hdf5"
# data_path = "data/PhyWorld/uniform_motion_eval.hdf5"

# data_path = "data/PhyWorld/collision_30K.hdf5"
data_path = "data/PhyWorld/collision_eval.hdf5"

# We will use lists to collect the chunks
all_video_chunks = []
all_position_chunks = []

# Open the file and extract the datasets
with h5py.File(data_path, 'r') as f:
    video_group = f['video_streams']
    position_group = f['position_streams']
    
    # Loop through EVERY member in the group (e.g., '00000', '00001', etc.)
    for key in video_group.keys():
        # Extract the chunk and append it to our list
        all_video_chunks.append(video_group[key][:])
        all_position_chunks.append(position_group[key][:])

# Concatenate all the smaller chunks into one massive NumPy array
video_data = np.concatenate(all_video_chunks, axis=0)
position_data = np.concatenate(all_position_chunks, axis=0)
print("Data loaded successfully!")
print(f"Total simulations available: {len(video_data)}")


#%%
# 1. Specify the index you want to look at (0 to 999)
# sim_index = np.random.randint(0, len(video_data))
sim_index = 20

# --- PART A: The Static Trajectory Plot ---
plt.figure(figsize=(5, 4))

# Grab the 32-frame trajectory for this specific simulation
x_coords = position_data[sim_index, :, 0]
y_coords = position_data[sim_index, :, 1]

plt.plot(x_coords, y_coords, marker='.', linestyle='-', color='b')
plt.title(f"Trajectory for Simulation {sim_index}")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()


# --- PART B: The GIF Animation ---
video_bytes = video_data[sim_index]
frames = []

# Write bytes to a temp file, use OpenCV to extract frames
with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_vid:
    temp_vid.write(video_bytes)
    temp_vid.flush()
    
    cap = cv2.VideoCapture(temp_vid.name)
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # OpenCV uses BGR, we need RGB for a correct-looking GIF
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        
    cap.release()

# Save the frames as a GIF and display it
if frames:
    gif_path = f"data/PhyWorld/gifs/simulation_{sim_index}.gif"
    
    # Save using Pillow
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100, # Milliseconds per frame (100ms = 10 FPS)
        loop=0 # 0 means loop forever
    )
    
    print(f"GIF successfully saved as: {gif_path}")
    
    # Render the GIF inline in the Jupyter output
    display(IPyImage(filename=gif_path))
else:
    print("Could not extract frames from the video bytes.")


#%%

import json

data_path = "data/PhyWorld/collision_30K.hdf5"
sim_index = 20 # The same video from your previous cell

with h5py.File(data_path, 'r') as f:
    init_group = f['init_streams']
    
    # Grab the first chunk/member
    first_member = list(init_group.keys())[0]
    
    # Extract the raw metadata bytes for our specific simulation
    raw_meta = init_group[first_member][sim_index]
    
    # Decode the bytes into a string
    meta_string = raw_meta.decode('utf-8') if isinstance(raw_meta, bytes) else str(raw_meta)

    # Let's try to print it nicely if it's JSON, otherwise just print the string
    try:
        meta_dict = json.loads(meta_string)
        print(json.dumps(meta_dict, indent=4))
    except json.JSONDecodeError:
        print(meta_string)



#%%
