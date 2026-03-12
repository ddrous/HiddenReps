#%% Cell 1: Imports, 
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
from torch.utils.data import DataLoader, Subset

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

#%% Cell 2: Collect the epoch and avg loss

#""" ### Example line in nohup.log
# Loading Moving MNIST Dataset...
# Original loaded MovingMNIST shape: (20, 10000, 64, 64) (T, N, H, W)
# Batched Video shape: (256, 20, 64, 64, 1)

# === Visualizing Video Augmentations ===

# 🚀 [PHASE 1] Starting Base Training (IDM + FDM + maybe Enc) -> Saving to experiments/260312-113749
# Total Trainable Parameters in Phase 1 WARP: 32539195
# Phase 1 - Epoch 10/2500 - Avg Loss: 0.027188
# Phase 1 - Epoch 20/2500 - Avg Loss: 0.024963 
#"""

file = "nohup.log"
epoch_list = []
loss_list = []

with open(file, 'r') as f:
    for line in f:
        if "Phase 1 - Epoch" in line and "Avg Loss" in line:
            parts = line.split(" - ")
            epoch_part = parts[1]  # "Epoch 10/2500"
            loss_part = parts[2]    # "Avg Loss: 0.027188"
            
            epoch_num = int(epoch_part.split()[1].split('/')[0])  # Extract the current epoch number
            avg_loss = float(loss_part.split()[2])  # Extract the average loss
            
            epoch_list.append(epoch_num)
            loss_list.append(avg_loss)

#%% Cell 3: Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(epoch_list, loss_list, marker='o', linestyle='-', color='b')
plt.title('Average Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.yscale('log')  # Use logarithmic scale for better visibility of loss changes
plt.grid()
# plt.xticks(np.arange(0, max(epoch_list)+1, 10))  # Set x-ticks every 10 epochs
plt.show()
