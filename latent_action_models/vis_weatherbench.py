# #%%
# import os
# import subprocess
# import glob
# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import matplotlib.pyplot as plt

# import seaborn as sns
# sns.set(style="white", context="talk")
# plt.rcParams['savefig.facecolor'] = 'white'

# try:
#     import xarray as xr
# except ImportError:
#     raise ImportError("Please install xarray and netcdf4: pip install xarray netcdf4")

# class WeatherBenchTemperature(Dataset):
#     def __init__(self, data_path="./data/WeatherBench", split="train", download=False, seq_len=20, mean=None, std=None):
#         """
#         PyTorch Dataset for WeatherBench 2m Temperature (5.625 degree resolution).
        
#         Args:
#             data_path (str): Directory where the .nc files will be stored.
#             split (str): 'train' (1979-2015), 'val' (2016), or 'test' (2017-2018).
#             download (bool): If True and files are missing, downloads and extracts the data.
#             seq_len (int): The continuous sequence length T returned per sample.
#             mean (float, optional): Mean for normalization. If None and split='train', it is computed.
#             std (float, optional): Std for normalization. If None and split='train', it is computed.
#         """
#         self.data_path = data_path
#         self.split = split
#         self.seq_len = seq_len
        
#         # 1. Download if requested
#         if download:
#             self._download_and_extract()
            
#         # 2. Determine which years belong to this split (OpenSTL/WeatherBench standard)
#         if split == "train":
#             years = [str(y) for y in range(1979, 2016)]
#         elif split == "val":
#             years = ['2016']
#         elif split == "test":
#             years = ['2017', '2018']
#         else:
#             raise ValueError("Split must be 'train', 'val', or 'test'.")
            
#         # 3. Load the data using Xarray
#         file_patterns = [os.path.join(data_path, f"*{y}*.nc") for y in years]
#         files_to_load = []
#         for pat in file_patterns:
#             files_to_load.extend(glob.glob(pat))
            
#         if len(files_to_load) == 0:
#             raise FileNotFoundError(f"No .nc files found for {split} split in {data_path}. Try setting download=True.")
            
#         print(f"[{split.upper()}] Loading {len(files_to_load)} years of data into memory...")
#         # Combine all files along the time dimension
#         dataset = xr.open_mfdataset(sorted(files_to_load), combine='by_coords')
        
#         # Extract the numpy array: Shape (Time, Latitude, Longitude) -> (T, 32, 64)
#         raw_data = dataset.get('t2m').values 
        
#         # 4. Preprocessing (Normalization)
#         if split == "train":
#             # Compute stats on training set to prevent data leakage
#             self.mean = raw_data.mean() if mean is None else mean
#             self.std = raw_data.std() if std is None else std
#         else:
#             # Validation/Test sets MUST use the training set's mean and std
#             if mean is None or std is None:
#                 print("Warning: You are normalizing a test/val set without providing train mean/std. Computing locally...")
#             self.mean = raw_data.mean() if mean is None else mean
#             self.std = raw_data.std() if std is None else std
            
#         # Normalize
#         norm_data = (raw_data - self.mean) / self.std
        
#         # Add Channel dimension -> (Time_Total, Channels, Height, Width) -> (T_total, 1, 32, 64)
#         self.data = np.expand_dims(norm_data, axis=1).astype(np.float32)
#         print(f"[{split.upper()}] Loaded Tensor Shape: {self.data.shape}")

#     def _download_and_extract(self):
#         # Check if we already have .nc files
#         if len(glob.glob(os.path.join(self.data_path, "*.nc"))) > 0:
#             return

#         os.makedirs(self.data_path, exist_ok=True)
#         zip_path = os.path.join(self.data_path, "2m_temperature.zip")
#         url = "https://dataserv.ub.tum.de/public.php/dav/files/m1524895/5.625deg/2m_temperature/?accept=zip"
        
#         # print("Downloading 2m Temperature (5.625 deg)... (~2GB)")
#         # subprocess.run(["wget", url, "-O", zip_path], check=True)

#         print("Extracting files...")
#         subprocess.run(["unzip", "-q", zip_path, "-d", self.data_path], check=True)
        
#         print("Cleaning up zip file...")
#         os.remove(zip_path)

#     def __len__(self):
#         # Total number of sliding windows of size seq_len
#         return len(self.data) - self.seq_len + 1

#     def __getitem__(self, idx):
#         # Returns a sequence of shape (seq_len, C, H, W)
#         seq = self.data[idx : idx + self.seq_len]
#         return torch.from_numpy(seq)

# # --- Visualization Helper ---
# def visualize_sequence(sequence_tensor, title="Weather Sequence"):
#     """
#     Visualizes a uniform tensor sequence of shape (Seq_Len, C, H, W)
#     """
#     seq_len = sequence_tensor.shape[0]
    
#     # We will pick 10 evenly spaced frames from the sequence to display
#     indices = np.linspace(0, seq_len - 1, min(10, seq_len), dtype=int)
    
#     fig, axes = plt.subplots(1, len(indices), figsize=(20, 3))
#     for i, idx in enumerate(indices):
#         # Shape is (1, 32, 64), squeeze to (32, 64) for matplotlib
#         img = sequence_tensor[idx, 0].numpy()
#         axes[i].imshow(img, cmap='coolwarm', origin='lower')
#         axes[i].set_title(f"t = {idx}")
#         axes[i].axis('off')
        
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()

# # ==========================================
# # Example Usage Pipeline
# # ==========================================
# if __name__ == "__main__":
#     # 1. Initialize Train Set (Will trigger download on first run)
#     train_dataset = WeatherBenchTemperature(
#         data_path="./data/WeatherBench/2m_temperature", 
#         split="train", 
#         download=False, 
#         seq_len=24
#     )
    
#     # Extract the scaling parameters from the train set to prevent data leakage!
#     train_mean = train_dataset.mean
#     train_std = train_dataset.std
    
#     # 2. Initialize Test Set using Train stats
#     test_dataset = WeatherBenchTemperature(
#         data_path="./data/WeatherBench/2m_temperature", 
#         split="test", 
#         download=False, 
#         seq_len=24,
#         mean=train_mean,
#         std=train_std
#     )
    
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    
#     print("\nData Loaders ready!")
#     print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    
#     # 3. Fetch one sequence from the test set and visualize it
#     sample_seq = test_dataset[0]  # Shape: (20, 1, 32, 64)
#     visualize_sequence(sample_seq, title="2m Temperature Rollout (Test Set)")




#%%

#%%
import os
import subprocess
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import seaborn as sns

# -- Enforce Crisp Visualizations --
sns.set(style="white", context="talk")
plt.rcParams['figure.dpi'] = 150  # High DPI for crisp images
plt.rcParams['savefig.facecolor'] = 'white'

try:
    import xarray as xr
except ImportError:
    raise ImportError("Please install xarray and netcdf4: pip install xarray netcdf4")

class WeatherBenchTemperature(Dataset):
    def __init__(self, data_path="./data/WeatherBench", split="train", download=False, seq_len=24, mean=None, std=None):
        """
        PyTorch Dataset for WeatherBench 2m Temperature (5.625 degree resolution).
        
        Args:
            data_path (str): Directory where the .nc files will be stored.
            split (str): 'train' (1979-2015), 'val' (2016), or 'test' (2017-2018).
            download (bool): If True and files are missing, downloads and extracts the data.
            seq_len (int): The continuous sequence length T. OpenSTL defaults to 24 (12 past -> 12 future).
            mean (float, optional): Mean for normalization. If None and split='train', it is computed.
            std (float, optional): Std for normalization. If None and split='train', it is computed.
        """
        self.data_path = data_path
        self.split = split
        self.seq_len = seq_len
        
        # 1. Download if requested
        if download:
            self._download_and_extract()
            
        # 2. Determine which years belong to this split (OpenSTL/WeatherBench standard)
        if split == "train":
            years = [str(y) for y in range(1979, 2016)]
        elif split == "val":
            years = ['2016']
        elif split == "test":
            years = ['2017', '2018']
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
            
        # 3. Load the data using Xarray
        file_patterns = [os.path.join(data_path, f"*{y}*.nc") for y in years]
        files_to_load = []
        for pat in file_patterns:
            files_to_load.extend(glob.glob(pat))
            
        if len(files_to_load) == 0:
            raise FileNotFoundError(f"No .nc files found for {split} split in {data_path}. Try setting download=True.")
            
        print(f"[{split.upper()}] Loading {len(files_to_load)} years of data into memory...")
        # Combine all files along the time dimension
        dataset = xr.open_mfdataset(sorted(files_to_load), combine='by_coords')
        
        # Extract the numpy array: Shape (Time, Latitude, Longitude) -> (T, 32, 64)
        raw_data = dataset.get('t2m').values 
        
        # 4. Preprocessing (Normalization)
        if split == "train":
            # Compute stats on training set to prevent data leakage
            self.mean = raw_data.mean() if mean is None else mean
            self.std = raw_data.std() if std is None else std
        else:
            # Validation/Test sets MUST use the training set's mean and std
            if mean is None or std is None:
                print("Warning: Normalizing a test/val set without train mean/std. Computing locally...")
            self.mean = raw_data.mean() if mean is None else mean
            self.std = raw_data.std() if std is None else std
            
        # Normalize
        norm_data = (raw_data - self.mean) / self.std
        
        # Add Channel dimension -> (Time_Total, Channels, Height, Width) -> (T_total, 1, 32, 64)
        self.data = np.expand_dims(norm_data, axis=1).astype(np.float32)
        print(f"[{split.upper()}] Loaded Tensor Shape: {self.data.shape}")

    def _download_and_extract(self):
        if len(glob.glob(os.path.join(self.data_path, "*.nc"))) > 0:
            return

        os.makedirs(self.data_path, exist_ok=True)
        zip_path = os.path.join(self.data_path, "2m_temperature.zip")
        url = "https://dataserv.ub.tum.de/public.php/dav/files/m1524895/5.625deg/2m_temperature/?accept=zip"
        
        print("Downloading 2m Temperature (5.625 deg)... (~2GB)")
        subprocess.run(["wget", url, "-O", zip_path], check=True)

        print("Extracting files...")
        subprocess.run(["unzip", "-q", zip_path, "-d", self.data_path], check=True)
        
        print("Cleaning up zip file...")
        os.remove(zip_path)

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len]
        return torch.from_numpy(seq)

# --- Visualization Helpers ---

def get_shared_vmin_vmax(sequence_tensor):
    """Calculates absolute min/max across the whole sequence for a unified color scale."""
    return sequence_tensor.min().item(), sequence_tensor.max().item()

def visualize_sequence(sequence_tensor, cmap, title="Weather Sequence"):
    """
    Visualizes a uniform tensor sequence frame-by-frame with a unified colorbar.
    """
    seq_len = sequence_tensor.shape[0]
    vmin, vmax = get_shared_vmin_vmax(sequence_tensor)
    
    num_frames_to_show = min(10, seq_len)
    indices = np.linspace(0, seq_len - 1, num_frames_to_show, dtype=int)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(20, 3), dpi=150)
    # cmap = 'blues'
    for i, idx in enumerate(indices):
        img = sequence_tensor[idx, 0].numpy()
        # nearest interpolation respects the actual 5.625 grid blocks
        im = axes[i].imshow(img, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
        
        axes[i].set_title(f"t = {idx} hrs", fontsize=12)
        axes[i].axis('off')
        
    # Add a shared colorbar to the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.2, 0.01, 0.6])
    fig.colorbar(im, cax=cbar_ax, label="Norm. Temperature")
    
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.show()

def animate_sequence(sequence_tensor, title="Weather Sequence", interval=200, cmap='RdBu_r'):
    """
    Generates an HTML5 video animation for the sequence.
    """
    seq_len = sequence_tensor.shape[0]
    vmin, vmax = get_shared_vmin_vmax(sequence_tensor)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    img_plot = ax.imshow(sequence_tensor[0, 0].numpy(), cmap=cmap, origin='lower', 
                         vmin=vmin, vmax=vmax, interpolation='nearest')
    
    ax.axis('off')
    title_text = ax.set_title(f"{title} | t = 0 hrs", fontsize=14)
    
    cbar = fig.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Norm. Temperature")
    
    def update(frame):
        img_plot.set_array(sequence_tensor[frame, 0].numpy())
        title_text.set_text(f"{title} | t = {frame} hrs")
        return [img_plot, title_text]
        
    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=interval, blit=False)
    
    # Close the static figure so it doesn't double-render below the video
    plt.close(fig)
    return HTML(ani.to_jshtml())

# ==========================================
# Example Usage Pipeline
# ==========================================
if __name__ == "__main__":
    # 1. Initialize Train Set
    train_dataset = WeatherBenchTemperature(
        data_path="./data/WeatherBench/2m_temperature", 
        split="train", 
        download=False, 
        seq_len=24  # Standard OpenSTL setting (12 pre -> 12 aft)
    )
    
    train_mean = train_dataset.mean
    train_std = train_dataset.std
    
    # 2. Initialize Test Set using Train stats
    test_dataset = WeatherBenchTemperature(
        data_path="./data/WeatherBench/2m_temperature", 
        split="test", 
        download=False, 
        seq_len=24,
        mean=train_mean,
        std=train_std
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    
    print("\nData Loaders ready!")
    
    # 3. Fetch one sequence from the test set
    sample_seq = test_dataset[0]  # Shape: (24, 1, 32, 64)
    
    # cmap = 'Blues'  # Color map for visualization
    cmap = 'RdBu_r'  # Diverging color map to show hot/cold deviations clearly
    # 4. Display Static Frame-by-Frame
    visualize_sequence(sample_seq, title="2m Temperature Rollout (Test Set)", cmap=cmap)
    
    # 5. Display Video Animation
    video_html = animate_sequence(sample_seq, title="2m Temperature Rollout", cmap=cmap)
    display(video_html)
