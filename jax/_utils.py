import equinox as eqx
import numpy as np
import json
import shutil
import os
from pathlib import Path
from datetime import datetime


def get_run_path():
    """Generates a run path based on timestamp inside ../runs"""
    # Assuming script is in `project/src`, runs will be in `project/runs`
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    base_dir = Path("../runs")
    run_dir = base_dir / timestamp
    return run_dir

def find_latest_run():
    """Finds the most recent folder in ../runs"""
    base_dir = Path("../runs")
    if not base_dir.exists():
        raise FileNotFoundError("No runs folder found.")
    runs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError("No run directories found.")
    return max(runs, key=os.path.getmtime)

def save_run(run_dir, model, config, history):
    """Saves model, config, and history."""
    run_dir.mkdir(parents=True, exist_ok=True)

    ## Make a dir names artefacts inside run_dir
    artefacts_dir = run_dir / "artefacts"
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    ## Make a plots folder inside run_dir
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 0. Save data.py, models.py, analysis.py, and main.py for reproducibility
    shutil.copy(__file__, run_dir / "_utils.py")
    shutil.copy(Path(__file__).parent / "data.py", run_dir / "data.py")
    shutil.copy(Path(__file__).parent / "models.py", run_dir / "models.py")
    shutil.copy(Path(__file__).parent / "analysis.py", run_dir / "analysis.py")
    shutil.copy(Path(__file__).parent / "main.py", run_dir / "main.py")

    # 1. Save Model (Equinox way) (inside artefacts)
    # eqx.tree_serialise_leaves(run_dir / "model.eqx", model)
    eqx.tree_serialise_leaves(artefacts_dir / "model.eqx", model)

    # 2. Save Config
    if config is not None:
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
    # 3. Save History
    if history is not None:
        np.save(artefacts_dir / "loss_history.npy", history)
    
    print(f"✅ Run saved to {run_dir}")

def load_experiment(run_dir, model_skeleton):
    """Loads model and history."""
    print(f"📂 Loading experiment from {run_dir}...")
    
    artefacts_dir = run_dir / "artefacts"

    # 1. Load Model
    model = eqx.tree_deserialise_leaves(artefacts_dir / "model.eqx", model_skeleton)
    
    # 2. Load History
    history = np.load(artefacts_dir / "loss_history.npy", allow_pickle=True).item()
    
    return model, history
