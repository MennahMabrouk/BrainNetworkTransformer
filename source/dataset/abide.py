import numpy as np
import torch
from omegaconf import DictConfig, open_dict
import sys
import os

# Dynamically add project root to sys.path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from source.dataset.preprocess import StandardScaler

def load_abide_data(cfg: DictConfig):

    # Load dataset
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    # Standardize time series data
    scaler = StandardScaler(mean=np.mean(final_timeseires), std=np.std(final_timeseires))
    final_timeseires = scaler.transform(final_timeseires)

    # Convert numpy arrays to PyTorch tensors
    final_timeseires, final_pearson, labels = [
        torch.from_numpy(data).float() for data in (final_timeseires, final_pearson, labels)
    ]

    # Update configuration with dataset dimensions
    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels, site
