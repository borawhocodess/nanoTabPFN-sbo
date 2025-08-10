import h5py
import random
import torch
import numpy as np

from pfns.bar_distribution import get_bucket_limits

def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_default_device():
    device = 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    if torch.cuda.is_available(): device = 'cuda'
    return device

def make_global_bucket_edges(filename, n_buckets=100, device=None):
    with h5py.File(filename, "r") as f:
        ys_all = f["y"][:]
        ys_concat = ys_all.reshape(-1)

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)

    return global_bucket_edges
