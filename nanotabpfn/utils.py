import h5py
import random
import torch
import numpy as np


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_default_device():
    device = 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    if torch.cuda.is_available(): device = 'cuda'
    return device


def compute_bucket_borders(num_buckets: int, ys: torch.Tensor) -> torch.Tensor:
    """
    decides equal mass bucket borders from ys
    inspired by pfns.model.bar_distribution get_bucket_borders
    """
    ys = torch.as_tensor(ys)
    if not torch.is_floating_point(ys):
        ys = ys.to(torch.float32)
    ys = ys.flatten()
    ys = ys[torch.isfinite(ys)]

    if ys.numel() <= num_buckets:
        raise ValueError(f"ys numel ({ys.numel()}) <= num buckets ({num_buckets})")

    n = (ys.numel() // num_buckets) * num_buckets
    ys = ys[:n]
    ys_per_bucket = n // num_buckets

    ys_sorted, _ = torch.sort(ys)

    chunks = ys_sorted.reshape(num_buckets, ys_per_bucket)
    interiors = (chunks[:-1, -1] + chunks[1:, 0]) / 2

    min_outer = ys_sorted[0].unsqueeze(0)
    max_outer = ys_sorted[-1].unsqueeze(0)

    borders = torch.cat((min_outer, interiors, max_outer))

    if borders.numel() - 1 != num_buckets:
        raise ValueError("num borders - 1 != num buckets")

    if torch.unique_consecutive(borders).numel() != borders.numel():
        raise ValueError("duplicate borders detected")

    return borders


def make_global_bucket_edges(filename, n_buckets=100, device=get_default_device(), max_y=5_000_000):
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, num_datapoints = y.shape
        total = num_tables * num_datapoints

        if max_y >= total:
            ys_concat = y[...].reshape(-1)
        else:
            full_rows = max_y // num_datapoints
            rem =  max_y % num_datapoints

            parts = []
            if full_rows > 0:
                parts.append(y[:full_rows, :].reshape(-1))
            if rem > 0:
                parts.append(y[full_rows, :rem].reshape(-1))
            ys_concat = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=y.dtype)

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = compute_bucket_borders(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges
