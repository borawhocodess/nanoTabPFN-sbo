import torch

from nanotabpfn.utils import compute_bucket_borders


def test_compute_bucket_borders_simple_case():
    ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ys = torch.tensor(ys, dtype=torch.float32)

    borders = compute_bucket_borders(num_buckets=5, ys=ys)

    expected = torch.tensor([1.0, 2.5, 4.5, 6.5, 8.5, 10.0])
    torch.testing.assert_close(borders, expected)
