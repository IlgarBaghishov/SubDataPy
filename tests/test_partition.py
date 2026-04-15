import torch
import numpy as np
import pytest
import tempfile
import os
from subdatapy.partition import (
    scan_config_boundaries,
    compute_partition_ranges,
    mmap_load_partition,
)


@pytest.fixture
def config_idxs_file(tmp_path):
    """Create a temporary config_idxs.npy with known structure."""
    # 5 configs, each with different row counts: 10, 5, 20, 3, 12 = 50 rows total
    ids = (
        [0] * 10 +
        [1] * 5 +
        [2] * 20 +
        [3] * 3 +
        [4] * 12
    )
    arr = np.array(ids, dtype=np.int64)
    path = str(tmp_path / "config_idxs.npy")
    np.save(path, arr)
    return path, arr


@pytest.fixture
def data_file(tmp_path):
    """Create a temporary X.npy with known values."""
    np.random.seed(0)
    X = np.random.randn(50, 10).astype(np.float64)
    path = str(tmp_path / "X.npy")
    np.save(path, X)
    return path, X


def test_scan_config_boundaries(config_idxs_file):
    path, arr = config_idxs_file
    unique_ids, start_rows, row_counts = scan_config_boundaries(path)

    assert list(unique_ids) == [0, 1, 2, 3, 4]
    assert list(start_rows) == [0, 10, 15, 35, 38]
    assert list(row_counts) == [10, 5, 20, 3, 12]
    assert row_counts.sum() == 50


def test_scan_config_boundaries_interleaved(tmp_path):
    """Interleaved configs should raise ValueError."""
    arr = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    path = str(tmp_path / "config_idxs.npy")
    np.save(path, arr)
    with pytest.raises(ValueError, match="interleaved"):
        scan_config_boundaries(path)


def test_compute_partition_ranges_2_ranks():
    # 5 configs: 10, 5, 20, 3, 12 = 50 rows. Target = 25/rank
    row_counts = np.array([10, 5, 20, 3, 12])
    ranges = compute_partition_ranges(row_counts, world_size=2)

    assert len(ranges) == 2
    # Check coverage
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 50
    assert ranges[0][1] == ranges[1][0]  # contiguous

    # Both partitions should be at config boundaries
    cumulative = np.cumsum(row_counts)
    cut = ranges[0][1]
    assert cut in cumulative  # cut is at a config boundary


def test_compute_partition_ranges_equal_configs():
    # 4 configs, each 10 rows = 40 total. 2 ranks → 20 each.
    row_counts = np.array([10, 10, 10, 10])
    ranges = compute_partition_ranges(row_counts, world_size=2)

    assert ranges == [(0, 20), (20, 40)]


def test_compute_partition_ranges_4_ranks():
    # 8 configs, each 5 rows = 40 total. 4 ranks → 10 each.
    row_counts = np.array([5, 5, 5, 5, 5, 5, 5, 5])
    ranges = compute_partition_ranges(row_counts, world_size=4)

    assert len(ranges) == 4
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 40
    for i in range(3):
        assert ranges[i][1] == ranges[i + 1][0]


def test_compute_partition_ranges_uneven():
    # 3 configs: 1, 1, 98 = 100 rows. 2 ranks.
    # Best split: configs 0-1 (2 rows) to rank 0, config 2 (98 rows) to rank 1
    # OR configs 0-2 all to rank 0 — depends on which is closer to target=50
    row_counts = np.array([1, 1, 98])
    ranges = compute_partition_ranges(row_counts, world_size=2)

    assert len(ranges) == 2
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 100
    # Each rank must have at least 1 config
    assert ranges[0][1] > 0
    assert ranges[1][1] > ranges[1][0]


def test_compute_partition_ranges_too_many_ranks():
    row_counts = np.array([10, 10, 10])
    # world_size == n_configs is valid (one config per rank)
    ranges = compute_partition_ranges(row_counts, world_size=3)
    assert ranges == [(0, 10), (10, 20), (20, 30)]
    # world_size > n_configs must raise
    with pytest.raises(ValueError, match="world_size.*>.*n_configs"):
        compute_partition_ranges(row_counts, world_size=4)


def test_mmap_load_partition(data_file):
    path, X = data_file
    partition = mmap_load_partition(path, 10, 35)

    assert partition.shape == (25, 10)
    assert partition.dtype == torch.float64
    assert partition.device == torch.device('cpu')
    assert torch.allclose(partition, torch.from_numpy(X[10:35]))


def test_mmap_load_partition_full(data_file):
    path, X = data_file
    partition = mmap_load_partition(path, 0, 50)

    assert partition.shape == (50, 10)
    assert torch.allclose(partition, torch.from_numpy(X))


def test_mmap_load_partition_1d(tmp_path):
    """1D arrays (like y or w) should also work."""
    arr = np.arange(100, dtype=np.float64)
    path = str(tmp_path / "y.npy")
    np.save(path, arr)

    partition = mmap_load_partition(path, 20, 50)
    assert partition.shape == (30,)
    assert torch.allclose(partition, torch.from_numpy(arr[20:50]))
