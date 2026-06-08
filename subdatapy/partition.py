"""Data partitioning utilities for distributed SubDataPy.

Pure functions for partitioning data across ranks by configuration
boundaries. Used by BaseData when torch.distributed is initialized
and file paths are provided.
"""

import numpy as np
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional


def scan_config_boundaries(config_idxs_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan a config_idxs .npy file to find configuration boundaries.

    Expects configs to appear in contiguous blocks (all rows of a config
    are adjacent). Raises ValueError if configs are interleaved.

    Args:
        config_idxs_path: Path to config_idxs.npy file.

    Returns:
        (unique_ids, start_rows, row_counts) where:
        - unique_ids: (n_configs,) array of config IDs in file order
        - start_rows: (n_configs,) array of first row index for each config
        - row_counts: (n_configs,) array of row count for each config
    """
    config_idxs = np.load(config_idxs_path, mmap_mode='r')
    n = len(config_idxs)

    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Find where config ID changes
    changes = np.where(config_idxs[1:] != config_idxs[:-1])[0] + 1
    start_rows = np.concatenate([[0], changes])
    unique_ids = np.array(config_idxs[start_rows])

    # Verify contiguity: each unique ID should appear exactly once in unique_ids
    if len(unique_ids) != len(np.unique(unique_ids)):
        raise ValueError(
            "config_idxs file has interleaved configurations. "
            "All rows of a config must be contiguous.")

    # Compute row counts
    end_rows = np.concatenate([changes, [n]])
    row_counts = end_rows - start_rows

    return unique_ids, start_rows, row_counts


def compute_partition_ranges(
    row_counts: np.ndarray,
    world_size: int,
) -> List[Tuple[int, int]]:
    """Compute contiguous row ranges for each rank, balanced by row count.

    Walks configs in order, accumulating rows. Cuts at the config boundary
    nearest to target = total_rows / world_size for each rank.

    Args:
        row_counts: (n_configs,) array of row counts per config.
        world_size: Number of ranks to partition across.

    Returns:
        List of (start_row, end_row) tuples, one per rank.
    """
    total_rows = int(row_counts.sum())
    n_configs = len(row_counts)

    if world_size > n_configs:
        raise ValueError(
            f"world_size ({world_size}) > n_configs ({n_configs}). "
            f"Cannot assign at least one config per rank.")

    target_per_rank = total_rows / world_size
    ranges = []
    current_start = 0
    cumulative_rows = np.cumsum(row_counts)
    config_idx = 0

    for rank in range(world_size):
        if rank == world_size - 1:
            # Last rank gets everything remaining
            ranges.append((current_start, total_rows))
        else:
            target_end = target_per_rank * (rank + 1)
            # Find config boundary closest to target_end
            # cumulative_rows[i] = end row after config i
            # We want the config boundary where cumulative_rows is closest to target_end
            remaining_configs = cumulative_rows[config_idx:]
            diffs = np.abs(remaining_configs - target_end)
            best_local = np.argmin(diffs)
            best_config = config_idx + best_local

            # Ensure at least one config per remaining rank
            remaining_ranks = world_size - rank - 1
            remaining_configs_count = n_configs - best_config - 1
            if remaining_configs_count < remaining_ranks:
                best_config = n_configs - remaining_ranks - 1

            # Ensure this rank gets at least one config
            if best_config < config_idx:
                best_config = config_idx

            end_row = int(cumulative_rows[best_config])
            ranges.append((current_start, end_row))
            current_start = end_row
            config_idx = best_config + 1

    return ranges


def mmap_load_partition(
    file_path: str,
    start_row: int,
    end_row: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Load a row range from a .npy file straight into one buffer (~1x RAM).

    The slice is read directly from disk with ``np.fromfile`` at the row's
    byte offset, so only the destination buffer is resident. The earlier
    ``np.load(mmap_mode='r')`` + ``np.array(slice)`` held BOTH the destination
    copy AND the memory-mapped source pages that get paged in to feed the
    copy — ~2x peak RSS per rank. ``np.fromfile`` reads via the OS read cache
    (kernel page cache, not counted against process RSS), so peak stays ~1x.

    Falls back to the mmap+copy path for Fortran-ordered files (where row
    slices are not contiguous on disk) or any header-parsing surprise.

    Args:
        file_path: Path to .npy file.
        start_row: First row (inclusive).
        end_row: Last row (exclusive).
        dtype: Target torch dtype.

    Returns:
        Tensor of shape (end_row - start_row, ...) on CPU.
    """
    partition = None
    try:
        # Read header metadata only; mmap is lazy so no data pages are touched.
        mm = np.load(file_path, mmap_mode='r')
        file_dtype = mm.dtype
        tail = tuple(mm.shape[1:])
        ncols = int(np.prod(tail)) if mm.ndim > 1 else 1
        data_offset = int(mm.offset)
        c_contig = bool(mm.flags['C_CONTIGUOUS'])
        del mm
        if c_contig:
            n = end_row - start_row
            with open(file_path, 'rb') as f:
                f.seek(data_offset + start_row * ncols * file_dtype.itemsize)
                partition = np.fromfile(f, dtype=file_dtype, count=n * ncols)
            if partition.size != n * ncols:
                partition = None  # short read (e.g. truncated) -> fallback
            elif tail:
                partition = partition.reshape((n,) + tail)
    except Exception:
        partition = None

    if partition is None:
        arr = np.load(file_path, mmap_mode='r')
        partition = np.array(arr[start_row:end_row])  # mmap+copy fallback
        del arr

    return torch.from_numpy(partition).to(dtype=dtype, device='cpu')


def validate_partitions(local_config_ids: torch.Tensor) -> List[List[int]]:
    """Validate that partitions across ranks are disjoint.

    All-gathers config IDs from all ranks and checks for overlap.

    Args:
        local_config_ids: Unique config IDs on this rank.

    Returns:
        List of config ID lists from all ranks (for building global list).

    Raises:
        ValueError: If config IDs overlap between ranks.
    """
    local_ids = local_config_ids.cpu().tolist()
    all_ids = [None] * dist.get_world_size()
    dist.all_gather_object(all_ids, local_ids)

    # Check disjointness
    seen = set()
    for rank, ids in enumerate(all_ids):
        id_set = set(ids)
        overlap = seen & id_set
        if overlap:
            raise ValueError(
                f"Rank {rank} has config IDs {overlap} that overlap with "
                f"previous ranks. Data is not properly partitioned.")
        seen.update(id_set)

    return all_ids


def build_global_config_ids(local_config_ids: torch.Tensor) -> torch.Tensor:
    """All-gather unique config IDs from all ranks and return sorted global list.

    Args:
        local_config_ids: Unique config IDs on this rank (CPU tensor).

    Returns:
        Sorted 1D tensor of all unique config IDs across all ranks (CPU).
    """
    # Gather the per-rank id tensors (not Python lists) and dedup with
    # torch.unique. The previous .tolist() + sorted(set(...)) allocated one
    # Python int object per config id (millions of them, several GB at scale).
    local = local_config_ids.detach().cpu().contiguous()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    cat = torch.cat([g.to(local.dtype) for g in gathered])
    return torch.unique(cat).to(dtype=local_config_ids.dtype, device='cpu')
