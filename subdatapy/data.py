import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import warnings

from . import linalg
from . import partition as _partition


def _synthesize_enrow_mask(config_idxs):
    """Mark the first row of each config as the energy row.

    Shared between the regular and partitioned loaders so both paths use
    the same scatter_reduce_amin convention.
    """
    unique_vals, inverse = torch.unique(config_idxs, return_inverse=True)
    perm = torch.arange(inverse.size(0))
    first_idx = torch.empty(unique_vals.size(0), dtype=torch.int64)
    first_idx.scatter_reduce_(0, inverse, perm, reduce="amin", include_self=False)
    mask = torch.zeros_like(config_idxs, dtype=torch.bool)
    mask[first_idx] = True
    return mask


def _validate_local_devices(local_devices):
    """Fail fast on bad `local_devices` input."""
    if not isinstance(local_devices, (list, tuple)) or len(local_devices) == 0:
        raise ValueError(
            f"local_devices must be a non-empty list/tuple, got {local_devices!r}")
    for d in local_devices:
        if not isinstance(d, str):
            raise ValueError(f"local_devices entries must be strings, got {d!r}")
        if d != 'cpu' and not d.startswith('cuda'):
            raise ValueError(
                f"local_devices entries must be 'cpu' or 'cuda[:N]', got {d!r}")


def process_data(x, dtype=torch.float64, device='cpu'):
    """
    Loads data into CPU memory (Pinned if possible for fast transfer).
    """
    if x is None:
        return None
    
    if isinstance(x, str):
        if x.endswith('.npy'):
            x = np.load(x)
        else:
            raise ValueError('File format not supported')

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, pd.DataFrame):
        x = torch.from_numpy(x.values)
    
    if not isinstance(x, torch.Tensor):
        raise ValueError('Input type not supported')

    # Keep on CPU by default to save VRAM
    if x.dtype != dtype or x.device!=device:
        x = x.to(dtype=dtype, device=device)
    return x


class BaseData:

    def __init__(self, X, y=None, w=None, config_idxs=None, enrow_mask=None,
                 intercept=True, device='cuda', local_devices=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None):
        """Args:
            partitioned_override: Force `_is_partitioned` to this value
                instead of auto-detecting from (distributed AND file-path
                inputs). Used by nested samplers in CookSubSampler to
                inherit the parent's partition state when they receive
                tensors (not paths).
            unique_config_idxs_train_override: If provided, replaces the
                per-rank unique_config_idxs_train computed during
                train_test_split. Used by nested samplers to see the global
                config-id list rather than their partition-local view.

        Device rule (see README): row-scale data (X and the per-row/per-config
        arrays) lives on the host CPU; the device only ever holds the small
        model-scale factors and one streamed chunk at a time. There is no
        train/test target switch — train_test_split keeps the design matrix on
        CPU and records row indices instead of copying out X_train/X_test.
        """
        self.device = device
        self.dtype = torch.float64
        self.coeffs = None
        self._unique_config_idxs_train_override = unique_config_idxs_train_override

        # Multi-GPU within a rank: list of local CUDA devices.
        if local_devices is None:
            self.local_devices = linalg.get_local_devices(device=device)
        else:
            self.local_devices = list(local_devices)
        _validate_local_devices(self.local_devices)

        # Auto-partition trigger: distributed + file paths for X and
        # config_idxs. Explicit override wins, otherwise auto-detect.
        if partitioned_override is not None:
            self._is_partitioned = bool(partitioned_override)
        else:
            self._is_partitioned = (
                linalg.is_distributed()
                and isinstance(X, str) and isinstance(config_idxs, str)
            )

        if self._is_partitioned and isinstance(X, str) and isinstance(config_idxs, str):
            self._load_partitioned(X, y, w, config_idxs, enrow_mask, intercept)
            return

        self.X = process_data(X, self.dtype, 'cpu')

        if intercept:
            ones = torch.ones((self.X.shape[0], 1), dtype=self.dtype).to(device='cpu')
            self.X = torch.hstack((ones, self.X))

        self.y = process_data(y, self.dtype, 'cpu')
        if self.y is not None: self.y = self.y.reshape(-1, 1)

        if w is None:
            self.w = torch.ones(self.X.shape[0], dtype=self.dtype).to(device='cpu')
        else:
            self.w = process_data(w, self.dtype, 'cpu')
        if self.w is not None: self.w = self.w.reshape(-1, 1)

        if enrow_mask is not None:
            self.enrow_mask = process_data(enrow_mask, torch.bool, 'cpu')
        else:
            self.enrow_mask = None

        if config_idxs is None:
            warnings.warn("config_idxs is None. No grouping.", UserWarning)
            self.config_idxs = torch.arange(self.X.shape[0]).to(dtype=torch.int64, device='cpu')
            self.unique_config_idxs = self.config_idxs
            if self.enrow_mask is None:
                self.enrow_mask = torch.ones_like(self.config_idxs, dtype=torch.bool).to('cpu')
        else:
            self.config_idxs = process_data(config_idxs, torch.int64, 'cpu')
            self.unique_config_idxs = torch.unique(self.config_idxs)
            if self.enrow_mask is None:
                warnings.warn("enrow_mask is None. Using first element as energy row.", UserWarning)
                self.enrow_mask = _synthesize_enrow_mask(self.config_idxs)

    def _load_partitioned(self, X_path, y_path, w_path, config_idxs_path,
                          enrow_mask_path, intercept):
        """Load only this rank's slice of data from .npy files via mmap."""
        # Partitioned mode keeps large data off any single GPU. The design
        # matrix stays on CPU per the device rule; no train target to set.
        world_size = linalg.get_world_size()
        rank = linalg.get_rank()

        # Rank 0 scans config boundaries, computes partition ranges, broadcasts.
        # NOTE: `dist.broadcast_object_list` pickles to CPU and uses the
        # backend's default device. With NCCL the caller must have run
        # `torch.cuda.set_device(...)` before init_process_group; the
        # benchmark launcher does this.
        if rank == 0:
            _, _, row_counts = _partition.scan_config_boundaries(config_idxs_path)
            ranges = _partition.compute_partition_ranges(row_counts, world_size)
        else:
            ranges = None
        obj_list = [ranges]
        dist.broadcast_object_list(obj_list, src=0)
        ranges = obj_list[0]
        start, end = ranges[rank]

        self.X = _partition.mmap_load_partition(X_path, start, end, dtype=self.dtype)
        if intercept:
            ones = torch.ones((self.X.shape[0], 1), dtype=self.dtype)
            self.X = torch.hstack((ones, self.X))

        self.y = (_partition.mmap_load_partition(y_path, start, end, dtype=self.dtype).reshape(-1, 1)
                  if y_path is not None else None)

        if w_path is None:
            self.w = torch.ones(self.X.shape[0], dtype=self.dtype).reshape(-1, 1)
        else:
            self.w = _partition.mmap_load_partition(w_path, start, end, dtype=self.dtype).reshape(-1, 1)

        self.config_idxs = _partition.mmap_load_partition(
            config_idxs_path, start, end, dtype=torch.int64)

        if enrow_mask_path is not None:
            self.enrow_mask = _partition.mmap_load_partition(
                enrow_mask_path, start, end, dtype=torch.bool)
        else:
            warnings.warn("enrow_mask is None. Using first element of each config as energy row.", UserWarning)
            self.enrow_mask = _synthesize_enrow_mask(self.config_idxs)

        # Build global config id list across ranks (sorted, unique).
        local_unique = torch.unique(self.config_idxs)
        self.unique_config_idxs = _partition.build_global_config_ids(local_unique)


    def train_test_split(self, test_fraction=0.0, seed=None, test_mask=None):
        self.test_fraction = test_fraction
        self.seed = seed

        if test_mask is None:
            if self.test_fraction == 0.0:
                self.test_mask = torch.zeros(self.config_idxs.shape, dtype=torch.bool)
            elif 0.0 < self.test_fraction < 1.0:
                if self.seed is not None:
                    torch.manual_seed(self.seed)

                # Use global unique_config_idxs so partitioned ranks produce
                # the same split decisions.
                num_test = int(len(self.unique_config_idxs) * self.test_fraction)
                perm = torch.randperm(len(self.unique_config_idxs))
                test_indices = self.unique_config_idxs[perm[:num_test]]

                self.test_mask = torch.isin(self.config_idxs, test_indices)
            else:
                raise ValueError("test_fraction must be between 0 and 1")
        else:
            self.test_mask = process_data(test_mask, torch.bool, 'cpu')

        self.train_mask = ~self.test_mask

        # Device rule: the design matrix stays on the host CPU. Record which
        # rows are train/test as index tensors instead of copying X_train /
        # X_test out of self.X — consumers stream self.X[train_idx] /
        # [test_idx] to the device in chunks. The per-row siblings (y, w,
        # enrow, config_idxs) are each 1/p the size of X, so they are kept
        # materialized on CPU; they index by chunk position, aligned with
        # train_idx / test_idx (which are in increasing row order).
        self.train_idx = self.train_mask.nonzero(as_tuple=True)[0]
        self.test_idx = self.test_mask.nonzero(as_tuple=True)[0]

        self.y_train = self.y[self.train_mask] if self.y is not None else None
        self.w_train = self.w[self.train_mask]
        self.enrow_mask_train = self.enrow_mask[self.train_mask] if self.enrow_mask is not None else None
        self.config_idxs_train = self.config_idxs[self.train_mask]

        self.y_test = self.y[self.test_mask] if self.y is not None else None
        self.w_test = self.w[self.test_mask]
        self.enrow_mask_test = self.enrow_mask[self.test_mask] if self.enrow_mask is not None else None
        self.config_idxs_test = self.config_idxs[self.test_mask]

        # Per-config tensors for sampling operations.
        if self._is_partitioned and linalg.is_distributed():
            # Global view assembled across ranks.
            local_train = torch.unique(self.config_idxs_train)
            local_test = torch.unique(self.config_idxs_test)
            self.unique_config_idxs_train = _partition.build_global_config_ids(local_train).to(self.device)
            self.unique_config_idxs_test = _partition.build_global_config_ids(local_test).to(self.device)
        else:
            self.unique_config_idxs_train = torch.unique(self.config_idxs_train).to(device=self.device)
            self.unique_config_idxs_test = torch.unique(self.config_idxs_test).to(device=self.device)

        # Explicit caller-supplied global config ids override the per-rank
        # view (used by nested samplers in CookSubSampler).
        if self._unique_config_idxs_train_override is not None:
            self.unique_config_idxs_train = self._unique_config_idxs_train_override.to(self.device)


    def train(self, method='auto', n_chunks=None):
        """Weighted Least Squares training.

        Args:
            method: 'auto' (default; single-pass lstsq when it fits one device,
                TSQR otherwise), 'lstsq', or 'qr'. The weighted matrix is never
                built on the host — chunks of self.X[train_idx] are gathered
                and scaled by w on the device.
            n_chunks: number of TSQR chunks (user-supplied; None = single pass
                per rank).
        """
        self.coeffs = linalg.solve_wls(
            self.X, self.y_train, self.w_train, x_idx=self.train_idx,
            method=method, device=self.device, n_chunks=n_chunks,
            partitioned=self._is_partitioned, local_devices=self.local_devices,
            dtype=self.dtype)


    def compute_errors(self, verbose=True, n_chunks=None):
        if self.coeffs is None:
            raise ValueError("Model not trained.")

        if n_chunks is None:
            n_chunks = getattr(self, 'n_chunks', None)
        is_rank0 = linalg.get_rank() == 0

        # Stream each split through the GPU in chunks so the full prediction
        # never materializes on one device (mirrors the train solver). The
        # squared-residual vectors come back on the data's device (CPU in
        # chunked/partitioned modes), aligned with the row masks below.
        train_sq = linalg.chunked_sq_residuals(
            self.X, self.y_train, self.coeffs, x_idx=self.train_idx,
            device=self.device, n_chunks=n_chunks, local_devices=self.local_devices)
        test_sq = linalg.chunked_sq_residuals(
            self.X, self.y_test, self.coeffs, x_idx=self.test_idx,
            device=self.device, n_chunks=n_chunks, local_devices=self.local_devices)

        def get_rmse(sq, mask, name):
            # distributed_rmse must be called by every rank the same number
            # of times (it is a collective), so always reduce even when this
            # rank's local contribution is empty.
            if mask is None or sq.shape[0] == 0:
                local_sq, local_n = 0.0, 0
            else:
                sel = sq.reshape(-1)[mask.to(sq.device).reshape(-1)]
                local_sq = float(sel.sum().item()) if sel.numel() > 0 else 0.0
                local_n = int(sel.numel())
            rmse = linalg.distributed_rmse(local_sq, local_n, device=self.device)
            if rmse is not None and verbose and is_rank0:
                print(f"{name} RMSE is {rmse}")
            return rmse

        en_train = self.enrow_mask_train
        en_test = self.enrow_mask_test
        e_train = get_rmse(train_sq, en_train, "Energy training")
        f_train = get_rmse(train_sq, ~en_train if en_train is not None else None, "Force training")
        e_test = get_rmse(test_sq, en_test, "Energy test")
        f_test = get_rmse(test_sq, ~en_test if en_test is not None else None, "Force test")

        return e_train, f_train, e_test, f_test