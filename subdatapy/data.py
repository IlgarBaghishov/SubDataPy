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
                 train_target_device=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None):
        """Args:
            train_target_device: Device to place X_train/y_train/w_train on
                after train_test_split. None -> self.device (single-process
                default); 'cpu' -> keep training data off GPU (used in
                chunked/distributed modes). Replaces the older pattern of
                setting a `_train_target_device` attribute before calling
                super().__init__().
            partitioned_override: Force `_is_partitioned` to this value
                instead of auto-detecting from (distributed AND file-path
                inputs). Used by nested samplers in CookSubSampler to
                inherit the parent's partition state when they receive
                tensors (not paths).
            unique_config_idxs_train_override: If provided, replaces the
                per-rank unique_config_idxs_train computed during
                train_test_split. Used by nested samplers to see the global
                config-id list rather than their partition-local view.
        """
        self.device = device
        self.dtype = torch.float64
        self.coeffs = None
        self._train_target_device = train_target_device
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
        # Partitioned mode exists to keep large data off any single GPU.
        # Default train target to CPU unless the caller set it explicitly.
        if self._train_target_device is None:
            self._train_target_device = 'cpu'

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

        # Training data target (explicit kwarg at construction time). None
        # => the primary device; 'cpu' => keep training data off GPU, used
        # by chunked/distributed modes where X doesn't fit on one GPU.
        train_target = self._train_target_device if self._train_target_device is not None else self.device

        self.X_train = self.X[self.train_mask].to(train_target)
        self.y_train = self.y[self.train_mask].to(train_target) if self.y is not None else None
        self.w_train = self.w[self.train_mask].to(train_target)
        self.enrow_mask_train = self.enrow_mask[self.train_mask].to(train_target) if self.enrow_mask is not None else None
        self.config_idxs_train = self.config_idxs[self.train_mask].to(device=train_target)

        # Test data always goes to GPU
        self.X_test = self.X[self.test_mask].to(self.device)
        self.y_test = self.y[self.test_mask].to(self.device) if self.y is not None else None
        self.w_test = self.w[self.test_mask].to(self.device)
        self.enrow_mask_test = self.enrow_mask[self.test_mask].to(self.device) if self.enrow_mask is not None else None
        self.config_idxs_test = self.config_idxs[self.test_mask].to(device=self.device)

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


    def train(self, method='lstsq', n_chunks=None):
        """Weighted Least Squares training.

        Args:
            method: 'lstsq' (default, torch.linalg.lstsq) or 'qr' (via TSQR)
            n_chunks: For 'qr' method, number of chunks. None = auto.
        """
        A = self.X_train.clone()
        A.mul_(self.w_train)

        B = self.y_train.clone()
        B.mul_(self.w_train)
        B = B.reshape(-1, 1)

        self.coeffs = linalg.solve_wls(
            A, B, method=method, device=self.device, n_chunks=n_chunks,
            partitioned=self._is_partitioned, local_devices=self.local_devices,
            dtype=self.dtype)
        del A, B


    def compute_errors(self, verbose=True):
        if self.coeffs is None:
            raise ValueError("Model not trained.")

        is_rank0 = linalg.get_rank() == 0

        def get_rmse(X_data, y_true, mask, name):
            mask_local = mask.to(X_data.device) if mask is not None else None
            if X_data.shape[0] == 0 or (mask_local is not None and mask_local.sum() == 0):
                local_sq, local_n = 0.0, 0
            else:
                coeffs = self.coeffs.to(X_data.device)
                y_preds = X_data[mask_local] @ coeffs
                sq_res = torch.square(y_preds - y_true[mask_local])
                local_sq = float(sq_res.sum().item())
                local_n = int(sq_res.numel())

            rmse = linalg.distributed_rmse(local_sq, local_n, device=self.device)
            if rmse is not None and verbose and is_rank0:
                print(f"{name} RMSE is {rmse}")
            return rmse

        e_train = get_rmse(self.X_train, self.y_train, self.enrow_mask_train, "Energy training")
        f_train = get_rmse(self.X_train, self.y_train, ~self.enrow_mask_train, "Force training")

        e_test = get_rmse(self.X_test, self.y_test, self.enrow_mask_test, "Energy test")
        f_test = get_rmse(self.X_test, self.y_test, ~self.enrow_mask_test, "Force test")

        return e_train, f_train, e_test, f_test