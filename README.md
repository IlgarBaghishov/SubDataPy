# SubDataPy

**SubDataPy** is a GPU-accelerated Python toolkit for subsampling large datasets, designed for Machine Learning Interatomic Potentials (MLIPs). It implements statistical subsampling methods — random, leverage score, and Cook's distance — that operate on grouped rows (atomic configurations with energy + force rows).

Subsampling has been shown to not only increase training speed by reducing dataset size, but also potentially increase testing accuracy by removing redundant data.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Data Loading](#data-loading)
- [Subsampling Methods](#subsampling-methods)
  - [Random Subsampling](#random-subsampling)
  - [Leverage Score Subsampling](#leverage-score-subsampling)
  - [Cook's Distance Subsampling](#cooks-distance-subsampling)
- [Training and Error Evaluation](#training-and-error-evaluation)
- [Batch Evaluation](#batch-evaluation)
- [Multi-GPU and Distributed Usage](#multi-gpu-and-distributed-usage)
- [Generating Descriptor Matrices](#generating-descriptor-matrices)
- [API Reference](#api-reference)
  - [BaseData](#basedata)
  - [RandomSubSampler](#randomsubsampler)
  - [LeverageSubSampler](#leveragesubsampler)
  - [CookSubSampler](#cooksubsampler)
  - [linalg Module](#linalg-module)
- [Examples](#examples)
- [Testing](#testing)
- [Mathematical Reference](#mathematical-reference)
- [Citation](#citation)

---

## Key Features

- **Three subsampling strategies:** Random, Leverage Score, Cook's Distance (one-step and stepwise)
- **Configuration-aware:** Treats groups of rows (1 energy + N force rows per atomic configuration) as atomic units — never splits a configuration
- **Block mode:** For leverage and Cook's, aggregate scores across all rows of a configuration rather than using only the energy row
- **GPU-accelerated:** All linear algebra (SVD, QR, least squares, Woodbury updates) runs on CUDA GPUs via PyTorch
- **Distributed TSQR:** Multi-GPU support via `torch.distributed` (NCCL backend) for datasets that don't fit in GPU memory
- **Flexible data input:** Accepts `.npy` file paths, NumPy arrays, Pandas DataFrames, or PyTorch tensors
- **Batch evaluation:** `create_subsample_errors_dataframe()` sweeps over multiple subsample fractions with repeats and returns a structured Pandas DataFrame

---

## Installation

```bash
git clone https://github.com/IlgarBaghishov/SubDataPy.git
cd SubDataPy

# Install with test dependencies
pip install -e ".[test]"

# Install runtime only
pip install -e .
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.8 (with CUDA support recommended)
- NumPy >= 1.20
- Pandas >= 1.0

---

## Quick Start

```python
import numpy as np
from subdatapy.subsampler import RandomSubSampler, LeverageSubSampler, CookSubSampler

# Load data (or pass file paths directly)
X = np.load("X.npy")            # (n_rows, n_features) descriptor matrix
y = np.load("y.npy")            # (n_rows,) targets (energies + forces)
w = np.load("w.npy")            # (n_rows,) weights
config_idxs = np.load("config_idxs.npy")  # (n_rows,) integer config IDs

# Create a random subsample at 50% of configurations
rs = RandomSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    test_fraction=0.2,
    seed=42,
    device='cuda',
)
rs.create_subsample(subsample_fraction=0.5, seed=123)
rs.train_subsample()
errors = rs.compute_subsample_errors(verbose=True)
# Returns: (sub_e_train, sub_f_train, e_train, f_train, e_test, f_test)
```

---

## Core Concepts

### Configurations and Grouping

In MLIP training, each atomic configuration produces **one energy row** and **N force rows** (where N = 3 * number of atoms). SubDataPy always selects or removes entire configurations, never individual rows.

- **`config_idxs`**: An integer array of length `n_rows` mapping each row to its configuration ID. All rows with the same ID belong to the same configuration.
- **`enrow_mask`**: A boolean array of length `n_rows` where `True` marks energy rows. If not provided, SubDataPy uses the first row of each configuration as the energy row.

### Block vs Non-Block Mode

- **Non-block mode** (`block=False`, default): Leverage scores and Cook's distances are computed using only the energy row of each configuration. Faster, good for initial exploration.
- **Block mode** (`block=True`): Scores are computed using all rows (energy + forces) of each configuration as a block. More statistically principled but more expensive.

### Weighted Least Squares

All fitting uses Weighted Least Squares (WLS). The weight vector `w` allows different importance for energy vs force rows (e.g., `w_energy=100, w_force=1`). Internally, the design matrix and targets are pre-multiplied by weights before solving.

### Data Flow

1. Data loads to **CPU** (from `.npy` files, NumPy arrays, or DataFrames)
2. An intercept column of ones is optionally prepended to X (`intercept=True`)
3. Train/test split moves subsets to **GPU** (`device='cuda'`)
4. All linear algebra runs on GPU in `torch.float64` (double precision)

---

## Data Loading

SubDataPy accepts multiple input formats for `X`, `y`, `w`, `config_idxs`, and `enrow_mask`:

```python
# File paths (.npy)
rs = RandomSubSampler(X="path/to/X.npy", y="path/to/y.npy", ...)

# NumPy arrays
rs = RandomSubSampler(X=np_array, y=np_array, ...)

# Pandas DataFrames
rs = RandomSubSampler(X=df_features, y=df_targets, ...)

# PyTorch tensors
rs = RandomSubSampler(X=torch_tensor, y=torch_tensor, ...)
```

All inputs are converted to `torch.float64` tensors internally.

---

## Subsampling Methods

### Random Subsampling

Uniformly random selection of configurations. Serves as a baseline.

```python
from subdatapy.subsampler import RandomSubSampler

rs = RandomSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    enrow_mask=enrow_mask,       # optional, auto-detected if None
    test_fraction=0.2,           # fraction of configs for test set
    seed=42,                     # seed for train/test split
    intercept=True,              # prepend intercept column (default)
    device='cuda',               # 'cuda' or 'cpu'
)

# Create subsample: keep 50% of training configurations
mask = rs.create_subsample(subsample_fraction=0.5, seed=123)
# mask is a boolean tensor over ALL rows (train + test)

# Train on the subsample
rs.train_subsample()

# Evaluate errors
e_sub, f_sub, e_train, f_train, e_test, f_test = rs.compute_subsample_errors(verbose=True)
```

### Leverage Score Subsampling

Samples configurations proportional to their statistical leverage scores. Configurations with higher leverage (more influential on the fit) are more likely to be selected.

```python
from subdatapy.subsampler import LeverageSubSampler

ls = LeverageSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    test_fraction=0.2,
    seed=42,
    device='cuda',

    # Leverage-specific parameters:
    block=False,              # True: sum leverage over all rows per config
                              # False: use energy row leverage only

    # Factorization method:
    factorization='auto',     # 'auto', 'svd', or 'qr'
                              # auto: SVD for single-GPU, QR if chunked/distributed
    n_chunks=None,            # number of TSQR chunks (for large matrices)

    # Pre-computed SVD factors (optional, skip SVD computation):
    U=None, S=None, Vh=None,
)

ls.create_subsample(subsample_fraction=0.3, seed=123)
ls.train_subsample()
errors = ls.compute_subsample_errors()

# Access leverage scores after create_subsample:
print(ls.leverage_scores)  # (n_configs,) tensor
```

**How it works:**
1. Compute weighted X: `X_w = diag(w) @ X`
2. Compute SVD: `U, S, Vh = svd(X_w)` (or QR factorization)
3. Row leverage scores: `h_i = sum(U[i,:]^2)` (or `h_i = ||R^{-T} x_i||^2` for QR)
4. Block mode: sum leverage scores within each configuration
5. Non-block mode: use the energy row's leverage score
6. Sample configurations proportional to normalized leverage scores

### Cook's Distance Subsampling

Selects configurations by their influence on the model fit, measured by Cook's distance. Supports two approaches:

#### One-Step Cook's Distance

Computes Cook's distance for each configuration in a single pass, then samples proportional to scores (or selects top-k).

```python
from subdatapy.subsampler import CookSubSampler

# One-step (non-block only)
cs = CookSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    test_fraction=0.2,
    seed=42,
    device='cuda',

    stepwise=False,           # one-step mode
    sampling=True,            # True: sample proportional to Cook's scores
                              # False: select top-k configurations
)

cs.create_subsample(subsample_fraction=0.5, seed=123)
cs.train_subsample()
errors = cs.compute_subsample_errors()

# Access Cook's distance scores:
print(cs.onestep_en_cooks)  # (n_configs,) tensor
```

**One-step formula** (energy rows only):

```
D_i = e_i^2 * h_i / (1 - h_i)^2
```

where `e_i` is the residual and `h_i` is the leverage score of the i-th energy row.

#### Stepwise Cook's Distance

Greedy iterative method: starts from a small initial subset, then adds (ascending) or removes (descending) one configuration per iteration based on which has the highest Cook's distance influence on the current model.

```python
# Stepwise ascending with random initial subset
cs = CookSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    test_fraction=0.2,
    seed=42,
    device='cuda',

    stepwise=True,            # stepwise mode
    ascending=True,           # True: grow subset from small initial
                              # False: shrink from large initial

    # Initial subset:
    initial_subsampler="random",    # "random" or "leverage"
    initial_subsample_fraction=0.05, # start with 5% of configs

    # Block mode:
    block=False,              # True: block Cook's (all rows per config)
                              # False: energy-row Cook's only

    # Factorization for initial (X'X)^{-1}:
    factorization='auto',     # 'auto', 'svd', or 'qr'
                              # auto: SVD single-process, QR if chunked/distributed

    # Update method per iteration:
    update_method='auto',     # 'auto', 'woodbury', or 'qr'
                              # auto: QR for block (stable), Woodbury for non-block (fast)

    # Chunked/distributed:
    n_chunks=None,            # TSQR chunks (for large data, or multi-GPU)
    tree_reduction_threshold=10,  # reduce R matrices after this many
)

cs.create_subsample(subsample_fraction=0.5, seed=123)
cs.train_subsample()
errors = cs.compute_subsample_errors()
```

**Stepwise algorithm:**
1. Create initial subset (random or leverage-based, at `initial_subsample_fraction`)
2. Compute initial factorization: `(X'X)^{-1}` via SVD or QR
3. For each iteration until target fraction is reached:
   - Compute coefficients: `beta = (X'X)^{-1} X'y`
   - Compute Cook's distance for each candidate configuration
   - Select the configuration with highest (ascending) or lowest (descending) Cook's distance
   - Update `(X'X)^{-1}` incrementally via Woodbury identity or QR re-factorization
4. Return the final subset mask

**Non-block Cook's formula** (per energy row):
```
D_i = e_i^2 * h_i / (1 + h_i)
```

**Block Cook's formula** (per configuration group c):
```
D_c = r_c' @ inv(I +/- H_c) @ H_c @ r_c
```
where `H_c = X_c @ (X'X)^{-1} @ X_c'` is the block leverage matrix and `r_c = X_c @ beta - y_c`.

---

## Training and Error Evaluation

### Full Training (No Subsampling)

```python
from subdatapy.data import BaseData

bd = BaseData(
    X, y=y, w=w,
    config_idxs=config_idxs,
    enrow_mask=enrow_mask,
    intercept=True,
    device='cuda',
)
bd.train_test_split(test_fraction=0.2, seed=42)

# Standard least squares
bd.train()                    # method='lstsq' (default)

# Or QR-based (supports chunking for large matrices)
bd.train(method='qr', n_chunks=4)

# Evaluate
e_train, f_train, e_test, f_test = bd.compute_errors(verbose=True)

# Access coefficients
print(bd.coeffs.shape)  # (n_features + 1, 1) if intercept=True
```

### Subsample Training

After creating a subsample with any subsampler:

```python
# Train on the subsample
subsampler.train_subsample()

# Evaluate errors on subsampled training, full training, and test sets
e_sub, f_sub, e_train, f_train, e_test, f_test = subsampler.compute_subsample_errors(verbose=True)
```

The 6 returned RMSE values are:
1. **Subsampled Training Energy RMSE** — energy rows in the subsample
2. **Subsampled Training Force RMSE** — force rows in the subsample
3. **Entire Training Energy RMSE** — all training energy rows
4. **Entire Training Force RMSE** — all training force rows
5. **Testing Energy RMSE** — test set energy rows
6. **Testing Force RMSE** — test set force rows

---

## Batch Evaluation

Sweep over multiple subsample fractions with repeated trials in a single call:

```python
errors_df = subsampler.create_subsample_errors_dataframe(
    subsample_fractions_list=[0.05, 0.1, 0.2, 0.5],
    repeat_count_list=5,        # 5 repeats per fraction (int applies to all)
    seed=42,
    verbose=False,
)

# errors_df is a Pandas DataFrame with MultiIndex columns:
#   Level 0: Error Type (e.g., "Testing Energy RMSE")
#   Level 1: Subsample Fraction (e.g., 0.05, 0.1, 0.2, 0.5)
# Rows are repeats.

print(errors_df["Testing Energy RMSE"])
```

For stepwise Cook's distance, each successive fraction in `subsample_fractions_list` reuses the previous subset as initial state (the greedy loop continues rather than restarting), allowing efficient evaluation of the subsampling curve.

**Variable repeats per fraction:**

```python
# More repeats for small fractions (high variance), fewer for large fractions
errors_df = subsampler.create_subsample_errors_dataframe(
    subsample_fractions_list=[0.05, 0.1, 0.2, 0.5],
    repeat_count_list=[10, 10, 5, 5],  # must be non-increasing
    seed=42,
)
```

---

## Multi-GPU and Distributed Usage

For datasets that exceed single-GPU memory, SubDataPy supports chunked processing and multi-GPU distribution via `torch.distributed` with the NCCL backend.

### Single-GPU Chunking

Stream data through GPU in chunks (no distributed setup needed):

```python
cs = CookSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    test_fraction=0.2, seed=42,
    device='cuda',
    stepwise=True, ascending=True,
    factorization='qr',       # must use QR for chunking
    n_chunks=8,               # process in 8 sequential chunks
)
cs.create_subsample(subsample_fraction=0.5, seed=123)
```

This keeps only one chunk on GPU at a time, reducing peak memory from O(n*p) to O(n/8 * p).

### Multi-GPU (torchrun)

```bash
# 4 GPUs on one node
torchrun --nproc_per_node=4 my_script.py

# Two nodes, one rank per node, each rank using all 4 local GPUs via
# the `local_devices` parameter (see Partitioned Mode below)
srun -N 2 --ntasks-per-node=1 --gpus-per-task=4 \
    torchrun --nnodes=2 --nproc_per_node=1 my_script.py
```

```python
# my_script.py
import torch.distributed as dist
from subdatapy.subsampler import CookSubSampler

dist.init_process_group(backend='nccl')

# Under torchrun, SubDataPy requires partitioned mode — pass .npy file
# paths for X and config_idxs so BaseData auto-enables it. Passing
# in-memory tensors while distributed raises ValueError.
cs = CookSubSampler(
    X="data/X.npy", y="data/y.npy", w="data/w.npy",
    config_idxs="data/config_idxs.npy",
    test_fraction=0.2, seed=42,
    device='cuda',
    stepwise=True, ascending=True,
    factorization='qr',
    n_chunks=16,               # 16 total chunks, 4 per GPU
)
cs.create_subsample(subsample_fraction=0.5, seed=123)
cs.train_subsample()
errors = cs.compute_subsample_errors(verbose=True)  # all ranks participate

dist.destroy_process_group()
```

### Partitioned Mode (OOM-free distributed)

When `BaseData` receives `.npy` file paths for both `X` and `config_idxs` under `torch.distributed`, it *auto-enters* partitioned mode:

1. Rank 0 scans the `config_idxs` file and computes partition boundaries at configuration boundaries (never mid-config) via `partition.compute_partition_ranges`. The ranges are broadcast to all ranks.
2. Each rank `mmap`-loads only its slice of `X`, `y`, `w`, `config_idxs`, `enrow_mask` — memory is `D/world_size` per rank.
3. Train/test split uses the global config-id list so every rank agrees on which configs are test. Subsamplers use a global `unique_config_idxs_train` (built via `partition.build_global_config_ids`) so sampling with the same seed yields the same chosen configs across ranks.
4. TSQR runs in `partitioned=True` mode: each rank QRs its local rows; rank 0 gathers per-rank R matrices and does a final QR; R is broadcast back to all ranks for downstream use. Only small tensors (p×p R, p×1 XTy, and the single winning config's rows per stepwise iteration) travel across NCCL.

```python
# Target: 2 nodes × 1 rank × 4 local GPUs = 8 GPUs, 2 CPU-RAM partitions.
# Each rank sees all 4 local GPUs and round-robins chunks across them.
cs = CookSubSampler(
    X="data/X.npy", y="data/y.npy", w="data/w.npy",
    config_idxs="data/config_idxs.npy",
    enrow_mask="data/enrow_mask.npy",
    device='cuda:0',
    local_devices=['cuda:0','cuda:1','cuda:2','cuda:3'],  # round-robin TSQR chunks
    stepwise=True, ascending=True,
    factorization='qr',
)
```

**`local_devices`** is auto-detected from `torch.cuda.device_count()` if not supplied. When `LOCAL_WORLD_SIZE==1` (one rank per node), it defaults to every visible GPU; when multiple ranks share a node, it defaults to `[device]` and each rank gets one GPU.

**`_is_partitioned`** is the flag set by the auto-detection. You can force it via the `partitioned_override=` kwarg (used internally by `CookSubSampler` to inherit the parent's partition state in nested samplers).

Partitioned mode requires `world_size > 1`. Calling `tsqr_r(partitioned=True)` under a single process raises `ValueError`.

### TSQR Execution Modes

| `n_chunks` | Setup | Mode | Description |
|:---:|---|---|---|
| None | single-process, one `local_devices` | Single-pass | `QR(X)` directly on the device |
| N | single-process | Chunked | Stream N chunks through the primary device (or round-robin across `local_devices`) |
| None / per-rank N | partitioned multi-rank (torchrun + file paths) | Partitioned TSQR | Each rank QRs its local partition; rank 0 gathers and reduces across ranks |

Only two NCCL collectives are needed: `gather` (R matrices) and `reduce` (X'y sum). All communication is GPU-to-GPU. Replicated-distributed mode — where every rank holds the full matrix and TSQR merely splits the row range — was removed because partitioned is strictly more memory-efficient.

### Distributed Leverage Scores

```python
from subdatapy.subsampler import LeverageSubSampler

ls = LeverageSubSampler(
    X, y=y, w=w,
    config_idxs=config_idxs,
    test_fraction=0.2, seed=42,
    device='cuda',
    factorization='qr',
    n_chunks=8,
)
ls.create_subsample(subsample_fraction=0.3, seed=123)
```

### Distributed Full Training

```python
from subdatapy.data import BaseData

bd = BaseData(X, y=y, w=w, config_idxs=config_idxs, device='cuda')
bd.train_test_split(test_fraction=0.2, seed=42)
bd.train(method='qr', n_chunks=8)
```

---

## Generating Descriptor Matrices

SubDataPy works with pre-computed descriptor matrices. For MLIP workflows, you typically generate these using a descriptor calculator like [FitSNAP](https://github.com/FitSNAP/FitSNAP).

An example FitSNAP helper script is provided in `tools/generate_matrices/FitSNAP/qSNAP.py`. It:

1. Reads atomic structures and DFT energies/forces from HDF5 files
2. Computes SNAP bispectrum descriptors via FitSNAP
3. Saves `X.npy`, `y.npy`, `w.npy`, `config_idxs.npy` for use with SubDataPy

For benchmarking with synthetic data, see `examples/mpi_benchmark/generate_data.py`.

---

## API Reference

### BaseData

Base class for data handling. All subsamplers inherit from this.

```python
from subdatapy.data import BaseData

bd = BaseData(X, y=None, w=None, config_idxs=None, enrow_mask=None,
              intercept=True, device='cuda')
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array-like or str | required | Feature matrix (n_rows, n_features). `.npy` path, NumPy array, DataFrame, or tensor. |
| `y` | array-like or str | `None` | Target vector (n_rows,). Same formats as X. |
| `w` | array-like or str | `None` | Weight vector (n_rows,). Defaults to all ones. |
| `config_idxs` | array-like or str | `None` | Configuration ID per row (n_rows,). If None, each row is its own config. |
| `enrow_mask` | array-like or str | `None` | Boolean mask for energy rows (n_rows,). If None, first row per config is energy. |
| `intercept` | bool | `True` | Prepend a column of ones to X. |
| `device` | str | `'cuda'` | PyTorch device for computation. |

**Methods:**

| Method | Description |
|---|---|
| `train_test_split(test_fraction, seed, test_mask)` | Split data into train/test sets by configuration. |
| `train(method='lstsq', n_chunks=None)` | Weighted least squares. `method='qr'` uses TSQR. |
| `compute_errors(verbose=True)` | Returns `(e_train, f_train, e_test, f_test)` RMSE tuple. |

---

### RandomSubSampler

```python
from subdatapy.subsampler import RandomSubSampler

rs = RandomSubSampler(X, y=None, w=None, test_fraction=0.0, seed=None,
                      test_mask=None, config_idxs=None, enrow_mask=None,
                      intercept=True, device='cuda')
```

Inherits all BaseData parameters plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `test_fraction` | float | `0.0` | Fraction of configurations for the test set. |
| `seed` | int | `None` | Random seed for train/test split. |
| `test_mask` | array-like | `None` | Explicit boolean test mask (overrides `test_fraction`). |

**Methods:**

| Method | Description |
|---|---|
| `create_subsample(subsample_fraction, seed=None)` | Create a subsample. Returns a boolean mask over all rows. |
| `train_subsample()` | Train WLS on the subsampled data. |
| `compute_subsample_errors(verbose=False)` | Returns 6-tuple of RMSE values (sub-train, full-train, test for energy + force). |
| `create_subsample_errors_dataframe(subsample_fractions_list, repeat_count_list=1, seed=None, verbose=False)` | Sweep fractions with repeats. Returns MultiIndex DataFrame. |

---

### LeverageSubSampler

```python
from subdatapy.subsampler import LeverageSubSampler

ls = LeverageSubSampler(X, y=None, w=None, test_fraction=0.0, seed=None,
                        config_idxs=None, enrow_mask=None, intercept=True,
                        device='cuda', block=False, U=None, S=None, Vh=None,
                        factorization='auto', n_chunks=None)
```

Inherits all RandomSubSampler parameters plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `block` | bool | `False` | Sum leverage scores across all rows per configuration. |
| `U, S, Vh` | tensors | `None` | Pre-computed SVD factors to skip SVD computation. |
| `factorization` | str | `'auto'` | `'auto'`, `'svd'`, or `'qr'`. Auto selects SVD unless chunked/distributed. |
| `n_chunks` | int | `None` | Number of TSQR chunks for QR factorization. |

**Attributes set after `create_subsample()`:**

| Attribute | Description |
|---|---|
| `leverage_scores` | Per-configuration leverage scores (n_configs,) tensor. |

---

### CookSubSampler

```python
from subdatapy.subsampler import CookSubSampler

cs = CookSubSampler(X, y, w=None, test_fraction=0.0, seed=None, test_mask=None,
                    config_idxs=None, enrow_mask=None, intercept=True,
                    device='cuda', block=False, stepwise=False, sampling=True,
                    ascending=True, initial_subsampler="random",
                    initial_subsample_fraction=1, U=None, S=None, Vh=None,
                    n_chunks=None, factorization='auto', update_method='auto',
                    tree_reduction_threshold=10)
```

Inherits all RandomSubSampler parameters plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stepwise` | bool | `False` | Stepwise greedy selection (True) or one-step scoring (False). |
| `block` | bool | `False` | Block Cook's distance using all rows per config. |
| `sampling` | bool | `True` | One-step only: sample proportional to Cook's (True) or top-k (False). |
| `ascending` | bool | `True` | Stepwise only: grow subset (True) or shrink (False). |
| `initial_subsampler` | str | `"random"` | Stepwise only: `"random"` or `"leverage"` initial subset. |
| `initial_subsample_fraction` | float | `1` | Stepwise only: fraction for the initial subset. |
| `U, S, Vh` | tensors | `None` | Pre-computed SVD factors (one-step mode). |
| `factorization` | str | `'auto'` | `'auto'`, `'svd'`, or `'qr'`. For initial (X'X)^{-1}. |
| `update_method` | str | `'auto'` | `'auto'`, `'woodbury'`, or `'qr'`. Per-iteration update. Auto = QR for block, Woodbury for non-block. |
| `n_chunks` | int | `None` | TSQR chunks for initial factorization and chunked processing. |
| `tree_reduction_threshold` | int | `10` | Reduce accumulated R matrices after this many chunks. |

**Attributes set after `create_subsample()`:**

| Attribute | Description |
|---|---|
| `onestep_en_cooks` | One-step Cook's scores (n_configs,). Only set if `stepwise=False`. |
| `XTX_inv` | Current (X'X)^{-1} at end of stepwise. |
| `XTy` | Current X'y at end of stepwise. |
| `R_final` | QR R factor (only when using QR factorization). |

---

### linalg Module

Pure functions for distributed linear algebra. Used internally by subsamplers but available for direct use.

```python
from subdatapy import linalg
```

**TSQR Functions:**

| Function | Description |
|---|---|
| `tsqr_r(X, device, n_chunks, tree_reduction_threshold)` | Compute R factor of QR(X). Returns R on rank 0, None on others. |
| `tsqr_r_xty(X, y, device, n_chunks, tree_reduction_threshold)` | Compute R and X'y simultaneously. Returns (R, X'y) on rank 0. |

**Inverse/Solve Functions:**

| Function | Description |
|---|---|
| `xtx_inv_from_r(R, device)` | (X'X)^{-1} = R^{-1} R^{-T}. Inversion on CPU in float64. |
| `xtx_inv_from_svd(X, device)` | (X'X)^{-1} via SVD with singular value filtering. Returns (inv, U, S, Vh). |
| `solve_from_r_xty(R, XTy)` | Solve beta = R^{-1} R^{-T} X'y via two triangular solves. |

**Update Functions:**

| Function | Description |
|---|---|
| `woodbury_update(XTX_inv, X_change, y_change, XTy, ascending)` | Rank-k Woodbury/Sherman-Morrison update for adding or removing rows. |
| `qr_update_add(R, X_new, y_new, XTy, device)` | QR-based update: append rows and re-factorize. Returns (R_new, inv_new, XTy_new). |

**Leverage:**

| Function | Description |
|---|---|
| `leverage_scores_from_r(X, R, device)` | h_i = \|\|R^{-T} x_i\|\|^2. QR-based leverage scores. |

**Distributed Helpers:**

| Function | Description |
|---|---|
| `get_rank()` | Current process rank (0 if not distributed). |
| `get_world_size()` | World size (1 if not distributed). |
| `is_distributed()` | True if torch.distributed is active with world_size > 1. |

---

## Examples

### Jupyter Notebooks

- `examples/Be_qSNAP_8/example1.ipynb` — Beryllium qSNAP descriptor subsampling
- `examples/lithium/example1.ipynb` — Lithium dataset subsampling

### Benchmark Script

The `examples/mpi_benchmark/` directory contains a comprehensive benchmark:

```bash
# Generate synthetic data (2000 configs, 200 features)
python examples/mpi_benchmark/generate_data.py

# Run single-GPU benchmarks
python examples/mpi_benchmark/benchmark.py --method random --subsample_fraction 0.5
python examples/mpi_benchmark/benchmark.py --method leverage --block
python examples/mpi_benchmark/benchmark.py --method cooks --stepwise --ascending

# Run distributed benchmark
torchrun --nproc_per_node=4 examples/mpi_benchmark/benchmark.py \
    --method cooks --stepwise --ascending --factorization qr

# Run the full benchmark suite (NERSC Perlmutter)
sbatch examples/mpi_benchmark/run_all.sh
```

---

## Testing

```bash
# Run all tests
pytest

# With verbose output
pytest -v

# With coverage
pytest --cov=subdatapy --cov-report=term-missing

# Run only linalg tests
pytest tests/test_linalg.py -v

# Run only subsampler tests
pytest tests/test_subsamplers.py -v
```

Tests require a CUDA GPU by default. On CPU-only systems, tests run with `device='cpu'` (different expected RMSE values are used).

---

## Mathematical Reference

For detailed derivations, equations, and code cross-references for all algorithms implemented in SubDataPy, see [MATH.md](MATH.md).

---

## Citation

If you use SubDataPy in your research, please cite:

```
@software{subdatapy,
  author = {Ilgar Baghishov},
  title = {SubDataPy: A Python Toolkit for Subsampling Large Datasets for MLIPs},
  url = {https://github.com/IlgarBaghishov/SubDataPy},
}
```
