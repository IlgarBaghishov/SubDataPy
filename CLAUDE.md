# CLAUDE.md - SubDataPy

## Project Overview

SubDataPy is a GPU-accelerated Python toolkit for subsampling large datasets for Machine Learning Interatomic Potentials (MLIPs). It implements statistical subsampling methods (random, leverage score, Cook's distance) that operate on grouped rows (atomic configurations with energy + force rows). See `README.md` for full user documentation.

## Tech Stack

- **Language:** Python 3.11+
- **Core dependency:** PyTorch (>=2.8) for all linear algebra (SVD, QR, least squares) — runs on GPU (CUDA) by default
- **Other deps:** NumPy (>=1.20), Pandas (>=1.0)
- **Distributed:** `torch.distributed` with NCCL backend for multi-GPU TSQR (no mpi4py dependency)
- **Build:** setuptools via `pyproject.toml`
- **Tests:** pytest with pytest-cov

## Project Structure

```
subdatapy/
  __init__.py              # Exports BaseData
  data.py                  # BaseData class: data loading, train/test split, WLS training
  linalg.py                # Reusable TSQR + distributed linear algebra (pure functions)
  partition.py             # Config-boundary partitioning + mmap loader for distributed mode
  subsampler/
    __init__.py            # Exports: RandomSubSampler, LeverageSubSampler, CookSubSampler
    random.py              # RandomSubSampler (base for all subsamplers)
    leverage.py            # LeverageSubSampler (SVD or QR-based leverage scores)
    cooks.py               # CookSubSampler (one-step, stepwise, chunked, distributed)
MATH.md                    # Mathematical reference: equations, derivations, code cross-references
tests/
  conftest.py              # pytest fixtures, loads .npy test data
  test_subsamplers.py      # Tests for Random, Leverage, Cook's subsamplers
  test_linalg.py           # Tests for linalg module (TSQR, inverses, updates, leverage)
  test_partition.py        # Tests for partition.py utilities
  test_distributed_regressions.py  # Regressions for the distributed/partitioned code paths
  test_data/               # .npy files (X, y, w, config_idxs)
tools/
  generate_matrices/FitSNAP/qSNAP.py  # FitSNAP helper to generate descriptor matrices
examples/
  Be_qSNAP_8/             # Beryllium qSNAP Jupyter notebook
  lithium/                 # Lithium Jupyter notebook
  mpi_benchmark/           # Benchmark script with torchrun support
```

## Class Hierarchy

```
BaseData (data.py)
  └── RandomSubSampler (random.py)        — uniform random config selection
        ├── LeverageSubSampler (leverage.py) — weighted sampling by leverage scores (SVD or QR)
        └── CookSubSampler (cooks.py)        — Cook's distance (one-step, stepwise, distributed)
```

All subsamplers inherit from `RandomSubSampler`, which inherits from `BaseData`.

## Key Concepts

- **config_idxs:** Integer array mapping each row to a configuration ID. Subsampling selects/removes entire configurations (energy + force rows together).
- **enrow_mask:** Boolean mask identifying energy rows (one per config). Force rows are `~enrow_mask`.
- **block mode:** Treats all rows of a config as a block for leverage/Cook's calculations.
- **subsample_fraction:** Fraction of unique configurations to keep.
- **Data flow:** Data loads to CPU, then moves to GPU (`device='cuda'`) for train/test splits and computation. Uses `torch.float64` throughout.
- **Distributed data strategy:** Under `torch.distributed` the only supported mode is **partitioned** — each rank `mmap`-loads only its slice. `BaseData` auto-enters it when given `.npy` *file paths* for both `X` and `config_idxs` under `torchrun`; `partition.py` cuts rows at configuration boundaries (no config is ever split across ranks), and the TSQR gather is done on per-rank local R blocks. Memory scales as `D/world_size` per rank. Passing in-memory tensors under `torchrun` raises `ValueError` — run single-process, or serialize to `.npy` first.
- **local_devices:** Optional list of CUDA device strings passed to `BaseData` (e.g. `['cuda:0','cuda:1','cuda:2','cuda:3']`) that allows one rank to round-robin TSQR/leverage chunks across multiple local GPUs. Auto-detected via `linalg.get_local_devices()`.
- **`_is_partitioned`:** Flag on `BaseData` set by auto-detection or by the `partitioned_override=` kwarg. Subsamplers pass `unique_config_idxs_train_override=` to their nested samplers so every rank picks identical configs.

## linalg.py Module

Reusable pure functions for distributed linear algebra. No mutable state. Mode detection is automatic based on `n_chunks` and `torch.distributed` state.

**TSQR:** `tsqr_r()`, `tsqr_r_xty()` — three supported modes: single-pass (single-process, no chunks, single device), chunked single-process (round-robin across `local_devices`), and partitioned multi-rank (each rank QRs its local partition, rank 0 gathers and reduces). Replicated-distributed mode was removed — partitioned is strictly more memory-efficient and has unambiguous parameter semantics.

**Inverses:** `xtx_inv_from_r()` (QR-based, CPU float64 for stability), `xtx_inv_from_svd()` (SVD with singular value filtering).

**Updates:** `woodbury_update()` (rank-k Sherman-Morrison), `qr_update_add()` (append rows, re-factorize R).

**Leverage:** `leverage_scores_from_r()` — `h_i = ||R^{-T} x_i||^2`, equivalent to SVD-based `sum(U_i^2)`.

**Solve:** `solve_from_r_xty()` — two triangular solves (more stable than explicit inverse).

## CookSubSampler Key Parameters

- **`factorization`**: `'auto'` (default), `'svd'`, or `'qr'`. Auto selects SVD for single-process, QR if chunked/distributed.
- **`n_chunks`**: Number of TSQR chunks. None = single-pass (or 1 per rank if distributed).
- **`update_method`**: `'auto'` (default), `'woodbury'`, or `'qr'`. Auto selects QR for block mode (stable), Woodbury for non-block (fast).
- **`tree_reduction_threshold`**: Reduce accumulated R matrices after this many (default 10).

## Common Commands

```bash
# Install in editable mode with test deps
pip install -e .[test]

# Run tests
pytest

# Run tests with coverage
pytest --cov=subdatapy --cov-report=term-missing

# Run distributed benchmark (via torchrun)
torchrun --nproc_per_node=4 examples/mpi_benchmark/benchmark.py --method cooks --stepwise --ascending --factorization qr
```

## Testing

- Tests are in `tests/test_subsamplers.py` (7 tests) and `tests/test_linalg.py` (8 tests)
- Fixture `mini_dataset` in `tests/conftest.py` loads small .npy files from `tests/test_data/`
- Tests verify exact RMSE values against known expected results (using `pytest.approx` with `rel=1e-8`)
- Tests use `seed=41` for train/test split, `seed=42` for subsampling, `test_fraction=0.5`
- Device-agnostic: separate expected values for `'cuda'` and `'cpu'`

## Architecture Notes

- `process_data()` in `data.py` handles loading from .npy files, NumPy arrays, Pandas DataFrames, converting all to PyTorch tensors
- `BaseData.__init__` optionally prepends an intercept column of ones to X
- `BaseData.train()` supports `method='lstsq'` (default) or `method='qr'` (via TSQR)
- `RandomSubSampler.train_subsample()` also supports `method='lstsq'` (default) or `method='qr'` with optional `n_chunks`, same pattern as `BaseData.train()`
- `RandomSubSampler.create_subsample()` is the main entry point — calls `_create_sub_mask()` (overridden by subclasses) then `_subsample()`
- `create_subsample_errors_dataframe()` runs multiple subsample fractions with repeats, returns a MultiIndex DataFrame. For stepwise Cook's, successive fractions reuse the previous subset (greedy loop continues).
- Cook's `_create_sub_mask()` applies weights in-place (`mul_`/`div_`) to X_train and y_train around the sampling call, restored inside `try/finally` so an exception never leaves weighted data
- In chunked/distributed mode, `_move_train_to_cpu()` saves GPU memory; `_move_train_to_device()` restores data for the stepwise loop
- `_prepare_block_metadata()` sorts rows by config_id, builds group_metadata `[config_id, start_row, count]`, sorts by count for efficient batched padding
- In distributed partitioned mode, every rank participates in the stepwise greedy loop; owner rank broadcasts the winning config's X/y rows to all ranks via `linalg.broadcast_tensor` so the Woodbury/QR update stays identical on every rank

## CI

GitHub Actions workflow (`.github/workflows/python-test.yml`) runs pytest across Python 3.11-3.12 on ubuntu-latest.

## Style

- No formatter/linter configured
- Uses `torch.float64` (double precision) everywhere for numerical stability
- Heavy use of PyTorch tensor operations, broadcasting, and GPU acceleration
- Comments explain mathematical formulas (Woodbury identity, Sherman-Morrison, Cook's distance)

## Watch Out For (past bugs, now fixed — don't reintroduce)

- **`argmax` returns an index, not a config ID.** When selecting configs by score (Cook's, leverage), always map via `unique_config_idxs_train[index]` (non-block) or `batch_config_ids[index]` (block).
- **Intercept double-add.** When creating internal subsamplers (e.g. in `_create_initial_sub_mask`), pass `intercept=False` since `X_train` already has the intercept column.
- **Device consistency.** `config_idxs` and `enrow_mask` live on CPU; `X_train`, `y_train`, `w_train` live on GPU. When mixing in `torch.isin` or boolean indexing, move the smaller tensor to match. In chunked mode, `_move_train_to_device()` must also move `sub_mask_train`.
- **Always use `self.device`**, never hardcode `'cuda'`.
- **RNG difference — CPU vs CUDA.** `torch.manual_seed` seeds both CPU and CUDA generators as independent streams, so `torch.randperm(n, device='cuda')` and `torch.multinomial(probs_on_cuda, ...)` produce different sequences from their CPU counterparts even with identical seeds. To keep config selection device-independent, all sampling in `random.py`, `leverage.py`, and `cooks.py` (one-step) is **forced onto CPU RNG**: `torch.randperm(n)`, `torch.multinomial(probs.cpu(), ...)`. The selected configs now match across devices; RMSE values still differ by ~1e-12 from matmul/SVD backend floating-point drift, so tests use `pytest.approx(..., rel=1e-6)`. If you add a new sampling step, remember to route its RNG through CPU or the expected-values dict in `conftest.py` will no longer hold on both devices.
- **NCCL contiguity trap.** `torch.linalg.qr` on CUDA returns a *column-major* R. Using `torch.empty_like(R)` for the NCCL gather buffer inherits that stride, so NCCL's raw-byte transfer arrives transposed (only `[0,0]` populated per block). Always allocate the receive buffer with explicit `torch.empty(R.shape, dtype=..., device=...)` and call `.contiguous()` on the send buffer.
- **Conda env:** Use `subdatapy` conda env for development and testing.
- **In-place weighting in Cook's `_create_sub_mask`.** `X_train` and `y_train` are multiplied by `w_train` in place and divided back after sampling. The restore must be inside a `try/finally`; otherwise an exception in the stepwise compute leaves `X_train` permanently weighted.
- **Partitioned + world_size==1 is rejected.** `tsqr_r(partitioned=True)` with `world_size==1` raises `ValueError`. If you see this, you probably forgot to launch under `torchrun`.
- **Replicated distributed is rejected.** `tsqr_r(partitioned=False)` with `world_size>1` raises `ValueError`. If you're under `torchrun`, pass `.npy` file paths to `BaseData` so the auto-partition trigger fires.
