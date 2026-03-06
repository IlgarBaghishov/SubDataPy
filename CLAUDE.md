# CLAUDE.md - SubDataPy

## Project Overview

SubDataPy is a Python toolkit for subsampling large datasets for Machine Learning Interatomic Potentials (MLIPs). It implements statistical subsampling methods (random, leverage score, Cook's distance) that operate on grouped rows (atomic configurations with energy + force rows).

## Tech Stack

- **Language:** Python 3.11+
- **Core dependency:** PyTorch (>=2.8) for all linear algebra (SVD, QR, least squares) — runs on GPU (CUDA) by default
- **Other deps:** NumPy (>=1.20), Pandas (>=1.0)
- **Optional:** mpi4py (for distributed TSQR in `mpi_cooks.py`)
- **Build:** setuptools via `pyproject.toml`
- **Tests:** pytest with pytest-cov

## Project Structure

```
subdatapy/
  __init__.py              # Exports BaseData
  data.py                  # BaseData class: data loading, train/test split, WLS training
  subsampler/
    __init__.py            # Exports all subsamplers
    random.py              # RandomSubSampler (base for all subsamplers)
    leverage.py            # LeverageSubSampler (SVD-based leverage scores)
    cooks.py               # CookSubSampler (one-step and stepwise Cook's distance)
    entropy.py             # EntropySubSampler (stub/placeholder)
    mpi_cooks.py           # TSQRMPICookSubSampler (distributed TSQR + Cook's)
  trainer/
    __init__.py            # Empty
tests/
  conftest.py              # pytest fixtures, loads .npy test data
  test_subsamplers.py      # Tests for Random, Leverage, Cook's subsamplers
  test_data/               # .npy files (X, y, w, config_idxs)
tools/
  generate_matrices/FitSNAP/qSNAP.py  # FitSNAP helper to generate descriptor matrices
examples/                  # Jupyter notebook examples
docs/                      # RST documentation
```

## Class Hierarchy

```
BaseData (data.py)
  └── RandomSubSampler (random.py)        — random config selection
        ├── LeverageSubSampler (leverage.py) — weighted sampling by leverage scores
        ├── CookSubSampler (cooks.py)        — Cook's distance (one-step SVD or stepwise Woodbury)
        └── TSQRMPICookSubSampler (mpi_cooks.py) — distributed Cook's via Tall-Skinny QR
```

All subsamplers inherit from `RandomSubSampler`, which inherits from `BaseData`.

## Key Concepts

- **config_idxs:** Integer array mapping each row to a configuration ID. Subsampling selects/removes entire configurations (energy + force rows together).
- **enrow_mask:** Boolean mask identifying energy rows (one per config). Force rows are `~enrow_mask`.
- **block mode:** Treats all rows of a config as a block for leverage/Cook's calculations.
- **subsample_fraction:** Fraction of unique configurations to keep.
- **Data flow:** Data loads to CPU, then moves to GPU (`device='cuda'`) for train/test splits and computation. Uses `torch.float64` throughout.

## Common Commands

```bash
# Install in editable mode with test deps
pip install -e .[test]

# Run tests
pytest

# Run tests with coverage
pytest --cov=subdatapy --cov-report=term-missing

# Run MPI version (requires mpi4py)
mpirun -np <N> python <script.py>
```

## Testing

- Tests are in `tests/test_subsamplers.py`
- Fixture `mini_dataset` in `tests/conftest.py` loads small .npy files from `tests/test_data/`
- Tests verify exact RMSE values against known expected results (using `pytest.approx` with `rel=1e-8`)
- Tests use `seed=41` for train/test split, `seed=42` for subsampling, `test_fraction=0.5`
- Tests require CUDA GPU to run (default `device='cuda'`)

## Architecture Notes

- `process_data()` in `data.py` handles loading from .npy files, NumPy arrays, Pandas DataFrames, converting all to PyTorch tensors
- `BaseData.__init__` optionally prepends an intercept column of ones to X
- `RandomSubSampler.create_subsample()` is the main entry point — calls `_create_sub_mask()` (overridden by subclasses) then `_subsample()`
- `create_subsample_errors_dataframe()` runs multiple subsample fractions with repeats, returns a MultiIndex DataFrame
- Cook's stepwise method uses Woodbury/Sherman-Morrison rank-k updates to incrementally update `(X'X)^{-1}`
- MPI Cook's (`mpi_cooks.py`) implements 4 execution modes: regular QR, sequential TSQR, parallel TSQR, hybrid TSQR
- In-place operations (`mul_`, `div_`) used in Cook's to save GPU memory

## CI

GitHub Actions workflow (`.github/workflows/python-test.yml`) runs pytest across Python 3.11-3.12 on ubuntu-latest.

## Style

- No formatter/linter configured
- Uses `torch.float64` (double precision) everywhere for numerical stability
- Heavy use of PyTorch tensor operations, broadcasting, and GPU acceleration
- Comments explain mathematical formulas (Woodbury identity, Sherman-Morrison, Cook's distance)

## Watch Out For (past bugs, now fixed — don't reintroduce)

- **`argmax` returns an index, not a config ID.** When selecting configs by score (Cook's, leverage), always map via `unique_config_idxs_train[index]`.
- **Intercept double-add.** When creating internal subsamplers (e.g. in `_create_initial_sub_mask`), pass `intercept=False` since `X_train` already has the intercept column.
- **Device consistency.** `config_idxs` and `enrow_mask` live on CPU; `X_train`, `y_train`, `w_train` live on GPU. When mixing in `torch.isin` or boolean indexing, move the smaller tensor to match.
- **Always use `self.device`**, never hardcode `'cuda'`.
- **RNG difference.** `torch.manual_seed` + `torch.randperm` gives different sequences than NumPy. Test expected values are tied to PyTorch's RNG — if upgrading PyTorch major versions, expected values may need regeneration.
- **Conda env:** Use `subdatapy` conda env (Python 3.11) for development and testing.
