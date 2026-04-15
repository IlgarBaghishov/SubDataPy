# SubDataPy usage examples

Minimal, copy-paste-ready scripts covering every combination of
subsampling algorithm × parallelization level supported by SubDataPy.
Organized **one factor at a time**: start at `01_full_training.py`, then
each subsequent script changes exactly one thing relative to the baseline.

## Setup

```bash
# From repo root (one time) — the conda env name is subdatapy
eval "$(conda shell.bash hook)" && conda activate subdatapy
pip install -e .            # editable install

# One-time random dataset
cd examples/usage
python gen_data.py          # writes data/{X,y,w,config_idxs,enrow_mask}.npy
```

Every example auto-detects the device: `cuda:0` if a GPU is visible,
`cpu` otherwise. Force CPU with `CUDA_VISIBLE_DEVICES=`:

```bash
CUDA_VISIBLE_DEVICES= python 02_random.py   # CPU run
python 02_random.py                         # GPU if available
```

## Index

| # | File | Varies | Launch |
|---|---|---|---|
| 01 | `01_full_training.py` | **Baseline** — full WLS, no subsampling | `python 01_full_training.py` |
| 02 | `02_random.py` | Method → uniform-random subsampling | `python 02_random.py` |
| 03 | `03_leverage.py` | Method → leverage-score (both block and non-block) | `python 03_leverage.py` |
| 04 | `04_cooks_onestep.py` | Method → one-step Cook's (sampling *and* top-k) | `python 04_cooks_onestep.py` |
| 05 | `05_cooks_stepwise.py` | Method → stepwise Cook's (ascending/descending, block/non-block) | `python 05_cooks_stepwise.py` |
| 06 | `06_chunked.py` | Parallelization → single-process chunked TSQR (`n_chunks=N`) | `python 06_chunked.py` |
| 07 | `07_multi_gpu.py` | Parallelization → one process, many GPUs (`local_devices`) | `python 07_multi_gpu.py` |
| 08 | `08_distributed.py` | Parallelization → multi-rank partitioned (torchrun + file paths) | see below |
| 09 | `09_learning_curve.py` | API surface → multi-fraction learning curve via DataFrame | `python 09_learning_curve.py` |

## Launch patterns

```bash
# Plain single-process (examples 01–07, 09)
python NN_example.py

# Distributed partitioned (example 08) — one node
torchrun --nproc_per_node=4 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 08_distributed.py

# Distributed partitioned — two nodes, one rank per node, each rank uses
# all local GPUs (NERSC Perlmutter style)
srun -N 2 --ntasks-per-node=1 --gpus-per-task=4 \
    torchrun --nnodes=2 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 08_distributed.py
```

## Coverage matrix

| Algorithm | Single-process | Chunked (`n_chunks`) | Multi-GPU (`local_devices`) | Distributed partitioned |
|---|:---:|:---:|:---:|:---:|
| `BaseData.train` | 01 | 06¹ | 07¹ | 08¹ |
| `RandomSubSampler` | 02 | 06¹ | 07¹ | 08¹ |
| `LeverageSubSampler` (non-block / block) | 03 | 06 | 07 | 08¹ |
| `CookSubSampler` one-step | 04 | n/a² | n/a² | n/a² |
| `CookSubSampler` stepwise (non-block / block) | 05 | 06¹ | 07¹ | 08 |
| Learning-curve DataFrame | 09 | 09¹ | 09¹ | 09¹ |

¹ Same pattern as the cited example — swap the subsampler class or add
  the relevant kwarg. The examples are orthogonal on purpose.

² One-step Cook's requires a full SVD of the training matrix and does
  not support chunked or distributed mode. Use stepwise (`stepwise=True`)
  when you need those.

## Rules of thumb

- **CPU vs GPU:** every script picks one via `_common.device()`. Same
  code path.
- **Memory pressure → chunked or distributed.** `n_chunks` streams the
  matrix through one GPU. `local_devices` spreads across GPUs on one
  process. `torchrun + file paths` splits both memory *and* compute
  across ranks.
- **Reproducibility:** sampling RNG is routed through CPU so the same
  seed picks the same configs on CPU and GPU.
- **One-step Cook's is single-process only.** For every other method,
  every parallelization level works.
