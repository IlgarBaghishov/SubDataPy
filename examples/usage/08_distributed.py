"""08 — Distributed partitioned mode (torchrun, each rank loads its slice).

Pass *file paths* to `CookSubSampler` (or any BaseData subclass) under
torchrun. `BaseData` auto-enters partitioned mode: rank 0 scans
config_idxs, computes per-rank ranges at config boundaries, broadcasts
them, and each rank `mmap`-loads only its partition. Peak memory per
rank scales as `D / world_size`.

Launch:
    # 4 ranks on one node, one GPU each
    torchrun --nproc_per_node=4 --rdzv_backend=c10d \\
        --rdzv_endpoint=localhost:29500 08_distributed.py

    # Two nodes × 1 rank × 4 local GPUs (NERSC-style)
    srun -N 2 --ntasks-per-node=1 --gpus-per-task=4 \\
        torchrun --nnodes=2 --nproc_per_node=1 \\
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 08_distributed.py
"""
import os
import torch
import torch.distributed as dist

from subdatapy.subsampler import CookSubSampler
from subdatapy import linalg
from _common import paths


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    n_visible = torch.cuda.device_count()

    if local_world_size == 1:
        # One rank per node — use every visible GPU.
        torch.cuda.set_device(0)
        device = "cuda:0"
        local_devices = [f"cuda:{i}" for i in range(n_visible)]
    else:
        # Multiple ranks per node — each rank owns one GPU.
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        local_devices = [device]

    p = paths()
    cs = CookSubSampler(
        X=p["X"], y=p["y"], w=p["w"],
        config_idxs=p["config_idxs"],
        enrow_mask=p["enrow_mask"],
        intercept=True,
        device=device,
        local_devices=local_devices,
        stepwise=True, ascending=True,
        initial_subsampler="random", initial_subsample_fraction=0.1,
        factorization="qr",
        test_fraction=0.5, seed=41,
    )
    cs.create_subsample(subsample_fraction=0.3, seed=42)
    cs.train_subsample(method="qr")
    cs.compute_subsample_errors(verbose=(linalg.get_rank() == 0))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
