"""07 — Multi-GPU inside a single process via `local_devices`.

Use every visible GPU from one Python process: TSQR and leverage chunks
are round-robined across the devices in `local_devices`. This is
different from torchrun-style multi-rank (example 08) — here there is
still only one process, one RNG stream, one Python interpreter; the
GPUs share work only inside the heavy linear-algebra kernels.

Launch:
    python 07_multi_gpu.py
"""
import torch
from subdatapy.subsampler import LeverageSubSampler
from _common import load


def main() -> None:
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Only {n_gpus} GPU(s) visible; this example needs >= 2.")
        return

    local_devices = [f"cuda:{i}" for i in range(n_gpus)]
    d = load()
    ls = LeverageSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=local_devices[0],
        local_devices=local_devices,          # round-robin across GPUs
        factorization="qr",
        n_chunks=2,                           # 2 chunks per local GPU
        test_fraction=0.5, seed=41,
    )
    ls.create_subsample(subsample_fraction=0.3, seed=42)
    ls.train_subsample(method="qr", n_chunks=n_gpus)
    ls.compute_subsample_errors(verbose=True)


if __name__ == "__main__":
    main()
