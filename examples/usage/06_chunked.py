"""06 — Single-process chunked TSQR (keep training data off GPU).

Setting `n_chunks=N` streams X through the primary device N chunks at a
time instead of materializing the whole matrix on GPU. Training data
stays on CPU until each chunk is needed, so peak VRAM is roughly
`n × p × 8 / n_chunks` bytes instead of the full `n × p × 8`. Combine
with `factorization='qr'` for any subsampler that supports it.

Launch:
    python 06_chunked.py
"""
from subdatapy.subsampler import LeverageSubSampler
from _common import load, device


def main() -> None:
    d = load()
    ls = LeverageSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=device(),
        factorization="qr",       # required for chunked leverage
        n_chunks=4,               # stream X through GPU in 4 passes
        test_fraction=0.5, seed=41,
    )
    ls.create_subsample(subsample_fraction=0.3, seed=42)
    ls.train_subsample(method="qr", n_chunks=4)   # chunked solver too
    ls.compute_subsample_errors(verbose=True)


if __name__ == "__main__":
    main()
