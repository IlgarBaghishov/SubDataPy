"""03 — Leverage-score subsampling.

Weighted multinomial sampling over configs, where each config's weight is
its leverage score `h_i = ||R^{-T} x_i||^2`. Set `block=True` to aggregate
leverage over every row of a config (energy + force); `block=False` uses
only the energy row.

Launch:
    python 03_leverage.py
"""
from subdatapy.subsampler import LeverageSubSampler
from _common import load, device


def run(block: bool) -> None:
    d = load()
    ls = LeverageSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=device(),
        block=block,
        factorization="auto",       # 'svd' single-process, 'qr' if chunked/distributed
        test_fraction=0.5, seed=41,
    )
    ls.create_subsample(subsample_fraction=0.3, seed=42)
    ls.train_subsample()
    print(f"\n--- block={block} ---")
    ls.compute_subsample_errors(verbose=True)


def main() -> None:
    run(block=False)
    run(block=True)


if __name__ == "__main__":
    main()
