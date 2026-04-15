"""05 — Stepwise Cook's distance subsampling.

Greedy loop that adds (ascending) or removes (descending) one config per
iteration by maximum/minimum Cook's distance. After each step the
factorization is refreshed via Woodbury or QR update so the greedy search
stays cheap.

`block=True` aggregates Cook's over all rows of a configuration (stable
for MLIP settings). Initial subset is a small random seed.

Launch:
    python 05_cooks_stepwise.py
"""
from subdatapy.subsampler import CookSubSampler
from _common import load, device


def run(ascending: bool, block: bool) -> None:
    d = load()
    cs = CookSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=device(),
        stepwise=True,
        ascending=ascending,
        block=block,
        initial_subsampler="random",
        initial_subsample_fraction=0.1,
        factorization="auto",
        update_method="auto",            # Woodbury for non-block, QR for block
        test_fraction=0.5, seed=41,
    )
    cs.create_subsample(subsample_fraction=0.3, seed=42)
    cs.train_subsample()
    print(f"\n--- ascending={ascending}, block={block} ---")
    cs.compute_subsample_errors(verbose=True)


def main() -> None:
    run(ascending=True,  block=False)
    run(ascending=False, block=False)
    run(ascending=True,  block=True)


if __name__ == "__main__":
    main()
