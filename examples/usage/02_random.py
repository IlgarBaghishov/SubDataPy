"""02 — Uniform-random subsampling at configuration granularity.

Selects `subsample_fraction` of unique configurations (not rows) at
random, trains on the subset, reports errors on the subsample, the
whole train set, and the held-out test set.

Launch:
    python 02_random.py
"""
from subdatapy.subsampler import RandomSubSampler
from _common import load, device


def main() -> None:
    d = load()
    rs = RandomSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=device(),
        test_fraction=0.5, seed=41,
    )
    rs.create_subsample(subsample_fraction=0.3, seed=42)
    rs.train_subsample()
    rs.compute_subsample_errors(verbose=True)


if __name__ == "__main__":
    main()
