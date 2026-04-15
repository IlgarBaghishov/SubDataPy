"""04 — One-step Cook's distance subsampling.

Compute Cook's distance for every configuration once, then either
weighted-sample from the resulting distribution (`sampling=True`) or
deterministically take the top-k highest (`sampling=False`). One-step is
non-block only — use the stepwise example (05) for block mode.

Launch:
    python 04_cooks_onestep.py
"""
from subdatapy.subsampler import CookSubSampler
from _common import load, device


def run(sampling: bool) -> None:
    d = load()
    cs = CookSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=device(),
        stepwise=False,                 # one-step Cook's
        sampling=sampling,
        test_fraction=0.5, seed=41,
    )
    cs.create_subsample(subsample_fraction=0.3, seed=42)
    cs.train_subsample()
    print(f"\n--- sampling={sampling} ---")
    cs.compute_subsample_errors(verbose=True)


def main() -> None:
    run(sampling=True)
    run(sampling=False)


if __name__ == "__main__":
    main()
