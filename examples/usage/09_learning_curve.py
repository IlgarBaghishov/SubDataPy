"""09 — Multi-fraction learning curve via `create_subsample_errors_dataframe`.

Run the same subsampler at several `subsample_fraction` values and
(optionally) with repeated seeds per fraction, returning a pandas
MultiIndex DataFrame (Error Type × Subsample Fraction). For stepwise
Cook's, fractions must be sorted ascending — the greedy loop reuses the
previously-selected subset and just extends it.

Launch:
    python 09_learning_curve.py
"""
from subdatapy.subsampler import RandomSubSampler, CookSubSampler
from _common import load, device


def main() -> None:
    d = load()

    # Random baseline: repeat each small fraction more times for variance.
    rs = RandomSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True, device=device(),
        test_fraction=0.5, seed=41,
    )
    rs_df = rs.create_subsample_errors_dataframe(
        subsample_fractions_list=[0.1, 0.2, 0.4, 0.6],
        repeat_count_list=[5, 3, 2, 1],          # more repeats for smaller fractions
        seed=42,
    )
    print("Random subsampling learning curve:\n", rs_df, "\n")

    # Stepwise Cook's reuses state across fractions (greedy loop continues).
    cs = CookSubSampler(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True, device=device(),
        stepwise=True, ascending=True,
        initial_subsampler="random", initial_subsample_fraction=0.05,
        test_fraction=0.5, seed=41,
    )
    cs_df = cs.create_subsample_errors_dataframe(
        subsample_fractions_list=[0.1, 0.2, 0.4, 0.6],
        repeat_count_list=1,
        seed=42,
    )
    print("Stepwise Cook's learning curve:\n", cs_df)


if __name__ == "__main__":
    main()
