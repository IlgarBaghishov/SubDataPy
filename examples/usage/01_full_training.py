"""01 — Full weighted least-squares fit, no subsampling.

This is the baseline every other example varies from. We load the full
dataset, split 50/50 into train/test at config granularity, fit a WLS
regression, and print train/test RMSEs for energy and force rows.

Launch:
    python 01_full_training.py
"""
from subdatapy.data import BaseData
from _common import load, device


def main() -> None:
    d = load()
    bd = BaseData(
        X=d["X"], y=d["y"], w=d["w"],
        config_idxs=d["config_idxs"],
        enrow_mask=d["enrow_mask"],
        intercept=True,
        device=device(),
    )
    bd.train_test_split(test_fraction=0.5, seed=41)
    bd.train(method="lstsq")           # or method="qr" for TSQR solver
    e_tr, f_tr, e_te, f_te = bd.compute_errors(verbose=True)


if __name__ == "__main__":
    main()
