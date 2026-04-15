"""Generate a small random dataset shared by every example in this folder.

Writes X.npy, y.npy, w.npy, config_idxs.npy, enrow_mask.npy into the
examples/usage/data/ subdirectory. Re-run to regenerate with the same seed.

Usage:
    python gen_data.py
"""
import os
import numpy as np


def generate(
    out_dir: str = os.path.join(os.path.dirname(__file__), "data"),
    n_configs: int = 400,
    rows_per_config: int = 30,
    n_features: int = 20,
    noise: float = 0.01,
    seed: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    n = n_configs * rows_per_config
    # True descriptor-like design matrix
    X = rng.standard_normal((n, n_features))
    beta = rng.standard_normal((n_features, 1))
    y = (X @ beta).reshape(-1) + noise * rng.standard_normal(n)
    w = np.ones(n)

    # Contiguous config blocks — required by the partitioned loader.
    config_idxs = np.repeat(np.arange(n_configs, dtype=np.int64), rows_per_config)
    # First row of each config is the energy row; the rest are force rows.
    enrow_mask = np.zeros(n, dtype=bool)
    enrow_mask[::rows_per_config] = True

    for name, arr in [("X", X), ("y", y), ("w", w),
                      ("config_idxs", config_idxs), ("enrow_mask", enrow_mask)]:
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    print(f"Wrote {n} rows × {n_features} features ({n_configs} configs × "
          f"{rows_per_config} rows/config) to {out_dir}/")


if __name__ == "__main__":
    generate()
