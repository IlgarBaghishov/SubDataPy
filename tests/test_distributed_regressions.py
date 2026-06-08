"""Regression tests for audit findings.

Each test documents one specific claim from the audit and verifies whether
it is a real bug. Tests are independent of distributed execution so they
can run in CI; behavior under torch.distributed is exercised by the
benchmark scripts.
"""
import os
import tempfile
import numpy as np
import pytest
import torch

from subdatapy import linalg
from subdatapy.data import BaseData
from subdatapy.subsampler import CookSubSampler


def _make_npy_dataset(tmp_path, n_configs=20, rows_per_config=5, n_features=8):
    """Write a small contiguous-config .npy dataset and return file paths."""
    n = n_configs * rows_per_config
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n_features))
    y = rng.standard_normal(n)
    w = np.ones(n)
    cfg = np.repeat(np.arange(n_configs, dtype=np.int64), rows_per_config)
    erm = np.zeros(n, dtype=bool)
    erm[::rows_per_config] = True

    paths = {}
    for name, arr in [("X", X), ("y", y), ("w", w),
                      ("config_idxs", cfg), ("enrow_mask", erm)]:
        p = str(tmp_path / f"{name}.npy")
        np.save(p, arr)
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# C1: BaseData full-train in partitioned mode honors a CPU train target
# ---------------------------------------------------------------------------

def test_partitioned_loader_keeps_design_matrix_on_cpu(tmp_path):
    """Per the device rule, partitioned BaseData keeps the design matrix on
    CPU and records train/test row indices instead of copying X_train/X_test
    onto the GPU (which previously OOMed on large partitions).

    The auto-trigger requires world_size>1, so we drive _load_partitioned
    directly via a single-rank gloo group (just enough for the broadcast +
    all_gather collectives inside the loader).
    """
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")
    if dist.is_initialized():
        pytest.skip("distributed already initialized in this process")

    paths = _make_npy_dataset(tmp_path)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        bd = BaseData.__new__(BaseData)
        bd.device = "cpu"
        bd.dtype = torch.float64
        bd.storage_dtype = torch.float64
        bd.coeffs = None
        bd.local_devices = ["cpu"]
        bd._is_partitioned = True
        bd._unique_config_idxs_train_override = None
        bd._load_partitioned(paths["X"], paths["y"], paths["w"],
                             paths["config_idxs"], paths["enrow_mask"],
                             intercept=True)
        bd.train_test_split(test_fraction=0.5, seed=41)
        # The design matrix stays on CPU and is never copied into X_train.
        assert bd.X.device.type == "cpu"
        assert not hasattr(bd, "X_train"), (
            "BaseData must not materialize X_train; it should stream "
            "self.X[train_idx] in chunks instead")
        assert bd.train_idx.numel() + bd.test_idx.numel() == bd.X.shape[0]
        # Indices are aligned with the per-row siblings (increasing row order).
        assert bd.y_train.shape[0] == bd.train_idx.numel()
    finally:
        dist.destroy_process_group()
        for k in ("RANK", "WORLD_SIZE"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# C2: partitioned=True with world_size==1 should not silently fall through
# ---------------------------------------------------------------------------

def test_tsqr_partitioned_world_size_1_raises():
    """Calling tsqr_r with partitioned=True and world_size==1 must raise.
    Previously it silently fell through to the replicated single-pass path,
    which let bugs hide because the partitioned branch wasn't actually
    exercised."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    X = torch.randn(100, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="partitioned=True"):
        linalg.tsqr_r(X, device="cuda", partitioned=True)
    with pytest.raises(ValueError, match="partitioned=True"):
        linalg.tsqr_r_xty(X, X[:, :1], device="cuda", partitioned=True)


def test_tsqr_replicated_distributed_raises(tmp_path):
    """Replicated distributed mode (world_size>1 with partitioned=False)
    was removed. Calling tsqr_r under torch.distributed without
    partitioned=True must raise a helpful ValueError rather than silently
    running a memory-inefficient replicated path."""
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")
    if dist.is_initialized():
        pytest.skip("distributed already initialized in this process")

    # Fake world_size > 1 by monkey-patching linalg.get_world_size.
    orig = linalg.get_world_size
    linalg.get_world_size = lambda: 2
    try:
        X = torch.randn(50, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="partitioned=True"):
            linalg.tsqr_r(X, device="cpu", partitioned=False)
    finally:
        linalg.get_world_size = orig


# ---------------------------------------------------------------------------
# C3: Cook's _create_sub_mask must not leave X_train weighted on exception
# ---------------------------------------------------------------------------

def test_cooks_never_mutates_host_design_matrix(mini_dataset):
    """Cook's must never weight/mutate the shared host design matrix in place
    (it weights per gather instead). Previously it multiplied X_train by w in
    place and relied on a try/finally restore; the index model removes that
    entirely, so self.X must be byte-identical before and after sampling, even
    if the stepwise compute raises."""
    d = mini_dataset
    css = CookSubSampler(d["X"], y=d["y"], w=d["w"],
                         config_idxs=d["config_idxs"],
                         intercept=True, device=d["device"],
                         stepwise=True, ascending=True,
                         test_fraction=0.5, seed=41,
                         factorization="svd")

    X_before = css.X.clone()
    y_train_before = css.y_train.clone()

    # Force an exception inside _stepwise_cooks_sampling
    def boom():
        raise RuntimeError("intentional")
    css._stepwise_cooks_sampling = boom

    with pytest.raises(RuntimeError, match="intentional"):
        css.create_subsample(subsample_fraction=0.5, seed=42)

    assert torch.equal(css.X, X_before), "Cook's mutated the host design matrix"
    assert torch.equal(css.y_train, y_train_before), "Cook's mutated y_train"


# ---------------------------------------------------------------------------
# Sanity: NCCL contiguity bug regression — verify that gather_list buffers
# allocated by torch.empty(shape, ...) are actually contiguous.  This is the
# non-distributed half of the regression we already fixed in linalg.py.
# ---------------------------------------------------------------------------

def test_torch_empty_returns_contiguous():
    """torch.empty(shape, dtype, device) must yield a contiguous buffer.
    The bug was using torch.empty_like(R) where R had column-major stride
    from torch.linalg.qr on CUDA. This guards against regressions where
    someone reverts to empty_like."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    X = torch.randn(200, 50, dtype=torch.float64, device="cuda")
    _, R = torch.linalg.qr(X, mode="r")
    # Document the platform-specific stride that motivated the fix
    expected_non_contig = not R.is_contiguous()

    # The fix: explicit torch.empty(shape, ...)
    buf = torch.empty(R.shape, dtype=R.dtype, device=R.device)
    assert buf.is_contiguous()

    # And the antipattern that caused the original bug
    if expected_non_contig:
        buf_bad = torch.empty_like(R)
        assert not buf_bad.is_contiguous(), (
            "Expected empty_like(QR-R) to inherit non-contiguous stride on "
            f"this platform; saw {buf_bad.stride()}. If this assert flips, "
            "the underlying torch behavior changed.")


# ---------------------------------------------------------------------------
# H2: -1 sentinel from _compute_*_cooks reaches change_mask without harm
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# H1: Block Cook's stepwise works under torch.distributed (world_size=1 gloo
# is enough to exercise the code path and assert mathematical equivalence
# to the non-distributed run).
# ---------------------------------------------------------------------------

def test_block_cooks_distributed_matches_single_process(mini_dataset):
    """Stepwise block=True Cook's was never exercised under any distributed
    code path. Drive it through the is_distributed() branches with a
    single-rank gloo group and verify the selected config set matches the
    non-distributed run bit-for-bit."""
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")
    if dist.is_initialized():
        pytest.skip("distributed already initialized in this process")

    d = mini_dataset

    # Reference run (no torch.distributed in scope).
    css_ref = CookSubSampler(d["X"], y=d["y"], w=d["w"],
                             config_idxs=d["config_idxs"],
                             intercept=True, device=d["device"],
                             stepwise=True, ascending=True,
                             block=True,
                             test_fraction=0.5, seed=41,
                             factorization="qr")
    css_ref.create_subsample(subsample_fraction=0.5, seed=42)
    ref_selected = sorted(
        css_ref.config_idxs_train[css_ref.sub_mask_train].unique().cpu().tolist())

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29556")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        css_dist = CookSubSampler(d["X"], y=d["y"], w=d["w"],
                                  config_idxs=d["config_idxs"],
                                  intercept=True, device=d["device"],
                                  stepwise=True, ascending=True,
                                  block=True,
                                  test_fraction=0.5, seed=41,
                                  factorization="qr")
        css_dist.create_subsample(subsample_fraction=0.5, seed=42)
        dist_selected = sorted(
            css_dist.config_idxs_train[css_dist.sub_mask_train].unique().cpu().tolist())
    finally:
        dist.destroy_process_group()
        for k in ("RANK", "WORLD_SIZE"):
            os.environ.pop(k, None)

    assert dist_selected == ref_selected, (
        "Block Cook's stepwise selected different configs under "
        f"torch.distributed: ref={ref_selected[:10]}..., "
        f"dist={dist_selected[:10]}...")


def test_compute_nonblock_cooks_sentinel_when_no_energy_rows(mini_dataset):
    """When a rank has zero energy rows, _compute_nonblock_cooks must
    return a sentinel that won't crash downstream consumers."""
    d = mini_dataset
    css = CookSubSampler(d["X"], y=d["y"], w=d["w"],
                         config_idxs=d["config_idxs"],
                         intercept=True, device=d["device"],
                         stepwise=True, ascending=True,
                         test_fraction=0.5, seed=41,
                         factorization="svd")
    # Drive enrow_mask_train to all-False locally
    css.enrow_mask_train = torch.zeros_like(css.enrow_mask_train)
    css.XTX_inv = torch.eye(css.X.shape[1], device=css.device,
                            dtype=css.dtype)
    coeffs = torch.zeros((css.X.shape[1], 1), device=css.device,
                         dtype=css.dtype)
    val, cid = css._compute_nonblock_cooks(coeffs)
    assert val == -float("inf")
    assert cid == -1
