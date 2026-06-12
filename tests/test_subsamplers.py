import pytest
import torch
import numpy as np
import pandas as pd
from subdatapy.data import BaseData
from subdatapy.subsampler import RandomSubSampler, LeverageSubSampler, CookSubSampler


def test_float32_storage_matches_float64_well_conditioned():
    """dtype=torch.float32 stores X in float32 (half the memory) but keeps all
    factors in float64; on well-conditioned data the coefficients match the
    float64 run to ~1e-5, and the default (float64) is unchanged."""
    rng = np.random.default_rng(0)
    n, p, rpc = 2000, 12, 5
    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, p - 1))])  # well conditioned
    true = rng.standard_normal((p, 1))
    y = (X @ true + 1e-3 * rng.standard_normal((n, 1))).reshape(-1)
    cfg = np.repeat(np.arange(n // rpc), rpc).astype(np.int64)
    erm = np.zeros(n, bool); erm[::rpc] = True

    out = {}
    for dt in (torch.float64, torch.float32):
        bd = BaseData(X, y=y, config_idxs=cfg, enrow_mask=erm, intercept=False,
                      device="cpu", dtype=dt)
        bd.train_test_split(test_fraction=0.3, seed=41)
        bd.train(method="auto")
        out[dt] = bd

    assert out[torch.float32].X.dtype == torch.float32       # X stored in float32
    assert out[torch.float64].X.dtype == torch.float64
    assert out[torch.float32].coeffs.dtype == torch.float64  # factors stay float64
    rel = (torch.norm(out[torch.float64].coeffs - out[torch.float32].coeffs)
           / torch.norm(out[torch.float64].coeffs)).item()
    assert rel < 1e-5, f"float32 coeffs diverged on well-conditioned data: {rel:.2e}"


def test_invalid_dtype_rejected():
    with pytest.raises(ValueError, match="float32 or torch.float64"):
        BaseData(np.ones((4, 2)), device="cpu", dtype=torch.int32)


def check_dataframe_results(df, expected_error=None, subsampler_name=None, column_name="Testing Energy RMSE"):
    """Helper function to check common DataFrame outputs."""
    assert isinstance(df, pd.DataFrame), f"{subsampler_name} did not return a Pandas DataFrame"
    assert not df.empty, f"{subsampler_name} returned an empty DataFrame"
    assert column_name in df.columns.get_level_values('Error Type'), \
        f"'{column_name}' not found in DataFrame columns for {subsampler_name}"

    errors = df[column_name]
    assert (errors >= 0).all().all(), f"{column_name} contains negative values for {subsampler_name}."
    # rel=1e-6: loose enough to absorb the ~1e-12 CPU-vs-CUDA drift from
    # backend-specific matmul/SVD rounding, tight enough to flag real
    # pipeline bugs (which change RMSE by orders of magnitude).
    assert errors.iloc[0,-1] == pytest.approx(expected_error, rel=1e-6), \
        f"{subsampler_name} {column_name} did not match expected value. " \
        f"Got {errors.iloc[0,-1]}, expected {expected_error}"


def test_random_subsampler_dataframe(mini_dataset):
    """Test RandomSubSampler.create_subsample_errors_dataframe."""
    data = mini_dataset
    rs = RandomSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"], device=data["device"])
    rs_df = rs.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(rs_df, data["expected_rs_test_energy_rmse"], "RandomSubSampler")


def test_leverage_subsampler_dataframe(mini_dataset):
    """Test LeverageSubSampler.create_subsample_errors_dataframe."""
    data = mini_dataset
    ls = LeverageSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"], device=data["device"])
    ls_df = ls.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(ls_df, data["expected_ls_test_energy_rmse"], "LeverageSubSampler")


def test_ascending_cooks_random_init_dataframe(mini_dataset):
    """Test CookSubSampler (stepwise, ascending, random init)."""
    data = mini_dataset
    cs_r = CookSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                          device=data["device"], stepwise=True, ascending=True, initial_subsampler="random", initial_subsample_fraction=0.05)
    cs_r_df = cs_r.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(cs_r_df, data["expected_asc_cooks_r_test_energy_rmse"], "CookSubSampler (Random Init)")


def test_ascending_cooks_leverage_init_dataframe(mini_dataset):
    """Test CookSubSampler (stepwise, ascending, leverage init)."""
    data = mini_dataset
    cs_l = CookSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                          device=data["device"], stepwise=True, ascending=True, initial_subsampler="leverage", initial_subsample_fraction=0.05)
    cs_l_df = cs_l.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(cs_l_df, data["expected_asc_cooks_l_test_energy_rmse"], "CookSubSampler (Leverage Init)")


def test_ascending_cooks_qr_factorization(mini_dataset):
    """CookSubSampler with factorization='qr' produces reasonable results."""
    data = mini_dataset
    cs = CookSubSampler(X=data["X"], y=data["y"], w=data["w"],
                        test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                        device=data["device"], stepwise=True, ascending=True,
                        initial_subsampler="random", initial_subsample_fraction=0.05,
                        factorization='qr')
    cs_df = cs.create_subsample_errors_dataframe(
        subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    errors = cs_df["Testing Energy RMSE"]
    assert (errors >= 0).all().all()
    assert errors.iloc[0, -1] < 0.1


def test_ascending_cooks_chunked_matches_unchunked(mini_dataset):
    """CookSubSampler with n_chunks=2 matches n_chunks=None (QR path)."""
    data = mini_dataset

    cs1 = CookSubSampler(X=data["X"], y=data["y"], w=data["w"],
                         test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                         device=data["device"], stepwise=True, ascending=True,
                         initial_subsampler="random", initial_subsample_fraction=0.05,
                         factorization='qr', n_chunks=None)
    cs1.create_subsample(subsample_fraction=0.2, seed=42)
    cs1.train_subsample()
    e1 = cs1.compute_subsample_errors(verbose=False)

    cs2 = CookSubSampler(X=data["X"], y=data["y"], w=data["w"],
                         test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                         device=data["device"], stepwise=True, ascending=True,
                         initial_subsampler="random", initial_subsample_fraction=0.05,
                         factorization='qr', n_chunks=2)
    cs2.create_subsample(subsample_fraction=0.2, seed=42)
    cs2.train_subsample()
    e2 = cs2.compute_subsample_errors(verbose=False)

    for v1, v2 in zip(e1, e2):
        assert v1 == pytest.approx(v2, rel=1e-6)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("block", [True, False])
def test_stepwise_cooks_incremental_inverse_matches_recompute(mini_dataset, ascending, block):
    """The (X^T W^2 X)^{-1} maintained incrementally through the greedy
    add/remove loop must match the inverse recomputed from scratch on the
    final selected subset, for BOTH ascending (add; QR update for block,
    Woodbury otherwise) and descending (remove; Woodbury only). The old
    descending+block bug used the append-only QR update for removals, so its
    inverse diverged from the real subset — this pins both directions."""
    d = mini_dataset
    css = CookSubSampler(X=d["X"], y=d["y"], w=d["w"], test_fraction=0.5, seed=41,
                         config_idxs=d["config_idxs"], device=d["device"],
                         stepwise=True, ascending=ascending, block=block,
                         factorization="qr",
                         initial_subsample_fraction=(0.1 if ascending else 1.0))
    css.create_subsample(subsample_fraction=0.5, seed=42)

    sm = css.sub_mask_train.cpu()
    Xw = (css.X[css.train_idx[sm]] * css.w_train[sm].reshape(-1, 1)).to(css.device)
    ref = torch.linalg.inv(Xw.T @ Xw)
    rel = (torch.linalg.norm(css.XTX_inv - ref) / torch.linalg.norm(ref)).item()
    assert rel < 1e-6, (
        f"incremental XTX_inv diverged from recompute "
        f"(ascending={ascending}, block={block}): relerr={rel:.2e}")


def test_onestep_cooks_svd_qr_match_unordered_configs():
    """One-step Cook's must select the same configs via SVD and QR even when
    config ids are not sorted by row position (contiguous blocks, but
    decreasing ids). The SVD path used to map scores to the sorted unique
    config-id list, pairing each score with the wrong config for unordered
    data; it now maps to the energy row's actual config id, like the QR path."""
    rng = np.random.default_rng(0)
    n_configs, rpc, p = 60, 6, 6
    n = n_configs * rpc
    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, p - 1))])
    true = rng.standard_normal((p, 1))
    y = (X @ true + 1e-2 * rng.standard_normal((n, 1))).reshape(-1)
    cfg = np.repeat(np.arange(n_configs)[::-1], rpc).astype(np.int64)  # decreasing ids
    erm = np.zeros(n, bool); erm[::rpc] = True
    kw = dict(config_idxs=cfg, enrow_mask=erm, intercept=False, device="cpu",
              test_fraction=0.3, seed=41, stepwise=False, sampling=False)

    def sel(fact):
        c = CookSubSampler(X, y=y, factorization=fact, **kw)
        c.create_subsample(0.5, seed=42)
        return sorted(c.config_idxs_train[c.sub_mask_train].unique().tolist())

    assert sel("svd") == sel("qr")


def test_descending_cooks_rejects_explicit_qr_update(mini_dataset):
    """The QR update can only append rows, so explicit update_method='qr' with
    descending must raise rather than silently update in the wrong direction."""
    d = mini_dataset
    css = CookSubSampler(X=d["X"], y=d["y"], w=d["w"], test_fraction=0.5, seed=41,
                         config_idxs=d["config_idxs"], device=d["device"],
                         stepwise=True, ascending=False, block=True,
                         factorization="qr", update_method="qr",
                         initial_subsample_fraction=1.0)
    with pytest.raises(ValueError, match="cannot remove rows"):
        css.create_subsample(subsample_fraction=0.5, seed=42)


def test_leverage_qr_matches_svd(mini_dataset):
    """LeverageSubSampler QR-based matches SVD-based leverage scores."""
    data = mini_dataset

    ls_svd = LeverageSubSampler(X=data["X"], y=data["y"], w=data["w"],
                                test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                                device=data["device"], factorization='svd')
    ls_svd.create_subsample(subsample_fraction=0.2, seed=42)

    ls_qr = LeverageSubSampler(X=data["X"], y=data["y"], w=data["w"],
                               test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                               device=data["device"], factorization='qr')
    ls_qr.create_subsample(subsample_fraction=0.2, seed=42)

    assert torch.allclose(ls_svd.leverage_scores, ls_qr.leverage_scores, atol=1e-8)