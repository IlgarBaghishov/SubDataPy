import pytest
import torch
import numpy as np
import pandas as pd
from subdatapy.subsampler import RandomSubSampler, LeverageSubSampler, CookSubSampler


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