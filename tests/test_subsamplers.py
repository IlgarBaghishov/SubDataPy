import pytest
import numpy as np
import pandas as pd
from subdatapy.subsampler import RandomSubSampler, LeverageSubSampler, CookSubSampler
# from subdatapy.data import BaseData # If you want to test BaseData separately


def check_dataframe_results(df, expected_error=None, subsampler_name=None, column_name="Testing Energy RMSE"):
    """Helper function to check common DataFrame outputs."""
    assert isinstance(df, pd.DataFrame), f"{subsampler_name} did not return a Pandas DataFrame"
    assert not df.empty, f"{subsampler_name} returned an empty DataFrame"
    assert column_name in df.columns.get_level_values('Error Type'), \
        f"'{column_name}' not found in DataFrame columns for {subsampler_name}"

    errors = df[column_name]
    assert (errors >= 0).all().all(), f"{column_name} contains negative values for {subsampler_name}."
    assert errors.iloc[0,-1] == pytest.approx(expected_error, rel=1e-8), \
        f"{subsampler_name} {column_name} did not match expected value. " \
        f"Got {errors.iloc[0,-1]}, expected {expected_error}"


def test_random_subsampler_dataframe(mini_dataset):
    """Test RandomSubSampler.create_subsample_errors_dataframe."""
    data = mini_dataset
    rs = RandomSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"])
    rs_df = rs.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(rs_df, data["expected_rs_test_energy_rmse"], "RandomSubSampler")


def test_leverage_subsampler_dataframe(mini_dataset):
    """Test LeverageSubSampler.create_subsample_errors_dataframe."""
    data = mini_dataset
    ls = LeverageSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"])
    ls_df = ls.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(ls_df, data["expected_ls_test_energy_rmse"], "LeverageSubSampler")


def test_ascending_cooks_random_init_dataframe(mini_dataset):
    """Test CookSubSampler (stepwise, ascending, random init)."""
    data = mini_dataset
    cs_r = CookSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                          stepwise=True, ascending=True, initial_subsampler="random", initial_subsample_fraction=0.05)
    cs_r_df = cs_r.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(cs_r_df, data["expected_asc_cooks_r_test_energy_rmse"], "CookSubSampler (Random Init)")


def test_ascending_cooks_leverage_init_dataframe(mini_dataset):
    """Test CookSubSampler (stepwise, ascending, leverage init)."""
    data = mini_dataset
    cs_l = CookSubSampler(X=data["X"], y=data["y"], w=data["w"], test_fraction=0.5, seed=41, config_idxs=data["config_idxs"],
                          stepwise=True, ascending=True, initial_subsampler="leverage", initial_subsample_fraction=0.05)
    cs_l_df = cs_l.create_subsample_errors_dataframe(subsample_fractions_list=[0.05, 0.2], repeat_count_list=1, seed=42)
    check_dataframe_results(cs_l_df, data["expected_asc_cooks_l_test_energy_rmse"], "CookSubSampler (Leverage Init)")