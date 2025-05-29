import pytest
import numpy as np

@pytest.fixture(scope="session") # Load data once per test session
def mini_dataset():
    
    data_dir = "tests/test_data/"
    print("\nLoading mini_dataset fixture...")
    X = np.load(data_dir+"X.npy")
    y = np.load(data_dir+"y.npy")
    w = np.load(data_dir+"w.npy")
    config_idxs = np.load(data_dir+"config_idxs.npy")

    return {
        "X": X,
        "y": y,
        "w": w,
        "config_idxs": config_idxs,
        "expected_rs_test_energy_rmse": 0.08652817482516958,
        "expected_asc_cooks_r_test_energy_rmse": 0.05032344020736297,
        "expected_ls_test_energy_rmse": 0.061184169732618365,
        "expected_asc_cooks_l_test_energy_rmse": 0.04707222902198353
    }