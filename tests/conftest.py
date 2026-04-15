import pytest
import torch
import numpy as np

@pytest.fixture(scope="session")
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope="session")
def mini_dataset(device):

    data_dir = "tests/test_data/"
    print(f"\nLoading mini_dataset fixture (device={device})...")
    X = np.load(data_dir+"X.npy")
    y = np.load(data_dir+"y.npy")
    w = np.load(data_dir+"w.npy")
    config_idxs = np.load(data_dir+"config_idxs.npy")

    # Expected values are shared between CPU and CUDA because the
    # subsamplers force sampling RNG to CPU (see random.py / leverage.py
    # / cooks.py) — identical seed => identical selected configs across
    # devices. Remaining CPU/CUDA drift is from matmul/SVD backend
    # floating-point differences, well inside the test tolerance.
    expected = {
        "expected_rs_test_energy_rmse": 0.07199577683153684,
        "expected_ls_test_energy_rmse": 0.08524375243567513,
        "expected_asc_cooks_r_test_energy_rmse": 0.03925666139635273,
        "expected_asc_cooks_l_test_energy_rmse": 0.04745908638574153,
    }

    return {
        "X": X,
        "y": y,
        "w": w,
        "config_idxs": config_idxs,
        "device": device,
        **expected,
    }