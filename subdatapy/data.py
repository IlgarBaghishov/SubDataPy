import numpy as np
import dask.array as da
import pandas as pd


def process_data(x, compute_lib):
    if isinstance(x, np.ndarray) or isinstance(x, da.Array):
        pass
    elif isinstance(x, pd.DataFrame):
        x = x.values
    elif isinstance(x, str):
        if x.endswith('.npy') and compute_lib == 'numpy':
            x = np.load(x)
        elif x.endswith('.npy') and compute_lib == 'dask':
            x = da.from_array(np.load(x))
        elif x.endswith('.zarr') and compute_lib == 'dask':
            x = da.from_zarr(x)
        else:
            raise ValueError('File format not supported')
    else:
        raise ValueError('Input not supported')
    return x


class BaseData:

    def __init__(self, X, compute_lib='numpy', y=None, w=None, config_idxs=None, enrow_mask=None):
        """
        Base class for data handling in SubDataPy.
        :param X: Design Matrix of predictor features (independent variables) X rows are data points and columns are features
        :param compute_lib: Library to use for computation ('numpy' or 'dask')
        :param y: Optional respose feature (dependent variable) y
        :param w: Optional weights vector
        :param config_idxs: Optional configuration index vector
        :param enrow_mask: Optional mask for energy rows
        """

        if compute_lib not in ['numpy', 'dask']:
            raise ValueError("compute_lib must be 'numpy' or 'dask'")
        self.compute_lib = compute_lib

        self.X = process_data(X, compute_lib=self.compute_lib)
        self.y = None if y is None else process_data(y, compute_lib=self.compute_lib)
        self.w = None if w is None else process_data(w, compute_lib=self.compute_lib)
        self.config_idxs = None if config_idxs is None else process_data(config_idxs, compute_lib=self.compute_lib)
        self.enrow_mask = None if enrow_mask is None else process_data(enrow_mask, compute_lib=self.compute_lib)
        