import numpy as np
import pandas as pd
import warnings
import cupynumeric as cpn
from legate.io.hdf5 import from_file
import gc
import h5py


def process_data(x, lib, dataset_name=None):

    if isinstance(x, lib.ndarray):
        print("Data is already a lib.ndarray", flush=True)
    elif isinstance(x, pd.DataFrame):
        x = lib.asarray(x.values)
    elif isinstance(x, str):
        if x.endswith('.npy'):
            x = lib.load(x)
        elif x.endswith('.hdf5'):
            if dataset_name:
                if lib == np:
                    with h5py.File(x, 'r') as f:
                        x = f[dataset_name][:]
                else:
                    x = lib.asarray(from_file(x, dataset_name=dataset_name))
            else:
                raise ValueError('dataset_name should be provided when using .hdf5')
        else:
            raise ValueError('File format not supported')
    else:
        raise ValueError('Input not supported')
    return x




class BaseData:

    def __init__(self, X, y=None, w=None, config_idxs=None, enrow_mask=None, compute_lib='numpy'):

        """
        Base class for data handling in SubDataPy.
        :param X: Design Matrix of predictor features (independent variables) X rows are data points and columns are features
        :param compute_lib: Library to use for computation ('numpy' or 'dask')
        :param y: Optional respose feature (dependent variable) y
        :param w: Optional weights vector
        :param test_fraction: Fraction of data to be used for testing
        :param config_idxs: Optional configuration index vector
        :param enrow_mask: Optional mask for energy rows
        """
        
        if compute_lib not in ['numpy', 'cupynumeric']:
            raise ValueError("compute_lib must be 'numpy' or 'cupynumeric'")
        self.compute_lib = compute_lib
        self.lib = np if compute_lib == 'numpy' else cpn
        print(f"Using compute library: {self.compute_lib}", flush=True)

        self.coeffs = None

        self.X = process_data(X, self.lib, dataset_name='X')

        self.y = None if y is None else process_data(y, self.lib, dataset_name='y')
        self.w = self.lib.ones(self.X.shape[0]) if w is None else process_data(w, self.lib, dataset_name='w')
        if enrow_mask is not None:
            self.enrow_mask = process_data(enrow_mask, self.lib, dataset_name='enrow_mask')
        if config_idxs is None:
            warnings.warn("config_idxs is None. So, no grouping into configurations.", UserWarning)
            self.config_idxs = self.lib.arange(self.X.shape[0])
            self.unique_config_idxs = self.config_idxs
            if enrow_mask is None:
                self.enrow_mask = self.lib.full_like(self.config_idxs, True, dtype=bool)
        else:
            self.config_idxs = process_data(config_idxs, self.lib, dataset_name='config_idxs')
            if self.compute_lib == 'numpy':
                self.unique_config_idxs, unique_config_first_idxs = self.lib.unique(self.config_idxs, return_index=True)
            else:
                sorted_idx = self.lib.argsort(self.config_idxs)
                sorted_config = self.config_idxs[sorted_idx]
                self.unique_config_idxs = self.lib.unique(sorted_config)
                unique_config_first_idxs = sorted_idx[self.lib.searchsorted(sorted_config, self.unique_config_idxs)]

            if enrow_mask is None:
                warnings.warn("enrow_mask is None. Considering first element of each configuration as energy row", UserWarning)
                self.enrow_mask = self.lib.zeros_like(self.config_idxs, dtype=bool)
                self.enrow_mask[unique_config_first_idxs] = True


    def train_test_split(self, test_fraction=0.0, seed=None, test_mask=None):
        
        self.test_fraction = test_fraction
        self.seed = seed

        if test_mask is None:
            if self.test_fraction == 0.0:
                self.test_mask = self.lib.zeros_like(self.config_idxs, dtype=bool)
                self.train_mask = self.lib.ones_like(self.config_idxs, dtype=bool)
            elif 0.0 < self.test_fraction < 1.0:
                np.random.seed(self.seed)
                print(f"Splitting with test_fraction={self.test_fraction}, seed={self.seed}", flush=True)
                unique_config_idxs_test = self.lib.random.choice(
                    self.unique_config_idxs,
                    size=int(len(self.unique_config_idxs) * self.test_fraction),
                    replace=False
                )
                self.test_mask = self.lib.isin(self.config_idxs, self.lib.asarray(unique_config_idxs_test))
                self.train_mask = ~self.test_mask
            else:
                raise ValueError("test_fraction must be between 0 and 1")
        else:
            if test_mask.shape != self.config_idxs.shape:
                raise ValueError("test_mask shape must be same as config_idxs shape")
            self.test_mask = test_mask
            self.train_mask = ~self.test_mask


        if self.compute_lib == "numpy":
            self.X_train, self.X_test = self.X[self.train_mask], self.X[self.test_mask]
            del self.X

        else:
            n_rows = self.X.shape[0]
            batch_size = 100_000

            train_batches = []
            for i in range(0, n_rows, batch_size):
                end = min(i + batch_size, n_rows)
                batch = self.X[i:end]
                batch_train_mask = self.train_mask[i:end]
                train_batches.append(batch[batch_train_mask])
    
            self.X_train = self.lib.concatenate(train_batches, axis=0)
            print(self.X_train[:2, :10], flush=True)
    
            del batch, batch_train_mask, train_batches
            gc.collect()
    
            test_batches = []
            for i in range(0, n_rows, batch_size):
                end = min(i + batch_size, n_rows)
                batch = self.X[i:end]
                batch_test_mask = self.test_mask[i:end]
                test_batches.append(batch[batch_test_mask])
            del self.X
            self.X_test = self.lib.concatenate(test_batches, axis=0)
            print(self.X_test[:2, :10], flush=True)
    
            del batch, batch_test_mask, test_batches
            gc.collect()

        if self.y is not None:
            self.y_train = self.y[self.train_mask]
            self.y_test = self.y[self.test_mask]
        else:
            self.y_train = self.y_test = None

        self.w_train = self.w[self.train_mask]
        self.w_test = self.w[self.test_mask]

        self.config_idxs_train = self.config_idxs[self.train_mask]
        self.config_idxs_test = self.config_idxs[self.test_mask]

        self.unique_config_idxs_train = self.lib.unique(self.config_idxs_train)
        self.unique_config_idxs_test = self.lib.unique(self.config_idxs_test)

        self.enrow_mask_train = self.enrow_mask[self.train_mask]
        self.enrow_mask_test = self.enrow_mask[self.test_mask]


    def train(self):

        if self.compute_lib == 'numpy':
            self.coeffs, *_ = self.lib.linalg.lstsq(
                self.w_train.reshape([-1,1]) * self.X_train,
                self.w_train.reshape([-1,1]) * self.y_train.reshape([-1,1])
            )
        else:
            Q, R = self.lib.linalg.qr(self.w_train.reshape([-1,1]) * self.X_train)
            b_w = self.w_train.reshape([-1,1]) * self.y_train.reshape([-1,1])
            self.coeffs = self.lib.linalg.solve(R, Q.T @ b_w)
            

    def compute_errors(self, X_test=None, y_test=None, enrow_mask_test=None):

        reshaped_y_train = self.y_train.reshape(-1, 1)
        residuals_train = self.lib.dot(self.X_train, self.coeffs) - reshaped_y_train

        squared_residuals_train = self.lib.square(residuals_train)
        masked_squared_residuals = squared_residuals_train[self.enrow_mask_train]
        energy_train_rmse = self.lib.sqrt(cpn.mean(masked_squared_residuals))
        print("Energy training RMSE:", energy_train_rmse, flush=True)


        force_train_rmse = None
        if not self.lib.all(self.enrow_mask_train):
            force_train_rmse = self.lib.sqrt(self.lib.mean(squared_residuals_train[~self.enrow_mask_train]))
            print("Force training RMSE:", force_train_rmse, flush=True)

        energy_test_rmse = None
        force_test_rmse = None
        if self.X_test.shape[0] != 0:
            residuals_test = self.lib.dot(self.X_test, self.coeffs) - self.y_test.reshape(-1, 1)
            squared_residuals_test = self.lib.square(residuals_test)

            energy_test_rmse = self.lib.sqrt(cpn.mean(squared_residuals_test[self.enrow_mask_test]))
            print("Energy test RMSE:", energy_test_rmse, flush=True)

            if not self.lib.all(self.enrow_mask_test):
                force_test_rmse = self.lib.sqrt(cpn.mean(squared_residuals_test[~self.enrow_mask_test]))
                print("Force test RMSE:", force_test_rmse, flush=True)

        if X_test is not None and y_test is not None:
            if enrow_mask_test is None:
                enrow_mask_test = self.lib.full_like(y_test.reshape(-1), True, dtype=bool)

            residuals_ext_test = self.lib.dot(X_test, self.coeffs) - y_test.reshape(-1, 1)
            squared_residuals_ext_test = self.lib.square(residuals_ext_test)

            ext_energy_test_rmse = self.lib.sqrt(self.lib.mean(squared_residuals_ext_test[enrow_mask_test]))
            print("External Energy test RMSE:", ext_energy_test_rmse, flush=True)

            if not self.lib.all(enrow_mask_test):
                ext_force_test_rmse = self.lib.sqrt(self.lib.mean(squared_residuals_ext_test[~enrow_mask_test]))
                print("External Force test RMSE:", ext_force_test_rmse, flush=True)

            return energy_train_rmse, force_train_rmse, ext_energy_test_rmse, ext_force_test_rmse

        return energy_train_rmse, force_train_rmse, energy_test_rmse, force_test_rmse