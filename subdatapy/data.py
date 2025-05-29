import numpy as np
import pandas as pd
import warnings


def process_data(x, compute_lib):
    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, pd.DataFrame):
        x = x.values
    elif isinstance(x, str):
        if x.endswith('.npy') and compute_lib == 'numpy':
            x = np.load(x)
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

        if compute_lib not in ['numpy', 'dask']:
            raise ValueError("compute_lib must be 'numpy' or 'dask'")
        self.compute_lib = compute_lib
        self.coeffs = None

        self.X = process_data(X, compute_lib=self.compute_lib)
        self.y = None if y is None else process_data(y, compute_lib=self.compute_lib)
        self.w = np.ones(self.X.shape[0]) if w is None else process_data(w, compute_lib=self.compute_lib)
        if enrow_mask is not None: self.enrow_mask = process_data(enrow_mask, compute_lib=self.compute_lib)

        if config_idxs is None:
            warnings.warn("config_idxs is None. So, no grouping into configurations.", UserWarning)
            self.config_idxs = np.arange(self.X.shape[0])
            self.unique_config_idxs = self.config_idxs
            if enrow_mask is None: self.enrow_mask = np.full_like(self.config_idxs, True, dtype=bool)
        else:
            self.config_idxs = process_data(config_idxs, compute_lib=self.compute_lib)
            self.unique_config_idxs, unique_config_first_idxs = np.unique(self.config_idxs, return_index=True)
            if enrow_mask is None:
                warnings.warn("enrow_mask is None. Considering first element of each configuration as energy row", UserWarning)
                self.enrow_mask = np.zeros_like(self.config_idxs, dtype=bool)
                self.enrow_mask[unique_config_first_idxs] = True



    def train_test_split(self, test_fraction=0.0, seed=None, test_mask=None):

        # if self.y is None: raise ValueError("y is None. Cannot split data without y.")
        self.test_fraction = test_fraction
        self.seed = seed

        if test_mask is None:
            if self.test_fraction == 0.0:
                self.test_mask = np.zeros_like(self.config_idxs, dtype=bool)
                self.train_mask = np.ones_like(self.config_idxs, dtype=bool)
            elif self.test_fraction > 0.0 and self.test_fraction < 1.0:
                np.random.seed(self.seed)
                unique_config_idxs_test = np.random.choice(self.unique_config_idxs,
                                                        size=int(len(self.unique_config_idxs)*self.test_fraction),
                                                        replace=False)
                self.test_mask = np.isin(self.config_idxs, unique_config_idxs_test)
                self.train_mask = ~self.test_mask
            else:
                raise ValueError("test_fraction must be between 0 and 1")
        else:
            if test_mask.shape != self.config_idxs.shape:
                raise ValueError("test_mask shape must be same as config_idxs shape")
            self.test_mask = test_mask
            self.train_mask = ~self.test_mask
        
        self.X_train, self.X_test = self.X[self.train_mask], self.X[self.test_mask]
        if self.y is None:
            self.y_train, self.y_test = None, None
        else:
            self.y_train, self.y_test = self.y[self.train_mask], self.y[self.test_mask]
        self.w_train, self.w_test = self.w[self.train_mask], self.w[self.test_mask]
        self.config_idxs_train, self.config_idxs_test = self.config_idxs[self.train_mask], self.config_idxs[self.test_mask]
        self.unique_config_idxs_train, self.unique_config_idxs_test = np.unique(self.config_idxs_train), np.unique(self.config_idxs_test)
        # self.unique_config_idxs_train, self.unique_config_idxs_test = self.unique_config_idxs[self.train_mask], self.unique_config_idxs[self.test_mask]
        self.enrow_mask_train, self.enrow_mask_test = self.enrow_mask[self.train_mask], self.enrow_mask[self.test_mask]



    def train(self):

        self.coeffs, *_ = np.linalg.lstsq(self.w_train.reshape([-1,1])*self.X_train, self.w_train.reshape([-1,1])*self.y_train.reshape([-1,1]))
    


    def compute_errors(self, X_test=None, y_test=None, enrow_mask_test=None):

        self.squared_residuals = np.square(np.dot(self.X,self.coeffs) - self.y.reshape(-1,1))

        energy_train_rmse = np.sqrt(np.mean(self.squared_residuals[self.train_mask*self.enrow_mask]))
        print("Energy training RMSE is", energy_train_rmse)
        if ~np.all(self.enrow_mask):
            force_train_rmse = np.sqrt(np.mean(self.squared_residuals[self.train_mask*(~self.enrow_mask)]))
            print("Force training RMSE is", force_train_rmse)

        if self.X_test.shape[0] != 0:
            energy_test_rmse = np.sqrt(np.mean(self.squared_residuals[self.test_mask*self.enrow_mask]))
            print("Energy test RMSE is", energy_test_rmse)
            if ~np.all(self.enrow_mask):
                force_test_rmse = np.sqrt(np.mean(self.squared_residuals[self.test_mask*(~self.enrow_mask)]))
                print("Force test RMSE is", force_test_rmse)

        if X_test is not None and y_test is not None:
            if enrow_mask_test is None: enrow_mask_test = np.full_like(y_test.reshape(-1), True, dtype=bool)
            squared_residuals_test = np.square(np.dot(X_test, self.coeffs) - y_test.reshape(-1,1))
            energy_test_rmse = np.sqrt(np.mean(squared_residuals_test[enrow_mask_test]))
            print("Energy test RMSE is", energy_test_rmse)
            if ~np.all(enrow_mask_test):
                force_test_rmse = np.sqrt(np.mean(squared_residuals_test[~enrow_mask_test]))
                print("Force test RMSE is", force_test_rmse)

        return energy_train_rmse, force_train_rmse, energy_test_rmse, force_test_rmse