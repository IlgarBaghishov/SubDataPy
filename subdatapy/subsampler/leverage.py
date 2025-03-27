import numpy as np
import warnings
from subdatapy.data import BaseData


class LeverageSubSampler(BaseData):

    def __init__(self, X, compute_lib='numpy', y=None, w=None, config_ind=None, block=False,
                 U=None, S=None, Vh=None):

        super().__init__(X, compute_lib=compute_lib, y=y, w=w, config_ind=config_ind)
        self.block = block
        self.U = U
        self.S = S
        self.Vh = Vh


    def create_subsample(self, sample_fraction, seed=None):

        np.random.seed(seed)
        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        
        if self.config_ind is None:
            warnings.warn("config_ind is None. Will remove rows with leverage but without "
                          "grouping into configurations.", UserWarning)
            self.config_ind = np.arange(self.X.shape[0])
            unique_configs = self.config_ind
        else:
            unique_configs = np.unique(self.config_ind)

        if self.U is None:
            self.U, self.S, self.Vh = np.linalg.svd(self.w.reshape([-1,1])*self.X, full_matrices=False)
        self.leverage_scores = np.sum(self.U**2, axis=1)
        if self.block:
            self.leverage_scores = np.array([np.sum(self.leverage_scores[self.config_ind == unique_config])
                                             for unique_config in unique_configs])
        else:
            self.leverage_scores = np.array([self.leverage_scores[self.config_ind == unique_config][0]
                                             for unique_config in unique_configs])
        leverage_probabilities = self.leverage_scores / np.sum(self.leverage_scores)
        sampled_indices = np.random.choice(unique_configs, size=int(len(unique_configs) * sample_fraction),
                                           replace=False, p=leverage_probabilities)
        mask = sampled_indices if self.config_ind is None else np.isin(self.config_ind, sampled_indices)
        
        self.config_ind = self.config_ind[mask]
        self.X_sub = self.X[mask]
        if self.y is not None:
            self.y_sub = self.y[mask]
        if self.w is not None:
            self.w_sub = self.w[mask]
        if self.row_mask is not None:
            self.row_mask = self.row_mask[mask]

        return self.config_ind