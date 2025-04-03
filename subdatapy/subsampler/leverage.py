import numpy as np
import warnings
from .random import RandomSubSampler


class LeverageSubSampler(RandomSubSampler):

    def __init__(self, X, y=None, w=None, compute_lib='numpy', config_idxs=None, block=False,
                 U=None, S=None, Vh=None):

        super().__init__(X, y=y, w=w, compute_lib=compute_lib, config_idxs=config_idxs)
        self.block = block
        self.U = U
        self.S = S
        self.Vh = Vh


    def _create_sub_mask(self):
        
        if self.config_idxs is None:
            warnings.warn("config_idxs is None. Will remove rows with leverage but without "
                          "grouping into configurations.", UserWarning)
            self.config_idxs = np.arange(self.X.shape[0])
            unique_config_idxs = self.config_idxs
        else:
            unique_config_idxs = np.unique(self.config_idxs)

        if self.U is None:
            self.U, self.S, self.Vh = np.linalg.svd(self.w.reshape([-1,1])*self.X, full_matrices=False)
        self.leverage_scores = np.sum(self.U**2, axis=1)
        if self.block:
            self.leverage_scores = np.array([np.sum(self.leverage_scores[self.config_idxs == unique_config_idx])
                                             for unique_config_idx in unique_config_idxs])
        else:
            self.leverage_scores = np.array([self.leverage_scores[self.config_idxs == unique_config_idx][0]
                                             for unique_config_idx in unique_config_idxs])
        leverage_probabilities = self.leverage_scores / np.sum(self.leverage_scores)
        sub_unique_config_idxs = np.random.choice(unique_config_idxs,
                                                  size=int(len(unique_config_idxs)*self.sample_fraction),
                                                  replace=False, p=leverage_probabilities)
        self.sub_mask = np.isin(self.config_idxs, sub_unique_config_idxs)
