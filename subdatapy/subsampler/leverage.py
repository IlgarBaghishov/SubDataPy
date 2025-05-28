import numpy as np
import warnings
from .random import RandomSubSampler


class LeverageSubSampler(RandomSubSampler):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, config_idxs=None, enrow_mask=None, compute_lib='numpy',
                 block=False, U=None, S=None, Vh=None):

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed, config_idxs=config_idxs, enrow_mask=enrow_mask,
                         compute_lib=compute_lib)
        self.block = block
        self.U = U
        self.S = S
        self.Vh = Vh



    def _create_sub_mask(self):

        if self.U is None:
            self.U, self.S, self.Vh = np.linalg.svd(self.w_train.reshape([-1,1])*self.X_train, full_matrices=False)
        self.leverage_scores = np.sum(self.U**2, axis=1)
        if self.block:
            self.leverage_scores = np.array([np.sum(self.leverage_scores[self.config_idxs_train == unique_config_idx])
                                             for unique_config_idx in self.unique_config_idxs_train])
        else:
            self.leverage_scores = np.array([self.leverage_scores[self.config_idxs_train == unique_config_idx][0]
                                             for unique_config_idx in self.unique_config_idxs_train])
        leverage_probabilities = self.leverage_scores / np.sum(self.leverage_scores)
        sub_unique_config_idxs_train = np.random.choice(self.unique_config_idxs_train, size=self.n_subsamples,
                                                  replace=False, p=leverage_probabilities)
        
        self.sub_mask = np.isin(self.config_idxs, sub_unique_config_idxs_train)
        self.sub_mask_train = np.isin(self.config_idxs_train, sub_unique_config_idxs_train)
