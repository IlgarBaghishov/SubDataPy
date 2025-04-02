import numpy as np
import warnings
from subdatapy.data import BaseData


class RandomSubSampler(BaseData):

    def __init__(self, X, compute_lib='numpy', y=None, w=None, config_idxs=None):

        super().__init__(X, compute_lib=compute_lib, y=y, w=w, config_idxs=config_idxs)


    def create_subsample(self, sample_fraction, seed=None):
        
        self.seed = seed
        self.sample_fraction = sample_fraction
        np.random.seed(self.seed)
        if self.sample_fraction <= 0 or self.sample_fraction > 1:
            raise ValueError("sample_fraction must be between 0 and 1")

        self._create_sub_mask()
        
        self._subsample()

        return self.sub_mask


    def _create_sub_mask(self):

        if self.config_idxs is None:
            warnings.warn("config_idxs is None. Will remove rows randomly without "
                          "grouping into configurations.", UserWarning)
            self.config_idxs = np.arange(self.X.shape[0])
            unique_config_idxs = self.config_idxs
        else:
            unique_config_idxs = np.unique(self.config_idxs)

        sub_unique_config_idxs = np.random.choice(unique_config_idxs, 
                                                  size=int(len(unique_config_idxs)*self.sample_fraction),
                                                  replace=False)
        self.sub_mask = sub_unique_config_idxs if self.config_idxs is None else np.isin(self.config_idxs, sub_unique_config_idxs)


    def _subsample(self):
                
        self.sub_config_idxs = self.config_idxs[self.sub_mask]
        self.sub_X = self.X[self.sub_mask]
        if self.y is not None:
            self.sub_y = self.y[self.sub_mask]
        if self.w is not None:
            self.sub_w = self.w[self.sub_mask]
        if self.enrow_mask is not None:
            self.sub_enrow_mask = self.enrow_mask[self.sub_mask]