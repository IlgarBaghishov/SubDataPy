import numpy as np
import warnings
from .random import RandomSubSampler
from .leverage import LeverageSubSampler


class CookSubSampler(RandomSubSampler):

    def __init__(self, X, compute_lib='numpy', y=None, w=None, config_idxs=None, block=False, sequential=True,
                 sampling=True, in_reverse=True, initial_subsampler="leverage", U=None, S=None, Vh=None):

        super().__init__(X, compute_lib=compute_lib, y=y, w=w, config_idxs=config_idxs)
        self.block = block
        self.sequential = sequential
        self.sampling = sampling
        self.in_reverse = in_reverse
        self.initial_subsampler = initial_subsampler
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
            if self.sequential:
                raise NotImplementedError("Block Sequential Cook's Distance methods are not implemented yet.")
            else:
                if self.sampling:
                    pass
                else:
                    pass
        else:
            if self.sequential:
                if self.in_reverse:
                    if self.initial_subsampler == "leverage":
                        lss = LeverageSubSampler(self.X, compute_lib=self.compute_lib, y=self.y, w=self.w,
                                                 config_idxs=self.config_idxs, block=False)
                        self.sub_mask = lss.create_subsample(sample_fraction=self.sample_fraction, seed=self.seed)
                    elif self.initial_subsampler == "random":
                        rss = RandomSubSampler(self.X, compute_lib=self.compute_lib, y=self.y, w=self.w,
                                               config_idxs=self.config_idxs)
                        self.sub_mask = rss.create_subsample(sample_fraction=self.sample_fraction, seed=self.seed)
                    else:
                        raise ValueError("initial_subsampler must be 'leverage' or 'random'.")
                else:
                    pass
            else:
                if self.sampling:
                    pass
                else:
                    pass