import numpy as np
import warnings
from .random import RandomSubSampler
from .leverage import LeverageSubSampler


class CookSubSampler(RandomSubSampler):

    def __init__(self, X, y, w=None, compute_lib='numpy', config_idxs=None, block=False, stepwise=True,
                 sampling=True, in_reverse=True, initial_subsampler="leverage", U=None, S=None, Vh=None):

        super().__init__(X, y=y, w=w, compute_lib=compute_lib, config_idxs=config_idxs)
        self.block = block
        self.stepwise = stepwise
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
            self.unique_config_idxs = self.config_idxs
        else:
            self.unique_config_idxs = np.unique(self.config_idxs)

        if self.stepwise:
            self._stepwise_cooks_sampling()
        else:
            self._onestep_cooks_sampling()

        
    def _stepwise_cooks_sampling(self):

        self.X = self.w.reshape([-1,1])*self.X
        self.y = self.w.reshape([-1,1])*self.y.reshape([-1,1])

        if self.initial_subsampler == "leverage":
            lss = LeverageSubSampler(self.X, y=self.y, w=self.w, compute_lib=self.compute_lib,
                                        config_idxs=self.config_idxs, block=False)
            self.sub_mask = lss.create_subsample(sample_fraction=self.sample_fraction, seed=self.seed)
        elif self.initial_subsampler == "random":
            rss = RandomSubSampler(self.X, y=self.y, w=self.w, compute_lib=self.compute_lib,
                                    config_idxs=self.config_idxs)
            self.sub_mask = rss.create_subsample(sample_fraction=self.sample_fraction, seed=self.seed)

        if self.U is None:
            self.U, self.S, self.Vh = np.linalg.svd(self.X, full_matrices=False)
        
        if self.block:
            raise NotImplementedError("Onestep Block Cook's Distance methods are not implemented yet.")
        else:
            self.leverage_scores = np.sum(self.U[self.enrow_mask]**2, axis=1)

        XTX_inv = self.Vh.T @ np.diag(np.reciprocal(self.S)**2) @ self.Vh

        leverage_scores = self.X @ XTX_inv
        leverage_scores = np.einsum('ij,ji->i', leverage_scores, self.X.T)

        coeffs = XTX_inv @ (self.X.T @ self.y)
        en_residuals_sq = np.square(self.X[self.enrow_mask] @ coeffs - self.y[self.enrow_mask].reshape(-1,1)).reshape(-1)

        # try this 2nd method and see what's faster and if it these two methods are equivalent:
        # leverage_scores = XTX_inv @ self.X.T
        # coeffs = leverage_scores @ self.y
        # en_residuals_sq = np.square(self.X[self.enrow_mask] @ coeffs - self.y[self.enrow_mask].reshape(-1,1)).reshape(-1)
        # leverage_scores = np.einsum('ij,ji->i', self.X, leverage_scores)

        e_cooks = en_residuals_sq * leverage_scores / (1-leverage_scores)**2

    
    def _onestep_cooks_sampling(self):

        self.X = self.w.reshape([-1,1])*self.X
        self.y = self.w.reshape([-1,1])*self.y.reshape([-1,1])

        if self.U is None:
            self.U, self.S, self.Vh = np.linalg.svd(self.X, full_matrices=False)
        
        if self.block:
            raise NotImplementedError("Onestep Block Cook's Distance methods are not implemented yet.")
        else:
            leverage_scores = np.sum(self.U[self.enrow_mask]**2, axis=1)

            tol = np.finfo(float).eps * max(self.X.shape) * self.S[0]
            self.S_inv = np.where(self.S > tol, 1/self.S, 0)
            coeffs = self.Vh.T @ (self.S_inv.reshape(-1,1) * (self.U.T @ self.y))
            # coeffs = self.Vh.T @ np.diag(self.S) @ self.U.T @ self.y
            en_residuals_sq = np.square(self.X[self.enrow_mask] @ coeffs - self.y[self.enrow_mask].reshape(-1,1)).reshape(-1)

            # # try this 2nd method and see what's faster and if it these two methods are equivalent:
            # leverage_scores = XTX_inv @ self.X.T
            # coeffs = leverage_scores @ self.y
            # en_residuals_sq = np.square(self.X[self.enrow_mask] @ coeffs - self.y[self.enrow_mask].reshape(-1,1)).reshape(-1)
            # leverage_scores = np.einsum('ij,ji->i', self.X, leverage_scores)

            e_cooks = en_residuals_sq * leverage_scores / (1-leverage_scores)**2
            e_cooks_probabilities = e_cooks / np.sum(e_cooks)

            n_samples = int(len(self.unique_config_idxs)*self.sample_fraction)
            if self.sampling:
                sub_unique_config_idxs = np.random.choice(self.unique_config_idxs, size=n_samples,
                                                            replace=False, p=e_cooks_probabilities)
            else:
                sub_unique_config_idxs = self.unique_config_idxs[np.argsort(e_cooks)[::-1][:n_samples]]

            self.sub_mask = np.isin(self.config_idxs, sub_unique_config_idxs)
