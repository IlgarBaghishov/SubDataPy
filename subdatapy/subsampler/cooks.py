import numpy as np
import warnings
from .random import RandomSubSampler
from .leverage import LeverageSubSampler


class CookSubSampler(RandomSubSampler):

    def __init__(self, X, y, w=None, test_fraction=0.0, seed=None, test_mask=None, config_idxs=None, enrow_mask=None, compute_lib='numpy',
                 block=False, stepwise=False, sampling=True, ascending=False, initial_subsampler="leverage",
                 initial_subsample_fraction=1, U=None, S=None, Vh=None):

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed, test_mask=test_mask, 
                         config_idxs=config_idxs, enrow_mask=enrow_mask, compute_lib=compute_lib)
        self.block = block
        self.stepwise = stepwise
        self.sampling = sampling
        self.ascending = ascending
        self.initial_subsampler = initial_subsampler
        self.initial_subsample_fraction = initial_subsample_fraction
        self.U = U
        self.S = S
        self.Vh = Vh

        self.onestep_en_cooks = None
        self.leverage_scores = None
        self.XTX_inv = None
        self.XTy = None



    def _create_sub_mask(self):

        self.X_train = self.w_train.reshape([-1,1]) * self.X_train
        self.y_train = self.w_train.reshape([-1,1]) * self.y_train.reshape([-1,1])

        if self.stepwise:
            self._stepwise_cooks_sampling()
        else:
            self._onestep_cooks_sampling()

        self.X_train = self.X_train / self.w_train.reshape([-1,1])
        self.y_train = self.y_train / self.w_train.reshape([-1,1])

    

    def _onestep_cooks_sampling(self):

        if self.block:
            raise NotImplementedError("Onestep Block Cook's Distance methods are not implemented yet.")
        else:
            if self.onestep_en_cooks is None:

                if self.U is None:
                    self.U, self.S, self.Vh = np.linalg.svd(self.X_train, full_matrices=False)
                    self.leverage_scores = np.sum(self.U[self.enrow_mask_train]**2, axis=1)

                tol = np.finfo(float).eps * max(self.X_train.shape) * self.S[0]
                self.S_inv = np.where(self.S > tol, 1/self.S, 0)
                coeffs = self.Vh.T @ (self.S_inv.reshape(-1,1) * (self.U.T @ self.y_train))
                # coeffs = self.Vh.T @ np.diag(self.S) @ self.U.T @ self.y
                en_residuals_sq = np.square(self.X_train[self.enrow_mask_train] @ coeffs - self.y_train[self.enrow_mask_train].reshape(-1,1)).reshape(-1)

                self.onestep_en_cooks = en_residuals_sq * self.leverage_scores / (1-self.leverage_scores)**2
                self.onestep_en_cooks_prob = self.onestep_en_cooks / np.sum(self.onestep_en_cooks)

            if self.sampling:
                sub_unique_config_idxs_train = np.random.choice(self.unique_config_idxs_train, size=self.n_subsamples,
                                                            replace=False, p=self.onestep_en_cooks_prob)
            else:
                sub_unique_config_idxs_train = self.unique_config_idxs_train[np.argsort(self.onestep_en_cooks)[::-1][:self.n_subsamples]]

            self.sub_mask = np.isin(self.config_idxs, sub_unique_config_idxs_train)
            self.sub_mask_train = np.isin(self.config_idxs_train, sub_unique_config_idxs_train)



    def _create_initial_sub_mask(self):
        if self.ascending and self.initial_subsample_fraction == 1:
            raise ValueError("initial_subsample_fraction must be lower than 1 for ascending=True")
        if self.initial_subsample_fraction <= 0 or self.initial_subsample_fraction > 1:
            raise ValueError("initial_subsample_fraction must be between 0 and 1")
        
        if self.initial_subsampler == "leverage":
            lss = LeverageSubSampler(self.X_train, y=self.y_train, seed=self.seed, compute_lib=self.compute_lib,
                                     config_idxs=self.config_idxs_train, block=self.block)
            self.sub_mask_train = lss.create_subsample(subsample_fraction=self.initial_subsample_fraction)
            self.sub_mask = np.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train])

        elif self.initial_subsampler == "random":
            rss = RandomSubSampler(self.X_train, y=self.y_train, w=self.w_train, seed=self.seed, compute_lib=self.compute_lib,
                                   config_idxs=self.config_idxs_train)
            self.sub_mask_train = rss.create_subsample(subsample_fraction=self.initial_subsample_fraction)
            self.sub_mask = np.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train])

        

    def _stepwise_cooks_sampling(self):
        
        if self.block:
            raise NotImplementedError("Stepwise Block Cook's Distance methods are not implemented yet.")
        
        if self.sub_mask is None: self._create_initial_sub_mask()
        n_subsamples_init = np.sum(self.sub_mask * self.enrow_mask)

        if self.S is None and self.Vh is None:
            self.U, self.S, self.Vh = np.linalg.svd(self.X_train[self.sub_mask_train], full_matrices=False)

        if self.XTX_inv is None: self.XTX_inv = self.Vh.T @ np.diag(np.reciprocal(self.S)**2) @ self.Vh
        if self.XTy is None: self.XTy = self.X_train[self.sub_mask_train].T @ self.y_train[self.sub_mask_train]

        if self.ascending:
            for i in range(n_subsamples_init,self.n_subsamples):

                leverage_scores = self.X_train[self.enrow_mask_train] @ self.XTX_inv
                leverage_scores = np.einsum('ij,ji->i', leverage_scores, self.X_train[self.enrow_mask_train].T)
                coeffs = self.XTX_inv @ self.XTy
                en_residuals_sq = np.square(self.X_train[self.enrow_mask_train] @ coeffs - self.y_train[self.enrow_mask_train]).reshape(-1)
                e_cooks = en_residuals_sq * leverage_scores / (1+leverage_scores)

                e_cooks[self.sub_mask_train[self.enrow_mask_train]] = -np.inf
                config_to_add = np.argmax(e_cooks)
                config_to_add_mask = np.isin(self.config_idxs_train, self.unique_config_idxs_train[config_to_add])
                self.sub_mask_train[config_to_add_mask] = True

                # Update self.XTX_inv and self.XTy
                left_update = self.XTX_inv @ self.X_train[config_to_add_mask].T
                inv_update = np.linalg.inv(np.eye(np.sum(config_to_add_mask)) + self.X_train[config_to_add_mask] @ left_update)
                right_update = self.X_train[config_to_add_mask] @ self.XTX_inv
                self.XTX_inv -= left_update @ inv_update @ right_update
                self.XTy += self.X_train[config_to_add_mask].T @ self.y_train[config_to_add_mask]
        else:
            for i in range(n_subsamples_init,self.n_subsamples,-1):
                                
                leverage_scores = self.X_train[self.enrow_mask_train] @ self.XTX_inv
                leverage_scores = np.einsum('ij,ji->i', leverage_scores, self.X_train[self.enrow_mask_train].T)
                coeffs = self.XTX_inv @ self.XTy
                en_residuals_sq = np.square(self.X_train[self.enrow_mask_train] @ coeffs - self.y_train[self.enrow_mask_train]).reshape(-1)
                e_cooks = en_residuals_sq * leverage_scores / (1-leverage_scores)**2

                e_cooks[self.sub_mask_train[self.enrow_mask_train]] = np.inf
                config_to_remove = np.argmin(e_cooks)
                config_to_remove_mask = np.isin(self.config_idxs_train, self.unique_config_idxs_train[config_to_remove])
                self.sub_mask_train[config_to_remove_mask] = False

                # Update self.XTX_inv and self.XTy
                left_update = self.XTX_inv @ self.X_train[config_to_remove_mask].T
                inv_update = np.linalg.inv(np.eye(np.sum(config_to_remove_mask)) - self.X_train[config_to_remove_mask] @ left_update)
                right_update = self.X_train[config_to_remove_mask] @ self.XTX_inv
                self.XTX_inv += left_update @ inv_update @ right_update
                self.XTy -= self.X_train[config_to_remove_mask].T @ self.y_train[config_to_remove_mask]

        self.sub_mask = np.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train])