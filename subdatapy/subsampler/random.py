import numpy as np
import warnings
from subdatapy.data import BaseData


class RandomSubSampler(BaseData):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, config_idxs=None, enrow_mask=None, compute_lib='numpy'):

        super().__init__(X, y=y, w=w, config_idxs=config_idxs, enrow_mask=enrow_mask,
                         compute_lib=compute_lib)

        self.sub_mask = None
        self.train_test_split(test_fraction=test_fraction, seed=seed)



    def create_subsample(self, subsample_fraction, seed=None):
        
        self.seed = seed
        np.random.seed(self.seed)
        self.subsample_fraction = subsample_fraction
        self.n_subsamples = int(len(self.unique_config_idxs_train)*self.subsample_fraction)

        if self.subsample_fraction <= 0 or self.subsample_fraction > 1:
            raise ValueError("sample_fraction must be between 0 and 1")

        self._create_sub_mask()
        
        self._subsample()

        return self.sub_mask
    


    def train_subsample(self):

        self.sub_coeffs, *_ = np.linalg.lstsq(self.sub_w_train.reshape([-1,1])*self.sub_X_train, self.sub_w_train.reshape([-1,1])*self.sub_y_train.reshape([-1,1]))
    


    def compute_subsample_errors(self, X_test=None, y_test=None, enrow_mask_test=None):
        energy_subtrain_rmse, force_subtrain_rmse, energy_nonsubtrain_rmse, force_nonsubtrain_rmse, energy_test_rmse, force_test_rmse = None, None, None, None, None, None
        self.squared_residuals = np.square(np.dot(self.X,self.sub_coeffs) - self.y.reshape(-1,1))

        energy_subtrain_rmse = np.sqrt(np.mean(self.squared_residuals[self.sub_mask*self.enrow_mask]))
        print("Subsampled data Energy training RMSE is", energy_subtrain_rmse)
        if ~np.all(self.enrow_mask):
            force_subtrain_rmse = np.sqrt(np.mean(self.squared_residuals[self.sub_mask*(~self.enrow_mask)]))
            print("Force training RMSE is", force_subtrain_rmse)

        energy_nonsubtrain_rmse = np.sqrt(np.mean(self.squared_residuals[(~self.sub_mask)*self.train_mask*self.enrow_mask]))
        print("Remaining data Energy training RMSE is", energy_nonsubtrain_rmse)
        if ~np.all(self.enrow_mask):
            force_nonsubtrain_rmse = np.sqrt(np.mean(self.squared_residuals[(~self.sub_mask)*self.train_mask*(~self.enrow_mask)]))
            print("Force training RMSE is", force_nonsubtrain_rmse)

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

        return energy_subtrain_rmse, force_subtrain_rmse, energy_nonsubtrain_rmse, force_nonsubtrain_rmse, energy_test_rmse, force_test_rmse



    def create_subsample_errors_sequence(self, list_of_subsample_fractions, seed=None):
        
        error_sequence = []

        for subsample_fraction in list_of_subsample_fractions:
            self.create_subsample(subsample_fraction=subsample_fraction, seed=seed)
            self.train_subsample()
            error_sequence.append([subsample_fraction,self.compute_subsample_errors()])

        return error_sequence
            


    def _create_sub_mask(self):

        sub_unique_config_idxs_train = np.random.choice(self.unique_config_idxs_train, size=self.n_subsamples, replace=False)
        self.sub_mask = np.isin(self.config_idxs, sub_unique_config_idxs_train)
        self.sub_mask_train = np.isin(self.config_idxs_train, sub_unique_config_idxs_train)



    def _subsample(self):
                
        self.sub_config_idxs_train = self.config_idxs[self.sub_mask]
        self.sub_X_train = self.X[self.sub_mask]
        if self.y is not None:
            self.sub_y_train = self.y[self.sub_mask]
        if self.w is not None:
            self.sub_w_train = self.w[self.sub_mask]
        if self.enrow_mask is not None:
            self.sub_enrow_mask_train = self.enrow_mask[self.sub_mask]