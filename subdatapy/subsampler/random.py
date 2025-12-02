import numpy as np
import pandas as pd
import warnings
from subdatapy.data import BaseData
from collections import defaultdict


class RandomSubSampler(BaseData):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, test_mask=None, config_idxs=None, 
                 enrow_mask=None, intercept=True, compute_lib='numpy'):

        super().__init__(X, y=y, w=w, config_idxs=config_idxs, enrow_mask=enrow_mask, intercept=intercept,
                         compute_lib=compute_lib)

        self.sub_mask = None
        self.train_test_split(test_fraction=test_fraction, seed=seed, test_mask=test_mask)



    def create_subsample(self, subsample_fraction, seed=None):
        
        if seed != self.seed:
            self.seed = seed
            np.random.seed(self.seed)
        self.subsample_fraction = subsample_fraction
        self.n_subsamples = round(len(self.unique_config_idxs_train)*self.subsample_fraction)

        if self.subsample_fraction <= 0 or self.subsample_fraction > 1:
            raise ValueError("sample_fraction must be between 0 and 1")

        self._create_sub_mask()
        
        self._subsample()

        return self.sub_mask
    


    def train_subsample(self):

        self.sub_coeffs, *_ = np.linalg.lstsq(self.sub_w_train.reshape([-1,1])*self.sub_X_train, self.sub_w_train.reshape([-1,1])*self.sub_y_train.reshape([-1,1]))
    


    def compute_subsample_errors(self, X_test=None, y_test=None, enrow_mask_test=None, verbose=False):
        force_subtrain_rmse, force_nonsubtrain_rmse, energy_test_rmse, force_test_rmse = None, None, None, None
        self.squared_residuals = np.square(np.dot(self.X,self.sub_coeffs) - self.y.reshape(-1,1))

        energy_subtrain_rmse = np.sqrt(np.mean(self.squared_residuals[self.sub_mask*self.enrow_mask]))
        if verbose: print("Subsampled data Energy training RMSE is", energy_subtrain_rmse)
        if ~np.all(self.enrow_mask):
            force_subtrain_rmse = np.sqrt(np.mean(self.squared_residuals[self.sub_mask*(~self.enrow_mask)]))
            if verbose: print("Subsampled data Force training RMSE is", force_subtrain_rmse)

        energy_nonsubtrain_rmse = np.sqrt(np.mean(self.squared_residuals[(~self.sub_mask)*self.train_mask*self.enrow_mask]))
        if verbose: print("Remaining data Energy training RMSE is", energy_nonsubtrain_rmse)
        if ~np.all(self.enrow_mask):
            force_nonsubtrain_rmse = np.sqrt(np.mean(self.squared_residuals[(~self.sub_mask)*self.train_mask*(~self.enrow_mask)]))
            if verbose: print("Remaining data Force training RMSE is", force_nonsubtrain_rmse)

        if X_test is not None and y_test is not None:
            if enrow_mask_test is None: enrow_mask_test = np.full_like(y_test.reshape(-1), True, dtype=bool)
            squared_residuals_test = np.square(np.dot(X_test, self.coeffs) - y_test.reshape(-1,1))
            energy_test_rmse = np.sqrt(np.mean(squared_residuals_test[enrow_mask_test]))
            if verbose: print("Energy test RMSE is", energy_test_rmse)
            if ~np.all(enrow_mask_test):
                force_test_rmse = np.sqrt(np.mean(squared_residuals_test[~enrow_mask_test]))
                if verbose: print("Force test RMSE is", force_test_rmse)
        elif self.X_test.shape[0] != 0:
            energy_test_rmse = np.sqrt(np.mean(self.squared_residuals[self.test_mask*self.enrow_mask]))
            if verbose: print("Energy test RMSE is", energy_test_rmse)
            if ~np.all(self.enrow_mask):
                force_test_rmse = np.sqrt(np.mean(self.squared_residuals[self.test_mask*(~self.enrow_mask)]))
                if verbose: print("Force test RMSE is", force_test_rmse)

        return energy_subtrain_rmse, force_subtrain_rmse, energy_nonsubtrain_rmse, force_nonsubtrain_rmse, energy_test_rmse, force_test_rmse



    def create_subsample_errors_sequence(self, subsample_fractions_list, repeat_count_list=1, seed=None, verbose=False):
        
        if isinstance(subsample_fractions_list, (float,int)):
            subsample_fractions_list = [subsample_fractions_list]
        if isinstance(repeat_count_list, int):
            repeat_count_list = [repeat_count_list] * len(subsample_fractions_list)
        elif len(subsample_fractions_list) != len(repeat_count_list):
            raise ValueError("subsample_fractions_list and repeat_count_list must have the same length if repeat_count_list is a list")
        
        error_sequence = []
        for i,subsample_fraction in enumerate(subsample_fractions_list):
            for _ in range(repeat_count_list[i]):
                self.create_subsample(subsample_fraction=subsample_fraction, seed=seed)
                self.train_subsample()
                error_sequence.append([subsample_fraction,self.compute_subsample_errors(verbose=verbose)])

        return error_sequence



    def create_subsample_errors_dataframe(self, subsample_fractions_list, repeat_count_list=1, seed=None, verbose=False):

        if isinstance(subsample_fractions_list, (float,int)):
            subsample_fractions_list = [subsample_fractions_list]
        if isinstance(repeat_count_list, int):
            repeat_count_list = [repeat_count_list] * len(subsample_fractions_list)
        elif len(subsample_fractions_list) != len(repeat_count_list):
            raise ValueError("subsample_fractions_list and repeat_count_list must have the same length if repeat_count_list is a list")
        paired_lists = list(zip(repeat_count_list, subsample_fractions_list))
        sorted_paired_lists = sorted(paired_lists, key=lambda item: item[1])
        repeat_count_list, subsample_fractions_list = map(list, zip(*sorted_paired_lists))
        if not all(repeat_count_list[i] >= repeat_count_list[i+1] for i in range(len(repeat_count_list) - 1)):
            raise ValueError("repeat_count_list must be non-increasing with respect to increasing subsample_fractions_list")

        error_names = [
            "Subsampled Training Energy RMSE", "Subsampled Training Force RMSE",
            "Remaining Training Energy RMSE", "Remaining Training Force RMSE",
            "Testing Energy RMSE", "Testing Force RMSE"
        ]
        collected_errors = defaultdict(list)

        repeat_count_old = 0
        for i, repeat_count in reversed(list(enumerate(repeat_count_list))):
            for j in range(repeat_count-repeat_count_old):
                self.sub_mask = None
                for subsample_fraction in subsample_fractions_list[:i+1]:

                    if verbose: print(f"Processing subsample_fraction {subsample_fraction}")
                    self.create_subsample(subsample_fraction=subsample_fraction, seed=seed)
                    self.train_subsample()
                    computed_errors = self.compute_subsample_errors(verbose=verbose)

                    for error_idx, error_value in enumerate(computed_errors):
                        error_name = error_names[error_idx]
                        collected_errors[(error_name, subsample_fraction)].append(error_value)
                        
            repeat_count_old = repeat_count

        errors_df = pd.DataFrame({
            col_tuple: pd.Series(error_list)
            for col_tuple, error_list in collected_errors.items()
        })

        multi_idx = pd.MultiIndex.from_tuples(
            errors_df.columns,
            names=['Error Type', 'Subsample Fraction']
        )
        error_cat_type = pd.CategoricalDtype(categories=error_names, ordered=True)
        frac_cat_type = pd.CategoricalDtype(categories=subsample_fractions_list, ordered=True)
        current_error_levels = multi_idx.levels[0]
        current_frac_levels = multi_idx.levels[1]
        new_error_levels = current_error_levels.astype(error_cat_type)
        new_frac_levels = current_frac_levels.astype(frac_cat_type)
        multi_idx = multi_idx.set_levels([new_error_levels, new_frac_levels])
        errors_df.columns = multi_idx
        errors_df = errors_df.sort_index(axis=1)

        return errors_df
            

        # for i, num_repeats in enumerate(repeat_count_list):
        #     for j in range(num_repeats):
        #         if verbose: print(f"Processing repeat {j}...")
        #         subsample_fraction = subsample_fractions_list[i]
        #         if verbose: print(f"  Processing subsample fraction {subsample_fraction}...")

        # for i, subsample_fraction in enumerate(subsample_fractions_list):
        #     num_repeats = repeat_count_list[i]
        #     if verbose: print(f"  Processing fraction {subsample_fraction} ({num_repeats} repeats)...")
        #     for rep in range(num_repeats):
        #         self.create_subsample(subsample_fraction=subsample_fraction, seed=seed)
        #         self.train_subsample()
        #         computed_errors = self.compute_subsample_errors(verbose=verbose)

        #         for error_idx, error_value in enumerate(computed_errors):
        #             error_name = error_names[error_idx]
        #             collected_errors[(error_name, subsample_fraction)].append(error_value)



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
