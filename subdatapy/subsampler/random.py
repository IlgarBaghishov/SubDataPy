import torch
import pandas as pd
from collections import defaultdict
from subdatapy.data import BaseData


class RandomSubSampler(BaseData):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, test_mask=None, config_idxs=None, 
                 enrow_mask=None, intercept=True, device='cuda'):

        super().__init__(X, y=y, w=w, config_idxs=config_idxs, enrow_mask=enrow_mask, intercept=intercept,
                         device=device)

        self.sub_mask = None
        self.train_test_split(test_fraction=test_fraction, seed=seed, test_mask=test_mask)


    def create_subsample(self, subsample_fraction, seed=None):
        if seed is not None:
            self.seed = seed
            torch.manual_seed(self.seed)
            
        self.subsample_fraction = subsample_fraction

        if not (0 < self.subsample_fraction <= 1):
            raise ValueError("subsample_fraction must be between 0 and 1")

        n_total = len(self.unique_config_idxs_train)
        self.n_subsamples = round(n_total * self.subsample_fraction)

        self._create_sub_mask()
        self._subsample()
        return self.sub_mask


    def _create_sub_mask(self):
        perm = torch.randperm(len(self.unique_config_idxs_train), device=self.device)
        chosen_indices = self.unique_config_idxs_train[perm[:self.n_subsamples]]
        
        self.sub_mask = torch.isin(self.config_idxs, chosen_indices.to(device='cpu'))
        self.sub_mask_train = torch.isin(self.config_idxs_train, chosen_indices)


    def _subsample(self):
        self.sub_X_train = self.X[self.sub_mask].to(device=self.device)
        if self.y is not None:
            self.sub_y_train = self.y[self.sub_mask].to(device=self.device)
        if self.w is not None:
            self.sub_w_train = self.w[self.sub_mask].to(device=self.device)


    def train_subsample(self):
        # A = self.sub_w_train.reshape(-1, 1) * self.sub_X_train
        # B = self.sub_w_train.reshape(-1, 1) * self.sub_y_train.reshape(-1, 1)
        A = self.sub_w_train * self.sub_X_train
        B = self.sub_w_train * self.sub_y_train

        result = torch.linalg.lstsq(A, B)
        self.sub_coeffs = result.solution


    def compute_subsample_errors(self, verbose=False):

        train_preds = self.X_train @ self.sub_coeffs
        train_sq_res = torch.square(train_preds - self.y_train)
        test_preds = self.X_test @ self.sub_coeffs
        test_sq_res = torch.square(test_preds - self.y_test)
        
        def get_rmse(sq_res, name):
            val = torch.sqrt(torch.mean(sq_res)).item()
            if verbose: print(f"{name}: {val}")
            return val

        # Subsampled Train
        mask_sub_en = self.sub_mask_train & self.enrow_mask_train
        e_sub = get_rmse(train_sq_res[mask_sub_en], "Subsampled Training Data Energy RMSE")
        
        mask_sub_f = self.sub_mask_train & (~self.enrow_mask_train)
        f_sub = get_rmse(train_sq_res[mask_sub_f], "Subsampled Training Data Force RMSE")
        
        # Entire Train
        e_train = get_rmse(train_sq_res[self.enrow_mask_train], "Entire Training Data Energy RMSE")
        f_train = get_rmse(train_sq_res[~self.enrow_mask_train], "Entire Training Data Force RMSE")
        
        # Test
        e_test = get_rmse(test_sq_res[self.enrow_mask_test], "Energy Test RMSE")
        f_test = get_rmse(test_sq_res[~self.enrow_mask_test], "Force Test RMSE")
        
        return e_sub, f_sub, e_train, f_train, e_test, f_test


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
            "Entire Training Energy RMSE", "Entire Training Force RMSE",
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
