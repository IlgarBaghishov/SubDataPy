import torch
import torch.distributed as dist
import pandas as pd
from collections import defaultdict
from subdatapy.data import BaseData
from subdatapy import linalg


class RandomSubSampler(BaseData):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, test_mask=None, config_idxs=None,
                 enrow_mask=None, intercept=True, device='cuda', dtype=torch.float64,
                 local_devices=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None):
        super().__init__(
            X, y=y, w=w, config_idxs=config_idxs, enrow_mask=enrow_mask,
            intercept=intercept, device=device, dtype=dtype, local_devices=local_devices,
            partitioned_override=partitioned_override,
            unique_config_idxs_train_override=unique_config_idxs_train_override,
        )

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
        # Sample indices on CPU so CPU and CUDA runs with the same seed pick
        # the same configs. The CUDA RNG is a separate stream even after
        # torch.manual_seed, so leaving `device=self.device` here made the
        # subsampler produce different subsets on each backend.
        perm = torch.randperm(len(self.unique_config_idxs_train))
        chosen_indices_cpu = self.unique_config_idxs_train.cpu()[perm[:self.n_subsamples]]

        self.sub_mask = torch.isin(self.config_idxs, chosen_indices_cpu)
        self.sub_mask_train = torch.isin(
            self.config_idxs_train,
            chosen_indices_cpu.to(self.config_idxs_train.device))


    def _subsample(self):
        # Record which training rows fall in the subsample as indices into
        # self.X (no design-matrix copy); train_subsample streams them. The
        # subsample is always a subset of train, so its rows are train rows.
        sub_mask_train_cpu = self.sub_mask_train.to('cpu')
        self.sub_idx = self.train_idx[sub_mask_train_cpu]
        if self.y is not None:
            self.sub_y_train = self.y_train[sub_mask_train_cpu]
        if self.w is not None:
            self.sub_w_train = self.w_train[sub_mask_train_cpu]


    def train_subsample(self, method='auto', n_chunks=None):
        # Stream the subsampled rows from self.X (weighted on the device per
        # chunk); the weighted subsample is never built on the host.
        self.sub_coeffs = linalg.solve_wls(
            self.X, self.sub_y_train, self.sub_w_train, x_idx=self.sub_idx,
            method=method, device=self.device, n_chunks=n_chunks,
            partitioned=self._is_partitioned, local_devices=self.local_devices,
            dtype=self.dtype)


    def compute_subsample_errors(self, verbose=False, n_chunks=None):

        if n_chunks is None:
            n_chunks = getattr(self, 'n_chunks', None)

        # Stream train and test through the device in chunks so neither full
        # prediction lands on one device. Squared residuals come back on
        # self.X.device (CPU), matching the CPU row masks applied below.
        train_sq_res = linalg.chunked_sq_residuals(
            self.X, self.y_train, self.sub_coeffs, x_idx=self.train_idx,
            device=self.device, n_chunks=n_chunks, local_devices=self.local_devices)
        test_sq_res = linalg.chunked_sq_residuals(
            self.X, self.y_test, self.sub_coeffs, x_idx=self.test_idx,
            device=self.device, n_chunks=n_chunks, local_devices=self.local_devices)

        is_rank0 = linalg.get_rank() == 0

        def get_rmse(sq_res, name):
            local_sq = float(sq_res.sum().item()) if sq_res.numel() > 0 else 0.0
            local_n = int(sq_res.numel())
            val = linalg.distributed_rmse(local_sq, local_n, device=self.device)
            if val is not None and verbose and is_rank0:
                print(f"{name}: {val}")
            return val

        # Row masks may live on a different device than the residuals (e.g.
        # stepwise Cook's moves some to the GPU); align them to the residuals.
        en_tr = self.enrow_mask_train.to(train_sq_res.device)
        sub_tr = self.sub_mask_train.to(train_sq_res.device)
        en_te = self.enrow_mask_test.to(test_sq_res.device)

        # Subsampled Train
        e_sub = get_rmse(train_sq_res[sub_tr & en_tr], "Subsampled Training Data Energy RMSE")
        f_sub = get_rmse(train_sq_res[sub_tr & (~en_tr)], "Subsampled Training Data Force RMSE")

        # Entire Train
        e_train = get_rmse(train_sq_res[en_tr], "Entire Training Data Energy RMSE")
        f_train = get_rmse(train_sq_res[~en_tr], "Entire Training Data Force RMSE")

        # Test
        e_test = get_rmse(test_sq_res[en_te], "Energy Test RMSE")
        f_test = get_rmse(test_sq_res[~en_te], "Force Test RMSE")

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
