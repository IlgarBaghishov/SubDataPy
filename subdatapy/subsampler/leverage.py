import torch
from .random import RandomSubSampler

class LeverageSubSampler(RandomSubSampler):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, config_idxs=None, enrow_mask=None, intercept=True,
                 device='cuda', block=False, U=None, S=None, Vh=None):

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed, config_idxs=config_idxs, enrow_mask=enrow_mask,
                         intercept=intercept, device=device)
        self.block = block
        self.U = U
        self.S = S
        self.Vh = Vh

    def _create_sub_mask(self):
        
        # 1. Compute SVD
        # Use Weighted X
        X_w = self.w_train.reshape(-1, 1) * self.X_train
        
        if self.U is None:
            # full_matrices=False is standard
            self.U, self.S, self.Vh = torch.linalg.svd(X_w, full_matrices=False)
            
        # 2. Compute Leverage Scores (Row-wise sum of U^2)
        # Shape: (N_rows,)
        row_leverage = torch.sum(self.U**2, dim=1)

        # 3. Handle Block Aggregation
        if self.block:
            # We need to sum row_leverage for each configuration.
            # We assume config_idxs_train maps rows to config IDs.
            # To use index_add, we need contiguous integers 0..N_groups.
            
            # Map unique_config_idxs to 0..N-1
            unique_vals, inverse_indices = torch.unique(self.config_idxs_train, return_inverse=True)
            n_groups = len(unique_vals)
            
            # Initialize group leverage scores
            group_leverage = torch.zeros(n_groups, device=self.device, dtype=row_leverage.dtype)
            
            # Scatter Add: much faster than loop
            group_leverage.index_add_(0, inverse_indices, row_leverage)
            
            self.leverage_scores = group_leverage
            # Align unique_config_idxs_train with the order of group_leverage (it is sorted by unique)
            # self.unique_config_idxs_train is already sorted by unique
        else:
            # For non-block, we just take the score of the 'energy row' or first row?
            # Original code: [lev[config==unique][0] for unique...]
            # This implies taking the first row's leverage of each group.
            
            # Find indices of first elements (similar to BaseData init)
            # Or simplified: just use the mask if available
            # Assuming 1-to-1 mapping if not block, or taking representative.
            # Let's assume we take the sum if block=True, and the first element if block=False?
            # The original code took the FIRST element.
            
            unique_vals, inverse_indices = torch.unique(self.config_idxs_train, return_inverse=True)
            
            # We need to pick the value corresponding to the first occurrence.
            # We can use scatter_reduce with "mean" or just pick manually?
            # Creating a mask for unique_first_indices is fastest.
            perm = torch.arange(inverse_indices.size(0), device=self.device)
            unique_first_indices = torch.empty(len(unique_vals), dtype=torch.long, device=self.device)
            # "amin" gets the smallest index (first occurrence)
            unique_first_indices.scatter_reduce_(0, inverse_indices, perm, reduce="amin", include_self=False)
            
            self.leverage_scores = row_leverage[unique_first_indices]

        # 4. Sampling
        leverage_sum = torch.sum(self.leverage_scores)
        if leverage_sum == 0:
            probs = torch.ones_like(self.leverage_scores) / len(self.leverage_scores)
        else:
            probs = self.leverage_scores / leverage_sum
            
        # torch.multinomial is used for weighted sampling
        # We need 'replace=False'. multinomial supports this.
        chosen_indices_idx = torch.multinomial(probs, self.n_subsamples, replacement=False)
        
        # Map back to real config IDs
        # If block=True or False, we aligned scores with unique_vals (sorted)
        # so unique_vals[chosen_indices_idx] gives the config IDs
        sub_unique_config_idxs_train = unique_vals[chosen_indices_idx]
        
        self.sub_mask = torch.isin(self.config_idxs, sub_unique_config_idxs_train.to('cpu'))
        self.sub_mask_train = torch.isin(self.config_idxs_train, sub_unique_config_idxs_train)