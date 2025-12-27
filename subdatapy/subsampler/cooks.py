import torch
import warnings
from .random import RandomSubSampler
from .leverage import LeverageSubSampler


class CookSubSampler(RandomSubSampler):

    def __init__(self, X, y, w=None, test_fraction=0.0, seed=None, test_mask=None, config_idxs=None, enrow_mask=None, intercept=True,
                 device='cuda', block=False, stepwise=False, sampling=True, ascending=True, initial_subsampler="random",
                 initial_subsample_fraction=1, U=None, S=None, Vh=None):

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed, test_mask=test_mask, 
                         config_idxs=config_idxs, enrow_mask=enrow_mask, intercept=intercept, device=device)
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
        self.XTX_inv = None
        self.XTy = None
        
        # PRE-COMPUTE GROUPS for block operations
        if self.block:
            self._prepare_block_metadata()


    def _prepare_block_metadata(self):
        
        sort_perm = torch.argsort(self.config_idxs_train)
        
        # 2. Reorder GPU Tensors
        # This aligns X_train with the sorted groups
        self.X_train = self.X_train[sort_perm]
        self.y_train = self.y_train[sort_perm]
        self.w_train = self.w_train[sort_perm]
        self.enrow_mask_train = self.enrow_mask_train[sort_perm]

        self.config_idxs_train = self.config_idxs_train[sort_perm]
        
        # 3. Build Metadata (on CPU)
        unique_vals, counts = torch.unique_consecutive(self.config_idxs_train, return_counts=True)
        end_indices = torch.cumsum(counts, dim=0)
        start_indices = end_indices - counts
        
        self.group_metadata = torch.stack((unique_vals, start_indices, counts), dim=1)
        
        # 4. Sort Metadata by size (for efficient batching)
        size_sort = torch.argsort(self.group_metadata[:, 2])
        self.group_metadata = self.group_metadata[size_sort]


    def _create_sub_mask(self):

        # self.X_train.to('cpu')
        # self.X_train_w = (self.w_train.reshape(-1, 1) * self.X_train).to(self.device)
        # self.y_train_w = self.w_train.reshape(-1, 1) * self.y_train.reshape(-1, 1)

        self.X_train.mul_(self.w_train)
        self.y_train.mul_(self.w_train)

        if self.stepwise:
            self._stepwise_cooks_sampling()
        else:
            self._onestep_cooks_sampling()
            
        # del self.X_train_w
        # del self.y_train_w
        # self.X_train.to(self.device)
        self.X_train.div_(self.w_train)
        self.y_train.div_(self.w_train)


    def _onestep_cooks_sampling(self):

        if self.block:
            raise NotImplementedError("Onestep Block Cook's Distance methods are not implemented yet.")
        else:
            if self.U is None:
                self.U, self.S, self.Vh = torch.linalg.svd(self.X_train, full_matrices=False)

            leverage_scores = torch.sum(self.U[self.enrow_mask_train]**2, dim=1)

            tol = torch.finfo(self.X_train.dtype).eps * max(self.X_train.shape) * self.S[0]
            S_inv = torch.where(self.S > tol, 1/self.S, torch.tensor(0.0, device=self.device, dtype=self.dtype))

            term1 = self.U.T @ self.y_train
            term2 = S_inv.reshape(-1, 1) * term1
            coeffs = self.Vh.T @ term2

            preds = self.X_train[self.enrow_mask_train] @ coeffs
            en_residuals_sq = torch.square(preds - self.y_train[self.enrow_mask_train]).reshape(-1)

            self.onestep_en_cooks = en_residuals_sq * leverage_scores / (1 - leverage_scores)**2
            
            if self.sampling:
                cooks_probs = self.onestep_en_cooks / torch.sum(self.onestep_en_cooks)
                indices = torch.multinomial(cooks_probs, self.n_subsamples, replacement=False)
                sub_unique_config_idxs_train = self.unique_config_idxs_train[indices]
            else:
                topk_vals, topk_indices = torch.topk(self.onestep_en_cooks, self.n_subsamples)
                sub_unique_config_idxs_train = self.unique_config_idxs_train[topk_indices]

            self.sub_mask = torch.isin(self.config_idxs, sub_unique_config_idxs_train.to(device='cpu'))
            self.sub_mask_train = torch.isin(self.config_idxs_train, sub_unique_config_idxs_train)


    def _create_initial_sub_mask(self):

        if self.initial_subsampler == "leverage":
            lss = LeverageSubSampler(self.X_train, seed=self.seed, device=self.device,
                                     config_idxs=self.config_idxs_train, block=self.block)
            self.sub_mask_train = lss.create_subsample(subsample_fraction=self.initial_subsample_fraction, seed=self.seed).to(device=self.device)
        elif self.initial_subsampler == "random":
            rss = RandomSubSampler(self.X_train, seed=self.seed, device=self.device, config_idxs=self.config_idxs_train)
            self.sub_mask_train = rss.create_subsample(subsample_fraction=self.initial_subsample_fraction, seed=self.seed).to(device=self.device)

        self.sub_mask = torch.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train].to(device='cpu'))


    def _stepwise_cooks_sampling(self):

        if self.sub_mask is None:
             self._create_initial_sub_mask()
        
        sub_X = self.X_train[self.sub_mask_train]
        sub_y = self.y_train[self.sub_mask_train]
        
        # Compute (X'X)^-1
        # For numerical stability with rank deficiency, use pinv or specialized update
        # Here using cholesky or inv is faster if full rank.
        # Using SVD as in original code is safest.
        U, S, Vh = torch.linalg.svd(sub_X, full_matrices=False)

        # XTX_inv = V @ S^-2 @ V.T
        # Filter small S
        tol = torch.finfo(S.dtype).eps * max(sub_X.shape) * S[0]
        S_inv_sq = torch.where(S > tol, 1/(S**2), torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.XTX_inv = Vh.T @ torch.diag(S_inv_sq) @ Vh
        self.XTy = sub_X.T @ sub_y
        
        n_subsamples_init = int(torch.sum(self.sub_mask & self.enrow_mask))
        
        # THE LOOP
        target_range = range(n_subsamples_init, self.n_subsamples) if self.ascending \
                       else range(n_subsamples_init, self.n_subsamples, -1)

        for _ in target_range:
            
            coeffs = self.XTX_inv @ self.XTy
            
            if self.block:
                
                BATCH_SIZE = 5000 # Adjust based on GPU memory.
                best_cooks_val = -float('inf') if self.ascending else float('inf')
                best_config_id = -1
                
                num_groups = self.group_metadata.shape[0]
                
                # Iterate chunks of groups
                for i in range(0, num_groups, BATCH_SIZE):
                    batch_meta = self.group_metadata[i : i + BATCH_SIZE]
                    
                    # Determine max size in this batch for padding
                    max_len = batch_meta[-1, 2].item()
                    curr_batch_size = batch_meta.shape[0]
                    
                    # Create Padded Tensor (Batch, Max_Len, Feat)
                    # We pull from X_train.
                    # Warning: X_train might be on CPU if 150GB.
                    # We copy strictly what we need to GPU.
                    
                    X_batch = torch.zeros((curr_batch_size, max_len, self.X_train.shape[1]), device=self.device, dtype=self.dtype)
                    y_batch = torch.zeros((curr_batch_size, max_len, 1), device=self.device, dtype=self.dtype)
                    
                    # Fill batch (This loop runs on CPU/GPU indices, but only 500 iterations)
                    # Optimized: Slice using indices
                    # Since data is contiguous (we sorted X_train), we can try to fast copy.
                    # But batch items are distinct in global X_train.
                    # A loop here is unavoidable unless we used packed sequences, but loop 500 is fine.
                    for b_i in range(curr_batch_size):
                        idx, start, count = batch_meta[b_i]
                        X_batch[b_i, :count, :] = self.X_train[start : start+count].to(device=self.device)
                        y_batch[b_i, :count, :] = self.y_train[start : start+count].to(device=self.device)
                    
                    # --- COMPUTE COOKS BATCHED ---
                    # 1. Fake Leverage: X_i @ XTX_inv @ X_i.T
                    # X_batch: (B, L, F), XTX_inv: (F, F)
                    # Res: (B, L, L)                    
                    temp = torch.bmm(X_batch, self.XTX_inv.unsqueeze(0).expand(curr_batch_size, -1, -1))
                    fake_lev = torch.bmm(temp, X_batch.transpose(1, 2))
                    
                    # 2. Residuals: X_i @ coeffs - y_i
                    # coeffs: (F, 1)
                    res = torch.bmm(X_batch, coeffs.unsqueeze(0).expand(curr_batch_size, -1, -1).to('cuda')) - y_batch
                    
                    # 3. Sherman-Morrison / Cook's Term
                    # For adding: inv(I + H)
                    # For removing: inv(I - H)
                    identity = torch.eye(max_len, device='cuda').unsqueeze(0)
                    
                    if self.ascending:
                        mat_to_inv = identity + fake_lev
                    else:
                        mat_to_inv = identity - fake_lev
                        
                    # Batched Inverse (B, L, L)
                    # Note: We must mask out the padding parts to avoid singular matrices if padding is 0?
                    # Actually I + 0 = I, which is invertible. So padding 0 is safe for 'ascending'.
                    # For 'descending', I - 0 = I. Safe.
                    inv_mat = torch.linalg.inv(mat_to_inv)
                    
                    # Final Calc: res.T @ inv_mat @ (fake_lev @ res)
                    # shape: (B, 1, L) @ (B, L, L) @ (B, L, 1) -> (B, 1, 1)
                    term_right = torch.bmm(fake_lev, res)
                    term_mid = torch.bmm(inv_mat, term_right)
                    cooks_vals = torch.bmm(res.transpose(1, 2), term_mid).squeeze()
                    
                    # Handle already selected masks
                    # If ascending, we can't select what's already in sub_mask_train
                    # Check metadata IDs against current mask
                    # This check is fast on GPU
                    batch_config_ids = batch_meta[:, 0].to('cuda')
                    
                    # Construct a boolean vector of "is_in_subset" for this batch
                    # This requires mapping batch_config_ids back to boolean.
                    # Or simpler: maintain a set of active IDs?
                    # Let's trust logic: e_cooks[already_in] = -inf
                    
                    # Ideally we maintain a global boolean tensor of active configs
                    # But batch_config_ids are raw IDs.
                    # Use isin
                    # active_ids = self.unique_config_idxs_train[torch.unique(self.config_idxs_train[self.sub_mask_train])]  # this gave error
                    active_ids = torch.unique(self.config_idxs_train[self.sub_mask_train])
                    is_active = torch.isin(batch_config_ids, active_ids.to('cuda'))
                    
                    if self.ascending:
                        cooks_vals[is_active] = -float('inf')
                        curr_max = torch.max(cooks_vals)
                        if curr_max > best_cooks_val:
                            best_cooks_val = curr_max
                            best_config_id = batch_config_ids[torch.argmax(cooks_vals)]
                    else:
                        cooks_vals[~is_active] = float('inf')
                        curr_min = torch.min(cooks_vals)
                        if curr_min < best_cooks_val:
                            best_cooks_val = curr_min
                            best_config_id = batch_config_ids[torch.argmin(cooks_vals)]

            else:
                en_residuals = self.X_train[self.enrow_mask_train] @ coeffs - self.y_train[self.enrow_mask_train]
                leverage_scores = self.X_train[self.enrow_mask_train] @ self.XTX_inv
                leverage_scores = torch.einsum('ij,ji->i', leverage_scores, self.X_train[self.enrow_mask_train].T)
                e_cooks = torch.square(en_residuals).reshape(-1) * leverage_scores / (1+leverage_scores)

                if self.ascending:
                    e_cooks[self.sub_mask_train[self.enrow_mask_train]] = -float('inf')
                    best_config_id = torch.argmax(e_cooks)
                else:
                    e_cooks[self.sub_mask_train[self.enrow_mask_train]] = float('inf')
                    best_config_id = torch.argmin(e_cooks)

            # Apply Update
            config_to_change = best_config_id.item()
            # Find mask for this specific config
            change_mask = (self.config_idxs_train == config_to_change)
            
            # Update sub_mask_train
            if self.ascending:
                self.sub_mask_train[change_mask] = True
            else:
                self.sub_mask_train[change_mask] = False

            # Update Matrix (Rank-k update)
            X_change = self.X_train[change_mask].to('cuda')
            y_change = self.y_train[change_mask].to('cuda')
            
            # Woodbury Identity / Sherman-Morrison for blocks
            # (A +/- UCV)^-1 = A^-1 -/+ A^-1 U (C^-1 +/- V A^-1 U)^-1 V A^-1
            # Here A = XTX, U = X_change.T, V = X_change, C = I
            
            left_update = self.XTX_inv @ X_change.T
            
            # Inner inverse size: (k, k) where k is atoms in group
            inner_term = torch.eye(X_change.shape[0], device='cuda')
            inner_right = X_change @ left_update
            
            if self.ascending:
                inner = torch.linalg.inv(inner_term + inner_right)
                self.XTX_inv -= left_update @ inner @ (X_change @ self.XTX_inv)
                self.XTy += X_change.T @ y_change
            else:
                inner = torch.linalg.inv(inner_term - inner_right)
                self.XTX_inv += left_update @ inner @ (X_change @ self.XTX_inv)
                self.XTy -= X_change.T @ y_change

        self.sub_mask = torch.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train].to(device='cpu'))