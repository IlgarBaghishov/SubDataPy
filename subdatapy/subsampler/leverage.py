import torch
from .random import RandomSubSampler
from subdatapy import linalg


class LeverageSubSampler(RandomSubSampler):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, config_idxs=None,
                 enrow_mask=None, intercept=True, device='cuda', block=False,
                 U=None, S=None, Vh=None,
                 factorization='auto',   # 'svd', 'qr', or 'auto'
                 n_chunks=None,
                 ):

        # Keep training data on CPU when chunking/distributed (set before super
        # so train_test_split sees it and never moves full X to GPU).
        if n_chunks is not None or linalg.is_distributed():
            self._train_target_device = 'cpu'

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed,
                         config_idxs=config_idxs, enrow_mask=enrow_mask,
                         intercept=intercept, device=device)
        self.block = block
        self.U = U
        self.S = S
        self.Vh = Vh
        self.factorization = factorization
        self.n_chunks = n_chunks

    def _create_sub_mask(self):

        # 1. Compute weighted X (stays on whatever device X_train is on)
        X_w = self.w_train.reshape(-1, 1) * self.X_train

        # 2. Compute leverage scores
        use_qr = (self.factorization == 'qr' or
                  (self.factorization == 'auto' and
                   (self.n_chunks is not None or linalg.is_distributed())))

        if use_qr:
            if self.n_chunks is None and not linalg.is_distributed():
                # Single-pass: retain Q, compute ||Q_i||^2 — O(np)
                row_leverage = linalg.leverage_scores_from_qr(X_w, device=self.device)
            else:
                # Chunked/distributed: TSQR for R, then chunked leverage — O(np^2)
                R = linalg.tsqr_r(X_w, device=self.device, n_chunks=self.n_chunks)
                if linalg.is_distributed() and linalg.get_rank() != 0:
                    return
                row_leverage = linalg.leverage_scores_from_r(
                    X_w, R, device=self.device, n_chunks=self.n_chunks)
        else:
            # SVD path (original behavior)
            if self.U is None:
                self.U, self.S, self.Vh = torch.linalg.svd(X_w.to(self.device), full_matrices=False)
            row_leverage = torch.sum(self.U ** 2, dim=1)

        # row_leverage is on self.device (GPU); config_idxs_train may be on CPU

        # 3. Handle Block Aggregation
        if self.block:
            unique_vals, inverse_indices = torch.unique(self.config_idxs_train, return_inverse=True)
            n_groups = len(unique_vals)
            group_leverage = torch.zeros(n_groups, device=self.device, dtype=row_leverage.dtype)
            group_leverage.index_add_(0, inverse_indices.to(self.device), row_leverage)
            self.leverage_scores = group_leverage
        else:
            unique_vals, inverse_indices = torch.unique(self.config_idxs_train, return_inverse=True)
            inv_dev = inverse_indices.device
            perm = torch.arange(inverse_indices.size(0), device=inv_dev)
            unique_first_indices = torch.empty(len(unique_vals), dtype=torch.long, device=inv_dev)
            unique_first_indices.scatter_reduce_(0, inverse_indices, perm, reduce="amin", include_self=False)
            self.leverage_scores = row_leverage[unique_first_indices.to(self.device)]

        # 4. Sampling (all on self.device)
        leverage_sum = torch.sum(self.leverage_scores)
        if leverage_sum == 0:
            probs = torch.ones_like(self.leverage_scores) / len(self.leverage_scores)
        else:
            probs = self.leverage_scores / leverage_sum

        chosen_indices_idx = torch.multinomial(probs, self.n_subsamples, replacement=False)
        # unique_vals is on config_idxs_train's device; index with CPU indices
        sub_unique_config_idxs_train = unique_vals[chosen_indices_idx.to(unique_vals.device)]

        self.sub_mask = torch.isin(self.config_idxs, sub_unique_config_idxs_train.to('cpu'))
        self.sub_mask_train = torch.isin(
            self.config_idxs_train,
            sub_unique_config_idxs_train.to(self.config_idxs_train.device))
