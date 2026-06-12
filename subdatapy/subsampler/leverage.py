import torch
import torch.distributed as dist
from .random import RandomSubSampler
from subdatapy import linalg


class LeverageSubSampler(RandomSubSampler):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, test_mask=None,
                 config_idxs=None,
                 enrow_mask=None, intercept=True, device='cuda', dtype=torch.float64, block=False,
                 U=None, S=None, Vh=None,
                 factorization='auto',   # 'svd', 'qr', or 'auto'
                 n_chunks=None,
                 local_devices=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None,
                 ):
        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed,
                         test_mask=test_mask,
                         config_idxs=config_idxs, enrow_mask=enrow_mask,
                         intercept=intercept, device=device, dtype=dtype,
                         local_devices=local_devices,
                         partitioned_override=partitioned_override,
                         unique_config_idxs_train_override=unique_config_idxs_train_override)
        self.block = block
        self.U = U
        self.S = S
        self.Vh = Vh
        self.factorization = factorization
        self.n_chunks = n_chunks

    def _create_sub_mask(self):

        # 1-2. Leverage scores over the weighted training rows, streamed from
        # the host design matrix by index — the weighted matrix is never built
        # on the host. p = number of features.
        p = self.X.shape[1]
        use_qr = (self.factorization == 'qr' or
                  (self.factorization == 'auto' and
                   (self.n_chunks is not None or linalg.is_distributed())))

        if use_qr:
            if self.n_chunks is None and not linalg.is_distributed():
                # Single-pass: gather+weight the train rows on the device,
                # retain Q, compute ||Q_i||^2 — O(np).
                X_w = (self.w_train.reshape(-1, 1) * self.X[self.train_idx]).to(self.device)
                row_leverage = linalg.leverage_scores_from_qr(X_w, device=self.device)
            else:
                # Chunked/distributed: TSQR for R, then chunked leverage — O(np^2)
                R = linalg.tsqr_r(
                    self.X, x_idx=self.train_idx, w=self.w_train,
                    device=self.device, n_chunks=self.n_chunks,
                    partitioned=self._is_partitioned,
                    local_devices=self.local_devices)
                if linalg.is_distributed():
                    # Broadcast R to all ranks so every rank can compute local h_i.
                    R = linalg.broadcast_tensor(
                        R if linalg.get_rank() == 0 else None,
                        src=0, shape=(p, p), dtype=self.dtype, device=self.device)
                row_leverage = linalg.leverage_scores_from_r(
                    self.X, R, x_idx=self.train_idx, w=self.w_train,
                    device=self.device, n_chunks=self.n_chunks,
                    local_devices=self.local_devices)
        else:
            # SVD path (single-pass): gather+weight the train rows on the device.
            if self.U is None:
                X_w = (self.w_train.reshape(-1, 1) * self.X[self.train_idx]).to(self.device)
                self.U, self.S, self.Vh = torch.linalg.svd(X_w, full_matrices=False)
            row_leverage = torch.sum(self.U ** 2, dim=1)

        # row_leverage is on self.device (GPU); config_idxs_train may be on CPU

        # 3. Aggregate row leverage into per-config leverage (on local configs).
        local_unique_vals, inverse_indices = torch.unique(self.config_idxs_train, return_inverse=True)
        if self.block:
            n_groups = len(local_unique_vals)
            group_leverage = torch.zeros(n_groups, device=self.device, dtype=row_leverage.dtype)
            group_leverage.index_add_(0, inverse_indices.to(self.device), row_leverage)
            local_leverage_per_config = group_leverage
        else:
            # Represent each config by its ENERGY row (per enrow_mask), to match
            # non-block Cook's and the documented semantics. Non-energy rows are
            # masked to a large index so amin selects the energy row; a config
            # with no energy row falls back to its first row.
            inv_dev = inverse_indices.device
            n_rows = inverse_indices.size(0)
            perm = torch.arange(n_rows, device=inv_dev)
            enrow = self.enrow_mask_train.to(inv_dev)
            en_perm = torch.where(enrow, perm, n_rows)
            rep_idx = torch.empty(len(local_unique_vals), dtype=torch.long, device=inv_dev)
            rep_idx.scatter_reduce_(0, inverse_indices, en_perm, reduce="amin", include_self=False)
            no_energy = rep_idx >= n_rows
            if no_energy.any():
                first_idx = torch.empty(len(local_unique_vals), dtype=torch.long, device=inv_dev)
                first_idx.scatter_reduce_(0, inverse_indices, perm, reduce="amin", include_self=False)
                rep_idx = torch.where(no_energy, first_idx, rep_idx)
            local_leverage_per_config = row_leverage[rep_idx.to(self.device)]

        # 4. Build per-config leverage vector matching global unique_config_idxs_train.
        if linalg.is_distributed():
            # All-gather (config_id_list, leverage_list) pairs from each rank.
            pairs = [None] * linalg.get_world_size()
            local_payload = (local_unique_vals.cpu().tolist(),
                             local_leverage_per_config.detach().cpu().tolist())
            dist.all_gather_object(pairs, local_payload)

            global_ids = self.unique_config_idxs_train.cpu()
            score_map = {}
            for ids, scores in pairs:
                for cid, sc in zip(ids, scores):
                    score_map[int(cid)] = float(sc)
            global_scores = torch.tensor(
                [score_map[int(i)] for i in global_ids.tolist()],
                dtype=row_leverage.dtype, device=self.device)
            unique_vals_global = self.unique_config_idxs_train.to(self.device)
            self.leverage_scores = global_scores
        else:
            self.leverage_scores = local_leverage_per_config
            unique_vals_global = local_unique_vals

        # 5. Sampling on CPU so CPU and CUDA runs with the same seed pick
        # identical configs (CUDA RNG is an independent stream).
        leverage_sum = torch.sum(self.leverage_scores)
        if leverage_sum == 0:
            probs = torch.ones_like(self.leverage_scores) / len(self.leverage_scores)
        else:
            probs = self.leverage_scores / leverage_sum

        chosen_indices_idx = torch.multinomial(probs.cpu(), self.n_subsamples,
                                               replacement=False)
        sub_unique_config_idxs_train = unique_vals_global.cpu()[chosen_indices_idx]

        self.sub_mask = torch.isin(self.config_idxs, sub_unique_config_idxs_train)
        self.sub_mask_train = torch.isin(
            self.config_idxs_train,
            sub_unique_config_idxs_train.to(self.config_idxs_train.device))
