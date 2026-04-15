import torch
import torch.distributed as dist
from .random import RandomSubSampler
from subdatapy import linalg


class LeverageSubSampler(RandomSubSampler):

    def __init__(self, X, y=None, w=None, test_fraction=0.0, seed=None, config_idxs=None,
                 enrow_mask=None, intercept=True, device='cuda', block=False,
                 U=None, S=None, Vh=None,
                 factorization='auto',   # 'svd', 'qr', or 'auto'
                 n_chunks=None,
                 local_devices=None,
                 train_target_device=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None,
                 ):
        # Keep training data on CPU when chunking/distributed so the full X
        # never ends up on one GPU.
        if train_target_device is None and (n_chunks is not None or linalg.is_distributed()):
            train_target_device = 'cpu'

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed,
                         config_idxs=config_idxs, enrow_mask=enrow_mask,
                         intercept=intercept, device=device,
                         local_devices=local_devices,
                         train_target_device=train_target_device,
                         partitioned_override=partitioned_override,
                         unique_config_idxs_train_override=unique_config_idxs_train_override)
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
                R = linalg.tsqr_r(
                    X_w, device=self.device, n_chunks=self.n_chunks,
                    partitioned=self._is_partitioned,
                    local_devices=self.local_devices)
                if linalg.is_distributed():
                    # Broadcast R to all ranks so every rank can compute local h_i.
                    p = X_w.shape[1]
                    R = linalg.broadcast_tensor(
                        R if linalg.get_rank() == 0 else None,
                        src=0, shape=(p, p), dtype=X_w.dtype, device=self.device)
                row_leverage = linalg.leverage_scores_from_r(
                    X_w, R, device=self.device, n_chunks=self.n_chunks,
                    local_devices=self.local_devices)
        else:
            # SVD path (original behavior)
            if self.U is None:
                self.U, self.S, self.Vh = torch.linalg.svd(X_w.to(self.device), full_matrices=False)
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
            inv_dev = inverse_indices.device
            perm = torch.arange(inverse_indices.size(0), device=inv_dev)
            unique_first_indices = torch.empty(len(local_unique_vals), dtype=torch.long, device=inv_dev)
            unique_first_indices.scatter_reduce_(0, inverse_indices, perm, reduce="amin", include_self=False)
            local_leverage_per_config = row_leverage[unique_first_indices.to(self.device)]

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
