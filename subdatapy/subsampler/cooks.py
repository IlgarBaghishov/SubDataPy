import torch
import torch.distributed as dist
import warnings
from .random import RandomSubSampler
from .leverage import LeverageSubSampler
from subdatapy import linalg


def _batched_matmul(A, B):
    """Batched matrix multiply with loop fallback for CUBLAS compatibility.

    torch.bmm can fail with CUBLAS_STATUS_INVALID_VALUE for float64 on
    some CUDA toolkit / driver combinations.  When that happens, fall back
    to a sequential loop which only uses single-batch GEMM calls.
    """
    try:
        return torch.bmm(A, B)
    except RuntimeError:
        return torch.stack([A[i] @ B[i] for i in range(A.shape[0])])


class CookSubSampler(RandomSubSampler):

    def __init__(self, X, y, w=None, test_fraction=0.0, seed=None, test_mask=None,
                 config_idxs=None, enrow_mask=None, intercept=True,
                 device='cuda', block=False, stepwise=False, sampling=True,
                 ascending=True, initial_subsampler="random",
                 initial_subsample_fraction=1, U=None, S=None, Vh=None,
                 # Chunked / distributed parameters:
                 n_chunks=None,
                 factorization='auto',       # 'svd', 'qr', or 'auto'
                 update_method='auto',        # 'woodbury', 'qr', or 'auto'
                 tree_reduction_threshold=10,
                 local_devices=None,
                 train_target_device=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None,
                 ):

        self.n_chunks = n_chunks
        # Keep training data on CPU in chunked/distributed so the full X
        # never lands on one GPU.
        if train_target_device is None and (n_chunks is not None or linalg.is_distributed()):
            train_target_device = 'cpu'

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed,
                         test_mask=test_mask, config_idxs=config_idxs,
                         enrow_mask=enrow_mask, intercept=intercept, device=device,
                         local_devices=local_devices,
                         train_target_device=train_target_device,
                         partitioned_override=partitioned_override,
                         unique_config_idxs_train_override=unique_config_idxs_train_override)

        self.block = block
        self.stepwise = stepwise
        self.sampling = sampling
        self.ascending = ascending
        self.initial_subsampler = initial_subsampler
        self.initial_subsample_fraction = initial_subsample_fraction
        self.U = U
        self.S = S
        self.Vh = Vh

        self.factorization = factorization
        self.update_method = update_method
        self.tree_reduction_threshold = tree_reduction_threshold

        self.onestep_en_cooks = None
        self.XTX_inv = None
        self.XTy = None
        self.R_final = None

        if self.block:
            self._prepare_block_metadata()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _use_qr_factorization(self):
        if self.factorization == 'qr':
            return True
        if self.factorization == 'svd':
            return False
        # 'auto': QR if chunked or distributed
        return self._needs_chunking()

    def _use_qr_update(self):
        if self.update_method == 'qr':
            if self.R_final is None:
                raise ValueError(
                    "update_method='qr' requires QR factorization (R_final). "
                    "Use factorization='qr' or update_method='woodbury'.")
            return True
        if self.update_method == 'woodbury':
            return False
        # 'auto': QR for block (stable), Woodbury for non-block (fast)
        # QR update requires R_final from QR factorization
        return self.block and self.R_final is not None

    def _needs_chunking(self):
        return self.n_chunks is not None or linalg.is_distributed()

    def _move_train_to_cpu(self):
        """Move training data to CPU for chunked processing."""
        self.X_train = self.X_train.to('cpu')
        self.y_train = self.y_train.to('cpu')
        self.w_train = self.w_train.to('cpu')
        self.config_idxs_train = self.config_idxs_train.to('cpu')
        self.enrow_mask_train = self.enrow_mask_train.to('cpu')

    def _move_train_to_device(self):
        """Move training data back to GPU after chunked processing."""
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.w_train = self.w_train.to(self.device)
        self.config_idxs_train = self.config_idxs_train.to(self.device)
        self.enrow_mask_train = self.enrow_mask_train.to(self.device)
        self.sub_mask_train = self.sub_mask_train.to(self.device)

    # ------------------------------------------------------------------
    # Block metadata (identical in old cooks.py and mpi_cooks.py)
    # ------------------------------------------------------------------

    def _prepare_block_metadata(self):
        sort_perm = torch.argsort(self.config_idxs_train)

        self.X_train = self.X_train[sort_perm]
        self.y_train = self.y_train[sort_perm]
        self.w_train = self.w_train[sort_perm]
        self.enrow_mask_train = self.enrow_mask_train[sort_perm]
        self.config_idxs_train = self.config_idxs_train[sort_perm]

        unique_vals, counts = torch.unique_consecutive(self.config_idxs_train, return_counts=True)
        end_indices = torch.cumsum(counts, dim=0)
        start_indices = end_indices - counts

        self.group_metadata = torch.stack((unique_vals, start_indices, counts), dim=1)

        size_sort = torch.argsort(self.group_metadata[:, 2])
        self.group_metadata = self.group_metadata[size_sort]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _create_sub_mask(self):
        self.X_train.mul_(self.w_train)
        self.y_train.mul_(self.w_train)
        try:
            if self.stepwise:
                self._stepwise_cooks_sampling()
            else:
                self._onestep_cooks_sampling()
        finally:
            self.X_train.div_(self.w_train)
            self.y_train.div_(self.w_train)

    # ------------------------------------------------------------------
    # One-step Cook's (unchanged from original — SVD only, single GPU)
    # ------------------------------------------------------------------

    def _onestep_cooks_sampling(self):
        if self.block:
            raise NotImplementedError("Onestep Block Cook's Distance methods are not implemented yet.")
        if self._needs_chunking():
            raise NotImplementedError(
                "One-step Cook's does not support chunked/distributed mode. "
                "Use stepwise=True for chunked operation.")

        if self.U is None:
            self.U, self.S, self.Vh = torch.linalg.svd(self.X_train, full_matrices=False)

        leverage_scores = torch.sum(self.U[self.enrow_mask_train] ** 2, dim=1)

        tol = torch.finfo(self.X_train.dtype).eps * max(self.X_train.shape) * self.S[0]
        S_inv = torch.where(self.S > tol, 1 / self.S,
                            torch.tensor(0.0, device=self.device, dtype=self.dtype))

        term1 = self.U.T @ self.y_train
        term2 = S_inv.reshape(-1, 1) * term1
        coeffs = self.Vh.T @ term2

        preds = self.X_train[self.enrow_mask_train] @ coeffs
        en_residuals_sq = torch.square(preds - self.y_train[self.enrow_mask_train]).reshape(-1)

        self.onestep_en_cooks = en_residuals_sq * leverage_scores / (1 - leverage_scores) ** 2

        if self.sampling:
            cooks_probs = self.onestep_en_cooks / torch.sum(self.onestep_en_cooks)
            # CPU RNG so CPU and CUDA runs with the same seed pick identical
            # configs (CUDA RNG is an independent stream).
            indices = torch.multinomial(cooks_probs.cpu(), self.n_subsamples,
                                        replacement=False)
            sub_unique_config_idxs_train = self.unique_config_idxs_train.cpu()[indices]
        else:
            topk_vals, topk_indices = torch.topk(self.onestep_en_cooks, self.n_subsamples)
            sub_unique_config_idxs_train = self.unique_config_idxs_train[topk_indices.cpu()]

        self.sub_mask = torch.isin(self.config_idxs, sub_unique_config_idxs_train.to(device='cpu'))
        self.sub_mask_train = torch.isin(
            self.config_idxs_train,
            sub_unique_config_idxs_train.to(self.config_idxs_train.device))

    # ------------------------------------------------------------------
    # Initial subset
    # ------------------------------------------------------------------

    def _create_initial_sub_mask(self):
        # Nested samplers inherit the parent's partition state and global
        # config-id list via explicit kwargs, so every rank picks the same
        # configs. device=self.device keeps RNG consistent (torch.randperm
        # draws different sequences on CPU vs GPU for the same seed).
        train_target = 'cpu' if self._needs_chunking() else None
        parent_partitioned = self._is_partitioned
        parent_global_cfg = (self.unique_config_idxs_train
                             if parent_partitioned else None)

        if self.initial_subsampler == "leverage":
            inner = LeverageSubSampler(
                self.X_train, seed=self.seed, device=self.device,
                config_idxs=self.config_idxs_train, block=self.block,
                intercept=False,
                n_chunks=self.n_chunks,
                factorization=self.factorization,
                local_devices=self.local_devices,
                train_target_device=train_target,
                partitioned_override=parent_partitioned,
                unique_config_idxs_train_override=parent_global_cfg)
        elif self.initial_subsampler == "random":
            inner = RandomSubSampler(
                self.X_train, seed=self.seed, device=self.device,
                config_idxs=self.config_idxs_train, intercept=False,
                local_devices=self.local_devices,
                train_target_device=train_target,
                partitioned_override=parent_partitioned,
                unique_config_idxs_train_override=parent_global_cfg)
        else:
            raise ValueError(
                f"Unknown initial_subsampler {self.initial_subsampler!r}")

        self.sub_mask_train = inner.create_subsample(
            subsample_fraction=self.initial_subsample_fraction, seed=self.seed
        ).to(device=self.config_idxs_train.device)

        self.sub_mask = torch.isin(
            self.config_idxs.cpu(), self.config_idxs_train[self.sub_mask_train].cpu())

    # ------------------------------------------------------------------
    # Stepwise Cook's — merged logic from cooks.py + mpi_cooks.py
    # ------------------------------------------------------------------

    def _stepwise_cooks_sampling(self):
        distributed = linalg.is_distributed()
        world_size = linalg.get_world_size()
        rank = linalg.get_rank()

        # 1. Initial subset
        if self.sub_mask is None:
            self._create_initial_sub_mask()

        sub_X = self.X_train[self.sub_mask_train]
        sub_y = self.y_train[self.sub_mask_train]

        # 2. Initial factorization
        if self._use_qr_factorization():
            R, XTy = linalg.tsqr_r_xty(
                sub_X, sub_y,
                device=self.device,
                n_chunks=self.n_chunks,
                tree_reduction_threshold=self.tree_reduction_threshold,
                partitioned=self._is_partitioned,
                local_devices=self.local_devices,
            )
            # Broadcast R and XTy so every rank has identical state for the
            # greedy update loop.
            p = sub_X.shape[1]
            if distributed:
                R = linalg.broadcast_tensor(
                    R if rank == 0 else None, src=0,
                    shape=(p, p), dtype=sub_X.dtype, device=self.device)
                XTy = linalg.broadcast_tensor(
                    XTy if rank == 0 else None, src=0,
                    shape=(p, 1), dtype=sub_X.dtype, device=self.device)
            self.R_final = R
            self.XTX_inv = linalg.xtx_inv_from_r(R, device=self.device)
            self.XTy = XTy.to(self.device)
        else:
            # SVD path (single-process only)
            self.XTX_inv, _, _, _ = linalg.xtx_inv_from_svd(sub_X, device=self.device)
            self.XTy = sub_X.T @ sub_y

        # 3. Data stays on its current device (CPU in chunked mode, GPU otherwise).
        # The stepwise loop moves only the needed rows to GPU per iteration.

        # 4. Stepwise greedy loop
        if distributed:
            local_init = torch.tensor(
                [int(torch.sum(self.sub_mask.cpu() & self.enrow_mask.cpu()))],
                dtype=torch.int64, device=self.device)
            dist.all_reduce(local_init, op=dist.ReduceOp.SUM)
            n_subsamples_init = int(local_init.item())
        else:
            n_subsamples_init = int(torch.sum(self.sub_mask.cpu() & self.enrow_mask.cpu()))

        target_range = (range(n_subsamples_init, self.n_subsamples) if self.ascending
                        else range(n_subsamples_init, self.n_subsamples, -1))

        for _ in target_range:
            coeffs = self.XTX_inv @ self.XTy

            # Find best config to add/remove
            if self.block:
                best_val, best_config_id = self._compute_block_cooks(coeffs)
            else:
                best_val, best_config_id = self._compute_nonblock_cooks(coeffs)

            if distributed:
                pairs = [None] * world_size
                dist.all_gather_object(pairs, (float(best_val), int(best_config_id)))
                if self.ascending:
                    owner_rank = max(range(world_size), key=lambda i: pairs[i][0])
                else:
                    owner_rank = min(range(world_size), key=lambda i: pairs[i][0])
                best_config_id = pairs[owner_rank][1]
            else:
                owner_rank = 0

            # Update local mask (only owner rank has matching rows; others are empty)
            change_mask = (self.config_idxs_train == best_config_id)

            if self.ascending:
                self.sub_mask_train[change_mask] = True
            else:
                self.sub_mask_train[change_mask] = False

            # Owner broadcasts X_change, y_change so all ranks apply the same update.
            if distributed:
                if rank == owner_rank:
                    X_change_local = self.X_train[change_mask].to(self.device)
                    y_change_local = self.y_train[change_mask].to(self.device)
                    n_change = torch.tensor([X_change_local.shape[0]],
                                            dtype=torch.int64, device=self.device)
                else:
                    n_change = torch.tensor([0], dtype=torch.int64, device=self.device)
                dist.broadcast(n_change, src=owner_rank)
                n = int(n_change.item())
                p = self.X_train.shape[1]
                X_change = linalg.broadcast_tensor(
                    X_change_local if rank == owner_rank else None, src=owner_rank,
                    shape=(n, p), dtype=self.dtype, device=self.device)
                y_change = linalg.broadcast_tensor(
                    y_change_local if rank == owner_rank else None, src=owner_rank,
                    shape=(n, 1), dtype=self.dtype, device=self.device)
            else:
                X_change = self.X_train[change_mask].to(self.device)
                y_change = self.y_train[change_mask].to(self.device)

            if self._use_qr_update():
                self.R_final, self.XTX_inv, self.XTy = linalg.qr_update_add(
                    self.R_final, X_change, y_change, self.XTy, device=self.device)
            else:
                self.XTX_inv, self.XTy = linalg.woodbury_update(
                    self.XTX_inv, X_change, y_change, self.XTy, self.ascending)

        # 5. Finalize
        self.sub_mask = torch.isin(
            self.config_idxs,
            self.config_idxs_train[self.sub_mask_train].to(device='cpu'))

    # ------------------------------------------------------------------
    # Block Cook's Distance
    # ------------------------------------------------------------------

    def _compute_block_cooks(self, coeffs):
        """Block Cook's Distance: batched computation over config groups.
        Returns (best_val, best_config_id)."""
        BATCH_SIZE = 5000
        num_groups = self.group_metadata.shape[0]

        best_cooks_val = -float('inf') if self.ascending else float('inf')
        best_config_id = -1

        if num_groups == 0:
            return best_cooks_val, best_config_id

        active_ids = torch.unique(self.config_idxs_train[self.sub_mask_train]).to(self.device)
        max_len = self.group_metadata[:, 2].max().item()
        identity = torch.eye(max_len, device=self.device, dtype=self.dtype)

        for batch_start in range(0, num_groups, BATCH_SIZE):
            with torch.no_grad():
                batch_end = min(batch_start + BATCH_SIZE, num_groups)
                batch_meta = self.group_metadata[batch_start:batch_end]
                curr_batch_size = batch_meta.shape[0]
                batch_max_len = batch_meta[-1, 2].item()
                n_features = self.X_train.shape[1]

                # Build padded batch on CPU then transfer
                X_batch_cpu = torch.zeros((curr_batch_size, batch_max_len, n_features), dtype=self.dtype)
                y_batch_cpu = torch.zeros((curr_batch_size, batch_max_len, 1), dtype=self.dtype)

                for b_i in range(curr_batch_size):
                    _, start, count = batch_meta[b_i]
                    start, count = int(start), int(count)
                    X_batch_cpu[b_i, :count] = self.X_train[start:start + count]
                    y_batch_cpu[b_i, :count] = self.y_train[start:start + count]

                X_batch = X_batch_cpu.to(self.device)
                y_batch = y_batch_cpu.to(self.device)
                del X_batch_cpu, y_batch_cpu

                # H_c = X_c @ XTX_inv @ X_c^T
                temp = _batched_matmul(X_batch, self.XTX_inv.unsqueeze(0).expand(curr_batch_size, -1, -1))
                fake_lev = _batched_matmul(temp, X_batch.transpose(1, 2))
                del temp

                # Residuals r = X_c @ beta - y_c
                res = _batched_matmul(X_batch, coeffs.unsqueeze(0).expand(curr_batch_size, -1, -1)) - y_batch
                del X_batch, y_batch

                # Cook's: r^T @ inv(I +/- H_c) @ (H_c @ r)
                I_k = identity[:batch_max_len, :batch_max_len].unsqueeze(0)
                mat_to_inv = fake_lev.clone().add_(I_k) if self.ascending else (I_k - fake_lev)
                del I_k

                inv_mat = torch.linalg.inv(mat_to_inv)
                del mat_to_inv

                term_right = _batched_matmul(fake_lev, res)
                term_mid = _batched_matmul(inv_mat, term_right)
                cooks_vals = _batched_matmul(res.transpose(1, 2), term_mid).squeeze(-1).squeeze(-1)
                del fake_lev, res, inv_mat, term_right, term_mid

                # Mask already-selected configs
                batch_config_ids = batch_meta[:, 0].to(self.device)
                is_active = torch.isin(batch_config_ids, active_ids)

                if self.ascending:
                    cooks_vals[is_active] = -float('inf')
                    curr_best = torch.max(cooks_vals)
                    if curr_best > best_cooks_val:
                        best_cooks_val = curr_best
                        best_config_id = batch_config_ids[torch.argmax(cooks_vals)]
                else:
                    cooks_vals[~is_active] = float('inf')
                    curr_best = torch.min(cooks_vals)
                    if curr_best < best_cooks_val:
                        best_cooks_val = curr_best
                        best_config_id = batch_config_ids[torch.argmin(cooks_vals)]

        del identity
        best_val = best_cooks_val.item() if torch.is_tensor(best_cooks_val) else float(best_cooks_val)
        best_id = best_config_id.cpu().item() if torch.is_tensor(best_config_id) else int(best_config_id)
        return best_val, best_id

    # ------------------------------------------------------------------
    # Non-block Cook's Distance
    # ------------------------------------------------------------------

    def _compute_nonblock_cooks(self, coeffs):
        """Non-block Cook's Distance on energy rows.
        D_i = e_i^2 * h_i / (1 + h_i). Returns (best_val, best_config_id)."""
        with torch.no_grad():
            # Energy rows only (small). Skip if this rank has none.
            if self.enrow_mask_train.sum().item() == 0:
                return (-float('inf') if self.ascending else float('inf')), -1

            X_en = self.X_train[self.enrow_mask_train].to(self.device)
            y_en = self.y_train[self.enrow_mask_train].to(self.device)

            en_residuals = X_en @ coeffs - y_en
            leverage_scores = torch.sum((X_en @ self.XTX_inv) * X_en, dim=1)
            e_cooks = en_residuals.reshape(-1) ** 2 * leverage_scores / (1 + leverage_scores)

            del X_en, y_en, en_residuals, leverage_scores

            is_active = self.sub_mask_train[self.enrow_mask_train].to(self.device)

            # Local config IDs: use the energy-row positions of this rank's
            # config_idxs_train, which is correct for partitioned data (global
            # unique_config_idxs_train won't align with e_cooks indexing).
            local_en_config_ids = self.config_idxs_train[self.enrow_mask_train].to(self.device)

            if self.ascending:
                e_cooks[is_active] = -float('inf')
                best_idx = torch.argmax(e_cooks)
            else:
                e_cooks[~is_active] = float('inf')
                best_idx = torch.argmin(e_cooks)

            best_val = e_cooks[best_idx].item()
            best_config_id = int(local_en_config_ids[best_idx].item())
            return best_val, best_config_id
