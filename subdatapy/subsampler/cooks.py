import torch
import torch.distributed as dist
import warnings
from .random import RandomSubSampler
from .leverage import LeverageSubSampler
from subdatapy import linalg
from subdatapy import partition as _partition


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
                 device='cuda', dtype=torch.float64, block=False, stepwise=False, sampling=True,
                 ascending=True, initial_subsampler="random",
                 initial_subsample_fraction=1, U=None, S=None, Vh=None,
                 # Chunked / distributed parameters:
                 n_chunks=None,
                 factorization='auto',       # 'svd', 'qr', or 'auto'
                 update_method='auto',        # 'woodbury', 'qr', or 'auto'
                 tree_reduction_threshold=10,
                 local_devices=None,
                 partitioned_override=None,
                 unique_config_idxs_train_override=None,
                 ):

        self.n_chunks = n_chunks

        super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed,
                         test_mask=test_mask, config_idxs=config_idxs,
                         enrow_mask=enrow_mask, intercept=intercept, device=device,
                         dtype=dtype, local_devices=local_devices,
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
        # The QR update (qr_update_add) can only APPEND rows; it has no
        # downdate. So descending (removing configs) must use Woodbury, whose
        # rank-k formula handles removal (inv(I - X A^{-1} X^T)). Note: that
        # removal inverse can be ill-conditioned when dropping a very
        # high-leverage config, so descending is inherently a touch less
        # robust than ascending (which can use the stabler QR update).
        if not self.ascending:
            if self.update_method == 'qr':
                raise ValueError(
                    "update_method='qr' cannot remove rows, so it does not "
                    "support descending stepwise Cook's. Use "
                    "update_method='woodbury' or 'auto' (which selects "
                    "Woodbury for descending).")
            return False
        if self.update_method == 'qr':
            if self.R_final is None:
                raise ValueError(
                    "update_method='qr' requires QR factorization (R_final). "
                    "Use factorization='qr' or update_method='woodbury'.")
            return True
        if self.update_method == 'woodbury':
            return False
        # 'auto' (ascending): QR for block (stable), Woodbury for non-block
        # (fast). QR update requires R_final from QR factorization.
        return self.block and self.R_final is not None

    def _needs_chunking(self):
        return self.n_chunks is not None or linalg.is_distributed()

    def _train_rows(self, sel):
        """Weighted (design, target) training rows selected by ``sel`` (a bool
        mask, slice, or index over the train rows), gathered onto self.device.

        Replaces the old materialized + in-place-weighted ``self.X_train[sel]``
        / ``self.y_train[sel]``: rows are gathered from the host ``self.X`` by
        their global index and weighted here (per call), so the design matrix
        is never copied or mutated.
        """
        if torch.is_tensor(sel):
            sel = sel.cpu()
        gi = self.train_idx[sel]
        wsel = self.w_train[sel].reshape(-1, 1)
        Xw = (self.X[gi] * wsel).to(self.device)
        yw = (self.y_train[sel].reshape(-1, 1) * wsel).to(self.device)
        return Xw, yw

    # ------------------------------------------------------------------
    # Block metadata (identical in old cooks.py and mpi_cooks.py)
    # ------------------------------------------------------------------

    def _prepare_block_metadata(self):
        sort_perm = torch.argsort(self.config_idxs_train)

        # Reorder the per-row train siblings and the row-index map by config so
        # group_metadata can slice contiguous [start:start+count] ranges. The
        # design matrix is never reordered — train_idx points back into self.X.
        self.y_train = self.y_train[sort_perm]
        self.w_train = self.w_train[sort_perm]
        self.enrow_mask_train = self.enrow_mask_train[sort_perm]
        self.config_idxs_train = self.config_idxs_train[sort_perm]
        self.train_idx = self.train_idx[sort_perm.to(self.train_idx.device)]

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
        # Weighting is applied per gather inside the methods below (see
        # _train_rows), so the host design matrix is never weighted in place.
        if self.stepwise:
            self._stepwise_cooks_sampling()
        else:
            self._onestep_cooks_sampling()

    # ------------------------------------------------------------------
    # One-step Cook's (unchanged from original — SVD only, single GPU)
    # ------------------------------------------------------------------

    def _onestep_cooks_sampling(self):
        if self.block:
            raise NotImplementedError("Onestep Block Cook's Distance methods are not implemented yet.")
        # auto/qr -> TSQR path (chunkable + distributed); svd -> single-pass.
        if self._use_qr_factorization():
            self._onestep_cooks_qr()
        else:
            if self._needs_chunking():
                raise NotImplementedError(
                    "One-step Cook's with factorization='svd' does not support "
                    "chunked/distributed mode. Use factorization='qr' or 'auto', "
                    "or stepwise=True.")
            self._onestep_cooks_svd()

    def _onestep_cooks_svd(self):
        # Single-pass WLS leverage from the SVD. Gather the weighted training
        # rows onto the device (the host design matrix is never SVD'd in place).
        Xtr, ytr = self._train_rows(slice(None))
        enrow = self.enrow_mask_train.to(self.device)

        if self.U is None:
            self.U, self.S, self.Vh = torch.linalg.svd(Xtr, full_matrices=False)

        leverage_scores = torch.sum(self.U[enrow] ** 2, dim=1)

        tol = torch.finfo(self.dtype).eps * max(Xtr.shape) * self.S[0]
        S_inv = torch.where(self.S > tol, 1 / self.S,
                            torch.tensor(0.0, device=self.device, dtype=self.dtype))

        term1 = self.U.T @ ytr
        term2 = S_inv.reshape(-1, 1) * term1
        coeffs = self.Vh.T @ term2

        preds = Xtr[enrow] @ coeffs
        en_residuals_sq = torch.square(preds - ytr[enrow]).reshape(-1)
        cooks = en_residuals_sq * leverage_scores / (1 - leverage_scores) ** 2

        # Map each cooks score back to the actual config id of its energy row
        # (same robust mapping as the QR path). Using unique_config_idxs_train
        # here would assume config ids are sorted by row position and pick the
        # wrong configs otherwise.
        self.onestep_en_cooks = cooks
        en_config_ids = self.config_idxs_train[self.enrow_mask_train].cpu()
        self._onestep_select(cooks.cpu(), en_config_ids)

    def _onestep_cooks_qr(self):
        # TSQR leverage: R/X^T y from all (in-place weighted) train rows,
        # streamed/chunked; leverage h_i = ||R^{-T} x_i||^2 and residuals on
        # the (small) energy rows only. Works single-process, chunked, and
        # partitioned-distributed.
        distributed = linalg.is_distributed()
        rank = linalg.get_rank()
        p = self.X.shape[1]

        R, XTy = linalg.tsqr_r_xty(
            self.X, self.y_train, x_idx=self.train_idx, w=self.w_train,
            device=self.device, n_chunks=self.n_chunks,
            tree_reduction_threshold=self.tree_reduction_threshold,
            partitioned=self._is_partitioned, local_devices=self.local_devices)
        if distributed:
            R = linalg.broadcast_tensor(R if rank == 0 else None, src=0,
                                        shape=(p, p), dtype=self.dtype, device=self.device)
            XTy = linalg.broadcast_tensor(XTy if rank == 0 else None, src=0,
                                          shape=(p, 1), dtype=self.dtype, device=self.device)
        coeffs = linalg.solve_from_r_xty(R, XTy)

        # Energy rows only (one per config) — small, gather to the device.
        en = self.enrow_mask_train
        if int(en.sum()) > 0:
            Xen, yen = self._train_rows(en)
            h = linalg.leverage_scores_from_r(Xen, R, device=self.device).reshape(-1)
            e = (Xen @ coeffs - yen).reshape(-1)
            cooks = (e ** 2 * h / (1 - h) ** 2).cpu()
            en_config_ids = self.config_idxs_train[en].cpu()
        else:
            cooks = torch.empty(0, dtype=self.dtype)
            en_config_ids = torch.empty(0, dtype=torch.int64)

        self.onestep_en_cooks = cooks
        # Robust config-id mapping (works when energy rows span ranks).
        if distributed:
            pairs = [None] * linalg.get_world_size()
            dist.all_gather_object(pairs, (en_config_ids.tolist(), cooks.tolist()))
            ids, vals = [], []
            for cid_list, v_list in pairs:
                ids.extend(cid_list)
                vals.extend(v_list)
            en_config_ids = torch.tensor(ids, dtype=torch.int64)
            cooks = torch.tensor(vals, dtype=self.dtype)
        self._onestep_select(cooks, en_config_ids)

    def _onestep_select(self, cooks, config_ids):
        """Pick n_subsamples configs from per-config Cook's scores (CPU) and set
        sub_mask / sub_mask_train. cooks and config_ids are aligned 1-D CPU
        tensors. Sampling uses CPU RNG so CPU and CUDA runs match."""
        if self.sampling:
            probs = cooks / torch.sum(cooks)
            idx = torch.multinomial(probs, self.n_subsamples, replacement=False)
        else:
            _, idx = torch.topk(cooks, self.n_subsamples)
        chosen = config_ids[idx]
        self.sub_mask = torch.isin(self.config_idxs, chosen)
        self.sub_mask_train = torch.isin(
            self.config_idxs_train, chosen.to(self.config_idxs_train.device))

    # ------------------------------------------------------------------
    # Initial subset
    # ------------------------------------------------------------------

    def _create_initial_sub_mask(self):
        # Pick the initial config set, then rebuild the masks via isin so they
        # are correct regardless of row order (block sort) and partitioning.
        parent_partitioned = self._is_partitioned
        parent_global_cfg = (self.unique_config_idxs_train
                             if parent_partitioned else None)

        if self.initial_subsampler == "random":
            # Random config pick over the (global) train configs — needs no X.
            # Matches RandomSubSampler.create_subsample's CPU-RNG selection.
            if self.seed is not None:
                torch.manual_seed(self.seed)
            n_total = len(self.unique_config_idxs_train)
            n_init = round(n_total * self.initial_subsample_fraction)
            perm = torch.randperm(n_total)
            chosen = self.unique_config_idxs_train.cpu()[perm[:n_init]]
        elif self.initial_subsampler == "leverage":
            # Leverage over the parent's TRAIN rows, sharing self.X by reference
            # (test_mask marks the non-train rows) — no design-matrix copy.
            inner = LeverageSubSampler(
                self.X, w=self.w, seed=self.seed, device=self.device,
                dtype=self.storage_dtype,
                config_idxs=self.config_idxs, enrow_mask=self.enrow_mask,
                test_mask=self.test_mask, block=self.block, intercept=False,
                n_chunks=self.n_chunks, factorization=self.factorization,
                local_devices=self.local_devices,
                partitioned_override=parent_partitioned,
                unique_config_idxs_train_override=parent_global_cfg)
            inner.create_subsample(
                subsample_fraction=self.initial_subsample_fraction, seed=self.seed)
            local_chosen = torch.unique(self.config_idxs[inner.sub_mask.cpu()])
            chosen = (_partition.build_global_config_ids(local_chosen)
                      if parent_partitioned else local_chosen)
        else:
            raise ValueError(
                f"Unknown initial_subsampler {self.initial_subsampler!r}")

        chosen = chosen.cpu()
        self.sub_mask = torch.isin(self.config_idxs, chosen)
        self.sub_mask_train = torch.isin(
            self.config_idxs_train, chosen.to(self.config_idxs_train.device))

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

        # Subsample rows (indices into the host X) and their weights/targets.
        sm = self.sub_mask_train.cpu()
        sub_idx = self.train_idx[sm]
        sub_w = self.w_train[sm]
        sub_y = self.y_train[sm]
        p = self.X.shape[1]

        # 2. Initial factorization
        if self._use_qr_factorization():
            R, XTy = linalg.tsqr_r_xty(
                self.X, sub_y, x_idx=sub_idx, w=sub_w,
                device=self.device,
                n_chunks=self.n_chunks,
                tree_reduction_threshold=self.tree_reduction_threshold,
                partitioned=self._is_partitioned,
                local_devices=self.local_devices,
            )
            # Broadcast R and XTy so every rank has identical state for the
            # greedy update loop.
            if distributed:
                R = linalg.broadcast_tensor(
                    R if rank == 0 else None, src=0,
                    shape=(p, p), dtype=self.dtype, device=self.device)
                XTy = linalg.broadcast_tensor(
                    XTy if rank == 0 else None, src=0,
                    shape=(p, 1), dtype=self.dtype, device=self.device)
            self.R_final = R
            self.XTX_inv = linalg.xtx_inv_from_r(R, device=self.device)
            self.XTy = XTy.to(self.device)
        else:
            # SVD path (single-process only): gather the weighted subset onto
            # the device for the dense SVD.
            Xw, yw = self._train_rows(self.sub_mask_train)
            self.XTX_inv, _, _, _ = linalg.xtx_inv_from_svd(Xw, device=self.device)
            self.XTy = Xw.T @ yw

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
                    X_change_local, y_change_local = self._train_rows(change_mask)
                    n_change = torch.tensor([X_change_local.shape[0]],
                                            dtype=torch.int64, device=self.device)
                else:
                    X_change_local = y_change_local = None
                    n_change = torch.tensor([0], dtype=torch.int64, device=self.device)
                dist.broadcast(n_change, src=owner_rank)
                n = int(n_change.item())
                X_change = linalg.broadcast_tensor(
                    X_change_local if rank == owner_rank else None, src=owner_rank,
                    shape=(n, p), dtype=self.dtype, device=self.device)
                y_change = linalg.broadcast_tensor(
                    y_change_local if rank == owner_rank else None, src=owner_rank,
                    shape=(n, 1), dtype=self.dtype, device=self.device)
            else:
                X_change, y_change = self._train_rows(change_mask)

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
                n_features = self.X.shape[1]

                # Build padded batch on CPU then transfer. Rows are gathered
                # from the host X by their (config-sorted) global index and
                # weighted here — the design matrix is never copied/weighted
                # in place.
                X_batch_cpu = torch.zeros((curr_batch_size, batch_max_len, n_features), dtype=self.dtype)
                y_batch_cpu = torch.zeros((curr_batch_size, batch_max_len, 1), dtype=self.dtype)

                for b_i in range(curr_batch_size):
                    _, start, count = batch_meta[b_i]
                    start, count = int(start), int(count)
                    gi = self.train_idx[start:start + count]
                    wsel = self.w_train[start:start + count].reshape(-1, 1)
                    X_batch_cpu[b_i, :count] = self.X[gi] * wsel
                    y_batch_cpu[b_i, :count] = self.y_train[start:start + count].reshape(-1, 1) * wsel

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

            X_en, y_en = self._train_rows(self.enrow_mask_train)

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
