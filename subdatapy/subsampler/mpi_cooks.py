import torch
from mpi4py import MPI
from contextlib import contextmanager
from .random import RandomSubSampler


@contextmanager
def gpu_memory_context():
    """Flush GPU cache before and after a block."""
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        torch.cuda.empty_cache()


class TSQRMPICookSubSampler(RandomSubSampler):
    """
        Case 1: n_ranks=None, mpi_size=1  -> Regular QR, no chunking
        Case 2: n_ranks=N,    mpi_size=1  -> Sequential TSQR, N chunks on one GPU
        Case 3: n_ranks=None, mpi_size>1  -> Parallel TSQR, one chunk per MPI rank
        Case 4: n_ranks=N*M,  mpi_size=M  -> Hybrid TSQR, N/M chunks per rank
    """

    def __init__(self, X, y, w=None, test_fraction=0.0, seed=None, test_mask=None,
                 config_idxs=None, enrow_mask=None, intercept=True,
                 device='cuda', block=False, stepwise=False, sampling=True, ascending=True,
                 initial_subsampler="random", initial_subsample_fraction=1,
                 n_ranks=None, tree_reduction_threshold=10):

        self.comm     = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        # Rank 0 holds the full dataset; other ranks hold a (1,1) placeholder.
        if self.mpi_rank == 0:
            super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed,
                             test_mask=test_mask, config_idxs=config_idxs,
                             enrow_mask=enrow_mask, intercept=intercept, device=device)
        else:
            super().__init__(torch.zeros((1, 1)), y=torch.zeros((1, 1)), device=device)

        self._select_execution_mode(n_ranks)
        self.tree_reduction_threshold = tree_reduction_threshold

        if self.mpi_rank == 0:
            mode_type = "Block" if block else "Non-Block"
            print(f"[Rank 0] {mode_type} Mode: {self.mode}, MPI size: {self.mpi_size}, "
                  f"n_ranks: {self.n_ranks}, local_chunks/rank: {self.local_n_chunks}", flush=True)

        self.block                      = block
        self.stepwise                   = stepwise
        self.sampling                   = sampling
        self.ascending                  = ascending
        self.initial_subsampler         = initial_subsampler
        self.initial_subsample_fraction = initial_subsample_fraction

        self._setup_data_storage()

        self.R_final = None
        self.XTX_inv = None
        self.XTy     = None

        if self.block and self.mpi_rank == 0:
            self._prepare_block_metadata()


    def _select_execution_mode(self, n_ranks):
        """Set self.mode and chunk counts from (n_ranks, mpi_size)."""
        if n_ranks is None:
            if self.mpi_size == 1:
                self.n_ranks = 1
                self.mode    = "regular_qr"
            else:
                self.n_ranks = self.mpi_size
                self.mode    = "parallel_tsqr"
        else:
            if self.mpi_size == 1:
                self.n_ranks = n_ranks
                self.mode    = "sequential_tsqr"
            else:
                if n_ranks % self.mpi_size != 0:
                    raise ValueError(
                        f"n_ranks ({n_ranks}) must be a multiple of mpi_size ({self.mpi_size})"
                    )
                self.n_ranks = n_ranks
                self.mode    = "hybrid_tsqr"

        self.local_n_chunks = self.n_ranks // self.mpi_size


    def _setup_data_storage(self):
        """Move training data to CPU for chunked modes (Cases 2, 3, 4)."""
        if self.mpi_rank == 0 and self.mode != "regular_qr":
            print(f"[Rank 0] {self.mode}: Using CPU storage for chunked processing", flush=True)
            self.X_train           = self.X_train.to('cpu')
            self.y_train           = self.y_train.to('cpu')
            self.w_train           = self.w_train.to('cpu')
            self.config_idxs_train = self.config_idxs_train.to('cpu')
            self.enrow_mask_train  = self.enrow_mask_train.to('cpu')


    def _prepare_block_metadata(self):
        """Sort rows by config ID and build per-config index metadata.

        group_metadata columns: [config_id, start_row, row_count]
        Sorted by row_count for uniform batch padding.
        """
        perm = torch.argsort(self.config_idxs_train)

        self.X_train           = self.X_train[perm]
        self.y_train           = self.y_train[perm]
        self.w_train           = self.w_train[perm]
        self.enrow_mask_train  = self.enrow_mask_train[perm]
        self.config_idxs_train = self.config_idxs_train[perm]

        unique_ids, counts = torch.unique_consecutive(self.config_idxs_train, return_counts=True)
        end_rows   = torch.cumsum(counts, dim=0)
        start_rows = end_rows - counts

        self.group_metadata = torch.stack((unique_ids, start_rows, counts), dim=1)

        # Sort by config size for efficient batch padding.
        size_order          = torch.argsort(self.group_metadata[:, 2])
        self.group_metadata = self.group_metadata[size_order]


    def _create_sub_mask(self):
        """Pre-weight data, run stepwise selection, then restore weights."""
        self.X_train = self.X_train * self.w_train.reshape(-1, 1)
        self.y_train = self.y_train * self.w_train.reshape(-1, 1)

        if self.stepwise:
            self._stepwise_tsqr_sampling()
        else:
            raise NotImplementedError("Only stepwise TSQR is implemented.")

        self.X_train = self.X_train / self.w_train.reshape(-1, 1)
        self.y_train = self.y_train / self.w_train.reshape(-1, 1)


    def _subsample(self):
        """Only rank 0 holds real data; skip on other ranks."""
        if self.mpi_rank == 0:
            super()._subsample()


    def _create_initial_sub_mask(self):
        """Build the starting subset using leverage or random sampling."""
        if self.initial_subsampler == "leverage":
            from .leverage import LeverageSubSampler
            lss = LeverageSubSampler(self.X_train, seed=self.seed, device=self.device,
                                     config_idxs=self.config_idxs_train, block=self.block,
                                     intercept=False)
            self.sub_mask_train = lss.create_subsample(
                subsample_fraction=self.initial_subsample_fraction, seed=self.seed)
        elif self.initial_subsampler == "random":
            
            rss = RandomSubSampler(self.X_train, seed=self.seed, device=self.device,
                                   config_idxs=self.config_idxs_train)
            self.sub_mask_train = rss.create_subsample(
                subsample_fraction=self.initial_subsample_fraction, seed=self.seed)

        self.sub_mask = torch.isin(
            self.config_idxs.cpu(), self.config_idxs_train[self.sub_mask_train].cpu())

        n_configs = len(torch.unique(self.config_idxs_train[self.sub_mask_train]))
        print(f"[Rank 0] Initial subset: {torch.sum(self.sub_mask_train).item()} rows, "
              f"{n_configs} configs", flush=True)


    def _compute_tsqr_qr(self, row_mask):
        """Route to the correct QR method for the active execution mode."""
        if self.mpi_rank == 0 and not self.block:
            print(f"[Rank 0] Non-block: {torch.sum(row_mask).item():,} rows", flush=True)

        if self.mode == "regular_qr":
            return self._compute_regular_qr(row_mask)
        elif self.mode == "sequential_tsqr":
            return self._compute_sequential_tsqr(row_mask)
        elif self.mode == "parallel_tsqr":
            return self._compute_parallel_tsqr(row_mask)
        elif self.mode == "hybrid_tsqr":
            return self._compute_hybrid_tsqr(row_mask)


    def _compute_regular_qr(self, row_mask):
        """Case 1: Single-pass QR on the full selected subset."""
        if self.mpi_rank == 0:
            print("[Rank 0] Case 1: Regular QR", flush=True)

        with torch.no_grad(), gpu_memory_context():
            selected_indices = torch.where(row_mask)[0]

            if self.mpi_rank == 0:
                print(f"[Rank 0] {len(selected_indices):,} rows", flush=True)

            sub_X = self.X_train[selected_indices].to(self.device)
            sub_y = self.y_train[selected_indices].to(self.device)

            _, R = torch.linalg.qr(sub_X, mode='r')
            XTy  = sub_X.T @ sub_y

            del sub_X, sub_y

        return R, XTy


    def _reduce_R_matrices(self, R_list, device='cuda'):
        """TSQR reduction: QR([R1; R2; ...]) -> R_combined."""
        if len(R_list) == 0:
            return None
        if len(R_list) == 1:
            return R_list[0]

        with torch.no_grad(), gpu_memory_context():
            R_stacked = torch.cat([r.to(device) for r in R_list], dim=0)
            _, R      = torch.linalg.qr(R_stacked, mode='r')
            result    = R.cpu()
            del R_stacked, R

        return result


    def _compute_sequential_tsqr(self, row_mask):
        """Case 2: TSQR over n_ranks sequential chunks on one GPU."""
        selected_indices = torch.where(row_mask)[0]
        n_selected       = len(selected_indices)
        chunk_size       = (n_selected + self.n_ranks - 1) // self.n_ranks

        if self.mpi_rank == 0:
            print(f"[Rank 0] Case 2: Sequential TSQR - {n_selected:,} rows, "
                  f"{self.n_ranks} chunks (~{chunk_size:,} rows each)", flush=True)

        R_matrices = []
        XTy_sum    = torch.zeros((self.X_train.shape[1], 1), dtype=self.dtype)

        for chunk_idx in range(self.n_ranks):
            start = chunk_idx * chunk_size
            end   = min(start + chunk_size, n_selected)

            if start >= n_selected:
                break

            with torch.no_grad(), gpu_memory_context():
                chunk_indices = selected_indices[start:end]
                X_chunk = self.X_train[chunk_indices].to(self.device)
                y_chunk = self.y_train[chunk_indices].to(self.device)

                _, R_local = torch.linalg.qr(X_chunk, mode='r')
                R_matrices.append(R_local.cpu())
                XTy_sum += (X_chunk.T @ y_chunk).cpu()

                del X_chunk, y_chunk, R_local

            # Periodic reduction to bound the number of R matrices in memory.
            if len(R_matrices) >= self.tree_reduction_threshold:
                if self.mpi_rank == 0:
                    print(f"[Rank 0]   Reducing {len(R_matrices)} R matrices", flush=True)
                R_matrices = [self._reduce_R_matrices(R_matrices, device=self.device)]

            if (chunk_idx + 1) % 10 == 0 and self.mpi_rank == 0:
                print(f"[Rank 0]   {chunk_idx + 1}/{self.n_ranks} chunks done", flush=True)

        if self.mpi_rank == 0:
            print(f"[Rank 0] Final reduction of {len(R_matrices)} R matrices", flush=True)

        R_final = self._reduce_R_matrices(R_matrices, device=self.device)
        return R_final, XTy_sum.to(self.device)


    def _compute_parallel_tsqr(self, row_mask):
        """Case 3: One chunk per MPI rank, QR computed in parallel."""
        if self.mpi_rank == 0:
            selected_indices = torch.where(row_mask)[0]
            n_selected       = len(selected_indices)
            n_features       = self.X_train.shape[1]
            print(f"[Rank 0] Case 3: Parallel TSQR - {n_selected:,} rows, {self.n_ranks} ranks", flush=True)
        else:
            selected_indices = None
            n_selected       = None
            n_features       = None

        n_selected       = self.comm.bcast(n_selected,       root=0)
        n_features       = self.comm.bcast(n_features,       root=0)
        selected_indices = self.comm.bcast(selected_indices, root=0)

        chunk_size = (n_selected + self.n_ranks - 1) // self.n_ranks
        start      = self.mpi_rank * chunk_size
        end        = min(start + chunk_size, n_selected)

        R_local_cpu   = None
        XTy_local_cpu = None

        if start < n_selected:
            chunk_indices = selected_indices[start:end]

            if self.mpi_rank == 0:
                # Send each rank its slice before processing rank 0's own chunk.
                print(f"[Rank 0] Distributing chunks to {self.mpi_size} ranks", flush=True)
                for other_rank in range(1, self.mpi_size):
                    other_start = other_rank * chunk_size
                    other_end   = min(other_start + chunk_size, n_selected)
                    if other_start < n_selected:
                        other_indices = selected_indices[other_start:other_end]
                        self.comm.send(
                            {'X': self.X_train[other_indices].cpu(),
                             'y': self.y_train[other_indices].cpu()},
                            dest=other_rank, tag=other_rank)
                print(f"[Rank 0] Distribution complete, processing own chunk", flush=True)
                chunk_data = {'X': self.X_train[chunk_indices],
                              'y': self.y_train[chunk_indices]}
            else:
                chunk_data = self.comm.recv(source=0, tag=self.mpi_rank)

            with torch.no_grad(), gpu_memory_context():
                X_chunk = chunk_data['X'].to(self.device)
                y_chunk = chunk_data['y'].to(self.device)

                _, R_local  = torch.linalg.qr(X_chunk, mode='r')
                XTy_local   = X_chunk.T @ y_chunk

                R_local_cpu   = R_local.cpu()
                XTy_local_cpu = XTy_local.cpu()

                del chunk_data, X_chunk, y_chunk, R_local, XTy_local

            print(f"[Rank {self.mpi_rank}] Local QR complete", flush=True)

        # Gather all local results to rank 0.
        all_R   = self.comm.gather(R_local_cpu,   root=0)
        all_XTy = self.comm.gather(XTy_local_cpu, root=0)

        if self.mpi_rank == 0:
            print(f"[Rank 0] Gathering complete, final reduction", flush=True)

            all_R   = [r   for r   in all_R   if r   is not None]
            all_XTy = [xty for xty in all_XTy if xty is not None]

            R_final = self._reduce_R_matrices(all_R, device=self.device)
            XTy_sum = sum(xty.to(self.device) for xty in all_XTy)

            del all_R, all_XTy
            print(f"[Rank 0] Parallel TSQR complete", flush=True)
            return R_final, XTy_sum
        else:
            return None, None


    def _compute_hybrid_tsqr(self, row_mask):
        """Case 4: Multiple chunks per rank, streamed from rank 0."""
        if self.mpi_rank == 0:
            selected_indices = torch.where(row_mask)[0]
            n_selected       = len(selected_indices)
            n_features       = self.X_train.shape[1]
            print(f"[Rank 0] Case 4: Hybrid TSQR - {n_selected:,} rows, "
                  f"{self.n_ranks} total chunks, {self.local_n_chunks} per rank", flush=True)
        else:
            selected_indices = None
            n_selected       = None
            n_features       = None

        n_selected       = self.comm.bcast(n_selected,       root=0)
        n_features       = self.comm.bcast(n_features,       root=0)
        selected_indices = self.comm.bcast(selected_indices, root=0)

        chunk_size  = (n_selected + self.n_ranks - 1) // self.n_ranks
        R_matrices  = []
        XTy_local   = torch.zeros((n_features, 1), dtype=self.dtype)

        if self.mpi_rank == 0:
            # Stream each rank's chunks then process rank 0's own chunks.
            for other_rank in range(1, self.mpi_size):
                print(f"[Rank 0]   Streaming to rank {other_rank}...", flush=True)

                num_chunks_for_rank = sum(
                    1 for li in range(self.local_n_chunks)
                    if (other_rank * self.local_n_chunks + li) * chunk_size < n_selected)
                self.comm.send(num_chunks_for_rank, dest=other_rank, tag=other_rank * 10000)

                for local_idx in range(self.local_n_chunks):
                    global_idx = other_rank * self.local_n_chunks + local_idx
                    start      = global_idx * chunk_size
                    end        = min(start + chunk_size, n_selected)
                    if start < n_selected:
                        chunk_data = {'X': self.X_train[selected_indices[start:end]].cpu(),
                                      'y': self.y_train[selected_indices[start:end]].cpu()}
                        self.comm.send(chunk_data, dest=other_rank,
                                       tag=other_rank * 10000 + local_idx + 1)
                        del chunk_data

                print(f"[Rank 0]   Streaming to rank {other_rank} complete", flush=True)

            print(f"[Rank 0] Processing own {self.local_n_chunks} chunks", flush=True)
            for local_idx in range(self.local_n_chunks):
                start = local_idx * chunk_size
                end   = min(start + chunk_size, n_selected)
                if start < n_selected:
                    with torch.no_grad(), gpu_memory_context():
                        X_chunk = self.X_train[selected_indices[start:end]].to(self.device)
                        y_chunk = self.y_train[selected_indices[start:end]].to(self.device)

                        _, R_local = torch.linalg.qr(X_chunk, mode='r')
                        R_matrices.append(R_local.cpu())
                        XTy_local += (X_chunk.T @ y_chunk).cpu()
                        del X_chunk, y_chunk, R_local

                    if len(R_matrices) >= self.tree_reduction_threshold:
                        print(f"[Rank 0]   Reducing {len(R_matrices)} R matrices", flush=True)
                        R_matrices = [self._reduce_R_matrices(R_matrices, device=self.device)]

                    if (local_idx + 1) % 5 == 0:
                        print(f"[Rank 0]   {local_idx + 1}/{self.local_n_chunks} chunks done", flush=True)
        else:
            # Receive and process each chunk immediately 
            num_chunks = self.comm.recv(source=0, tag=self.mpi_rank * 10000)
            print(f"[Rank {self.mpi_rank}] Receiving {num_chunks} chunks", flush=True)

            for local_idx in range(num_chunks):
                chunk_data = self.comm.recv(source=0, tag=self.mpi_rank * 10000 + local_idx + 1)

                with torch.no_grad(), gpu_memory_context():
                    X_chunk = chunk_data['X'].to(self.device)
                    y_chunk = chunk_data['y'].to(self.device)
                    del chunk_data

                    _, R_local = torch.linalg.qr(X_chunk, mode='r')
                    R_matrices.append(R_local.cpu())
                    XTy_local += (X_chunk.T @ y_chunk).cpu()
                    del X_chunk, y_chunk, R_local

                if len(R_matrices) >= self.tree_reduction_threshold:
                    print(f"[Rank {self.mpi_rank}]   Reducing {len(R_matrices)} R matrices", flush=True)
                    R_matrices = [self._reduce_R_matrices(R_matrices, device=self.device)]

                if (local_idx + 1) % 5 == 0:
                    print(f"[Rank {self.mpi_rank}]   {local_idx + 1}/{num_chunks} chunks done", flush=True)

        # Local reduction before gathering
        print(f"[Rank {self.mpi_rank}] Local reduction: {len(R_matrices)} R matrices", flush=True)

        if R_matrices:
            R_local_final = self._reduce_R_matrices(R_matrices, device=self.device).cpu()
            XTy_local_cpu = XTy_local
            del R_matrices
        else:
            R_local_final = None
            XTy_local_cpu = None

        torch.cuda.empty_cache()

        print(f"[Rank {self.mpi_rank}] Gathering results", flush=True)
        all_R   = self.comm.gather(R_local_final, root=0)
        all_XTy = self.comm.gather(XTy_local_cpu, root=0)

        if self.mpi_rank == 0:
            print(f"[Rank 0] Final reduction", flush=True)

            all_R   = [r   for r   in all_R   if r   is not None]
            all_XTy = [xty for xty in all_XTy if xty is not None]

            R_final = self._reduce_R_matrices(all_R, device=self.device)
            XTy_sum = sum(xty.to(self.device) for xty in all_XTy)

            del all_R, all_XTy
            print(f"[Rank 0] Hybrid TSQR complete", flush=True)
            return R_final, XTy_sum
        else:
            return None, None


    def _update_tsqr_add_config(self, config_id):
        """Block mode: add one config via rank-k QR update.

        Appends new rows to R and re-factorise: R_new = QR([R_old; X_new])[1]
        """
        if self.mpi_rank != 0:
            return

        with torch.no_grad(), gpu_memory_context():
            config_to_change = config_id if isinstance(config_id, int) else config_id.cpu().item()
            change_mask      = (self.config_idxs_train == config_to_change)

            X_change = self.X_train[change_mask].to(self.device)
            y_change = self.y_train[change_mask].to(self.device)

            # QR update: append new rows and re-factorise.
            _, R_new     = torch.linalg.qr(torch.cat([self.R_final, X_change], dim=0), mode='r')
            R_inv        = torch.linalg.inv(R_new)
            self.R_final = R_new
            self.XTX_inv = R_inv @ R_inv.T
            self.XTy    += X_change.T @ y_change

            del X_change, y_change, R_new, R_inv


    def _update_tsqr_add_config_nonblock(self, config_id):
        """Non-block mode: add one config via Woodbury rank-k update.

        Woodbury identity: (A + X'X)^{-1} = A^{-1} - A^{-1}X'(I + XA^{-1}X')^{-1}XA^{-1}
        """
        if self.mpi_rank != 0:
            return

        with torch.no_grad(), gpu_memory_context():
            config_to_change = config_id if isinstance(config_id, int) else config_id.cpu().item()
            change_mask      = (self.config_idxs_train == config_to_change)

            X_change = self.X_train[change_mask].to(self.device)
            y_change = self.y_train[change_mask].to(self.device)

            left_update = self.XTX_inv @ X_change.T
            inner_term  = torch.eye(X_change.shape[0], device=self.device)
            inner_right = X_change @ left_update

            if self.ascending:
                inner         = torch.linalg.inv(inner_term + inner_right)
                self.XTX_inv -= left_update @ inner @ (X_change @ self.XTX_inv)
                self.XTy     += X_change.T @ y_change
            else:
                inner         = torch.linalg.inv(inner_term - inner_right)
                self.XTX_inv += left_update @ inner @ (X_change @ self.XTX_inv)
                self.XTy     -= X_change.T @ y_change

            del X_change, y_change, left_update, inner_term, inner_right, inner


    def _compute_block_cooks_single_rank(self, coeffs):
        """Block Cook's Distance: D_c = r'(I +/- H_c)^{-1} H_c r per config."""
        BATCH_SIZE = 5000
        num_groups = self.group_metadata.shape[0]

        best_cooks_val = -float('inf') if self.ascending else float('inf')
        best_config_id = -1

        active_ids = torch.unique(self.config_idxs_train[self.sub_mask_train])
        max_len    = self.group_metadata[:, 2].max().item()
        identity   = torch.eye(max_len, device=self.device, dtype=self.dtype)

        for batch_start in range(0, num_groups, BATCH_SIZE):
            with torch.no_grad(), gpu_memory_context():
                batch_end       = min(batch_start + BATCH_SIZE, num_groups)
                batch_meta      = self.group_metadata[batch_start:batch_end]
                curr_batch_size = batch_meta.shape[0]
                batch_max_len   = batch_meta[-1, 2].item()
                n_features      = self.X_train.shape[1]

                # build padded batch on CPU then transfer at once
                X_batch_cpu = torch.zeros((curr_batch_size, batch_max_len, n_features), dtype=self.dtype)
                y_batch_cpu = torch.zeros((curr_batch_size, batch_max_len, 1),          dtype=self.dtype)

                for b_i in range(curr_batch_size):
                    _, start, count = batch_meta[b_i]
                    start, count = int(start), int(count)
                    X_batch_cpu[b_i, :count] = self.X_train[start:start + count]
                    y_batch_cpu[b_i, :count] = self.y_train[start:start + count]

                X_batch = X_batch_cpu.to(self.device)
                y_batch = y_batch_cpu.to(self.device)
                del X_batch_cpu, y_batch_cpu

                # 1. Block leverage matrix: H_c = X_c XTX_inv X_c'
                temp     = torch.bmm(X_batch, self.XTX_inv.unsqueeze(0).expand(curr_batch_size, -1, -1))
                fake_lev = torch.bmm(temp, X_batch.transpose(1, 2))
                del temp

                # 2. Residuals: r = X_c beta - y_c
                res = torch.bmm(X_batch, coeffs.unsqueeze(0).expand(curr_batch_size, -1, -1)) - y_batch
                del X_batch, y_batch

                # 3. Cook's term: inv(I + H_c) for adding, inv(I - H_c) for removing
                I_k = identity[:batch_max_len, :batch_max_len].unsqueeze(0)
                mat_to_inv = fake_lev.clone().add_(I_k) if self.ascending else (I_k - fake_lev)
                del I_k

                inv_mat = torch.linalg.inv(mat_to_inv)
                del mat_to_inv

                term_right = torch.bmm(fake_lev, res)
                term_mid   = torch.bmm(inv_mat, term_right)
                cooks_vals = torch.bmm(res.transpose(1, 2), term_mid).squeeze(-1).squeeze(-1)
                del fake_lev, res, inv_mat, term_right, term_mid

                # mask already-selected configs
                batch_config_ids = batch_meta[:, 0]
                is_active        = torch.isin(batch_config_ids, active_ids)

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

                del cooks_vals, is_active

            if batch_end % 10000 == 0:
                print(f"[Rank 0]     {batch_end}/{num_groups} configs done", flush=True)

        del identity
        return best_config_id.cpu().item() if torch.is_tensor(best_config_id) else best_config_id


    def _compute_nonblock_cooks_single_rank(self, coeffs):
        """Non-block Cook's Distance on energy rows: D_i = e_i^2 h_i / (1 + h_i)."""
        with torch.no_grad(), gpu_memory_context():
            X_en = self.X_train[self.enrow_mask_train].to(self.device)
            y_en = self.y_train[self.enrow_mask_train].to(self.device)

            en_residuals   = X_en @ coeffs - y_en
            leverage_scores = torch.sum((X_en @ self.XTX_inv) * X_en, dim=1)
            e_cooks        = en_residuals.reshape(-1) ** 2 * leverage_scores / (1 + leverage_scores)

            del X_en, y_en, en_residuals, leverage_scores

            is_active = self.sub_mask_train.cpu()[self.enrow_mask_train.cpu()]

            if self.ascending:
                e_cooks[is_active] = -float('inf')
                best_idx = torch.argmax(e_cooks)
            else:
                e_cooks[~is_active] = float('inf')
                best_idx = torch.argmin(e_cooks)

            del e_cooks

            # Map energy-row index back to config ID 
            energy_config_ids = self.config_idxs_train[self.enrow_mask_train]
            return energy_config_ids[best_idx.cpu()].item()


    def _stepwise_tsqr_sampling(self):
        """Stepwise Cook's Distance selection via TSQR.

        Rank 0 runs the full greedy loop.
        Other ranks participate only in the initial parallel factorisation, then exit.
        """
        if self.sub_mask is None and self.mpi_rank == 0:
            self._create_initial_sub_mask()

        # Broadcast initial masks so non-zero ranks can contribute to TSQR
        if self.mpi_size > 1 and self.mode in ["parallel_tsqr", "hybrid_tsqr"]:
            self.sub_mask_train = self.comm.bcast(
                self.sub_mask_train if self.mpi_rank == 0 else None, root=0)
            self.sub_mask = self.comm.bcast(
                self.sub_mask       if self.mpi_rank == 0 else None, root=0)

        if self.mpi_rank == 0:
            mode_str = "Block" if self.block else "Non-Block"
            print(f"[Rank 0] TSQR: Initial factorisation ({mode_str})", flush=True)

        # Initial QR / TSQR factorisation
        if self.mode in ["parallel_tsqr", "hybrid_tsqr"]:
            R_final, XTy = self._compute_tsqr_qr(
                self.sub_mask_train if self.mpi_rank == 0 else None)

            if self.mpi_rank == 0 and R_final is not None:
                self.XTy     = XTy.to(self.device)
                self.R_final = R_final.to(self.device)
                # compute XTX_inv on CPU in float64 for numerical consistency across cases
                R_cpu        = R_final.cpu().double()
                R_inv        = torch.linalg.inv(R_cpu)
                self.XTX_inv = (R_inv @ R_inv.T).to(self.device).to(self.dtype)
                del R_cpu, R_inv
                print("[Rank 0] Parallel factorisation complete", flush=True)
        else:
            R_final, XTy = self._compute_tsqr_qr(self.sub_mask_train)

            self.XTy     = XTy.to(self.device)
            self.R_final = R_final.to(self.device)
            # compute XTX_inv on CPU in float64 for numerical consistency across cases
            R_cpu        = R_final.cpu().double()
            R_inv        = torch.linalg.inv(R_cpu)
            self.XTX_inv = (R_inv @ R_inv.T).to(self.device).to(self.dtype)
            del R_cpu, R_inv
            print("[Rank 0] Initial factorisation complete", flush=True)

        # non-zero ranks exit after the parallel factorisation phase.
        if self.mode in ["parallel_tsqr", "hybrid_tsqr"] and self.mpi_rank != 0:
            print(f"[Rank {self.mpi_rank}] Parallel phase complete, exiting", flush=True)
            return

        # stepwise selection runs on rank 0 only.
        if self.mpi_rank == 0:
            n_subsamples_init = int(torch.sum(self.sub_mask.cpu() & self.enrow_mask.cpu()))

            target_range = (range(n_subsamples_init, self.n_subsamples) if self.ascending
                            else range(n_subsamples_init, self.n_subsamples, -1))
            n_iterations = len(target_range)
            mode_str     = "Block" if self.block else "Non-Block"
            print(f"[Rank 0] Stepwise selection ({mode_str}), {n_iterations} iterations", flush=True)

            for iter_idx, _ in enumerate(target_range):
                with torch.no_grad():
                    coeffs = self.XTX_inv @ self.XTy

                    if self.block:
                        best_config_id = self._compute_block_cooks_single_rank(coeffs)
                    else:
                        best_config_id = self._compute_nonblock_cooks_single_rank(coeffs)

                    if self.ascending:
                        if self.block:
                            self._update_tsqr_add_config(best_config_id)
                        else:
                            self._update_tsqr_add_config_nonblock(best_config_id)
                        change_mask = (self.config_idxs_train == best_config_id)
                        self.sub_mask_train[change_mask] = True
                    else:
                        raise NotImplementedError("Descending not implemented")

                if (iter_idx + 1) % 100 == 0:
                    print(f"[Rank 0]   Iteration {iter_idx + 1}/{n_iterations}", flush=True)

                if (iter_idx + 1) % 1000 == 0:
                    torch.cuda.empty_cache()

            self.sub_mask = torch.isin(
                self.config_idxs.cpu(),
                self.config_idxs_train[self.sub_mask_train].cpu())

            print("[Rank 0] Stepwise selection complete", flush=True)

            # Move data back to device for downstream training.
            self.X_train           = self.X_train.to(self.device)
            self.y_train           = self.y_train.to(self.device)
            self.w_train           = self.w_train.to(self.device)
            self.config_idxs_train = self.config_idxs_train.to(self.device)
            self.enrow_mask_train  = self.enrow_mask_train.to(self.device)
            self.sub_mask_train    = self.sub_mask_train.to(self.device)

            print("[Rank 0] TSQR complete", flush=True)
