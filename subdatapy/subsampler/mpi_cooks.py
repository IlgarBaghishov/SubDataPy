import torch
import warnings
from mpi4py import MPI
from .random import RandomSubSampler


class TSQRMPICookSubSampler(RandomSubSampler):
    """
    TSQR-based Cook's Distance subsampler with MPI support.
    
    Four execution modes based on MPI environment and n_ranks:
        Case 1: n_ranks=None, mpi_size=1  → Regular QR (no TSQR)
        Case 2: n_ranks=N, mpi_size=1     → Sequential TSQR (N chunks)
        Case 3: n_ranks=None, mpi_size>1  → Parallel TSQR (one chunk per rank)
        Case 4: n_ranks=N*M, mpi_size=M   → Hybrid TSQR (N chunks per rank)
    """

    def __init__(self, X, y, w=None, test_fraction=0.0, seed=None, test_mask=None,
                 config_idxs=None, enrow_mask=None, intercept=True,
                 device='cuda', block=False, stepwise=False, sampling=True, ascending=True,
                 initial_subsampler="random", initial_subsample_fraction=1,
                 n_ranks=None):

        # Initialize MPI communication environment
        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()  # Process rank (0 to mpi_size-1)
        self.mpi_size = self.comm.Get_size()  # Total MPI processes

        # Rank 0 loads full dataset; other ranks initialize with placeholders
        if self.mpi_rank == 0:
            super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed, test_mask=test_mask,
                             config_idxs=config_idxs, enrow_mask=enrow_mask, intercept=intercept, device=device)
        else:
            super().__init__(torch.zeros((1, 1)), y=torch.zeros((1, 1)), device=device)

        # Execution mode selection based on MPI environment and user input
        if n_ranks is None:
            if self.mpi_size == 1:
                # Case 1: Single process → standard QR factorization
                self.n_ranks = 1
                self.mode = "regular_qr"
            else:
                # Case 3: Multiple processes → parallel TSQR (one chunk per rank)
                self.n_ranks = self.mpi_size
                self.mode = "parallel_tsqr"
        else:
            if self.mpi_size == 1:
                # Case 2: Single process with manual chunking → sequential TSQR
                self.n_ranks = n_ranks
                self.mode = "sequential_tsqr"
            else:
                # Case 4: Multiple processes with chunking → hybrid TSQR
                if n_ranks % self.mpi_size != 0:
                    raise ValueError(
                        f"n_ranks ({n_ranks}) must be multiple of MPI size ({self.mpi_size})"
                    )
                self.n_ranks = n_ranks
                self.mode = "hybrid_tsqr"

        # Chunks assigned to each MPI rank (1 for parallel, >1 for hybrid)
        self.local_n_chunks = self.n_ranks // self.mpi_size

        if self.mpi_rank == 0:
            print(f"Mode: {self.mode}, MPI size: {self.mpi_size}, n_ranks: {self.n_ranks}, "
                  f"local_chunks/rank: {self.local_n_chunks}", flush=True)

        # Configuration parameters
        self.block = block
        self.stepwise = stepwise
        self.sampling = sampling
        self.ascending = ascending
        self.initial_subsampler = initial_subsampler
        self.initial_subsample_fraction = initial_subsample_fraction

        # Maintain training data on CPU to prevent GPU memory overflow
        # Data is streamed to GPU in chunks during computation
        if self.mpi_rank == 0:
            self.X_train = self.X_train.to('cpu')
            self.y_train = self.y_train.to('cpu')
            self.w_train = self.w_train.to('cpu')
            self.config_idxs_train = self.config_idxs_train.to('cpu')
            self.enrow_mask_train = self.enrow_mask_train.to('cpu')

        # QR factorization results
        self.R_final = None      # R factor from QR decomposition
        self.XTX_inv = None      # (X^T X)^(-1) computed as R^(-1) R^(-T)
        self.XTy = None          # X^T y for least squares

        # Prepare metadata for block-wise Cook's Distance evaluation
        if self.block and self.mpi_rank == 0:
            self._prepare_block_metadata()


    def _prepare_block_metadata(self):
        """Sort and group configurations for efficient block Cook's Distance computation."""
        
        # Sort training data by configuration ID
        sort_perm = torch.argsort(self.config_idxs_train)

        self.X_train = self.X_train[sort_perm]
        self.y_train = self.y_train[sort_perm]
        self.w_train = self.w_train[sort_perm]
        self.enrow_mask_train = self.enrow_mask_train[sort_perm]
        self.config_idxs_train = self.config_idxs_train[sort_perm]

        # Extract configuration boundaries: [config_id, start_index, count]
        unique_vals, counts = torch.unique_consecutive(self.config_idxs_train, return_counts=True)
        end_indices = torch.cumsum(counts, dim=0)
        start_indices = end_indices - counts

        self.group_metadata = torch.stack((unique_vals, start_indices, counts), dim=1)

        # Sort by configuration size (number of atoms)
        # Batching similar-sized configs reduces memory allocation overhead
        size_sort = torch.argsort(self.group_metadata[:, 2])
        self.group_metadata = self.group_metadata[size_sort]


    def _create_sub_mask(self):
        if self.stepwise:
            self._stepwise_tsqr_sampling()
        else:
            raise NotImplementedError("Only stepwise TSQR implemented")


    def _create_initial_sub_mask(self):
        """Create initial subset via random sampling (only rank 0)."""
        if self.mpi_rank != 0:
            return
            
        if self.initial_subsampler == "random":
            unique_cpu = self.unique_config_idxs_train.to('cpu')
            n_init = int(len(unique_cpu) * self.initial_subsample_fraction)
            perm = torch.randperm(len(unique_cpu))
            init_configs = unique_cpu[perm[:n_init]]
            self.sub_mask_train = torch.isin(self.config_idxs_train, init_configs)
        else:
            raise NotImplementedError("Only random initial subsampler for TSQR")

        self.sub_mask = torch.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train])


    def _compute_tsqr_qr(self, row_mask):
        """Route to appropriate QR method based on execution mode."""
        if self.mode == "regular_qr":
            return self._compute_regular_qr(row_mask)
        elif self.mode == "sequential_tsqr":
            return self._compute_sequential_tsqr(row_mask)
        elif self.mode == "parallel_tsqr":
            return self._compute_parallel_tsqr(row_mask)
        elif self.mode == "hybrid_tsqr":
            return self._compute_hybrid_tsqr(row_mask)


    def _compute_regular_qr(self, row_mask):
        """Case 1: Standard QR factorization without TSQR."""
        if self.mpi_rank == 0:
            print("Case 1: Computing Regular QR (no chunking)", flush=True)
        
        selected_indices = torch.where(row_mask)[0]
        n_selected = len(selected_indices)
        
        if self.mpi_rank == 0:
            print(f"Processing {n_selected:,} rows with regular QR", flush=True)
        
        X_selected = self.X_train[selected_indices].to(self.device)
        y_selected = self.y_train[selected_indices].to(self.device)
        w_selected = self.w_train[selected_indices].to(self.device)
        
        # Apply sample weights
        X_weighted = w_selected.reshape(-1, 1) * X_selected
        y_weighted = w_selected * y_selected
        
        # Single QR decomposition: X = QR
        Q, R = torch.linalg.qr(X_weighted, mode='reduced')
        XTy = X_weighted.T @ y_weighted
        
        del X_selected, y_selected, w_selected, X_weighted, y_weighted, Q
        torch.cuda.empty_cache()
        
        return R, XTy


    def _compute_sequential_tsqr(self, row_mask):
        """Case 2: Sequential TSQR with n_ranks chunks on single GPU."""
        selected_indices = torch.where(row_mask)[0]
        n_selected = len(selected_indices)
        
        # Divide into exactly n_ranks chunks
        chunk_size = (n_selected + self.n_ranks - 1) // self.n_ranks
        
        if self.mpi_rank == 0:
            print(f"Case 2: Sequential TSQR - {n_selected:,} rows, {self.n_ranks} chunks, "
                  f"~{chunk_size} rows/chunk", flush=True)
        
        R_matrices = []
        XTy_sum = torch.zeros((self.X_train.shape[1], 1),
                              device=self.device, dtype=self.dtype)
        
        # Process chunks sequentially: load → QR → discard
        for chunk_idx in range(self.n_ranks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_selected)
            
            if start_idx >= n_selected:
                break
                
            chunk_indices = selected_indices[start_idx:end_idx]
            
            X_chunk = self.X_train[chunk_indices].to(self.device)
            y_chunk = self.y_train[chunk_indices].to(self.device)
            w_chunk = self.w_train[chunk_indices].to(self.device)
            
            X_chunk_w = w_chunk.reshape(-1, 1) * X_chunk
            y_chunk_w = w_chunk * y_chunk
            
            # Local QR decomposition for this chunk
            Q_local, R_local = torch.linalg.qr(X_chunk_w, mode='reduced')
            R_matrices.append(R_local.clone())
            XTy_sum += X_chunk_w.T @ y_chunk_w
            
            del X_chunk, y_chunk, w_chunk, X_chunk_w, y_chunk_w, Q_local, R_local
            torch.cuda.empty_cache()
            
            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed chunk {chunk_idx + 1}/{self.n_ranks}", flush=True)
        
        # Combine R factors via final QR: [R0; R1; ...] → R_final
        R_stacked = torch.cat(R_matrices, dim=0)
        Q_final, R_final = torch.linalg.qr(R_stacked, mode='reduced')
        
        del R_matrices, R_stacked, Q_final
        torch.cuda.empty_cache()
        
        return R_final, XTy_sum


    def _compute_parallel_tsqr(self, row_mask):
        """Case 3: Parallel TSQR where each rank processes one chunk."""
        # Broadcast metadata from rank 0 to all ranks
        if self.mpi_rank == 0:
            selected_indices = torch.where(row_mask)[0]
            n_selected = len(selected_indices)
        else:
            selected_indices = None
            n_selected = None
        
        n_selected = self.comm.bcast(n_selected, root=0)
        selected_indices = self.comm.bcast(selected_indices, root=0)
        
        chunk_size = (n_selected + self.n_ranks - 1) // self.n_ranks
        
        if self.mpi_rank == 0:
            print(f"Case 3: Parallel TSQR - {n_selected:,} rows, {self.n_ranks} MPI ranks", flush=True)
        
        # Each rank identifies its chunk based on rank ID
        chunk_idx = self.mpi_rank
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_selected)
        
        if start_idx < n_selected:
            chunk_indices = selected_indices[start_idx:end_idx]
            
            # Data distribution: rank 0 sends chunks to other ranks
            if self.mpi_rank == 0:
                X_chunk = self.X_train[chunk_indices].to(self.device)
                y_chunk = self.y_train[chunk_indices].to(self.device)
                w_chunk = self.w_train[chunk_indices].to(self.device)
            else:
                # Receive data from rank 0
                X_chunk = self.comm.recv(source=0, tag=self.mpi_rank*3)
                y_chunk = self.comm.recv(source=0, tag=self.mpi_rank*3+1)
                w_chunk = self.comm.recv(source=0, tag=self.mpi_rank*3+2)
                
                X_chunk = X_chunk.to(self.device)
                y_chunk = y_chunk.to(self.device)
                w_chunk = w_chunk.to(self.device)
            
            # Rank 0 sends data to other ranks
            if self.mpi_rank == 0:
                for other_rank in range(1, self.mpi_size):
                    other_start = other_rank * chunk_size
                    other_end = min(other_start + chunk_size, n_selected)
                    if other_start < n_selected:
                        other_indices = selected_indices[other_start:other_end]
                        self.comm.send(self.X_train[other_indices].cpu(), dest=other_rank, tag=other_rank*3)
                        self.comm.send(self.y_train[other_indices].cpu(), dest=other_rank, tag=other_rank*3+1)
                        self.comm.send(self.w_train[other_indices].cpu(), dest=other_rank, tag=other_rank*3+2)
            
            X_chunk_w = w_chunk.reshape(-1, 1) * X_chunk
            y_chunk_w = w_chunk * y_chunk
            
            # Local QR on each rank (parallel execution)
            Q_local, R_local = torch.linalg.qr(X_chunk_w, mode='reduced')
            XTy_local = X_chunk_w.T @ y_chunk_w
            
            # Move to CPU for MPI communication
            R_local_cpu = R_local.cpu()
            XTy_local_cpu = XTy_local.cpu()
            
            del X_chunk, y_chunk, w_chunk, X_chunk_w, y_chunk_w, Q_local, R_local, XTy_local
            torch.cuda.empty_cache()
        else:
            R_local_cpu = None
            XTy_local_cpu = None
        
        # Gather all R matrices to rank 0
        all_R = self.comm.gather(R_local_cpu, root=0)
        all_XTy = self.comm.gather(XTy_local_cpu, root=0)
        
        if self.mpi_rank == 0:
            all_R = [r for r in all_R if r is not None]
            all_XTy = [xty for xty in all_XTy if xty is not None]
            
            # Final QR combination on rank 0
            R_stacked = torch.cat([r.to(self.device) for r in all_R], dim=0)
            Q_final, R_final = torch.linalg.qr(R_stacked, mode='reduced')
            
            XTy_sum = sum([xty.to(self.device) for xty in all_XTy])
            
            del all_R, all_XTy, R_stacked, Q_final
            torch.cuda.empty_cache()
            
            return R_final, XTy_sum
        else:
            return None, None


    def _compute_hybrid_tsqr(self, row_mask):
        """Case 4: Hybrid TSQR - each rank processes multiple chunks sequentially."""
        # Broadcast metadata
        if self.mpi_rank == 0:
            selected_indices = torch.where(row_mask)[0]
            n_selected = len(selected_indices)
            n_features = self.X_train.shape[1]
        else:
            selected_indices = None
            n_selected = None
            n_features = None
        
        n_selected = self.comm.bcast(n_selected, root=0)
        selected_indices = self.comm.bcast(selected_indices, root=0)
        n_features = self.comm.bcast(n_features, root=0)
        
        chunk_size = (n_selected + self.n_ranks - 1) // self.n_ranks
        
        if self.mpi_rank == 0:
            print(f"Case 4: Hybrid TSQR - {n_selected:,} rows, {self.n_ranks} total chunks, "
                  f"{self.local_n_chunks} chunks/rank", flush=True)
        
        # Each rank processes local_n_chunks sequentially
        R_matrices_local = []
        XTy_local = torch.zeros((n_features, 1), device=self.device, dtype=self.dtype)
        
        for local_chunk_idx in range(self.local_n_chunks):
            # Calculate global chunk index for this rank
            global_chunk_idx = self.mpi_rank * self.local_n_chunks + local_chunk_idx
            start_idx = global_chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_selected)
            
            if start_idx >= n_selected:
                break
            
            chunk_indices = selected_indices[start_idx:end_idx]
            
            # Data distribution synchronized across all ranks
            if self.mpi_rank == 0:
                # Rank 0: process own chunk and send to others
                X_chunk = self.X_train[chunk_indices].to(self.device)
                y_chunk = self.y_train[chunk_indices].to(self.device)
                w_chunk = self.w_train[chunk_indices].to(self.device)
                
                # Send chunks to other ranks for this iteration
                for other_rank in range(1, self.mpi_size):
                    other_global_idx = other_rank * self.local_n_chunks + local_chunk_idx
                    other_start = other_global_idx * chunk_size
                    other_end = min(other_start + chunk_size, n_selected)
                    if other_start < n_selected:
                        other_indices = selected_indices[other_start:other_end]
                        tag_base = other_rank * 1000 + local_chunk_idx * 3
                        self.comm.send(self.X_train[other_indices].cpu(), dest=other_rank, tag=tag_base)
                        self.comm.send(self.y_train[other_indices].cpu(), dest=other_rank, tag=tag_base+1)
                        self.comm.send(self.w_train[other_indices].cpu(), dest=other_rank, tag=tag_base+2)
            else:
                # Other ranks: receive chunk
                tag_base = self.mpi_rank * 1000 + local_chunk_idx * 3
                X_chunk = self.comm.recv(source=0, tag=tag_base).to(self.device)
                y_chunk = self.comm.recv(source=0, tag=tag_base+1).to(self.device)
                w_chunk = self.comm.recv(source=0, tag=tag_base+2).to(self.device)
            
            X_chunk_w = w_chunk.reshape(-1, 1) * X_chunk
            y_chunk_w = w_chunk * y_chunk
            
            Q_local, R_local = torch.linalg.qr(X_chunk_w, mode='reduced')
            R_matrices_local.append(R_local.clone())
            XTy_local += X_chunk_w.T @ y_chunk_w
            
            del X_chunk, y_chunk, w_chunk, X_chunk_w, y_chunk_w, Q_local, R_local
            torch.cuda.empty_cache()
        
        # Combine local R matrices on each rank
        if len(R_matrices_local) > 0:
            R_local_combined = torch.cat(R_matrices_local, dim=0)
            Q_local, R_local_final = torch.linalg.qr(R_local_combined, mode='reduced')
            
            R_local_final_cpu = R_local_final.cpu()
            XTy_local_cpu = XTy_local.cpu()
            
            del R_matrices_local, R_local_combined, Q_local, R_local_final
            torch.cuda.empty_cache()
        else:
            R_local_final_cpu = None
            XTy_local_cpu = None
        
        # Gather from all ranks
        all_R = self.comm.gather(R_local_final_cpu, root=0)
        all_XTy = self.comm.gather(XTy_local_cpu, root=0)
        
        if self.mpi_rank == 0:
            all_R = [r for r in all_R if r is not None]
            all_XTy = [xty for xty in all_XTy if xty is not None]
            
            # Final combination on rank 0
            R_stacked = torch.cat([r.to(self.device) for r in all_R], dim=0)
            Q_final, R_final = torch.linalg.qr(R_stacked, mode='reduced')
            
            XTy_sum = sum([xty.to(self.device) for xty in all_XTy])
            
            del all_R, all_XTy, R_stacked, Q_final
            torch.cuda.empty_cache()
            
            return R_final, XTy_sum
        else:
            return None, None

    def _update_tsqr_add_config(self, config_id):
        """Update R and XTX_inv via rank-k update (Woodbury identity)."""
        if self.mpi_rank != 0:
            return
            
        config_id_cpu = config_id if isinstance(config_id, int) else config_id.cpu().item()
        
        config_mask = (self.config_idxs_train == config_id_cpu)

        X_new = self.X_train[config_mask].to(self.device)
        y_new = self.y_train[config_mask].to(self.device)
        w_new = self.w_train[config_mask].to(self.device)

        X_new_w = w_new.reshape(-1, 1) * X_new
        y_new_w = w_new * y_new

        # Rank-k update: QR of [R_old; X_new]
        combined = torch.cat([self.R_final, X_new_w], dim=0)
        Q_update, R_new = torch.linalg.qr(combined, mode='reduced')

        # Update derived quantities
        self.R_final = R_new
        R_inv = torch.inverse(R_new)
        self.XTX_inv = R_inv @ R_inv.T  # (X^T X)^(-1) = R^(-1) R^(-T)
        self.XTy += X_new_w.T @ y_new_w

        del X_new, y_new, w_new, X_new_w, y_new_w, combined, Q_update, R_inv
        torch.cuda.empty_cache()


    def _compute_block_cooks_single_rank(self, coeffs):
        """Compute block Cook's Distance for all candidate configurations."""
        BATCH_SIZE = 5000
        num_groups = self.group_metadata.shape[0]

        best_cooks_val = -float('inf') if self.ascending else float('inf')
        best_config_id = -1

        # Process configurations in batches to manage GPU memory
        for i in range(0, num_groups, BATCH_SIZE):
            batch_meta = self.group_metadata[i:i+BATCH_SIZE]

            max_len = batch_meta[-1, 2].item()  # Maximum atoms in batch
            curr_batch_size = batch_meta.shape[0]

            # Allocate padded batch tensors
            X_batch = torch.zeros((curr_batch_size, max_len, self.X_train.shape[1]),
                                 device=self.device, dtype=self.dtype)
            y_batch = torch.zeros((curr_batch_size, max_len, 1),
                                 device=self.device, dtype=self.dtype)

            # Load configurations into batch (with zero-padding)
            for b_i in range(curr_batch_size):
                idx, start, count = batch_meta[b_i]
                start, count = int(start), int(count)
                X_batch[b_i, :count, :] = self.X_train[start:start+count].to(self.device)
                y_batch[b_i, :count, :] = self.y_train[start:start+count].to(self.device)

            # Compute block leverage matrix: H_m = X_m (X^T X)^(-1) X_m^T
            temp = torch.bmm(X_batch, self.XTX_inv.unsqueeze(0).expand(curr_batch_size, -1, -1))
            fake_lev = torch.bmm(temp, X_batch.transpose(1, 2))

            # Compute residuals: e_m = y_m - X_m β
            res = torch.bmm(X_batch, coeffs.unsqueeze(0).expand(curr_batch_size, -1, -1)) - y_batch

            # Block Cook's Distance: D_m = e_m^T (I ± H_m)^(-1) H_m e_m
            identity = torch.eye(max_len, device=self.device).unsqueeze(0)
            mat_to_inv = identity + fake_lev if self.ascending else identity - fake_lev
            inv_mat = torch.linalg.inv(mat_to_inv)

            term_right = torch.bmm(fake_lev, res)
            term_mid = torch.bmm(inv_mat, term_right)
            cooks_vals = torch.bmm(res.transpose(1, 2), term_mid).squeeze()

            # Mask out already-selected configurations
            batch_config_ids = batch_meta[:, 0].to(self.device)
            active_ids = torch.unique(self.config_idxs_train[self.sub_mask_train]).to(self.device)
            is_active = torch.isin(batch_config_ids, active_ids)

            # Find configuration with extremum Cook's Distance
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

            del X_batch, y_batch, fake_lev, res, inv_mat, term_right, term_mid, cooks_vals

        return best_config_id.cpu().item() if torch.is_tensor(best_config_id) else best_config_id


    def _stepwise_tsqr_sampling(self):
        """Stepwise selection via TSQR and Additive Cook's Distance."""
        # Create initial subset (rank 0 only)
        if self.sub_mask is None and self.mpi_rank == 0:
            self._create_initial_sub_mask()

        # Broadcast initial mask to all ranks (parallel modes)
        if self.mpi_size > 1 and self.mode in ["parallel_tsqr", "hybrid_tsqr"]:
            self.sub_mask_train = self.comm.bcast(self.sub_mask_train if self.mpi_rank == 0 else None, root=0)
            self.sub_mask = self.comm.bcast(self.sub_mask if self.mpi_rank == 0 else None, root=0)

        if self.mpi_rank == 0:
            print("TSQR: Initial factorization", flush=True)
        
        # Initial QR/TSQR factorization
        if self.mode in ["parallel_tsqr", "hybrid_tsqr"]:
            # Parallel modes: all ranks participate
            R_final, XTy = self._compute_tsqr_qr(self.sub_mask_train if self.mpi_rank == 0 else None)
            
            if self.mpi_rank == 0:
                if R_final is not None:
                    self.R_final = R_final.to(self.device)
                    self.XTy = XTy.to(self.device)
                    
                    R_inv = torch.inverse(self.R_final)
                    self.XTX_inv = R_inv @ R_inv.T
                    del R_inv
                    
                    print("Parallel QR complete. Rank 0 continuing with stepwise selection...", flush=True)
        else:
            # Sequential modes: rank 0 only
            R_final, XTy = self._compute_tsqr_qr(self.sub_mask_train)
            self.R_final = R_final.to(self.device)
            self.XTy = XTy.to(self.device)
            
            R_inv = torch.inverse(self.R_final)
            self.XTX_inv = R_inv @ R_inv.T
            del R_inv

        # Ranks 1+ exit for parallel modes
        if self.mode in ["parallel_tsqr", "hybrid_tsqr"] and self.mpi_rank != 0:
            print(f"Rank {self.mpi_rank}: Parallel QR complete, exiting.", flush=True)
            return

        # Stepwise selection (rank 0 only)
        if self.mpi_rank == 0:
            n_subsamples_init = int(torch.sum(self.sub_mask & self.enrow_mask))
            target_range = range(n_subsamples_init, self.n_subsamples) if self.ascending \
                           else range(n_subsamples_init, self.n_subsamples, -1)

            print(f"TSQR: Stepwise selection, {len(target_range)} iterations", flush=True)

            for iter_idx, _ in enumerate(target_range):
                coeffs = self.XTX_inv @ self.XTy

                if self.block:
                    best_config_id = self._compute_block_cooks_single_rank(coeffs)
                else:
                    raise NotImplementedError("Non-block not implemented")

                if self.ascending:
                    self._update_tsqr_add_config(best_config_id)
                    config_mask = (self.config_idxs_train == best_config_id)
                    self.sub_mask_train[config_mask] = True

                if iter_idx % 100 == 0:
                    print(f"Iteration {iter_idx}/{len(target_range)}", flush=True)

            self.sub_mask = torch.isin(self.config_idxs, self.config_idxs_train[self.sub_mask_train])

            print("TSQR: Moving data to CUDA for training", flush=True)
            self.X_train = self.X_train.to(self.device)
            self.y_train = self.y_train.to(self.device)
            self.w_train = self.w_train.to(self.device)
            self.config_idxs_train = self.config_idxs_train.to(self.device)
            self.enrow_mask_train = self.enrow_mask_train.to(self.device)
            self.sub_mask_train = self.sub_mask_train.to(self.device)

            print("TSQR: Complete", flush=True)