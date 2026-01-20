import torch
import warnings
from mpi4py import MPI
from .random_utils import RandomSubSampler

class TSQRMPICookSubSampler(RandomSubSampler):

    def __init__(self, X, y, w=None, test_fraction=0.0, seed=None, test_mask=None,
                 config_idxs=None, enrow_mask=None, intercept=True,
                 device='cuda', block=False, stepwise=False, sampling=True, ascending=True,
                 initial_subsampler="random", initial_subsample_fraction=1,
                 n_ranks=None, chunk_size=None):

        # Initialize MPI communication environment
        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()  
        self.mpi_size = self.comm.Get_size()  

        # Rank 0 loads full dataset; other ranks initialize with placeholders
        # Data will be broadcast to all ranks during computation
        if self.mpi_rank == 0:
            super().__init__(X, y=y, w=w, test_fraction=test_fraction, seed=seed, 
                           test_mask=test_mask, config_idxs=config_idxs, 
                           enrow_mask=enrow_mask, intercept=intercept, device=device)
        else:
            super().__init__(torch.zeros((1, 1)), y=torch.zeros((1, 1)), device=device)

        # Execution mode selection based on MPI environment and user input
        if n_ranks is None:
            
            if self.mpi_size == 1:
                # case 1: Single process → standard QR factorization
                self.n_ranks = 1
                self.mode = "regular_qr"
            else:
                # case 3: Multiple processes → parallel TSQR 
                self.n_ranks = self.mpi_size
                self.mode = "parallel_tsqr"
        else:
            # User-specified number of chunks
            if self.mpi_size == 1:
                # case 2: single process with manual chunking → sequential TSQR
                self.n_ranks = n_ranks
                self.mode = "sequential_tsqr"
            else:
                # case 4: Multiple processes with chunking → hybrid 
                if n_ranks % self.mpi_size != 0:
                    raise ValueError(
                        f"n_ranks ({n_ranks}) must be multiple of MPI size ({self.mpi_size}). "
                        f"each rank processes n_ranks/mpi_size chunks."
                    )
                self.n_ranks = n_ranks
                self.mode = "hybrid_tsqr"

        # Chunks assigned to each MPI rank (1 for parallel, >1 for hybrid)
        self.local_n_chunks = self.n_ranks // self.mpi_size

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

        # Sort by size 
        # Batching similar-sized configs reduces memory allocation overhead
        size_sort = torch.argsort(self.group_metadata[:, 2])
        self.group_metadata = self.group_metadata[size_sort]


    def _compute_regular_qr(self, row_mask):
       
        selected_indices = torch.where(row_mask)[0]
        
        X_selected = self.X_train[selected_indices].to(self.device)
        y_selected = self.y_train[selected_indices].to(self.device)
        w_selected = self.w_train[selected_indices].to(self.device)
        
       
        X_weighted = w_selected.reshape(-1, 1) * X_selected
        y_weighted = w_selected * y_selected
        
        # standard QR factorization: X = QR
        Q, R = torch.linalg.qr(X_weighted, mode='reduced')
        XTy = X_weighted.T @ y_weighted
        
        # clean up GPU memory
        del X_selected, y_selected, w_selected, X_weighted, y_weighted, Q
        torch.cuda.empty_cache()
        
        return R, XTy


    def _compute_sequential_tsqr(self, row_mask):

        selected_indices = torch.where(row_mask)[0]
        n_selected = len(selected_indices)
        chunk_size = (n_selected + self.n_ranks - 1) // self.n_ranks
        
        R_matrices = []
        XTy_sum = torch.zeros((self.X_train.shape[1], 1), device=self.device, dtype=self.dtype)
        
        # Process chunks sequentially 
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
            
            # Free GPU memory after processing
            del X_chunk, y_chunk, w_chunk, X_chunk_w, y_chunk_w, Q_local, R_local
            torch.cuda.empty_cache()
        
        # Combine R factors via final QR factorization
        # Produce R_final equivalent to QR(X)
        R_stacked = torch.cat(R_matrices, dim=0)
        _, R_final = torch.linalg.qr(R_stacked, mode='reduced')
        
        return R_final, XTy_sum


    def _update_tsqr_add_config(self, config_id):
     
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
        _, R_new = torch.linalg.qr(combined, mode='reduced')

        # Update derived quantities
        self.R_final = R_new
        R_inv = torch.inverse(R_new)
        self.XTX_inv = R_inv @ R_inv.T  # (X^T X)^(-1) = R^(-1) R^(-T)
        self.XTy += X_new_w.T @ y_new_w

        del X_new, y_new, w_new, X_new_w, y_new_w, combined, R_inv
        torch.cuda.empty_cache()


    def _compute_block_cooks_single_rank(self, coeffs):
       
        BATCH_SIZE = 5000
        num_groups = self.group_metadata.shape[0]

        best_cooks_val = -float('inf') if self.ascending else float('inf')
        best_config_id = -1

        # Process configurations in batches 
        for i in range(0, num_groups, BATCH_SIZE):
            batch_meta = self.group_metadata[i:i+BATCH_SIZE]
            max_len = batch_meta[-1, 2].item()  
            curr_batch_size = batch_meta.shape[0]

            # Allocate padded batch tensors on GPU
            X_batch = torch.zeros((curr_batch_size, max_len, self.X_train.shape[1]), 
                                device=self.device, dtype=self.dtype)
            y_batch = torch.zeros((curr_batch_size, max_len, 1), 
                                device=self.device, dtype=self.dtype)

            # Load configurations into batch (with padding)
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
            # Sign depends on ascending (add) vs descending (remove) mode
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

            # Find configuration with extremum influence in this batch
            if self.ascending:
                cooks_vals[is_active] = -float('inf')  # Exclude selected
                curr_best = torch.max(cooks_vals)
                if curr_best > best_cooks_val:
                    best_cooks_val = curr_best
                    best_config_id = batch_config_ids[torch.argmax(cooks_vals)]
            else:
                cooks_vals[~is_active] = float('inf')  # Exclude unselected
                curr_best = torch.min(cooks_vals)
                if curr_best < best_cooks_val:
                    best_cooks_val = curr_best
                    best_config_id = batch_config_ids[torch.argmin(cooks_vals)]

            del X_batch, y_batch, fake_lev, res, inv_mat, term_right, term_mid, cooks_vals

        return best_config_id.cpu().item() if torch.is_tensor(best_config_id) else best_config_id. is this good and same