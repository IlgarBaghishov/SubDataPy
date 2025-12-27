import torch
import numpy as np
import pandas as pd
import warnings


def process_data(x, dtype=torch.float64, device='cpu'):
    """
    Loads data into CPU memory (Pinned if possible for fast transfer).
    """
    if x is None:
        return None
    
    if isinstance(x, str):
        if x.endswith('.npy'):
            x = np.load(x)
        else:
            raise ValueError('File format not supported')

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, pd.DataFrame):
        x = torch.from_numpy(x.values)
    
    if not isinstance(x, torch.Tensor):
        raise ValueError('Input type not supported')

    # Keep on CPU by default to save VRAM
    if x.dtype != dtype or x.device!=device:
        x = x.to(dtype=dtype, device=device)
    return x


class BaseData:

    def __init__(self, X, y=None, w=None, config_idxs=None, enrow_mask=None, intercept=True, device='cuda'):
        self.device = device
        self.dtype = torch.float64 
        self.coeffs = None

        self.X = process_data(X, self.dtype, 'cpu')

        if intercept:
            ones = torch.ones((self.X.shape[0], 1), dtype=self.dtype).to(device='cpu')
            self.X = torch.hstack((ones, self.X))

        self.y = process_data(y, self.dtype, 'cpu')
        if self.y is not None: self.y = self.y.reshape(-1, 1)

        if w is None:
            self.w = torch.ones(self.X.shape[0], dtype=self.dtype).to(device='cpu')
        else:
            self.w = process_data(w, self.dtype, 'cpu')
        if self.w is not None: self.w = self.w.reshape(-1, 1)

        if enrow_mask is not None:
            self.enrow_mask = process_data(enrow_mask, torch.bool, 'cpu')
        else:
            self.enrow_mask = None

        if config_idxs is None:
            warnings.warn("config_idxs is None. No grouping.", UserWarning)
            self.config_idxs = torch.arange(self.X.shape[0]).to(dtype=torch.int64, device='cpu')
            self.unique_config_idxs = self.config_idxs
            if self.enrow_mask is None:
                self.enrow_mask = torch.ones_like(self.config_idxs, dtype=torch.bool).to('cpu')
        else:
            self.config_idxs = process_data(config_idxs, torch.int64, 'cpu')
            self.unique_config_idxs, inverse_indices = torch.unique(self.config_idxs, return_inverse=True)
            if self.enrow_mask is None:
                warnings.warn("enrow_mask is None. Using first element as energy row.", UserWarning)
                perm = torch.arange(inverse_indices.size(0))
                unique_first_indices = torch.empty(self.unique_config_idxs.size(0), dtype=torch.int64)
                unique_first_indices.scatter_reduce_(0, inverse_indices, perm, reduce="amin", include_self=False)
                
                self.enrow_mask = torch.zeros_like(self.config_idxs, dtype=torch.bool).to('cpu')
                self.enrow_mask[unique_first_indices] = True


    def train_test_split(self, test_fraction=0.0, seed=None, test_mask=None):
        self.test_fraction = test_fraction
        self.seed = seed

        if test_mask is None:
            if self.test_fraction == 0.0:
                self.test_mask = torch.zeros(self.config_idxs.shape, dtype=torch.bool)
            elif 0.0 < self.test_fraction < 1.0:
                if self.seed is not None:
                    torch.manual_seed(self.seed)

                num_test = int(len(self.unique_config_idxs) * self.test_fraction)
                perm = torch.randperm(len(self.unique_config_idxs))
                test_indices = self.unique_config_idxs[perm[:num_test]]

                self.test_mask = torch.isin(self.config_idxs, test_indices)
            else:
                raise ValueError("test_fraction must be between 0 and 1")
        else:
            self.test_mask = process_data(test_mask, torch.bool, 'cpu')
            
        self.train_mask = ~self.test_mask

        # IMPORTANT: Only move TRAINING and TESTING sets to GPU
        self.X_train = self.X[self.train_mask].to(self.device)
        self.y_train = self.y[self.train_mask].to(self.device) if self.y is not None else None
        self.w_train = self.w[self.train_mask].to(self.device)
        self.enrow_mask_train = self.enrow_mask[self.train_mask].to(self.device) if self.enrow_mask is not None else None

        self.X_test = self.X[self.test_mask].to(self.device)
        self.y_test = self.y[self.test_mask].to(self.device) if self.y is not None else None
        self.w_test = self.w[self.test_mask].to(self.device)
        self.enrow_mask_test = self.enrow_mask[self.test_mask].to(self.device) if self.enrow_mask is not None else None

        self.config_idxs_train = self.config_idxs[self.train_mask].to(device=self.device)
        self.config_idxs_test = self.config_idxs[self.test_mask].to(device=self.device)
        self.unique_config_idxs_train = torch.unique(self.config_idxs_train).to(device=self.device)
        self.unique_config_idxs_test = torch.unique(self.config_idxs_test).to(device=self.device)


    def train(self):
        # Weighted Least Squares on GPU
        # Solve: (X^T W X) c = X^T W y
        # Or: min || W^0.5 (X c - y) ||
        
        # Apply weights efficiently without blowing up memory
        # We modify a clone of X_train temporarily
        
        # 1. Prepare A and B
        # NOTE: self.X_train is already on self.device (GPU)
        
        # In-place multiplication to save memory
        A = self.X_train.clone()
        A.mul_(self.w_train) 
        
        B = self.y_train.clone()
        B.mul_(self.w_train)
        B = B.reshape(-1, 1)

        # 2. Solve
        # result = torch.linalg.lstsq(A, B)
        # self.coeffs = result.solution
        self.coeffs = torch.linalg.pinv(A) @ B
        
        # Cleanup
        del A
        del B


    def compute_errors(self, verbose=True):
        if self.coeffs is None:
            raise ValueError("Model not trained.")
        
        def get_rmse(X_data, y_true, mask, name):
            if X_data.shape[0] == 0: return None

            mask_gpu = mask.to(self.device)
            if mask_gpu.sum() == 0: return None

            y_preds = X_data[mask_gpu] @ self.coeffs
            
            sq_res = torch.square(y_preds - y_true[mask_gpu])
            rmse = torch.sqrt(torch.mean(sq_res)).item()
            if verbose: print(f"{name} RMSE is {rmse}")
            return rmse

        e_train = get_rmse(self.X_train, self.y_train, self.enrow_mask_train, "Energy training")
        f_train = get_rmse(self.X_train, self.y_train, ~self.enrow_mask_train, "Force training")
        
        e_test = get_rmse(self.X_test, self.y_test, self.enrow_mask_test, "Energy test")
        f_test = get_rmse(self.X_test, self.y_test, ~self.enrow_mask_test, "Force test")

        return e_train, f_train, e_test, f_test