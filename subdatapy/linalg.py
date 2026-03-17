"""Reusable TSQR and distributed linear algebra for SubDataPy.

Pure functions, no mutable state. Mode detection is automatic based on
whether torch.distributed is initialized and n_chunks is set.

Execution modes for tsqr_r / tsqr_r_xty:
    n_chunks=None, world_size=1  -> Single-pass QR on GPU
    n_chunks>1,    world_size=1  -> Sequential TSQR (stream chunks through GPU)
    n_chunks=None, world_size>1  -> Parallel TSQR (1 chunk per rank)
    n_chunks>1,    world_size>1  -> Hybrid TSQR (n_chunks/world_size per rank)
"""

import torch
import torch.distributed as dist
from contextlib import contextmanager
from typing import Optional, Tuple


@contextmanager
def _cusolver_backend():
    """Temporarily switch to default (cusolver) backend for QR.

    Magma's float64 QR is inaccurate on some GPUs (e.g. A100).
    Cusolver/default gives correct results. Only affects CUDA tensors.
    """
    old = torch.backends.cuda.preferred_linalg_library()
    try:
        torch.backends.cuda.preferred_linalg_library('default')
        yield
    finally:
        torch.backends.cuda.preferred_linalg_library(old)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Return current process rank, or 0 if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return world size, or 1 if not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_distributed() -> bool:
    """True if torch.distributed is initialized with world_size > 1."""
    return get_world_size() > 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reduce_r_matrices(r_list, device):
    """Stack list of R matrices and QR-reduce: QR([R1; R2; ...]) -> R_combined.

    Args:
        r_list: list of (p, p) upper-triangular tensors
        device: device for the computation

    Returns:
        R: (p, p) upper-triangular on *device*
    """
    if len(r_list) == 1:
        return r_list[0].to(device)
    with torch.no_grad():
        R_stacked = torch.cat([r.to(device) for r in r_list], dim=0)
        with _cusolver_backend():
            _, R = torch.linalg.qr(R_stacked, mode='r')
    return R


# ---------------------------------------------------------------------------
# Core TSQR
# ---------------------------------------------------------------------------

def tsqr_r(X, *, device='cuda', n_chunks=None, tree_reduction_threshold=10):
    """Compute R factor of QR(X) for a tall-skinny matrix.

    Args:
        X: (n, p) tensor, can be on CPU or GPU.
        device: device for computation.
        n_chunks: Total number of chunks. None = auto.
        tree_reduction_threshold: Reduce accumulated R matrices after this many.

    Returns:
        R: (p, p) upper-triangular on *device*. Rank 0 only in distributed
           mode; other ranks get None.
    """
    R, _ = _tsqr_core(X, y=None, device=device, n_chunks=n_chunks,
                       tree_reduction_threshold=tree_reduction_threshold,
                       compute_xty=False)
    return R


def tsqr_r_xty(X, y, *, device='cuda', n_chunks=None, tree_reduction_threshold=10):
    """Compute R factor and X^T y simultaneously.

    Same modes as tsqr_r. X^T y is accumulated per-chunk and summed.
    In distributed mode, X^T y uses dist.reduce(op=SUM).

    Returns:
        (R, XTy) on rank 0; (None, None) on other ranks.
    """
    return _tsqr_core(X, y=y, device=device, n_chunks=n_chunks,
                      tree_reduction_threshold=tree_reduction_threshold,
                      compute_xty=True)


def _tsqr_core(X, *, y=None, device='cuda', n_chunks=None,
               tree_reduction_threshold=10, compute_xty=True):
    """Unified implementation for tsqr_r and tsqr_r_xty."""
    world_size = get_world_size()
    rank = get_rank()
    n_features = X.shape[1]
    dtype = X.dtype

    # --- Mode 1: single-pass (no chunks, no distribution) ---
    if n_chunks is None and world_size == 1:
        with torch.no_grad():
            X_dev = X.to(device)
            with _cusolver_backend():
                _, R = torch.linalg.qr(X_dev, mode='r')
            XTy = None
            if compute_xty and y is not None:
                y_dev = y.to(device)
                XTy = X_dev.T @ y_dev
            return R, XTy

    # --- Modes 2/3/4: chunked and/or distributed ---
    if n_chunks is None:
        # Mode 3: parallel, 1 chunk per rank
        n_chunks = world_size

    if world_size > 1:
        assert n_chunks % world_size == 0, \
            f"n_chunks ({n_chunks}) must be divisible by world_size ({world_size})"

    total_rows = X.shape[0]
    local_n_chunks = n_chunks // world_size
    chunk_size = (total_rows + n_chunks - 1) // n_chunks

    R_list = []
    XTy_accum = torch.zeros((n_features, 1), dtype=dtype)

    for local_i in range(local_n_chunks):
        global_i = rank * local_n_chunks + local_i
        start = global_i * chunk_size
        end = min(start + chunk_size, total_rows)
        if start >= total_rows:
            break

        with torch.no_grad():
            X_chunk = X[start:end].to(device)
            with _cusolver_backend():
                _, R_local = torch.linalg.qr(X_chunk, mode='r')
            R_list.append(R_local.cpu())

            if compute_xty and y is not None:
                y_chunk = y[start:end].to(device)
                XTy_accum += (X_chunk.T @ y_chunk).cpu()
                del y_chunk
            del X_chunk

        # Periodic tree reduction to bound memory
        if len(R_list) >= tree_reduction_threshold:
            R_list = [_reduce_r_matrices(R_list, device).cpu()]

    # Local reduction
    if len(R_list) == 0:
        R_local_final = torch.zeros((n_features, n_features), dtype=dtype, device=device)
    else:
        R_local_final = _reduce_r_matrices(R_list, device)

    # Distributed gather + global reduction
    if world_size > 1:
        if rank == 0:
            gather_list = [torch.empty_like(R_local_final) for _ in range(world_size)]
        else:
            gather_list = None
        dist.gather(R_local_final.contiguous(), gather_list, dst=0)

        XTy_dev = XTy_accum.to(device)
        dist.reduce(XTy_dev, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            R_final = _reduce_r_matrices(gather_list, device)
            return R_final, XTy_dev if compute_xty else None
        else:
            return None, None
    else:
        XTy_out = XTy_accum.to(device) if compute_xty else None
        return R_local_final, XTy_out


# ---------------------------------------------------------------------------
# Inverse / solve helpers
# ---------------------------------------------------------------------------

def xtx_inv_from_r(R, device='cuda'):
    """Compute (X^T X)^{-1} = R^{-1} R^{-T} from the R factor.

    Triangular solve on CPU in float64 for stability (avoids CUBLAS float64
    bugs and exploits triangular structure — faster than general inv for
    large p). Final matmul on device for speed when p is large.
    """
    R_cpu = R.cpu().double()
    I = torch.eye(R_cpu.shape[0], dtype=torch.float64)
    R_inv = torch.linalg.solve_triangular(R_cpu, I, upper=True)
    R_inv_dev = R_inv.to(device=device, dtype=R.dtype)
    XTX_inv = R_inv_dev @ R_inv_dev.T
    return XTX_inv


def xtx_inv_from_svd(X, device='cuda'):
    """Compute (X^T X)^{-1} via SVD. Returns (XTX_inv, U, S, Vh).

    Filters singular values below machine epsilon for numerical safety.
    """
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    tol = torch.finfo(S.dtype).eps * max(X.shape) * S[0]
    S_inv_sq = torch.where(S > tol, 1 / (S ** 2),
                           torch.tensor(0.0, device=device, dtype=X.dtype))
    XTX_inv = Vh.T @ torch.diag(S_inv_sq) @ Vh
    return XTX_inv, U, S, Vh


def solve_from_r_xty(R, XTy):
    """Solve for coefficients: beta = R^{-1} R^{-T} X^T y.

    Uses two triangular solves (more stable than explicit inverse).
    """
    z = torch.linalg.solve_triangular(R.T, XTy, upper=False)
    beta = torch.linalg.solve_triangular(R, z, upper=True)
    return beta


# ---------------------------------------------------------------------------
# Update methods
# ---------------------------------------------------------------------------

def woodbury_update(XTX_inv, X_change, y_change, XTy, ascending):
    """Rank-k Woodbury/Sherman-Morrison update for (X^T X)^{-1} and X^T y.

    When ascending=True (adding rows):
        (A + X'X)^{-1} = A^{-1} - A^{-1} X' (I + X A^{-1} X')^{-1} X A^{-1}
    When ascending=False (removing rows):
        (A - X'X)^{-1} = A^{-1} + A^{-1} X' (I - X A^{-1} X')^{-1} X A^{-1}

    Returns:
        (XTX_inv, XTy) — updated, same device as inputs
    """
    left = XTX_inv @ X_change.T                     # (p, k)
    I_k = torch.eye(X_change.shape[0], device=X_change.device, dtype=X_change.dtype)
    inner_right = X_change @ left                    # (k, k)

    if ascending:
        inner = torch.linalg.inv(I_k + inner_right)
        XTX_inv = XTX_inv - left @ inner @ (X_change @ XTX_inv)
        XTy = XTy + X_change.T @ y_change
    else:
        inner = torch.linalg.inv(I_k - inner_right)
        XTX_inv = XTX_inv + left @ inner @ (X_change @ XTX_inv)
        XTy = XTy - X_change.T @ y_change

    return XTX_inv, XTy


def qr_update_add(R, X_new, y_new, XTy, device='cuda'):
    """QR-based rank-k update: add rows and recompute R, (X^T X)^{-1}, X^T y.

    R_new = QR([R; X_new])[1]

    Returns:
        (R_new, XTX_inv_new, XTy_new) — all on device
    """
    with torch.no_grad():
        stacked = torch.cat([R.to(device), X_new.to(device)], dim=0)
        with _cusolver_backend():
            _, R_new = torch.linalg.qr(stacked, mode='r')
        XTX_inv = xtx_inv_from_r(R_new, device=device)
        XTy = XTy + X_new.T @ y_new
    return R_new, XTX_inv, XTy


# ---------------------------------------------------------------------------
# Leverage scores
# ---------------------------------------------------------------------------

def leverage_scores_from_qr(X, *, device='cuda'):
    """Single-pass QR leverage: retain Q, compute h_i = ||Q_i||^2.

    O(np) for leverage scores (after O(np^2) QR). More efficient and
    numerically stable than leverage_scores_from_r, but requires the full
    X to fit in GPU memory (no chunking support).

    Args:
        X: (n, p) matrix (moved to device for QR)
        device: device for computation

    Returns:
        h: (n,) leverage scores on device
    """
    X_dev = X.to(device)
    with torch.no_grad():
        with _cusolver_backend():
            Q, _ = torch.linalg.qr(X_dev, mode='reduced')
    h = torch.sum(Q ** 2, dim=1)
    return h


def leverage_scores_from_r(X, R, *, device='cuda', n_chunks=None):
    """Compute leverage scores h_i = ||R^{-T} x_i||^2.

    When n_chunks is set, streams X through GPU in chunks to avoid
    loading the full matrix. X can reside on CPU.

    Args:
        X: (n, p) matrix (can be on CPU)
        R: (p, p) upper-triangular R factor
        device: device for computation
        n_chunks: number of chunks (None = single pass, moves full X to device)

    Returns:
        h: (n,) leverage scores on device
    """
    R_dev = R.to(device)
    n = X.shape[0]

    if n_chunks is None or n_chunks <= 1:
        X_dev = X.to(device)
        B = torch.linalg.solve_triangular(R_dev.T, X_dev.T, upper=False).T
        return torch.sum(B ** 2, dim=1)

    # Chunked: stream X through GPU, never load full matrix
    chunk_size = (n + n_chunks - 1) // n_chunks
    h = torch.empty(n, device=device, dtype=X.dtype)
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            X_chunk = X[start:end].to(device)
            B_chunk = torch.linalg.solve_triangular(
                R_dev.T, X_chunk.T, upper=False).T
            h[start:end] = torch.sum(B_chunk ** 2, dim=1)
            del X_chunk, B_chunk
    return h
