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


def get_local_devices(device='cuda'):
    """Return list of local CUDA devices, or [device] if CUDA unavailable.

    Each visible CUDA device becomes an entry. When CUDA_VISIBLE_DEVICES is
    set by torchrun per local-rank, this returns only that rank's slice.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    return [device]


def solve_wls(A, B, *, method, device, n_chunks, partitioned, local_devices, dtype):
    """Weighted least-squares solve: coeffs = argmin ||A coeffs - B||^2.

    Exactly the pattern used by both ``BaseData.train`` and
    ``RandomSubSampler.train_subsample``: auto-switch ``'lstsq'`` to
    ``'qr'`` under torch.distributed (lstsq needs the full replicated
    matrix), run TSQR when ``method == 'qr'``, and broadcast the rank-0
    coeffs so every rank ends with the same answer.

    Returns the (p, 1) coefficients on ``device``.
    """
    if is_distributed() and method == 'lstsq':
        method = 'qr'

    if method == 'qr':
        R, XTy = tsqr_r_xty(A, B, device=device, n_chunks=n_chunks,
                            partitioned=partitioned, local_devices=local_devices)
        coeffs = solve_from_r_xty(R, XTy) if R is not None else None
        if is_distributed():
            coeffs = broadcast_tensor(
                coeffs if get_rank() == 0 else None,
                src=0, shape=(A.shape[1], 1), dtype=dtype, device=device)
        return coeffs

    if method == 'lstsq':
        return torch.linalg.lstsq(A, B).solution

    raise ValueError(f"Unknown method {method!r}; expected 'lstsq' or 'qr'.")


def distributed_rmse(local_sq, local_n, *, device):
    """RMSE from local sum-of-squares and local count with an all-reduce.

    Both ``random.compute_subsample_errors`` and ``data.compute_errors`` need
    this; keep the reduction in one place to avoid drift. In single-process
    mode the reduction is a no-op.

    Args:
        local_sq: Local sum of squared residuals (Python float or 0-d tensor).
        local_n:  Local count of residuals (Python int).
        device:   CUDA device on which to perform the reduction (required
                  for NCCL; ignored by gloo).

    Returns:
        A Python float RMSE, or ``None`` if the global count is zero.
    """
    local_sq = float(local_sq)
    local_n = int(local_n)
    if is_distributed():
        buf = torch.tensor([local_sq, float(local_n)], dtype=torch.float64,
                           device=device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        total_sq, total_n = buf[0].item(), buf[1].item()
    else:
        total_sq, total_n = local_sq, local_n
    if total_n == 0:
        return None
    return (total_sq / total_n) ** 0.5


def broadcast_tensor(
    tensor: Optional[torch.Tensor],
    src: int,
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Broadcast a tensor from src to all ranks. Receivers allocate a buffer.

    Args:
        tensor: Source tensor on src rank; ignored on other ranks.
        src: Source rank.
        shape, dtype, device: Keyword-only. Used to allocate the receive
            buffer on non-src ranks. Kept keyword-only so callers can't
            accidentally swap them — a real hazard with three similar args.

    Returns:
        Tensor on all ranks with the broadcast contents (contiguous, on
        `device`).
    """
    if get_rank() == src:
        buf = tensor.contiguous().to(device)
    else:
        buf = torch.empty(shape, dtype=dtype, device=device)
    dist.broadcast(buf, src=src)
    return buf


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

def tsqr_r(X, *, device='cuda', n_chunks=None, tree_reduction_threshold=10,
           partitioned=False, local_devices=None):
    """Compute R factor of QR(X) for a tall-skinny matrix.

    Args:
        X: (n, p) tensor, can be on CPU or GPU.
        device: primary device for computation (first local GPU).
        n_chunks: Number of chunks to split the rows into before QR'ing.
              * Single-process (world_size=1): total chunk count. None
                selects a single-pass QR when `n_local_gpus == 1`, or
                `n_local_gpus` chunks otherwise.
              * Partitioned multi-rank (partitioned=True, world_size>1):
                per-rank chunk count; the loop runs `n_chunks *
                len(local_devices)` local chunks (default 1 per local GPU
                when None).
        tree_reduction_threshold: Reduce accumulated R matrices after this
            many. Balances peak GPU memory (threshold * p^2 doubles) against
            the number of extra QR passes. Default 10.
        partitioned: If True, each rank holds only its own partition of X
            and the cross-rank gather/reduce runs. Required whenever
            world_size > 1. Partitioned with world_size==1 raises; so does
            world_size > 1 with partitioned=False (see _tsqr_core docstring
            for why replicated-distributed was removed).
        local_devices: Optional list of local CUDA devices to round-robin
            chunks across. None = single device (fall back to `device`).

    Returns:
        R: (p, p) upper-triangular on *device*. Rank 0 only when
           partitioned+distributed; other ranks get None.
    """
    R, _ = _tsqr_core(X, y=None, device=device, n_chunks=n_chunks,
                       tree_reduction_threshold=tree_reduction_threshold,
                       compute_xty=False, partitioned=partitioned,
                       local_devices=local_devices)
    return R


def tsqr_r_xty(X, y, *, device='cuda', n_chunks=None, tree_reduction_threshold=10,
               partitioned=False, local_devices=None):
    """Compute R factor and X^T y simultaneously.

    Same modes and `n_chunks` semantics as :func:`tsqr_r` (see its
    docstring for the per-rank vs global distinction). X^T y is
    accumulated per-chunk and summed; in distributed mode it uses
    dist.reduce(op=SUM).

    Returns:
        (R, XTy) on rank 0; (None, None) on other ranks.
    """
    return _tsqr_core(X, y=y, device=device, n_chunks=n_chunks,
                      tree_reduction_threshold=tree_reduction_threshold,
                      compute_xty=True, partitioned=partitioned,
                      local_devices=local_devices)


def _tsqr_core(X, *, y=None, device='cuda', n_chunks=None,
               tree_reduction_threshold=10, compute_xty=True,
               partitioned=False, local_devices=None):
    """Unified implementation for tsqr_r and tsqr_r_xty.

    Supported modes:
      * Single-process, n_chunks=None, n_local_gpus=1 → single-pass QR.
      * Single-process, n_chunks set or n_local_gpus>1 → sequential/
        round-robin chunked TSQR; `n_chunks` is the total chunk count
        (defaulting to n_local_gpus when None).
      * Distributed (world_size>1), partitioned=True → each rank QRs its
        local partition, rank 0 gathers and reduces. `n_chunks` is
        per-rank (multiplied by n_local_gpus to get local chunks).

    Replicated distributed mode (world_size>1 with partitioned=False) is
    not supported — partitioned is strictly more memory-efficient and
    uses unambiguous parameter semantics. Callers that hit that combo
    get a ValueError pointing at the fix.
    """
    world_size = get_world_size()
    rank = get_rank()
    n_features = X.shape[1]
    dtype = X.dtype

    if local_devices is None or len(local_devices) == 0:
        local_devices = [device]
    n_local_gpus = len(local_devices)

    if world_size > 1 and not partitioned:
        raise ValueError(
            "tsqr_r/tsqr_r_xty under torch.distributed (world_size>1) "
            "requires partitioned=True. Pass .npy file paths for X and "
            "config_idxs to BaseData so it auto-enables partitioned "
            "loading, or run without torch.distributed for replicated "
            "in-memory inputs.")
    if partitioned and world_size == 1:
        raise ValueError(
            "tsqr_r/tsqr_r_xty called with partitioned=True but "
            "world_size==1. Partitioned TSQR requires torch.distributed "
            "with world_size>1. For single-process use, pass "
            "partitioned=False (the default).")

    # --- Mode 1: single-pass (single process, single device, no chunks) ---
    if n_chunks is None and not partitioned and n_local_gpus == 1:
        with torch.no_grad():
            X_dev = X.to(device)
            with _cusolver_backend():
                _, R = torch.linalg.qr(X_dev, mode='r')
            XTy = None
            if compute_xty and y is not None:
                y_dev = y.to(device)
                XTy = X_dev.T @ y_dev
            return R, XTy

    # --- Chunked path (single-process-chunked OR partitioned multi-rank) ---
    local_rows = X.shape[0]
    if partitioned:
        # Per-rank chunks; defaults to one chunk per local GPU.
        per_rank_chunks = n_chunks if n_chunks is not None else 1
        total_local = per_rank_chunks * n_local_gpus
    else:
        # Single-process: n_chunks is the total chunk count.
        total_local = n_chunks if n_chunks is not None else n_local_gpus
    chunk_size = (local_rows + total_local - 1) // total_local if local_rows > 0 else 0

    R_list = []
    XTy_accum = torch.zeros((n_features, 1), dtype=dtype, device=device)

    for ci in range(total_local):
        start = ci * chunk_size
        end = min(start + chunk_size, local_rows)
        if start >= end:
            break
        gpu = local_devices[ci % n_local_gpus]
        with torch.no_grad():
            X_chunk = X[start:end].to(gpu)
            with _cusolver_backend():
                _, R_local = torch.linalg.qr(X_chunk, mode='r')
            R_list.append(R_local.to(device))
            if compute_xty and y is not None:
                y_chunk = y[start:end].to(gpu)
                XTy_accum += (X_chunk.T @ y_chunk).to(device)
                del y_chunk
            del X_chunk
        if len(R_list) >= tree_reduction_threshold:
            R_list = [_reduce_r_matrices(R_list, device)]

    if len(R_list) == 0:
        R_local_final = torch.zeros((n_features, n_features), dtype=dtype, device=device)
    else:
        R_local_final = _reduce_r_matrices(R_list, device)
        # Pad to (p, p) when a rank's local partition has fewer rows than
        # columns (only reachable in partitioned mode with very skinny
        # partitions — zero rows contribute zero to the stacked QR).
        if R_local_final.shape[0] < n_features:
            padded = torch.zeros((n_features, n_features), dtype=dtype, device=device)
            padded[:R_local_final.shape[0], :R_local_final.shape[1]] = R_local_final
            R_local_final = padded

    R_local_final = R_local_final.contiguous()

    # Single-process: no collectives needed.
    if not partitioned:
        XTy_out = XTy_accum if compute_xty else None
        return R_local_final, XTy_out

    # Partitioned multi-rank: gather R, reduce XTy. torch.linalg.qr on
    # CUDA returns column-major R; NCCL transfers raw bytes, so gather
    # buffers must be explicitly contiguous via torch.empty(shape, ...)
    # — never torch.empty_like(R).
    if rank == 0:
        gather_list = [torch.empty(R_local_final.shape, dtype=R_local_final.dtype,
                                   device=R_local_final.device)
                       for _ in range(world_size)]
    else:
        gather_list = None
    dist.gather(R_local_final, gather_list, dst=0)

    if compute_xty:
        dist.reduce(XTy_accum, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        R_final = _reduce_r_matrices(gather_list, device)
        return R_final, (XTy_accum if compute_xty else None)
    return None, None


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


def leverage_scores_from_r(X, R, *, device='cuda', n_chunks=None,
                           local_devices=None):
    """Compute leverage scores h_i = ||R^{-T} x_i||^2.

    When n_chunks is set, streams X through GPU in chunks to avoid
    loading the full matrix. X can reside on CPU.

    Args:
        X: (n, p) matrix (can be on CPU)
        R: (p, p) upper-triangular R factor
        device: primary device for output and single-GPU path
        n_chunks: number of chunks (None = single pass, moves full X to device)
        local_devices: optional list of local CUDA devices to round-robin
            chunks across (multi-GPU within a rank). None = single device.

    Returns:
        h: (n,) leverage scores on device
    """
    if local_devices is None or len(local_devices) == 0:
        local_devices = [device]
    n_local_gpus = len(local_devices)

    n = X.shape[0]

    if (n_chunks is None or n_chunks <= 1) and n_local_gpus == 1:
        R_dev = R.to(device)
        X_dev = X.to(device)
        B = torch.linalg.solve_triangular(R_dev.T, X_dev.T, upper=False).T
        return torch.sum(B ** 2, dim=1)

    # Chunked and/or multi-GPU: round-robin chunks across local GPUs
    per_rank_chunks = n_chunks if (n_chunks is not None and n_chunks > 1) else 1
    total_chunks = per_rank_chunks * n_local_gpus
    chunk_size = (n + total_chunks - 1) // total_chunks if n > 0 else 0
    # Pre-place R on each local GPU once
    R_by_gpu = {g: R.to(g) for g in local_devices}

    h = torch.empty(n, device=device, dtype=X.dtype)
    with torch.no_grad():
        for ci in range(total_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n)
            if start >= end:
                break
            gpu = local_devices[ci % n_local_gpus]
            X_chunk = X[start:end].to(gpu)
            B_chunk = torch.linalg.solve_triangular(
                R_by_gpu[gpu].T, X_chunk.T, upper=False).T
            h[start:end] = torch.sum(B_chunk ** 2, dim=1).to(device)
            del X_chunk, B_chunk
    return h


# ---------------------------------------------------------------------------
# Residuals (chunked, memory-safe)
# ---------------------------------------------------------------------------

def chunked_sq_residuals(X, y, coeffs, *, device='cuda', n_chunks=None,
                         local_devices=None):
    """Squared residuals (X @ coeffs - y)^2, streaming X through GPU in chunks.

    Mirrors the chunking used by tsqr / leverage so the full prediction
    never materializes on a single GPU. X and y may live on CPU; coeffs is
    moved to each local GPU. The squared-residual vector is returned on the
    SAME device as X (CPU in chunked/partitioned modes, GPU otherwise) so it
    aligns with the row masks the caller applies. Per-row residuals are
    independent, so chunking is bit-for-bit equivalent to a single matmul.

    Args:
        X: (n, p) design matrix (can be on CPU).
        y: (n,) or (n, 1) targets (same device as X).
        coeffs: (p, 1) coefficients.
        device: primary GPU for the single-device fast path.
        n_chunks: number of chunks. None/<=1 with one local GPU => single
            pass; otherwise n_chunks * len(local_devices) chunks.
        local_devices: optional list of local CUDA devices to round-robin.

    Returns:
        sq_res: (n, 1) tensor of squared residuals on X.device.
    """
    if local_devices is None or len(local_devices) == 0:
        local_devices = [device]
    n_local_gpus = len(local_devices)
    n = X.shape[0]
    out_device = X.device
    coeffs = coeffs.reshape(-1, 1)

    if n == 0:
        return torch.empty((0, 1), dtype=X.dtype, device=out_device)

    # Single-pass when there is nothing to chunk across.
    if (n_chunks is None or n_chunks <= 1) and n_local_gpus == 1:
        with torch.no_grad():
            r = X.to(device) @ coeffs.to(device) - y.to(device).reshape(-1, 1)
            return (r * r).to(out_device)

    per_chunks = n_chunks if (n_chunks is not None and n_chunks > 1) else 1
    total_chunks = per_chunks * n_local_gpus
    chunk_size = (n + total_chunks - 1) // total_chunks
    coeffs_by_gpu = {g: coeffs.to(g) for g in local_devices}

    out = torch.empty((n, 1), dtype=X.dtype, device=out_device)
    with torch.no_grad():
        for ci in range(total_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n)
            if start >= end:
                break
            gpu = local_devices[ci % n_local_gpus]
            X_chunk = X[start:end].to(gpu)
            y_chunk = y[start:end].to(gpu).reshape(-1, 1)
            r = X_chunk @ coeffs_by_gpu[gpu] - y_chunk
            out[start:end] = (r * r).to(out_device)
            del X_chunk, y_chunk, r
    return out
