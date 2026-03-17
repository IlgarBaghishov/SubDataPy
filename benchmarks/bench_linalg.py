"""Benchmark suite: PyTorch linalg.py vs NumPy/SciPy equivalents.

Compares accuracy (max absolute error) and wall-clock performance of every
function in subdatapy.linalg against a pure NumPy/SciPy implementation.

For large matrices, chunked variants (TSQR, chunked leverage) stream data
through GPU without loading the full matrix — so you can benchmark matrices
much larger than GPU memory.

Usage:
    python benchmarks/bench_linalg.py                       # CPU-only
    python benchmarks/bench_linalg.py --device cuda          # GPU
    python benchmarks/bench_linalg.py --sizes 500 2000 5000  # custom sizes
    python benchmarks/bench_linalg.py --n-features 100       # wider matrix
"""

import argparse
import time

import numpy as np
import torch
try:
    import scipy.linalg as sla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from subdatapy import linalg


# ---------------------------------------------------------------------------
# NumPy/SciPy reference implementations
# ---------------------------------------------------------------------------

def np_qr_r(X_np):
    """R factor via numpy QR."""
    _, R = np.linalg.qr(X_np, mode='reduced')
    return R


def np_qr_qr(X_np):
    """Full Q, R via numpy QR (for leverage_scores_from_qr equivalent)."""
    return np.linalg.qr(X_np, mode='reduced')


def np_qr_r_xty(X_np, y_np):
    """R factor + X^T y via numpy."""
    R = np_qr_r(X_np)
    XTy = X_np.T @ y_np
    return R, XTy


def np_xtx_inv_from_r(R_np):
    """(X^T X)^{-1} = R^{-1} R^{-T} via numpy."""
    R_inv = np.linalg.inv(R_np)
    return R_inv @ R_inv.T


def np_xtx_inv_from_svd(X_np):
    """(X^T X)^{-1} via SVD, with singular-value filtering."""
    U, S, Vh = np.linalg.svd(X_np, full_matrices=False)
    tol = np.finfo(S.dtype).eps * max(X_np.shape) * S[0]
    S_inv_sq = np.where(S > tol, 1.0 / (S ** 2), 0.0)
    XTX_inv = Vh.T @ np.diag(S_inv_sq) @ Vh
    return XTX_inv, U, S, Vh


def np_solve_from_r_xty(R_np, XTy_np):
    """Solve beta = R^{-1} R^{-T} X^T y."""
    if HAS_SCIPY:
        z = sla.solve_triangular(R_np.T, XTy_np, lower=True)
        beta = sla.solve_triangular(R_np, z, lower=False)
    else:
        z = np.linalg.solve(R_np.T, XTy_np)
        beta = np.linalg.solve(R_np, z)
    return beta


def np_woodbury_update(XTX_inv_np, X_change_np, y_change_np, XTy_np, ascending):
    """Woodbury/Sherman-Morrison update (pure numpy)."""
    left = XTX_inv_np @ X_change_np.T
    I_k = np.eye(X_change_np.shape[0], dtype=X_change_np.dtype)
    inner_right = X_change_np @ left
    if ascending:
        inner = np.linalg.inv(I_k + inner_right)
        XTX_inv_new = XTX_inv_np - left @ inner @ (X_change_np @ XTX_inv_np)
        XTy_new = XTy_np + X_change_np.T @ y_change_np
    else:
        inner = np.linalg.inv(I_k - inner_right)
        XTX_inv_new = XTX_inv_np + left @ inner @ (X_change_np @ XTX_inv_np)
        XTy_new = XTy_np - X_change_np.T @ y_change_np
    return XTX_inv_new, XTy_new


def np_qr_update_add(R_np, X_new_np, y_new_np, XTy_np):
    """QR-based rank-k update: R_new = QR([R; X_new])."""
    stacked = np.vstack([R_np, X_new_np])
    R_new = np_qr_r(stacked)
    XTX_inv_new = np_xtx_inv_from_r(R_new)
    XTy_new = XTy_np + X_new_np.T @ y_new_np
    return R_new, XTX_inv_new, XTy_new


def np_leverage_from_qr(X_np):
    """Leverage via QR: h_i = ||Q_i||^2."""
    Q, _ = np.linalg.qr(X_np, mode='reduced')
    return np.sum(Q ** 2, axis=1)


def np_leverage_from_r(X_np, R_np):
    """Leverage via R: h_i = ||R^{-T} x_i||^2."""
    if HAS_SCIPY:
        BT = sla.solve_triangular(R_np.T, X_np.T, lower=True)
    else:
        BT = np.linalg.solve(R_np.T, X_np.T)
    return np.sum(BT ** 2, axis=0)


# ---------------------------------------------------------------------------
# Timing + formatting
# ---------------------------------------------------------------------------

def bench(fn, warmup=2, repeats=5, sync_cuda=False):
    """Time a function. Returns (result, median_time_seconds)."""
    for _ in range(warmup):
        result = fn()
    if sync_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return result, np.median(times)


def fmt_t(t):
    if t < 1e-3:
        return f"{t*1e6:8.1f} us"
    elif t < 1:
        return f"{t*1e3:8.2f} ms"
    else:
        return f"{t:8.3f}  s"


def fmt_e(e):
    if e == 0:
        return "         0"
    return f"{e:10.2e}"


def gpu_free_gb():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3
    return 0


def gpu_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def row(name, t_pt, t_np, err, note=""):
    speedup = t_np / t_pt if t_pt > 0 else float('inf')
    print(f"  {name:<38} {fmt_t(t_pt)} {fmt_t(t_np)} {speedup:7.2f}x {fmt_e(err)}  {note}")


def subrow(name, err):
    print(f"    {name:<36} {'':>12} {'':>12} {'':>8} {fmt_e(err)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark linalg.py: PyTorch vs NumPy/SciPy")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 20000])
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    sync = (args.device == "cuda")
    dev = args.device
    p = args.n_features
    reps = args.repeats

    if not HAS_SCIPY:
        print("WARNING: scipy not installed. Triangular solves use np.linalg.solve fallback.\n")

    print(f"Device: {dev} | dtype: float64 | features: {p} | repeats: {reps}")
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()} ({gpu_free_gb():.1f} GB free)")
    print(f"scipy: {HAS_SCIPY}")

    for n in args.sizes:
        mat_bytes = n * p * 8
        mat_gb = mat_bytes / 1024**3
        # Estimate: need ~3x matrix for QR/SVD working space
        fits_gpu = (dev == "cpu") or (mat_bytes * 3 < torch.cuda.mem_get_info()[0])

        print(f"\n{'='*105}")
        print(f"  {n:,} x {p}  ({mat_gb:.2f} GB)   GPU fit: {'yes' if fits_gpu else 'NO -> chunked only'}")
        print(f"{'='*105}")
        print(f"  {'Function':<38} {'PyTorch':>12} {'NumPy/SciPy':>12} {'Speedup':>8} {'Max |err|':>12}")
        print(f"  {'-'*100}")

        # Generate data on CPU
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        X_t = torch.randn(n, p, dtype=torch.float64)
        y_t = torch.randn(n, 1, dtype=torch.float64)
        X_n = X_t.numpy().copy()
        y_n = y_t.numpy().copy()

        def safe_run(name, fn):
            """Run a benchmark block; catch and report errors without dying."""
            try:
                fn()
            except Exception as e:
                print(f"  {name:<38} ** FAILED: {type(e).__name__}: {e}")
                gpu_cleanup()

        # --- 1. tsqr_r single-pass (needs full X on GPU) ---
        def _bench_1():
            if not fits_gpu:
                return
            r_pt, t_pt = bench(lambda: linalg.tsqr_r(X_t, device=dev), repeats=reps, sync_cuda=sync)
            r_np, t_np = bench(lambda: np_qr_r(X_n), repeats=reps)
            err = np.max(np.abs(np.abs(r_pt.cpu().numpy()) - np.abs(r_np)))
            row("tsqr_r (single-pass)", t_pt, t_np, err, "|R| compared")
            gpu_cleanup()
        safe_run("tsqr_r (single-pass)", _bench_1)

        # --- 2. tsqr_r chunked (X stays on CPU) ---
        R_pt = R_np = XTy_pt = XTy_np = None
        def _bench_2():
            nonlocal R_pt, R_np
            r_pt, t_pt = bench(
                lambda: linalg.tsqr_r(X_t, device=dev, n_chunks=args.n_chunks),
                repeats=reps, sync_cuda=sync)
            r_np, t_np = bench(lambda: np_qr_r(X_n), repeats=reps)
            err = np.max(np.abs(np.abs(r_pt.cpu().numpy()) - np.abs(r_np)))
            row(f"tsqr_r (chunked, k={args.n_chunks})", t_pt, t_np, err, "X on CPU, chunks->GPU")
            R_pt, R_np = r_pt, r_np
            gpu_cleanup()
        safe_run("tsqr_r (chunked)", _bench_2)

        # --- 3. tsqr_r_xty (X stays on CPU when chunked) ---
        def _bench_3():
            nonlocal XTy_pt, XTy_np
            (R2, xty_pt), t_pt = bench(
                lambda: linalg.tsqr_r_xty(X_t, y_t, device=dev, n_chunks=args.n_chunks),
                repeats=reps, sync_cuda=sync)
            (_, xty_np), t_np = bench(lambda: np_qr_r_xty(X_n, y_n), repeats=reps)
            err_R = np.max(np.abs(np.abs(R2.cpu().numpy()) - np.abs(R_np)))
            err_XTy = np.max(np.abs(xty_pt.cpu().numpy() - xty_np))
            row("tsqr_r_xty (chunked)", t_pt, t_np, max(err_R, err_XTy), "max(err_R, err_XTy)")
            XTy_pt, XTy_np = xty_pt, xty_np
            gpu_cleanup()
        safe_run("tsqr_r_xty (chunked)", _bench_3)

        # --- 4. xtx_inv_from_r (p x p only — always fits) ---
        def _bench_4():
            r_pt2, t_pt = bench(lambda: linalg.xtx_inv_from_r(R_pt, device=dev), repeats=reps, sync_cuda=sync)
            r_np2, t_np = bench(lambda: np_xtx_inv_from_r(R_np), repeats=reps)
            err = np.max(np.abs(r_pt2.cpu().numpy() - r_np2))
            row("xtx_inv_from_r", t_pt, t_np, err, "p x p only")
            gpu_cleanup()
        safe_run("xtx_inv_from_r", _bench_4)

        # --- 5. xtx_inv_from_svd (needs full X on GPU) ---
        def _bench_5():
            if not fits_gpu:
                return
            X_dev = X_t.to(dev)
            r_pt3, t_pt = bench(lambda: linalg.xtx_inv_from_svd(X_dev, device=dev), repeats=reps, sync_cuda=sync)
            r_np3, t_np = bench(lambda: np_xtx_inv_from_svd(X_n), repeats=reps)
            err = np.max(np.abs(r_pt3[0].cpu().numpy() - r_np3[0]))
            row("xtx_inv_from_svd", t_pt, t_np, err, "needs full X on GPU")
            del X_dev
            gpu_cleanup()
        safe_run("xtx_inv_from_svd", _bench_5)

        # --- 6. solve_from_r_xty (p x p only) ---
        def _bench_6():
            R_dev = R_pt.to(dev)
            xty_dev = XTy_pt.to(dev)
            r_pt4, t_pt = bench(lambda: linalg.solve_from_r_xty(R_dev, xty_dev), repeats=reps, sync_cuda=sync)
            r_np4, t_np = bench(lambda: np_solve_from_r_xty(R_np, XTy_np), repeats=reps)
            err = np.max(np.abs(r_pt4.cpu().numpy() - r_np4))
            backend = "scipy" if HAS_SCIPY else "np.solve"
            row("solve_from_r_xty", t_pt, t_np, err, f"p x p, vs {backend}")
            gpu_cleanup()
        safe_run("solve_from_r_xty", _bench_6)

        # --- 7. woodbury_update (k=100, always CPU — tiny matrices) ---
        def _bench_7():
            k = min(100, n // 10)
            split = n - k
            # Use TSQR on GPU for setup, then move to CPU for the actual update
            R_init = linalg.tsqr_r(X_t[:split], device=dev, n_chunks=args.n_chunks)
            XTX_inv_t = linalg.xtx_inv_from_r(R_init, device='cpu')
            XTy_init_t = linalg.tsqr_r_xty(X_t[:split], y_t[:split], device=dev, n_chunks=args.n_chunks)[1].cpu()
            del R_init
            gpu_cleanup()
            # Numpy: reuse R_np (from bench 2) for inverse instead of recomputing QR
            XTX_inv_n = np_xtx_inv_from_r(R_np)
            XTy_init_n = X_n.T @ y_n  # Cheap: just matmul
            # Subtract the update rows' contribution to get initial XTy
            XTy_init_n = XTy_init_n - X_n[split:].T @ y_n[split:]
            r_pt5, t_pt = bench(
                lambda: linalg.woodbury_update(XTX_inv_t, X_t[split:], y_t[split:], XTy_init_t, ascending=True),
                repeats=reps)
            r_np5, t_np = bench(
                lambda: np_woodbury_update(XTX_inv_n, X_n[split:], y_n[split:], XTy_init_n, ascending=True),
                repeats=reps)
            err = max(
                np.max(np.abs(r_pt5[0].numpy() - r_np5[0])),
                np.max(np.abs(r_pt5[1].numpy() - r_np5[1])))
            row(f"woodbury_update (k={k})", t_pt, t_np, err, "CPU, p x p ops")
        safe_run("woodbury_update", _bench_7)

        # --- 8. qr_update_add (k=200, always CPU — tiny matrices) ---
        def _bench_8():
            k2 = min(200, n // 10)
            split2 = n - k2
            # Reuse R from bench 2 as the "initial" R (close enough for benchmark timing)
            R1_t = R_pt.cpu()
            XTy1_t = torch.from_numpy(X_n[:split2].T @ y_n[:split2]).to(torch.float64)
            R1_n = R_np.copy()
            XTy1_n = X_n[:split2].T @ y_n[:split2]
            r_pt6, t_pt = bench(
                lambda: linalg.qr_update_add(R1_t, X_t[split2:], y_t[split2:], XTy1_t, device='cpu'),
                repeats=reps)
            r_np6, t_np = bench(
                lambda: np_qr_update_add(R1_n, X_n[split2:], y_n[split2:], XTy1_n),
                repeats=reps)
            err = np.max(np.abs(np.abs(r_pt6[0].numpy()) - np.abs(r_np6[0])))
            row(f"qr_update_add (k={k2})", t_pt, t_np, err, "CPU, p x p ops")
        safe_run("qr_update_add", _bench_8)
        gpu_cleanup()

        # --- 9. leverage_scores_from_qr (needs full X on GPU) ---
        def _bench_9():
            if not fits_gpu:
                return
            r_pt7, t_pt = bench(
                lambda: linalg.leverage_scores_from_qr(X_t, device=dev),
                repeats=reps, sync_cuda=sync)
            r_np7, t_np = bench(lambda: np_leverage_from_qr(X_n), repeats=reps)
            err = np.max(np.abs(r_pt7.cpu().numpy() - r_np7))
            row("leverage_scores_from_qr", t_pt, t_np, err, "single-pass, ||Q_i||^2")
            gpu_cleanup()
        safe_run("leverage_scores_from_qr", _bench_9)

        # --- 10. leverage_scores_from_r single-pass (needs full X on GPU) ---
        def _bench_10():
            if not fits_gpu:
                return
            r_pt8, t_pt = bench(
                lambda: linalg.leverage_scores_from_r(X_t, R_pt, device=dev),
                repeats=reps, sync_cuda=sync)
            r_np8, t_np = bench(lambda: np_leverage_from_r(X_n, R_np), repeats=reps)
            err = np.max(np.abs(r_pt8.cpu().numpy() - r_np8))
            row("leverage_from_r (single)", t_pt, t_np, err, "||R^{-T} x_i||^2")
            gpu_cleanup()
        safe_run("leverage_from_r (single)", _bench_10)

        # --- 11. leverage_scores_from_r chunked (X stays on CPU) ---
        def _bench_11():
            r_pt9, t_pt = bench(
                lambda: linalg.leverage_scores_from_r(X_t, R_pt, device=dev, n_chunks=args.n_chunks),
                repeats=reps, sync_cuda=sync)
            r_np9, t_np = bench(lambda: np_leverage_from_r(X_n, R_np), repeats=reps)
            err = np.max(np.abs(r_pt9.cpu().numpy() - r_np9))
            row(f"leverage_from_r (chunked, k={args.n_chunks})", t_pt, t_np, err, "X on CPU, chunks->GPU")
            gpu_cleanup()
        safe_run("leverage_from_r (chunked)", _bench_11)

        # Free data before next size
        del X_t, y_t, X_n, y_n
        if R_pt is not None:
            del R_pt
        if R_np is not None:
            del R_np
        gpu_cleanup()

    # --- Coverage summary ---
    print(f"\n{'='*105}")
    print("NumPy/SciPy Coverage Summary")
    print(f"{'='*105}")
    coverage = [
        ("tsqr_r",                "numpy",  "np.linalg.qr(X, mode='reduced')"),
        ("tsqr_r (chunked)",      "numpy",  "No chunked mode; single-pass np.linalg.qr only"),
        ("tsqr_r_xty",            "numpy",  "np.linalg.qr + X.T @ y"),
        ("xtx_inv_from_r",        "numpy",  "np.linalg.inv(R) @ np.linalg.inv(R).T"),
        ("xtx_inv_from_svd",      "numpy",  "np.linalg.svd + Vh.T @ diag(1/S^2) @ Vh"),
        ("solve_from_r_xty",      "SCIPY",  "scipy.linalg.solve_triangular (x2). NO numpy triangular solve."),
        ("woodbury_update",        "numpy",  "Pure matrix ops (np.linalg.inv for k x k inner block)"),
        ("qr_update_add",          "numpy",  "np.linalg.qr(vstack([R, X_new])) + inv + matmul"),
        ("leverage_from_qr",       "numpy",  "np.linalg.qr + sum(Q^2, axis=1)"),
        ("leverage_from_r",        "SCIPY",  "scipy.linalg.solve_triangular + sum(B^2). NO numpy equiv."),
        ("leverage_from_r chunked","SCIPY",  "Same as above; numpy has no chunked mode."),
    ]
    print(f"  {'Function':<30} {'Needs':>6}  {'NumPy/SciPy equivalent'}")
    print(f"  {'-'*100}")
    for fn, needs, equiv in coverage:
        marker = "**" if needs == "SCIPY" else "  "
        print(f"  {fn:<30} {needs:>6}  {marker} {equiv}")
    print(f"\n  ** = Requires scipy. Without it, falls back to np.linalg.solve (slower, ignores triangularity).")


if __name__ == "__main__":
    main()
