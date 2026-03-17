import torch
import pytest
from subdatapy import linalg


@pytest.fixture
def tall_skinny():
    torch.manual_seed(0)
    X = torch.randn(1000, 50, dtype=torch.float64)
    y = torch.randn(1000, 1, dtype=torch.float64)
    return X, y


def test_tsqr_r_single_pass(tall_skinny):
    """Single-pass QR matches torch.linalg.qr."""
    X, _ = tall_skinny
    R = linalg.tsqr_r(X, device='cpu')
    _, R_ref = torch.linalg.qr(X, mode='r')
    # R unique up to sign flips on rows
    assert torch.allclose(R.abs(), R_ref.abs(), atol=1e-10)


def test_tsqr_r_sequential_chunks(tall_skinny):
    """Sequential TSQR with n_chunks=4 matches single-pass."""
    X, _ = tall_skinny
    R_single = linalg.tsqr_r(X, device='cpu')
    R_chunked = linalg.tsqr_r(X, device='cpu', n_chunks=4)
    assert torch.allclose(R_single.abs(), R_chunked.abs(), atol=1e-10)


def test_tsqr_r_xty_matches_direct(tall_skinny):
    """XTy from tsqr_r_xty matches X.T @ y."""
    X, y = tall_skinny
    R, XTy = linalg.tsqr_r_xty(X, y, device='cpu', n_chunks=4)
    XTy_ref = X.T @ y
    assert torch.allclose(XTy, XTy_ref, atol=1e-10)


def test_xtx_inv_from_r_matches_svd(tall_skinny):
    """QR-based and SVD-based (X'X)^{-1} match."""
    X, _ = tall_skinny
    _, R = torch.linalg.qr(X, mode='r')
    inv_qr = linalg.xtx_inv_from_r(R, device='cpu')
    inv_svd, _, _, _ = linalg.xtx_inv_from_svd(X, device='cpu')
    assert torch.allclose(inv_qr, inv_svd, atol=1e-8)


def test_woodbury_update_matches_recompute(tall_skinny):
    """Woodbury update matches recomputing inverse from scratch after adding rows."""
    X, y = tall_skinny
    X1, y1 = X[:800], y[:800]
    X2, y2 = X[800:], y[800:]

    # Direct computation on full data
    XTX_full = X.T @ X
    inv_full = torch.linalg.inv(XTX_full)
    XTy_full = X.T @ y

    # Woodbury: start with partial, add remaining
    XTX_partial = X1.T @ X1
    inv_partial = torch.linalg.inv(XTX_partial)
    XTy_partial = X1.T @ y1

    inv_updated, XTy_updated = linalg.woodbury_update(
        inv_partial, X2, y2, XTy_partial, ascending=True)

    assert torch.allclose(inv_updated, inv_full, atol=1e-6)
    assert torch.allclose(XTy_updated, XTy_full, atol=1e-10)


def test_qr_update_add_matches_full(tall_skinny):
    """QR update matches full QR of [X_old; X_new]."""
    X, y = tall_skinny
    X1, y1 = X[:800], y[:800]
    X2, y2 = X[800:], y[800:]

    _, R1 = torch.linalg.qr(X1, mode='r')
    XTy1 = X1.T @ y1

    R_new, inv_new, XTy_new = linalg.qr_update_add(R1, X2, y2, XTy1, device='cpu')

    _, R_ref = torch.linalg.qr(X, mode='r')
    assert torch.allclose(R_new.abs(), R_ref.abs(), atol=1e-10)


def test_leverage_scores_from_r_matches_svd(tall_skinny):
    """QR-based leverage matches SVD-based sum(U^2)."""
    X, _ = tall_skinny
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    lev_svd = torch.sum(U ** 2, dim=1)

    _, R = torch.linalg.qr(X, mode='r')
    lev_qr = linalg.leverage_scores_from_r(X, R, device='cpu')

    assert torch.allclose(lev_svd, lev_qr, atol=1e-10)


def test_leverage_scores_from_qr_matches_svd(tall_skinny):
    """Single-pass QR leverage (||Q_i||^2) matches SVD-based sum(U^2)."""
    X, _ = tall_skinny
    U, _, _ = torch.linalg.svd(X, full_matrices=False)
    lev_svd = torch.sum(U ** 2, dim=1)

    lev_qr = linalg.leverage_scores_from_qr(X, device='cpu')

    assert torch.allclose(lev_svd, lev_qr, atol=1e-10)


def test_leverage_scores_from_r_chunked_matches_single(tall_skinny):
    """Chunked leverage_scores_from_r matches single-pass."""
    X, _ = tall_skinny
    _, R = torch.linalg.qr(X, mode='r')

    lev_single = linalg.leverage_scores_from_r(X, R, device='cpu')
    lev_chunked = linalg.leverage_scores_from_r(X, R, device='cpu', n_chunks=4)

    assert torch.allclose(lev_single, lev_chunked, atol=1e-10)


def test_solve_from_r_xty(tall_skinny):
    """Triangular solve matches lstsq."""
    X, y = tall_skinny
    _, R = torch.linalg.qr(X, mode='r')
    XTy = X.T @ y

    beta_qr = linalg.solve_from_r_xty(R, XTy)
    beta_lstsq = torch.linalg.lstsq(X, y).solution

    assert torch.allclose(beta_qr, beta_lstsq, atol=1e-10)
