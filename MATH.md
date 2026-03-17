# Mathematical Reference

This document provides detailed mathematical derivations and implementation notes for every algorithm in SubDataPy. Each equation is mapped to the specific function and code location that implements it.

For full derivations and proofs, see the paper in `paper_latex/main.tex`.

**Notation conventions:** Tildes (e.g., X&#771;) denote weight-premultiplied quantities. Subscripts like _(n)_ or _(n+m)_ indicate which data was used for training. The `ps^2` normalization factor from the paper is omitted in the implementation since it is a constant that does not affect configuration ranking.

---

## 1. Notation

| Symbol | Meaning | Shape | Code variable |
|--------|---------|-------|---------------|
| **X** | Descriptor matrix (with optional intercept column) | (n, p) | `self.X` / `self.X_train` |
| **y** | Target vector (energies + forces) | (n, 1) | `self.y` / `self.y_train` |
| **W** | Diagonal weight matrix | (n, n) | `self.w` / `self.w_train` |
| **X&#771;** | Weighted descriptor matrix: W X | (n, p) | `self.w_train * self.X_train` |
| **y&#771;** | Weighted target vector: W y | (n, 1) | `self.w_train * self.y_train` |
| n | Number of rows (energy + force) | scalar | `X_train.shape[0]` |
| p | Number of features (+ intercept) | scalar | `X_train.shape[1]` |
| **beta** | Regression coefficients | (p, 1) | `self.coeffs` / `self.sub_coeffs` |
| **H** | Hat matrix: X&#771;(X&#771;^T X&#771;)^{-1} X&#771;^T | (n, n) | never formed explicitly |
| h_i | Leverage score for row i | scalar | `leverage_scores[i]` |
| H_c | Block hat matrix for config c: X_c (X&#771;^T X&#771;)^{-1} X_c^T | (k_c, k_c) | `fake_lev` in `cooks.py` |
| e_i | Residual: x_i^T beta - y_i | scalar | `en_residuals` |
| e_c | Block residual vector: X_c beta - y_c | (k_c, 1) | `res` in `cooks.py` |
| config_idxs | Maps rows to configurations | (n,) | `self.config_idxs` |
| enrow_mask | True for energy rows | (n,) | `self.enrow_mask` |
| **R** | Upper-triangular factor from QR(X&#771;) | (p, p) | `R` / `self.R_final` |
| **U, S, Vh** | SVD factors: X&#771; = U S Vh | (n,p), (p,), (p,p) | `self.U`, `self.S`, `self.Vh` |

---

## 2. Weighted Least Squares

**Problem:** Find beta that minimizes the weighted residual sum of squares:

```
min_beta  || W(X beta - y) ||^2  =  || X~ beta - y~ ||^2
```

where X&#771; = WX and y&#771; = Wy are the weight-premultiplied matrix and vector.

**Normal equations:**

```
(X~^T X~) beta = X~^T y~
```

### 2.1 Solution via lstsq (default)

Solves the normal equations using `torch.linalg.lstsq(X~, y~)`, which internally uses a stable factorization.

**Code:** `data.py:train()` (lines 140-141), `random.py:train_subsample()` (lstsq path)

### 2.2 Solution via QR

Factor X&#771; = QR, then solve via two triangular solves:

```
R^T z = X~^T y~     (forward substitution)
R beta = z           (back substitution)
```

This avoids forming (X&#771;^T X&#771;)^{-1} explicitly and is numerically stable.

**Code:** `data.py:train()` (lines 134-138, method='qr'), `random.py:train_subsample()` (qr path)
- R and X^Ty computed by: `linalg.tsqr_r_xty()` (`linalg.py:86`)
- Triangular solve: `linalg.solve_from_r_xty()` (`linalg.py:212`)

---

## 3. Matrix Factorizations

### 3.1 SVD Path

Full SVD of the weighted matrix:

```
X~ = U S Vh
```

where U is (n, p), S is (p,), Vh is (p, p).

The inverse of the normal matrix:

```
(X~^T X~)^{-1} = Vh^T  diag(1/s_i^2)  Vh
```

with singular value filtering: s_i is set to zero if `s_i < eps * max(n, p) * s_1`, where `eps` is machine epsilon for the dtype. This prevents numerical instability from near-zero singular values.

**Code:** `linalg.xtx_inv_from_svd()` (`linalg.py:199`)
- Returns `(XTX_inv, U, S, Vh)`

### 3.2 QR Path

Economy QR factorization:

```
X~ = Q R
```

where Q is (n, p) with orthonormal columns and R is (p, p) upper triangular.

The inverse of the normal matrix:

```
(X~^T X~)^{-1} = (R^T R)^{-1} = R^{-1} R^{-T}
```

**Numerical note:** The inversion of R is performed on CPU in float64 to avoid precision loss, then cast back to the working dtype and device.

**Code:** `linalg.xtx_inv_from_r()` (`linalg.py:188`)

### 3.3 TSQR (Tall-Skinny QR)

For matrices too large to factorize in a single pass, TSQR partitions X&#771; into chunks and reduces:

```
X~ = [X~_1; X~_2; ...; X~_k]

Step 1 (local QR):    X~_i = Q_i R_i       for each chunk i
Step 2 (reduce):      QR([R_1; R_2; ...; R_k]) -> R_final
```

**Key property:** R_final from TSQR equals R from QR(X&#771;) up to sign of rows. This means all downstream computations (inverse, leverage, solve) are equivalent.

X^T y is accumulated per-chunk: `X~^T y~ = sum_i X~_i^T y~_i`, then summed across ranks in distributed mode via `dist.reduce(op=SUM)`.

**4 Execution Modes:**

| `n_chunks` | `world_size` | Mode | Description |
|:---:|:---:|---|---|
| None | 1 | Single-pass | Direct `torch.linalg.qr(X~)` |
| >1 | 1 | Sequential TSQR | Stream chunks through one GPU |
| None | >1 | Parallel TSQR | 1 chunk per rank, gather + reduce |
| >1 | >1 | Hybrid TSQR | `n_chunks/world_size` chunks per rank |

**Tree reduction:** After `tree_reduction_threshold` (default 10) R matrices accumulate, they are reduced to bound memory at O(threshold * p^2).

**Distributed collectives:** Only two NCCL collectives are needed:
1. `dist.gather` — collect R matrices from all ranks to rank 0
2. `dist.reduce` — sum X^Ty across ranks

Results (R, X^Ty) are returned on rank 0 only; other ranks get `(None, None)`.

**Code:** `linalg.tsqr_r()` (`linalg.py:67`), `linalg.tsqr_r_xty()` (`linalg.py:86`), `linalg._tsqr_core()` (`linalg.py:100`)

---

## 4. Random Subsampling

### 4.1 Method

Uniform random selection of configurations without replacement:

```
P(select config c) = 1/N    for all c in {1, ..., N}
```

Select k = round(N * subsample_fraction) configs via `torch.randperm(N)[:k]`.

**Code:** `random.py:_create_sub_mask()` (lines 37-42)

### 4.2 Training on Subsample

Same WLS problem restricted to subsample rows S:

```
min_beta  || W_S (X_S beta - y_S) ||^2
```

Supports both `lstsq` (default) and `qr` methods, matching `BaseData.train()`.

**Code:** `random.py:train_subsample()` (lines 53-65)

### 4.3 Error Evaluation

Coefficients beta_S trained on the subsample are evaluated on three partitions:
- Subsampled training data (energy + force RMSE)
- Entire training data (energy + force RMSE)
- Test data (energy + force RMSE)

```
RMSE = sqrt( mean( (X @ beta_S - y)^2 ) )
```

computed separately for energy rows (`enrow_mask`) and force rows (`~enrow_mask`).

**Code:** `random.py:compute_subsample_errors()` (lines 63-90)

---

## 5. Leverage Score Subsampling

### 5.1 Definition

The leverage score of row i is the i-th diagonal element of the hat matrix:

```
h_i = [H]_{ii} = x_i^T (X~^T X~)^{-1} x_i
```

Leverage measures how much influence row i has on its own fitted value. High-leverage points are far from the center of the feature space.

### 5.2 SVD-based Computation

Since H = X&#771;(X&#771;^T X&#771;)^{-1} X&#771;^T and X&#771; = U S Vh:

```
H = U S Vh (Vh^T S^{-2} Vh) Vh^T S U^T = U U^T
```

Therefore:

```
h_i = sum_j  U_{ij}^2 = ||U_i||^2
```

**Code:** `leverage.py:_create_sub_mask()` SVD path (lines 42-44)

### 5.3 QR-based Computation

Since (X&#771;^T X&#771;)^{-1} = R^{-1} R^{-T}:

```
h_i = x_i^T R^{-1} R^{-T} x_i = ||R^{-T} x_i||^2
```

**Implementation:** Solve R^T B^T = X&#771;^T for B using `torch.linalg.solve_triangular`, then h_i = ||B_i||^2. This avoids forming R^{-1} explicitly.

**Code:** `linalg.leverage_scores_from_r()` (`linalg.py:273`)
- Called from `leverage.py:_create_sub_mask()` QR path (lines 36-39)

### 5.4 Proof of Equivalence (SVD vs QR)

The SVD and QR paths compute the same leverage scores:

```
X~ = QR   and   X~ = U S Vh

=> Q = U S Vh R^{-1}

=> QQ^T = U S Vh R^{-1} R^{-T} Vh^T S U^T
```

Since R^T R = X&#771;^T X&#771; = Vh^T S^2 Vh, we have R^{-1} R^{-T} = Vh^T S^{-2} Vh, so:

```
QQ^T = U S Vh (Vh^T S^{-2} Vh) Vh^T S U^T = U U^T = H
```

Therefore `diag(QQ^T) = diag(UU^T)`, and both `||R^{-T} x_i||^2` and `||U_i||^2` give identical leverage scores.

### 5.5 Why R^{-T} instead of QQ^T?

If the full Q factor is available, leverage scores are simply `h_i = ||Q_i||^2` at cost **O(np)** — just squaring and summing each row of Q. No matrix inversion or triangular solves needed.

The `||R^{-T} x̃_i||^2` path is **p times more expensive: O(np^2)**, because it solves the triangular system `R^T B^T = X̃^T` at O(p^2) per row × n rows.

We use the R^{-T} path anyway because **TSQR discards Q and only returns R**:
1. **Memory:** TSQR only computes R (p × p), discarding Q (n × p). For tall matrices (n >> p), storing Q would dominate memory.
2. **Distributed:** In distributed mode, only the small R matrices are communicated across ranks, not the large Q factors. The full Q is never formed.
3. **Streaming:** In chunked mode, each chunk's local Q_i is discarded after extracting R_i — the global Q never exists.

**Cost comparison:**

| Approach | Requires | Cost |
|---|---|---|
| `\|\|Q_i\|\|^2` | Full Q (n × p) | O(np) |
| `\|\|R^{-T} x̃_i\|\|^2` | Only R (p × p) | O(np²) |
| SVD `\|\|U_i\|\|^2` | Full SVD | O(np²) for SVD + O(np) for row norms |

The R^{-T} path pays an extra factor of p in compute to avoid ever forming or storing the (n × p) orthogonal factor.

**Numerical accuracy:** The `||Q_i||^2` and `||R^{-T} x̃_i||^2` approaches are mathematically identical but can differ in floating-point arithmetic. The Q factor from Householder QR is orthogonal to machine precision, so `||Q_i||^2` is as accurate as the factorization itself. The R^{-T} path solves a triangular system, which has forward error bounded by O(p · κ(R) · ε), where κ(R) = κ(X̃) is the condition number and ε is machine epsilon. For well-conditioned problems the results are indistinguishable; for ill-conditioned X̃, the Q path is more numerically stable since it avoids amplifying condition number through the triangular solve. In practice, since we use float64 throughout, the difference is negligible for the problem sizes we target.

**Implementation:** The code selects the appropriate path automatically:
- **Single-pass** (no chunks, not distributed): Uses `leverage_scores_from_qr()` — retains Q, computes `||Q_i||^2`. O(np) leverage after O(np²) QR.
- **Chunked/distributed**: Uses `leverage_scores_from_r()` with `n_chunks` — TSQR computes R, then streams X through GPU in chunks for the triangular solve. Never loads full X to GPU.

**Code:** `linalg.leverage_scores_from_qr()` (`linalg.py:293`), `linalg.leverage_scores_from_r()` (`linalg.py:316`), `leverage.py:_create_sub_mask()` (path selection at lines 35-49)

### 5.6 Block Mode

In block mode, leverage scores are summed over all rows of each configuration:

```
h_c = sum_{i in config c}  h_i
```

This gives the total influence of the entire configuration (energy + all force rows), not just the energy row.

**Code:** `leverage.py` block path (lines 47-52) using `index_add_`

### 5.7 Non-block Mode

In non-block mode, only the energy row's leverage score is used:

```
h_c = h_{energy_row(c)}
```

This is cheaper but ignores force-row influence.

**Code:** `leverage.py` non-block path (lines 54-58)

### 5.8 Sampling

Configurations are sampled without replacement with probability proportional to their leverage scores:

```
P(select config c) = h_c / sum_c' h_c'
```

Uses `torch.multinomial(probs, n_subsamples, replacement=False)`.

**Code:** `leverage.py:_create_sub_mask()` sampling section (lines 61-71)

---

## 6. Cook's Distance Subsampling

### 6.1 Subtractive Cook's Distance (One-step, Descending)

**Paper Eq 2 (eq:cooks_2), non-block:**

```
D_i = e_i^2 * h_i / (1 - h_i)^2
```

where e_i = x_i^T beta - y_i is the residual and h_i is the leverage score.

This measures the change in all predictions when row i is removed from the training set. Computed on energy rows only.

**Paper Eq 3 (eq:final_block_cooks), block:**

```
D_c = e_c^T (I - H_c)^{-1} H_c (I - H_c)^{-1} e_c
```

where H_c = X_c (X&#771;^T X&#771;)^{-1} X_c^T and e_c = X_c beta - y_c.

The one-step method computes D for all configs, then either samples proportionally or selects top-k.

**Code:** `cooks.py:_onestep_cooks_sampling()` (lines 153-184)

### 6.2 Additive Cook's Distance (Stepwise, Ascending)

**Paper Eq 4 (eq:final_additive_cooks_forces), non-block:**

```
D_i = e_i^2 * h_i / (1 + h_i)
```

Note the `+` sign in the denominator (vs `-` for subtractive) and no squaring.

**Paper Eq 5 (eq:final_block_add_cooks), block:**

```
D_c = e_c^T (I + H_c)^{-1} H_c e_c
```

This measures the change in predictions when config c is **added** to the current training set. The stepwise algorithm greedily adds the config with highest D at each iteration.

**Code:** `cooks.py:_stepwise_cooks_sampling()` with `ascending=True` (lines 212-282)

### 6.3 Stepwise Descending

Same formulas as subtractive Cook's (Section 6.1) but applied iteratively: at each step, remove the config with **smallest** D (least influential).

The sign in the Woodbury update flips, and the mask logic is inverted.

**Code:** `cooks.py:_stepwise_cooks_sampling()` with `ascending=False`

### 6.4 Non-block Cook's Implementation

For energy rows only (one per config):

```
h_i = x_i^T (X~^T X~)^{-1} x_i = sum_j (x_i * [XTX_inv]_j)^2
```

Implemented as a row-wise dot product: `torch.sum((X_en @ XTX_inv) * X_en, dim=1)`.

```
D_i = e_i^2 * h_i / (1 + h_i)      (ascending)
D_i = e_i^2 * h_i / (1 - h_i)^2    (descending / one-step)
```

**Code:** `cooks.py:_compute_nonblock_cooks()` (lines 369-393)

### 6.5 Block Cook's Implementation

For each configuration c with k_c rows:

```
X_c in R^{k_c x p}         (rows of X~ belonging to config c)
H_c = X_c (X~^T X~)^{-1} X_c^T    in R^{k_c x k_c}
e_c = X_c beta - y_c               in R^{k_c x 1}
```

Cook's distance:

```
D_c = e_c^T (I +/- H_c)^{-1} H_c e_c      (scalar)
```

where `+` for ascending (additive) and `-` for descending (subtractive).

Configs are processed in batches of `BATCH_SIZE=5000`:
1. Pad X_c and y_c to max group size within the batch
2. Compute H_c = X_c @ XTX_inv @ X_c^T via batched matrix multiply
3. Compute residuals e_c = X_c @ beta - y_c
4. Compute (I +/- H_c)^{-1} via `torch.linalg.inv`
5. Compute D_c = e_c^T @ inv @ H_c @ e_c via batched matmul chain

Uses `_batched_matmul()` with a sequential loop fallback for CUBLAS compatibility on some GPU/driver combinations.

**Code:** `cooks.py:_compute_block_cooks()` (lines 288-363)

---

## 7. Incremental Updates

### 7.1 Woodbury / Sherman-Morrison Identity

**Paper Eq 6 (eq:woodbury).**

When **adding** k rows X' to the current subset (ascending):

```
(A + X'^T X')^{-1} = A^{-1} - A^{-1} X'^T (I + X' A^{-1} X'^T)^{-1} X' A^{-1}
```

When **removing** k rows X' (descending):

```
(A - X'^T X')^{-1} = A^{-1} + A^{-1} X'^T (I - X' A^{-1} X'^T)^{-1} X' A^{-1}
```

Also updates X^T y:

```
X^T y  <-  X^T y + X'^T y'    (ascending)
X^T y  <-  X^T y - X'^T y'    (descending)
```

**Cost:** O(p^2 k + k^3) per step, where k is the number of rows being added/removed (typically k_c for one config). This is much cheaper than the O(np^2) cost of full recomputation.

**Code:** `linalg.woodbury_update()` (`linalg.py:226`)

### 7.2 QR Update (Block Mode)

Stack the current R with new rows and re-factorize:

```
R_new = QR([R_old; X_new])[1]     (R factor only)
(X~^T X~)^{-1}_new = R_new^{-1} R_new^{-T}
X^T y_new = X^T y_old + X_new^T y_new
```

More numerically stable than Woodbury for block mode where the update chunks can be large relative to p.

**Cost:** O((p + k) p^2) per step.

**Code:** `linalg.qr_update_add()` (`linalg.py:253`)

### 7.3 Update Method Selection

`update_method='auto'` (default) selects:
- **QR** for block mode: block updates involve larger chunks (k_c rows per config), where QR's numerical stability matters more
- **Woodbury** for non-block mode: single-row updates (k=1), where Woodbury is cheaper and stable enough

**Code:** `cooks.py:_use_qr_update()` (lines 79-84)

---

## 8. Stepwise Algorithm

Pseudocode for the full stepwise Cook's loop:

```
Input: X~, y~, initial_fraction, target_fraction, ascending

1. Create initial subset S_0 (random or leverage-based)

2. Factorize X~[S_0]:
   - QR path: R, X^Ty = TSQR(X~[S_0], y~[S_0])
              (X~^T X~)^{-1} = R^{-1} R^{-T}
   - SVD path: (X~^T X~)^{-1}, U, S, Vh = SVD(X~[S_0])
              X^Ty = X~[S_0]^T y~[S_0]

3. For each step until |S| reaches target:

   a. Compute coefficients:
      beta = (X~^T X~)^{-1} X^T y

   b. For each candidate config c not in S (ascending)
      or c in S (descending):
      - Compute Cook's distance D_c
        (block or non-block, see Section 6)

   c. Select config c* with:
      - max D_c (ascending: add most influential)
      - min D_c (descending: remove least influential)

   d. Update S:
      - S <- S U {c*}   (ascending)
      - S <- S \ {c*}   (descending)

   e. Update (X~^T X~)^{-1} and X^T y:
      - Woodbury update (non-block, see Section 7.1)
      - QR update (block, see Section 7.2)

4. Return final subset mask S
```

**Code:** `cooks.py:_stepwise_cooks_sampling()` (lines 212-282)

- Initial subset: `cooks.py:_create_initial_sub_mask()` (lines 190-206)
- Initial factorization: `linalg.tsqr_r_xty()` (QR) or `linalg.xtx_inv_from_svd()` (SVD)
- Cook's computation: `cooks.py:_compute_block_cooks()` or `_compute_nonblock_cooks()`
- Incremental update: `linalg.woodbury_update()` or `linalg.qr_update_add()`

---

## 9. Numerical Stability Notes

### SVD Filtering
Singular values below `eps * max(n, p) * s_1` are treated as zero when computing (X&#771;^T X&#771;)^{-1}. This prevents amplification of noise in near-singular directions.

**Code:** `linalg.xtx_inv_from_svd()` (`linalg.py:205-207`)

### QR Inverse on CPU float64
The triangular inverse R^{-1} is computed on CPU in float64, then cast back to the working dtype/device. This avoids precision loss that can occur in GPU float32/float64 triangular inversion for ill-conditioned R.

**Code:** `linalg.xtx_inv_from_r()` (`linalg.py:193-196`)

### Triangular Solves over Explicit Inverse
Where possible, triangular solves (`solve_triangular`) are preferred over forming R^{-1} explicitly. Used in:
- Coefficient solve: `linalg.solve_from_r_xty()` (`linalg.py:217-218`)
- Leverage scores: `linalg.leverage_scores_from_r()` (`linalg.py:289`)

### Tree Reduction in TSQR
Accumulated R matrices are reduced after `tree_reduction_threshold` (default 10) to bound peak memory at O(threshold * p^2) rather than O(n_chunks * p^2).

**Code:** `linalg._tsqr_core()` (lines 154-155)

### CUBLAS Compatibility
`_batched_matmul()` wraps `torch.bmm` with a loop fallback for CUBLAS errors that can occur with float64 on some GPU/driver combinations.

**Code:** `cooks.py:_batched_matmul()` (lines 8-18)

---

## 10. Paper Cross-Reference

| Paper Equation | Label | Description | Code Location |
|:-:|---|---|---|
| Eq 2 | `eq:cooks_2` | Subtractive Cook's distance (non-block) | `cooks.py:_onestep_cooks_sampling()` (line 173) |
| Eq 3 | `eq:final_block_cooks` | Block subtractive Cook's distance | `cooks.py:_compute_block_cooks()` (descending path) |
| Eq 4 | `eq:final_additive_cooks_forces` | Additive Cook's distance (non-block) | `cooks.py:_compute_nonblock_cooks()` (line 378) |
| Eq 5 | `eq:final_block_add_cooks` | Block additive Cook's distance | `cooks.py:_compute_block_cooks()` (ascending path, line 334) |
| Eq 6 | `eq:woodbury` | Woodbury matrix identity | `linalg.woodbury_update()` (lines 237-248) |
| Eq A.8 | `eq:Final_gen_cooks` | Derived block SCD formula | `cooks.py:_compute_block_cooks()` with `ascending=False` |
| Eq B.6 | `eq:Final_gen_cooks_add` | Derived block ACD formula | `cooks.py:_compute_block_cooks()` with `ascending=True` |

**Implementation note:** The paper formulas include a `1/(ps^2)` normalization factor. The code omits this because it is a constant across all configurations and does not affect the argmax/argmin selection used for stepwise greedy ranking. The one-step method (`_onestep_cooks_sampling`) also omits `ps^2` since it either samples proportionally (ratio preserved) or selects top-k (ranking preserved).
