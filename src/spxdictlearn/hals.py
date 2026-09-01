"""
Hierarchical Alternating Least Squares Weighted NMF (HALS-WNMF)

This module implements the HALS algorithm for spectral decomposition,
addressing the "twin spectra" problem of standard ALS-WNMF through
sequential deflation updates.

Key features:
- Precomputed constants (C, global_B_data) for efficient iteration
- BLAS-optimized aggregation tensor M computation
- Projected gradient descent with non-negativity constraint
- Support for regularization (L2, smoothness)

Algorithm reference:
Cichocki, A., & Phan, A. H. (2009). Fast local algorithms for large scale
nonnegative matrix and tensor factorizations. IEICE Transactions on
Fundamentals, 92(3), 708-721.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .als_wnmf import compute_regularized_loss, e_step
from .utils import validate_non_negativity


def precompute_hals_constants(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    N: int,
    T: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute C matrix and aligned B data for BLAS-optimized HALS.

    Two-pass approach to minimize peak memory:
    - Pass 1: Compute C and build global B_n non-zero pattern via dense mask (~4 MB)
    - Pass 2: Recompute B_n and align to global template via vectorized indexing

    C_n = R_n^T @ Σ_n^(-1) @ y_n
    B_n = R_n^T @ Σ_n^(-1) @ R_n

    Parameters
    ----------
    sources_data : List[np.ndarray]
        List of D_n arrays for all sources
        D_n[:, 2] = flux (y_n), D_n[:, 3] = error (σ_n)
    response_matrices : List[csr_matrix]
        List of R_n sparse response matrices
    N : int
        Number of sources
    T : int
        Number of wavelength bins
    verbose : bool
        Print progress during precomputation

    Returns
    -------
    C : np.ndarray, shape (N, T)
        First-order constants. C[n, :] = R_n^T @ Σ_n^(-1) @ y_n
    global_B_data : np.ndarray, shape (N, nnz_global)
        Dense matrix of aligned B_n data.
    global_indices : np.ndarray, shape (nnz_global,)
        CSR column indices for global template.
    global_indptr : np.ndarray, shape (T+1,)
        CSR row pointers for global template.

    Notes
    -----
    Memory comparison for T=2048, N=4114, nnz_global ~ 80k:

    Old (single-pass): B_list (~8 GB) + COO arrays (~8 GB) + concat (~8 GB)
                       + global_B_data (~2.7 GB) = ~27 GB additional peak
    New (two-pass):    pattern_mask (~4 MB) + reverse_map (~16 MB)
                       + global_B_data (~2.7 GB) = ~2.7 GB additional peak

    The two-pass approach recomputes B_n in pass 2 (doubling compute time)
    but reduces peak memory by ~24 GB.
    """
    report_interval = max(1, N // 20)

    # ── Pass 1: Compute C and build global non-zero pattern ──────────
    if verbose:
        print("  Pass 1/2: Computing C and building global template pattern...")

    C = np.zeros((N, T), dtype=np.float64)

    # Dense boolean mask for global B_n pattern: T² entries ≈ 4 MB for T=2048
    pattern_mask = np.zeros(T * T, dtype=bool)

    for n in range(N):
        if verbose and (n % report_interval == 0 or n == N - 1):
            pct = 100.0 * (n + 1) / N
            print(f"    [{n + 1:>5d}/{N}] ({pct:5.1f}%) Computing C[{n}] and B_n pattern...")

        D_n = sources_data[n]
        R_n = response_matrices[n]

        flux = D_n[:, 2]
        error = D_n[:, 3]
        precision = 1.0 / (error * error)

        # C_n = R_n^T @ Σ_n^(-1) @ y_n
        C[n, :] = R_n.T @ (precision * flux)

        # B_n = R_n^T @ diag(precision) @ R_n  (extract pattern only)
        R_n_weighted = R_n.multiply(precision[:, np.newaxis])
        B_n = (R_n.T @ R_n_weighted).tocsr()
        B_coo = B_n.tocoo()

        # Accumulate non-zero pattern into dense mask
        linear_idx = B_coo.row.astype(np.int64) * T + B_coo.col.astype(np.int64)
        pattern_mask[linear_idx] = True

        del B_n, R_n_weighted, B_coo

    # Extract and sort global non-zero pattern for CSR structure
    global_rc = np.where(pattern_mask)[0]
    del pattern_mask
    nnz_global = len(global_rc)

    global_rows = global_rc // T
    global_cols = global_rc % T

    # Sort by (row, col) for CSR format
    sort_order = np.lexsort((global_cols, global_rows))
    global_rows = global_rows[sort_order]
    global_cols = global_cols[sort_order]
    global_rc_sorted = global_rc[sort_order]
    del global_rc, sort_order

    # Build CSR indptr from row counts
    row_counts = np.bincount(global_rows, minlength=T)
    global_indptr = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)

    global_indices = global_cols.astype(np.int32)
    del global_cols

    if verbose:
        avg_bw = nnz_global / T if T > 0 else 0
        print(f"    Global template: {nnz_global:,} non-zeros, avg bandwidth ≈ {avg_bw:.0f} bins/row")

    # ── Pass 2: Recompute B_n and align to global template ───────────
    if verbose:
        print("  Pass 2/2: Recomputing B_n and aligning to global template...")

    # Reverse mapping: linear_index -> position in global_B_data columns
    # T² entries × 4 bytes ≈ 16 MB for T=2048
    reverse_map = np.full(T * T, -1, dtype=np.int32)
    reverse_map[global_rc_sorted] = np.arange(nnz_global, dtype=np.int32)
    del global_rc_sorted

    global_B_data = np.zeros((N, nnz_global), dtype=np.float64)

    for n in range(N):
        if verbose and (n % report_interval == 0 or n == N - 1):
            pct = 100.0 * (n + 1) / N
            print(f"    [{n + 1:>5d}/{N}] ({pct:5.1f}%) Aligning B_n data...")

        D_n = sources_data[n]
        R_n = response_matrices[n]

        flux = D_n[:, 2]
        error = D_n[:, 3]
        precision = 1.0 / (error * error)

        # Recompute B_n = R_n^T @ diag(precision) @ R_n
        R_n_weighted = R_n.multiply(precision[:, np.newaxis])
        B_n = (R_n.T @ R_n_weighted).tocsr()
        B_coo = B_n.tocoo()

        # Vectorized alignment via reverse mapping (replaces Python dict loop)
        linear_idx = B_coo.row.astype(np.int64) * T + B_coo.col.astype(np.int64)
        positions = reverse_map[linear_idx]
        valid = positions >= 0
        global_B_data[n, positions[valid]] = B_coo.data[valid]

        del B_n, R_n_weighted, B_coo

    del reverse_map

    if verbose:
        mem_C = C.nbytes / (1024**3)
        mem_B = global_B_data.nbytes / (1024**3)
        print(f"    Done. C = {mem_C:.2f} GB, global_B_data = {mem_B:.2f} GB")

    return C, global_B_data, global_indices, global_indptr


def compute_M_tensor(
    W: np.ndarray,
    global_B_data: np.ndarray,
    global_indices: np.ndarray,
    global_indptr: np.ndarray,
    K: int,
    T: int,
) -> Dict[Tuple[int, int], csr_matrix]:
    """
    Compute aggregation tensor M_jk = Σ_n w_nj * w_nk * B_n using BLAS.

    This function uses a single BLAS matrix multiplication to compute all
    M_jk blocks simultaneously, replacing the original O(N * K²) sparse
    matrix additions with O(K² * nnz_global) dense operations.

    Parameters
    ----------
    W : np.ndarray
        Current weight matrix, shape (N, K)
    global_B_data : np.ndarray
        Dense matrix of aligned B_n data, shape (N, nnz_global)
        From precompute_hals_constants()
    global_indices : np.ndarray
        CSR column indices for global template, shape (nnz_global,)
    global_indptr : np.ndarray
        CSR row pointers for global template, shape (T+1,)
    K : int
        Number of components
    T : int
        Number of wavelength bins

    Returns
    -------
    M : Dict[Tuple[int, int], csr_matrix]
        Aggregation tensor as dictionary mapping (j, k) -> sparse matrix.
        Only upper triangle (j <= k) is computed and stored.
        Access M[(j, k)] for j > k returns M[(k, j)] (symmetric).

    Notes
    -----
    The BLAS optimization works as follows:

    1. Build PairWeights matrix (n_pairs × N):
       PairWeights[i, n] = W[n, j] * W[n, k] for pair (j, k)

    2. Single BLAS matmul computes all weighted sums:
       All_M_data = PairWeights @ global_B_data
       Shape: (n_pairs, nnz_global)

    3. Reconstruct sparse matrices from results:
       M[(j, k)] = CSR from All_M_data[i, :] + global structure

    Performance: ~30x speedup over original sparse matrix loops.
    """
    N = W.shape[0]
    nnz_global = len(global_indices)

    # Build list of (j, k) pairs for upper triangle
    pairs = [(j, k) for j in range(K) for k in range(j, K)]
    n_pairs = len(pairs)

    # Build PairWeights matrix: shape (n_pairs, N)
    # PairWeights[i, :] = W[:, j] * W[:, k] for pair i = (j, k)
    PairWeights = np.empty((n_pairs, N), dtype=np.float64)
    for i, (j, k) in enumerate(pairs):
        PairWeights[i, :] = W[:, j] * W[:, k]

    # BLAS magic: single matrix multiplication computes all M_jk data
    # All_M_data[i, :] = Σ_n PairWeights[i, n] * global_B_data[n, :]
    All_M_data = PairWeights @ global_B_data  # Shape: (n_pairs, nnz_global)

    # Reconstruct sparse matrices from results
    M = {}
    for i, (j, k) in enumerate(pairs):
        M[(j, k)] = csr_matrix((All_M_data[i, :], global_indices, global_indptr), shape=(T, T))

    return M


def get_M_block(M: Dict[Tuple[int, int], csr_matrix], j: int, k: int) -> csr_matrix:
    """
    Get M_jk block from tensor, handling symmetry.

    Parameters
    ----------
    M : Dict
        Aggregation tensor (upper triangle stored)
    j, k : int
        Block indices

    Returns
    -------
    M_jk : csr_matrix
        The (j, k) block, transpose of (k, j) if j > k
    """
    if j <= k:
        return M[(j, k)]
    else:
        # Return transpose for lower triangle access
        return M[(k, j)].T


def compute_regularization_gradient_single(
    v_k: np.ndarray,
    k: int,
    V: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    normalize: bool,
    T: int,
    K: int,
    reg_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute regularization gradient for a single component v_k.

    Handles:
    - L2: 2 * alpha * v_k
    - First-order smoothness: β gradient component
    - Second-order smoothness: γ gradient component

    Parameters
    ----------
    v_k : np.ndarray
        Single component vector, shape (T,)
    k : int
        Component index (not used, for API consistency)
    V : np.ndarray
        Full V matrix (needed for smoothness computation), shape (T, K)
    alpha, beta, gamma : float
        Regularization coefficients
    normalize : bool
        Whether to normalize regularization terms
    T, K : int
        Dimensions
    reg_scale : float
        Scaling factor to match data term gradient scaling.
        In ALS-WNMF: reg_scale = M_total / 2.0 when normalize=True

    Returns
    -------
    grad_reg : np.ndarray
        Regularization gradient for component k, shape (T,)

    Notes
    -----
    The gradient formulas match those in als_wnmf.py m_step():
    - L2: 2 * alpha * v_k * reg_scale
    - First-order smoothness: β gradient component * reg_scale
    - Second-order smoothness: γ gradient component * reg_scale

    The reg_scale factor is critical for matching the loss function scaling.
    Without it, regularization can dominate or be negligible.
    """
    grad_reg = np.zeros(T, dtype=np.float64)

    # Normalization factors
    norm_l2 = 1.0 / (T * K) if normalize else 1.0
    norm_first = 1.0 / ((T - 1) * K) if normalize else 1.0
    norm_second = 1.0 / ((T - 2) * K) if normalize else 1.0

    # L2 gradient: ∂/∂v_k (alpha * ||V||²_F) = 2 * alpha * v_k
    if alpha > 0:
        grad_reg += 2.0 * alpha * norm_l2 * v_k * reg_scale

    # First-order smoothness gradient
    # The gradient involves neighbors: for interior points
    # ∂C/∂v_t = 2(v_t - v_{t-1}) - 2(v_{t+1} - v_t)
    #          = 4*v_t - 2*v_{t-1} - 2*v_{t+1}
    if beta > 0:
        grad_smooth = np.zeros(T, dtype=np.float64)

        # Interior points
        grad_smooth[1:-1] = 4.0 * v_k[1:-1] - 2.0 * v_k[:-2] - 2.0 * v_k[2:]

        # Boundary conditions (matching als_wnmf.py)
        grad_smooth[0] = 2.0 * (v_k[0] - v_k[1])
        grad_smooth[-1] = 2.0 * (v_k[-1] - v_k[-2])

        grad_reg += beta * norm_first * grad_smooth * reg_scale

    # Second-order smoothness gradient
    # ∂C/∂v_t = 12*v_t + 2*v_{t-2} + 2*v_{t+2} - 8*v_{t-1} - 8*v_{t+1}
    if gamma > 0:
        grad_curv = np.zeros(T, dtype=np.float64)

        # Interior points: 2 <= t <= T-3
        if T > 4:
            grad_curv[2:-2] = 12.0 * v_k[2:-2] + 2.0 * v_k[:-4] + 2.0 * v_k[4:] - 8.0 * v_k[1:-3] - 8.0 * v_k[3:-1]

        # Boundary t=1: simplified stencil
        if T > 3:
            grad_curv[1] = 10.0 * v_k[1] + 2.0 * v_k[3] - 4.0 * v_k[0] - 8.0 * v_k[2]

        # Boundary t=T-2: simplified stencil
        if T > 3:
            grad_curv[-2] = 10.0 * v_k[-2] + 2.0 * v_k[-4] - 4.0 * v_k[-1] - 8.0 * v_k[-3]

        # Boundaries t=0 and t=T-1: no second-order penalty (match loss function)
        grad_curv[0] = 0.0
        grad_curv[-1] = 0.0

        grad_reg += gamma * norm_second * grad_curv * reg_scale

    return grad_reg


def compute_hessian_diagonal_single(
    M_kk: csr_matrix,
    v_k: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    normalize: bool,
    T: int,
    K: int,
    reg_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute Hessian diagonal for a single component.

    diag(H) = diag(M_kk) + regularization terms

    Parameters
    ----------
    M_kk : csr_matrix
        Block (k, k) of aggregation tensor, shape (T, T)
    v_k : np.ndarray
        Component vector (for regularization), shape (T,)
    alpha, beta, gamma : float
        Regularization coefficients
    normalize : bool
        Whether to normalize
    T, K : int
        Dimensions
    reg_scale : float
        Scaling factor to match data term Hessian scaling.
        In ALS-WNMF: reg_scale = M_total / 2.0 when normalize=True

    Returns
    -------
    diag_H : np.ndarray
        Hessian diagonal, shape (T,)
    """
    # Extract diagonal of M_kk using Majorization-Minimization to guarantee safe step size
    diag_H = np.asarray(M_kk.sum(axis=1)).flatten()

    # Normalization factors
    norm_l2 = 1.0 / (T * K) if normalize else 1.0
    norm_first = 1.0 / ((T - 1) * K) if normalize else 1.0
    norm_second = 1.0 / ((T - 2) * K) if normalize else 1.0

    # L2 contribution to Hessian: 2 * alpha
    if alpha > 0:
        diag_H += 2.0 * alpha * norm_l2 * reg_scale

    # First-order smoothness contribution
    # True Hessian diagonal is 4.0, but for PGD stability we need a Majorizer.
    # Max row sum of absolute values of smoothness Hessian is 4 + |-2| + |-2| = 8.0.
    if beta > 0:
        diag_smooth = np.zeros(T, dtype=np.float64)
        diag_smooth[1:-1] = 8.0 * beta * norm_first * reg_scale
        diag_smooth[0] = 4.0 * beta * norm_first * reg_scale
        diag_smooth[-1] = 4.0 * beta * norm_first * reg_scale
        diag_H += diag_smooth

    # Second-order smoothness contribution
    # True Hessian diagonal is 12.0, but max eigenvalue (or max abs row sum) is
    # 12 + |-8| + |-8| + |2| + |2| = 32.0. To guarantee safe PGD steps without explosion:
    if gamma > 0:
        diag_curv = np.zeros(T, dtype=np.float64)
        if T > 4:
            diag_curv[2:-2] = 32.0 * gamma * norm_second * reg_scale
        if T > 3:
            diag_curv[1] = 24.0 * gamma * norm_second * reg_scale
            diag_curv[-2] = 24.0 * gamma * norm_second * reg_scale
        # t=0 and t=T-1: no contribution
        diag_H += diag_curv

    # Ensure positive diagonal for numerical stability
    diag_H = np.maximum(diag_H, 1e-12)

    return diag_H


def m_step_hals(
    V: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
    M: Dict[Tuple[int, int], csr_matrix],
    alpha: float,
    beta: float,
    gamma: float,
    normalize: bool,
    reg_scale: float,
    n_inner_iter: int = 5,
    fix_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """
    HALS M-step: Sequential update of each V_k component.

    For each component k:
    1. Compute target U_k = P[:, k] - Σ_{j≠k} M_jk @ V[:, j]
    2. Compute Hessian diagonal
    3. Inner loop: projected gradient descent with non-negativity

    Parameters
    ----------
    V : np.ndarray
        Current basis matrix, shape (T, K)
    W : np.ndarray
        Current weight matrix, shape (N, K)
    C : np.ndarray
        Precomputed first-order constants, shape (N, T)
    M : Dict
        Aggregation tensor (computed by compute_M_tensor)
    alpha, beta, gamma : float
        Regularization coefficients
    normalize : bool
        Whether to normalize regularization
    reg_scale : float
        Scaling factor for regularization gradients.
        Must match the scaling used in compute_regularized_loss.
        In ALS-WNMF: reg_scale = M_total / 2.0 when normalize=True
    n_inner_iter : int
        Number of inner PGD iterations per component
    fix_mask : np.ndarray, optional
        Boolean mask of shape (K,), where fix_mask[k] = True means
        component k is frozen (not updated). If None (default), all
        components are updated normally (original behavior).
        Used for incremental training with fixed galaxy basis.

    Returns
    -------
    V_new : np.ndarray
        Updated basis matrix, shape (T, K)
    t_compute : float
        Computation time in seconds

    Notes
    -----
    The key difference from MUR is that each component is updated
    sequentially against the residual left by other components.
    This naturally breaks symmetry and prevents "twin spectra".

    When fix_mask is provided, frozen components are skipped entirely.
    They still contribute to the residual computation for other components
    (via the U_k subtraction), but their own V[:, k] values are unchanged.
    """
    t_start = time.perf_counter()

    T, K = V.shape
    V_new = V.copy()

    # Default: all components are free (backward compatible)
    if fix_mask is None:
        fix_mask = np.zeros(K, dtype=bool)

    # Precompute P = C^T @ W (shape: T x K)
    P = C.T @ W  # (T, N) @ (N, K) = (T, K)

    # Sequential update of each component
    for k in range(K):
        # Skip frozen components (e.g., fixed galaxy basis)
        if fix_mask[k]:
            continue

        v_k = V_new[:, k].copy()

        # Step 1: Compute target U_k = P[:, k] - Σ_{j≠k} M_jk @ V[:, j]
        U_k = P[:, k].copy()
        for j in range(K):
            if j != k:
                M_jk = get_M_block(M, j, k)
                U_k -= M_jk @ V_new[:, j]

        # Step 2: Compute Hessian diagonal
        M_kk = get_M_block(M, k, k)
        diag_H = compute_hessian_diagonal_single(M_kk, v_k, alpha, beta, gamma, normalize, T, K, reg_scale)

        # Step 3: Inner loop - projected gradient descent
        for _ in range(n_inner_iter):
            # Compute gradient: grad = M_kk @ v_k - U_k + grad_reg
            M_kk_vk = M_kk @ v_k
            grad_reg = compute_regularization_gradient_single(
                v_k, k, V_new, alpha, beta, gamma, normalize, T, K, reg_scale
            )
            grad = M_kk_vk - U_k + grad_reg

            # Update with projected gradient descent
            v_k = v_k - grad / diag_H

            # Project to non-negative orthant
            v_k = np.maximum(v_k, 0.0)

        V_new[:, k] = v_k

    t_compute = time.perf_counter() - t_start
    return V_new, t_compute


def hals_wnmf(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V_init: np.ndarray,
    W_init: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
    tol: float = 1e-5,
    max_iter: int = 100,
    n_jobs: int = -1,
    n_inner_iter: int = 5,
    e_step_method: str = "numba",
    warm_start: bool = True,
    verbose: bool = True,
    fix_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Main HALS-WNMF algorithm with BLAS-optimized M tensor computation.

    Workflow:
    1. Precompute C and global_B_data (once)
    2. For each iteration:
       a. E-step: Update W (reuse e_step from als_wnmf.py)
       b. Compute M tensor via BLAS matmul
       c. M-step HALS: Sequential V_k updates (respecting fix_mask)
       d. Check convergence

    Parameters
    ----------
    sources_data : List[np.ndarray]
        List of D_n arrays for all N sources
    response_matrices : List[csr_matrix]
        List of pre-computed R_n matrices
    V_init : np.ndarray
        Initial basis matrix (from pruning), shape (T, K)
    W_init : np.ndarray
        Initial weight matrix (from pruning), shape (N, K)
    alpha : float
        L2 regularization coefficient (default: 0.0)
    beta : float
        First-order smoothness coefficient (default: 0.0)
    gamma : float
        Second-order smoothness coefficient (default: 0.0)
    normalize : bool
        Whether to normalize regularization terms (default: True)
    tol : float
        Convergence tolerance for relative loss change (default: 1e-5)
    max_iter : int
        Maximum number of iterations (default: 100)
    n_jobs : int
        Number of parallel jobs for E-step (-1 uses all CPUs)
    n_inner_iter : int
        Number of inner PGD iterations per component (default: 5)
    e_step_method : str
        NNLS solver for E-step: 'scipy' or 'numba' (default: 'numba')
    warm_start : bool
        Use previous W as initial guess for E-step (default: True)
    verbose : bool
        Print iteration progress (default: True)
    fix_mask : np.ndarray, optional
        Boolean mask of shape (K,), where fix_mask[k] = True means
        component k is frozen during M-step (not updated).
        If None (default), all components are updated normally.
        Used for incremental training: fix galaxy basis, learn AGN components.

    Returns
    -------
    V : np.ndarray
        Final basis matrix, shape (T, K), non-negative
    W : np.ndarray
        Final weight matrix, shape (N, K), non-negative
    loss_history : List[float]
        Loss at each iteration (including initial)

    Notes
    -----
    HALS addresses the "twin spectra" problem of ALS-WNMF by updating
    each component sequentially against the residual left by others.
    This naturally breaks symmetry and produces more independent components.

    The BLAS optimization for M tensor computation achieves ~30x speedup
    by replacing sparse matrix additions with a single dense matmul.

    The algorithm guarantees:
    - Monotonic decrease (or no increase) in loss
    - Non-negativity of V and W at all times
    - Convergence to a local minimum

    Typical usage:
    1. Run ALS-WNMF to get initial V, W
    2. Prune dictionary to remove twin spectra
    3. Run HALS with pruned V_init, W_init for fine-tuning

    For incremental AGN training:
    1. Set V_init = [V_galaxy_fix | V_agn_random]  (concatenated)
    2. Set fix_mask = [True, ..., True, False, ..., False]
    3. Galaxy basis columns are frozen; AGN columns are learned
    """
    N = len(sources_data)
    T, K = V_init.shape

    if verbose:
        print("=" * 70)
        print("HALS-WNMF: Hierarchical Alternating Least Squares (BLAS-optimized)")
        print("=" * 70)
        print(f"Sources: {N}, Components: {K}, Wavelength bins: {T}")
        print(f"Max iterations: {max_iter}, Tolerance: {tol}")
        print(f"Inner iterations: {n_inner_iter}")
        print(f"Regularization: α={alpha}, β={beta}, γ={gamma} (normalize={normalize})")
        if fix_mask is not None:
            n_fixed = int(np.sum(fix_mask))
            n_free = K - n_fixed
            print(f"Fixed components: {n_fixed}, Free components: {n_free}")
        print()

    # Step 1: Precompute constants (once)
    if verbose:
        print("Precomputing constants (C, global_B_data)...")

    t_precompute_start = time.perf_counter()
    C, global_B_data, global_indices, global_indptr = precompute_hals_constants(
        sources_data, response_matrices, N, T, verbose=verbose
    )
    t_precompute = time.perf_counter() - t_precompute_start

    nnz_global = len(global_indices)

    if verbose:
        print(f"  C matrix: {C.shape}")
        print(f"  global_B_data: {global_B_data.shape} (N × nnz_global)")
        print(f"  Global template nnz: {nnz_global:,}")
        print(f"  Precompute time: {t_precompute:.2f}s")
        print()

    # Initialize from pruned results
    V = V_init.copy()
    W = W_init.copy()

    # Compute regularization scaling factor to match loss function
    # This is critical for proper gradient scaling
    M_total = sum(D.shape[0] for D in sources_data)
    reg_scale = (M_total / 2.0) if normalize else 0.5

    if verbose:
        print(f"  M_total = {M_total:,}, reg_scale = {reg_scale:.2e}")

    # Validate initial non-negativity
    v_valid, w_valid = validate_non_negativity(V, W)
    if not (v_valid and w_valid):
        raise ValueError("Initial V or W contains negative values!")

    # Compute initial loss
    loss, chi2, l2_term, smooth_term, curv_term = compute_regularized_loss(
        sources_data, response_matrices, V, W, alpha, beta, gamma, normalize
    )
    loss_history = [loss]

    if verbose:
        print(f"Initial Loss = {loss:.6e} (χ² = {chi2:.6e})")
        print("-" * 70)

    # Main HALS loop
    for iteration in range(1, max_iter + 1):
        # E-step: Update W (fix V)
        t_e_start = time.perf_counter()
        W_new, _ = e_step(
            sources_data,
            response_matrices,
            V,
            n_jobs=n_jobs,
            verbose=False,
            method=e_step_method,
            W_prev=W if warm_start else None,
        )
        t_e_step = time.perf_counter() - t_e_start

        # Validate non-negativity after E-step
        _, w_valid = validate_non_negativity(V, W_new)
        if not w_valid:
            raise RuntimeError("E-step produced negative weights!")

        # Compute M tensor from updated W using BLAS
        t_m_tensor_start = time.perf_counter()
        M = compute_M_tensor(W_new, global_B_data, global_indices, global_indptr, K, T)
        t_m_tensor = time.perf_counter() - t_m_tensor_start

        # M-step HALS: Update V (fix W), respecting fix_mask
        V_new, t_m_step = m_step_hals(
            V, W_new, C, M, alpha, beta, gamma, normalize, reg_scale, n_inner_iter,
            fix_mask=fix_mask,
        )

        # Validate non-negativity after M-step
        v_valid, _ = validate_non_negativity(V_new, W_new)
        if not v_valid:
            raise RuntimeError("M-step produced negative spectra!")

        # Compute loss
        loss_new, chi2_new, l2_new, smooth_new, curv_new = compute_regularized_loss(
            sources_data, response_matrices, V_new, W_new, alpha, beta, gamma, normalize
        )

        # Check convergence
        delta_loss = loss - loss_new
        rel_change = abs(delta_loss) / loss_new if loss_new > 0 else 0

        # Update
        V, W, loss = V_new, W_new, loss_new
        chi2, l2_term, smooth_term, curv_term = chi2_new, l2_new, smooth_new, curv_new
        loss_history.append(loss)

        # Print progress
        if verbose:
            has_reg = (alpha > 0) or (beta > 0) or (gamma > 0)
            reg_str = ""
            if has_reg:
                parts = []
                if alpha > 0:
                    parts.append(f"L2={l2_term:.4e}")
                if beta > 0:
                    parts.append(f"smooth={smooth_term:.4e}")
                if gamma > 0:
                    parts.append(f"curv={curv_term:.4e}")
                reg_str = f", {', '.join(parts)}"

            print(
                f"Iter {iteration:3d}: "
                f"E={t_e_step:.2f}s, M_tensor={t_m_tensor:.2f}s, M_step={t_m_step:.2f}s | "
                f"Loss={loss:.6e} (χ²={chi2:.6e}{reg_str}), "
                f"Δ={delta_loss:.4e} ({rel_change:.2e})"
            )

        # Check convergence
        if rel_change < tol:
            if verbose:
                print("-" * 70)
                print(f"Converged after {iteration} iterations!")
                print(f"Final Loss = {loss:.6e} (χ² = {chi2:.6e})")
            break
    else:
        if verbose:
            print("-" * 70)
            print(f"Reached maximum iterations ({max_iter})")
            print(f"Final Loss = {loss:.6e} (χ² = {chi2:.6e})")

    return V, W, loss_history
