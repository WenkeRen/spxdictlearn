"""
Alternating Least Squares Weighted Non-Negative Matrix Factorization (ALS-WNMF)

This module implements the core ALS-WNMF algorithm for extracting physically
meaningful spectra from sparse, unaligned, and noisy observational data.

Algorithm structure:
- E-step: Update weights W using NNLS (parallelizable)
- M-step: Update basis V using Multiplicative Update Rules (MUR)
- Iterate until convergence

Reference: StartUp.md specification document
"""

import time
from typing import List, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.sparse import csr_matrix

from .numba_nnls import NUMBA_AVAILABLE, nnls_pgd, nnls_pgd_fallback
from .utils import (
    compute_global_chi2,
    compute_second_order_smoothness_penalty,
    compute_smoothness_penalty,
    initialize_parameters,
    validate_non_negativity,
)


def e_step_single_source(
    D_n: np.ndarray,
    R_n: csr_matrix,
    V: np.ndarray,
    method: str = "scipy",
    x_init: np.ndarray | None = None,
) -> np.ndarray:
    """
    E-step: Solve for weights w_n of a single source using NNLS.

    This solves: min_{w_n >= 0} || A_n @ w_n - b_n ||_2^2
    where A_n = (R_n @ V) / σ_n[:, newaxis]
          b_n = y_n / σ_n

    Parameters
    ----------
    D_n : np.ndarray
        Raw observation data, shape (M_n, 4)
        Column 2: flux y_n, Column 3: error σ_n
    R_n : scipy.sparse.csr_matrix
        Response matrix, shape (M_n, T)
    V : np.ndarray
        Global basis matrix (fixed during E-step), shape (T, K)
    method : str
        NNLS solver: 'scipy' (default) or 'numba'.
        - 'scipy': uses scipy.optimize.nnls (does not release GIL)
        - 'numba': uses Numba PGD (nogil=True, enables true parallelism)
    x_init : np.ndarray, optional
        Warm-start initial guess, shape (K,). Only used when method='numba'.
        Passing the previous iteration's w_n typically reduces inner iterations.

    Returns
    -------
    w_n : np.ndarray
        Weight vector for this source, shape (K,), non-negative
    """
    # Extract flux and error
    flux = D_n[:, 2]
    error = D_n[:, 3]

    # Compute precision-weighted observation: b = y / σ
    b = flux / error

    # Compute design matrix: A = (R @ V) / σ[:, newaxis]
    U = R_n @ V  # Sparse × Dense → Dense, shape (M_n, K)
    A = U / error[:, np.newaxis]  # Row-wise scaling, shape (M_n, K)

    # Solve NNLS with selected backend
    if method == "numba":
        if NUMBA_AVAILABLE:
            w_n, _ = nnls_pgd(A, b, x_init)
        else:
            # Fallback to pure NumPy PGD when Numba is unavailable
            w_n, _ = nnls_pgd_fallback(A, b)
    else:  # "scipy" (default)
        w_n, _ = nnls(A, b)

    return w_n


def e_step(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V: np.ndarray,
    n_jobs: int = -1,
    verbose: bool = True,
    method: str = "scipy",
    W_prev: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """
    E-step: Update weights W for all sources in parallel.

    For each source n, solve independent NNLS problem:
    min_{w_n >= 0} || A_n @ w_n - b_n ||_2^2

    Parameters
    ----------
    sources_data : list of np.ndarray
        List of D_n arrays for all sources
    response_matrices : list of csr_matrix
        List of R_n matrices for all sources
    V : np.ndarray
        Global basis matrix (fixed), shape (T, K)
    n_jobs : int
        Number of parallel jobs (-1 uses all CPUs, default: -1)
    verbose : bool
        Show progress bar
    method : str
        NNLS solver backend: 'scipy' (default) or 'numba'.
        - 'scipy': scipy.optimize.nnls; holds GIL, limiting thread parallelism
        - 'numba': Numba PGD with nogil=True; enables true thread parallelism
    W_prev : np.ndarray, optional
        Previous iteration weights, shape (N, K). When provided with
        method='numba', each source is warm-started from its prior solution,
        reducing the number of inner PGD iterations needed.

    Returns
    -------
    W : np.ndarray
        Updated weight matrix, shape (N, K), non-negative
    t_parallel : float
        Time spent in parallel computation (seconds)
    """
    N = len(sources_data)

    # Parallel computation of weights
    if verbose:
        backend_info = (
            f"numba-pgd ({'GIL-free' if NUMBA_AVAILABLE else 'fallback'})" if method == "numba" else "scipy-nnls"
        )
        warm_info = ", warm-start" if (W_prev is not None and method == "numba") else ""
        print(f"E-step: Solving NNLS for {N} sources [{backend_info}{warm_info}]...")

    t_start = time.perf_counter()
    weights_list = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(e_step_single_source)(
            D_n,
            R_n,
            V,
            method=method,
            x_init=W_prev[n] if W_prev is not None else None,
        )
        for n, (D_n, R_n) in enumerate(zip(sources_data, response_matrices))
    )
    t_parallel = time.perf_counter() - t_start

    # Stack into weight matrix
    W = np.vstack(weights_list)  # Shape (N, K)

    return W, t_parallel


def m_step_single_source(
    D_n: np.ndarray,
    R_n: csr_matrix,
    V: np.ndarray,
    w_n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute single source contribution to M-step accumulators.

    This function extracts the per-source computation from the M-step loop,
    enabling parallelization across sources using joblib.Parallel.

    Parameters
    ----------
    D_n : np.ndarray
        Raw observation data, shape (M_n, 4)
        Column 2: flux y_n, Column 3: error σ_n
    R_n : scipy.sparse.csr_matrix
        Response matrix, shape (M_n, T)
    V : np.ndarray
        Global basis matrix (fixed during M-step), shape (T, K)
    w_n : np.ndarray
        Weight vector for this source, shape (K,)

    Returns
    -------
    P_n : np.ndarray
        Numerator contribution: (R_n^T @ s_n) ⊗ w_n^T, shape (T, K)
    Q_n : np.ndarray
        Denominator contribution: (R_n^T @ q_n) ⊗ w_n^T, shape (T, K)

    Notes
    -----
    The contributions follow the MUR derivation:
    - s_n = y_n / σ_n² (precision-weighted observation)
    - q_n = ŷ_n / σ_n² (precision-weighted model prediction)
    - ŷ_n = R_n @ V @ w_n (model prediction)
    """
    # Extract flux and error
    flux = D_n[:, 2]
    error = D_n[:, 3]

    # Compute precision-weighted observation: s = y / σ²
    s = flux / (error * error)

    # Compute model prediction: ŷ = R @ V @ w
    y_pred = R_n @ (V @ w_n)

    # Compute precision-weighted prediction: q = ŷ / σ²
    q = y_pred / (error * error)

    # Compute numerator contribution: P_n = (R^T @ s) ⊗ w^T
    RnT_s = R_n.T @ s  # Sparse.T @ Dense_Vector → Dense, shape (T,)
    P_n = np.outer(RnT_s, w_n)

    # Compute denominator contribution: Q_n = (R^T @ q) ⊗ w^T
    RnT_q = R_n.T @ q  # Sparse.T @ Dense_Vector → Dense, shape (T,)
    Q_n = np.outer(RnT_q, w_n)

    return P_n, Q_n


def m_step(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V: np.ndarray,
    W: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
    epsilon_reg: float = 1e-12,
    n_jobs: int = -1,
    chunk_size: int = 500,
) -> Tuple[np.ndarray, float]:
    """
    M-step: Update global basis V using Multiplicative Update Rules (MUR).

    MUR guarantees non-negativity and monotonic convergence:
    V_new = V_old ⊙ [(P + P_reg) / (Q + Q_reg + ε)]

    where P and Q are accumulated over all sources:
    P += (R_n^T @ s_n) ⊗ w_n^T
    Q += (R_n^T @ q_n) ⊗ w_n^T

    with s_n = y_n / σ_n^2 and q_n = ŷ_n / σ_n^2

    Regularization terms (with optional normalization):
    - Q_reg += 2*α*n_0*V (L2 regularization, from gradient of ||V||²_F)
    - Q_reg += 4*β*n_1*V (First-order smoothness, positive part of gradient)
    - P_reg += 2*β*n_1*(V_prev + V_next) (First-order smoothness, negative part)
    - Q_reg += 2*γ*n_2*Q_2nd (Second-order smoothness, positive part)
    - P_reg += 2*γ*n_2*P_2nd (Second-order smoothness, negative part)

    Normalization factors (when normalize=True):
    - n_0 = 1/(T*K) for L2
    - n_1 = 1/((T-1)*K) for first-order smoothness
    - n_2 = 1/((T-2)*K) for second-order smoothness

    This implementation uses array slicing for efficient computation of the
    smoothness gradient without constructing explicit matrices.

    Parameters
    ----------
    sources_data : list of np.ndarray
        List of D_n arrays for all sources
    response_matrices : list of csr_matrix
        List of R_n matrices for all sources
    V : np.ndarray
        Current basis matrix, shape (T, K)
    W : np.ndarray
        Fixed weight matrix, shape (N, K)
    alpha : float
        L2 regularization coefficient (default: 0.0, no regularization).
        When normalized, typical value: 0.001 - 0.1
    beta : float
        First-order smoothness regularization coefficient (default: 0.0).
        Penalizes first-order differences: Σ(V_{t+1} - V_t)².
        When normalized, typical value: 0.01 - 1.0
    gamma : float
        Second-order smoothness regularization coefficient (default: 0.0).
        Penalizes curvature (discrete Laplacian squared).
        When normalized, typical value: 0.01 - 1.0
    normalize : bool
        If True (default), apply normalization to regularization terms to make
        alpha, beta, gamma dimensionless and consistent across problem sizes.
    epsilon_reg : float
        Small regularization to prevent division by zero (default: 1e-12)
    n_jobs : int
        Number of parallel jobs for source contribution computation
        (default: -1, all CPUs). Parallelization uses reduction pattern.
    chunk_size : int
        Number of sources per chunk for incremental reduction (default: 500).
        Reduces peak memory from O(N*T*K) to O(chunk_size*T*K) by processing
        sources in chunks and freeing each chunk's results immediately.
        Algorithm result is identical regardless of chunk_size.

    Returns
    -------
    V_new : np.ndarray
        Updated basis matrix, shape (T, K), non-negative
    t_parallel : float
        Time spent in parallel computation (seconds)

    Notes
    -----
    The first-order smoothness gradient is computed using array slicing:
    - V_neighbor_sum[1:-1] = V[0:-2] + V[2:] (inner points)
    - V_neighbor_sum[0] = V[0] + V[1] (top boundary, Neumann)
    - V_neighbor_sum[-1] = V[-2] + V[-1] (bottom boundary, Neumann)

    The second-order smoothness gradient uses:
    - Interior points (2 <= t <= T-3):
      P_2nd[t] = 8(V_{t-1} + V_{t+1}), Q_2nd[t] = 12V_t + 2(V_{t-2} + V_{t+2})
    - Boundary points t=1, t=T-2: reduced stencil (fewer neighbors)
    - Boundary points t=0, t=T-1: NO regularization (matches loss function)

    Performance Optimization:
    - Uses joblib.Parallel with threading backend for parallel computation
    - Reduction pattern: each worker computes P_n, Q_n, then sums at the end
    - scipy.sparse operations release GIL, enabling true parallelism
    """
    N = len(sources_data)
    T, K = V.shape

    # Scale regularizers to match the objective function in compute_regularized_loss.
    # The data term gradients P and Q correspond to 1/2 * chi2_raw. To align with the
    # evaluated loss, we must scale the regularization gradients appropriately.
    M_total = sum(D.shape[0] for D in sources_data)
    reg_scale = (M_total / 2.0) if normalize else 0.5

    # Normalization factors (makes alpha, beta, gamma dimensionless)
    norm_l2 = 1.0 / (T * K) if normalize else 1.0
    norm_first = 1.0 / ((T - 1) * K) if normalize else 1.0
    norm_second = 1.0 / ((T - 2) * K) if normalize else 1.0

    # Parallel computation of source contributions with chunked reduction.
    # Processing in chunks avoids materializing all contributions simultaneously,
    # reducing peak memory from O(N * T * K) to O(chunk_size * T * K).
    # Algorithm result is identical: P and Q are the same sum, just accumulated incrementally.
    P = np.zeros((T, K))
    Q = np.zeros((T, K))
    t_parallel_start = time.perf_counter()

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_contributions = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(m_step_single_source)(
                sources_data[n], response_matrices[n], V, W[n, :]
            )
            for n in range(chunk_start, chunk_end)
        )
        # Incremental reduction: accumulate and free each chunk immediately
        for P_n, Q_n in chunk_contributions:
            P += P_n
            Q += Q_n

    t_parallel = time.perf_counter() - t_parallel_start

    # Apply L2 regularization to denominator (gradient of ||V||²_F is 2*α*V)
    # We use 2*α*V to match the multiplicative update rule derivation
    # Skip computation when alpha=0 for performance
    if alpha > 0:
        Q += 2 * alpha * norm_l2 * V * reg_scale

    # Apply first-order smoothness regularization using array slicing
    if beta > 0:
        # Compute V_neighbor_sum = V_{t-1} + V_{t+1}
        # This corresponds to the negative part of the smoothness gradient
        V_neighbor_sum = np.zeros_like(V)

        # Inner points: V[1:-1] gets V[0:-2] + V[2:]
        V_neighbor_sum[1:-1, :] = V[0:-2, :] + V[2:, :]

        # Boundaries (Dirichlet condition: V[0] pulled toward V[1], V[-1] pulled toward V[-2])
        # This prevents edge pixel amplitude elevation by constraining boundary pixels
        # to follow their inner neighbors, rather than allowing them to grow freely.
        V_neighbor_sum[0, :] = 2 * V[1, :]  # Top boundary: V[0] is pulled toward V[1]
        V_neighbor_sum[-1, :] = 2 * V[-2, :]  # Bottom boundary: V[-1] is pulled toward V[-2]

        # Apply smoothness gradient to MUR:
        # Numerator gets: 2*β*V_neighbor_sum (negative part of gradient)
        # Denominator gets: 4*β*V (positive part of gradient)
        P += 2 * beta * norm_first * V_neighbor_sum * reg_scale
        Q += 4 * beta * norm_first * V * reg_scale

    # Apply second-order smoothness regularization (gamma)
    # Penalty: Σ_k Σ_t (V_{t+1} - 2*V_t + V_{t-1})²
    # Gradient: 2 * L_2^T @ L_2 @ V where L_2 is second-difference operator
    if gamma > 0:
        # For MUR, split gradient into P (numerator) and Q (denominator)
        # Interior: ∇ = 2[6V_t + V_{t-2} + V_{t+2} - 4V_{t-1} - 4V_{t+1}]
        #   P_t = 4(V_{t-1} + V_{t+1}), Q_t = 6V_t + V_{t-2} + V_{t+2}
        P_2nd = np.zeros_like(V)
        Q_2nd = np.zeros_like(V)

        # Interior points: 2 <= t <= T-3
        # Gradient of C = Σ_t (V_{t+1} - 2V_t + V_{t-1})² is:
        #   ∂C/∂V_t = 12V_t + 2V_{t-2} + 2V_{t+2} - 8V_{t-1} - 8V_{t+1}
        # MUR decomposition: P gets negative terms (8V_{t-1} + 8V_{t+1}),
        #                    Q gets positive terms (12V_t + 2V_{t-2} + 2V_{t+2})
        if T > 4:
            P_2nd[2:-2, :] = 8 * (V[1:-3, :] + V[3:-1, :])  # Negative: 8(V_{t-1} + V_{t+1})
            Q_2nd[2:-2, :] = 12 * V[2:-2, :] + 2 * (V[:-4, :] + V[4:, :])  # Positive: 12V_t + 2(V_{t-2} + V_{t+2})

        # Boundary t=1: P_1 = 4(V_0 + V_2), Q_1 = 5V_1 + V_3
        if T > 3:
            P_2nd[1, :] = 4 * (V[0, :] + V[2, :])
            Q_2nd[1, :] = 5 * V[1, :] + V[3, :]

        # Boundary t=T-2: P_{T-2} = 4(V_{T-1} + V_{T-3}), Q_{T-2} = 5V_{T-2} + V_{T-4}
        if T > 3:
            P_2nd[-2, :] = 4 * (V[-1, :] + V[-3, :])
            Q_2nd[-2, :] = 5 * V[-2, :] + V[-4, :]

        # Boundary t=0 and t=T-1: NO second-order smoothness regularization
        # The loss function compute_second_order_smoothness_penalty only computes
        # interior points (V[1] to V[T-2]), so we set boundary gradients to zero
        # to ensure M-step gradient matches the loss function.
        P_2nd[0, :] = 0.0
        Q_2nd[0, :] = 0.0
        P_2nd[-1, :] = 0.0
        Q_2nd[-1, :] = 0.0

        # Apply to MUR accumulators
        P += 2 * gamma * norm_second * P_2nd * reg_scale
        Q += 2 * gamma * norm_second * Q_2nd * reg_scale

    # Multiplicative update: V_new = V ⊙ (P / (Q + ε))
    V_new = V * (P / (Q + epsilon_reg))

    # Project onto the non-negative orthant.  Low-S/N measurements can be negative
    # (noise realizations), which drives isolated numerator entries P < 0 and
    # produces tiny negative V values.  NMF requires V >= 0; the HALS path already
    # enforces this through its inner PGD (max(0, .)), so the MUR path needs the
    # explicit projection here.
    V_new = np.maximum(V_new, 0.0)

    return V_new, t_parallel


def compute_regularized_loss(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V: np.ndarray,
    W: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
) -> Tuple[float, float, float, float, float]:
    """
    Compute the regularized objective function with L2, smoothness, and curvature penalties on V.

    Loss = χ²/M_total + α/(TK) ||V||²_F + β/((T-1)K) · Smoothness + γ/((T-2)K) · Curvature

    where:
    - χ²/M_total is the reduced chi-squared (normalized by total observations)
    - ||V||²_F is the Frobenius norm squared of V (sum of squares)
    - Smoothness = Σ_k Σ_t (V_{t+1,k} - V_{t,k})² (first-order differences)
    - Curvature = Σ_k Σ_t (V_{t+1,k} - 2V_{t,k} + V_{t-1,k})² (second-order differences)
    - α is the L2 regularization coefficient
    - β is the first-order smoothness regularization coefficient
    - γ is the second-order smoothness (curvature) regularization coefficient

    Parameters
    ----------
    sources_data : list of np.ndarray
        List of D_n arrays for all sources
    response_matrices : list of csr_matrix
        List of R_n matrices for all sources
    V : np.ndarray
        Global basis matrix, shape (T, K)
    W : np.ndarray
        Weight matrix, shape (N, K)
    alpha : float
        L2 regularization coefficient (default: 0.0, no regularization)
    beta : float
        First-order smoothness regularization coefficient (default: 0.0, no regularization).
        Penalizes roughness in the extracted spectra.
    gamma : float
        Second-order smoothness (curvature) regularization coefficient (default: 0.0).
        Penalizes curvature, encouraging linear/flat spectral regions.
    normalize : bool
        If True (default), apply normalization to all terms. This makes parameters
        dimensionless and consistent across different problem sizes.

    Returns
    -------
    total_loss : float
        Total loss = chi2_norm + l2_term + smooth_term + curvature_term
    chi2_norm : float
        Normalized chi-squared (reduced chi-squared = χ²/M_total)
    l2_term : float
        L2 penalty term: alpha * norm_l2 * ||V||²_F
    smooth_term : float
        First-order smoothness penalty term: beta * norm_first * Smoothness
    curvature_term : float
        Second-order smoothness (curvature) penalty term: gamma * norm_second * Curvature

    Notes
    -----
    When normalize=True (default), all regularization terms are normalized:
    - L2: divided by (T*K) - average squared element
    - First-order smoothness: divided by ((T-1)*K) - average squared slope
    - Second-order smoothness: divided by ((T-2)*K) - average squared curvature
    - Chi-squared: divided by M_total - reduced chi-squared

    This makes alpha, beta, gamma dimensionless and consistent across different
    problem sizes. Typical normalized values:
    - alpha: 0.001 - 0.1 (prevents large spectral values)
    - beta: 0.01 - 1.0 (encourages smooth slopes)
    - gamma: 0.01 - 1.0 (encourages linear/flat regions)

    Setting alpha=beta=gamma=0.0 recovers the unregularized chi-squared objective.
    """
    T, K = V.shape

    # Total observations for chi2 normalization
    M_total = sum(D.shape[0] for D in sources_data)

    # Normalization factors
    norm_chi2 = 1.0 / M_total if normalize else 1.0
    norm_l2 = 1.0 / (T * K) if normalize else 1.0
    norm_first = 1.0 / ((T - 1) * K) if normalize else 1.0
    norm_second = 1.0 / ((T - 2) * K) if normalize else 1.0

    # Compute chi-squared (normalized to reduced chi-squared)
    chi2_raw = compute_global_chi2(sources_data, response_matrices, V, W)
    chi2_norm = chi2_raw * norm_chi2

    # Compute L2 regularization: Frobenius norm squared
    # Skip computation when alpha=0 for performance
    l2_term = 0.0
    if alpha > 0:
        frobenius_norm_sq = np.sum(V**2)
        l2_term = alpha * norm_l2 * frobenius_norm_sq

    # Compute first-order smoothness regularization
    # Skip computation when beta=0 for performance
    smooth_term = 0.0
    if beta > 0:
        smoothness_penalty = compute_smoothness_penalty(V)
        smooth_term = beta * norm_first * smoothness_penalty

    # Compute second-order smoothness (curvature) regularization
    # Skip computation when gamma=0 for performance
    curvature_term = 0.0
    if gamma > 0:
        curvature_penalty = compute_second_order_smoothness_penalty(V)
        curvature_term = gamma * norm_second * curvature_penalty

    # Total loss
    total_loss = chi2_norm + l2_term + smooth_term + curvature_term

    return total_loss, chi2_norm, l2_term, smooth_term, curvature_term


def als_wnmf(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    K: int = 0,
    T: int = 4096,
    lambda_min: float = 0.75,
    lambda_max: float = 5.0,
    alpha: float | None = None,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
    tol: float = 1e-4,
    max_iter: int = 100,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: bool = True,
    e_step_method: str = "numba",
    warm_start: bool = True,
    loss_eval_every: int = 1,
    m_step_chunk_size: int = 500,
    V_init: np.ndarray | None = None,
    W_init: np.ndarray | None = None,
    fix_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Alternating Least Squares Weighted Non-Negative Matrix Factorization.

    This is the main algorithm that extracts global basis spectra V and
    source weights W from heterogeneous observational data.

    Algorithm:
    1. Initialize V and W (random or from V_init/W_init)
    2. Loop until convergence:
       a. E-step: Update W using NNLS (parallel)
       b. M-step: Update V using MUR with regularization
       c. If fix_mask is provided, restore frozen columns after M-step
       d. Check convergence: |ΔLoss| / Loss < tol

    Objective function (with all regularization terms):
    Loss = χ²/M_total + α/(TK)||V||²_F + β/((T-1)K)·Smoothness + γ/((T-2)K)·Curvature

    Regularization terms (when normalize=True):
    - L2 (α): Prevents large spectral values
    - First-order smoothness (β): Encourages smooth slopes
    - Second-order smoothness (γ): Encourages linear/flat regions (penalizes curvature)

    Parameters
    ----------
    sources_data : list of np.ndarray
        List of D_n arrays for all N sources
    response_matrices : list of csr_matrix
        List of pre-computed R_n matrices
    K : int
        Number of non-negative components to extract. Ignored when V_init is provided
        (K is inferred from V_init.shape[1]). Default: 0 (must be set when V_init is None).
    T : int
        Number of wavelength bins (default: 4096). Ignored when V_init is provided.
    lambda_min : float
        Minimum wavelength for target grid (default: 0.75)
    lambda_max : float
        Maximum wavelength for target grid (default: 5.0)
    alpha : float, optional
        L2 regularization coefficient. If None (default), sets to 0.01 (normalized).
        When normalize=True, typical range: 0.001 - 0.1.
        When normalize=False, typical range: 0.01*N to 1.0*N.
    e_step_method : str
        NNLS solver for E-step: 'scipy' (default) or 'numba'.
        - 'scipy': scipy.optimize.nnls; proven exact solver but holds GIL
        - 'numba': Numba PGD (nogil=True); releases GIL for true parallelism,
          typically 2-4× faster than scipy on multi-core systems
        Both methods produce equivalent results for well-conditioned problems.
    warm_start : bool
        If True (default), initialize each E-step from the previous iteration's
        weights. Only effective when e_step_method='numba'. Reduces inner PGD
        iterations in later iterations when W changes slowly.
    beta : float, optional
        First-order smoothness regularization coefficient. Default is 0.0 (disabled).
        When normalize=True, typical range: 0.01 - 1.0.
        Penalizes first-order differences: Σ(V_{t+1} - V_t)².
    gamma : float, optional
        Second-order smoothness (curvature) regularization coefficient. Default is 0.0 (disabled).
        When normalize=True, typical range: 0.01 - 1.0.
        Penalizes curvature (discrete Laplacian squared), encouraging linear/flat regions.
    normalize : bool
        If True (default), apply normalization to all regularization terms. This makes
        alpha, beta, gamma dimensionless and consistent across different problem sizes.
    tol : float
        Convergence tolerance for relative Loss change (default: 1e-4)
    max_iter : int
        Maximum number of iterations (default: 100)
    n_jobs : int
        Number of parallel jobs for E-step (default: -1, all CPUs)
    seed : int
        Random seed for initialization (default: 42)
    verbose : bool
        Print iteration progress (default: True)
    loss_eval_every : int
        Evaluate exact loss every N iterations (default: 1, every iteration).
        Exact loss computation requires a full data scan; setting this to a
        larger value (e.g., 5 or 10) reduces per-iteration cost significantly
        for large datasets. Convergence is only checked at evaluation iterations.
        The final iteration always evaluates exact loss regardless of this setting.
    m_step_chunk_size : int
        Number of sources per chunk for M-step incremental reduction (default: 500).
        Reduces peak memory from O(N*T*K) to O(chunk_size*T*K) without changing
        the algorithmic result. Set to a large value (e.g., N) to disable chunking.
    V_init : np.ndarray, optional
        Initial basis matrix, shape (T, K). When provided, K and T are inferred
        from its shape and random initialization is skipped. Used for incremental
        training where part of the basis (e.g., galaxy components) is pre-learned.
    W_init : np.ndarray, optional
        Initial weight matrix, shape (N, K). When provided, random initialization
        is skipped. Must be provided together with V_init.
    fix_mask : np.ndarray, optional
        Boolean mask of shape (K,), where fix_mask[k] = True means component k
        is frozen during M-step (not updated). After each MUR update, frozen
        columns are restored to their V_init values. Used for incremental training:
        fix galaxy basis, learn AGN components.

    Returns
    -------
    V : np.ndarray
        Global basis matrix (eigenspectra), shape (T, K), non-negative
    W : np.ndarray
        Weight matrix, shape (N, K), non-negative
    loss_history : list of float
        Total loss at each iteration

    Notes
    -----
    The algorithm guarantees:
    - Monotonic decrease (or no increase) in total Loss after each step
    - Non-negativity of V and W at all times
    - Convergence to a local minimum

    When fix_mask is provided, frozen components still participate in E-step
    (they receive weights) but their V columns are never modified. The M-step
    computes MUR updates for all columns including frozen ones, then restores
    frozen columns to their original values. This ensures that frozen components
    contribute correctly to the residual computation for free components.

    Regularization helps prevent overfitting when:
    - Number of observations is limited
    - Data is very noisy
    - K (number of components) is large relative to N

    Recommended starting values (normalized mode):
    - alpha = 0.01 (weak L2)
    - beta = 0.1 (moderate first-order smoothness)
    - gamma = 0.0 (disabled by default, enable with 0.1 if needed)
    """
    N = len(sources_data)

    # Validate V_init / W_init consistency
    if V_init is not None and W_init is None:
        raise ValueError("W_init must be provided when V_init is given")
    if W_init is not None and V_init is None:
        raise ValueError("V_init must be provided when W_init is given")
    if V_init is not None and W_init is not None:
        if V_init.shape[1] != W_init.shape[1]:
            raise ValueError(
                f"V_init columns ({V_init.shape[1]}) != W_init columns ({W_init.shape[1]})"
            )
        if W_init.shape[0] != N:
            raise ValueError(
                f"W_init rows ({W_init.shape[0]}) != number of sources ({N})"
            )

    # Validate fix_mask
    if fix_mask is not None:
        if V_init is None:
            raise ValueError("fix_mask requires V_init to be provided")
        if fix_mask.shape[0] != V_init.shape[1]:
            raise ValueError(
                f"fix_mask length ({fix_mask.shape[0]}) != V_init columns ({V_init.shape[1]})"
            )

    # Infer K and T from V_init when provided
    if V_init is not None:
        T = V_init.shape[0]
        K = V_init.shape[1]
    else:
        if K <= 0:
            raise ValueError("K must be a positive integer when V_init is not provided")

    # Validate e_step_method
    if e_step_method not in ("scipy", "numba"):
        raise ValueError(f"e_step_method must be 'scipy' or 'numba', got '{e_step_method}'")

    # Trigger Numba JIT compilation before the main loop to avoid measuring
    # compilation time in the first iteration.
    if e_step_method == "numba" and NUMBA_AVAILABLE:
        if verbose:
            print("Warming up Numba JIT (first call compiles and caches the kernel)...")
        _t_jit = time.perf_counter()
        _dummy_A = np.ones((4, 2), dtype=np.float64)
        _dummy_b = np.ones(4, dtype=np.float64)
        nnls_pgd(_dummy_A, _dummy_b)
        if verbose:
            print(f"  JIT warmup done in {time.perf_counter() - _t_jit:.2f}s")
    elif e_step_method == "numba" and not NUMBA_AVAILABLE:
        if verbose:
            print("WARNING: Numba not available, falling back to pure-NumPy PGD for E-step.")

    # Set default alpha value based on normalization mode
    if alpha is None:
        if normalize:
            alpha = 0.01  # Normalized default
            if verbose:
                print(f"Using default L2 regularization: α = {alpha:.4f} (normalized)")
        else:
            alpha = 0.1 * N  # Legacy default
            if verbose:
                print(f"Using default L2 regularization: α = {alpha:.2f} (0.1 * N_sources)")
    elif alpha > 0:
        if verbose:
            mode_str = "normalized" if normalize else "unnormalized"
            print(f"Using L2 regularization: α = {alpha:.6f} ({mode_str})")
    else:
        if verbose:
            print("L2 regularization disabled (α = 0.0)")

    if verbose:
        if beta > 0:
            mode_str = "normalized" if normalize else "unnormalized"
            print(f"Using first-order smoothness: β = {beta:.6f} ({mode_str})")
        if gamma > 0:
            mode_str = "normalized" if normalize else "unnormalized"
            print(f"Using second-order smoothness: γ = {gamma:.6f} ({mode_str})")
        norm_str = "enabled" if normalize else "disabled"
        if e_step_method == "numba":
            numba_str = f"numba-pgd ({'available' if NUMBA_AVAILABLE else 'unavailable → fallback'})"
            warm_str = ", warm-start" if warm_start else ""
            estep_str = f"{numba_str}{warm_str}"
        else:
            estep_str = "scipy-nnls"
        init_mode = "from V_init/W_init" if V_init is not None else "random"
        print(f"Initializing ALS-WNMF with N={N} sources, K={K} components, T={T} bins [{init_mode}]")
        print(f"Normalization: {norm_str} | E-step: {estep_str}")
        if fix_mask is not None:
            n_fixed = int(np.sum(fix_mask))
            n_free = K - n_fixed
            print(f"fix_mask: {n_fixed} frozen + {n_free} free components")

    # Initialize parameters
    if V_init is not None and W_init is not None:
        V = V_init.copy()
        W = W_init.copy()
    else:
        V, W = initialize_parameters(N, T, K, seed=seed)

    # Validate initial non-negativity
    v_valid, w_valid = validate_non_negativity(V, W)
    assert v_valid and w_valid, "Initialization produced negative values!"

    # Compute initial loss (chi2 + regularization)
    loss, chi2, l2_term, smooth_term, curv_term = compute_regularized_loss(
        sources_data, response_matrices, V, W, alpha, beta, gamma, normalize
    )
    loss_history = [loss]

    if verbose:
        has_reg = (alpha > 0) or (beta > 0) or (gamma > 0)
        if has_reg:
            reg_str = ""
            if alpha > 0:
                reg_str += f"L2 = {l2_term:.6e}"
            if beta > 0:
                if reg_str:
                    reg_str += ", "
                reg_str += f"smooth = {smooth_term:.6e}"
            if gamma > 0:
                if reg_str:
                    reg_str += ", "
                reg_str += f"curv = {curv_term:.6e}"
            print(f"Initial Loss = {loss:.6e} (χ² = {chi2:.6e}, {reg_str})")
        else:
            print(f"Initial χ² = {chi2:.6e}")
        print("-" * 70)

    # Helper to format regularization string (reused below)
    def _fmt_reg(l2_v, smooth_v, curv_v):
        has_reg = (alpha > 0) or (beta > 0) or (gamma > 0)
        if not has_reg:
            return ""
        parts = []
        if alpha > 0:
            parts.append(f"L2 = {l2_v:.6e}")
        if beta > 0:
            parts.append(f"smooth = {smooth_v:.6e}")
        if gamma > 0:
            parts.append(f"curv = {curv_v:.6e}")
        return ", ".join(parts)

    # Main ALS loop
    converged = False
    for iteration in range(1, max_iter + 1):
        t_iter_start = time.perf_counter()

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

        # Lightweight non-negativity check after E-step
        _, w_valid = validate_non_negativity(V, W_new)
        assert w_valid, "E-step produced negative weights!"

        # M-step: Update V (fix W) with regularization (chunked reduction)
        t_m_start = time.perf_counter()
        V_new, _ = m_step(
            sources_data,
            response_matrices,
            V,
            W_new,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            normalize=normalize,
            n_jobs=n_jobs,
            chunk_size=m_step_chunk_size,
        )
        t_m_step = time.perf_counter() - t_m_start

        # Restore frozen columns after MUR update
        if fix_mask is not None:
            V_new[:, fix_mask] = V[:, fix_mask]

        # Validate non-negativity after M-step
        v_valid, w_valid = validate_non_negativity(V_new, W_new)
        assert v_valid and w_valid, "Negative values detected in V or W!"

        # Loss evaluation (controlled frequency)
        # Exact loss requires a full data scan; skip on non-evaluation iterations.
        # The final iteration always evaluates to ensure loss_history ends with exact value.
        do_eval = (iteration % loss_eval_every == 0) or (iteration == max_iter)

        if do_eval:
            t_loss_start = time.perf_counter()
            loss_new, chi2_new, l2_new, smooth_new, curv_new = compute_regularized_loss(
                sources_data, response_matrices, V_new, W_new, alpha, beta, gamma, normalize
            )
            t_loss = time.perf_counter() - t_loss_start
        else:
            t_loss = 0.0

        t_total = time.perf_counter() - t_iter_start

        # Print progress with full timing breakdown
        if verbose:
            if do_eval:
                timing = f"E={t_e_step:.1f}s, M={t_m_step:.1f}s, Loss={t_loss:.1f}s, Total={t_total:.1f}s"
                reg_s = _fmt_reg(l2_new, smooth_new, curv_new)
                delta_loss = loss - loss_new
                rel_change = abs(delta_loss) / loss_new if loss_new > 0 else 0
                if reg_s:
                    print(
                        f"Iter {iteration:3d}: {timing}"
                        f" | Loss = {loss_new:.6e} (χ² = {chi2_new:.6e}, {reg_s})"
                        f", ΔLoss = {delta_loss:.6e} ({rel_change:.6e})"
                    )
                else:
                    print(
                        f"Iter {iteration:3d}: {timing}"
                        f" | χ² = {chi2_new:.6e}, Δχ² = {delta_loss:.6e} ({rel_change:.6e})"
                    )
            else:
                timing = f"E={t_e_step:.1f}s, M={t_m_step:.1f}s, Loss=skip, Total={t_total:.1f}s"
                print(f"Iter {iteration:3d}: {timing}")

        # Update parameters
        V, W = V_new, W_new

        if do_eval:
            delta_loss = loss - loss_new
            rel_change = abs(delta_loss) / loss_new if loss_new > 0 else 0
            loss = loss_new
            chi2, l2_term, smooth_term, curv_term = chi2_new, l2_new, smooth_new, curv_new
            loss_history.append(loss)

            # Convergence check (only at evaluation iterations)
            if rel_change < tol:
                converged = True
                if verbose:
                    print("-" * 70)
                    print(f"Converged after {iteration} iterations!")
                    reg_s = _fmt_reg(l2_term, smooth_term, curv_term)
                    if reg_s:
                        print(f"Final Loss = {loss:.6e} (χ² = {chi2:.6e}, {reg_s})")
                    else:
                        print(f"Final χ² = {chi2:.6e}")
                break
        else:
            # Between evaluations: append previous loss as upper bound
            loss_history.append(loss)

    if not converged:
        if verbose:
            print("-" * 70)
            print(f"Reached maximum iterations ({max_iter})")
            reg_s = _fmt_reg(l2_term, smooth_term, curv_term)
            if reg_s:
                print(f"Final Loss = {loss:.6e} (χ² = {chi2:.6e}, {reg_s})")
            else:
                print(f"Final χ² = {chi2:.6e}")

    return V, W, loss_history
