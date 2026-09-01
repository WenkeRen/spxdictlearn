"""
Numba-accelerated Non-Negative Least Squares (NNLS) using Projected Gradient Descent.

This module provides a GIL-free NNLS solver that enables true parallelism in the
E-step of ALS-WNMF. The scipy.optimize.nnls function does not release the GIL,
limiting parallel speedup even with joblib's threading backend.

Algorithm: Projected Gradient Descent with Armijo Line Search
- Precompute A.T @ A and A.T @ b for efficiency
- Project onto non-negative orthant after each gradient step
- Use Armijo backtracking to ensure monotonic convergence
- Converges in 50-200 iterations for typical SPXPCA problems (K=3-20, M=500-12000)

Performance:
- @njit(nogil=True) releases GIL for true parallelism
- fastmath=True enables SIMD vectorization
- cache=True avoids recompilation overhead

Reference: Lin (2007) "Projected Gradient Methods for Nonnegative Matrix Factorization"
"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators for fallback

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):
        return range(*args)


@njit(nogil=True, cache=True, fastmath=True)
def _nnls_pgd_core(
    AtA: np.ndarray,
    Atb: np.ndarray,
    x_init: np.ndarray,
    use_init: bool,
    max_iter: int,
    tol: float,
    sigma: float,
    beta: float,
) -> tuple:
    """
    Core PGD solver with precomputed A.T @ A and A.T @ b.

    Solves: min_{x >= 0} ||Ax - b||_2^2

    Parameters
    ----------
    AtA : np.ndarray
        Precomputed A.T @ A, shape (K, K)
    Atb : np.ndarray
        Precomputed A.T @ b, shape (K,)
    x_init : np.ndarray
        Initial guess, shape (K,). Ignored when use_init=False.
    use_init : bool
        If True, warm-start from x_init; otherwise start from zeros.
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance on projected gradient norm
    sigma : float
        Armijo parameter (sufficient decrease)
    beta : float
        Step size reduction factor for backtracking

    Returns
    -------
    x : np.ndarray
        Solution vector, shape (K,), non-negative
    converged : bool
        True if converged within tolerance
    """
    K = Atb.shape[0]

    # Initialize
    x = np.empty(K)
    if use_init:
        for k in range(K):
            val = x_init[k]
            x[k] = val if val > 0.0 else 0.0
    else:
        for k in range(K):
            x[k] = 0.0

    # Main PGD loop
    for iteration in range(max_iter):
        # Compute gradient: grad = AtA @ x - Atb
        grad = AtA @ x - Atb

        # Projected gradient norm (KKT: only count binding constraints correctly)
        grad_norm = 0.0
        for k in range(K):
            if x[k] > 0.0 or grad[k] < 0.0:
                grad_norm += grad[k] * grad[k]
        grad_norm = np.sqrt(grad_norm)

        # Check convergence
        if grad_norm < tol:
            for k in range(K):
                if x[k] < 0.0:
                    x[k] = 0.0
            return x, True

        # Armijo line search with projection
        d = -grad
        alpha = 1.0
        f_x = 0.5 * np.dot(x, AtA @ x) - np.dot(Atb, x)

        for _ in range(50):  # Max 50 backtracking steps
            x_trial = np.empty(K)
            for k in range(K):
                val = x[k] + alpha * d[k]
                x_trial[k] = val if val > 0.0 else 0.0

            f_trial = 0.5 * np.dot(x_trial, AtA @ x_trial) - np.dot(Atb, x_trial)
            directional_deriv = np.dot(grad, x_trial - x)

            if f_trial <= f_x + sigma * directional_deriv + 1e-12:
                break
            alpha *= beta

        x = x_trial

    for k in range(K):
        if x[k] < 0.0:
            x[k] = 0.0

    return x, False


def nnls_pgd(
    A: np.ndarray,
    b: np.ndarray,
    x_init: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """
    Solve Non-Negative Least Squares using Projected Gradient Descent.

    Solves: min_{x >= 0} ||Ax - b||_2^2

    Uses Armijo line search with projection to ensure monotonic convergence
    and non-negativity constraint satisfaction.  When Numba is available the
    core loop runs with nogil=True, enabling true thread-level parallelism
    inside joblib's threading backend.

    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape (M, K). Must be C-contiguous.
    b : np.ndarray
        Observation vector, shape (M,). Must be C-contiguous.
    x_init : np.ndarray, optional
        Warm-start initial guess, shape (K,). Passing the previous iteration's
        weight vector typically reduces inner iterations significantly.
    max_iter : int
        Maximum iterations (default: 500)
    tol : float
        Convergence tolerance on projected gradient norm (default: 1e-8)

    Returns
    -------
    x : np.ndarray
        Solution vector, shape (K,), non-negative
    converged : bool
        True if converged within tolerance
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)

    # Precompute normal equations
    AtA = A.T @ A
    Atb = A.T @ b

    # Prepare warm-start arrays (Numba kernels cannot accept Python None)
    if x_init is not None:
        x0 = np.ascontiguousarray(x_init, dtype=np.float64)
        use_init = True
    else:
        x0 = np.zeros(A.shape[1], dtype=np.float64)
        use_init = False

    sigma = 0.01  # Armijo sufficient-decrease parameter
    beta = 0.5  # Backtracking reduction factor

    x, converged = _nnls_pgd_core(AtA, Atb, x0, use_init, max_iter, tol, sigma, beta)

    # Defensive clamp
    x = np.maximum(x, 0.0)
    return x, converged


def nnls_pgd_fallback(
    A: np.ndarray,
    b: np.ndarray,
    x_init: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """
    Fallback NNLS using pure NumPy PGD when Numba is not available.

    Provides the same interface and algorithm as nnls_pgd but runs in pure
    Python/NumPy.  Slower than the Numba version, but always available.

    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape (M, K)
    b : np.ndarray
        Observation vector, shape (M,)
    x_init : np.ndarray, optional
        Warm-start initial guess, shape (K,)
    max_iter : int
        Maximum iterations (default: 500)
    tol : float
        Convergence tolerance (default: 1e-8)

    Returns
    -------
    x : np.ndarray
        Solution vector, shape (K,), non-negative
    converged : bool
        True if converged within tolerance
    """
    AtA = (A.T @ A).astype(np.float64)
    Atb = (A.T @ b).astype(np.float64)

    if x_init is not None:
        x = np.maximum(x_init.astype(np.float64), 0.0)
    else:
        x = np.zeros(A.shape[1], dtype=np.float64)

    sigma = 0.01
    beta_bt = 0.5

    for _ in range(max_iter):
        grad = AtA @ x - Atb
        # Projected gradient (KKT)
        grad_proj = np.where((x > 0) | (grad < 0), grad, 0.0)
        grad_norm = np.linalg.norm(grad_proj)

        if grad_norm < tol:
            return x, True

        d = -grad
        alpha = 1.0
        f_x = 0.5 * x @ AtA @ x - Atb @ x

        for _ in range(50):
            x_trial = np.maximum(x + alpha * d, 0.0)
            f_trial = 0.5 * x_trial @ AtA @ x_trial - Atb @ x_trial
            directional_deriv = grad @ (x_trial - x)
            if f_trial <= f_x + sigma * directional_deriv + 1e-12:
                break
            alpha *= beta_bt

        x = x_trial

    return x, False
