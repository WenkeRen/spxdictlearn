"""
Utility functions for ALS-WNMF algorithm.

This module provides helper functions for initialization, objective function
computation, and data validation.
"""

import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix


def initialize_parameters(
    N: int, T: int, K: int, low: float = 0.1, high: float = 1.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize the latent variables V and W with random uniform values.

    Parameters
    ----------
    N : int
        Number of sources
    T : int
        Number of wavelength bins
    K : int
        Number of components
    low : float
        Lower bound for random initialization (default: 0.1)
    high : float
        Upper bound for random initialization (default: 1.0)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    V : np.ndarray
        Global basis matrix, shape (T, K), non-negative
    W : np.ndarray
        Weight matrix, shape (N, K), non-negative

    Notes
    -----
    Using low=0.1 instead of 0.0 avoids numerical issues with zero initialization.
    """
    rng = np.random.default_rng(seed)
    V = rng.uniform(low, high, size=(T, K))
    W = rng.uniform(low, high, size=(N, K))
    return V, W


def extract_source_data(D_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract flux and error arrays from raw observation data.

    Parameters
    ----------
    D_n : np.ndarray
        Raw observation data, shape (M_n, 4)
        - Column 0: Central wavelength
        - Column 1: Bandwidth
        - Column 2: Flux
        - Column 3: Error

    Returns
    -------
    lambda_c : np.ndarray
        Central wavelengths, shape (M_n,)
    fwhm : np.ndarray
        FWHM bandwidths, shape (M_n,)
    flux : np.ndarray
        Observed flux, shape (M_n,)
    error : np.ndarray
        Gaussian error standard deviations, shape (M_n,)
    """
    lambda_c = D_n[:, 0]
    fwhm = D_n[:, 1]
    flux = D_n[:, 2]
    error = D_n[:, 3]
    return lambda_c, fwhm, flux, error


def compute_source_chi2(
    flux: np.ndarray,
    error: np.ndarray,
    R_n: csr_matrix,
    V: np.ndarray,
    w_n: np.ndarray,
) -> float:
    """
    Compute the chi-squared contribution from a single source.

    chi2_n = || Σ_n^(-1/2) (y_n - R_n @ V @ w_n) ||_2^2

    Parameters
    ----------
    flux : np.ndarray
        Observed flux y_n, shape (M_n,)
    error : np.ndarray
        Error standard deviations σ_n, shape (M_n,)
    R_n : scipy.sparse.csr_matrix
        Response matrix, shape (M_n, T)
    V : np.ndarray
        Global basis matrix, shape (T, K)
    w_n : np.ndarray
        Weight vector for this source, shape (K,)

    Returns
    -------
    chi2_n : float
        Chi-squared contribution from this source
    """
    # Model prediction: ŷ = R @ V @ w
    y_pred = R_n @ (V @ w_n)  # Shape: (M_n,)

    # Weighted residual
    residual = (flux - y_pred) / error  # Element-wise division

    # Chi-squared
    chi2_n = np.sum(residual**2)

    return chi2_n


def compute_global_chi2(
    sources_data: list,
    response_matrices: list,
    V: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    Compute the global objective function (chi-squared).

    χ² = Σ_n || Σ_n^(-1/2) (y_n - R_n @ V @ w_n) ||_2^2

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

    Returns
    -------
    chi2 : float
        Global chi-squared value
    """
    N = len(sources_data)
    chi2 = 0.0

    for n in range(N):
        D_n = sources_data[n]
        R_n = response_matrices[n]
        w_n = W[n, :]  # Weight vector for source n

        # Extract flux and error
        _, _, flux, error = extract_source_data(D_n)

        # Compute chi2 contribution
        chi2_n = compute_source_chi2(flux, error, R_n, V, w_n)
        chi2 += chi2_n

    return chi2


def compute_smoothness_penalty(V: np.ndarray) -> float:
    """
    Compute first-order smoothness penalty for basis spectra V.

    This penalizes roughness in the extracted spectra by measuring the
    sum of squared differences between adjacent wavelength bins:

    Smoothness = Σ_k Σ_t (V_{t+1,k} - V_{t,k})²

    This corresponds to a discrete approximation of the L2 norm of the
    first derivative: ||∇V||². Smaller values indicate smoother spectra.

    Parameters
    ----------
    V : np.ndarray
        Global basis matrix, shape (T, K) where T is number of wavelength
        bins and K is number of components

    Returns
    -------
    smoothness_penalty : float
        Sum of squared first-order differences across all components.
        High values indicate rough/noisy spectra; low values indicate
        smooth spectra.

    Notes
    -----
    This is used in smoothness regularization to prevent overfitting to
    noise. The regularization term in the objective function is:
        β · Smoothness
    where β is the smoothness coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> V_smooth = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    >>> compute_smoothness_penalty(V_smooth)
    0.02  # Small penalty for smooth spectra
    >>> V_rough = np.array([[1.0, 2.0], [5.0, 0.5], [1.0, 3.0]])
    >>> compute_smoothness_penalty(V_rough)
    33.25  # Large penalty for rough spectra
    """
    # Compute first-order differences: V[1:] - V[:-1]
    # This creates an array of shape (T-1, K)
    diff = V[1:] - V[:-1]

    # Sum of squared differences (sum over all wavelength bins and components)
    smoothness_penalty = np.sum(diff**2)

    return smoothness_penalty


def compute_second_order_smoothness_penalty(V: np.ndarray) -> float:
    """
    Compute second-order smoothness penalty (discrete Laplacian squared).

    C_2nd = Σ_k Σ_{t=1}^{T-2} (V_{t+1,k} - 2*V_{t,k} + V_{t-1,k})²

    This penalizes curvature, encouraging linear/flat spectral regions.
    Linear spectra have zero penalty; quadratic and higher have positive penalty.

    Parameters
    ----------
    V : np.ndarray
        Global basis matrix, shape (T, K) where T is number of wavelength
        bins and K is number of components

    Returns
    -------
    penalty : float
        Sum of squared second-order differences (Laplacian squared).
        High values indicate curved/peaked spectra; low values indicate
        linear/flat spectra.

    Notes
    -----
    This is used in second-order smoothness regularization to prevent
    overfitting to noise while allowing linear trends. The regularization
    term in the objective function is:
        γ · C_2nd
    where γ is the second-order smoothness coefficient.

    Unlike first-order smoothness which encourages flat regions (zero slope),
    second-order smoothness encourages linear regions (zero curvature),
    preserving overall spectral shape while reducing noise.

    Examples
    --------
    >>> import numpy as np
    >>> # Linear spectrum: zero second-order penalty
    >>> V_linear = np.linspace(0, 1, 100).reshape(-1, 1)
    >>> compute_second_order_smoothness_penalty(V_linear)
    0.0  # Exactly zero for linear spectra

    >>> # Quadratic spectrum: non-zero penalty
    >>> V_quad = (np.linspace(0, 1, 100)**2).reshape(-1, 1)
    >>> compute_second_order_smoothness_penalty(V_quad)
    >0.0  # Positive for curved spectra
    """
    # Compute second-order differences (discrete Laplacian): V[2:] - 2*V[1:-1] + V[:-2]
    # This creates an array of shape (T-2, K)
    laplacian = V[2:] - 2 * V[1:-1] + V[:-2]

    # Sum of squared Laplacian values (sum over all wavelength bins and components)
    penalty = np.sum(laplacian**2)

    return penalty


def validate_non_negativity(V: np.ndarray, W: np.ndarray) -> Tuple[bool, bool]:
    """
    Validate that V and W are non-negative.

    Parameters
    ----------
    V : np.ndarray
        Global basis matrix
    W : np.ndarray
        Weight matrix

    Returns
    -------
    v_valid : bool
        True if all(V >= 0)
    w_valid : bool
        True if all(W >= 0)
    """
    v_valid = np.all(V >= 0)
    w_valid = np.all(W >= 0)
    return v_valid, w_valid


def print_iteration_stats(iteration: int, chi2: float, delta_chi2: float | None = None):
    """
    Print iteration statistics in a formatted way.

    Parameters
    ----------
    iteration : int
        Current iteration number
    chi2 : float
        Current chi-squared value
    delta_chi2 : float, optional
        Change in chi-squared from previous iteration
    """
    if delta_chi2 is not None:
        rel_change = delta_chi2 / chi2 if chi2 > 0 else 0
        print(f"Iter {iteration:3d}: χ² = {chi2:.6e}, Δχ² = {delta_chi2:.6e} ({rel_change:.6e})")
    else:
        print(f"Iter {iteration:3d}: χ² = {chi2:.6e}")
