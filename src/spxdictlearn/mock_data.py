"""
Mock Data Generation for ALS-WNMF Testing

This module generates synthetic observational data for testing the ALS-WNMF
algorithm with known ground truth.

Ground truth components (K_true = 3):
- Component 0: Broad Gaussian continuum
- Component 1: Power-law spectrum
- Component 2: Sparse narrow emission lines
"""

from typing import List, Literal, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .response_matrix import build_response_matrix, build_target_grid


def generate_true_basis(target_edges: np.ndarray, K_true: int = 3) -> np.ndarray:
    """
    Generate K_true distinct artificial spectra on the target wavelength grid.

    Component 0: Broad Gaussian continuum (e.g., centered at 2.0 μm)
    Component 1: Power-law curve (f(λ) = λ^(-1.5))
    Component 2: Sparse narrow Gaussian emission lines

    Parameters
    ----------
    target_edges : np.ndarray
        Target grid bin edges, shape (T+1,)
    K_true : int
        Number of true components (default: 3)

    Returns
    -------
    V_true : np.ndarray
        True basis matrix, shape (T, K_true), non-negative
    """
    T = len(target_edges) - 1

    # Compute bin centers
    bin_centers = np.sqrt(target_edges[:-1] * target_edges[1:])

    # Initialize basis matrix
    V_true = np.zeros((T, K_true))

    # Component 0: Broad Gaussian continuum
    # Center at 2.0 μm, width 1.0 μm
    V_true[:, 0] = np.exp(-0.5 * ((bin_centers - 2.0) / 1.0) ** 2)

    # Component 1: Power-law (f(λ) = λ^(-1.5))
    # Normalize to avoid tiny values
    V_true[:, 1] = bin_centers ** (-1.5)
    V_true[:, 1] /= np.max(V_true[:, 1])  # Normalize to max=1

    # Component 2: Sparse narrow emission lines
    # Lines at 1.5, 2.0, 3.0, 4.0 μm
    line_centers = np.array([1.5, 2.0, 3.0, 4.0])
    line_widths = np.array([0.02, 0.02, 0.02, 0.02])  # Narrow lines

    for center, width in zip(line_centers, line_widths):
        V_true[:, 2] += np.exp(-0.5 * ((bin_centers - center) / width) ** 2)

    # Normalize component 2
    V_true[:, 2] /= np.max(V_true[:, 2])

    return V_true


def generate_true_weights(N: int, K_true: int = 3, seed: int = 42) -> np.ndarray:
    """
    Generate random true weights for N sources.

    Parameters
    ----------
    N : int
        Number of sources
    K_true : int
        Number of components (default: 3)
    seed : int
        Random seed

    Returns
    -------
    W_true : np.ndarray
        True weight matrix, shape (N, K_true), non-negative
    """
    rng = np.random.default_rng(seed)
    W_true = rng.uniform(0.1, 5.0, size=(N, K_true))
    return W_true


def generate_mock_source_observations(
    n_source: int,
    V_true: np.ndarray,
    w_true: np.ndarray,
    M_n: int = 500,
    fwhm_const: float = 0.02,
    noise_level: float = 0.05,
    lambda_min: float = 0.75,
    lambda_max: float = 5.0,
    target_edges: np.ndarray | None = None,
    response_type: Literal["rectangular", "gaussian"] = "rectangular",
    seed: int = 42,
) -> Tuple[np.ndarray, csr_matrix]:
    """
    Generate mock observations for a single source.

    Process:
    1. Generate M_n random central wavelengths
    2. Build true high-res spectrum: x = V_true @ w_true
    3. Build response matrix R_n
    4. Compute true observation: y_true = R_n @ x
    5. Add Gaussian noise: y = y_true + N(0, σ²)
    6. Pack into D_n array

    Parameters
    ----------
    n_source : int
        Source index (for seed variation)
    V_true : np.ndarray
        True basis matrix, shape (T, K_true)
    w_true : np.ndarray
        True weight vector for this source, shape (K_true,)
    M_n : int
        Number of observations per source (default: 500)
    fwhm_const : float
        Constant FWHM for all observations (default: 0.02)
    noise_level : float
        Noise as fraction of mean flux (default: 0.05)
    lambda_min : float
        Min central wavelength for observations (default: 0.8)
    lambda_max : float
        Max central wavelength for observations (default: 4.8)
    target_edges : np.ndarray, optional
        Target grid edges (if None, uses default)
    response_type : {"rectangular", "gaussian"}, default "rectangular"
        Response function model for mock observation generation.
    seed : int
        Base random seed

    Returns
    -------
    D_n : np.ndarray
        Mock observation data, shape (M_n, 4)
        - Column 0: Central wavelength
        - Column 1: FWHM bandwidth
        - Column 2: Noisy flux
        - Column 3: Error standard deviation
    R_n : scipy.sparse.csr_matrix
        Response matrix, shape (M_n, T)
    """
    # Use source-specific seed
    rng = np.random.default_rng(seed + n_source)

    # Generate random central wavelengths
    lambda_c = rng.uniform(lambda_min, lambda_max, size=M_n)

    # Constant FWHM
    fwhm = np.full(M_n, fwhm_const)

    # Build target grid if not provided
    if target_edges is None:
        target_edges, _ = build_target_grid()

    T = len(target_edges) - 1

    # Build true high-res spectrum
    x_true = V_true @ w_true  # Shape (T,)

    # Build response matrix
    # Pack D_n for response matrix construction
    D_n_temp = np.column_stack([lambda_c, fwhm, np.zeros(M_n), np.zeros(M_n)])
    R_n = build_response_matrix(D_n_temp, target_edges, response_type=response_type)

    # Compute true observations
    y_true = R_n @ x_true  # Shape (M_n,)

    # Add noise
    # Error as fraction of mean flux
    error = noise_level * np.mean(y_true) * np.ones(M_n)

    # Add Gaussian noise
    y_noisy = y_true + rng.normal(0, error)

    # Pack into D_n array
    D_n = np.column_stack([lambda_c, fwhm, y_noisy, error])

    return D_n, R_n


def generate_mock_data(
    N: int = 50,
    M_per_source: int = 500,
    K_true: int = 3,
    T: int = 4096,
    noise_level: float = 0.05,
    response_type: Literal["rectangular", "gaussian"] = "rectangular",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[csr_matrix], np.ndarray, np.ndarray]:
    """
    Generate complete mock dataset for testing ALS-WNMF.

    Parameters
    ----------
    N : int
        Number of sources (default: 50)
    M_per_source : int
        Number of observations per source (default: 500)
    K_true : int
        Number of true components (default: 3)
    T : int
        Number of wavelength bins (default: 4096)
    noise_level : float
        Noise level as fraction of mean flux (default: 0.05)
    response_type : {"rectangular", "gaussian"}, default "rectangular"
        Response function model for mock observation generation.
    seed : int
        Random seed (default: 42)
    verbose : bool
        Print progress (default: True)

    Returns
    -------
    sources_data : list of np.ndarray
        List of D_n arrays for all sources
    response_matrices : list of csr_matrix
        List of R_n matrices for all sources
    V_true : np.ndarray
        True basis matrix, shape (T, K_true)
    W_true : np.ndarray
        True weight matrix, shape (N, K_true)
    """
    if verbose:
        print(f"Generating mock data: N={N} sources, M={M_per_source} obs/source")
        print(f"K_true={K_true} components, T={T} bins, noise={noise_level * 100}%")

    # Build target grid
    target_edges, _ = build_target_grid(n_bins=T)
    assert len(target_edges) - 1 == T, f"Grid size mismatch: {len(target_edges) - 1} != {T}"

    # Generate ground truth
    V_true = generate_true_basis(target_edges, K_true=K_true)
    W_true = generate_true_weights(N, K_true=K_true, seed=seed)

    if verbose:
        print(f"Generated true basis V: {V_true.shape}")
        print(f"Generated true weights W: {W_true.shape}")

    # Generate observations for each source
    sources_data = []
    response_matrices = []

    for n in range(N):
        D_n, R_n = generate_mock_source_observations(
            n_source=n,
            V_true=V_true,
            w_true=W_true[n, :],
            M_n=M_per_source,
            noise_level=noise_level,
            target_edges=target_edges,
            response_type=response_type,
            seed=seed,
        )

        sources_data.append(D_n)
        response_matrices.append(R_n)

        if verbose and (n + 1) % max(1, N // 10) == 0:
            print(f"Generated {n + 1}/{N} sources")

    if verbose:
        print("Mock data generation complete!")

    return sources_data, response_matrices, V_true, W_true
