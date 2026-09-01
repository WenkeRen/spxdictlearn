"""
Response Matrix Construction for ALS-WNMF

This module implements the construction of sparse response matrices
for mapping high-resolution spectra to observed flux measurements.

Supported response models:
- Rectangular (top-hat): Unity transmission within FWHM, zero outside.
  Band edges at λ_c ± FWHM/2. Default choice.
- Gaussian: Truncated Gaussian response (3σ) with σ = FWHM / 2.35482.
  Uses exact CDF integration. More realistic for gradient spectroscopy.

Key features:
- Logarithmic wavelength grid from 0.75 to 5.0 microns
- Exact bin-wise integration (overlap for rectangular, CDF for Gaussian)
- Flux conservation normalization (row sums = 1.0)
- Memory-efficient sparse matrix representation (CSR format)
"""

import numpy as np
from scipy.stats import norm
from scipy.sparse import coo_matrix, csr_matrix
from typing import Literal, Tuple


def build_target_grid(
    lambda_min: float = 0.75, lambda_max: float = 5.0, n_bins: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the target high-resolution logarithmic wavelength grid.

    Parameters
    ----------
    lambda_min : float
        Minimum wavelength in microns (default: 0.75)
    lambda_max : float
        Maximum wavelength in microns (default: 5.0)
    n_bins : int
        Number of wavelength bins (default: 4096)

    Returns
    -------
    edges : np.ndarray
        Array of n_bins+1 bin edges in linear space (microns)
    centers : np.ndarray
        Array of n_bin bin centers in linear space (microns)

    Notes
    -----
    The grid is linearly spaced in log-space:
    ln(edges) = linspace(ln(lambda_min), ln(lambda_max), n_bins+1)
    """
    # Linear spacing in log space
    log_edges = np.linspace(np.log(lambda_min), np.log(lambda_max), n_bins + 1)
    edges = np.exp(log_edges)

    # Compute bin centers (geometric mean)
    centers = np.sqrt(edges[:-1] * edges[1:])

    return edges, centers


def build_response_matrix(
    D_n: np.ndarray,
    target_edges: np.ndarray,
    response_type: Literal["rectangular", "gaussian"] = "rectangular",
) -> csr_matrix:
    """
    Build a sparse response matrix R_n for a single source.

    The response matrix maps high-resolution spectrum (T bins) to observed
    flux measurements (M_n observations) using a bandpass model.

    Construction method:
    1. For each observation m with central wavelength λ_c and FWHM:
       - Rectangular: band edges at λ_c ± FWHM/2, unity transmission inside
       - Gaussian: σ_f = FWHM / 2.35482, truncate at ±3σ_f
    2. For each target bin t overlapping with band edges:
       - Compute integration bounds: a = max(λ_start, edge[t]), b = min(λ_end, edge[t+1])
       - Rectangular: R_n[m,t] = b - a  (overlap length)
       - Gaussian: R_n[m,t] = Φ(b; λ_c, σ_f) - Φ(a; λ_c, σ_f)  (CDF integral)
    3. Normalize each row to sum to 1.0 (flux conservation)

    Parameters
    ----------
    D_n : np.ndarray
        Raw observation data for source n, shape (M_n, 4)
        - Column 0: Central wavelength λ_c (microns)
        - Column 1: Bandwidth FWHM (microns)
        - Column 2: Observed flux (not used here, for data structure consistency)
        - Column 3: Error (not used here, for data structure consistency)
    target_edges : np.ndarray
        Target grid bin edges, shape (T+1,) in microns
    response_type : {"rectangular", "gaussian"}, default "rectangular"
        Response function model:
        - "rectangular": Top-hat bandpass. Band edges at λ_c ± FWHM/2.
          Integration reduces to computing overlap lengths between band and
          target bins. Simpler and more localized than Gaussian.
        - "gaussian": Truncated Gaussian response (3σ) with σ = FWHM / 2.35482.
          Uses exact CDF integration. More realistic for slit/gradient spectroscopy
          but extends beyond the nominal FWHM.

    Returns
    -------
    R_n : scipy.sparse.csr_matrix
        Sparse response matrix, shape (M_n, T) where T = len(target_edges) - 1.
        Row m contains the response weights for observation m, normalized to sum
        to 1.0 (flux conservation).

    Notes
    -----
    - Uses COO format for efficient construction, then converts to CSR
    - CSR format enables efficient row slicing and matrix-vector multiplication
    - For Gaussian: truncation at 3σ loses ~0.27% of flux, corrected by row normalization
    - For rectangular: no flux is lost outside the band, normalization distributes
      fractional transmission across overlapping target bins
    """
    M_n = D_n.shape[0]
    T = len(target_edges) - 1

    # Extract observation parameters
    lambda_c = D_n[:, 0]  # Central wavelengths, shape (M_n,)
    fwhm = D_n[:, 1]  # FWHM bandwidths, shape (M_n,)

    # Compute band boundaries and Gaussian sigma (if needed)
    if response_type == "rectangular":
        # Top-hat bandpass: edges at λ_c ± FWHM/2
        lambda_start = lambda_c - fwhm / 2.0
        lambda_end = lambda_c + fwhm / 2.0
        sigma_f = None  # Not used
    elif response_type == "gaussian":
        # σ_f = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.35482
        sigma_f = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        lambda_start = lambda_c - 3.0 * sigma_f
        lambda_end = lambda_c + 3.0 * sigma_f
    else:
        raise ValueError(
            f"Unknown response_type: {response_type!r}. " "Use 'rectangular' or 'gaussian'."
        )

    # Build COO format sparse matrix
    # We'll collect all non-zero entries
    row_indices = []
    col_indices = []
    data = []

    for m in range(M_n):
        # Find overlapping bins
        # Bins fully before or after the truncated range have zero response
        mask_start = target_edges[:-1] < lambda_end[m]
        mask_end = target_edges[1:] > lambda_start[m]
        overlap_mask = mask_start & mask_end

        if not np.any(overlap_mask):
            # No overlap - this shouldn't happen with proper wavelength coverage
            continue

        overlap_bins = np.where(overlap_mask)[0]

        for t in overlap_bins:
            # Integration bounds for this bin
            a = max(lambda_start[m], target_edges[t])
            b = min(lambda_end[m], target_edges[t + 1])

            if a < b:
                if response_type == "gaussian":
                    # Exact Gaussian integral using CDF
                    integral = norm.cdf(b, loc=lambda_c[m], scale=sigma_f[m]) - norm.cdf(
                        a, loc=lambda_c[m], scale=sigma_f[m]
                    )
                else:
                    # Rectangular: integral is simply the overlap length
                    integral = b - a

                if integral > 0:
                    row_indices.append(m)
                    col_indices.append(t)
                    data.append(integral)

    # Construct COO matrix and convert to CSR
    R_n = coo_matrix((data, (row_indices, col_indices)), shape=(M_n, T), dtype=np.float64).tocsr()

    # Flux conservation normalization
    # Since we truncated at 3σ, rows don't sum to exactly 1.0
    # Normalize each row to preserve total flux
    row_sums = np.asarray(R_n.sum(axis=1)).ravel()

    # Avoid division by zero
    valid_rows = row_sums > 0

    if np.any(valid_rows):
        # Normalize valid rows
        R_n = R_n.multiply(1.0 / np.where(valid_rows, row_sums, 1.0)[:, np.newaxis])

    # Ensure CSR format
    R_n = R_n.tocsr()

    return R_n


def build_all_response_matrices(
    sources_data: list,
    target_edges: np.ndarray,
    response_type: Literal["rectangular", "gaussian"] = "rectangular",
    verbose: bool = True,
) -> list:
    """
    Build response matrices for all sources.

    Parameters
    ----------
    sources_data : list of np.ndarray
        List of D_n arrays, one per source
    target_edges : np.ndarray
        Target grid bin edges
    response_type : {"rectangular", "gaussian"}, default "rectangular"
        Response function model passed to build_response_matrix.
    verbose : bool
        Print progress information

    Returns
    -------
    response_matrices : list of scipy.sparse.csr_matrix
        List of R_n matrices, one per source
    """
    N = len(sources_data)
    response_matrices = []

    for n in range(N):
        R_n = build_response_matrix(sources_data[n], target_edges, response_type=response_type)
        response_matrices.append(R_n)

        if verbose and (n + 1) % max(1, N // 10) == 0:
            print(f"Built {n + 1}/{N} response matrices")

    return response_matrices
