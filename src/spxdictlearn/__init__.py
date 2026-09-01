"""
SPXPCA - Global NMF for Heterogeneous Spectra

A Python package implementing Alternating Least Squares Weighted
Non-Negative Matrix Factorization (ALS-WNMF) for extracting physically
meaningful spectra from sparse, unaligned, and noisy astronomical data.

Main components:
- response_matrix: Sparse Gaussian response matrix construction
- als_wnmf: Core ALS-WNMF algorithm
- hals: Hierarchical ALS (HALS) for symmetry breaking
- pruning: Dictionary pruning utilities
- normalization: Basis spectrum normalization for weight comparability
- mock_data: Synthetic data generation for testing
- utils: Helper functions for initialization and validation

Version 0.4.0 adds:
- HALS algorithm for improved spectral orthogonality
- Dictionary pruning for removing similar components
- Normalization module for weight comparability
"""

from . import als_wnmf, hals, mock_data, normalization, numba_nnls, pruning, response_matrix, utils
from .als_wnmf import m_step_single_source
from .hals import compute_M_tensor, hals_wnmf, m_step_hals, precompute_hals_constants
from .normalization import denormalize_basis, normalize_basis, normalize_basis_l1, normalize_basis_l2
from .numba_nnls import NUMBA_AVAILABLE, nnls_pgd, nnls_pgd_fallback
from .pruning import (
    cluster_similar_components,
    compute_cosine_similarity_matrix,
    merge_components_by_cluster,
    prune_and_sort_dictionary,
)
from .utils import compute_second_order_smoothness_penalty

__version__ = "0.4.0"

__all__ = [
    # Modules
    "als_wnmf",
    "hals",
    "mock_data",
    "normalization",
    "numba_nnls",
    "pruning",
    "response_matrix",
    "utils",
    # ALS-WNMF
    "m_step_single_source",
    # HALS
    "hals_wnmf",
    "m_step_hals",
    "precompute_hals_constants",
    "compute_M_tensor",
    # Normalization
    "normalize_basis",
    "normalize_basis_l1",
    "normalize_basis_l2",
    "denormalize_basis",
    # Pruning
    "compute_cosine_similarity_matrix",
    "cluster_similar_components",
    "merge_components_by_cluster",
    "prune_and_sort_dictionary",
    # Numba
    "NUMBA_AVAILABLE",
    "nnls_pgd",
    "nnls_pgd_fallback",
    # Utils
    "compute_second_order_smoothness_penalty",
]
