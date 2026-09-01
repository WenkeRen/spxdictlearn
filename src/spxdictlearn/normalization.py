"""
Basis Spectrum Normalization Module

This module provides normalization utilities for NMF basis spectra (V) and
weights (W). Normalization makes weights comparable across components by
standardizing the amplitude of each basis spectrum.

Mathematical Background
-----------------------
In NMF, the decomposition V @ W^T has a scale ambiguity: multiplying V[:,k]
by a factor s and dividing W[:,k] by s preserves the reconstruction.

Without normalization, components with larger amplitudes have smaller weights,
making weight-based comparisons (e.g., for importance ranking) misleading.

Normalization Methods
---------------------
- L1 normalization: mean(V[:,k]) = 1 for all k
  - Intuitive for astronomical spectra (average flux = 1)
  - Recommended for weight-based component importance ranking

- L2 normalization: ||V[:,k]||_2 = 1 for all k
  - Standard for orthonormal basis functions
  - Useful for PCA-like interpretations

Both methods preserve the reconstruction: V @ W^T = V_norm @ W_norm^T
"""

import warnings
from typing import Tuple

import numpy as np


def normalize_basis_l1(V: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L1 normalization: mean(V[:, k]) = 1 for all components k.

    This normalization scales each component so that its mean value equals 1,
    and adjusts the corresponding weights to preserve reconstruction.

    Parameters
    ----------
    V : np.ndarray
        Basis matrix, shape (T, K) where T is the number of wavelength bins
        and K is the number of components.
    W : np.ndarray
        Weight matrix, shape (N, K) where N is the number of sources.

    Returns
    -------
    V_norm : np.ndarray
        Normalized basis matrix, shape (T, K), with mean(V_norm[:, k]) = 1.
    W_norm : np.ndarray
        Normalized weight matrix, shape (N, K), scaled to preserve reconstruction.
    scales : np.ndarray
        Scale factors, shape (K,), where scales[k] = mean(V[:, k]).
        Useful for potential denormalization: V = V_norm * scales.

    Raises
    ------
    ValueError
        If V and W have incompatible shapes (different K dimensions).

    Warns
    -----
    RuntimeWarning
        If any component has zero or near-zero mean, which would cause
        division by zero. Such components are left unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> V = np.array([[1, 2], [3, 4], [5, 6]])  # T=3, K=2
    >>> W = np.array([[0.5, 0.3], [0.2, 0.4]])  # N=2, K=2
    >>> V_norm, W_norm, scales = normalize_basis_l1(V, W)
    >>> np.mean(V_norm, axis=0)  # Should be [1.0, 1.0]
    array([1., 1.])
    >>> # Reconstruction is preserved
    >>> np.allclose(V @ W.T, V_norm @ W_norm.T)
    True

    Notes
    -----
    The reconstruction V @ W^T is mathematically equivalent to V_norm @ W_norm^T:
        V_norm[:, k] = V[:, k] / scale_k
        W_norm[:, k] = W[:, k] * scale_k
        V_norm @ W_norm^T = sum_k (V[:,k]/scale_k) @ (W[:,k]*scale_k)^T
                         = sum_k V[:,k] @ W[:,k]^T
                         = V @ W^T
    """
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if V.ndim != 2 or W.ndim != 2:
        raise ValueError(f"V and W must be 2D arrays, got shapes {V.shape} and {W.shape}")

    T, K = V.shape
    N, K_w = W.shape

    if K != K_w:
        raise ValueError(f"V has {K} components but W has {K_w} components")

    # Compute L1 scale: mean value of each component
    scales = np.mean(V, axis=0)  # Shape: (K,)

    # Check for zero or near-zero scales
    zero_threshold = 1e-12
    zero_mask = np.abs(scales) < zero_threshold

    if np.any(zero_mask):
        zero_components = np.where(zero_mask)[0].tolist()
        warnings.warn(
            f"Components {zero_components} have zero or near-zero mean (< {zero_threshold}). "
            "These components will not be normalized (scale = 1.0). "
            "Consider removing them from the analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
        scales[zero_mask] = 1.0  # Avoid division by zero

    # Normalize V and adjust W
    V_norm = V / scales[np.newaxis, :]  # Broadcasting: (T, K) / (1, K)
    W_norm = W * scales[np.newaxis, :]  # Broadcasting: (N, K) * (1, K)

    return V_norm, W_norm, scales


def normalize_basis_l2(V: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L2 normalization: ||V[:, k]||_2 = 1 for all components k.

    This normalization scales each component to unit L2 norm, making the
    basis vectors orthonormal in magnitude. Useful for PCA-like interpretations.

    Parameters
    ----------
    V : np.ndarray
        Basis matrix, shape (T, K) where T is the number of wavelength bins
        and K is the number of components.
    W : np.ndarray
        Weight matrix, shape (N, K) where N is the number of sources.

    Returns
    -------
    V_norm : np.ndarray
        Normalized basis matrix, shape (T, K), with ||V_norm[:, k]||_2 = 1.
    W_norm : np.ndarray
        Normalized weight matrix, shape (N, K), scaled to preserve reconstruction.
    scales : np.ndarray
        Scale factors, shape (K,), where scales[k] = ||V[:, k]||_2.

    Raises
    ------
    ValueError
        If V and W have incompatible shapes (different K dimensions).

    Warns
    -----
    RuntimeWarning
        If any component has zero or near-zero L2 norm, which would cause
        division by zero. Such components are left unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> V = np.array([[1, 0], [0, 1], [0, 0]])  # Orthonormal columns
    >>> W = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> V_norm, W_norm, scales = normalize_basis_l2(V, W)
    >>> np.linalg.norm(V_norm, axis=0)  # Should be [1.0, 1.0]
    array([1., 1.])

    Notes
    -----
    The reconstruction V @ W^T is mathematically equivalent to V_norm @ W_norm^T.
    L2 normalization is standard in PCA and SVD contexts.
    """
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if V.ndim != 2 or W.ndim != 2:
        raise ValueError(f"V and W must be 2D arrays, got shapes {V.shape} and {W.shape}")

    T, K = V.shape
    N, K_w = W.shape

    if K != K_w:
        raise ValueError(f"V has {K} components but W has {K_w} components")

    # Compute L2 scale: Euclidean norm of each component
    scales = np.linalg.norm(V, axis=0)  # Shape: (K,)

    # Check for zero or near-zero scales
    zero_threshold = 1e-12
    zero_mask = np.abs(scales) < zero_threshold

    if np.any(zero_mask):
        zero_components = np.where(zero_mask)[0].tolist()
        warnings.warn(
            f"Components {zero_components} have zero or near-zero L2 norm (< {zero_threshold}). "
            "These components will not be normalized (scale = 1.0). "
            "Consider removing them from the analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
        scales[zero_mask] = 1.0  # Avoid division by zero

    # Normalize V and adjust W
    V_norm = V / scales[np.newaxis, :]  # Broadcasting: (T, K) / (1, K)
    W_norm = W * scales[np.newaxis, :]  # Broadcasting: (N, K) * (1, K)

    return V_norm, W_norm, scales


def normalize_basis(
    V: np.ndarray,
    W: np.ndarray,
    method: str = "l1",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    General normalization interface for basis spectra and weights.

    This is the recommended entry point for normalization, providing a
    unified interface to different normalization methods.

    Parameters
    ----------
    V : np.ndarray
        Basis matrix, shape (T, K) where T is the number of wavelength bins
        and K is the number of components.
    W : np.ndarray
        Weight matrix, shape (N, K) where N is the number of sources.
    method : str, default="l1"
        Normalization method:
        - "l1": L1 normalization (mean = 1). Recommended for weight comparison.
        - "l2": L2 normalization (norm = 1). Standard for orthonormal bases.

    Returns
    -------
    V_norm : np.ndarray
        Normalized basis matrix, shape (T, K).
    W_norm : np.ndarray
        Normalized weight matrix, shape (N, K).
    scales : np.ndarray
        Scale factors, shape (K,). Useful for potential denormalization.

    Raises
    ------
    ValueError
        If an unknown normalization method is specified.
    ValueError
        If V and W have incompatible shapes.

    See Also
    --------
    normalize_basis_l1 : L1 normalization (mean = 1).
    normalize_basis_l2 : L2 normalization (norm = 1).

    Examples
    --------
    >>> import numpy as np
    >>> from spxdictlearn import normalize_basis
    >>> V = np.random.rand(100, 5)  # 100 wavelength bins, 5 components
    >>> W = np.random.rand(50, 5)   # 50 sources
    >>> V_norm, W_norm, scales = normalize_basis(V, W, method="l1")
    >>> np.mean(V_norm, axis=0)  # All close to 1.0
    array([1., 1., 1., 1., 1.])

    Notes
    -----
    **When to use L1 vs L2:**

    - **L1 (default)**: For astronomical spectra where average flux has physical
      meaning. Makes weights directly comparable for component importance.

    - **L2**: For PCA-like analysis or when orthonormality is desired.

    **Reconstruction Preservation:**

    Both methods preserve the reconstruction exactly:
        V @ W.T == V_norm @ W_norm.T

    This means normalized results can be used anywhere the original V, W were used.
    """
    method = method.lower()

    if method == "l1":
        return normalize_basis_l1(V, W)
    elif method == "l2":
        return normalize_basis_l2(V, W)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'l1' or 'l2'.")


def denormalize_basis(
    V_norm: np.ndarray,
    W_norm: np.ndarray,
    scales: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reverse normalization to recover original scale.

    Given normalized V_norm, W_norm and the scale factors from a previous
    normalization, recover the original V, W.

    Parameters
    ----------
    V_norm : np.ndarray
        Normalized basis matrix, shape (T, K).
    W_norm : np.ndarray
        Normalized weight matrix, shape (N, K).
    scales : np.ndarray
        Scale factors from the original normalization, shape (K,).

    Returns
    -------
    V : np.ndarray
        Original-scale basis matrix, shape (T, K).
    W : np.ndarray
        Original-scale weight matrix, shape (N, K).

    Examples
    --------
    >>> import numpy as np
    >>> from spxdictlearn import normalize_basis_l1, denormalize_basis
    >>> V_orig = np.random.rand(100, 5)
    >>> W_orig = np.random.rand(50, 5)
    >>> V_norm, W_norm, scales = normalize_basis_l1(V_orig, W_orig)
    >>> V_recovered, W_recovered = denormalize_basis(V_norm, W_norm, scales)
    >>> np.allclose(V_orig, V_recovered)
    True
    >>> np.allclose(W_orig, W_recovered)
    True

    Notes
    -----
    This function is useful when you need to:
    - Convert normalized weights back to physical units
    - Compare with other results that use original scaling
    - Debug normalization effects
    """
    V_norm = np.asarray(V_norm, dtype=np.float64)
    W_norm = np.asarray(W_norm, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)

    # Reverse the normalization
    V = V_norm * scales[np.newaxis, :]
    W = W_norm / scales[np.newaxis, :]

    return V, W
