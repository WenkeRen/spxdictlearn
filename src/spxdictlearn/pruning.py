"""
Dictionary Pruning for ALS-WNMF Spectral Components.

This module provides utilities for identifying and merging similar spectral
components extracted by ALS-WNMF, addressing the "twin spectra" problem
before HALS refinement.

Key functions:
- compute_cosine_similarity_matrix: Pairwise cosine similarity
- cluster_similar_components: Hierarchical clustering
- merge_components_by_cluster: Weighted averaging merge
- prune_and_sort_dictionary: Complete pruning pipeline
"""

from typing import Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform


def compute_cosine_similarity_matrix(V: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between spectral components.

    S_ij = (v_i · v_j) / (||v_i|| ||v_j||)

    Parameters
    ----------
    V : np.ndarray
        Basis spectra matrix, shape (T, K) where T is number of wavelength
        bins and K is number of components

    Returns
    -------
    S : np.ndarray
        Similarity matrix, shape (K, K). Diagonal elements are 1.0.
        Values range from -1 to 1 (for normalized vectors).

    Notes
    -----
    Uses L2 normalization for numerical stability. Components with zero
    norm (all zeros) will have similarity 0 with all other components.
    """
    T, K = V.shape

    # Compute L2 norms for each component
    norms = np.linalg.norm(V, axis=0, keepdims=True)  # Shape: (1, K)

    # Avoid division by zero for zero-norm components
    norms = np.where(norms > 0, norms, 1.0)

    # Normalize each component to unit length
    V_normalized = V / norms  # Shape: (T, K)

    # Compute cosine similarity: S = V^T @ V (for normalized V)
    S = V_normalized.T @ V_normalized  # Shape: (K, K)

    # Ensure diagonal is exactly 1.0 (numerical precision)
    np.fill_diagonal(S, 1.0)

    # Clip to valid range [-1, 1] (handles numerical errors)
    S = np.clip(S, -1.0, 1.0)

    return S


def cluster_similar_components(
    V: np.ndarray,
    similarity_threshold: float = 0.95,
    method: str = "complete",
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """
    Use hierarchical clustering to group similar spectral components.

    Parameters
    ----------
    V : np.ndarray
        Basis spectra matrix, shape (T, K)
    similarity_threshold : float
        Components with cosine similarity >= this threshold are merged.
        Default: 0.95 (very similar)
        Typical range: 0.90 - 0.99
    method : str
        Linkage method for hierarchical clustering:
        - 'complete': Maximum distance (most conservative, default)
        - 'average': Average distance
        - 'single': Minimum distance (most aggressive)
    smooth_sigma : float
        Gaussian smoothing sigma for similarity computation (proxy smoothing).
        If > 0, applies 1D Gaussian filter along wavelength axis before
        computing cosine similarity to reduce high-frequency noise sensitivity.
        The clustering labels are then applied to the original (unsmoothed) V.
        Default: 0.0 (no smoothing)
        Typical range: 1.0 - 3.0

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each component, shape (K,)
        Labels are integers starting from 0.

    Notes
    -----
    Uses complete linkage on (1 - similarity) as the distance metric.
    This ensures that all components in a cluster have similarity >= threshold
    with every other component in the same cluster.

    The smooth_sigma parameter enables "proxy smoothing":
    - Similarity is computed on smoothed spectra (noise-insensitive)
    - Clustering labels are used to merge the original (noisy) spectra
    - Statistical averaging during merge naturally cancels random noise

    Examples
    --------
    >>> import numpy as np
    >>> # Create 5 components, where components 0 and 1 are identical
    >>> V = np.random.rand(100, 5)
    >>> V[:, 1] = V[:, 0]  # Make component 1 identical to component 0
    >>> labels = cluster_similar_components(V, similarity_threshold=0.99)
    >>> # Components 0 and 1 will have the same label
    """
    K = V.shape[1]

    if K <= 1:
        return np.array([0])

    # Apply Gaussian smoothing if sigma > 0 (proxy smoothing for similarity)
    if smooth_sigma > 0:
        V_for_dist = gaussian_filter1d(V, sigma=smooth_sigma, axis=0)
    else:
        V_for_dist = V

    # Compute cosine similarity matrix on (potentially smoothed) V
    S = compute_cosine_similarity_matrix(V_for_dist)

    # Convert similarity to distance: d = 1 - S
    # This gives distance in range [0, 2]
    distance_matrix = 1.0 - S

    # Ensure distance matrix is non-negative (handle numerical errors)
    distance_matrix = np.maximum(distance_matrix, 0.0)

    # Convert to condensed distance matrix for scipy
    # scipy.cluster.hierarchy.linkage expects condensed form
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method=method)

    # Cut dendrogram at distance threshold
    # Components with distance < (1 - threshold) are in same cluster
    distance_threshold = 1.0 - similarity_threshold
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Convert to 0-indexed labels
    labels = labels - 1

    return labels


def merge_components_by_cluster(
    V: np.ndarray,
    W: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Merge spectral components in the same cluster by weighted averaging.

    Merging rules:
    - V_merged: Weighted average of spectral shapes, weighted by global activation
    - W_merged: Sum of weights (preserves total activation)

    Parameters
    ----------
    V : np.ndarray
        Original basis spectra, shape (T, K_old)
    W : np.ndarray
        Original weight matrix, shape (N, K_old)
    labels : np.ndarray
        Cluster assignment for each component, shape (K_old,)

    Returns
    -------
    V_merged : np.ndarray
        Merged basis spectra, shape (T, K_new) where K_new = n_unique_labels
    W_merged : np.ndarray
        Merged weight matrix, shape (N, K_new)
    merge_info : dict
        Metadata about the merge process:
        - 'n_old': K_old (original number of components)
        - 'n_new': K_new (new number of components)
        - 'merge_mapping': Dict mapping new_index -> list of old indices
        - 'global_weights_old': Global weights before merge
        - 'global_weights_new': Global weights after merge

    Notes
    -----
    The weighted averaging formula for V:
        V_merged[:, c] = Σ_{k ∈ c} (w_global[k] * V[:, k]) / Σ_{k ∈ c} w_global[k]

    where w_global[k] = Σ_n W[n, k] is the total activation of component k.

    This ensures that components with higher activation have more influence
    on the merged spectrum shape.

    Examples
    --------
    >>> import numpy as np
    >>> V = np.random.rand(100, 4)  # 4 components
    >>> W = np.random.rand(50, 4)   # 50 sources
    >>> labels = np.array([0, 0, 1, 1])  # Merge 0+1 and 2+3
    >>> V_new, W_new, info = merge_components_by_cluster(V, W, labels)
    >>> print(f"Components: {V.shape[1]} -> {V_new.shape[1]}")
    """
    T, K_old = V.shape
    N = W.shape[0]

    # Get unique cluster labels (sorted)
    unique_labels = np.unique(labels)
    K_new = len(unique_labels)

    # Compute global weights for each component: w_global[k] = Σ_n W[n, k]
    global_weights = np.sum(W, axis=0)  # Shape: (K_old,)

    # Initialize merged matrices
    V_merged = np.zeros((T, K_new))
    W_merged = np.zeros((N, K_new))

    # Build merge mapping
    merge_mapping = {}

    for new_idx, cluster_label in enumerate(unique_labels):
        # Find all components in this cluster
        old_indices = np.where(labels == cluster_label)[0]
        merge_mapping[int(new_idx)] = old_indices.tolist()

        if len(old_indices) == 1:
            # No merging needed, just copy
            k = old_indices[0]
            V_merged[:, new_idx] = V[:, k]
            W_merged[:, new_idx] = W[:, k]
        else:
            # Weighted averaging for V
            cluster_weights = global_weights[old_indices]
            total_weight = np.sum(cluster_weights)

            if total_weight > 0:
                # Normalize weights for averaging
                norm_weights = cluster_weights / total_weight
                V_merged[:, new_idx] = np.sum(V[:, old_indices] * norm_weights, axis=1)
            else:
                # All weights are zero, use simple average
                V_merged[:, new_idx] = np.mean(V[:, old_indices], axis=1)

            # Sum weights for W (preserves total activation)
            W_merged[:, new_idx] = np.sum(W[:, old_indices], axis=1)

    # Compute new global weights
    global_weights_new = np.sum(W_merged, axis=0)

    # Build merge info
    merge_info = {
        "n_old": K_old,
        "n_new": K_new,
        "merge_mapping": merge_mapping,
        "global_weights_old": global_weights.tolist(),
        "global_weights_new": global_weights_new.tolist(),
    }

    return V_merged, W_merged, merge_info


def sort_components_by_weight(
    V: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort spectral components by total weight (descending order).

    Parameters
    ----------
    V : np.ndarray
        Basis spectra, shape (T, K)
    W : np.ndarray
        Weight matrix, shape (N, K)

    Returns
    -------
    V_sorted : np.ndarray
        Sorted basis spectra, shape (T, K)
    W_sorted : np.ndarray
        Sorted weight matrix, shape (N, K)
    sort_indices : np.ndarray
        Indices that map old -> new order: V_sorted = V[:, sort_indices]
    """
    # Compute global weights
    global_weights = np.sum(W, axis=0)  # Shape: (K,)

    # Sort by descending weight
    sort_indices = np.argsort(global_weights)[::-1]

    # Apply sorting
    V_sorted = V[:, sort_indices]
    W_sorted = W[:, sort_indices]

    return V_sorted, W_sorted, sort_indices


def prune_and_sort_dictionary(
    V: np.ndarray,
    W: np.ndarray,
    similarity_threshold: float = 0.95,
    method: str = "complete",
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Complete pruning pipeline for spectral dictionary.

    Steps:
    1. Compute cosine similarity matrix
    2. Hierarchical clustering of similar components
    3. Merge by weighted averaging
    4. Sort by total weight (descending)

    Parameters
    ----------
    V : np.ndarray
        Original basis spectra, shape (T, K_old)
    W : np.ndarray
        Original weight matrix, shape (N, K_old)
    similarity_threshold : float
        Components with cosine similarity >= this threshold are merged.
        Default: 0.95
        Typical range: 0.90 - 0.99
    method : str
        Linkage method for hierarchical clustering ('complete', 'average', 'single')
    smooth_sigma : float
        Gaussian smoothing sigma for similarity computation (proxy smoothing).
        If > 0, applies 1D Gaussian filter along wavelength axis before
        computing cosine similarity to reduce high-frequency noise sensitivity.
        The clustering labels are then applied to the original (unsmoothed) V.
        Default: 0.0 (no smoothing)
        Typical range: 1.0 - 3.0

    Returns
    -------
    V_pruned : np.ndarray
        Pruned basis spectra, shape (T, K_new) where K_new <= K_old
    W_pruned : np.ndarray
        Pruned weight matrix, shape (N, K_new)
    pruning_info : dict
        Complete metadata about the pruning process:
        - 'similarity_threshold': Input threshold
        - 'clustering_method': Linkage method
        - 'smooth_sigma': Smoothing sigma used
        - 'similarity_matrix': K_old x K_old similarity matrix
        - 'cluster_labels': Original cluster assignments
        - 'merge_info': From merge_components_by_cluster()
        - 'sort_indices': Final sorting indices
        - 'n_components_removed': K_old - K_new

    Notes
    -----
    This is the main entry point for dictionary pruning. Use this function
    to prepare ALS-WNMF results for HALS refinement.

    The pruning reduces K (number of components) while preserving:
    - Total activation: ΣW is approximately conserved
    - Spectral information: Merged spectra are weighted averages

    The smooth_sigma parameter enables "proxy smoothing":
    - Similarity is computed on smoothed spectra (noise-insensitive)
    - Clustering labels are used to merge the original (noisy) spectra
    - Statistical averaging during merge naturally cancels random noise

    Examples
    --------
    >>> import numpy as np
    >>> # Load ALS-WNMF results
    >>> V = np.load('results/02_alswnmf/V_estimated.npy')  # (2048, 40)
    >>> W = np.load('results/02_alswnmf/W_estimated.npy')  # (2026, 40)
    >>> # Prune dictionary with smoothing for robust similarity
    >>> V_pruned, W_pruned, info = prune_and_sort_dictionary(V, W, 0.95, smooth_sigma=2.0)
    >>> print(f"Components: {V.shape[1]} -> {V_pruned.shape[1]}")
    >>> # Save for HALS
    >>> np.save('results/03_pruning/V_pruned.npy', V_pruned)
    >>> np.save('results/03_pruning/W_pruned.npy', W_pruned)
    """
    T, K_old = V.shape
    N = W.shape[0]

    # Step 1: Compute similarity matrix on (potentially smoothed) V
    if smooth_sigma > 0:
        V_for_sim = gaussian_filter1d(V, sigma=smooth_sigma, axis=0)
    else:
        V_for_sim = V
    S = compute_cosine_similarity_matrix(V_for_sim)

    # Step 2: Cluster similar components (pass smooth_sigma for consistency)
    labels = cluster_similar_components(V, similarity_threshold, method, smooth_sigma)

    # Step 3: Merge components by cluster
    V_merged, W_merged, merge_info = merge_components_by_cluster(V, W, labels)

    # Step 4: Sort by weight (descending)
    V_sorted, W_sorted, sort_indices = sort_components_by_weight(V_merged, W_merged)

    # Build complete pruning info
    K_new = V_sorted.shape[1]
    pruning_info = {
        "similarity_threshold": similarity_threshold,
        "clustering_method": method,
        "smooth_sigma": smooth_sigma,
        "similarity_matrix": S.tolist(),
        "cluster_labels": labels.tolist(),
        "merge_info": merge_info,
        "sort_indices": sort_indices.tolist(),
        "n_components_removed": K_old - K_new,
        "n_components_old": K_old,
        "n_components_new": K_new,
    }

    return V_sorted, W_sorted, pruning_info


def compute_max_pairwise_similarity(S: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute the maximum pairwise similarity (excluding diagonal).

    Parameters
    ----------
    S : np.ndarray
        Similarity matrix, shape (K, K)

    Returns
    -------
    max_sim : float
        Maximum off-diagonal similarity
    i : int
        Row index of maximum
    j : int
        Column index of maximum

    Notes
    -----
    Useful for diagnosing "twin spectra" issues.
    """
    K = S.shape[0]

    if K <= 1:
        return 0.0, 0, 0

    # Create mask for off-diagonal elements
    mask = ~np.eye(K, dtype=bool)

    # Find maximum off-diagonal similarity
    S_masked = np.where(mask, S, -np.inf)
    max_sim = np.max(S_masked)
    i, j = np.unravel_index(np.argmax(S_masked), S.shape)

    return max_sim, i, j
