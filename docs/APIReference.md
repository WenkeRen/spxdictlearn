# API Reference: spxdictlearn

**Version:** 0.4.1
**Last Updated:** 2026-03-04

This document provides comprehensive API documentation for all public functions in the spxdictlearn package.

---

## Table of Contents

- [Main Algorithm](#main-algorithm)
  - [als_wnmf](#als_wnmf)
- [HALS Functions](#hals-functions)
  - [hals_wnmf](#hals_wnmf)
  - [precompute_hals_constants](#precompute_hals_constants)
  - [compute_M_tensor](#compute_m_tensor)
  - [m_step_hals](#m_step_hals)
- [Pruning Functions](#pruning-functions)
  - [prune_and_sort_dictionary](#prune_and_sort_dictionary)
  - [compute_cosine_similarity_matrix](#compute_cosine_similarity_matrix)
  - [cluster_similar_components](#cluster_similar_components)
  - [merge_components_by_cluster](#merge_components_by_cluster)
- [Normalization Functions](#normalization-functions)
  - [normalize_basis](#normalize_basis)
  - [normalize_basis_l1](#normalize_basis_l1)
  - [normalize_basis_l2](#normalize_basis_l2)
  - [denormalize_basis](#denormalize_basis)
- [Numba Functions](#numba-functions)
  - [nnls_pgd](#nnls_pgd)
  - [nnls_pgd_fallback](#nnls_pgd_fallback)
  - [NUMBA_AVAILABLE](#numba_available)
- [E-step Functions](#e-step-functions)
  - [e_step](#e_step)
  - [e_step_single_source](#e_step_single_source)
- [M-step Functions](#m-step-functions)
  - [m_step](#m_step)
- [Loss Computation](#loss-computation)
  - [compute_regularized_loss](#compute_regularized_loss)
- [Response Matrix](#response-matrix)
  - [build_target_grid](#build_target_grid)
  - [build_response_matrix](#build_response_matrix)
  - [build_all_response_matrices](#build_all_response_matrices)
- [Mock Data Generation](#mock-data-generation)
  - [generate_mock_data](#generate_mock_data)
  - [generate_true_basis](#generate_true_basis)
  - [generate_true_weights](#generate_true_weights)
- [Utility Functions](#utility-functions)
  - [initialize_parameters](#initialize_parameters)
  - [compute_global_chi2](#compute_global_chi2)
  - [compute_smoothness_penalty](#compute_smoothness_penalty)
  - [compute_second_order_smoothness_penalty](#compute_second_order_smoothness_penalty)
  - [validate_non_negativity](#validate_non_negativity)

---

## Main Algorithm

### als_wnmf

```python
als_wnmf(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    K: int,
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
    e_step_method: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[float]]
```

Alternating Least Squares Weighted Non-Negative Matrix Factorization.

This is the main algorithm that extracts global basis spectra V and source weights W from heterogeneous observational data.

**Algorithm:**
1. Initialize V and W with random uniform values
2. Loop until convergence:
   - E-step: Update W using NNLS (parallel)
   - M-step: Update V using MUR with regularization
   - Check convergence: |ΔLoss| / Loss < tol

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sources_data` | `List[np.ndarray]` | required | List of D_n arrays for all N sources. Each D_n has shape (M_n, 4) |
| `response_matrices` | `List[csr_matrix]` | required | List of pre-computed R_n matrices |
| `K` | `int` | required | Number of non-negative components to extract |
| `T` | `int` | 4096 | Number of wavelength bins |
| `lambda_min` | `float` | 0.75 | Minimum wavelength for target grid (μm) |
| `lambda_max` | `float` | 5.0 | Maximum wavelength for target grid (μm) |
| `alpha` | `float \| None` | None | L2 regularization coefficient. If None, sets to 0.01 (normalized) or 0.1*N (unnormalized) |
| `beta` | `float` | 0.0 | First-order smoothness coefficient |
| `gamma` | `float` | 0.0 | Second-order smoothness (curvature) coefficient |
| `normalize` | `bool` | True | Apply normalization to regularization terms |
| `tol` | `float` | 1e-4 | Convergence tolerance for relative Loss change |
| `max_iter` | `int` | 100 | Maximum number of iterations |
| `n_jobs` | `int` | -1 | Number of parallel jobs for E-step (-1 = all CPUs) |
| `e_step_method` | `str` | "auto" | NNLS solver: 'auto', 'numba' (GIL-free), or 'scipy' |
| `seed` | `int` | 42 | Random seed for initialization |
| `verbose` | `bool` | True | Print iteration progress |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `V` | `np.ndarray` | Global basis matrix (eigenspectra), shape (T, K), non-negative |
| `W` | `np.ndarray` | Weight matrix, shape (N, K), non-negative |
| `loss_history` | `List[float]` | Total loss at each iteration |

**Example:**

```python
from spxcore import als_wnmf, generate_mock_data, build_all_response_matrices, build_target_grid

# Generate or load data
sources_data, response_matrices, V_true, W_true = generate_mock_data(N=50)

# Run ALS-WNMF with default regularization
V, W, loss_history = als_wnmf(
    sources_data=sources_data,
    response_matrices=response_matrices,
    K=3,
    alpha=0.01,    # L2 regularization
    beta=0.1,      # First-order smoothness
    gamma=0.0,     # No curvature penalty
    normalize=True
)

# With second-order smoothness (curvature penalty)
V, W, loss_history = als_wnmf(
    sources_data=sources_data,
    response_matrices=response_matrices,
    K=5,
    alpha=0.01,
    beta=0.1,
    gamma=0.1,     # Penalize curvature
    normalize=True
)
```

---

## HALS Functions

### hals_wnmf

```python
hals_wnmf(
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
    e_step_method: str = "auto",
    warm_start: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[float]]
```

Hierarchical Alternating Least Squares Weighted NMF.

HALS addresses the "twin spectra" problem through sequential deflation updates. Each component V[:,k] is updated while holding others fixed, naturally breaking symmetry.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sources_data` | `List[np.ndarray]` | required | List of D_n arrays for all N sources |
| `response_matrices` | `List[csr_matrix]` | required | List of R_n matrices |
| `V_init` | `np.ndarray` | required | Initial basis matrix (typically from als_wnmf), shape (T, K) |
| `W_init` | `np.ndarray` | required | Initial weight matrix, shape (N, K) |
| `alpha` | `float` | 0.0 | L2 regularization coefficient |
| `beta` | `float` | 0.0 | First-order smoothness coefficient |
| `gamma` | `float` | 0.0 | Second-order smoothness coefficient |
| `normalize` | `bool` | True | Apply normalization to regularization terms |
| `tol` | `float` | 1e-5 | Convergence tolerance |
| `max_iter` | `int` | 100 | Maximum iterations |
| `n_jobs` | `int` | -1 | Parallel jobs for E-step |
| `n_inner_iter` | `int` | 5 | PGD iterations per component update |
| `e_step_method` | `str` | "auto" | NNLS solver: 'auto', 'numba', or 'scipy' |
| `warm_start` | `bool` | True | Reuse precomputed constants across iterations |
| `verbose` | `bool` | True | Print iteration progress |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `V` | `np.ndarray` | Refined basis matrix, shape (T, K) |
| `W` | `np.ndarray` | Refined weight matrix, shape (N, K) |
| `loss_history` | `List[float]` | Total loss at each iteration |

**Example:**

```python
from spxdictlearn import als_wnmf, prune_and_sort_dictionary, hals_wnmf

# Step 1: Run ALS-WNMF
V_als, W_als, _ = als_wnmf(sources_data, response_matrices, K=40)

# Step 2: Prune similar components
V_pruned, W_pruned, info = prune_and_sort_dictionary(V_als, W_als, 0.95)

# Step 3: HALS refinement
V_final, W_final, loss = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_pruned, W_init=W_pruned,
    alpha=0.01, beta=0.1, n_inner_iter=5
)
```

---

### precompute_hals_constants

```python
precompute_hals_constants(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    N: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Precompute C matrix and global B data for BLAS-optimized HALS.

Computes constants ONCE before HALS iterations:
- C_n = R_n^T @ Σ_n^(-1) @ y_n
- B_n = R_n^T @ Σ_n^(-1) @ R_n (aligned for BLAS)

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `sources_data` | `List[np.ndarray]` | List of D_n arrays |
| `response_matrices` | `List[csr_matrix]` | List of R_n matrices |
| `N` | `int` | Number of sources |
| `T` | `int` | Number of wavelength bins |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `C` | `np.ndarray` | First-order constants, shape (N, T) |
| 1 | `global_B_data` | `np.ndarray` | Aligned B_n data, shape (N, nnz_global) |
| 2 | `global_indices` | `np.ndarray` | CSR column indices |
| 3 | `global_indptr` | `np.ndarray` | CSR row pointers |

---

### compute_M_tensor

```python
compute_M_tensor(
    W: np.ndarray,
    global_B_data: np.ndarray,
    global_indices: np.ndarray,
    global_indptr: np.ndarray,
    K: int,
) -> np.ndarray
```

Compute aggregation tensor M using BLAS-optimized matmul.

M^(j,k) = Σ_n w_nj * w_nk * B_n

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `W` | `np.ndarray` | Weight matrix, shape (N, K) |
| `global_B_data` | `np.ndarray` | Aligned B_n data from precompute_hals_constants |
| `global_indices` | `np.ndarray` | CSR column indices |
| `global_indptr` | `np.ndarray` | CSR row pointers |
| `K` | `int` | Number of components |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `M_tensor` | `np.ndarray` | Aggregation tensor, shape (K, K, T) |

---

### m_step_hals

```python
m_step_hals(
    V: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
    M_tensor: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
    n_inner_iter: int = 5,
) -> np.ndarray
```

HALS M-step: Sequential component updates with PGD.

Updates each component V[:,k] using deflation:
U_k = P - Σ_{j≠k} M^(j,k) @ V[:,j]

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `V` | `np.ndarray` | required | Current basis matrix |
| `W` | `np.ndarray` | required | Current weight matrix |
| `C` | `np.ndarray` | required | Precomputed constants from precompute_hals_constants |
| `M_tensor` | `np.ndarray` | required | Aggregation tensor from compute_M_tensor |
| `alpha` | `float` | 0.0 | L2 regularization |
| `beta` | `float` | 0.0 | First-order smoothness |
| `gamma` | `float` | 0.0 | Second-order smoothness |
| `normalize` | `bool` | True | Apply normalization |
| `n_inner_iter` | `int` | 5 | PGD iterations per component |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `V_new` | `np.ndarray` | Updated basis matrix |

---

## Pruning Functions

### prune_and_sort_dictionary

```python
prune_and_sort_dictionary(
    V: np.ndarray,
    W: np.ndarray,
    similarity_threshold: float = 0.95,
    method: str = "complete",
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]
```

Complete pruning pipeline for removing similar spectral components.

Computes cosine similarity, clusters components, merges similar ones, and sorts by importance.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `V` | `np.ndarray` | required | Basis matrix, shape (T, K) |
| `W` | `np.ndarray` | required | Weight matrix, shape (N, K) |
| `similarity_threshold` | `float` | 0.95 | Merge components with S >= threshold |
| `method` | `str` | "complete" | Linkage method: 'complete', 'average', or 'single' |
| `smooth_sigma` | `float` | 0.0 | Gaussian smoothing sigma for proxy smoothing. If > 0, applies 1D Gaussian filter to V before computing similarity. Typical range: 1.0 - 3.0 |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `V_pruned` | `np.ndarray` | Pruned basis matrix, shape (T, K_new) |
| 1 | `W_pruned` | `np.ndarray` | Pruned weight matrix, shape (N, K_new) |
| 2 | `info` | `dict` | Metadata (clusters, K_before, K_after, importance, smooth_sigma) |

**Proxy Smoothing Strategy:**

When `smooth_sigma > 0`, the function implements "proxy smoothing":
1. Apply Gaussian filter to V along wavelength axis (noise reduction)
2. Compute cosine similarity on smoothed spectra
3. Apply clustering labels to **original** (unsmoothed) V
4. Merge uses original components; statistical averaging cancels noise

This ensures grouping decisions are based on spectral shapes, not noise artifacts.

**Example:**

```python
from spxdictlearn import als_wnmf, prune_and_sort_dictionary

V, W, _ = als_wnmf(sources_data, response_matrices, K=40)
print(f"Original: {V.shape[1]} components")

# Basic pruning
V_pruned, W_pruned, info = prune_and_sort_dictionary(V, W, 0.95)
print(f"Pruned: {info['K_before']} → {info['K_after']} components")

# With proxy smoothing for noisy spectra
V_pruned, W_pruned, info = prune_and_sort_dictionary(
    V, W, similarity_threshold=0.95, smooth_sigma=2.0
)
```

---

### compute_cosine_similarity_matrix

```python
compute_cosine_similarity_matrix(
    V: np.ndarray,
) -> np.ndarray
```

Compute pairwise cosine similarity between spectral components.

S_ij = (v_i · v_j) / (||v_i|| ||v_j||)

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `V` | `np.ndarray` | Basis matrix, shape (T, K) |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `S` | `np.ndarray` | Similarity matrix, shape (K, K), range [-1, 1] |

---

### cluster_similar_components

```python
cluster_similar_components(
    V: np.ndarray,
    similarity_threshold: float = 0.95,
    method: str = "complete",
    smooth_sigma: float = 0.0,
) -> np.ndarray
```

Use hierarchical clustering to group similar spectral components.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `V` | `np.ndarray` | required | Basis matrix, shape (T, K) |
| `similarity_threshold` | `float` | 0.95 | Components with S >= threshold are merged |
| `method` | `str` | "complete" | Linkage method |
| `smooth_sigma` | `float` | 0.0 | Gaussian smoothing sigma for proxy smoothing. If > 0, applies 1D Gaussian filter to V before computing similarity. Typical range: 1.0 - 3.0 |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `labels` | `np.ndarray` | Cluster labels, shape (K,) |

**Proxy Smoothing:**

When `smooth_sigma > 0`, similarity is computed on smoothed spectra to reduce sensitivity to high-frequency noise. Clustering labels are applied to the original (unsmoothed) V.

---

### merge_components_by_cluster

```python
merge_components_by_cluster(
    V: np.ndarray,
    W: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]
```

Merge components within clusters using weighted averaging.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `V` | `np.ndarray` | Basis matrix, shape (T, K) |
| `W` | `np.ndarray` | Weight matrix, shape (N, K) |
| `labels` | `np.ndarray` | Cluster labels from cluster_similar_components |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `V_merged` | `np.ndarray` | Merged basis matrix |
| 1 | `W_merged` | `np.ndarray` | Merged weight matrix |

---

## Normalization Functions

### normalize_basis

```python
normalize_basis(
    V: np.ndarray,
    W: np.ndarray,
    method: str = "l1",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

General normalization interface for basis spectra and weights.

This is the recommended entry point for normalization, providing a unified interface to different normalization methods. Normalization makes weights comparable across components by standardizing the amplitude of each basis spectrum.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `V` | `np.ndarray` | required | Basis matrix, shape (T, K) |
| `W` | `np.ndarray` | required | Weight matrix, shape (N, K) |
| `method` | `str` | "l1" | Normalization method: 'l1' (mean=1) or 'l2' (norm=1) |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `V_norm` | `np.ndarray` | Normalized basis matrix, shape (T, K) |
| 1 | `W_norm` | `np.ndarray` | Normalized weight matrix, shape (N, K) |
| 2 | `scales` | `np.ndarray` | Scale factors, shape (K,) |

**Example:**

```python
from spxdictlearn import normalize_basis

# L1 normalization (default, recommended for weight comparison)
V_norm, W_norm, scales = normalize_basis(V, W, method="l1")

# L2 normalization (for orthonormal basis)
V_norm, W_norm, scales = normalize_basis(V, W, method="l2")

# Verify reconstruction is preserved
assert np.allclose(V @ W.T, V_norm @ W_norm.T)
```

---

### normalize_basis_l1

```python
normalize_basis_l1(
    V: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

L1 normalization: mean(V[:, k]) = 1 for all components k.

This normalization scales each component so that its mean value equals 1, and adjusts the corresponding weights to preserve reconstruction. Recommended for astronomical spectra where average flux has physical meaning.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `V` | `np.ndarray` | Basis matrix, shape (T, K) |
| `W` | `np.ndarray` | Weight matrix, shape (N, K) |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `V_norm` | `np.ndarray` | Normalized basis matrix with mean=1 per column |
| 1 | `W_norm` | `np.ndarray` | Normalized weight matrix |
| 2 | `scales` | `np.ndarray` | L1 scale factors (column means) |

**Example:**

```python
from spxdictlearn import normalize_basis_l1, prune_and_sort_dictionary

# Step 1: Normalize basis spectra (L1: mean = 1)
V_norm, W_norm, scales = normalize_basis_l1(V, W)

# Step 2: Now weights are comparable - use for pruning
V_pruned, W_pruned, info = prune_and_sort_dictionary(
    V_norm, W_norm, similarity_threshold=0.85
)
```

---

### normalize_basis_l2

```python
normalize_basis_l2(
    V: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

L2 normalization: ||V[:, k]||_2 = 1 for all components k.

This normalization scales each component to unit L2 norm, making the basis vectors orthonormal in magnitude. Useful for PCA-like interpretations.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `V` | `np.ndarray` | Basis matrix, shape (T, K) |
| `W` | `np.ndarray` | Weight matrix, shape (N, K) |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `V_norm` | `np.ndarray` | Normalized basis matrix with ||v_k||_2 = 1 |
| 1 | `W_norm` | `np.ndarray` | Normalized weight matrix |
| 2 | `scales` | `np.ndarray` | L2 scale factors (column norms) |

---

### denormalize_basis

```python
denormalize_basis(
    V_norm: np.ndarray,
    W_norm: np.ndarray,
    scales: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]
```

Reverse normalization to recover original scale.

Given normalized V_norm, W_norm and the scale factors from a previous normalization, recover the original V, W.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `V_norm` | `np.ndarray` | Normalized basis matrix, shape (T, K) |
| `W_norm` | `np.ndarray` | Normalized weight matrix, shape (N, K) |
| `scales` | `np.ndarray` | Scale factors from original normalization, shape (K,) |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `V` | `np.ndarray` | Original-scale basis matrix |
| 1 | `W` | `np.ndarray` | Original-scale weight matrix |

**Example:**

```python
from spxdictlearn import normalize_basis_l1, denormalize_basis

# Normalize
V_norm, W_norm, scales = normalize_basis_l1(V, W)

# ... do some processing ...

# Recover original scale
V_recovered, W_recovered = denormalize_basis(V_norm, W_norm, scales)
assert np.allclose(V, V_recovered)
assert np.allclose(W, W_recovered)
```

---

## Numba Functions

### nnls_pgd

```python
nnls_pgd(
    A: np.ndarray,
    b: np.ndarray,
    x_init: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray
```

GIL-free Non-Negative Least Squares using Projected Gradient Descent.

Requires numba to be installed. Falls back to nnls_pgd_fallback if unavailable.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `A` | `np.ndarray` | required | Design matrix, shape (M, K) |
| `b` | `np.ndarray` | required | Target vector, shape (M,) |
| `x_init` | `np.ndarray \| None` | None | Initial guess (warm start) |
| `max_iter` | `int` | 200 | Maximum iterations |
| `tol` | `float` | 1e-6 | Convergence tolerance |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `x` | `np.ndarray` | Solution vector, shape (K,), non-negative |

---

### nnls_pgd_fallback

```python
nnls_pgd_fallback(
    A: np.ndarray,
    b: np.ndarray,
    x_init: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray
```

Pure NumPy implementation of NNLS-PGD (no JIT compilation).

Used when numba is not installed.

**Parameters:** Same as `nnls_pgd`

**Returns:** Same as `nnls_pgd`

---

### NUMBA_AVAILABLE

```python
NUMBA_AVAILABLE: bool
```

Feature flag indicating whether numba is installed.

**Example:**

```python
from spxdictlearn import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    print("Numba acceleration enabled")
else:
    print("Using scipy.optimize.nnls (install with: pip install spxdictlearn[numba])")
```

---

## E-step Functions

### e_step

```python
e_step(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V: np.ndarray,
    n_jobs: int = -1,
    verbose: bool = True,
) -> np.ndarray
```

E-step: Update weights W for all sources in parallel.

For each source n, solves independent NNLS problem:
$$\min_{w_n \ge 0} || A_n @ w_n - b_n ||_2^2$$

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sources_data` | `List[np.ndarray]` | required | List of D_n arrays for all sources |
| `response_matrices` | `List[csr_matrix]` | required | List of R_n matrices for all sources |
| `V` | `np.ndarray` | required | Global basis matrix (fixed), shape (T, K) |
| `n_jobs` | `int` | -1 | Number of parallel jobs (-1 uses all CPUs) |
| `verbose` | `bool` | True | Show progress bar |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `W` | `np.ndarray` | Updated weight matrix, shape (N, K), non-negative |

---

### e_step_single_source

```python
e_step_single_source(
    D_n: np.ndarray,
    R_n: csr_matrix,
    V: np.ndarray,
) -> np.ndarray
```

E-step: Solve for weights w_n of a single source using NNLS.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `D_n` | `np.ndarray` | Raw observation data, shape (M_n, 4). Column 2: flux, Column 3: error |
| `R_n` | `csr_matrix` | Response matrix, shape (M_n, T) |
| `V` | `np.ndarray` | Global basis matrix (fixed), shape (T, K) |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `w_n` | `np.ndarray` | Weight vector for this source, shape (K,), non-negative |

---

## M-step Functions

### m_step

```python
m_step(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V: np.ndarray,
    W: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
    epsilon_reg: float = 1e-12,
) -> np.ndarray
```

M-step: Update global basis V using Multiplicative Update Rules (MUR).

MUR guarantees non-negativity and monotonic convergence:
$$V_{\text{new}} = V_{\text{old}} \odot \frac{P + P_{\text{reg}}}{Q + Q_{\text{reg}} + \epsilon}$$

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sources_data` | `List[np.ndarray]` | required | List of D_n arrays for all sources |
| `response_matrices` | `List[csr_matrix]` | required | List of R_n matrices for all sources |
| `V` | `np.ndarray` | required | Current basis matrix, shape (T, K) |
| `W` | `np.ndarray` | required | Fixed weight matrix, shape (N, K) |
| `alpha` | `float` | 0.0 | L2 regularization coefficient |
| `beta` | `float` | 0.0 | First-order smoothness coefficient |
| `gamma` | `float` | 0.0 | Second-order smoothness coefficient |
| `normalize` | `bool` | True | Apply normalization to regularization terms |
| `epsilon_reg` | `float` | 1e-12 | Small value to prevent division by zero |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `V_new` | `np.ndarray` | Updated basis matrix, shape (T, K), non-negative |

---

## Loss Computation

### compute_regularized_loss

```python
compute_regularized_loss(
    sources_data: List[np.ndarray],
    response_matrices: List[csr_matrix],
    V: np.ndarray,
    W: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    normalize: bool = True,
) -> Tuple[float, float, float, float, float]
```

Compute the regularized objective function with L2, smoothness, and curvature penalties.

$$\text{Loss} = \frac{\chi^2}{M_{\text{total}}} + \frac{\alpha}{TK} ||V||_F^2 + \frac{\beta}{(T-1)K} \cdot \text{Smoothness} + \frac{\gamma}{(T-2)K} \cdot \text{Curvature}$$

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sources_data` | `List[np.ndarray]` | required | List of D_n arrays for all sources |
| `response_matrices` | `List[csr_matrix]` | required | List of R_n matrices for all sources |
| `V` | `np.ndarray` | required | Global basis matrix, shape (T, K) |
| `W` | `np.ndarray` | required | Weight matrix, shape (N, K) |
| `alpha` | `float` | 0.0 | L2 regularization coefficient |
| `beta` | `float` | 0.0 | First-order smoothness coefficient |
| `gamma` | `float` | 0.0 | Second-order smoothness coefficient |
| `normalize` | `bool` | True | Apply normalization to all terms |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `total_loss` | `float` | Total loss = chi2_norm + l2_term + smooth_term + curvature_term |
| 1 | `chi2_norm` | `float` | Normalized chi-squared (reduced chi-squared = χ²/M_total) |
| 2 | `l2_term` | `float` | L2 penalty term |
| 3 | `smooth_term` | `float` | First-order smoothness penalty term |
| 4 | `curvature_term` | `float` | Second-order smoothness penalty term |

**Example:**

```python
from spxcore import compute_regularized_loss

total_loss, chi2, l2, smooth, curv = compute_regularized_loss(
    sources_data, response_matrices, V, W,
    alpha=0.01, beta=0.1, gamma=0.1
)
print(f"Total Loss: {total_loss:.4e}")
print(f"  χ²: {chi2:.4e}")
print(f"  L2: {l2:.4e}")
print(f"  Smooth: {smooth:.4e}")
print(f"  Curv: {curv:.4e}")
```

---

## Response Matrix

### build_target_grid

```python
build_target_grid(
    lambda_min: float = 0.75,
    lambda_max: float = 5.0,
    n_bins: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]
```

Build the target high-resolution logarithmic wavelength grid.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lambda_min` | `float` | 0.75 | Minimum wavelength (μm) |
| `lambda_max` | `float` | 5.0 | Maximum wavelength (μm) |
| `n_bins` | `int` | 4096 | Number of wavelength bins |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `edges` | `np.ndarray` | Bin edges, shape (n_bins+1,) in μm |
| 1 | `centers` | `np.ndarray` | Bin centers (geometric mean), shape (n_bins,) in μm |

---

### build_response_matrix

```python
build_response_matrix(
    D_n: np.ndarray,
    target_edges: np.ndarray,
) -> csr_matrix
```

Build a sparse Gaussian response matrix R_n for a single source.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `D_n` | `np.ndarray` | Raw observation data, shape (M_n, 4). Column 0: λ_c, Column 1: FWHM |
| `target_edges` | `np.ndarray` | Target grid bin edges, shape (T+1,) in μm |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `R_n` | `csr_matrix` | Sparse response matrix, shape (M_n, T), row sums = 1.0 |

**Example:**

```python
from spxcore.response_matrix import build_response_matrix, build_target_grid

# Build target grid
edges, centers = build_target_grid(n_bins=4096)

# Build response matrix for a source
R_n = build_response_matrix(D_n, edges)
print(f"Response matrix shape: {R_n.shape}")
print(f"Sparsity: {1 - R_n.nnz / (R_n.shape[0] * R_n.shape[1]):.4%}")
```

---

### build_all_response_matrices

```python
build_all_response_matrices(
    sources_data: list,
    target_edges: np.ndarray,
    verbose: bool = True,
) -> list
```

Build response matrices for all sources.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sources_data` | `list` | required | List of D_n arrays |
| `target_edges` | `np.ndarray` | required | Target grid bin edges |
| `verbose` | `bool` | True | Print progress |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `response_matrices` | `list` | List of R_n csr_matrix matrices |

---

## Mock Data Generation

### generate_mock_data

```python
generate_mock_data(
    N: int = 50,
    M_per_source: int = 500,
    K_true: int = 3,
    T: int = 4096,
    noise_level: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[csr_matrix], np.ndarray, np.ndarray]
```

Generate complete mock dataset for testing ALS-WNMF.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `N` | `int` | 50 | Number of sources |
| `M_per_source` | `int` | 500 | Observations per source |
| `K_true` | `int` | 3 | Number of true components |
| `T` | `int` | 4096 | Number of wavelength bins |
| `noise_level` | `float` | 0.05 | Noise as fraction of mean flux |
| `seed` | `int` | 42 | Random seed |
| `verbose` | `bool` | True | Print progress |

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `sources_data` | `List[np.ndarray]` | List of D_n arrays |
| 1 | `response_matrices` | `List[csr_matrix]` | List of R_n matrices |
| 2 | `V_true` | `np.ndarray` | True basis matrix, shape (T, K_true) |
| 3 | `W_true` | `np.ndarray` | True weight matrix, shape (N, K_true) |

**Example:**

```python
from spxcore import generate_mock_data, als_wnmf

# Generate mock data with known ground truth
sources_data, response_matrices, V_true, W_true = generate_mock_data(
    N=100,
    M_per_source=500,
    K_true=3,
    noise_level=0.05
)

# Run decomposition
V_est, W_est, loss_history = als_wnmf(
    sources_data, response_matrices, K=3
)

# Compare with ground truth
print(f"True V shape: {V_true.shape}")
print(f"Estimated V shape: {V_est.shape}")
```

---

### generate_true_basis

```python
generate_true_basis(
    target_edges: np.ndarray,
    K_true: int = 3,
) -> np.ndarray
```

Generate K_true distinct artificial spectra.

- **Component 0:** Broad Gaussian continuum (center=2.0 μm, σ=1.0 μm)
- **Component 1:** Power-law (f(λ) = λ^(-1.5))
- **Component 2:** Narrow emission lines (at 1.5, 2.0, 3.0, 4.0 μm)

**Returns:** `V_true` with shape (T, K_true)

---

### generate_true_weights

```python
generate_true_weights(
    N: int,
    K_true: int = 3,
    seed: int = 42,
) -> np.ndarray
```

Generate random true weights for N sources.

**Returns:** `W_true` with shape (N, K_true), values in [0.1, 5.0]

---

## Utility Functions

### initialize_parameters

```python
initialize_parameters(
    N: int,
    T: int,
    K: int,
    low: float = 0.1,
    high: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]
```

Initialize V and W with random uniform values.

**Returns:** `(V, W)` with shapes (T, K) and (N, K)

---

### compute_global_chi2

```python
compute_global_chi2(
    sources_data: list,
    response_matrices: list,
    V: np.ndarray,
    W: np.ndarray,
) -> float
```

Compute the global chi-squared (unregularized data-fitting term).

$$\chi^2 = \sum_{n=1}^N || \Sigma_n^{-1/2} (y_n - R_n V w_n) ||_2^2$$

---

### compute_smoothness_penalty

```python
compute_smoothness_penalty(V: np.ndarray) -> float
```

Compute first-order smoothness penalty for basis spectra V.

$$\text{Smoothness} = \sum_k \sum_t (V_{t+1,k} - V_{t,k})^2$$

**Returns:** Sum of squared first-order differences. High values = rough spectra.

---

### compute_second_order_smoothness_penalty

```python
compute_second_order_smoothness_penalty(V: np.ndarray) -> float
```

Compute second-order smoothness penalty (discrete Laplacian squared).

$$\text{Curvature} = \sum_k \sum_t (V_{t+1,k} - 2V_{t,k} + V_{t-1,k})^2$$

**Returns:** Sum of squared Laplacian values. Linear spectra have zero penalty.

**Example:**

```python
import numpy as np
from spxcore import compute_second_order_smoothness_penalty

# Linear spectrum: zero penalty
V_linear = np.linspace(0, 1, 100).reshape(-1, 1)
print(compute_second_order_smoothness_penalty(V_linear))  # ~0.0

# Quadratic spectrum: positive penalty
V_quad = (np.linspace(0, 1, 100)**2).reshape(-1, 1)
print(compute_second_order_smoothness_penalty(V_quad))   # >0
```

---

### validate_non_negativity

```python
validate_non_negativity(
    V: np.ndarray,
    W: np.ndarray,
) -> Tuple[bool, bool]
```

Validate that V and W are non-negative.

**Returns:** `(v_valid, w_valid)` - True if all values >= 0

---

## See Also

- [Technical Specification](./TechnicalSpec.md) - Mathematical details and algorithm specification
- [Regularization Guide](./Regularization.md) - Parameter tuning guide
- [Changelog](./CHANGELOG.md) - Version history
