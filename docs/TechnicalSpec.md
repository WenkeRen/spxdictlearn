# Technical Specification: ALS-WNMF for SPHEREx Spectral Decomposition

**Version:** 0.4.0
**Last Updated:** 2026-03-04

This document provides the complete technical specification for the Alternating Least Squares Weighted Non-Negative Matrix Factorization (ALS-WNMF) algorithm used to extract physically meaningful spectra from sparse, unaligned, and noisy SPHEREx observational data.

---

## Table of Contents

1. [Data Structures & Tensor Dimensions](#1-data-structures--tensor-dimensions)
2. [Mathematical Model](#2-mathematical-model)
3. [Core Algorithm Logic](#3-core-algorithm-logic)
4. [Response Matrix Construction](#4-response-matrix-construction)
5. [Regularization](#5-regularization)
6. [M-step Update Rules](#6-m-step-update-rules)
7. [HALS Algorithm (v0.4.0)](#7-hals-algorithm-v040)
8. [Dictionary Pruning (v0.4.0)](#8-dictionary-pruning-v040)
9. [Mock Data Generation](#9-mock-data-generation)
10. [Testing Requirements](#10-testing-requirements)

---

## 1. Data Structures & Tensor Dimensions

All implementations MUST strictly adhere to these dimensions.

### Global Parameters

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| $N$ | Number of distinct astronomical sources | $\approx 3000$ |
| $T$ | Number of target high-resolution logarithmic wavelength bins | $4096$ |
| $K$ | Number of non-negative components to extract | $20$ |
| $M_n$ | Number of raw observations for source $n$ | $8000 - 12000$ |

### Input Data (Per Source $n \in [1, N]$)

The raw input for source $n$ is a 2D array $\mathbf{D}_n \in \mathbb{R}^{M_n \times 4}$:

| Column | Symbol | Description |
|--------|--------|-------------|
| 0 | $\lambda_c$ | Observation central wavelength (μm) |
| 1 | $bw$ | Observation bandwidth as FWHM (μm) |
| 2 | $flux$ | Observed flux $\mathbf{y}_n \in \mathbb{R}^{M_n}$ |
| 3 | $error$ | Gaussian error standard deviation $\boldsymbol{\sigma}_n \in \mathbb{R}^{M_n}$ |

### Latent Variables (To be Optimized)

- **$\mathbf{V} \in \mathbb{R}^{T \times K}$**: Global high-resolution basis matrix (Eigenspectra). Constraint: $\mathbf{V} \ge 0$.
- **$\mathbf{W} \in \mathbb{R}^{N \times K}$**: Weight matrix. Constraint: $\mathbf{W} \ge 0$.

### Response Matrix

$\mathbf{R}_n \in \mathbb{R}^{M_n \times T}$: Truncated Gaussian response matrix. **Must be implemented as `scipy.sparse.csr_matrix`** for memory efficiency.

---

## 2. Mathematical Model

### Global Objective Function (v0.3.0)

$$\min_{\mathbf{V} \ge 0, \mathbf{W} \ge 0} \left[ \frac{\chi^2}{M_{\text{total}}} + \frac{\alpha}{TK} ||\mathbf{V}||_F^2 + \frac{\beta}{(T-1)K} \cdot \text{Smoothness}(\mathbf{V}) + \frac{\gamma}{(T-2)K} \cdot \text{Curvature}(\mathbf{V}) \right]$$

where the data-fitting term (chi-squared) is:

$$\chi^2 = \sum_{n=1}^N || \boldsymbol{\Sigma}_n^{-1/2} (\mathbf{y}_n - \mathbf{R}_n \mathbf{V} \mathbf{w}_n) ||_2^2$$

### Regularization Terms

| Term | Formula | Description |
|------|---------|-------------|
| L2 (α) | $\|\mathbf{V}\|_F^2 = \sum_{t,k} V_{t,k}^2$ | Frobenius norm squared |
| First-order smoothness (β) | $\sum_{k} \sum_{t=1}^{T-1} (V_{t+1,k} - V_{t,k})^2$ | Squared adjacent differences |
| Second-order smoothness (γ) | $\sum_{k} \sum_{t=1}^{T-2} (V_{t+1,k} - 2V_{t,k} + V_{t-1,k})^2$ | Discrete Laplacian squared |

### Normalization Factors

When `normalize=True` (default), all terms are normalized to be dimensionless:

| Term | Normalization | Physical Meaning |
|------|---------------|------------------|
| $\chi^2$ | $1/M_{\text{total}}$ | Reduced chi-squared |
| L2 (α) | $1/(TK)$ | Average squared element |
| First-order (β) | $1/((T-1)K)$ | Average squared slope |
| Second-order (γ) | $1/((T-2)K)$ | Average squared curvature |

**Note:** $\boldsymbol{\Sigma}_n^{-1/2}$ is equivalent to element-wise multiplication by $1/\boldsymbol{\sigma}_n$. Do NOT construct dense diagonal covariance matrices.

---

## 3. Core Algorithm Logic: ALS-WNMF

Implement the optimization using Alternating Least Squares. Loop the following E-step and M-step until convergence ($|\Delta \text{Loss}| / \text{Loss} < \text{tolerance}$).

### 3.1 E-step: Update Weights $\mathbf{w}_n$ (Parallelizable)

Fix $\mathbf{V}$, optimize $\mathbf{w}_n$ for each source independently. This is a standard Non-Negative Least Squares (NNLS) problem:

$$\min_{\mathbf{w}_n \ge 0} || \mathbf{A}_n \mathbf{w}_n - \mathbf{b}_n ||_2^2$$

**Computational steps per source $n$:**

1. Compute precision-weighted observation: $\mathbf{y}'_n = \mathbf{y}_n \oslash \boldsymbol{\sigma}_n$ (Shape: $M_n$)
2. Compute dense matrix: $\mathbf{U}_n = \mathbf{R}_n \mathbf{V}$ (Sparse × Dense → Dense, Shape: $M_n \times K$)
3. Compute design matrix: $\mathbf{A}_n = \mathbf{U}_n \oslash \boldsymbol{\sigma}_n[:, \text{newaxis}]$ (Shape: $M_n \times K$)
4. Solve: $\mathbf{w}_n = \text{scipy.optimize.nnls}(\mathbf{A}_n, \mathbf{y}'_n)[0]$

**Engineering Directive:** Use `joblib.Parallel` to parallelize this loop over all $N$ sources.

### 3.2 M-step: Update Basis $\mathbf{V}$ (Global Reduction)

Fix all $\mathbf{w}_n$, update the global matrix $\mathbf{V}$ using Multiplicative Update Rules (MUR) to guarantee non-negativity and monotonic convergence.

See [Section 6: M-step Update Rules](#6-m-step-update-rules) for detailed implementation.

---

## 4. Response Matrix Construction

The response matrix MUST implement exact analytical integration of a Gaussian response over logarithmic grid bins using `scipy.stats.norm.cdf`.

### Target Grid Definition

- $\lambda_{\text{min}} = 0.75$ μm, $\lambda_{\text{max}} = 5.0$ μm
- Grid edges: $T+1$ linearly spaced points in log-space: $\ln \boldsymbol{\lambda}_{\text{edges}} = \text{linspace}(\ln 0.75, \ln 5.0, 4097)$
- Bin centers: Geometric mean of edges: $\text{centers}[t] = \sqrt{\text{edges}[t] \times \text{edges}[t+1]}$

### Observation Mapping (For the $m$-th observation of source $n$)

Extract parameters from $\mathbf{D}_n[m, :]$: $\lambda_c = \text{col } 0$, $FWHM = \text{col } 1$.

1. Calculate Gaussian Standard Deviation: $\sigma_f = FWHM / (2 \sqrt{2 \ln 2}) \approx FWHM / 2.35482$
2. Truncation Bounds: $\lambda_{\text{start}} = \lambda_c - 3\sigma_f$, $\lambda_{\text{end}} = \lambda_c + 3\sigma_f$

### Integration Logic

For every target grid bin $t \in [0, 4095]$ that overlaps with $[\lambda_{\text{start}}, \lambda_{\text{end}}]$:

1. Find actual integration bounds: $a = \max(\lambda_{\text{start}}, \text{edges}[t])$, $b = \min(\lambda_{\text{end}}, \text{edges}[t+1])$
2. If $a < b$, compute the exact Gaussian integral:

$$\mathbf{R}_n[m, t] = \text{norm.cdf}(b, \text{loc}=\lambda_c, \text{scale}=\sigma_f) - \text{norm.cdf}(a, \text{loc}=\lambda_c, \text{scale}=\sigma_f)$$

### Flux Conservation Normalization

Since truncating at $3\sigma$ loses ~0.27% of the area, normalize each row:

```python
row_sum = np.sum(Rn[m, :])
if row_sum > 0:
    Rn[m, :] /= row_sum  # Row sums exactly to 1.0
```

---

## 5. Regularization

### 5.1 L2 Regularization (α)

**Purpose:** Prevents large spectral values and improves numerical conditioning.

**Penalty:** $\|\mathbf{V}\|_F^2 = \sum_{t,k} V_{t,k}^2$

**Gradient:** $\nabla_{\mathbf{V}} \|\mathbf{V}\|_F^2 = 2\mathbf{V}$

**Typical normalized values:** 0.001 - 0.1

### 5.2 First-Order Smoothness (β)

**Purpose:** Encourages smooth slopes by penalizing roughness.

**Penalty:** $\text{Smoothness}(\mathbf{V}) = \sum_{k} \sum_{t=1}^{T-1} (V_{t+1,k} - V_{t,k})^2$

**Gradient:** $\nabla_{\mathbf{V}} \text{Smoothness} = 2\mathbf{L}^T \mathbf{L} \mathbf{V}$

where $\mathbf{L}$ is the first-difference operator.

**Typical normalized values:** 0.01 - 1.0

### 5.3 Second-Order Smoothness / Curvature (γ)

**Purpose:** Encourages linear/flat regions by penalizing curvature.

**Penalty:** $\text{Curvature}(\mathbf{V}) = \sum_{k} \sum_{t=1}^{T-2} (V_{t+1,k} - 2V_{t,k} + V_{t-1,k})^2$

**Properties:**
- Linear spectra ($V_t = a + bt$) have **zero** curvature penalty
- Quadratic and higher-order spectra have **positive** penalty
- Unlike first-order (β), curvature penalty (γ) allows linear trends

**Typical normalized values:** 0.01 - 1.0

### 5.4 Recommended Parameter Values (Normalized Mode)

| Parameter | Default | Typical Range | Physical Interpretation |
|-----------|---------|---------------|------------------------|
| α | 0.01 | 0.001 - 0.1 | Prevents large spectral values |
| β | 0.0 | 0.01 - 1.0 | Encourages smooth slopes |
| γ | 0.0 | 0.01 - 1.0 | Encourages linear/flat regions |

**Recommended starting values:**
- α = 0.01 (weak L2)
- β = 0.1 (moderate first-order smoothness)
- γ = 0.0 (disabled by default; enable with 0.1 if needed)

---

## 6. M-step Update Rules

The M-step uses Multiplicative Update Rules (MUR) to guarantee non-negativity and monotonic convergence.

### 6.1 Basic MUR (Without Regularization)

$$\mathbf{V}_{\text{new}} = \mathbf{V}_{\text{old}} \odot \frac{\mathbf{P}}{\mathbf{Q} + \epsilon}$$

where $\epsilon = 10^{-12}$ prevents division by zero.

### 6.2 Accumulator Computation

Initialize $\mathbf{P} = \mathbf{0}_{T \times K}$ and $\mathbf{Q} = \mathbf{0}_{T \times K}$.

For each source $n \in [1, N]$:

1. Compute precision-weighted observation: $\mathbf{s}_n = \mathbf{y}_n \oslash (\boldsymbol{\sigma}_n \odot \boldsymbol{\sigma}_n)$
2. Compute model prediction: $\mathbf{\hat{y}}_n = \mathbf{R}_n (\mathbf{V} \mathbf{w}_n)$
3. Compute precision-weighted prediction: $\mathbf{q}_n = \mathbf{\hat{y}}_n \oslash (\boldsymbol{\sigma}_n \odot \boldsymbol{\sigma}_n)$
4. Accumulate to Numerator: $\mathbf{P} \mathrel{+}= (\mathbf{R}_n^T \mathbf{s}_n) \otimes \mathbf{w}_n^T$
5. Accumulate to Denominator: $\mathbf{Q} \mathrel{+}= (\mathbf{R}_n^T \mathbf{q}_n) \otimes \mathbf{w}_n^T$

### 6.3 MUR with All Regularization (v0.3.3)

$$\mathbf{V}_{\text{new}} = \mathbf{V}_{\text{old}} \odot \frac{\mathbf{P} + \mathbf{P}_{\text{reg}}}{\mathbf{Q} + \mathbf{Q}_{\text{reg}} + \epsilon}$$

where the regularization gradient terms are:

$$\mathbf{P}_{\text{reg}} = (2\beta n_1 \mathbf{V}_{\text{neighbor}} + 2\gamma n_2 \mathbf{P}_{2\text{nd}}) \cdot \text{reg\_scale}$$

$$\mathbf{Q}_{\text{reg}} = (2\alpha n_0 \mathbf{V} + 4\beta n_1 \mathbf{V} + 2\gamma n_2 \mathbf{Q}_{2\text{nd}}) \cdot \text{reg\_scale}$$

with normalization factors: $n_0 = 1/(TK)$, $n_1 = 1/((T-1)K)$, $n_2 = 1/((T-2)K)$

**Gradient Scaling Factor (v0.3.3 fix):**

$$\text{reg\_scale} = \begin{cases} M_{\text{total}} / 2 & \text{if normalize=True} \\ 0.5 & \text{if normalize=False} \end{cases}$$

This factor aligns the regularization gradients with the loss function evaluation in `compute_regularized_loss()`. The data term gradients P and Q correspond to `∇(1/2 · χ²_raw)`, but the loss normalizes χ² by `M_total`. Without this scaling, the effective regularization strength would be off by a factor of `M_total/2` (normalized) or `2` (unnormalized).

### 6.4 First-Order Smoothness Gradient Arrays

Compute $\mathbf{V}_{\text{neighbor}}$ using array slicing:

```python
V_neighbor_sum = np.zeros_like(V)
# Inner points
V_neighbor_sum[1:-1, :] = V[0:-2, :] + V[2:, :]
# Boundaries (Neumann condition: zero slope)
V_neighbor_sum[0, :] = V[0, :] + V[1, :]      # Top
V_neighbor_sum[-1, :] = V[-2, :] + V[-1, :]   # Bottom
```

### 6.5 Second-Order Smoothness Gradient Arrays

For MUR with reflective/Neumann boundaries:

| Position | $P_{2\text{nd}}[t]$ | $Q_{2\text{nd}}[t]$ |
|----------|---------------------|---------------------|
| Interior ($2 \le t \le T-3$) | $4(V_{t-1} + V_{t+1})$ | $6V_t + V_{t-2} + V_{t+2}$ |
| Near-boundary ($t=1$) | $4(V_0 + V_2)$ | $5V_1 + V_3$ |
| Near-boundary ($t=T-2$) | $4(V_{T-1} + V_{T-3})$ | $5V_{T-2} + V_{T-4}$ |
| Boundary ($t=0$) | $2V_1$ | $2V_0$ |
| Boundary ($t=T-1$) | $2V_{T-2}$ | $2V_{T-1}$ |

**Implementation:**

```python
P_2nd = np.zeros_like(V)
Q_2nd = np.zeros_like(V)

# Interior points (2 <= t <= T-3)
if T > 4:
    P_2nd[2:-2, :] = 4 * (V[1:-3, :] + V[3:-1, :])
    Q_2nd[2:-2, :] = 6 * V[2:-2, :] + V[:-4, :] + V[4:, :]

# Boundary t=1
if T > 3:
    P_2nd[1, :] = 4 * (V[0, :] + V[2, :])
    Q_2nd[1, :] = 5 * V[1, :] + V[3, :]

# Boundary t=T-2
if T > 3:
    P_2nd[-2, :] = 4 * (V[-1, :] + V[-3, :])
    Q_2nd[-2, :] = 5 * V[-2, :] + V[-4, :]

# Boundaries t=0 and t=T-1 (Neumann)
P_2nd[0, :] = 2 * V[1, :]
Q_2nd[0, :] = 2 * V[0, :]
P_2nd[-1, :] = 2 * V[-2, :]
Q_2nd[-1, :] = 2 * V[-1, :]
```

---

## 7. HALS Algorithm (v0.4.0)

### Purpose: Addressing "Twin Spectra" Problem

The standard Multiplicative Update Rule (MUR) in ALS-WNMF can produce nearly identical spectral components ("twin spectra") due to the symmetry of simultaneous updates. HALS (Hierarchical Alternating Least Squares) addresses this through:

1. **Sequential deflation updates**: Each component V[:,k] is updated while holding others fixed
2. **Natural symmetry breaking**: Sequential updates break degeneracy
3. **BLAS-optimized aggregation**: Precomputed constants for efficient iteration

### Key Differences from MUR

| Feature | MUR (ALS-WNMF) | HALS |
|---------|---------------|------|
| Update | Simultaneous (all V[:,k]) | Sequential (one V[:,k] at a time) |
| Symmetry | May have twins | Naturally broken |
| Speed | Faster per iteration | Slower per iteration |
| Orthogonality | May suffer | Better |
| Memory | Lower | Higher (precomputed constants) |

### Precomputation (One-Time Cost)

Before HALS iterations, compute constants that are reused across iterations:

**C Matrix (First-Order Constants):**
$$C_n = \mathbf{R}_n^T \boldsymbol{\Sigma}_n^{-1} \mathbf{y}_n$$

- Shape: (N, T)
- Physical meaning: Precision-weighted observation back-projected to wavelength grid
- Computed ONCE before HALS iterations

**B_n Matrices (Second-Order Constants):**
$$B_n = \mathbf{R}_n^T \boldsymbol{\Sigma}_n^{-1} \mathbf{R}_n$$

- Shape: (T, T) for each source n
- Physical meaning: Precision-weighted response autocorrelation
- **BLAS Optimization**: All B_n matrices have similar sparse structure (narrow bandwidth)

**Global B Data Structure:**

To enable BLAS optimization, extract all B_n matrices into a unified structure:
- **global_B_data**: Dense matrix (N, nnz_global) containing aligned non-zero values
- **global_indices**: CSR column indices for global template
- **global_indptr**: CSR row pointers for global template

Memory estimate for T=2048, N=3000, nnz_global ~ 40k:
- global_B_data: 960 MB (vs ~2-5 GB for naive B_list)

### Aggregation Tensor M

The key innovation in HALS is precomputing the aggregation tensor M:

$$M^{(j,k)} = \sum_{n=1}^N w_{nj} w_{nk} B_n$$

- **Shape**: (T, T, K, K) but computed in a factorized form
- **Physical meaning**: Weighted sum of B_n matrices using pairwise weight products

**BLAS-Optimized Computation:**

Instead of summing K² sparse matrices of size (T, T), use:
```python
PairWeights = W.T @ W  # Shape: (K, K)
All_M_data = PairWeights @ global_B_data  # Single BLAS matmul
```

This replaces ~630,000 sparse matrix additions with one dense matmul (~30× speedup).

### Sequential Update Rule

For each component k = 0, ..., K-1:

**Deflation Computation:**
$$\mathbf{U}_k = \mathbf{P} - \sum_{j \neq k} M^{(j,k)} @ \mathbf{V}[:,j]$$

- P = Σ_n w_nk * C_n (precision-weighted observation)
- U_k represents the "residual" for component k after removing contributions from other components

**Diagonal Hessian:**
$$\text{diag}(\mathbf{H}_k) = \text{diag}(M^{(k,k)}) + \text{regularization}$$

Regularization gradients (same as MUR):
- L2 (α): +2αn₀
- First-order (β): +4βn₁
- Second-order (γ): curvature penalty

**Inner PGD Loop:**

Solve non-negative least squares for V[:,k]:
$$\min_{\mathbf{V}[:,k] \ge 0} ||\mathbf{H}_k^{1/2} \mathbf{V}[:,k] - \mathbf{H}_k^{-1/2} \mathbf{U}_k||_2^2$$

Using Projected Gradient Descent with 3-5 iterations:
1. Compute gradient: g = H_k @ V[:,k] - U_k
2. Gradient step: V_temp = V[:,k] - step_size * g
3. Project: V[:,k] = max(V_temp, 0)
4. Check convergence or repeat

### Algorithm Workflow

**Initialization (Warm Start):**
```python
# Precompute constants (once)
C, global_B_data, global_indices, global_indptr = precompute_hals_constants(
    sources_data, response_matrices, N, T
)

# Initialize V, W from ALS-WNMF or random
```

**Iteration Loop:**
```python
for iteration in range(max_iter):
    # E-step: Update W (reuse from als_wnmf)
    W = e_step(sources_data, response_matrices, V, n_jobs=-1)

    # Compute aggregation tensor M (BLAS-optimized)
    M_tensor = compute_M_tensor(W, global_B_data, global_indices, global_indptr, K)

    # M-step: Sequential V updates
    V_new = m_step_hals(V, W, C, M_tensor, alpha, beta, gamma, normalize, n_inner_iter)

    # Check convergence
    loss = compute_regularized_loss(...)
    if |loss - prev_loss| / loss < tol:
        break
```

**Warm Start Mode:**

When `warm_start=True`, constants C and global_B_data are reused across iterations:
- First iteration: Precompute constants (~2-5 seconds)
- Subsequent iterations: Reuse constants (negligible overhead)

### Parameter Guide

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| n_inner_iter | 5 | 3-10 | PGD iterations per component |
| tol | 1e-5 | 1e-4 - 1e-6 | Convergence tolerance |
| warm_start | True | - | Reuse precomputed constants |
| n_jobs | -1 | 1-N | Parallel jobs for E-step |

### Performance Characteristics

**Per-Iteration Cost:**
- E-step: ~6-9s (same as ALS-WNMF with threading backend)
- M tensor: ~0.3s (BLAS matmul, ~30× speedup)
- M-step: ~15-30s (K sequential updates, ~2-3× slower than MUR)
- Total: ~20-40s/iter (vs ~8s/iter for ALS-WNMF)

**Convergence:**
- Typically converges in 50-100 iterations (vs 50-200 for ALS-WNMF)
- Better orthogonality reduces need for many components
- Warm-start from ALS-WNMF solution accelerates convergence

**Memory:**
- C: N × T × 8 bytes ≈ 49 MB (for N=3000, T=2048)
- global_B_data: N × nnz_global × 8 bytes ≈ 960 MB
- Total: ~1 GB additional memory (acceptable for modern workstations)

---

## 8. Dictionary Pruning (v0.4.0)

### Purpose: Remove Redundant Similar Components

ALS-WNMF with large K can extract nearly identical spectral components. Dictionary pruning identifies and merges these components before HALS refinement, reducing computational cost and improving interpretability.

### Cosine Similarity Metric

Compute pairwise cosine similarity between spectral components:

$$S_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \, ||\mathbf{v}_j||}$$

Properties:
- Range: [-1, 1] (for normalized vectors)
- S_ii = 1.0 (identical components)
- S_ij ≈ 1.0 → nearly identical (potential "twins")
- S_ij ≈ 0 → orthogonal
- S_ij ≈ -1 → anti-correlated (rare for non-negative spectra)

**Numerical Stability:**

```python
# L2 normalization for stability
norms = np.linalg.norm(V, axis=0, keepdims=True)
norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
V_normalized = V / norms

# Cosine similarity via matrix multiplication
S = V_normalized.T @ V_normalized
np.fill_diagonal(S, 1.0)  # Ensure exact diagonal
S = np.clip(S, -1.0, 1.0)  # Handle numerical errors
```

### Hierarchical Clustering

Group similar components using hierarchical clustering with complete linkage:

**Distance Metric:**
$$d_{ij} = 1 - S_{ij}$$

- Similar components (S_ij ≈ 1) have small distance
- Orthogonal components (S_ij ≈ 0) have distance ≈ 1
- Anti-correlated (S_ij ≈ -1) have distance ≈ 2

**Linkage Method:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| 'complete' | Maximum distance in cluster (default) | Conservative, ensures all pairs exceed threshold |
| 'average' | Average distance in cluster | Balanced |
| 'single' | Minimum distance in cluster | Aggressive, may include distant pairs |

**Threshold Selection:**

| Threshold | Effect | Typical K Reduction |
|-----------|--------|---------------------|
| 0.99 | Very conservative | Minimal merging |
| 0.95 (default) | Merge near-identical twins | ~10-30% reduction |
| 0.90 | Aggressive merging | ~30-50% reduction |
| 0.85 | Very aggressive | May over-merge |

### Weighted Merging

Merge components within each cluster using global importance weighting:

**Global Importance:**
$$w_k^{\text{global}} = \sum_{n=1}^N w_{nk}$$

- Components with larger total weight contribute more to observations
- Important components dominate the merged spectrum

**Merging Formula:**

For cluster c containing components {k₁, k₂, ..., k_m}:

$$\mathbf{V}_{\text{merged}}[:,c] = \frac{\sum_{k \in c} w_k^{\text{global}} \cdot \mathbf{V}[:,k]}{\sum_{k \in c} w_k^{\text{global}}}$$

$$\mathbf{W}_{\text{merged}}[:,c] = \sum_{k \in c} \mathbf{W}[:,k]$$

Physical interpretation:
- V_merged is a weighted average of similar spectra
- W_merged adds the weights (preserves total flux)

### Complete Pruning Pipeline

```python
def prune_and_sort_dictionary(V, W, similarity_threshold=0.95, method='complete'):
    """
    Complete pruning pipeline:
    1. Compute pairwise cosine similarity
    2. Cluster similar components
    3. Merge within clusters
    4. Sort by global importance

    Returns:
        V_pruned: (T, K_new) pruned basis
        W_pruned: (N, K_new) pruned weights
        info: dict with metadata (clusters, K_before, K_after)
    """
    # Step 1: Cosine similarity
    S = compute_cosine_similarity_matrix(V)

    # Step 2: Hierarchical clustering
    labels = cluster_similar_components(V, similarity_threshold, method)

    # Step 3: Merge by cluster
    V_merged, W_merged = merge_components_by_cluster(V, W, labels)

    # Step 4: Sort by global importance
    importance = W_merged.sum(axis=0)  # Sum across sources
    sort_order = np.argsort(importance)[::-1]  # Descending
    V_sorted = V_merged[:, sort_order]
    W_sorted = W_merged[:, sort_order]

    return V_sorted, W_sorted, {
        'clusters': labels,
        'K_before': V.shape[1],
        'K_after': V_merged.shape[1],
        'importance': importance[sort_order]
    }
```

### Usage in ALS-WNMF → HALS Pipeline

**Recommended Workflow:**
```python
# Step 1: Run ALS-WNMF with large K
V_als, W_als, _ = als_wnmf(
    sources_data, response_matrices, K=40
)

# Step 2: Prune to remove twins
V_pruned, W_pruned, info = prune_and_sort_dictionary(
    V_als, W_als, similarity_threshold=0.95
)
print(f"Pruned: {info['K_before']} → {info['K_after']} components")

# Step 3: HALS refinement (faster with smaller K)
V_final, W_final, _ = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_pruned, W_init=W_pruned,
    warm_start=True
)
```

### Validation Checklist

After pruning, verify:
- [ ] **Flux conservation**: Sum of all weights should be approximately preserved
- [ ] **Reconstruction quality**: χ² should not significantly increase
- [ ] **Component diversity**: Cosine similarity matrix should have lower off-diagonal values
- [ ] **Physical meaningfulness**: Merged components should look like astronomical spectra

---

## 9. Mock Data Generation

### Ground Truth Generation

**True Basis ($\mathbf{V}_{\text{true}}$):** Create $K_{\text{true}} = 3$ distinct artificial spectra:

- **Component 0:** Broad Gaussian continuum (center=2.0 μm, σ=1.0 μm)
- **Component 1:** Power-law curve ($f(\lambda) = \lambda^{-1.5}$)
- **Component 2:** Sparse narrow Gaussian emission lines (at λ=1.5, 2.0, 3.0, 4.0 μm)

**True Weights ($\mathbf{W}_{\text{true}}$):** Generate for $N$ sources:

```python
np.random.uniform(0.1, 5.0, size=(N, K_true))
```

### Observation Simulation

For each source $n \in [1, N]$:

1. Generate $M_n$ random central wavelengths $\lambda_c \sim \text{Uniform}(0.8, 4.8)$
2. Set constant $FWHM = 0.02$
3. Build ground truth spectrum: $\mathbf{x}_n = \mathbf{V}_{\text{true}} (\mathbf{W}_{\text{true}})_{n}^T$
4. Build $\mathbf{R}_n$ using the exact integration logic
5. Compute true observation: $\mathbf{y}_{\text{true}} = \mathbf{R}_n \mathbf{x}_n$
6. Add noise: $\boldsymbol{\sigma}_n = 0.05 \times \text{mean}(\mathbf{y}_{\text{true}})$
7. Add Gaussian noise: $\mathbf{y}_n = \mathbf{y}_{\text{true}} + \mathcal{N}(0, \boldsymbol{\sigma}_n^2)$
8. Pack into $\mathbf{D}_n \in \mathbb{R}^{M_n \times 4}$

---

## 10. Testing Requirements

### Verification Assertions

The testing script MUST assert/verify:

1. **Monotonicity:** The total Loss MUST decrease or remain flat after every E-step and M-step. If Loss increases, the implementation is flawed.

2. **Non-negativity:** Assert `np.all(V >= 0)` and `np.all(W >= 0)` at all times.

3. **Convergence:** The algorithm should trigger the tolerance stopping condition within 100 iterations.

4. **Response Matrix Properties:**
   - Sparsity (>99% for typical parameters)
   - Non-negativity (all values ≥ 0)
   - Normalization (row sums = 1.0 within machine precision)

5. **Reconstruction Accuracy:**
   - RMSE < 20% on noisy test data
   - Components recover ground truth structure

### Memory Requirements

**Critical:** $\mathbf{R}_n$ memory footprint is $M_n \times T \approx 4 \times 10^7$ floats ≈ 160 MB if dense. With $N=3000$, total is ~480 GB. **MUST use `scipy.sparse.csr_matrix`**. Total footprint drops to <2 GB.

### Initialization

Initialize $\mathbf{V}$ and $\mathbf{W}$ using `np.random.uniform(low=0.1, high=1.0)`. Do not initialize with zeros.

---

## Related Documentation

- [API Reference](./APIReference.md) - Function signatures and usage examples
- [Regularization Guide](./Regularization.md) - Detailed parameter tuning guide
- [Changelog](./CHANGELOG.md) - Version history and migration guides
