# HALS: Hierarchical Alternating Least Squares

**Version:** 0.4.1
**Last Updated:** 2026-03-04

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Algorithm Workflow](#algorithm-workflow)
4. [Performance Characteristics](#performance-characteristics)
5. [Usage Examples](#usage-examples)
6. [Parameter Guide](#parameter-guide)
7. [Comparison: ALS-WNMF vs HALS](#comparison-als-wnmf-vs-hals)
8. [When to Use HALS](#when-to-use-hals)

---

## Overview

### What is HALS?

Hierarchical Alternating Least Squares (HALS) is a variant of Non-Negative Matrix Factorization (NMF) that updates basis components **sequentially** rather than simultaneously. This sequential update naturally breaks symmetries that can cause "twin spectra" in standard ALS-WNMF.

### The "Twin Spectra" Problem

In standard ALS-WNMF with Multiplicative Update Rules (MUR), all components V[:,k] are updated simultaneously:

$$V_{\text{new}} = V_{\text{old}} \odot \frac{P}{Q}$$

This simultaneous update can lead to:
- **Nearly identical components**: Multiple components converge to the same spectrum
- **Symmetry degeneracy**: Permutations of components produce identical loss
- **Poor orthogonality**: Components are not well-distinguished

### HALS Solution

HALS breaks this symmetry through:
1. **Sequential deflation**: Each component is updated while holding others fixed
2. **Natural asymmetry**: Update order breaks permutation symmetry
3. **BLAS optimization**: Precomputed constants enable efficient iteration

---

## Mathematical Foundation

### Objective Function

HALS optimizes the same objective as ALS-WNMF:

$$\min_{V \ge 0, W \ge 0} \left[ \frac{\chi^2}{M_{\text{total}}} + \alpha R_{\text{L2}} + \beta R_{\text{smooth}} + \gamma R_{\text{curv}} \right]$$

where:
$$\chi^2 = \sum_{n=1}^N || \Sigma_n^{-1/2} (y_n - R_n V w_n) ||_2^2$$

### Precomputation (One-Time Cost)

Before HALS iterations, compute constants that are reused:

**C Matrix (First-Order Constants):**
$$C_n = R_n^T \Sigma_n^{-1} y_n$$

- Shape: (N, T)
- Physical: Precision-weighted observation back-projected to wavelength grid

**B_n Matrices (Second-Order Constants):**
$$B_n = R_n^T \Sigma_n^{-1} R_n$$

- Shape: (T, T) for each source n
- Physical: Precision-weighted response autocorrelation

### Aggregation Tensor M

The key innovation: precompute weighted sum of B_n matrices:

$$M^{(j,k)} = \sum_{n=1}^N w_{nj} w_{nk} B_n$$

**BLAS Optimization:**

Instead of K² sparse matrix additions, use:
```python
PairWeights = W.T @ W  # Shape: (K, K)
All_M_data = PairWeights @ global_B_data  # Single BLAS matmul
```

This provides ~30× speedup over naive implementation.

### Sequential Update Rule

For each component k = 0, ..., K-1:

**Deflation Computation:**
$$U_k = \sum_n w_{nk} C_n - \sum_{j \neq k} M^{(j,k)} V[:,j]$$

The first term is the observation signal for component k. The second term removes contributions from all other components (deflation).

**Hessian Diagonal (Majorizer for PGD):**
$$H_k = M^{(k,k)} + \nabla^2 R_{\text{reg}}$$

For Projected Gradient Descent stability, we use the **maximum absolute row sum** of the regularization Hessian as the diagonal majorizer, not the true Hessian diagonal. This guarantees the step size is conservative enough to prevent gradient explosion.

Regularization Hessian diagonal (majorizer values):
- L2 (α): +2αn₀
- First-order (β): +8βn₁ (interior), +4βn₁ (boundary)
  - True diagonal is 4, but max row sum is 4 + |-2| + |-2| = 8
- Second-order (γ): +32γn₂ (interior), +24γn₂ (near-boundary)
  - True diagonal is 12, but max row sum is 12 + |-8| + |-8| + |2| + |2| = 32

**Projected Gradient Descent:**

Solve: $\min_{V[:,k] \ge 0} ||H_k^{1/2} V[:,k] - H_k^{-1/2} U_k||_2^2$

Algorithm (3-5 iterations):
1. Compute gradient: g = H_k @ V[:,k] - U_k
2. Gradient step: V_temp = V[:,k] - η * g (with line search)
3. Project: V[:,k] = max(V_temp, 0)
4. Check convergence or repeat

---

## Algorithm Workflow

### Initialization (Warm Start)

```python
from spxdictlearn import als_wnmf, hals_wnmf

# Step 1: Run ALS-WNMF to get initial solution
V_init, W_init, loss_als = als_wnmf(
    sources_data, response_matrices, K=40,
    alpha=0.01, beta=0.1, e_step_method='numba'
)
```

### HALS Iteration

```python
# Step 2: Precompute constants (once)
C, global_B_data, global_indices, global_indptr = precompute_hals_constants(
    sources_data, response_matrices, N, T
)

# Step 3: HALS iterations
V, W = V_init, W_init
for iteration in range(max_iter):
    # E-step: Update W (parallel)
    W = e_step(sources_data, response_matrices, V, n_jobs=-1)

    # Compute M tensor (BLAS-optimized)
    M_tensor = compute_M_tensor(W, global_B_data, global_indices, global_indptr, K)

    # M-step: Sequential V updates
    V = m_step_hals(V, W, C, M_tensor, alpha, beta, gamma, n_inner_iter)

    # Check convergence
    loss = compute_regularized_loss(...)
    if |loss - prev_loss| / loss < tol:
        break
```

### Convergence Criteria

HALS typically converges in 50-100 iterations with:
- Tolerance: 1e-5 (stricter than ALS-WNMF's 1e-4)
- Relative loss change: |ΔLoss| / Loss < tol

---

## Performance Characteristics

### Per-Iteration Cost

| Operation | Time | Speedup vs Naive |
|-----------|------|------------------|
| E-step (parallel) | 6-9s | Same as ALS-WNMF |
| M tensor (BLAS) | 0.3s | ~30× |
| M-step (sequential) | 15-30s | ~2-3× slower than MUR |
| **Total** | **20-40s** | **~2-4× slower than ALS-WNMF** |

### Memory Overhead

| Data Structure | Size (N=3000, T=2048) | Description |
|----------------|----------------------|-------------|
| C | 49 MB | First-order constants |
| global_B_data | 960 MB | Aligned B_n matrices |
| M_tensor | 10 MB | Aggregation tensor (K²×T) |
| **Total** | **~1 GB** | Additional vs ALS-WNMF |

### Convergence Speed

| Metric | ALS-WNMF | HALS |
|--------|----------|------|
| Iterations to converge | 50-200 | 50-100 |
| Per-iteration time | 8s | 20-40s |
| **Total time** | **400-1600s** | **1000-4000s** |
| Component quality | May have twins | Better orthogonality |

**When to use HALS:**
- If component quality is more important than speed
- If ALS-WNMF produces "twin spectra"
- For final refinement after ALS-WNMF initialization

---

## Usage Examples

### Basic Usage

```python
from spxdictlearn import als_wnmf, hals_wnmf

# Initialize with ALS-WNMF
V_als, W_als, _ = als_wnmf(
    sources_data, response_matrices, K=40,
    alpha=0.01, beta=0.1, e_step_method='numba'
)

# Refine with HALS
V_final, W_final, loss = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_als, W_init=W_als,
    alpha=0.01, beta=0.1, gamma=0.0,
    tol=1e-5, max_iter=100,
    n_inner_iter=5, verbose=True
)
```

### With Pruning (Recommended)

```python
from spxdictlearn import als_wnmf, prune_and_sort_dictionary, hals_wnmf

# Step 1: ALS-WNMF with large K
V_als, W_als, _ = als_wnmf(
    sources_data, response_matrices, K=40
)

# Step 2: Prune similar components
V_pruned, W_pruned, info = prune_and_sort_dictionary(
    V_als, W_als, similarity_threshold=0.95
)
print(f"Pruned: {info['K_before']} → {info['K_after']} components")

# Step 3: HALS refinement (faster with smaller K)
V_final, W_final, loss = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_pruned, W_init=W_pruned,
    alpha=0.01, beta=0.1
)
```

### With All Regularization

```python
V, W, loss = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_als, W_init=W_als,
    alpha=0.01,    # L2 (prevent large values)
    beta=0.1,      # First-order smoothness
    gamma=0.1,     # Second-order smoothness
    normalize=True
)
```

### Diagnostic Output

```python
# With verbose=True, HALS prints iteration progress:
# Iteration 1: Loss = 1.234e-3 (E-step: 7.2s, M-tensor: 0.3s, M-step: 18.5s)
# Iteration 2: Loss = 9.876e-4 (Δ = 20.0%, E-step: 7.1s, M-tensor: 0.3s, M-step: 18.3s)
# ...
# Converged at iteration 47: Loss = 7.654e-4
```

---

## Parameter Guide

| Parameter | Default | Range | Purpose | Recommendation |
|-----------|---------|-------|---------|----------------|
| `n_inner_iter` | 5 | 3-10 | PGD iterations per component | 5 is sufficient for most cases |
| `tol` | 1e-5 | 1e-4 - 1e-6 | Convergence tolerance | Stricter than ALS-WNMF (1e-4) |
| `warm_start` | True | - | Reuse precomputed constants | Always True for production |
| `e_step_method` | "auto" | "numba"/"scipy" | NNLS solver | Use "numba" for large N |
| `max_iter` | 100 | 50-200 | Maximum iterations | 100 is typically sufficient |

### Tuning Recommendations

**For faster convergence:**
- Increase `tol` to 1e-4 (less strict)
- Decrease `n_inner_iter` to 3 (fewer PGD iterations)

**For better component quality:**
- Decrease `tol` to 1e-6 (stricter convergence)
- Increase `n_inner_iter` to 10 (more accurate PGD)

**For large datasets (N > 1000):**
- Use `e_step_method='numba'` for GIL-free parallelism
- Enable `warm_start=True` to reuse constants

---

## Comparison: ALS-WNMF vs HALS

| Feature | ALS-WNMF | HALS |
|---------|----------|------|
| **Update Strategy** | Simultaneous (all V[:,k]) | Sequential (one V[:,k] at a time) |
| **Symmetry** | May have twins | Naturally broken |
| **Speed (per iter)** | ~8s | ~20-40s |
| **Convergence** | 50-200 iterations | 50-100 iterations |
| **Total Time** | 400-1600s | 1000-4000s |
| **Memory** | ~2GB | ~3GB (with constants) |
| **Orthogonality** | May suffer | Better |
| **Implementation** | Multiplicative Update Rules | Projected Gradient Descent |
| **Best For** | Initial exploration | Final refinement |

### When to Choose Which

**Use ALS-WNMF when:**
- Exploring the data for the first time
- Speed is more important than component orthogonality
- K is small (< 20)
- Need quick results

**Use HALS when:**
- ALS-WNMF produces "twin spectra"
- Component quality is critical
- Have time for slower convergence
- Final refinement needed

**Use Both (Recommended Pipeline):**
```python
# 1. Fast exploration with ALS-WNMF
V_als, W_als, _ = als_wnmf(data, responses, K=40)

# 2. Prune redundant components
V_pruned, W_pruned, _ = prune_and_sort_dictionary(V_als, W_als, 0.95)

# 3. Refine with HALS for quality
V_final, W_final, _ = hals_wnmf(data, responses, V_pruned, W_pruned)
```

---

## When to Use HALS

### Indicators That HALS May Help

1. **Visual Inspection**: Extracted components look nearly identical
2. **Cosine Similarity**: S_ij > 0.95 for multiple pairs
3. **Loss Plateau**: ALS-WNMF loss stops decreasing but components are uncertain
4. **Physical Interpretation**: Need distinct spectral components for classification

### Use Cases

**Case 1: Astronomical Source Classification**
```python
# Extract distinct spectral types (star-forming, AGN, quiescent)
V_als, W_als, _ = als_wnmf(data, responses, K=30)

# If AGN components split into multiple similar spectra:
V_final, W_final, _ = hals_wnmf(data, responses, V_als, W_als)
# Result: AGN components merge into one distinct spectrum
```

**Case 2: Emission Line Decomposition**
```python
# Isolate individual emission lines
V_als, W_als, _ = als_wnmf(data, responses, K=50)

# Prune and refine to get clean line components
V_pruned, W_pruned, _ = prune_and_sort_dictionary(V_als, W_als, 0.95)
V_final, W_final, _ = hals_wnmf(data, responses, V_pruned, W_pruned)
```

**Case 3: Continuum vs Line Separation**
```python
# Separate continuum from emission lines
V_als, W_als, _ = als_wnmf(data, responses, K=10, beta=0.5)
V_final, W_final, _ = hals_wnmf(data, responses, V_als, W_als, beta=0.5)
# Result: First 2-3 components are smooth continua, rest are lines
```

---

## See Also

- [Technical Specification](./TechnicalSpec.md) - Complete mathematical details
- [API Reference](./APIReference.md) - Function signatures
- [Regularization Guide](./Regularization.md) - Parameter tuning
- [Changelog](./CHANGELOG.md) - Version history
