# Regularization Guide: ALS-WNMF and HALS

**Version:** 0.4.0
**Last Updated:** 2026-03-04

This guide explains how to use regularization parameters in ALS-WNMF and HALS to improve spectral decomposition quality and prevent overfitting.

---

## Table of Contents

1. [Why Regularization Matters](#why-regularization-matters)
2. [Overview of Regularization Terms](#overview-of-regularization-terms)
3. [L2 Regularization (α)](#l2-regularization-α)
4. [First-Order Smoothness (β)](#first-order-smoothness-β)
5. [Second-Order Smoothness (γ)](#second-order-smoothness-γ)
6. [Normalization Mode](#normalization-mode)
7. [Parameter Tuning Guide](#parameter-tuning-guide)
8. [Migration Guide: v0.2.x -> v0.3.0](#migration-guide-v02x--v030)

---

## Why Regularization Matters

Without regularization, ALS-WNMF can produce:

1. **Unphysical large values** in extracted spectra
2. **High-frequency noise** amplified from observational errors
3. **Overfitting** to noise rather than true spectral features
4. **Poor generalization** to new observations

Regularization addresses these issues by adding penalty terms to the objective function that encourage physically meaningful solutions.

---

## Overview of Regularization Terms

The ALS-WNMF objective function (v0.3.0, normalized mode) is:

$$\text{Loss} = \frac{\chi^2}{M_{\text{total}}} + \frac{\alpha}{TK} ||\mathbf{V}||_F^2 + \frac{\beta}{(T-1)K} \cdot \text{Smoothness}(\mathbf{V}) + \frac{\gamma}{(T-2)K} \cdot \text{Curvature}(\mathbf{V})$$

| Parameter | Term | What It Penalizes | Physical Effect |
|-----------|------|-------------------|-----------------|
| **α** | L2 | Large spectral values | Prevents unphysical amplitudes |
| **β** | First-order smoothness | Adjacent bin differences | Encourages smooth slopes |
| **γ** | Second-order smoothness | Local curvature | Encourages linear/flat regions |

**Key Insight:** Each term addresses a different type of unphysical behavior:
- **α** -> "Don't make the spectrum too bright"
- **β** -> "Don't make the spectrum too jagged"
- **γ** -> "Don't make the spectrum too curved"

---

## L2 Regularization (α)

### Mathematical Formulation

**Penalty:** $||\mathbf{V}||_F^2 = \sum_{t,k} V_{t,k}^2$ (Frobenius norm squared)

**Gradient:** $\nabla_{\mathbf{V}} ||\mathbf{V}||_F^2 = 2\mathbf{V}$

### Physical Interpretation

L2 regularization penalizes the sum of squared spectral values. This:
- Prevents individual spectral components from becoming extremely large
- Encourages the model to distribute flux across multiple components
- Improves numerical conditioning

### When to Use

- **Limited observations:** When sources have few observations (<5000 per source)
- **High noise:** When measurement errors are large (noise_level > 0.1)
- **Large K:** When extracting many components relative to N sources
- **Overfitting signs:** If χ² is very low but extracted spectra look unphysical

### Parameter Range (Normalized Mode)

| Value | Effect |
|-------|--------|
| 0.0 | No L2 penalty (not recommended) |
| 0.001 - 0.01 | Weak penalty (default starting point) |
| 0.01 - 0.1 | Moderate penalty |
| 0.1 - 1.0 | Strong penalty (may underfit) |

### Example Usage

```python
# Weak L2 (default)
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01, normalize=True
)

# Strong L2 for noisy data
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.1, normalize=True
)

# Disable L2 (not recommended)
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.0, normalize=True
)
```

---

## First-Order Smoothness (β)

### Mathematical Formulation

**Penalty:** $\text{Smoothness}(\mathbf{V}) = \sum_{k} \sum_{t=1}^{T-1} (V_{t+1,k} - V_{t,k})^2$

**Gradient:** $\nabla_{\mathbf{V}} \text{Smoothness} = 2\mathbf{L}^T \mathbf{L} \mathbf{V}$

where $\mathbf{L}$ is the first-difference operator.

### Physical Interpretation

First-order smoothness penalizes differences between adjacent wavelength bins. This:
- Encourages smooth, continuous spectra
- Suppresses high-frequency noise
- Preserves overall spectral shape and slopes

### When to Use

- **Noisy observations:** When data has high-frequency noise
- **Sparse wavelength coverage:** When wavelength sampling is uneven
- **Smooth continua:** When extracting continuum spectra (blackbody, power-law)
- **Emission line suppression:** When you want to suppress spurious narrow features

### Parameter Range (Normalized Mode)

| Value | Effect |
|-------|--------|
| 0.0 | No smoothness constraint (default) |
| 0.01 - 0.1 | Weak smoothing (preserves most features) |
| 0.1 - 1.0 | Moderate smoothing |
| 1.0 - 10.0 | Strong smoothing (may blur real features) |

### Example Usage

```python
# Moderate smoothness
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01, beta=0.1, normalize=True
)

# Strong smoothness for very noisy data
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01, beta=1.0, normalize=True
)
```

### Caveat

**First-order smoothness penalizes ALL slopes**, including physically meaningful ones (e.g., rising continua, power-law slopes). If your spectra should have significant overall slopes, consider using **second-order smoothness (γ)** instead, which allows linear trends.

---

## Second-Order Smoothness (γ)

### Mathematical Formulation

**Penalty:** $\text{Curvature}(\mathbf{V}) = \sum_{k} \sum_{t=1}^{T-2} (V_{t+1,k} - 2V_{t,k} + V_{t-1,k})^2$

This is the **discrete Laplacian squared**.

### Physical Interpretation

Second-order smoothness penalizes curvature (deviations from linearity). Key properties:

- **Linear spectra have zero penalty:** $V_t = a + bt$ -> $\text{Curvature} = 0$
- **Quadratic and higher have positive penalty:** Encourages linear/flat regions
- **Unlike β, allows linear trends:** Preserves overall spectral slopes

### When to Use

- **Curved continua:** When spectra should be approximately linear (e.g., power-law continua)
- **Noise reduction without slope penalty:** When you want smoothness but need to preserve slopes
- **Emission line preservation:** Allows linear trends while reducing noise peaks
- **Combined with β:** Use both for maximum smoothness (flat + linear regions)

### Parameter Range (Normalized Mode)

| Value | Effect |
|-------|--------|
| 0.0 | No curvature penalty (default) |
| 0.01 - 0.1 | Weak curvature penalty |
| 0.1 - 1.0 | Moderate curvature penalty |
| 1.0 - 10.0 | Strong curvature penalty (forces linear spectra) |

### Example Usage

```python
# Moderate curvature penalty
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01, beta=0.0, gamma=0.1, normalize=True
)

# Combined: L2 + first-order + second-order
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01,   # Weak L2
    beta=0.1,     # Moderate smoothness
    gamma=0.1,    # Moderate curvature penalty
    normalize=True
)
```

### β vs γ: When to Use Which

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Noisy flat continua | β only | Want to suppress all variations |
| Noisy power-law continua | γ only | Want to allow linear trends |
| Mixed features (continua + lines) | γ > β | Allow slopes, suppress sharp curves |
| Maximum noise reduction | β + γ together | Both constraints active |
| Preserving emission lines | Small β, γ | Minimal smoothing |

---

## Normalization Mode

### What Normalization Does

When `normalize=True` (default in v0.3.0), all regularization terms are divided by appropriate factors to make them dimensionless:

| Term | Normalization | Physical Meaning |
|------|---------------|------------------|
| χ² | $1/M_{\text{total}}$ | Reduced chi-squared |
| L2 (α) | $1/(TK)$ | Average squared element |
| First-order (β) | $1/((T-1)K)$ | Average squared slope |
| Second-order (γ) | $1/((T-2)K)$ | Average squared curvature |

### Why Normalization Matters

**Without normalization** (`normalize=False`):
- Parameters scale with problem size (T, K, N)
- α = 0.1 might work for T=4096 but fail for T=2048
- Need to retune parameters when changing grid resolution

**With normalization** (`normalize=True`):
- Parameters are dimensionless
- Same α, β, γ work across different problem sizes
- Easier to develop intuition for parameter values

### When to Disable Normalization

Only use `normalize=False` for:
- Backward compatibility with v0.2.x code
- Reproducing historical results
- Specific use cases requiring unnormalized behavior

### Example

```python
# Normalized mode (recommended)
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01, beta=0.1, gamma=0.1,
    normalize=True  # Default
)

# Unnormalized mode (legacy)
# Note: Parameters need different values!
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.1 * N,    # Scales with N
    beta=10.0,        # Large value needed
    gamma=0.0,
    normalize=False
)
```

---

## Parameter Tuning Guide

### Recommended Starting Values

```python
# Default configuration (good for most cases)
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01,   # Weak L2
    beta=0.1,     # Moderate first-order smoothness
    gamma=0.0,    # No curvature penalty
    normalize=True
)
```

### Tuning Strategy

1. **Start with defaults:** α=0.01, β=0.1, γ=0.0
2. **Check convergence:** Ensure loss decreases monotonically
3. **Inspect extracted spectra:** Look for unphysical features
4. **Adjust based on issues:**

| Issue | Solution |
|-------|----------|
| Spectra too noisy | Increase β (or add γ) |
| Spectra too flat | Decrease β |
| Spectra too bright | Increase α |
| Spectra too dim | Decrease α |
| Spurious narrow peaks | Increase β |
| Missing narrow features | Decrease β |
| Spectra too curved | Increase γ |
| Missing slopes | Decrease γ, or use γ only (β=0) |

### Validation Checklist

After tuning, verify:

- [ ] **Monotonic convergence:** Loss decreases at every iteration
- [ ] **Non-negativity:** All V and W values ≥ 0
- [ ] **Physical spectra:** Extracted components look like real astronomical spectra
- [ ] **Reconstruction quality:** Model predictions match observations reasonably
- [ ] **Reasonable χ²:** Final χ² ≈ 1.0 (for well-calibrated errors)

### Cross-Validation (Advanced)

For robust parameter selection:

```python
from sklearn.model_selection import KFold

# Split sources into train/validation sets
kf = KFold(n_splits=5)

for alpha in [0.001, 0.01, 0.1]:
    for beta in [0.0, 0.1, 1.0]:
        val_losses = []
        for train_idx, val_idx in kf.split(sources_data):
            # Train on subset
            V, W, _ = als_wnmf(
                [sources_data[i] for i in train_idx],
                [response_matrices[i] for i in train_idx],
                K=5, alpha=alpha, beta=beta, verbose=False
            )
            # Evaluate on held-out sources
            loss, *_ = compute_regularized_loss(
                [sources_data[i] for i in val_idx],
                [response_matrices[i] for i in val_idx],
                V, W
            )
            val_losses.append(loss)

        avg_loss = np.mean(val_losses)
        print(f"α={alpha}, β={beta}: val_loss={avg_loss:.4e}")
```

---

## Migration Guide: v0.2.x -> v0.3.0

### Important: v0.3.3 Gradient Scaling Fix

**If you used v0.3.0 - v0.3.2 with custom regularization parameters, read this:**

A critical bug was fixed in v0.3.3 where regularization gradients were not properly scaled to match the loss function evaluation. This caused the effective regularization strength to be incorrect:

| Mode | Bug Impact | Migration Action |
|------|------------|------------------|
| Normalized (`normalize=True`) | Regularization was ~M_total/2 weaker | Reduce α, β, γ by ~M_total/2 to maintain same behavior |
| Unnormalized (`normalize=False`) | Regularization was ~2× stronger | Increase α, β, γ by ~2× to maintain same behavior |

For typical SPHEREx datasets (M_total ~ 20 million), normalized mode regularization was approximately **10 million times weaker** than intended.

**Default parameters (α=0.01, β=0.0, γ=0.0)** now have the correct effective strength.

### Breaking API Change: `compute_regularized_loss()`

**v0.2.x (4-tuple):**
```python
total_loss, chi2, l2_term, smooth_term = compute_regularized_loss(...)
```

**v0.3.0 (5-tuple):**
```python
total_loss, chi2_norm, l2_term, smooth_term, curv_term = compute_regularized_loss(...)
```

### Default Alpha Changed

**v0.2.x:** Default α = 0.1 * N (unnormalized)

**v0.3.0:** Default α = 0.01 (normalized)

### New Parameter: gamma

v0.3.0 adds `gamma` parameter for second-order smoothness (curvature penalty).

### Migration Example

**v0.2.x code:**
```python
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=10.0,   # Unnormalized
    beta=5.0,     # Unnormalized
)
```

**v0.3.0 equivalent (normalized mode):**
```python
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=0.01,   # Normalized
    beta=0.1,     # Normalized
    gamma=0.0,    # New parameter
    normalize=True
)
```

**v0.3.0 (backward compatible):**
```python
V, W, history = als_wnmf(
    sources_data, response_matrices, K=5,
    alpha=10.0,   # Same as before
    beta=5.0,     # Same as before
    gamma=0.0,
    normalize=False  # Disable normalization
)
```

### Parameter Conversion Table

To convert from unnormalized (v0.2.x) to normalized (v0.3.0) parameters:

| Unnormalized | Normalized | Conversion |
|--------------|------------|------------|
| α = 0.1 * N | α ≈ 0.01 | Divide by ~10*N |
| β = 10.0 | β ≈ 0.1 | Divide by ~100 |
| γ = N/A | γ = 0.0 | New parameter |

Note: Exact conversion depends on T, K, and N. The table provides rough guidance.

---

## HALS Regularization (v0.4.0)

### Regularization in HALS

HALS (Hierarchical Alternating Least Squares) uses the same regularization parameters as ALS-WNMF (α, β, γ) with identical physical interpretations. The key difference is in how the regularization gradients are applied:

**ALS-WNMF (Simultaneous):**
- All components V[:,k] updated simultaneously using Multiplicative Update Rules
- Regularization applied uniformly to all components

**HALS (Sequential):**
- Components updated one at a time using Projected Gradient Descent
- Regularization applied per-component with deflation from other components

### Parameter Recommendations for HALS

Since HALS typically converges to a more orthogonal solution, you may need different regularization strengths:

| Parameter | ALS-WNMF | HALS | Reason |
|-----------|----------|------|--------|
| α (L2) | 0.01 | 0.005 - 0.01 | HALS naturally avoids large values |
| β (smooth) | 0.1 | 0.05 - 0.1 | Sequential updates already reduce noise |
| γ (curvature) | 0.0 | 0.0 - 0.05 | Optional for linear continua |

### HALS-Specific Considerations

**1. Regularization in Deflation**

When updating component k, HALS computes:
$$U_k = P - \sum_{j \neq k} M^{(j,k)} V[:,j]$$

The deflation term (sum over j≠k) implicitly reduces the need for strong regularization:
- Other components already capture shared structure
- Component k specializes to residual features
- Lower β often sufficient

**2. Convergence Tolerance**

HALS typically uses stricter tolerance than ALS-WNMF:
```python
# ALS-WNMF
V_als, W_als, _ = als_wnmf(..., tol=1e-4)

# HALS (stricter)
V_hals, W_hals, _ = hals_wnmf(..., tol=1e-5)
```

With stricter tolerance, you may reduce regularization slightly:
```python
# With tol=1e-5, use weaker regularization
V_hals, W_hals, _ = hals_wnmf(
    ..., alpha=0.005, beta=0.05, tol=1e-5
)
```

**3. Warm Start from ALS-WNMF**

When using HALS as refinement after ALS-WNMF, maintain same regularization:
```python
# Consistent regularization across algorithms
V_als, W_als, _ = als_wnmf(
    data, responses, K=40,
    alpha=0.01, beta=0.1, gamma=0.0
)

V_hals, W_hals, _ = hals_wnmf(
    data, responses, V_als, W_als,
    alpha=0.01, beta=0.1, gamma=0.0  # Same values
)
```

### Example Usage

**HALS with L2 + Smoothness:**
```python
from spxdictlearn import als_wnmf, hals_wnmf

# Initialize with ALS-WNMF
V_init, W_init, _ = als_wnmf(
    sources_data, response_matrices, K=40,
    alpha=0.01, beta=0.1, gamma=0.0
)

# Refine with HALS (slightly weaker regularization)
V_final, W_final, loss = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_init, W_init=W_init,
    alpha=0.005,   # Reduced: HALS naturally controls magnitude
    beta=0.05,     # Reduced: Deflation already smooths
    gamma=0.0,     # No curvature penalty
    tol=1e-5,      # Stricter convergence
    n_inner_iter=5
)
```

**HALS with Curvature Penalty:**
```python
# For extracting linear continuum components
V_final, W_final, loss = hals_wnmf(
    sources_data, response_matrices,
    V_init=V_init, W_init=W_init,
    alpha=0.005,   # Weak L2
    beta=0.0,      # No first-order (allow slopes)
    gamma=0.05,    # Curvature penalty (encourage linear)
    tol=1e-5
)
```

### Validation Checklist

After HALS with regularization, verify:
- [ ] **Monotonic convergence**: Loss decreases at every iteration
- [ ] **Non-negativity**: All V and W values ≥ 0
- [ ] **Component diversity**: Cosine similarity < 0.9 for most pairs
- [ ] **Physical spectra**: Components look like astronomical spectra
- [ ] **Reasonable χ²**: Final χ² ≈ 1.0 (for well-calibrated errors)

---

## Summary Table

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| α | 0.01 | 0.001 - 0.1 | Prevents large spectral values |
| β | 0.0 | 0.01 - 1.0 | Encourages smooth slopes |
| γ | 0.0 | 0.01 - 1.0 | Encourages linear/flat regions |
| normalize | True | - | Makes parameters dimensionless |

**Recommended starting point:**
```python
als_wnmf(..., alpha=0.01, beta=0.1, gamma=0.0, normalize=True)
```

---

## See Also

- [Technical Specification](./TechnicalSpec.md) - Mathematical details
- [API Reference](./APIReference.md) - Function signatures
- [Changelog](./CHANGELOG.md) - Version history
