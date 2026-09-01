# Basis Spectrum Normalization

**Version:** 0.4.0
**Last Updated:** 2026-03-04

This document explains the normalization module for making NMF weights comparable across components.

---

## The Scale Ambiguity Problem

In Non-negative Matrix Factorization (NMF), the decomposition has a fundamental **scale ambiguity**:

$$Y \approx V \cdot W^T$$

For any non-zero scalar $s_k$:

$$V_{:,k} \cdot W_{:,k}^T = (s_k^{-1} \cdot V_{:,k}) \cdot (s_k \cdot W_{:,k})^T$$

This means multiplying a basis spectrum by a factor and dividing the corresponding weights by the same factor produces an **identical reconstruction**.

### Why This Matters for Weight Comparison

Without normalization:
- A component with large amplitude in V will have **small weights** in W
- A component with small amplitude in V will have **large weights** in W
- **Weight values are not comparable across components**

This is problematic for:
- **Importance ranking**: Sorting components by total weight is misleading
- **Thresholding**: Setting a weight threshold affects components differently
- **Clustering**: Weight-based similarity measures are distorted

---

## Normalization Methods

The module provides two normalization methods:

### L1 Normalization (Recommended)

$$\text{mean}(V_{:,k}) = 1 \quad \forall k$$

**Scale factor:** $s_k = \text{mean}(V_{:,k})$

**Normalized values:**
- $V_{\text{norm}}[:, k] = V[:, k] / s_k$
- $W_{\text{norm}}[:, k] = W[:, k] \cdot s_k$

**When to use:**
- Weight-based importance ranking
- Astronomical spectra (average flux = 1 is intuitive)
- Component thresholding

### L2 Normalization

$$||V_{:,k}||_2 = 1 \quad \forall k$$

**Scale factor:** $s_k = ||V_{:,k}||_2$

**When to use:**
- PCA-like interpretations
- Orthonormal basis analysis
- When L2 norm has physical meaning

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from spxdictlearn import normalize_basis_l1

# Load ALS-WNMF results
V = np.load("V_estimated.npy")  # Shape: (T, K)
W = np.load("W_estimated.npy")  # Shape: (N, K)

# Normalize (L1: mean = 1)
V_norm, W_norm, scales = normalize_basis_l1(V, W)

# Verify normalization
print(f"Mean of each component: {np.mean(V_norm, axis=0)}")
# Output: [1.0, 1.0, ..., 1.0]

# Verify reconstruction is preserved
print(f"Reconstruction preserved: {np.allclose(V @ W.T, V_norm @ W_norm.T)}")
# Output: True
```

### Integration with Pruning

```python
from spxdictlearn import normalize_basis_l1, prune_and_sort_dictionary

# Step 1: Normalize basis spectra
V_norm, W_norm, scales = normalize_basis_l1(V, W)

# Step 2: Now weights are comparable - use for pruning
V_pruned, W_pruned, info = prune_and_sort_dictionary(
    V_norm, W_norm,
    similarity_threshold=0.85
)

print(f"Components: {V.shape[1]} -> {V_pruned.shape[1]}")
```

### General Interface

```python
from spxdictlearn import normalize_basis

# L1 normalization (default)
V_norm, W_norm, scales = normalize_basis(V, W, method="l1")

# L2 normalization
V_norm, W_norm, scales = normalize_basis(V, W, method="l2")
```

### Denormalization

If you need to recover the original scale:

```python
from spxdictlearn import normalize_basis_l1, denormalize_basis

# Normalize
V_norm, W_norm, scales = normalize_basis_l1(V, W)

# Do some processing...

# Recover original scale
V_recovered, W_recovered = denormalize_basis(V_norm, W_norm, scales)

# V_recovered ≈ V, W_recovered ≈ W
```

---

## Mathematical Details

### Reconstruction Preservation

The normalization is designed to preserve the exact reconstruction:

$$\begin{aligned}
V_{\text{norm}} \cdot W_{\text{norm}}^T &= \sum_{k=1}^{K} V_{\text{norm}}[:,k] \cdot W_{\text{norm}}[:,k]^T \\
&= \sum_{k=1}^{K} \frac{V[:,k]}{s_k} \cdot (W[:,k] \cdot s_k)^T \\
&= \sum_{k=1}^{K} V[:,k] \cdot W[:,k]^T \\
&= V \cdot W^T
\end{aligned}$$

This means normalized results can be used anywhere the original V, W were used.

### Weight Distribution Changes

After L1 normalization:

| Metric | Before | After |
|--------|--------|-------|
| Component 1 total weight | 1000 | 500 |
| Component 2 total weight | 100 | 500 |
| Ratio | 10:1 | 1:1 |

If both components contribute equally to the reconstruction (same integrated flux), their normalized weights will be comparable.

---

## Edge Cases

### Zero-Mean Components

If a component has zero or near-zero mean:

```python
V[:, k] = np.array([1, -1, 1, -1, ...])  # Zero mean
```

The normalization will:
1. Issue a `RuntimeWarning`
2. Skip normalization for that component (scale = 1.0)
3. Suggest removing the component

### Handling Warnings

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    V_norm, W_norm, scales = normalize_basis_l1(V, W)

    if w:
        print(f"Warning: {w[0].message}")
        # Check which components have issues
        zero_components = [i for i, s in enumerate(scales) if s == 1.0]
        print(f"Problematic components: {zero_components}")
```

---

## Best Practices

1. **Always normalize before weight-based operations:**
   - Importance ranking
   - Thresholding
   - Clustering based on weights

2. **Use L1 normalization for astronomical spectra:**
   - More intuitive (average flux = 1)
   - Better for physical interpretation

3. **Save the scale factors:**
   ```python
   V_norm, W_norm, scales = normalize_basis_l1(V, W)
   np.save("scales.npy", scales)  # Save for potential denormalization
   ```

4. **Normalize once at the beginning:**
   - Don't re-normalize after pruning
   - The pruned results maintain the normalization

5. **Verify reconstruction preservation:**
   ```python
   assert np.allclose(V @ W.T, V_norm @ W_norm.T), "Reconstruction not preserved!"
   ```

---

## See Also

- [API Reference: Normalization](./APIReference.md#normalization-functions)
- [Dictionary Pruning](./Pruning.md) - Uses normalized weights for importance ranking
- [Technical Specification](./TechnicalSpec.md) - Mathematical background
