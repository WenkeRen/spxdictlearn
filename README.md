# SPXPCA - Global NMF for Heterogeneous Spectra

Alternating Least Squares Weighted Non-Negative Matrix Factorization (ALS-WNMF) for extracting physically meaningful spectra from sparse, unaligned, and noisy astronomical observations.

## Overview

This package implements a specialized NMF algorithm designed for SPHEREx mission data analysis, particularly for the North Ecliptic Pole (NEP) region. The algorithm handles:

- **Sparse observations**: Each source has M_n ∈ [8000, 12000] observations
- **Unaligned wavelengths**: Observations at different central wavelengths
- **Heterogeneous data**: Variable bandwidth and noise characteristics
- **Non-negativity constraints**: Physical spectra and weights are non-negative

## Project Structure

```
spxdictlearn/
├── docs/
│   ├── StartUp.md              # Algorithm specification
│   ├── TechnicalSpec.md        # Full technical specification (v0.4.x)
│   ├── APIReference.md         # API reference
│   ├── HALS.md                 # HALS algorithm derivation
│   ├── Regularization.md       # alpha/beta/gamma guide
│   ├── Normalization.md        # basis normalization for comparable weights
│   └── CHANGELOG.md            # version history
├── src/spxdictlearn/
│   ├── response_matrix.py      # Sparse Gaussian response matrix (CSR)
│   ├── als_wnmf.py             # ALS-WNMF (parallel NNLS E-step + MUR M-step, fix_mask)
│   ├── hals.py                 # HALS-WNMF refinement (PGD inner loop, fix_mask)
│   ├── pruning.py              # Similarity-based dictionary pruning/merging
│   ├── normalization.py        # L1/L2 basis normalization + denormalization
│   ├── numba_nnls.py           # Optional GIL-free projected-gradient NNLS
│   ├── mock_data.py            # Synthetic ground-truth generation
│   └── utils.py                # Shared helpers (validation, penalties)
├── tests/
│   ├── test_pipeline.py        # End-to-end ALS-WNMF suite
│   ├── test_regularization.py  # L2/smoothness/curvature regularization suite
│   └── test_fix_mask.py        # Frozen-column (two-stage) correctness
├── scripts/
│   └── run_nmf.py              # Mock-data demo run (ALS-WNMF + QA plots)
├── pyproject.toml              # Project configuration
└── requirements.txt            # Dependencies
```

## Installation

```bash
# Using conda environment (recommended)
conda create -n spxdictlearn python=3.11
conda activate spxdictlearn

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Running Tests

```bash
# Run complete test suite (from the package root)
pytest -v
```

The test suite verifies:
- ✓ Response matrix sparsity and normalization
- ✓ Mock data generation correctness
- ✓ χ² monotonic decrease (math correctness)
- ✓ Non-negativity constraints
- ✓ Convergence within 100 iterations

### Running on Mock Data

```bash
# Basic usage (N=50 sources, K=3 components)
python scripts/run_nmf.py

# Custom parameters
python scripts/run_nmf.py \
    --n-sources 100 \
    --n-components 5 \
    --max-iter 50 \
    --tol 1e-3 \
    --output-dir results/my_run
```

Results are saved in both **human-readable CSV** and **NumPy** formats:

**CSV Files** (open with Excel, Pandas, R, or any text editor):
- `V_estimated.csv`: Basis spectra with wavelength column
  - wavelength_microns: Wavelength grid (first column)
  - component_0, component_1, ...: Spectral components
- `W_estimated.csv`: Source weights with IDs
  - source_id: Source identifier
  - component_0, component_1, ...: Weight values
- `loss_history.csv`: Convergence tracking
  - iteration, loss, delta_loss, rel_change (total loss including regularization)

**NumPy Files** (for Python/NumPy users):
- `V_estimated.npy`: T×K array
- `W_estimated.npy`: N×K array
- `loss_history.npy`: Convergence array

**Ground Truth** (for mock data runs):
- `V_true.npy`, `W_true.npy`: For comparison

## Algorithm Details

### Objective Function

min_{V ≥ 0, W ≥ 0} Σ_n || Σ_n^(-1/2) (y_n - R_n @ V @ w_n) ||_2^2

Where:
- `V ∈ R^(T×K)`: Global basis matrix (eigenspectra)
- `W ∈ R^(N×K)`: Weight matrix
- `R_n ∈ R^(M_n×T)`: Sparse Gaussian response matrix
- `y_n ∈ R^(M_n)`: Observed flux
- `Σ_n`: Diagonal covariance matrix (σ_n²)

### Algorithm Flow

1. **Initialize**: V, W ~ Uniform(0.1, 1.0)
2. **E-step**: Fix V, solve NNLS for each w_n (parallel)
3. **M-step**: Fix W, update V using Multiplicative Update Rules
4. **Iterate**: Until |Δχ²| / χ² < tolerance

### Key Features

- **Memory efficient**: Response matrices in CSR sparse format (~480GB → ~2GB)
- **Parallel computation**: E-step parallelized with joblib; optional GIL-free
  Numba NNLS backend for true thread parallelism
- **Mathematical correctness**: Exact Gaussian integration via CDF
- **Two-stage support**: `fix_mask` freezes chosen columns of V so a first-stage
  basis can be held fixed while new components are learned (see `test_fix_mask.py`)
- **HALS refinement**: block-coordinate updates break the MUR "twin spectra"
  symmetry; each column is solved by diagonally preconditioned projected gradient
- **Dictionary pruning**: cosine-similarity clustering + weighted merging
  compresses overcomplete dictionaries
- **Guaranteed convergence**: MUR ensures monotonic χ² decrease; negative
  low-S/N measurements are handled by projecting V back onto the feasible set

## Technical Specifications

- **Target grid**: user-defined number of logarithmic wavelength bins
  (e.g. 4096 bins over 0.75 - 5.0 μm)
- **Gaussian response**: Truncated at ±3σ with flux conservation
- **NNLS solver**: Numba projected-gradient (default when available, warm-started)
  with automatic fallback to `scipy.optimize.nnls`
- **Parallelization**: joblib.Parallel (all CPUs by default)

## Testing

The implementation has been rigorously tested against the specification in `docs/StartUp.md`:

1. **Response Matrix**:
   - ✓ CSR sparse format
   - ✓ Row normalization (sum = 1.0)
   - ✓ Non-negative values
   - ✓ Exact Gaussian integration

2. **Algorithm**:
   - ✓ χ² monotonic decrease (verified at every step)
   - ✓ Non-negativity maintained (V ≥ 0, W ≥ 0)
   - ✓ Convergence within 100 iterations (tol = 1e-4)

3. **Reconstruction**:
   - ✓ RMSE < 20% on noisy test data
   - ✓ Component recovery

## References

See `docs/StartUp.md` for complete mathematical derivation and implementation details.

## License

MIT

## Authors

Wenke Ren (2025-2026)
