# SYSTEM SPECIFICATION: Global NMF for Heterogeneous Spectra

**Target Audience:** AI Coding Agent / Algorithm Engineer
**Task:** Implement an Alternating Least Squares Weighted Non-Negative Matrix Factorization (ALS-WNMF) algorithm to extract physically meaningful spectra from sparse, unaligned, and noisy observational data.

---

## 1. DATA STRUCTURES & TENSOR DIMENSIONS

All implementations MUST strictly adhere to these dimensions.

- **$N$**: Number of distinct astronomical sources. ($N \approx 3000$)
- **$T$**: Number of target high-resolution logarithmic wavelength bins. ($T = 4096$)
- **$K$**: Number of non-negative components to extract. ($K = 20$)
- **$M_n$**: Number of raw observations for the $n$-th source. (Variable per source, $M_n \in [8000, 12000]$)

### Input Data (Per Source $n \in [1, N]$)

The raw input for source $n$ is a 2D array $\mathbf{D}_n \in \mathbb{R}^{M_n \times 4}$.

- **Column 0 ($\lambda_c$)**: Observation central wavelength.
- **Column 1 ($bw$)**: Observation bandwidth, defined as the Full Width at Half Maximum (FWHM) of a Gaussian response.
- **Column 2 ($flux$)**: Maps to $\mathbf{y}_n \in \mathbb{R}^{M_n}$ (Observed flux array).
- **Column 3 ($error$)**: Maps to $\boldsymbol{\sigma}_n \in \mathbb{R}^{M_n}$ (1D array of Gaussian error standard deviations).

$\mathbf{R}_n \in \mathbb{R}^{M_n \times T}$: Truncated Gaussian response matrix. Must be strictly implemented as `scipy.sparse.csr_matrix`.

### Latent Variables (To be Optimized)

- **$\mathbf{V} \in \mathbb{R}^{T \times K}$**: The global high-resolution basis matrix (Eigenspectra). Constraint: $\mathbf{V} \ge 0$.
- **$\mathbf{W} \in \mathbb{R}^{N \times K}$**: The weight matrix. For a specific source $n$, its weight vector is $\mathbf{w}_n \in \mathbb{R}^{K}$ (the $n$-th row of $\mathbf{W}$). Constraint: $\mathbf{W} \ge 0$.

---

## 2. EXPLICIT MATHEMATICAL MODEL

### Global Objective Function ($\chi^2$)

$$\min_{\mathbf{V} \ge 0, \mathbf{W} \ge 0} \sum_{n=1}^N || \boldsymbol{\Sigma}_n^{-1/2} (\mathbf{y}_n - \mathbf{R}_n \mathbf{V} \mathbf{w}_n) ||_2^2$$

**Note for Agent:** $\boldsymbol{\Sigma}_n^{-1/2}$ is equivalent to element-wise multiplication by $1/\boldsymbol{\sigma}_n$. Do NOT construct dense diagonal covariance matrices.

---

## 3. CORE ALGORITHM LOGIC: ALS-WNMF

Implement the optimization using Alternating Least Squares. Loop the following E-step and M-step until convergence ($\Delta \chi^2 / \chi^2 < \text{tolerance}$).

### 3.1. E-step: Update Weights $\mathbf{w}_n$ (Parallelizable)

Fix $\mathbf{V}$, optimize $\mathbf{w}_n$ for each source independently. This is a standard Non-Negative Least Squares (NNLS) problem: $\min_{\mathbf{w}_n \ge 0} || \mathbf{A}_n \mathbf{w}_n - \mathbf{b}_n ||_2^2$.

**Computational steps per source $n$:**

1. Compute $\mathbf{y}'_n = \mathbf{y}_n \oslash \boldsymbol{\sigma}_n$ (Element-wise division, shape: $M_n$)
2. Compute dense matrix $\mathbf{U}_n = \mathbf{R}_n \mathbf{V}$ (Sparse dot Dense $\rightarrow$ Dense, shape: $M_n \times K$)
3. Compute $\mathbf{A}_n = \mathbf{U}_n \oslash \boldsymbol{\sigma}_n[:, \text{np.newaxis}]$ (Row-wise scaling, shape: $M_n \times K$)
4. Solve $\mathbf{w}_n = \text{scipy.optimize.nnls}(\mathbf{A}_n, \mathbf{y}'_n)[0]$

**Engineering Directive:** Use `joblib.Parallel` to parallelize this loop over all $N$ sources.

### 3.2. M-step: Update Basis $\mathbf{V}$ (Global Reduction)

Fix all $\mathbf{w}_n$, update the global matrix $\mathbf{V}$ using Multiplicative Update Rules (MUR) to guarantee non-negativity and monotonic convergence.

**Computational steps (Vectorized over $n$):**

Initialize Numerator $\mathbf{P} = \mathbf{0}_{T \times K}$ and Denominator $\mathbf{Q} = \mathbf{0}_{T \times K}$.

For each source $n \in [1, N]$:

1. Compute precision-weighted observation: $\mathbf{s}_n = \mathbf{y}_n \oslash (\boldsymbol{\sigma}_n \odot \boldsymbol{\sigma}_n)$ (Shape: $M_n$)
2. Compute model prediction: $\mathbf{\hat{y}}_n = \mathbf{R}_n (\mathbf{V} \mathbf{w}_n)$ (Shape: $M_n$)
3. Compute precision-weighted prediction: $\mathbf{q}_n = \mathbf{\hat{y}}_n \oslash (\boldsymbol{\sigma}_n \odot \boldsymbol{\sigma}_n)$ (Shape: $M_n$)
4. Accumulate to Numerator: $\mathbf{P} \mathrel{+}= (\mathbf{R}_n^T \mathbf{s}_n) \otimes \mathbf{w}_n^T$ (Outer product, $\mathbb{R}^T \times \mathbb{R}^K \rightarrow \mathbb{R}^{T \times K}$)
5. Accumulate to Denominator: $\mathbf{Q} \mathrel{+}= (\mathbf{R}_n^T \mathbf{q}_n) \otimes \mathbf{w}_n^T$ (Outer product, $\mathbb{R}^T \times \mathbb{R}^K \rightarrow \mathbb{R}^{T \times K}$)

**Final MUR Update:**

$$\mathbf{V}_{\text{new}} = \mathbf{V}_{\text{old}} \odot \frac{\mathbf{P}}{\mathbf{Q} + \epsilon_{\text{reg}}}$$

Where $\odot$ is element-wise multiplication, division is element-wise, and $\epsilon_{\text{reg}} = 10^{-12}$ to prevent division by zero.

---

## 4. CRITICAL IMPLEMENTATION DETAIL: Truncated Gaussian $\mathbf{R}_n$ Construction

The agent MUST implement an exact analytical integration of a Gaussian response over the logarithmic grid bins using `scipy.stats.norm.cdf`.

### Target Grid Definition

- $\lambda_{\text{min}} = 0.75$, $\lambda_{\text{max}} = 5.0$
- Grid edges: $T+1$ linearly spaced points in log-space: $\ln \boldsymbol{\lambda}_{\text{edges}} = \text{linspace}(\ln 0.75, \ln 5.0, 4097)$
- Array edges $= \exp(\ln \boldsymbol{\lambda}_{\text{edges}})$. Bin $t$ is bounded by [edges[t], edges[t+1]].

### Observation Mapping (For the $m$-th observation of source $n$)

Extract parameters from $\mathbf{D}_n[m, :]$: $\lambda_c = \text{col } 0$, $FWHM = \text{col } 1$.

1. Calculate Gaussian Standard Deviation: $\sigma_f = FWHM / (2 \sqrt{2 \ln 2}) \approx FWHM / 2.35482$
2. Truncation Bounds: $\lambda_{\text{start}} = \lambda_c - 3\sigma_f$, $\lambda_{\text{end}} = \lambda_c + 3\sigma_f$.

### Integration Logic

For every target grid bin $t \in [0, 4095]$ that overlaps with $[\lambda_{\text{start}}, \lambda_{\text{end}}]$:

1. Find actual integration bounds for this bin: $a = \max(\lambda_{\text{start}}, \text{edges}[t])$, $b = \min(\lambda_{\text{end}}, \text{edges}[t+1])$.
2. If $a < b$, compute the exact Gaussian integral over $[a, b]$ using the built-in Cumulative Distribution Function:

$$\mathbf{R}_n[m, t] = \text{norm.cdf}(b, \text{loc}=\lambda_c, \text{scale}=\sigma_f) - \text{norm.cdf}(a, \text{loc}=\lambda_c, \text{scale}=\sigma_f)$$

**(Agent directive: import norm from scipy.stats)**

### Flux Conservation Normalization

Since truncating at $3\sigma$ loses $\sim 0.27\%$ of the area, calculate the row sum `row_sum = np.sum(Rn[m, :])`. If `row_sum > 0`, divide the entire $m$-th row by `row_sum` so the row sums exactly to 1.0.

---

## 5. ROBUSTNESS & EDGE CASES (Agent Checklist)

- **Sparse Matrix Requirement:** $\mathbf{R}_n$ memory footprint is $M_n \times T \approx 4 \times 10^7$ floats $\approx 160$ MB if dense. With $N=3000$, total is $\sim 480$ GB. Agent MUST pre-compute $\mathbf{R}_n$ as `scipy.sparse.csr_matrix` and keep it in memory. Total footprint drops to $<2$ GB.

- **Initialization:** Initialize $\mathbf{V}$ and $\mathbf{W}$ using `np.random.uniform(low=0.1, high=1.0)`. Do not initialize with zeros.

- **Sparsity of Updates:** In the M-step, $\mathbf{R}_n^T \mathbf{s}_n$ is Sparse.T * Dense_Vector, which is highly efficient. Ensure scipy's sparse matrix-vector multiplication is utilized.

---

## 6. MOCK DATA GENERATION & TESTING SUITE

The agent MUST write a dedicated testing script (`test_pipeline.py`) to verify the mathematical correctness and convergence of the ALS-WNMF implementation.

### 6.1. Ground Truth Generation

**True Basis ($\mathbf{V}_{\text{true}}$):** Create $K_{\text{true}} = 3$ distinct artificial spectra on the $T=4096$ log grid:

- **Component 0:** A broad Gaussian continuum (e.g., center=2.0, $\sigma=1.0$).
- **Component 1:** A power-law curve (e.g., $f(\lambda) = \lambda^{-1.5}$).
- **Component 2:** Sparse narrow Gaussian emission lines (e.g., at $\lambda=1.5, 3.0$).

**True Weights ($\mathbf{W}_{\text{true}}$):** Generate for $N=50$ (small scale for quick testing) sources:

```python
np.random.uniform(0.1, 5.0, size=(50, 3))
```

### 6.2. Observation Simulation

For each source $n \in [1, 50]$:

1. Generate $M_n = 500$ random central wavelengths $\lambda_c \sim \text{Uniform}(0.8, 4.8)$.
2. Set constant $FWHM = 0.02$.
3. Build the ground truth high-res spectrum: $\mathbf{x}_n = \mathbf{V}_{\text{true}} (\mathbf{W}_{\text{true}})_{n}^T$
4. Build $\mathbf{R}_n$ using the exact logic defined in Section 4.
5. Compute true observation: $\mathbf{y}_{\text{true}} = \mathbf{R}_n \mathbf{x}_n$.
6. Add Noise: Set $\boldsymbol{\sigma}_n = 0.05 \times \text{mean}(\mathbf{y}_{\text{true}})$ (constant error array). Add Gaussian noise: $\mathbf{y}_n = \mathbf{y}_{\text{true}} + \mathcal{N}(0, \boldsymbol{\sigma}_n^2)$.
7. Pack into the target 2D array $\mathbf{D}_n \in \mathbb{R}^{500 \times 4}$.

### 6.3. Verification Assertions

Run the ALS-WNMF algorithm on the generated $\mathbf{D}_n$ datasets setting $K=3$. The testing script MUST assert/verify:

- **Monotonicity:** The global $\chi^2$ value MUST decrease or remain flat after every single E-step and M-step. If $\chi^2$ increases, the math implementation is flawed.
- **Non-negativity:** Assert `np.all(V >= 0)` and `np.all(W >= 0)` at all times.
- **Convergence:** The algorithm should trigger the tolerance stopping condition (e.g., $\Delta \chi^2 / \chi^2 < 10^{-4}$) within 100 iterations.
