# run_nmf.py — ALS-WNMF Mock-Data Demo

`run_nmf.py` runs ALS-WNMF (with L2 regularization) on synthetic data, saves the
estimated basis/weights and loss history, and produces QA plots comparing the
reconstruction against the known ground truth. It requires the optional `viz`/`io`
dependencies (`pip install -e ".[all]"`).

## Usage

```bash
# defaults: 50 sources, 3 components, alpha = 0.1 * N
python scripts/run_nmf.py

# custom run
python scripts/run_nmf.py -N 100 -M 10000 -K 5 --alpha 5.0 --max-iter 200 --seed 42
```

## Key parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--n-sources` | `-N` | 50 | number of synthetic sources |
| `--m-per-source` | `-M` | 5000 | observations per source |
| `--n-components` | `-K` | 3 | number of components (>= 3; the mock basis defines 3 features) |
| `--n-bins` | `-T` | 4096 | wavelength bins of the target grid |
| `--noise-level` | | 0.2 | fractional noise level |
| `--alpha` | | None | L2 regularization strength; `None` uses `0.1 * N`, `0.0` disables |
| `--max-iter` / `--tol` | | 300 / 1e-4 | iteration budget and convergence tolerance |
| `--n-jobs` | `-j` | -1 | parallel jobs for the E-step |
| `--output-dir` | `-o` | `results` | output directory |
| `--seed` | | 42 | random seed (data generation + initialization) |

## Output

- `V_estimated.csv/.npy`, `W_estimated.csv/.npy` — estimated basis and weights
- `loss_history.csv/.npy` — total loss (`chi2 + alpha*||V||^2`) per iteration
- `V_true.npy`, `W_true.npy` — ground truth (mock data only)
- `spectra_comparison.png`, `weights_comparison.png`, `reconstruction_quality.png` — QA plots

## Choosing alpha

Rule of thumb: `alpha ~ 0.1 * N` (default, balanced), `~0.01 * N` for high-S/N
data, `~1.0 * N` for noisy data or when extracted spectra develop non-physical
spikes. Compare a regularized run against `--alpha 0.0` and inspect
`spectra_comparison.png`.
