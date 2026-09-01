"""
ALS-WNMF Execution Script

This script runs the ALS-WNMF algorithm on either mock or real data.
Saves results (V, W, chi2_history) to output files.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spxdictlearn.als_wnmf import als_wnmf
from spxdictlearn.mock_data import generate_mock_data


def run_on_mock_data(
    N: int = 50,
    M_per_source: int = 500,
    K: int = 3,
    T: int = 4096,
    noise_level: float = 0.05,
    alpha: float | None = None,
    beta: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_jobs: int = -1,
    output_dir: str = "results",
    seed: int = 42,
):
    """
    Run ALS-WNMF on mock data.

    Parameters
    ----------
    N : int
        Number of sources
    M_per_source : int
        Observations per source
    K : int
        Number of components
    T : int
        Number of wavelength bins
    noise_level : float
        Noise level (fraction of mean flux)
    alpha : float, optional
        L2 regularization coefficient. If None (default), uses 0.1*N.
        Set to 0.0 to disable L2 regularization.
    beta : float, optional
        Smoothness regularization coefficient. Default is 0.0 (disabled).
        Set beta > 0.0 to enable smoothness constraint on basis spectra.
        Typical range: 0.1 to 100.0 depending on flux scale.
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs)
    output_dir : str
        Output directory for results
    seed : int
        Random seed
    """
    print("=" * 70)
    print("ALS-WNMF on Mock Data")
    print("=" * 70)
    print()

    # Generate mock data
    print("Generating mock data...")
    sources_data, response_matrices, V_true, W_true = generate_mock_data(
        N=N,
        M_per_source=M_per_source,
        K_true=K,
        T=T,
        noise_level=noise_level,
        seed=seed,
        verbose=True,
    )

    print()
    print("Running ALS-WNMF...")
    print()

    # Run algorithm (with optional L2 and smoothness regularization)
    V, W, loss_history = als_wnmf(
        sources_data=sources_data,
        response_matrices=response_matrices,
        K=K,
        T=T,
        alpha=alpha,
        beta=beta,
        tol=tol,
        max_iter=max_iter,
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save as CSV for cross-language compatibility
    import pandas as pd

    # Get wavelength grid for V matrix
    from spxdictlearn.response_matrix import build_target_grid

    _, wavelength_centers = build_target_grid(lambda_min=0.75, lambda_max=5.0, n_bins=T)

    # Save V (basis spectra) with wavelength column
    V_df = pd.DataFrame(V, columns=[f"component_{k}" for k in range(K)])
    V_df.insert(0, "wavelength_microns", wavelength_centers)
    V_df.to_csv(output_path / "V_estimated.csv", index=False, float_format="%.6e")

    # Save W (weights) with source IDs
    W_df = pd.DataFrame(W, columns=[f"component_{k}" for k in range(K)])
    W_df.insert(0, "source_id", range(N))
    W_df.to_csv(output_path / "W_estimated.csv", index=False, float_format="%.6e")

    # Save loss history (includes chi2 + regularization term)
    n_iter = len(loss_history)
    delta_loss = np.zeros(n_iter)
    delta_loss[1:] = np.diff(loss_history)
    rel_change = np.zeros(n_iter)
    for i in range(1, n_iter):
        if loss_history[i] > 0:
            rel_change[i] = abs(loss_history[i] - loss_history[i - 1]) / loss_history[i]

    loss_df = pd.DataFrame(
        {"iteration": range(n_iter), "loss": loss_history, "delta_loss": delta_loss, "rel_change": rel_change}
    )
    loss_df.to_csv(output_path / "loss_history.csv", index=False, float_format="%.6e")

    # Also save NPY for numpy users
    np.save(output_path / "V_estimated.npy", V)
    np.save(output_path / "W_estimated.npy", W)
    np.save(output_path / "loss_history.npy", np.array(loss_history))

    print()
    print("=" * 70)
    print(f"Results saved to {output_path}/")
    print("  - V_estimated.csv (human-readable)")
    print("  - W_estimated.csv (human-readable)")
    print("  - loss_history.csv (human-readable)")
    print("  - V_estimated.npy (NumPy format)")
    print("  - W_estimated.npy (NumPy format)")
    print("  - loss_history.npy (NumPy format)")

    # Log regularization info
    if alpha is None:
        actual_alpha = 0.1 * N
    else:
        actual_alpha = alpha
    print(f"  - L2 regularization: α = {actual_alpha:.6f}")
    print(f"  - Smoothness regularization: β = {beta:.6f}")
    print("=" * 70)

    # Generate QA plots comparing estimated vs true values
    print()
    print("Generating QA plots...")
    print()

    # Find optimal permutation to match components
    perm, correlations = find_optimal_permutation(V_true, V)
    V_estimated_permuted = V[:, perm]
    W_estimated_permuted = W[:, perm]

    # Print matching summary
    print("Component matching (true -> estimated):")
    for k in range(K):
        print(f"  True component {k} -> Estimated component {perm[k]} (correlation: {correlations[k]:.3f})")
    print()

    # Generate plots
    print("Saving plots:")
    plot_spectra_comparison(V_true, V_estimated_permuted, wavelength_centers, output_path, perm, correlations)
    plot_weights_comparison(W_true, W_estimated_permuted, output_path, perm, correlations)
    plot_reconstruction_quality(sources_data, response_matrices, V, W, output_path, n_samples=5)

    print()
    print("=" * 70)
    print("QA plots successfully generated!")
    print("=" * 70)


def find_optimal_permutation(V_true: np.ndarray, V_estimated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find optimal permutation to match estimated components to true components.

    Uses correlation between normalized spectra to find the best matching.
    Handles the permutation ambiguity of NMF.

    Parameters
    ----------
    V_true : np.ndarray
        True basis spectra, shape (T, K_true)
    V_estimated : np.ndarray
        Estimated basis spectra, shape (T, K_est)

    Returns
    -------
    perm : np.ndarray
        Permutation indices to reorder V_estimated to match V_true
    correlations : np.ndarray
        Correlation coefficients for each matched pair
    """
    from scipy.optimize import linear_sum_assignment

    K_true = V_true.shape[1]
    K_est = V_estimated.shape[1]

    # Normalize spectra to unit length for correlation computation
    V_true_norm = V_true / np.linalg.norm(V_true, axis=0, keepdims=True)
    V_est_norm = V_estimated / np.linalg.norm(V_estimated, axis=0, keepdims=True)

    # Compute correlation matrix
    corr_matrix = V_true_norm.T @ V_est_norm

    # Use Hungarian algorithm to find optimal matching (maximize total correlation)
    # linear_sum_assignment minimizes cost, so we use negative correlation
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)

    # Create permutation array
    perm = np.zeros(K_true, dtype=int)
    correlations = np.zeros(K_true)

    for i, j in zip(row_ind, col_ind):
        perm[i] = j
        correlations[i] = corr_matrix[i, j]

    return perm, correlations


def plot_spectra_comparison(
    V_true: np.ndarray,
    V_estimated: np.ndarray,
    wavelength: np.ndarray,
    output_path: Path,
    perm: np.ndarray,
    correlations: np.ndarray,
) -> None:
    """
    Plot comparison between true and estimated basis spectra.

    Parameters
    ----------
    V_true : np.ndarray
        True basis spectra, shape (T, K)
    V_estimated : np.ndarray
        Estimated basis spectra (permuted), shape (T, K)
    wavelength : np.ndarray
        Wavelength grid in microns, shape (T,)
    output_path : Path
        Directory to save the plot
    perm : np.ndarray
        Permutation used for matching
    correlations : np.ndarray
        Correlation coefficients for each component
    """
    K = V_true.shape[1]

    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K), sharex=True)
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        ax.plot(wavelength, V_true[:, k], "b-", label="True", linewidth=2, alpha=0.8)
        ax.plot(
            wavelength,
            V_estimated[:, k],
            color="orange",
            linestyle="-",
            label="Estimated",
            linewidth=2,
            alpha=0.8,
        )
        ax.set_ylabel(r"Flux [arbitrary units]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Component {k} (from estimated component {perm[k]}) - Correlation: {correlations[k]:.3f}")

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
    plt.suptitle("Basis Spectra Comparison: True vs Estimated", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / "spectra_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  - spectra_comparison.png")


def plot_weights_comparison(
    W_true: np.ndarray,
    W_estimated: np.ndarray,
    output_path: Path,
    perm: np.ndarray,
    correlations: np.ndarray,
) -> None:
    """
    Plot comparison between true and estimated source weights.

    Parameters
    ----------
    W_true : np.ndarray
        True source weights, shape (N, K)
    W_estimated : np.ndarray
        Estimated source weights (permuted), shape (N, K)
    output_path : Path
        Directory to save the plot
    perm : np.ndarray
        Permutation used for matching
    correlations : np.ndarray
        Correlation coefficients for each component
    """
    K = W_true.shape[1]

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        ax.scatter(W_true[:, k], W_estimated[:, k], alpha=0.5, s=20)

        # Add 1:1 reference line
        min_val = min(W_true[:, k].min(), W_estimated[:, k].min())
        max_val = max(W_true[:, k].max(), W_estimated[:, k].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1, label="1:1 line")

        # Calculate R^2
        correlation_matrix = np.corrcoef(W_true[:, k], W_estimated[:, k])
        r_squared = correlation_matrix[0, 1] ** 2

        ax.set_xlabel(r"True Weight")
        ax.set_ylabel(r"Estimated Weight")
        ax.set_title(f"Component {k} (R$^2$ = {r_squared:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Source Weights Comparison: True vs Estimated", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "weights_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  - weights_comparison.png")


def plot_reconstruction_quality(
    sources_data: list[np.ndarray],
    response_matrices: list,
    V: np.ndarray,
    W: np.ndarray,
    output_path: Path,
    n_samples: int = 5,
) -> None:
    """
    Plot reconstruction quality for a sample of sources.

    Parameters
    ----------
    sources_data : list of np.ndarray
        List of source data arrays, each shape (M_n, 4)
    response_matrices : list of sparse matrices
        List of response matrices for each source
    V : np.ndarray
        Estimated basis spectra, shape (T, K)
    W : np.ndarray
        Estimated source weights, shape (N, K)
    output_path : Path
        Directory to save the plot
    n_samples : int
        Number of sources to sample for visualization
    """
    N = len(sources_data)
    sample_indices = np.random.RandomState(seed=42).choice(N, size=min(n_samples, N), replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for idx, source_idx in enumerate(sample_indices):
        ax = axes[idx]

        # Get data for this source
        data = sources_data[source_idx]
        R = response_matrices[source_idx]

        # Extract wavelength, flux, and error
        lambda_c = data[:, 0]  # Central wavelength
        flux_obs = data[:, 2]  # Observed flux
        flux_err = data[:, 3]  # Measurement error

        # Reconstruct flux: y_pred = R @ V @ w
        w_n = W[source_idx, :]
        flux_pred = R @ V @ w_n

        # Calculate RMSE
        rmse = np.sqrt(np.mean((flux_obs - flux_pred) ** 2))
        mean_flux = np.mean(flux_obs)
        relative_rmse = (rmse / mean_flux) * 100 if mean_flux > 0 else 0

        # Sort by wavelength for smooth predicted curve
        sort_idx = np.argsort(lambda_c)
        lambda_c_sorted = lambda_c[sort_idx]
        flux_obs_sorted = flux_obs[sort_idx]
        flux_err_sorted = flux_err[sort_idx]
        flux_pred_sorted = flux_pred[sort_idx]

        # Plot
        ax.errorbar(
            lambda_c_sorted,
            flux_obs_sorted,
            yerr=flux_err_sorted,
            fmt="o",
            alpha=0.5,
            markersize=3,
            label="Observed",
            capsize=2,
            zorder=1,
        )
        ax.plot(lambda_c_sorted, flux_pred_sorted, "r-", linewidth=1.5, alpha=0.8, label="Predicted", zorder=5)

        ax.set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
        ax.set_ylabel(r"Flux [arbitrary units]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Source {source_idx} - RMSE: {rmse:.3f} ({relative_rmse:.1f}%)")

    plt.suptitle("Reconstruction Quality: Observed vs Predicted", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / "reconstruction_quality.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  - reconstruction_quality.png")


def main():
    parser = argparse.ArgumentParser(
        description="Run ALS-WNMF algorithm on mock data with L2 and smoothness regularization (v0.2.1)"
    )
    parser.add_argument("-N", "--n-sources", type=int, default=300, help="Number of sources")
    parser.add_argument(
        "-M",
        "--m-per-source",
        type=int,
        default=1000,
        help="Observations per source",
    )
    parser.add_argument("-K", "--n-components", type=int, default=3, help="Number of components")
    parser.add_argument("-T", "--n-bins", type=int, default=4096, help="Number of wavelength bins")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Noise level (fraction of mean flux)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="L2 regularization coefficient (default: 0.1*N, use 0.0 to disable)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=100.0,
        help="Smoothness regularization coefficient (default: 0.0, disabled)",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance")
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all CPUs)",
    )
    parser.add_argument("-o", "--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_on_mock_data(
        N=args.n_sources,
        M_per_source=args.m_per_source,
        K=args.n_components,
        T=args.n_bins,
        noise_level=args.noise_level,
        alpha=args.alpha,
        beta=args.beta,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
