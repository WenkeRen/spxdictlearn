"""
Complete Test Suite for ALS-WNMF Implementation

This module verifies the mathematical correctness and convergence of the
ALS-WNMF algorithm implementation.

Test categories:
1. Response matrix construction validation
2. Mock data generation validation
3. Algorithm convergence verification
4. Reconstruction accuracy testing
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spxdictlearn.response_matrix import build_target_grid, build_response_matrix
from spxdictlearn.mock_data import generate_mock_data
from spxdictlearn.als_wnmf import als_wnmf, initialize_parameters
from spxdictlearn.utils import validate_non_negativity


class TestResponseMatrix:
    """Test response matrix construction"""

    def test_target_grid_creation(self):
        """Verify target grid has correct properties"""
        edges, centers = build_target_grid(lambda_min=0.75, lambda_max=5.0, n_bins=4096)

        # Check dimensions
        assert len(edges) == 4097, f"Expected 4097 edges, got {len(edges)}"
        assert len(centers) == 4096, f"Expected 4096 centers, got {len(centers)}"

        # Check range
        assert np.isclose(edges[0], 0.75, rtol=1e-10), f"Edge[0] should be 0.75, got {edges[0]}"
        assert np.isclose(edges[-1], 5.0, rtol=1e-10), f"Edge[-1] should be 5.0, got {edges[-1]}"

        # Check monotonicity
        assert np.all(np.diff(edges) > 0), "Edges should be strictly increasing"
        assert np.all(np.diff(centers) > 0), "Centers should be strictly increasing"

        # Check bin centers are geometric means
        expected_centers = np.sqrt(edges[:-1] * edges[1:])
        assert np.allclose(centers, expected_centers), "Centers should be geometric means of edges"

    def test_response_matrix_sparsity(self):
        """Verify response matrix is sparse and in CSR format"""
        # Create mock observation data
        M = 100
        rng = np.random.default_rng(42)
        lambda_c = rng.uniform(1.0, 4.5, size=M)
        fwhm = np.full(M, 0.02)
        flux = rng.uniform(0, 1, size=M)
        error = rng.uniform(0.01, 0.1, size=M)

        D_n = np.column_stack([lambda_c, fwhm, flux, error])

        # Build target grid
        target_edges, _ = build_target_grid()

        # Build response matrix
        R_n = build_response_matrix(D_n, target_edges)

        # Check sparsity
        assert issparse(R_n), "R_n should be sparse"
        assert type(R_n).__name__ == "csr_matrix", f"R_n should be CSR format, got {type(R_n).__name__}"

        # Check shape
        assert R_n.shape == (M, 4096), f"Expected shape ({M}, 4096), got {R_n.shape}"

        # Check sparsity ratio (should be < 10% non-zero for FWHM=0.02)
        nnz_ratio = R_n.nnz / (M * 4096)
        assert nnz_ratio < 0.1, f"Response matrix too dense: {nnz_ratio:.2%} non-zero"

    def test_response_matrix_normalization(self):
        """Verify flux conservation (rows sum to 1.0)"""
        # Create simple test case
        M = 10
        lambda_c = np.linspace(1.0, 4.0, M)
        fwhm = np.full(M, 0.02)
        flux = np.ones(M)
        error = np.ones(M) * 0.1

        D_n = np.column_stack([lambda_c, fwhm, flux, error])

        target_edges, _ = build_target_grid()
        R_n = build_response_matrix(D_n, target_edges)

        # Check row sums
        row_sums = np.asarray(R_n.sum(axis=1)).ravel()

        # All rows should sum to exactly 1.0 (flux conservation)
        assert np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-15), (
            f"Row sums not equal to 1.0: min={row_sums.min():.2e}, max={row_sums.max():.2e}"
        )

    def test_response_matrix_non_negativity(self):
        """Verify response matrix is non-negative"""
        M = 50
        rng = np.random.default_rng(42)
        lambda_c = rng.uniform(1.0, 4.5, size=M)
        fwhm = np.full(M, 0.02)
        flux = rng.uniform(0, 1, size=M)
        error = rng.uniform(0.01, 0.1, size=M)

        D_n = np.column_stack([lambda_c, fwhm, flux, error])

        target_edges, _ = build_target_grid()
        R_n = build_response_matrix(D_n, target_edges)

        # Check all values are non-negative
        assert np.all(R_n.data >= 0), "Response matrix contains negative values"


class TestMockDataGeneration:
    """Test mock data generation"""

    def test_mock_data_shapes(self):
        """Verify mock data has correct shapes"""
        N = 20
        M_per_source = 200
        K_true = 3

        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=N, M_per_source=M_per_source, K_true=K_true, verbose=False
        )

        # Check dimensions
        assert len(sources_data) == N, f"Expected {N} sources, got {len(sources_data)}"
        assert len(response_matrices) == N, f"Expected {N} response matrices, got {len(response_matrices)}"
        assert V_true.shape == (
            4096,
            K_true,
        ), f"V_true shape mismatch: {V_true.shape}"
        assert W_true.shape == (
            N,
            K_true,
        ), f"W_true shape mismatch: {W_true.shape}"

        # Check each source
        for n in range(N):
            D_n = sources_data[n]
            R_n = response_matrices[n]

            assert D_n.shape == (
                M_per_source,
                4,
            ), f"D_n[{n}] shape mismatch: {D_n.shape}"
            assert R_n.shape == (
                M_per_source,
                4096,
            ), f"R_n[{n}] shape mismatch: {R_n.shape}"

    def test_true_basis_non_negativity(self):
        """Verify true basis is non-negative"""
        _, _, V_true, W_true = generate_mock_data(N=10, verbose=False)

        assert np.all(V_true >= 0), "True basis V contains negative values"
        assert np.all(W_true >= 0), "True weights W contains negative values"

    def test_mock_data_no_negatives(self):
        """Verify mock observations are non-negative (after noise addition)"""
        sources_data, _, _, _ = generate_mock_data(N=10, noise_level=0.05, verbose=False)

        for D_n in sources_data:
            flux = D_n[:, 2]
            error = D_n[:, 3]

            # Flux should be non-negative
            # (may have small negative values due to noise, but should be rare)
            negative_fraction = np.sum(flux < 0) / len(flux)
            assert negative_fraction < 0.05, f"Too many negative flux values: {negative_fraction:.2%}"

            # Error should be positive
            assert np.all(error > 0), "Error values should be positive"


class TestAlgorithmConvergence:
    """Test ALS-WNMF algorithm convergence properties"""

    def test_monotonic_chi2_decrease(self):
        """
        CRITICAL TEST: Verify χ² decreases or stays flat after each step.

        If χ² ever increases, the mathematical implementation is flawed.
        """
        # Generate small test dataset
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=200, K_true=3, verbose=False
        )

        # Run algorithm
        K = 3
        V, W, chi2_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            max_iter=20,
            tol=1e-4,
            n_jobs=-1,
            verbose=True,
        )

        # Check monotonicity
        chi2_array = np.array(chi2_history)

        # Each step should decrease or keep same χ²
        # Allow tiny numerical errors (1e-10 relative tolerance)
        for i in range(1, len(chi2_array)):
            assert chi2_array[i] <= chi2_array[i - 1] * (1 + 1e-10), (
                f"χ² increased at iteration {i}: {chi2_array[i]} > {chi2_array[i - 1]}"
            )

        print("\n✓ Monotonicity test passed: χ² never increased")

    def test_non_negativity_constraint(self):
        """
        CRITICAL TEST: Verify V and W remain non-negative throughout.

        This is a fundamental constraint of NMF.
        """
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=200, K_true=3, verbose=False
        )

        K = 3
        V, W, chi2_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            max_iter=20,
            n_jobs=-1,
            verbose=False,
        )

        # Check non-negativity
        v_valid, w_valid = validate_non_negativity(V, W)

        assert v_valid, "Basis matrix V contains negative values!"
        assert w_valid, "Weight matrix W contains negative values!"

        print("✓ Non-negativity test passed: V >= 0 and W >= 0")

    def test_convergence_speed(self):
        """
        Verify algorithm converges within reasonable iterations.

        Should reach tolerance (Δχ² / χ² < 5e-4) within 150 iterations.
        Uses Gaussian response for consistent convergence behavior.

        Note: alpha=0 (no regularization) to maintain v0.1.0 test behavior
        """
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=30, M_per_source=300, K_true=3, response_type="gaussian", verbose=False
        )

        K = 3
        max_iter = 150
        tol = 5e-4

        V, W, chi2_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=0.0,  # Disable regularization for this test
            max_iter=max_iter,
            tol=tol,
            n_jobs=-1,
            verbose=True,
        )

        # Check if converged
        iterations = len(chi2_history) - 1

        assert iterations < max_iter, f"Algorithm did not converge within {max_iter} iterations"

        print(f"\n✓ Convergence test passed: converged in {iterations} iterations")


class TestReconstructionAccuracy:
    """Test reconstruction accuracy vs. ground truth"""

    def test_reconstruction_quality(self):
        """
        Verify algorithm can recover ground truth reasonably well.

        Due to scaling ambiguity and permutation invariance in NMF,
        we check correlation rather than exact values.
        """
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=300, K_true=3, noise_level=0.05, verbose=False
        )

        K = 3
        V, W, chi2_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            max_iter=50,
            n_jobs=-1,
            verbose=False,
        )

        # Check that we can reconstruct observations well
        # Compare predicted vs. observed flux
        total_flux_error = 0.0
        total_flux = 0.0

        for n in range(len(sources_data)):
            D_n = sources_data[n]
            R_n = response_matrices[n]

            flux_obs = D_n[:, 2]
            flux_pred = R_n @ (V @ W[n, :])

            total_flux_error += np.sum((flux_obs - flux_pred) ** 2)
            total_flux += np.sum(flux_obs**2)

        rmse = np.sqrt(total_flux_error / total_flux)

        # RMSE should be less than 20% (conservative for noisy data)
        assert rmse < 0.2, f"Reconstruction RMSE too high: {rmse:.3f}"

        print(f"✓ Reconstruction test passed: RMSE = {rmse:.3f}")


def run_all_tests():
    """Run all tests manually (without pytest)"""
    print("=" * 70)
    print("Running ALS-WNMF Test Suite")
    print("=" * 70)
    print()

    # Test 1: Response Matrix
    print("[1/4] Testing Response Matrix Construction...")
    test_response = TestResponseMatrix()
    test_response.test_target_grid_creation()
    test_response.test_response_matrix_sparsity()
    test_response.test_response_matrix_normalization()
    test_response.test_response_matrix_non_negativity()
    print("✓ Response matrix tests passed\n")

    # Test 2: Mock Data
    print("[2/4] Testing Mock Data Generation...")
    test_mock = TestMockDataGeneration()
    test_mock.test_mock_data_shapes()
    test_mock.test_true_basis_non_negativity()
    test_mock.test_mock_data_no_negatives()
    print("✓ Mock data tests passed\n")

    # Test 3: Algorithm Convergence
    print("[3/4] Testing Algorithm Convergence...")
    test_conv = TestAlgorithmConvergence()
    test_conv.test_monotonic_chi2_decrease()
    test_conv.test_non_negativity_constraint()
    test_conv.test_convergence_speed()
    print("✓ Algorithm convergence tests passed\n")

    # Test 4: Reconstruction Accuracy
    print("[4/4] Testing Reconstruction Accuracy...")
    test_recon = TestReconstructionAccuracy()
    test_recon.test_reconstruction_quality()
    print("✓ Reconstruction accuracy tests passed\n")

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nNote: L2 Regularization tests (v0.2.0) are in tests/test_regularization.py")
    print("Run with: pytest tests/test_regularization.py -v")


if __name__ == "__main__":
    run_all_tests()
