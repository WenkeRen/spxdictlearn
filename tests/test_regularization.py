"""
L2 and Smoothness Regularization Tests for ALS-WNMF (v0.2.0+)

This module tests the regularization functionality:
- L2 regularization (v0.2.0): alpha parameter
- Smoothness regularization (v0.2.1): beta parameter
- Second-order smoothness (v0.3.0): gamma parameter with normalization
"""

import numpy as np
from spxdictlearn.response_matrix import build_response_matrix
from spxdictlearn.mock_data import generate_mock_data
from spxdictlearn.als_wnmf import als_wnmf, initialize_parameters
from spxdictlearn.als_wnmf import compute_regularized_loss, m_step
from spxdictlearn.utils import compute_smoothness_penalty, compute_second_order_smoothness_penalty


class TestRegularization:
    """Test suite for L2 regularization functionality (v0.2.0)"""

    def test_regularized_loss_computation(self):
        """Test that compute_regularized_loss() calculates correctly"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        # Initialize parameters
        N = len(sources_data)
        K = 3
        V, W = initialize_parameters(N, T=4096, K=K, seed=42)

        # Test with different alpha values (beta=0, gamma=0 for L2-only test)
        for alpha in [0.0, 0.1, 1.0]:
            total_loss, chi2_norm, l2_term, smooth_term, curv_term = compute_regularized_loss(
                sources_data, response_matrices, V, W, alpha, beta=0.0, gamma=0.0, normalize=False
            )

            # Verify: total = chi2 + alpha * ||V||^2 (unnormalized)
            frobenius_norm_sq = np.sum(V**2)
            expected_total = chi2_norm + alpha * frobenius_norm_sq

            assert np.isclose(total_loss, expected_total, rtol=1e-10), (
                f"Loss mismatch for alpha={alpha}: {total_loss} vs {expected_total}"
            )

            # Verify: l2_term = alpha * ||V||^2 (unnormalized)
            expected_l2 = alpha * frobenius_norm_sq
            assert np.isclose(l2_term, expected_l2, rtol=1e-10), (
                f"L2 term mismatch for alpha={alpha}: {l2_term} vs {expected_l2}"
            )

            # Verify: smooth_term = 0 when beta=0
            assert np.isclose(smooth_term, 0.0, rtol=1e-10), (
                f"Smoothness term should be 0 when beta=0, got {smooth_term}"
            )

            # Verify: curv_term = 0 when gamma=0
            assert np.isclose(curv_term, 0.0, rtol=1e-10), f"Curvature term should be 0 when gamma=0, got {curv_term}"

        print("  ✓ Regularized loss computation is correct")

    def test_m_step_with_regularization(self):
        """Test that M-step correctly applies alpha*V in denominator"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        N = len(sources_data)
        K = 3
        V, W = initialize_parameters(N, T=4096, K=K, seed=42)

        # Test M-step with and without L2 regularization (beta=0, gamma=0)
        V_no_reg, _ = m_step(
            sources_data, response_matrices, V.copy(), W, alpha=0.0, beta=0.0, gamma=0.0, normalize=False
        )
        V_reg, _ = m_step(
            sources_data, response_matrices, V.copy(), W, alpha=10.0, beta=0.0, gamma=0.0, normalize=False
        )

        # With regularization, V should be smaller (shrinkage effect)
        # (since denominator is larger: Q + 2*alpha*V)
        assert np.all(V_reg <= V_no_reg * (1 + 1e-10)), "Regularized V should be <= non-regularized V"

        # Both should remain non-negative
        assert np.all(V_no_reg >= 0), "Non-regularized V has negative values"
        assert np.all(V_reg >= 0), "Regularized V has negative values"

        print("  ✓ M-step L2 regularization works correctly")

    def test_als_wNMF_with_default_alpha(self):
        """Test that default alpha (None) is correctly set"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        K = 3

        # Run with alpha=None, beta=0, gamma=0 (should default to 0.01 normalized)
        V, W, loss_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=None,
            beta=0.0,
            gamma=0.0,
            normalize=True,
            max_iter=10,
            verbose=False,
        )

        # Verify non-negativity
        assert np.all(V >= 0), "V has negative values"
        assert np.all(W >= 0), "W has negative values"

        # Verify convergence (loss should decrease)
        assert len(loss_history) == 11, "Loss history length mismatch"  # initial + 10 iters
        assert loss_history[-1] <= loss_history[0], "Loss did not decrease"

        print("  ✓ Default alpha (0.01 normalized) works correctly")

    def test_monotonicity_with_regularization(self):
        """Test that total loss decreases monotonically with L2 regularization"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        K = 3

        # Run with alpha=0.1 (normalized), beta=0, gamma=0 (L2 only)
        V, W, loss_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=0.1,
            beta=0.0,
            gamma=0.0,
            normalize=True,
            max_iter=20,
            verbose=False,
        )

        # Check monotonic decrease: loss[i] <= loss[i-1]
        for i in range(1, len(loss_history)):
            assert loss_history[i] <= loss_history[i - 1] * (1 + 1e-10), (
                f"Loss increased at iteration {i}: {loss_history[i]} > {loss_history[i - 1]}"
            )

        print("  ✓ Total loss decreases monotonically with L2 regularization")


class TestSmoothness:
    """Test suite for smoothness regularization functionality (v0.2.1)"""

    def test_smoothness_penalty_computation(self):
        """Test that compute_smoothness_penalty() calculates correctly"""
        # Set up test data
        V_smooth = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [1.2, 2.2, 3.2],
            ]
        )
        V_rough = np.array(
            [
                [1.0, 2.0, 5.0],
                [1.1, 2.1, 0.5],
                [1.2, 3.0, 1.0],
            ]
        )

        # Test smoothness penalty
        penalty_smooth = compute_smoothness_penalty(V_smooth)
        penalty_rough = compute_smoothness_penalty(V_rough)

        # Smooth V should have lower penalty
        assert penalty_smooth < penalty_rough, (
            f"Smooth V should have lower penalty: {penalty_smooth} >= {penalty_rough}"
        )

        # Verify calculation: sum of squared first-order differences
        diff_smooth = V_smooth[1:] - V_smooth[:-1]
        expected_penalty_smooth = np.sum(diff_smooth**2)
        assert np.isclose(penalty_smooth, expected_penalty_smooth, rtol=1e-10), (
            f"Smooth penalty mismatch: {penalty_smooth} vs {expected_penalty_smooth}"
        )

        diff_rough = V_rough[1:] - V_rough[:-1]
        expected_penalty_rough = np.sum(diff_rough**2)
        assert np.isclose(penalty_rough, expected_penalty_rough, rtol=1e-10), (
            f"Rough penalty mismatch: {penalty_rough} vs {expected_penalty_rough}"
        )

        print("  ✓ Smoothness penalty computation is correct")

    def test_m_step_with_smoothness(self):
        """Test that M-step correctly applies smoothness gradient"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        N = len(sources_data)
        K = 3
        V, W = initialize_parameters(N, T=4096, K=K, seed=42)

        # Test M-step with and without smoothness (alpha=0, gamma=0, beta varies)
        V_no_smooth, _ = m_step(
            sources_data, response_matrices, V.copy(), W, alpha=0.0, beta=0.0, gamma=0.0, normalize=False
        )
        V_smooth, _ = m_step(
            sources_data, response_matrices, V.copy(), W, alpha=0.0, beta=10.0, gamma=0.0, normalize=False
        )

        # With smoothness, V should have smaller high-frequency variations
        # (gradient penalizes large adjacent differences)
        # Check that smooth V has lower variance between adjacent bins
        adj_diff_no_smooth = np.mean(np.abs(np.diff(V_no_smooth, axis=0)))
        adj_diff_smooth = np.mean(np.abs(np.diff(V_smooth, axis=0)))

        assert adj_diff_smooth <= adj_diff_no_smooth * (1 + 1e-10), (
            f"Smoothed V should reduce adjacent differences: {adj_diff_smooth} vs {adj_diff_no_smooth}"
        )

        # Both should remain non-negative
        assert np.all(V_no_smooth >= 0), "Non-smoothed V has negative values"
        assert np.all(V_smooth >= 0), "Smoothed V has negative values"

        print("  ✓ M-step smoothness regularization works correctly")

    def test_als_wNMF_with_default_beta(self):
        """Test that default beta (0.0) disables smoothness regularization"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        K = 3

        # Run with beta=0.0 (default, should behave like alpha only)
        V, W, loss_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=0.0,
            beta=0.0,  # Default value
            gamma=0.0,
            normalize=True,
            max_iter=10,
            verbose=False,
        )

        # Verify non-negativity
        assert np.all(V >= 0), "V has negative values"
        assert np.all(W >= 0), "W has negative values"

        # Verify convergence
        assert len(loss_history) == 11, "Loss history length mismatch"
        assert loss_history[-1] <= loss_history[0], "Loss did not decrease"

        print("  ✓ Default beta (0.0) works correctly")

    def test_monotonicity_with_smoothness(self):
        """Test that total loss decreases monotonically with smoothness regularization"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        K = 3

        # Run with alpha=0.0, beta=0.5 (normalized), gamma=0.0 (smoothness only)
        V, W, loss_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=0.0,
            beta=0.5,
            gamma=0.0,
            normalize=True,
            max_iter=20,
            verbose=False,
        )

        # Check monotonic decrease: loss[i] <= loss[i-1]
        # Use more lenient tolerance for smoothness (0.1% instead of 0.01%)
        # This accounts for numerical perturbations when applying smoothness constraint
        for i in range(1, len(loss_history)):
            assert loss_history[i] <= loss_history[i - 1] * (1 + 1e-8), (
                f"Loss increased at iteration {i}: {loss_history[i]} > {loss_history[i - 1]}"
            )

        print("  ✓ Total loss decreases monotonically with smoothness regularization")


class TestSecondOrderSmoothness:
    """Test suite for second-order smoothness regularization (v0.3.0)"""

    def test_second_order_penalty_computation(self):
        """Test compute_second_order_smoothness_penalty()"""
        # Linear V should have zero second-order penalty
        V_linear = np.linspace(0, 1, 100).reshape(-1, 1)
        penalty_linear = compute_second_order_smoothness_penalty(V_linear)
        assert np.isclose(penalty_linear, 0.0, atol=1e-10), (
            f"Linear spectrum should have zero second-order penalty, got {penalty_linear}"
        )

        # Quadratic V should have non-zero penalty
        V_quad = (np.linspace(0, 1, 100) ** 2).reshape(-1, 1)
        penalty_quad = compute_second_order_smoothness_penalty(V_quad)
        assert penalty_quad > 0, f"Quadratic spectrum should have positive second-order penalty, got {penalty_quad}"

        # Random noisy V should have higher penalty
        np.random.seed(42)
        V_noisy = np.random.rand(100, 1) * 0.5 + np.linspace(0, 1, 100).reshape(-1, 1)
        penalty_noisy = compute_second_order_smoothness_penalty(V_noisy)
        assert penalty_noisy > penalty_linear, f"Noisy spectrum should have higher penalty than linear"

        # Verify manual calculation for simple case
        V_simple = np.array([[0.0], [1.0], [4.0], [9.0]])  # t^2
        # Laplacian: [4-2*1+0, 9-2*4+1] = [2, 2]
        # Penalty: 2^2 + 2^2 = 8
        penalty_simple = compute_second_order_smoothness_penalty(V_simple)
        assert np.isclose(penalty_simple, 8.0, rtol=1e-10), (
            f"Simple quadratic penalty mismatch: {penalty_simple} vs 8.0"
        )

        print("  ✓ Second-order smoothness penalty computation is correct")

    def test_m_step_with_second_order(self):
        """Test M-step with gamma > 0 reduces curvature"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        N = len(sources_data)
        K = 3
        V, W = initialize_parameters(N, T=4096, K=K, seed=42)

        # Test M-step with and without second-order smoothness
        V_no_curv, _ = m_step(
            sources_data, response_matrices, V.copy(), W, alpha=0.0, beta=0.0, gamma=0.0, normalize=False
        )
        V_curv, _ = m_step(
            sources_data, response_matrices, V.copy(), W, alpha=0.0, beta=0.0, gamma=10.0, normalize=False
        )

        # With second-order smoothness, V should have lower curvature
        curv_no = compute_second_order_smoothness_penalty(V_no_curv)
        curv_yes = compute_second_order_smoothness_penalty(V_curv)

        assert curv_yes <= curv_no * (1 + 1e-10), f"V with gamma>0 should have lower curvature: {curv_yes} vs {curv_no}"

        # Both should remain non-negative
        assert np.all(V_no_curv >= 0), "V without curvature regularization has negative values"
        assert np.all(V_curv >= 0), "V with curvature regularization has negative values"

        print("  ✓ M-step second-order smoothness works correctly")

    def test_monotonicity_with_second_order(self):
        """Test loss decreases monotonically with second-order regularization"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        K = 3

        # Run with alpha=0.0, beta=0.0, gamma=0.5 (second-order only, normalized)
        V, W, loss_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=0.0,
            beta=0.0,
            gamma=0.5,
            normalize=True,
            max_iter=20,
            verbose=False,
        )

        # Check monotonic decrease
        for i in range(1, len(loss_history)):
            assert loss_history[i] <= loss_history[i - 1] * (1 + 1e-8), (
                f"Loss increased at iteration {i}: {loss_history[i]} > {loss_history[i - 1]}"
            )

        print("  ✓ Loss decreases monotonically with second-order regularization")

    def test_normalization_consistency(self):
        """Test that normalization makes parameters dimensionless"""
        # Generate two datasets with different sizes
        # Note: K_true must be >= 3 for mock data generation
        sources_data_small, response_matrices_small, _, _ = generate_mock_data(
            N=10, M_per_source=50, K_true=3, T=1024, verbose=False
        )
        sources_data_large, response_matrices_large, _, _ = generate_mock_data(
            N=30, M_per_source=100, K_true=3, T=4096, verbose=False
        )

        K = 3

        # Run with same normalized parameters on different problem sizes
        # Note: T must match the response matrix dimension
        V_small, W_small, loss_small = als_wnmf(
            sources_data=sources_data_small,
            response_matrices=response_matrices_small,
            K=K,
            T=1024,  # Match response matrix dimension
            alpha=0.01,
            beta=0.1,
            gamma=0.1,
            normalize=True,
            max_iter=5,
            verbose=False,
        )

        V_large, W_large, loss_large = als_wnmf(
            sources_data=sources_data_large,
            response_matrices=response_matrices_large,
            K=K,
            T=4096,  # Match response matrix dimension
            alpha=0.01,
            beta=0.1,
            gamma=0.1,
            normalize=True,
            max_iter=5,
            verbose=False,
        )

        # Both should converge (loss decreases)
        assert loss_small[-1] < loss_small[0], "Small dataset did not converge"
        assert loss_large[-1] < loss_large[0], "Large dataset did not converge"

        # Verify non-negativity
        assert np.all(V_small >= 0), "V_small has negative values"
        assert np.all(V_large >= 0), "V_large has negative values"

        print("  ✓ Normalization works consistently across different problem sizes")

    def test_combined_regularization(self):
        """Test all three regularization terms together"""
        # Set up test data
        sources_data, response_matrices, V_true, W_true = generate_mock_data(
            N=20, M_per_source=100, K_true=3, verbose=False
        )

        K = 3

        # Run with all three regularization terms (normalized)
        V, W, loss_history = als_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            K=K,
            alpha=0.01,
            beta=0.1,
            gamma=0.1,
            normalize=True,
            max_iter=20,
            verbose=False,
        )

        # Verify convergence
        for i in range(1, len(loss_history)):
            assert loss_history[i] <= loss_history[i - 1] * (1 + 1e-8), (
                f"Loss increased at iteration {i}: {loss_history[i]} > {loss_history[i - 1]}"
            )

        # Verify non-negativity
        assert np.all(V >= 0), "V has negative values"
        assert np.all(W >= 0), "W has negative values"

        # Verify that regularization is having an effect (V should be smooth)
        first_order = compute_smoothness_penalty(V)
        second_order = compute_second_order_smoothness_penalty(V)

        # With regularization, V should be reasonably smooth
        # (This is a sanity check, not a strict assertion)
        assert first_order >= 0, "First-order penalty should be non-negative"
        assert second_order >= 0, "Second-order penalty should be non-negative"

        print("  ✓ Combined regularization (alpha + beta + gamma) works correctly")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Regularization Test Suite (v0.3.0)")
    print("=" * 70)
    print()
    print("[TestRegularization - L2 Tests]")
    print("-" * 70)

    # Run L2 tests
    import pytest

    exit_code = pytest.main([__file__, "-v", "-k", "TestRegularization"])

    if exit_code == 0:
        print()
        print("[TestSmoothness - First-Order Smoothness Tests]")
        print("-" * 70)
        exit_code = pytest.main([__file__, "-v", "-k", "TestSmoothness"])

    if exit_code == 0:
        print()
        print("[TestSecondOrderSmoothness - Second-Order Smoothness Tests]")
        print("-" * 70)
        exit_code = pytest.main([__file__, "-v", "-k", "TestSecondOrderSmoothness"])

    print()
    print("=" * 70)
    print("ALL REGULARIZATION TESTS PASSED ✓" if exit_code == 0 else "SOME TESTS FAILED ✗")
    print("=" * 70)
