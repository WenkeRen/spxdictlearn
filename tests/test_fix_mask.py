"""
Tests for HALS fix_mask feature (AGN incremental training).

Validates that:
1. fix_mask=None produces identical results to original behavior
2. fix_mask correctly freezes specified components
3. E-step weights are updated for all components (fixed + free)
4. Loss is non-increasing with fix_mask
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from spxdictlearn.hals import hals_wnmf, m_step_hals, precompute_hals_constants, compute_M_tensor
from spxdictlearn.mock_data import generate_mock_data


# Use small T to keep memory footprint low for CI/testing
_T_TEST = 256
_M_TEST = 100


def _make_small_dataset(N=15, K_true=3, seed=42):
    """Generate small test dataset for fix_mask tests."""
    sources_data, response_matrices, V_true, W_true = generate_mock_data(
        N=N,
        M_per_source=_M_TEST,
        K_true=K_true,
        T=_T_TEST,
        noise_level=0.05,
        seed=seed,
        verbose=False,
    )
    return sources_data, response_matrices


class TestMStepFixMask:
    """Test m_step_hals with fix_mask parameter."""

    def test_fix_mask_none_is_default(self):
        """fix_mask=None should produce same result as no fix_mask."""
        sources_data, response_matrices = _make_small_dataset()
        N = len(sources_data)
        T = _T_TEST
        K = 5

        rng = np.random.default_rng(42)
        V = np.abs(rng.standard_normal((T, K))) * 0.1
        W = np.abs(rng.standard_normal((N, K))) * 0.1

        C, global_B_data, global_indices, global_indptr = precompute_hals_constants(
            sources_data, response_matrices, N, T
        )
        M = compute_M_tensor(W, global_B_data, global_indices, global_indptr, K, T)

        # Run with fix_mask=None
        V_none, _ = m_step_hals(V, W, C, M, 0.001, 0.0, 0.0, True, 1.0, 5, fix_mask=None)

        # Run without fix_mask parameter (original behavior)
        V_default, _ = m_step_hals(V, W, C, M, 0.001, 0.0, 0.0, True, 1.0, 5)

        np.testing.assert_array_equal(V_none, V_default)

    def test_frozen_components_unchanged(self):
        """Components with fix_mask[k]=True must not change."""
        sources_data, response_matrices = _make_small_dataset()
        N = len(sources_data)
        T = _T_TEST
        K = 5

        rng = np.random.default_rng(42)
        V = np.abs(rng.standard_normal((T, K))) * 0.1
        W = np.abs(rng.standard_normal((N, K))) * 0.1

        # Freeze first 3 components
        fix_mask = np.array([True, True, True, False, False])

        C, global_B_data, global_indices, global_indptr = precompute_hals_constants(
            sources_data, response_matrices, N, T
        )
        M = compute_M_tensor(W, global_B_data, global_indices, global_indptr, K, T)

        V_new, _ = m_step_hals(
            V, W, C, M, 0.001, 0.0, 0.0, True, 1.0, 5, fix_mask=fix_mask
        )

        # Frozen columns must be identical
        np.testing.assert_array_equal(V_new[:, :3], V[:, :3])

        # Free columns should still be non-negative
        assert np.all(V_new[:, 3:] >= 0)

    def test_all_frozen_no_change(self):
        """If all components are frozen, V should not change."""
        sources_data, response_matrices = _make_small_dataset()
        N = len(sources_data)
        T = _T_TEST
        K = 5

        rng = np.random.default_rng(42)
        V = np.abs(rng.standard_normal((T, K))) * 0.1
        W = np.abs(rng.standard_normal((N, K))) * 0.1

        fix_mask = np.ones(K, dtype=bool)  # All frozen

        C, global_B_data, global_indices, global_indptr = precompute_hals_constants(
            sources_data, response_matrices, N, T
        )
        M = compute_M_tensor(W, global_B_data, global_indices, global_indptr, K, T)

        V_new, _ = m_step_hals(
            V, W, C, M, 0.001, 0.0, 0.0, True, 1.0, 5, fix_mask=fix_mask
        )

        np.testing.assert_array_equal(V_new, V)


class TestHalsWnmfFixMask:
    """Test full hals_wnmf with fix_mask parameter."""

    def test_fix_mask_convergence(self):
        """HALS with fix_mask should converge (loss non-increasing)."""
        sources_data, response_matrices = _make_small_dataset(N=15)
        N = len(sources_data)
        T = _T_TEST
        K_fix = 2
        K_free = 2
        K_total = K_fix + K_free

        rng = np.random.default_rng(42)
        V_init = np.abs(rng.standard_normal((T, K_total))) * 0.1
        W_init = np.abs(rng.standard_normal((N, K_total))) * 0.1

        fix_mask = np.zeros(K_total, dtype=bool)
        fix_mask[:K_fix] = True

        V, W, loss_history = hals_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            V_init=V_init,
            W_init=W_init,
            alpha=0.001,
            max_iter=10,
            tol=1e-10,
            verbose=False,
            fix_mask=fix_mask,
        )

        # Check convergence: loss should be non-increasing
        for i in range(1, len(loss_history)):
            assert loss_history[i] <= loss_history[i - 1] + 1e-10, (
                f"Loss increased at iteration {i}: {loss_history[i]} > {loss_history[i-1]}"
            )

        # Check frozen components unchanged
        np.testing.assert_array_equal(V[:, :K_fix], V_init[:, :K_fix])

        # Check non-negativity
        assert np.all(V >= 0)
        assert np.all(W >= 0)

    def test_w_updated_for_fixed_components(self):
        """E-step should update W for fixed components (NNLS uses full V)."""
        sources_data, response_matrices = _make_small_dataset(N=15, K_true=3)
        N = len(sources_data)
        T = _T_TEST
        K_total = 4

        rng = np.random.default_rng(42)
        V_init = np.abs(rng.standard_normal((T, K_total))) * 0.1
        W_init = np.zeros((N, K_total))  # Zero initialization

        fix_mask = np.array([True, True, False, False])

        V, W, loss_history = hals_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            V_init=V_init,
            W_init=W_init,
            alpha=0.001,
            max_iter=3,
            tol=1e-10,
            verbose=False,
            fix_mask=fix_mask,
        )

        # Fixed-component weights should be non-zero
        # (E-step solves NNLS with the full V matrix, including fixed columns)
        for k in range(2):  # Check fixed components
            assert np.sum(np.abs(W[:, k])) > 0, f"W[:, {k}] is all zeros"

    def test_backward_compatible_no_fix_mask(self):
        """Calling hals_wnmf without fix_mask should work as before."""
        sources_data, response_matrices = _make_small_dataset(N=15)
        N = len(sources_data)
        T = _T_TEST
        K = 4

        rng = np.random.default_rng(42)
        V_init = np.abs(rng.standard_normal((T, K))) * 0.1
        W_init = np.abs(rng.standard_normal((N, K))) * 0.1

        # Without fix_mask
        V1, W1, loss1 = hals_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            V_init=V_init.copy(),
            W_init=W_init.copy(),
            alpha=0.001,
            max_iter=5,
            tol=1e-10,
            verbose=False,
        )

        # With fix_mask=None
        V2, W2, loss2 = hals_wnmf(
            sources_data=sources_data,
            response_matrices=response_matrices,
            V_init=V_init.copy(),
            W_init=W_init.copy(),
            alpha=0.001,
            max_iter=5,
            tol=1e-10,
            verbose=False,
            fix_mask=None,
        )

        np.testing.assert_array_almost_equal(V1, V2)
        np.testing.assert_array_almost_equal(W1, W2)
        np.testing.assert_array_almost_equal(loss1, loss2)
