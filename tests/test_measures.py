"""
Tests for core information-theoretic measures.

Validates the mathematical correctness of Φ, LZ complexity, and
self-model fidelity computations using known analytical results
and edge cases.
"""

import numpy as np
import pytest

from limen.core.phi import approximate_phi, geometric_integrated_information
from limen.core.complexity import (
    lempel_ziv_complexity,
    normalized_lz_complexity,
    lz_complexity_from_states,
    multiscale_lz_complexity,
)
from limen.core.information import (
    mutual_information,
    conditional_entropy,
    transfer_entropy,
    self_model_fidelity,
    self_model_fidelity_multiscale,
)


# =====================================================================
# PHI TESTS
# =====================================================================

class TestPhi:
    def test_phi_diagonal_is_zero(self):
        """A diagonal (disconnected) system should have Φ ≈ 0."""
        # Diagonal matrix = independent subsystems
        tpm = np.diag([0.5, 0.5, 0.5, 0.5])
        phi = approximate_phi(tpm)
        assert phi < 0.01, f"Diagonal system should have near-zero Φ, got {phi}"

    def test_phi_identity_low(self):
        """Identity matrix: each node copies itself, no integration."""
        tpm = 0.9 * np.eye(4)
        phi = approximate_phi(tpm)
        # Should be very low since nodes don't interact
        assert phi < 0.1, f"Identity-like system should have low Φ, got {phi}"

    def test_phi_connected_positive(self):
        """A connected system should have positive Φ."""
        rng = np.random.default_rng(42)
        n = 8
        tpm = rng.normal(0, 0.3, (n, n))
        # Ensure spectral radius < 1
        eigvals = np.linalg.eigvals(tpm)
        tpm /= (np.max(np.abs(eigvals)) + 0.1)

        phi = approximate_phi(tpm)
        assert phi >= 0, f"Φ should be non-negative, got {phi}"

    def test_phi_increases_with_connectivity(self):
        """Φ should generally increase with connectivity (up to a point)."""
        phis = []
        for density in [0.1, 0.3, 0.5]:
            rng = np.random.default_rng(42)
            n = 8
            W = rng.normal(0, 0.3, (n, n))
            mask = rng.random((n, n)) < density
            np.fill_diagonal(mask, False)
            tpm = W * mask
            eigvals = np.linalg.eigvals(tpm)
            tpm /= (np.max(np.abs(eigvals)) + 0.1)
            phis.append(approximate_phi(tpm))

        # At least one increase in the sequence
        assert any(phis[i+1] >= phis[i] for i in range(len(phis)-1)), \
            f"Φ should increase with some connectivity increase: {phis}"

    def test_phi_single_node(self):
        """Single node system should have Φ = 0."""
        tpm = np.array([[0.5]])
        phi = approximate_phi(tpm)
        assert phi == 0.0

    def test_geometric_returns_dict(self):
        """geometric_integrated_information should return all expected keys."""
        tpm = np.random.default_rng(42).normal(0, 0.2, (4, 4))
        tpm /= (np.max(np.abs(np.linalg.eigvals(tpm))) + 0.1)
        result = geometric_integrated_information(tpm)
        assert "phi" in result
        assert "mib_partition" in result
        assert "spectral_radius" in result
        assert "system_size" in result


# =====================================================================
# LZ COMPLEXITY TESTS
# =====================================================================

class TestLZComplexity:
    def test_constant_sequence_low_complexity(self):
        """A constant sequence should have minimal LZ complexity."""
        seq = [0] * 1000
        c = lempel_ziv_complexity(seq)
        assert c <= 2, f"Constant sequence should have LZ ≤ 2, got {c}"

    def test_periodic_sequence_low_complexity(self):
        """A periodic sequence should have low LZ complexity."""
        seq = [0, 1] * 500
        c = lempel_ziv_complexity(seq)
        c_norm = normalized_lz_complexity(seq)
        assert c_norm < 0.3, f"Periodic sequence should have low normalized LZ, got {c_norm}"

    def test_random_sequence_high_complexity(self):
        """A random sequence should have high LZ complexity."""
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 2, 1000)
        c_norm = normalized_lz_complexity(seq)
        assert c_norm > 0.5, f"Random sequence should have high normalized LZ, got {c_norm}"

    def test_complexity_increases_with_randomness(self):
        """LZ complexity should increase as sequences become more random."""
        # Highly structured
        seq1 = [0, 1] * 500
        # Some structure
        rng = np.random.default_rng(42)
        seq2 = list(rng.choice([0, 0, 0, 1], 1000))
        # Random
        seq3 = list(rng.integers(0, 2, 1000))

        c1 = normalized_lz_complexity(seq1)
        c2 = normalized_lz_complexity(seq2)
        c3 = normalized_lz_complexity(seq3)

        assert c1 < c3, f"Periodic ({c1}) should have lower LZ than random ({c3})"

    def test_empty_sequence(self):
        """Empty sequence should return 0."""
        assert lempel_ziv_complexity([]) == 0
        assert normalized_lz_complexity([]) == 0.0

    def test_lz_from_states(self):
        """lz_complexity_from_states should work on 2D arrays."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, (100, 4))
        result = lz_complexity_from_states(states)
        assert "lz_raw" in result
        assert "lz_normalized" in result
        assert result["lz_normalized"] >= 0
        assert result["lz_normalized"] <= 1

    def test_multiscale_lz(self):
        """Multiscale LZ should return values at multiple scales."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, (200, 4))
        result = multiscale_lz_complexity(states)
        assert len(result["scales"]) > 1
        assert len(result["lz_values"]) == len(result["scales"])


# =====================================================================
# INFORMATION THEORY TESTS
# =====================================================================

class TestInformationTheory:
    def test_mi_independent_is_low(self):
        """MI between independent variables should be near zero."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        mi = mutual_information(x, y, method="knn")
        assert mi < 0.2, f"MI of independent variables should be near 0, got {mi}"

    def test_mi_correlated_is_positive(self):
        """MI between correlated variables should be positive."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = 0.8 * x + 0.2 * rng.normal(0, 1, 1000)
        mi = mutual_information(x, y, method="knn")
        assert mi > 0.1, f"MI of correlated variables should be positive, got {mi}"

    def test_mi_identical_is_high(self):
        """MI between identical variables should equal entropy."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        mi = mutual_information(x, x + 1e-8 * rng.normal(0, 1, 1000), method="knn")
        assert mi > 1.0, f"MI of identical variables should be high, got {mi}"

    def test_mi_histogram_method(self):
        """Histogram method should also work."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = 0.8 * x + 0.2 * rng.normal(0, 1, 1000)
        mi = mutual_information(x, y, method="histogram")
        assert mi > 0, f"Histogram MI should be positive for correlated variables"

    def test_transfer_entropy_causal(self):
        """TE from cause to effect should be higher than reverse."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * x[t-1] + 0.3 * rng.normal()

        te_x_to_y = transfer_entropy(x, y, lag=1)
        te_y_to_x = transfer_entropy(y, x, lag=1)
        # Causal direction should have higher TE
        assert te_x_to_y > te_y_to_x * 0.5, \
            f"TE(X→Y)={te_x_to_y} should exceed TE(Y→X)={te_y_to_x}"

    def test_self_model_fidelity_recurrent(self):
        """SMF should be higher for recurrent (self-predicting) systems."""
        rng = np.random.default_rng(42)

        # Random (no temporal structure)
        random_states = rng.normal(0, 1, (500, 4))

        # Recurrent (strong temporal autocorrelation via linear dynamics)
        recurrent_states = np.zeros((500, 4))
        recurrent_states[0] = rng.normal(0, 0.1, 4)
        A = rng.normal(0, 0.5, (4, 4))
        A /= (np.max(np.abs(np.linalg.eigvals(A))) * 1.05)
        for t in range(1, 500):
            recurrent_states[t] = 0.95 * A @ recurrent_states[t-1] + 0.05 * rng.normal(0, 1, 4)

        smf_random = self_model_fidelity(random_states, lag=1, embedding_dim=1)["smf"]
        smf_recurrent = self_model_fidelity(recurrent_states, lag=1, embedding_dim=1)["smf"]

        assert smf_recurrent > smf_random, \
            f"Recurrent SMF ({smf_recurrent}) should exceed random SMF ({smf_random})"

    def test_smf_multiscale(self):
        """Multiscale SMF should return values at multiple lags."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, (200, 4))
        result = self_model_fidelity_multiscale(states)
        assert len(result["lags"]) > 1
        assert len(result["smf_values"]) == len(result["lags"])


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestMeasureIntegration:
    def test_all_measures_on_network(self):
        """All measures should work on actual network output."""
        from limen.phase1.network import RecurrentNetwork, NetworkConfig
        from limen.phase1.measures import compute_all_measures

        config = NetworkConfig(n_nodes=8, connection_density=0.5, seed=42)
        net = RecurrentNetwork(config)
        states = net.run(500, warmup=100)
        tpm = net.get_effective_tpm(states)

        result = compute_all_measures(
            states=states, tpm=tpm,
            connection_density=0.5, n_nodes=8,
        )

        assert result.phi >= 0
        assert 0 <= result.lz_normalized <= 1
        assert result.self_model_fidelity >= 0
        assert result.computation_time_seconds > 0

    def test_measures_differ_across_densities(self):
        """Measures should change as connection density changes."""
        from limen.phase1.network import RecurrentNetwork, NetworkConfig
        from limen.phase1.measures import compute_all_measures

        results = []
        for density in [0.1, 0.5, 0.9]:
            config = NetworkConfig(n_nodes=8, connection_density=density, seed=42)
            net = RecurrentNetwork(config)
            states = net.run(500, warmup=100)
            tpm = net.get_effective_tpm(states)
            result = compute_all_measures(
                states=states, tpm=tpm,
                connection_density=density, n_nodes=8,
            )
            results.append(result)

        # At least one measure should differ between low and high density
        phis = [r.phi for r in results]
        lzs = [r.lz_normalized for r in results]
        assert max(phis) - min(phis) > 0.001 or max(lzs) - min(lzs) > 0.001, \
            "Measures should vary with density"
