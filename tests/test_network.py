"""
Tests for the recurrent network simulation.
"""

import numpy as np
import pytest

from limen.phase1.network import RecurrentNetwork, NetworkConfig


class TestRecurrentNetwork:
    def test_basic_construction(self):
        """Network should construct with default config."""
        config = NetworkConfig(n_nodes=8, seed=42)
        net = RecurrentNetwork(config)
        assert net.n == 8
        assert net.W.shape == (8, 8)

    def test_no_self_connections(self):
        """Weight matrix should have zero diagonal (no self-connections)."""
        config = NetworkConfig(n_nodes=16, connection_density=1.0, seed=42)
        net = RecurrentNetwork(config)
        assert np.allclose(np.diag(net.W), 0), "Diagonal should be zero"

    def test_connection_density(self):
        """Actual density should approximate requested density."""
        for target_density in [0.1, 0.5, 0.9]:
            config = NetworkConfig(n_nodes=32, connection_density=target_density, seed=42)
            net = RecurrentNetwork(config)
            adjacency = (np.abs(net.W) > 1e-10).astype(float)
            np.fill_diagonal(adjacency, 0)
            actual_density = adjacency.sum() / (32 * 31)
            assert abs(actual_density - target_density) < 0.15, \
                f"Density mismatch: target={target_density}, actual={actual_density}"

    def test_run_returns_correct_shape(self):
        """run() should return array of correct shape."""
        config = NetworkConfig(n_nodes=8, seed=42)
        net = RecurrentNetwork(config)
        states = net.run(100, warmup=50)
        assert states.shape == (100, 8)

    def test_run_produces_finite_values(self):
        """States should not contain NaN or Inf."""
        config = NetworkConfig(n_nodes=16, connection_density=0.5, seed=42)
        net = RecurrentNetwork(config)
        states = net.run(500, warmup=100)
        assert np.all(np.isfinite(states)), "States contain non-finite values"

    def test_states_bounded(self):
        """States should remain bounded (not diverge)."""
        config = NetworkConfig(
            n_nodes=16, connection_density=0.5,
            noise_amplitude=0.01, seed=42,
        )
        net = RecurrentNetwork(config)
        states = net.run(1000, warmup=200)
        max_val = np.max(np.abs(states))
        assert max_val < 100, f"States diverged: max |x| = {max_val}"

    def test_tpm_estimation(self):
        """Estimated TPM should have reasonable spectral radius."""
        config = NetworkConfig(n_nodes=8, connection_density=0.5, seed=42)
        net = RecurrentNetwork(config)
        states = net.run(500, warmup=100)
        tpm = net.get_effective_tpm(states)

        assert tpm.shape == (8, 8)
        sr = np.max(np.abs(np.linalg.eigvals(tpm)))
        # Spectral radius should be < 2 (system is stable)
        assert sr < 2.0, f"TPM spectral radius too large: {sr}"

    def test_different_seeds_different_weights(self):
        """Different seeds should produce different networks."""
        config1 = NetworkConfig(n_nodes=8, seed=42)
        config2 = NetworkConfig(n_nodes=8, seed=123)
        net1 = RecurrentNetwork(config1)
        net2 = RecurrentNetwork(config2)
        assert not np.allclose(net1.W, net2.W), "Different seeds should give different weights"

    def test_spectral_radius_targeting(self):
        """Network with target spectral radius should hit it."""
        target_sr = 0.95
        config = NetworkConfig(
            n_nodes=16, connection_density=0.5,
            spectral_radius_target=target_sr, seed=42,
        )
        net = RecurrentNetwork(config)
        eigvals = np.linalg.eigvals(net.W)
        actual_sr = np.max(np.abs(eigvals))
        assert abs(actual_sr - target_sr) < 0.05, \
            f"Spectral radius: target={target_sr}, actual={actual_sr}"

    def test_network_stats(self):
        """get_network_stats should return expected keys."""
        config = NetworkConfig(n_nodes=16, connection_density=0.5, seed=42)
        net = RecurrentNetwork(config)
        stats = net.get_network_stats()

        assert "n_nodes" in stats
        assert "spectral_radius" in stats
        assert "mean_degree" in stats
        assert stats["n_nodes"] == 16

    def test_zero_density_disconnected(self):
        """Zero density should produce a disconnected network."""
        config = NetworkConfig(n_nodes=8, connection_density=0.0, seed=42)
        net = RecurrentNetwork(config)
        # Only noise on diagonal matters; off-diagonal should be zero
        off_diag = net.W.copy()
        np.fill_diagonal(off_diag, 0)
        assert np.allclose(off_diag, 0), "Zero density should have no connections"
