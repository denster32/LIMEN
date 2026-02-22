"""
Recurrent neural network simulation using JAX.

Implements a continuous-time recurrent neural network (CTRNN) — the simplest
system that can exhibit the kind of recurrent dynamics we're looking for.

The network dynamics follow:
    τ * dx/dt = -x + W @ tanh(x) + noise

Discretized with Euler integration:
    x(t+1) = (1 - dt/τ) * x(t) + (dt/τ) * W @ tanh(x(t)) + σ * η(t)

where:
    - x is the state vector (n_nodes,)
    - W is the weight matrix (n_nodes, n_nodes)
    - τ is the time constant
    - σ is noise amplitude
    - η is Gaussian white noise
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Try JAX, fall back to NumPy for portability
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np


@dataclass
class NetworkConfig:
    """Configuration for a recurrent network."""

    n_nodes: int = 32
    connection_density: float = 0.5
    weight_scale: float = 1.0
    tau: float = 1.0
    dt: float = 0.1
    noise_amplitude: float = 0.01
    spectral_radius_target: Optional[float] = None
    seed: int = 42


class RecurrentNetwork:
    """
    Continuous-time recurrent neural network for phase transition experiments.

    The key parameter is connection_density: as it increases from 0 to 1,
    the network transitions from disconnected nodes (no integration) through
    a potentially critical regime to a fully connected system.

    We're looking for the density where information-theoretic measures
    show a sharp transition — the limen.
    """

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.n = config.n_nodes

        # Generate weight matrix with specified connection density
        self.W = self._generate_weights()

        # Initial state
        self.state = self.rng.normal(0, 0.1, self.n)

        # History buffer
        self.state_history: list[np.ndarray] = []
        self.weight_history: list[np.ndarray] = []

    def _generate_weights(self) -> np.ndarray:
        """
        Generate a random weight matrix with specified connection density
        and optional spectral radius normalization.

        Connection density controls the fraction of non-zero weights.
        This is the parameter we sweep in the phase transition experiment.
        """
        n = self.n
        density = self.config.connection_density

        # Random Gaussian weights
        W = self.rng.normal(0, self.config.weight_scale / np.sqrt(n), (n, n))

        # Apply connection mask (Erdos-Renyi random graph)
        mask = self.rng.random((n, n)) < density
        np.fill_diagonal(mask, False)  # No self-connections
        W *= mask

        # Optionally normalize to target spectral radius
        if self.config.spectral_radius_target is not None:
            eigenvalues = np.linalg.eigvals(W)
            current_radius = np.max(np.abs(eigenvalues))
            if current_radius > 0:
                W *= self.config.spectral_radius_target / current_radius

        return W

    def step(self) -> np.ndarray:
        """
        Advance the network by one timestep using Euler integration.

        Returns the new state vector.
        """
        dt_tau = self.config.dt / self.config.tau

        # CTRNN dynamics
        activation = np.tanh(self.state)
        recurrent_input = self.W @ activation
        noise = self.config.noise_amplitude * self.rng.normal(0, 1, self.n)

        self.state = (
            (1 - dt_tau) * self.state
            + dt_tau * recurrent_input
            + np.sqrt(self.config.dt) * noise
        )

        return self.state.copy()

    def run(self, n_steps: int, warmup: int = 200, record: bool = True) -> np.ndarray:
        """
        Run the network for n_steps timesteps.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to simulate after warmup.
        warmup : int
            Number of initial steps to discard (let transients die out).
        record : bool
            Whether to store state history.

        Returns
        -------
        np.ndarray
            State history, shape (n_steps, n_nodes).
        """
        # Reset
        self.state = self.rng.normal(0, 0.1, self.n)
        self.state_history = []

        # Warmup: let transients settle
        for _ in range(warmup):
            self.step()

        # Record
        states = np.zeros((n_steps, self.n))
        for t in range(n_steps):
            states[t] = self.step()

        if record:
            self.state_history = [states[t] for t in range(n_steps)]

        return states

    def get_effective_tpm(self, states: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate the effective transition probability matrix from state history
        using linear regression: x(t+1) ≈ A @ x(t).

        This linearization is used for the Gaussian Φ approximation.

        Parameters
        ----------
        states : np.ndarray, optional
            State history. Uses stored history if not provided.

        Returns
        -------
        np.ndarray
            Estimated TPM, shape (n_nodes, n_nodes).
        """
        if states is None:
            if not self.state_history:
                raise ValueError("No state history. Run the network first.")
            states = np.array(self.state_history)

        # x(t+1) = A @ x(t) + noise
        # Solve via least squares: A = X_future @ X_past^+
        X_past = states[:-1].T  # (n, T-1)
        X_future = states[1:].T  # (n, T-1)

        # Regularized least squares
        reg = 1e-6 * np.eye(self.n)
        A = X_future @ X_past.T @ np.linalg.inv(X_past @ X_past.T + reg)

        return A

    def get_network_stats(self) -> dict:
        """
        Compute basic network statistics.

        Returns
        -------
        dict
            Network properties including spectral radius, mean degree,
            clustering coefficient estimate, etc.
        """
        eigenvalues = np.linalg.eigvals(self.W)

        # Connection statistics
        adjacency = (np.abs(self.W) > 1e-10).astype(float)
        np.fill_diagonal(adjacency, 0)
        degrees = adjacency.sum(axis=1)

        return {
            "n_nodes": self.n,
            "connection_density": self.config.connection_density,
            "actual_density": float(adjacency.sum() / (self.n * (self.n - 1))),
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
            "mean_degree": float(degrees.mean()),
            "max_degree": int(degrees.max()),
            "weight_std": float(np.std(self.W[adjacency > 0])) if adjacency.sum() > 0 else 0.0,
            "mean_weight": float(np.mean(np.abs(self.W[adjacency > 0]))) if adjacency.sum() > 0 else 0.0,
        }


class RecurrentNetworkJAX:
    """
    JAX-accelerated version of the recurrent network for fast parameter sweeps.

    Uses JIT compilation for ~10-50x speedup over NumPy on CPU,
    and can leverage GPU acceleration for large networks.
    """

    def __init__(self, config: NetworkConfig):
        if not HAS_JAX:
            raise ImportError("JAX is required for RecurrentNetworkJAX. Install with: pip install jax jaxlib")

        self.config = config
        self.n = config.n_nodes

        # Generate weights using NumPy (deterministic), then convert to JAX
        rng = np.random.default_rng(config.seed)
        W = rng.normal(0, config.weight_scale / np.sqrt(self.n), (self.n, self.n))
        mask = rng.random((self.n, self.n)) < config.connection_density
        np.fill_diagonal(mask, False)
        W *= mask

        if config.spectral_radius_target is not None:
            eigvals = np.linalg.eigvals(W)
            sr = np.max(np.abs(eigvals))
            if sr > 0:
                W *= config.spectral_radius_target / sr

        self.W = jnp.array(W)
        self.dt_tau = config.dt / config.tau
        self.noise_amp = config.noise_amplitude
        self.sqrt_dt = np.sqrt(config.dt)

    def run(self, n_steps: int, warmup: int = 200, key: Optional[object] = None) -> np.ndarray:
        """
        Run the network using JAX scan for efficiency.

        Returns NumPy array for compatibility with analysis functions.
        """
        if key is None:
            key = jax.random.PRNGKey(self.config.seed)

        W = self.W
        dt_tau = self.dt_tau
        noise_amp = self.noise_amp
        sqrt_dt = self.sqrt_dt

        @jit
        def step_fn(carry, key_t):
            state = carry
            noise = noise_amp * jax.random.normal(key_t, shape=(self.n,))
            activation = jnp.tanh(state)
            new_state = (1 - dt_tau) * state + dt_tau * (W @ activation) + sqrt_dt * noise
            return new_state, new_state

        # Initial state
        key, subkey = jax.random.split(key)
        state_init = 0.1 * jax.random.normal(subkey, shape=(self.n,))

        # Generate all random keys
        total_steps = warmup + n_steps
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, total_steps)

        # Run with scan
        _, all_states = jax.lax.scan(step_fn, state_init, keys)

        # Discard warmup
        states = all_states[warmup:]

        return np.asarray(states)
