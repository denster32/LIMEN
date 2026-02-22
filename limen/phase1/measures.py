"""
Unified measurement pipeline for Phase 1 experiments.

Computes all three consciousness candidate measures on a network's
state history and returns them in a standardized format for comparison.

The central question: do Φ, LZ complexity, and self-model fidelity
show a convergent discontinuity at the same critical parameter?
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

from limen.core.phi import approximate_phi, geometric_integrated_information
from limen.core.complexity import (
    lz_complexity_from_states,
    multiscale_lz_complexity,
    normalized_lz_complexity,
)
from limen.core.information import (
    self_model_fidelity,
    self_model_fidelity_multiscale,
    mutual_information,
    transfer_entropy,
)


@dataclass
class MeasurementResult:
    """Results from computing all consciousness candidate measures."""

    # Primary measures (the three legs of the tripod)
    phi: float
    lz_complexity: float
    lz_normalized: float
    self_model_fidelity: float
    self_model_fidelity_normalized: float

    # Secondary measures
    total_mutual_info: float
    spectral_radius: float
    mean_transfer_entropy: float

    # Multiscale analysis
    lz_multiscale: dict
    smf_multiscale: dict

    # Network properties
    n_nodes: int
    connection_density: float
    n_timesteps: int

    # Diagnostics
    computation_time_seconds: float

    def to_dict(self) -> dict:
        """Convert to flat dictionary for easy DataFrame construction."""
        return {
            "phi": self.phi,
            "lz_complexity": self.lz_complexity,
            "lz_normalized": self.lz_normalized,
            "self_model_fidelity": self.self_model_fidelity,
            "self_model_fidelity_normalized": self.self_model_fidelity_normalized,
            "total_mutual_info": self.total_mutual_info,
            "spectral_radius": self.spectral_radius,
            "mean_transfer_entropy": self.mean_transfer_entropy,
            "n_nodes": self.n_nodes,
            "connection_density": self.connection_density,
            "n_timesteps": self.n_timesteps,
            "computation_time_seconds": self.computation_time_seconds,
        }


def compute_all_measures(
    states: np.ndarray,
    tpm: np.ndarray,
    connection_density: float,
    n_nodes: Optional[int] = None,
    verbose: bool = False,
) -> MeasurementResult:
    """
    Compute all three primary measures plus diagnostics on a state history.

    This is the main measurement function called during parameter sweeps.
    It computes:
    1. Approximate Φ (integrated information) from the linearized TPM
    2. LZ complexity from the binarized state history
    3. Self-model fidelity (mutual info between present and past)

    Plus secondary measures for additional analysis.

    Parameters
    ----------
    states : np.ndarray
        Network state history, shape (timesteps, n_nodes).
    tpm : np.ndarray
        Estimated transition probability matrix, shape (n_nodes, n_nodes).
    connection_density : float
        The connection density parameter for this run.
    n_nodes : int, optional
        Number of nodes. Inferred from states if not provided.
    verbose : bool
        Print progress.

    Returns
    -------
    MeasurementResult
        All computed measures.
    """
    t_start = time.time()

    T, n = states.shape
    if n_nodes is None:
        n_nodes = n

    # =====================================================================
    # 1. INTEGRATED INFORMATION (Φ)
    # =====================================================================
    if verbose:
        print(f"  Computing Φ for {n_nodes}-node network...")

    phi_result = geometric_integrated_information(tpm)
    phi = phi_result["phi"]
    total_mi = phi_result["total_mutual_info"]
    spectral_radius = phi_result["spectral_radius"]

    if verbose:
        print(f"  Φ = {phi:.4f}, spectral radius = {spectral_radius:.3f}")

    # =====================================================================
    # 2. LEMPEL-ZIV COMPLEXITY
    # =====================================================================
    if verbose:
        print(f"  Computing LZ complexity...")

    lz_result = lz_complexity_from_states(states, method="concatenate")
    lz_raw = lz_result["lz_raw"]
    lz_norm = lz_result["lz_normalized"]

    # Multiscale LZ
    lz_multi = multiscale_lz_complexity(states)

    if verbose:
        print(f"  LZ = {lz_raw} (normalized: {lz_norm:.4f})")

    # =====================================================================
    # 3. SELF-MODEL FIDELITY
    # =====================================================================
    if verbose:
        print(f"  Computing self-model fidelity...")

    # Reduce dimensionality for SMF computation — kNN mutual information
    # estimation requires n_samples >> n_dims. Project onto top PCA
    # components to keep the computation tractable.
    max_smf_dims = min(n, 6)  # Cap at 6 dimensions for reliable kNN
    if n > max_smf_dims:
        centered = states - states.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        states_reduced = centered @ vt[:max_smf_dims].T
    else:
        states_reduced = states

    smf_result = self_model_fidelity(states_reduced, lag=1, embedding_dim=2)
    smf = smf_result["smf"]
    smf_norm = smf_result["smf_normalized"]

    # Multiscale SMF
    smf_multi = self_model_fidelity_multiscale(states_reduced, embedding_dim=2)

    if verbose:
        print(f"  SMF = {smf:.4f} (normalized: {smf_norm:.4f})")

    # =====================================================================
    # 4. SECONDARY MEASURES
    # =====================================================================

    # Mean pairwise transfer entropy (sample a subset of pairs for efficiency)
    if n_nodes <= 16:
        te_sum = 0.0
        te_count = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    te = transfer_entropy(states[:, i], states[:, j], lag=1)
                    te_sum += te
                    te_count += 1
        mean_te = te_sum / te_count if te_count > 0 else 0.0
    else:
        # Sample random pairs for larger networks
        rng = np.random.default_rng(42)
        n_pairs = min(100, n_nodes * (n_nodes - 1))
        pairs = set()
        while len(pairs) < n_pairs:
            i, j = rng.integers(0, n_nodes, 2)
            if i != j:
                pairs.add((i, j))
        te_sum = sum(
            transfer_entropy(states[:, i], states[:, j], lag=1) for i, j in pairs
        )
        mean_te = te_sum / len(pairs)

    if verbose:
        print(f"  Mean TE = {mean_te:.4f}")

    t_end = time.time()

    return MeasurementResult(
        phi=phi,
        lz_complexity=lz_raw,
        lz_normalized=lz_norm,
        self_model_fidelity=smf,
        self_model_fidelity_normalized=smf_norm,
        total_mutual_info=total_mi,
        spectral_radius=spectral_radius,
        mean_transfer_entropy=mean_te,
        lz_multiscale=lz_multi,
        smf_multiscale=smf_multi,
        n_nodes=n_nodes,
        connection_density=connection_density,
        n_timesteps=T,
        computation_time_seconds=t_end - t_start,
    )


def compute_phase_derivatives(
    densities: np.ndarray,
    phi_values: np.ndarray,
    lz_values: np.ndarray,
    smf_values: np.ndarray,
) -> dict:
    """
    Compute numerical derivatives and detect candidate phase transitions.

    A phase transition manifests as a peak in the derivative (susceptibility)
    of the order parameter with respect to the control parameter (density).

    In physics, this is analogous to the divergence of susceptibility
    at a second-order phase transition.

    Parameters
    ----------
    densities : np.ndarray
        Connection density values.
    phi_values, lz_values, smf_values : np.ndarray
        Measure values at each density.

    Returns
    -------
    dict
        Derivative analysis including:
        - Numerical derivatives of each measure
        - Candidate transition points (peaks of derivatives)
        - Convergence score (how close the three peaks are)
    """
    # Smooth derivatives using Savitzky-Golay-like finite differences
    def smooth_derivative(x, y, window=3):
        """Central differences with optional smoothing."""
        dy = np.gradient(y, x)
        # Simple moving average smoothing
        if len(dy) > window:
            kernel = np.ones(window) / window
            dy_smooth = np.convolve(dy, kernel, mode="same")
            return dy_smooth
        return dy

    d_phi = smooth_derivative(densities, phi_values)
    d_lz = smooth_derivative(densities, lz_values)
    d_smf = smooth_derivative(densities, smf_values)

    # Find peaks (candidate transition points)
    def find_peak(derivative):
        """Find the density at maximum derivative magnitude."""
        idx = np.argmax(np.abs(derivative))
        return densities[idx], float(derivative[idx])

    peak_phi = find_peak(d_phi)
    peak_lz = find_peak(d_lz)
    peak_smf = find_peak(d_smf)

    # Convergence score: how close are the three peaks?
    peak_densities = np.array([peak_phi[0], peak_lz[0], peak_smf[0]])
    convergence_std = float(np.std(peak_densities))
    convergence_mean = float(np.mean(peak_densities))

    # Score from 0 to 1: 1 means perfect convergence
    # Using exponential decay with characteristic scale of 0.05 density units
    convergence_score = float(np.exp(-convergence_std / 0.05))

    return {
        "derivatives": {
            "phi": d_phi,
            "lz": d_lz,
            "smf": d_smf,
        },
        "peaks": {
            "phi": {"density": peak_phi[0], "magnitude": peak_phi[1]},
            "lz": {"density": peak_lz[0], "magnitude": peak_lz[1]},
            "smf": {"density": peak_smf[0], "magnitude": peak_smf[1]},
        },
        "convergence": {
            "mean_density": convergence_mean,
            "std_density": convergence_std,
            "score": convergence_score,
        },
    }
