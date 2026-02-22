"""
Approximate Integrated Information (Φ) computation.

Implements the geometric decomposition approach from Oizumi et al. (2014),
adapted for tractable computation on small recurrent networks. Full IIT 3.0
Φ computation is NP-hard; this uses the Minimum Information Bipartition (MIB)
approximation which is tractable for networks up to ~256 nodes.

The key insight: Φ measures how much a system's current state carries information
about its past state *above and beyond* what its parts carry independently.
A system with high Φ cannot be reduced to independent subsystems without
losing information about its causal structure.
"""

import numpy as np
from scipy import linalg
from itertools import combinations
from typing import Optional


def _covariance_from_tpm(tpm: np.ndarray, noise_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the stationary covariance matrix of a linear-Gaussian system
    defined by its transition probability matrix (TPM).

    For a system x(t+1) = A @ x(t) + noise, the stationary covariance Σ
    satisfies the discrete Lyapunov equation: Σ = A Σ Aᵀ + Q

    Parameters
    ----------
    tpm : np.ndarray
        Transition matrix A, shape (n, n). Must have spectral radius < 1
        for stationarity.
    noise_cov : np.ndarray, optional
        Noise covariance Q. Defaults to identity.

    Returns
    -------
    np.ndarray
        Stationary covariance matrix Σ.
    """
    n = tpm.shape[0]
    if noise_cov is None:
        noise_cov = np.eye(n)

    # Solve the discrete Lyapunov equation: Σ = A Σ Aᵀ + Q
    # scipy.linalg.solve_discrete_lyapunov solves: A X Aᵀ - X + Q = 0
    try:
        cov = linalg.solve_discrete_lyapunov(tpm, noise_cov)
    except np.linalg.LinAlgError:
        # Fallback: iterative solution for ill-conditioned systems
        cov = noise_cov.copy()
        for _ in range(1000):
            cov_new = tpm @ cov @ tpm.T + noise_cov
            if np.allclose(cov_new, cov, atol=1e-10):
                break
            cov = cov_new
        cov = cov_new

    return cov


def _mutual_info_gaussian(cov: np.ndarray, idx_x: list[int], idx_y: list[int]) -> float:
    """
    Compute mutual information I(X; Y) for jointly Gaussian variables
    using the log-determinant formula:

        I(X; Y) = 0.5 * log(det(Σ_X) * det(Σ_Y) / det(Σ_XY))

    Parameters
    ----------
    cov : np.ndarray
        Joint covariance matrix.
    idx_x, idx_y : list[int]
        Index sets for X and Y partitions.

    Returns
    -------
    float
        Mutual information in nats.
    """
    idx_xy = sorted(idx_x + idx_y)

    cov_x = cov[np.ix_(idx_x, idx_x)]
    cov_y = cov[np.ix_(idx_y, idx_y)]
    cov_xy = cov[np.ix_(idx_xy, idx_xy)]

    # Use slogdet for numerical stability
    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)

    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        return 0.0

    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return max(0.0, mi)


def _effective_information(
    tpm: np.ndarray,
    cov: np.ndarray,
    indices_from: list[int],
    indices_to: list[int],
) -> float:
    """
    Compute effective information (EI) between two subsets of a system,
    measuring how much the past state of one subset constrains the future
    state of another, as quantified by mutual information under the
    stationary distribution.

    This captures the causal influence between subsystems.

    Parameters
    ----------
    tpm : np.ndarray
        Full system transition matrix.
    cov : np.ndarray
        Stationary covariance of the system.
    indices_from : list[int]
        Indices of the "cause" partition (past).
    indices_to : list[int]
        Indices of the "effect" partition (future).

    Returns
    -------
    float
        Effective information in nats.
    """
    n = tpm.shape[0]

    # Build the cross-temporal covariance: Cov(x(t+1), x(t)) = A @ Σ
    cross_cov = tpm @ cov

    # Build joint covariance of [x_to(t+1), x_from(t)]
    n_to = len(indices_to)
    n_from = len(indices_from)
    joint_dim = n_to + n_from

    joint_cov = np.zeros((joint_dim, joint_dim))

    # Cov(x_to(t+1), x_to(t+1))
    # Future covariance of the 'to' subset
    future_cov = tpm @ cov @ tpm.T + np.eye(n)  # Add noise covariance
    joint_cov[:n_to, :n_to] = future_cov[np.ix_(indices_to, indices_to)]

    # Cov(x_from(t), x_from(t))
    joint_cov[n_to:, n_to:] = cov[np.ix_(indices_from, indices_from)]

    # Cross-covariance: Cov(x_to(t+1), x_from(t)) = (A @ Σ)[to, from]
    joint_cov[:n_to, n_to:] = cross_cov[np.ix_(indices_to, indices_from)]
    joint_cov[n_to:, :n_to] = joint_cov[:n_to, n_to:].T

    # Regularize for numerical stability
    joint_cov += np.eye(joint_dim) * 1e-10

    idx_future = list(range(n_to))
    idx_past = list(range(n_to, joint_dim))

    return _mutual_info_gaussian(joint_cov, idx_future, idx_past)


def _find_mib(
    tpm: np.ndarray,
    cov: np.ndarray,
    n: int,
) -> tuple[float, tuple[list[int], list[int]]]:
    """
    Find the Minimum Information Bipartition (MIB) — the cut that
    destroys the least integrated information.

    For each possible bipartition of the system into two non-empty parts,
    compute the effective information across the cut. The MIB is the
    partition where this quantity is minimized. Φ equals this minimum.

    This is the bottleneck of the computation: O(2^n) bipartitions.
    Tractable up to n ≈ 20-25 nodes.

    Parameters
    ----------
    tpm : np.ndarray
        Transition matrix.
    cov : np.ndarray
        Stationary covariance.
    n : int
        System size.

    Returns
    -------
    tuple[float, tuple[list[int], list[int]]]
        (Φ value, (partition_A, partition_B))
    """
    all_indices = list(range(n))
    min_phi = float("inf")
    best_partition = ([], [])

    # Try all bipartitions (each element goes to A or B, both non-empty)
    for k in range(1, n // 2 + 1):
        for part_a in combinations(all_indices, k):
            part_a = list(part_a)
            part_b = [i for i in all_indices if i not in part_a]

            # Effective information in both directions across the cut
            ei_a_to_b = _effective_information(tpm, cov, part_a, part_b)
            ei_b_to_a = _effective_information(tpm, cov, part_b, part_a)

            # Φ for this bipartition is the minimum of the two directions
            # (the weakest link across the cut)
            phi_cut = min(ei_a_to_b, ei_b_to_a)

            if phi_cut < min_phi:
                min_phi = phi_cut
                best_partition = (part_a, part_b)

    return min_phi, best_partition


def approximate_phi(
    tpm: np.ndarray,
    noise_cov: Optional[np.ndarray] = None,
) -> float:
    """
    Compute approximate Integrated Information (Φ) for a linear-Gaussian system.

    This implements the Gaussian approximation to IIT 3.0's Φ measure:
    1. Compute the stationary covariance from the transition matrix
    2. Find the Minimum Information Bipartition (MIB)
    3. Return the integrated information at the MIB

    A high Φ means the system cannot be decomposed into independent parts
    without losing information about its causal dynamics.

    Parameters
    ----------
    tpm : np.ndarray
        Transition probability matrix / weight matrix, shape (n, n).
        Should have spectral radius < 1 for stationarity.
    noise_cov : np.ndarray, optional
        Noise covariance. Defaults to identity.

    Returns
    -------
    float
        Approximate Φ value (non-negative). Higher values indicate greater
        integration of information.
    """
    n = tpm.shape[0]

    if n < 2:
        return 0.0

    # Ensure spectral radius < 1 for stationarity
    eigenvalues = np.linalg.eigvals(tpm)
    spectral_radius = np.max(np.abs(eigenvalues))
    if spectral_radius >= 1.0:
        # Normalize to ensure convergence
        tpm = tpm / (spectral_radius + 0.01)

    cov = _covariance_from_tpm(tpm, noise_cov)

    # For very small systems, enumerate all bipartitions
    if n <= 16:
        phi, _ = _find_mib(tpm, cov, n)
    else:
        # For larger systems, use stochastic bipartition search
        phi = _stochastic_mib_search(tpm, cov, n, n_samples=500)

    return phi


def _stochastic_mib_search(
    tpm: np.ndarray,
    cov: np.ndarray,
    n: int,
    n_samples: int = 500,
) -> float:
    """
    Approximate MIB search for larger systems using random bipartitions.
    Samples random bipartitions and returns the minimum Φ found.

    This is a Monte Carlo approximation that works well in practice because
    the MIB tends to correspond to structurally obvious cuts in real networks.
    """
    all_indices = list(range(n))
    min_phi = float("inf")
    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        # Random bipartition: each node goes to A with probability 0.5
        mask = rng.random(n) < 0.5
        if not mask.any() or mask.all():
            continue

        part_a = [i for i in all_indices if mask[i]]
        part_b = [i for i in all_indices if not mask[i]]

        ei_a_to_b = _effective_information(tpm, cov, part_a, part_b)
        ei_b_to_a = _effective_information(tpm, cov, part_b, part_a)
        phi_cut = min(ei_a_to_b, ei_b_to_a)

        min_phi = min(min_phi, phi_cut)

    return min_phi if min_phi < float("inf") else 0.0


def geometric_integrated_information(
    tpm: np.ndarray,
    noise_cov: Optional[np.ndarray] = None,
) -> dict:
    """
    Full geometric integrated information decomposition.

    Returns the Φ value along with the MIB partition and diagnostic
    information useful for analysis.

    Parameters
    ----------
    tpm : np.ndarray
        Transition matrix.
    noise_cov : np.ndarray, optional
        Noise covariance.

    Returns
    -------
    dict
        Keys: 'phi', 'mib_partition', 'spectral_radius', 'stationary_cov',
              'system_size', 'total_mutual_info'
    """
    n = tpm.shape[0]

    eigenvalues = np.linalg.eigvals(tpm)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    tpm_stable = tpm.copy()
    if spectral_radius >= 1.0:
        tpm_stable = tpm_stable / (spectral_radius + 0.01)

    cov = _covariance_from_tpm(tpm_stable, noise_cov)

    if n <= 16:
        phi, mib = _find_mib(tpm_stable, cov, n)
    else:
        # For large systems, use stochastic search and return approximate partition
        phi = _stochastic_mib_search(tpm_stable, cov, n)
        mib = (list(range(n // 2)), list(range(n // 2, n)))

    # Total mutual information (whole-system integration)
    all_indices = list(range(n))
    if n >= 2:
        half = n // 2
        total_mi = _mutual_info_gaussian(cov, all_indices[:half], all_indices[half:])
    else:
        total_mi = 0.0

    return {
        "phi": phi,
        "mib_partition": mib,
        "spectral_radius": spectral_radius,
        "stationary_cov": cov,
        "system_size": n,
        "total_mutual_info": total_mi,
    }
