"""
Algorithmic complexity measures for state histories.

Implements Lempel-Ziv complexity (LZ76) and related measures to quantify
the algorithmic complexity of a network's temporal dynamics. Higher LZ
complexity indicates richer, less compressible dynamics — a necessary
(but not sufficient) condition for consciousness-like processing.

The intuition: a system that just repeats the same pattern has low complexity.
A system that's purely random has high complexity but no structure.
A conscious-like system should have *intermediate* complexity — structured
but not predictable, the edge between order and chaos.
"""

import numpy as np
from typing import Union


def _binarize_state_history(states: np.ndarray, threshold: str = "median") -> np.ndarray:
    """
    Convert continuous state histories to binary sequences for LZ analysis.

    Parameters
    ----------
    states : np.ndarray
        State history, shape (timesteps, n_nodes) or (timesteps,).
    threshold : str
        Binarization method: 'median', 'mean', or 'zero'.

    Returns
    -------
    np.ndarray
        Binary array of same shape, dtype int.
    """
    if threshold == "median":
        thresh = np.median(states, axis=0)
    elif threshold == "mean":
        thresh = np.mean(states, axis=0)
    elif threshold == "zero":
        thresh = 0.0
    else:
        raise ValueError(f"Unknown threshold method: {threshold}")

    return (states > thresh).astype(int)


def lempel_ziv_complexity(sequence: Union[np.ndarray, list, str]) -> int:
    """
    Compute the Lempel-Ziv complexity (LZ76) of a binary sequence.

    The algorithm scans the sequence left to right, counting the number
    of distinct patterns (words) encountered. Each time a new substring
    is found that hasn't been seen as an extension of previous patterns,
    the complexity counter increments.

    This is a measure of the algorithmic complexity of the sequence —
    how many distinct patterns it contains, which relates to how
    compressible it is.

    Parameters
    ----------
    sequence : array-like or str
        Binary sequence (0s and 1s).

    Returns
    -------
    int
        LZ complexity (number of distinct words in the LZ parsing).

    References
    ----------
    Lempel, A. & Ziv, J. (1976). On the Complexity of Finite Sequences.
    IEEE Transactions on Information Theory, 22(1), 75-81.
    """
    if isinstance(sequence, np.ndarray):
        s = "".join(str(int(x)) for x in sequence.flatten())
    elif isinstance(sequence, list):
        s = "".join(str(int(x)) for x in sequence)
    else:
        s = str(sequence)

    n = len(s)
    if n == 0:
        return 0

    # LZ76 parsing algorithm
    complexity = 1  # First symbol is always a new word
    i = 0  # Start of current component
    k = 1  # Current position
    k_max = 1  # Longest match found
    l = 1  # Length of current match attempt

    while k + l <= n:
        # Check if s[k:k+l] appears in s[i:k+l-1]
        substring = s[k : k + l]
        search_space = s[i : k + l - 1]

        if substring in search_space:
            l += 1
            if k + l > n:
                complexity += 1
        else:
            # Found a new word
            k_max = max(k_max, l)
            complexity += 1
            k += l if l > 1 else 1
            i = 0  # Reset search to beginning
            l = 1

    return complexity


def normalized_lz_complexity(
    sequence: Union[np.ndarray, list, str],
) -> float:
    """
    Compute normalized Lempel-Ziv complexity, scaled to [0, 1].

    Normalization uses the theoretical upper bound for a random binary
    sequence of length n: c(n) ~ n / log2(n).

    A value near 0 indicates highly regular/compressible dynamics.
    A value near 1 indicates random/incompressible dynamics.
    Values in the "interesting" range (0.3-0.7) suggest structured
    but non-trivial dynamics — the sweet spot for consciousness-like
    information processing.

    Parameters
    ----------
    sequence : array-like or str
        Binary sequence.

    Returns
    -------
    float
        Normalized LZ complexity in [0, 1].
    """
    if isinstance(sequence, np.ndarray):
        n = sequence.size
    elif isinstance(sequence, str):
        n = len(sequence)
    else:
        n = len(sequence)

    if n <= 1:
        return 0.0

    c = lempel_ziv_complexity(sequence)

    # Theoretical upper bound for random binary sequence
    # c_max ≈ n / log2(n) for large n
    log2_n = np.log2(n)
    if log2_n == 0:
        return 0.0

    c_max = n / log2_n

    return min(c / c_max, 1.0)


def lz_complexity_from_states(
    states: np.ndarray,
    method: str = "concatenate",
    threshold: str = "median",
) -> dict:
    """
    Compute LZ complexity from a network state history.

    Offers multiple methods for converting a multivariate continuous
    state history into a single sequence for LZ analysis.

    Parameters
    ----------
    states : np.ndarray
        State history, shape (timesteps, n_nodes).
    method : str
        How to create the binary sequence:
        - 'concatenate': Binarize each node, concatenate all node sequences
        - 'spatial': At each timestep, concatenate the binary state across nodes
        - 'pca': Project onto first PC, then binarize the 1D sequence
    threshold : str
        Binarization threshold method.

    Returns
    -------
    dict
        Keys: 'lz_raw', 'lz_normalized', 'sequence_length', 'method'
    """
    if states.ndim == 1:
        states = states.reshape(-1, 1)

    T, n_nodes = states.shape
    binary = _binarize_state_history(states, threshold)

    if method == "concatenate":
        # Concatenate all node time series into one long sequence
        sequence = binary.T.flatten()  # Node-major order

    elif method == "spatial":
        # At each timestep, read off the spatial pattern as a binary word
        sequence = binary.flatten()  # Time-major order

    elif method == "pca":
        # Project onto first principal component
        centered = states - states.mean(axis=0)
        if n_nodes > 1:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ vt[0]
        else:
            projected = centered.flatten()
        sequence = (projected > np.median(projected)).astype(int)

    else:
        raise ValueError(f"Unknown method: {method}")

    lz_raw = lempel_ziv_complexity(sequence)
    lz_norm = normalized_lz_complexity(sequence)

    return {
        "lz_raw": lz_raw,
        "lz_normalized": lz_norm,
        "sequence_length": len(sequence),
        "method": method,
    }


def multiscale_lz_complexity(
    states: np.ndarray,
    scales: list[int] = None,
    threshold: str = "median",
) -> dict:
    """
    Compute LZ complexity at multiple temporal scales by coarse-graining.

    At each scale factor s, average consecutive blocks of s timesteps,
    then compute LZ complexity. This reveals whether complexity is
    concentrated at fine timescales (noise) or persists across scales
    (genuine structure).

    A consciousness-relevant signal should show persistent complexity
    across scales, not just at the finest resolution.

    Parameters
    ----------
    states : np.ndarray
        State history, shape (timesteps, n_nodes).
    scales : list[int], optional
        Coarse-graining factors. Defaults to [1, 2, 4, 8, 16].
    threshold : str
        Binarization method.

    Returns
    -------
    dict
        Keys: 'scales', 'lz_values', 'lz_normalized_values'
    """
    if scales is None:
        T = states.shape[0]
        max_scale = max(1, T // 8)
        scales = [s for s in [1, 2, 4, 8, 16, 32] if s <= max_scale]

    lz_values = []
    lz_norm_values = []

    for s in scales:
        if s == 1:
            coarsened = states
        else:
            # Average blocks of s consecutive timesteps
            T = states.shape[0]
            n_blocks = T // s
            if n_blocks < 4:
                continue
            coarsened = states[: n_blocks * s].reshape(n_blocks, s, -1).mean(axis=1)

        result = lz_complexity_from_states(coarsened, method="concatenate", threshold=threshold)
        lz_values.append(result["lz_raw"])
        lz_norm_values.append(result["lz_normalized"])

    # Trim scales to match computed values
    scales = scales[: len(lz_values)]

    return {
        "scales": scales,
        "lz_values": lz_values,
        "lz_normalized_values": lz_norm_values,
    }
