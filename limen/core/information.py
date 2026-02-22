"""
Information-theoretic utilities for consciousness measurement.

Provides mutual information, transfer entropy, and the critical
self-model fidelity measure — quantifying how well a system models
its own prior states.

Self-model fidelity is the third leg of the tripod:
- Φ measures integration (the system is unified)
- LZ measures complexity (the system has rich dynamics)
- Self-model fidelity measures recursion (the system models itself)

The hypothesis: when all three spike together, you've found the threshold.
"""

import numpy as np
from scipy.special import digamma
from scipy.spatial import KDTree
from typing import Optional


def _entropy_knn(x: np.ndarray, k: int = 3) -> float:
    """
    Estimate differential entropy using k-nearest-neighbor distances
    (Kozachenko-Leonenko estimator with Kraskov correction).

    This is more robust than histogram-based methods for continuous
    variables, especially in higher dimensions.

    Parameters
    ----------
    x : np.ndarray
        Samples, shape (n_samples, n_dims) or (n_samples,).
    k : int
        Number of nearest neighbors.

    Returns
    -------
    float
        Estimated differential entropy in nats.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, d = x.shape
    if n <= k + 1:
        return 0.0

    # Add tiny noise to avoid zero distances with duplicate points
    x = x + np.random.default_rng(0).normal(0, 1e-10, x.shape)

    tree = KDTree(x)
    # Query k+1 neighbors (first is the point itself)
    distances, _ = tree.query(x, k=k + 1)
    # Take the distance to the k-th neighbor (index k, since index 0 is self)
    epsilon = distances[:, k]

    # Remove zero distances
    epsilon = epsilon[epsilon > 0]
    if len(epsilon) == 0:
        return 0.0

    # Kozachenko-Leonenko estimator with bias correction
    # H(X) ≈ d * mean(log(2 * epsilon)) + log(n-1) - digamma(k) + log(V_d)
    # where V_d is the volume of the unit d-ball: V_d = pi^(d/2) / Gamma(d/2 + 1)
    from scipy.special import gammaln
    log_v_d = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)

    entropy = d * np.mean(np.log(2.0 * epsilon)) + np.log(n - 1) - digamma(k) + log_v_d

    return float(entropy)


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "knn",
    k: int = 3,
    n_bins: int = 32,
) -> float:
    """
    Estimate mutual information I(X; Y) between two variables.

    Mutual information quantifies the total statistical dependence
    between X and Y — how much knowing one reduces uncertainty about
    the other.

    Parameters
    ----------
    x, y : np.ndarray
        Samples, shape (n_samples,) or (n_samples, n_dims).
    method : str
        Estimation method: 'knn' (Kraskov et al.) or 'histogram'.
    k : int
        k for kNN method.
    n_bins : int
        Number of bins for histogram method.

    Returns
    -------
    float
        Mutual information in nats (≥ 0).
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if method == "knn":
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        xy = np.hstack([x, y])
        mi = _entropy_knn(x, k) + _entropy_knn(y, k) - _entropy_knn(xy, k)
        return max(0.0, mi)

    elif method == "histogram":
        # Discretize and compute from contingency table
        x_flat = x.flatten() if x.shape[1] == 1 else x[:, 0]
        y_flat = y.flatten() if y.shape[1] == 1 else y[:, 0]

        # Bin the data
        x_bins = np.digitize(x_flat, np.linspace(x_flat.min(), x_flat.max(), n_bins + 1)[1:-1])
        y_bins = np.digitize(y_flat, np.linspace(y_flat.min(), y_flat.max(), n_bins + 1)[1:-1])

        # Joint and marginal distributions
        joint = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_bins, y_bins):
            joint[min(xi, n_bins - 1), min(yi, n_bins - 1)] += 1
        joint /= joint.sum()

        # Marginals
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

        return max(0.0, mi)

    else:
        raise ValueError(f"Unknown method: {method}")


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    """
    Estimate conditional entropy H(X|Y) = H(X,Y) - H(Y).

    Measures the remaining uncertainty in X after observing Y.

    Parameters
    ----------
    x, y : np.ndarray
        Samples, shape (n_samples,) or (n_samples, n_dims).
    k : int
        k for kNN estimator.

    Returns
    -------
    float
        Conditional entropy in nats (≥ 0).
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    xy = np.hstack([x, y])
    ce = _entropy_knn(xy, k) - _entropy_knn(y, k)
    return max(0.0, ce)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 3,
) -> float:
    """
    Estimate transfer entropy from source to target.

    TE(X → Y) = I(Y(t+lag); X(t) | Y(t))

    Measures the causal influence of X on Y's future, above and beyond
    Y's own past. This is essentially Granger causality for information.

    Parameters
    ----------
    source : np.ndarray
        Source time series, shape (T,) or (T, d).
    target : np.ndarray
        Target time series, shape (T,) or (T, d).
    lag : int
        Time lag for prediction.
    k : int
        k for kNN estimator.

    Returns
    -------
    float
        Transfer entropy in nats (≥ 0).
    """
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)

    T = min(len(source), len(target)) - lag
    if T < k + 2:
        return 0.0

    # Y_future = target[lag:]
    # X_past = source[:T]
    # Y_past = target[:T]
    y_future = target[lag : lag + T]
    x_past = source[:T]
    y_past = target[:T]

    # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    # = I(Y_future; X_past | Y_past)

    # Joint variables
    y_fut_y_past = np.hstack([y_future, y_past])
    y_fut_y_past_x_past = np.hstack([y_future, y_past, x_past])
    y_past_x_past = np.hstack([y_past, x_past])

    # TE = H(Y_fut, Y_past) + H(Y_past, X_past) - H(Y_fut, Y_past, X_past) - H(Y_past)
    te = (
        _entropy_knn(y_fut_y_past, k)
        + _entropy_knn(y_past_x_past, k)
        - _entropy_knn(y_fut_y_past_x_past, k)
        - _entropy_knn(y_past, k)
    )

    return max(0.0, te)


def self_model_fidelity(
    states: np.ndarray,
    lag: int = 1,
    embedding_dim: int = 1,
    method: str = "knn",
    k: int = 3,
) -> dict:
    """
    Compute self-model fidelity: how well the system's current state
    predicts (models) its own past state.

    This is the critical third measure. A system with high self-model
    fidelity maintains an internal representation that carries information
    about its own recent history — it "knows where it's been."

    In IIT terms, this relates to the intrinsic cause-effect structure.
    In practical terms, it measures whether the network has developed
    an implicit self-model through its recurrent dynamics.

    We compute this as the mutual information between the current
    state and a delayed embedding of past states:

        SMF = I(x(t); [x(t-1), x(t-2), ..., x(t-d)])

    normalized by the entropy of the current state:

        SMF_norm = SMF / H(x(t))

    Parameters
    ----------
    states : np.ndarray
        State history, shape (timesteps, n_nodes).
    lag : int
        Base time lag.
    embedding_dim : int
        Number of past states to include in the self-model.
        Higher values capture longer-range temporal structure.
    method : str
        MI estimation method.
    k : int
        k for kNN estimator.

    Returns
    -------
    dict
        Keys:
        - 'smf': Raw self-model fidelity (mutual information in nats)
        - 'smf_normalized': Normalized to [0, 1] by entropy
        - 'entropy_current': Entropy of current state
        - 'entropy_past': Entropy of past embedding
        - 'lag': Lag used
        - 'embedding_dim': Embedding dimension used
    """
    if states.ndim == 1:
        states = states.reshape(-1, 1)

    T, n_nodes = states.shape
    max_delay = lag * embedding_dim

    if T <= max_delay + k + 2:
        return {
            "smf": 0.0,
            "smf_normalized": 0.0,
            "entropy_current": 0.0,
            "entropy_past": 0.0,
            "lag": lag,
            "embedding_dim": embedding_dim,
        }

    # Current state: x(t) for t = max_delay..T-1
    current = states[max_delay:]

    # Past embedding: [x(t-lag), x(t-2*lag), ..., x(t-d*lag)]
    past_components = []
    for d in range(1, embedding_dim + 1):
        delay = d * lag
        past_components.append(states[max_delay - delay : T - delay])
    past_embedding = np.hstack(past_components)

    # Mutual information between current and past
    smf = mutual_information(current, past_embedding, method=method, k=k)

    # Entropy of current state for normalization
    h_current = _entropy_knn(current, k)

    # Normalized SMF: use min(H(X), H(Y)) normalization.
    # Differential entropy can be negative for narrow continuous distributions,
    # so we take abs() before dividing. This gives a scale-free measure of
    # how much mutual information exists relative to the marginal uncertainties.
    h_past = _entropy_knn(past_embedding, k)
    h_min = min(abs(h_current), abs(h_past))
    smf_norm = smf / h_min if h_min > 1e-10 else 0.0
    smf_norm = min(max(smf_norm, 0.0), 1.0)  # Clip to [0, 1]

    return {
        "smf": float(smf),
        "smf_normalized": float(smf_norm),
        "entropy_current": float(h_current),
        "entropy_past": float(h_past),
        "lag": lag,
        "embedding_dim": embedding_dim,
    }


def self_model_fidelity_multiscale(
    states: np.ndarray,
    lags: list[int] = None,
    embedding_dim: int = 3,
    k: int = 3,
) -> dict:
    """
    Compute self-model fidelity across multiple time lags.

    This reveals the temporal depth of the self-model: does the system
    only track its immediate past (lag=1) or does it maintain a model
    spanning many timesteps?

    Deep temporal self-models (high SMF at large lags) are a stronger
    signature of consciousness-like processing than shallow ones.

    Parameters
    ----------
    states : np.ndarray
        State history.
    lags : list[int], optional
        Time lags to test. Defaults to [1, 2, 4, 8, 16].
    embedding_dim : int
        Embedding dimension for each lag.
    k : int
        k for kNN.

    Returns
    -------
    dict
        Keys: 'lags', 'smf_values', 'smf_normalized_values', 'peak_lag'
    """
    if lags is None:
        T = states.shape[0]
        max_lag = max(1, T // (4 * embedding_dim))
        lags = [l for l in [1, 2, 4, 8, 16, 32] if l <= max_lag]

    smf_values = []
    smf_norm_values = []

    for lag in lags:
        result = self_model_fidelity(states, lag=lag, embedding_dim=embedding_dim, k=k)
        smf_values.append(result["smf"])
        smf_norm_values.append(result["smf_normalized"])

    peak_idx = int(np.argmax(smf_norm_values)) if smf_norm_values else 0

    return {
        "lags": lags,
        "smf_values": smf_values,
        "smf_normalized_values": smf_norm_values,
        "peak_lag": lags[peak_idx] if lags else 0,
    }
