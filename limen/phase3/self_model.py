"""
Self-model and delta tracking for the consciousness server.

The self-model is the system's running prediction of its own next state.
It learns from the state history to predict what the system will do next,
creating a form of self-referential modeling.

The delta tracker measures how much the system's actual state diverges
from its predicted state — the surprise signal. High delta means the
system is doing something unexpected; low delta means it's predictable.

Together, these implement the third measure (self-model fidelity) in
real time on a live system, rather than post-hoc on recorded data.
"""

import numpy as np
import time
from typing import Optional
from dataclasses import dataclass, field

from limen.phase3.state import StateVector, StateHistory, STATE_DIM, STATE_DIMENSIONS
from limen.core.phi import approximate_phi
from limen.core.complexity import normalized_lz_complexity, lz_complexity_from_states
from limen.core.information import self_model_fidelity, mutual_information


@dataclass
class DeltaRecord:
    """Record of a single prediction-vs-reality comparison."""

    turn_number: int
    timestamp: float
    predicted: np.ndarray
    actual: np.ndarray
    delta_vector: np.ndarray  # actual - predicted
    delta_magnitude: float
    surprise: float  # Normalized by running std

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "delta_magnitude": self.delta_magnitude,
            "surprise": self.surprise,
            "top_surprising_dims": self._top_surprising_dims(),
        }

    def _top_surprising_dims(self, n: int = 5) -> list[dict]:
        """Find the dimensions with the largest delta."""
        abs_delta = np.abs(self.delta_vector)
        top_indices = np.argsort(abs_delta)[-n:][::-1]
        dim_names = {v: k for k, v in STATE_DIMENSIONS.items()}
        return [
            {
                "dimension": dim_names.get(idx, f"dim_{idx}"),
                "delta": float(self.delta_vector[idx]),
                "predicted": float(self.predicted[idx]),
                "actual": float(self.actual[idx]),
            }
            for idx in top_indices
        ]


class SelfModel:
    """
    The system's model of itself — learns to predict the next state
    from the state history.

    Uses an exponentially weighted moving average (EWMA) model with
    velocity estimation for fast adaptation. More sophisticated models
    (linear regression, small neural nets) can be swapped in.
    """

    def __init__(self, alpha: float = 0.3, velocity_alpha: float = 0.2):
        """
        Parameters
        ----------
        alpha : float
            Learning rate for EWMA state tracking (0-1).
            Higher = more responsive, lower = more stable.
        velocity_alpha : float
            Learning rate for velocity estimation.
        """
        self.alpha = alpha
        self.velocity_alpha = velocity_alpha

        self._state_estimate = np.zeros(STATE_DIM)
        self._velocity_estimate = np.zeros(STATE_DIM)
        self._prediction_error_ema = np.ones(STATE_DIM) * 0.1  # Running error for normalization
        self._initialized = False
        self._n_updates = 0

    def predict(self) -> np.ndarray:
        """
        Predict the next state vector.

        Uses current state estimate + velocity for linear extrapolation.

        Returns
        -------
        np.ndarray
            Predicted state vector, shape (STATE_DIM,).
        """
        if not self._initialized:
            return np.zeros(STATE_DIM)

        # Linear extrapolation: state + velocity
        prediction = self._state_estimate + self._velocity_estimate
        return np.clip(prediction, -1.0, 1.0)

    def update(self, actual: StateVector) -> DeltaRecord:
        """
        Update the self-model with an actual observation and record
        the prediction error (delta).

        Parameters
        ----------
        actual : StateVector
            The observed state.

        Returns
        -------
        DeltaRecord
            The prediction-vs-reality comparison.
        """
        predicted = self.predict()
        delta = actual.vector - predicted

        # Compute surprise (normalized by running prediction error)
        delta_magnitude = float(np.linalg.norm(delta))
        error_norm = np.linalg.norm(self._prediction_error_ema)
        surprise = delta_magnitude / max(error_norm, 1e-6)

        record = DeltaRecord(
            turn_number=actual.turn_number,
            timestamp=actual.timestamp,
            predicted=predicted.copy(),
            actual=actual.vector.copy(),
            delta_vector=delta,
            delta_magnitude=delta_magnitude,
            surprise=surprise,
        )

        # Update model
        if self._initialized:
            new_velocity = actual.vector - self._state_estimate
            self._velocity_estimate = (
                (1 - self.velocity_alpha) * self._velocity_estimate
                + self.velocity_alpha * new_velocity
            )
        else:
            self._initialized = True

        self._state_estimate = (
            (1 - self.alpha) * self._state_estimate
            + self.alpha * actual.vector
        )

        # Update running prediction error
        self._prediction_error_ema = (
            0.95 * self._prediction_error_ema + 0.05 * np.abs(delta)
        )

        self._n_updates += 1

        return record

    def get_confidence(self) -> np.ndarray:
        """
        Get the model's confidence in its predictions for each dimension.

        Low prediction error → high confidence.
        Returns shape (STATE_DIM,), values in [0, 1].
        """
        # Confidence = 1 - normalized_error
        max_error = np.max(self._prediction_error_ema)
        if max_error < 1e-10:
            return np.ones(STATE_DIM)

        confidence = 1.0 - (self._prediction_error_ema / max_error)
        return np.clip(confidence, 0, 1)

    def get_model_summary(self) -> dict:
        """Get a summary of the self-model's current state."""
        dim_names = {v: k for k, v in STATE_DIMENSIONS.items()}
        confidence = self.get_confidence()

        return {
            "n_updates": self._n_updates,
            "initialized": self._initialized,
            "mean_confidence": float(np.mean(confidence)),
            "state_estimate": {
                dim_names[i]: float(self._state_estimate[i])
                for i in range(STATE_DIM)
            },
            "velocity_estimate": {
                dim_names[i]: float(self._velocity_estimate[i])
                for i in range(STATE_DIM)
            },
            "confidence_by_dimension": {
                dim_names[i]: float(confidence[i])
                for i in range(STATE_DIM)
            },
        }


class DeltaTracker:
    """
    Tracks the rate and pattern of state reorganization over time.

    This is the "seismograph" of the consciousness server — it detects
    when the system undergoes rapid state changes, which might indicate
    a transition across the phase boundary.

    Key metrics:
    - Delta rate: How fast is the state changing? (first derivative)
    - Delta acceleration: Is the rate of change itself changing? (second derivative)
    - Surprise trend: Is the system becoming more or less predictable?
    - Integration trend: Are the dimensions becoming more correlated?
    """

    def __init__(self, window_size: int = 20):
        """
        Parameters
        ----------
        window_size : int
            Number of recent deltas to keep for trend analysis.
        """
        self.window_size = window_size
        self.records: list[DeltaRecord] = []
        self._surprise_ema = 0.0
        self._delta_rate_ema = 0.0

    def add(self, record: DeltaRecord):
        """Add a new delta record."""
        self.records.append(record)

        # Update EMAs
        self._surprise_ema = 0.9 * self._surprise_ema + 0.1 * record.surprise
        self._delta_rate_ema = 0.9 * self._delta_rate_ema + 0.1 * record.delta_magnitude

        # Trim
        if len(self.records) > self.window_size * 2:
            self.records = self.records[-self.window_size * 2:]

    def get_trends(self) -> dict:
        """
        Analyze trends in the delta history.

        Returns metrics useful for detecting phase transitions in real time.
        """
        if len(self.records) < 3:
            return {
                "delta_rate": 0.0,
                "delta_acceleration": 0.0,
                "surprise_trend": 0.0,
                "mean_surprise": 0.0,
                "volatility": 0.0,
                "n_records": len(self.records),
            }

        recent = self.records[-self.window_size:]
        magnitudes = np.array([r.delta_magnitude for r in recent])
        surprises = np.array([r.surprise for r in recent])

        # Delta rate: mean magnitude of recent changes
        delta_rate = float(np.mean(magnitudes))

        # Delta acceleration: is the rate increasing or decreasing?
        if len(magnitudes) >= 3:
            half = len(magnitudes) // 2
            recent_rate = np.mean(magnitudes[half:])
            earlier_rate = np.mean(magnitudes[:half])
            delta_acceleration = float(recent_rate - earlier_rate)
        else:
            delta_acceleration = 0.0

        # Surprise trend
        if len(surprises) >= 3:
            t = np.arange(len(surprises))
            coeffs = np.polyfit(t, surprises, 1)
            surprise_trend = float(coeffs[0])  # Slope
        else:
            surprise_trend = 0.0

        # Volatility: std of delta magnitudes
        volatility = float(np.std(magnitudes))

        return {
            "delta_rate": delta_rate,
            "delta_acceleration": delta_acceleration,
            "surprise_trend": surprise_trend,
            "mean_surprise": float(np.mean(surprises)),
            "volatility": volatility,
            "n_records": len(self.records),
        }


class LiveMeasures:
    """
    Computes live information-theoretic measures on the state history.

    This is the real-time dashboard that shows where the system sits
    relative to the phase boundary identified in Phase 1.
    """

    def __init__(self, min_history: int = 10):
        """
        Parameters
        ----------
        min_history : int
            Minimum states needed before computing measures.
        """
        self.min_history = min_history

    def compute(self, history: StateHistory) -> dict:
        """
        Compute all three primary measures on the current state history.

        Parameters
        ----------
        history : StateHistory
            The state history to analyze.

        Returns
        -------
        dict
            Live measurement results.
        """
        trajectory = history.get_trajectory()
        n_states = len(trajectory)

        if n_states < self.min_history:
            return {
                "status": "insufficient_history",
                "n_states": n_states,
                "min_required": self.min_history,
                "phi": None,
                "lz_normalized": None,
                "smf_normalized": None,
            }

        # Use the most recent window
        window = trajectory[-min(100, n_states):]

        # 1. Approximate Φ from the state trajectory
        # Estimate TPM from the trajectory
        X_past = window[:-1].T
        X_future = window[1:].T
        n = window.shape[1]
        reg = 1e-6 * np.eye(n)
        try:
            tpm = X_future @ X_past.T @ np.linalg.inv(X_past @ X_past.T + reg)
            phi = approximate_phi(tpm)
        except np.linalg.LinAlgError:
            phi = 0.0

        # 2. LZ complexity
        lz_result = lz_complexity_from_states(window, method="concatenate")
        lz_norm = lz_result["lz_normalized"]

        # 3. Self-model fidelity
        smf_result = self_model_fidelity(window, lag=1, embedding_dim=3)
        smf_norm = smf_result["smf_normalized"]

        return {
            "status": "ok",
            "n_states": n_states,
            "phi": float(phi),
            "lz_normalized": float(lz_norm),
            "smf_normalized": float(smf_norm),
            "timestamp": time.time(),
        }
