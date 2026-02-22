"""
Phase transition detector.

Given a calibrated threshold from Phase 1 and validated signatures from
Phase 2, this module provides a real-time detector that classifies whether
an input signal (EEG, network state, AI state history) is above or below
the consciousness phase boundary.

The detector uses all three measures (Φ, LZ, SMF) and their convergence
to make a binary classification with a confidence score.

This is deliberately simple — a logistic regression on the three measures
and their derivatives. The complexity is in the measures, not the classifier.
If the phase transition is real, a simple classifier should suffice.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration for the phase transition detector."""

    # Threshold parameters (calibrated from Phase 1)
    phi_threshold: float = 0.3
    lz_threshold: float = 0.4
    smf_threshold: float = 0.3

    # Weights for combining measures
    phi_weight: float = 0.4
    lz_weight: float = 0.3
    smf_weight: float = 0.3

    # Derivative thresholds (for detecting transitions in progress)
    derivative_threshold: float = 0.1

    # Convergence requirement (how close do measures need to be?)
    min_convergence_score: float = 0.5

    # Smoothing (exponential moving average)
    ema_alpha: float = 0.3

    # History for derivative computation
    window_size: int = 10


@dataclass
class DetectionResult:
    """Result of a single detection step."""

    # Binary classification
    above_threshold: bool
    confidence: float  # 0 to 1

    # Individual measures
    phi: float
    lz_normalized: float
    smf_normalized: float

    # Composite score (weighted combination)
    composite_score: float

    # Transition detection
    transition_detected: bool
    transition_direction: Optional[str]  # 'ascending' or 'descending' or None
    derivative_magnitude: float

    # Convergence
    convergence_score: float

    # Smoothed values
    phi_smoothed: float
    lz_smoothed: float
    smf_smoothed: float

    def to_dict(self) -> dict:
        return {
            "above_threshold": self.above_threshold,
            "confidence": round(self.confidence, 4),
            "composite_score": round(self.composite_score, 4),
            "phi": round(self.phi, 4),
            "lz_normalized": round(self.lz_normalized, 4),
            "smf_normalized": round(self.smf_normalized, 4),
            "transition_detected": self.transition_detected,
            "transition_direction": self.transition_direction,
            "derivative_magnitude": round(self.derivative_magnitude, 4),
            "convergence_score": round(self.convergence_score, 4),
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        state = "ABOVE" if self.above_threshold else "BELOW"
        trans = ""
        if self.transition_detected:
            trans = f" [TRANSITION {self.transition_direction}]"
        return (
            f"{state} threshold (confidence: {self.confidence:.2f}, "
            f"score: {self.composite_score:.3f}){trans}"
        )


class PhaseTransitionDetector:
    """
    Real-time phase transition detector.

    Maintains a running estimate of the three consciousness measures
    and classifies whether the system is above or below the phase
    boundary. Also detects transitions in progress.

    Usage:
        detector = PhaseTransitionDetector()
        result = detector.detect(phi=0.5, lz=0.6, smf=0.4)
        print(result.summary())
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

        # Smoothed values
        self._phi_ema = 0.0
        self._lz_ema = 0.0
        self._smf_ema = 0.0

        # History for derivatives
        self._phi_history: list[float] = []
        self._lz_history: list[float] = []
        self._smf_history: list[float] = []
        self._composite_history: list[float] = []

        self._initialized = False

    def detect(
        self,
        phi: float,
        lz_normalized: float,
        smf_normalized: float,
    ) -> DetectionResult:
        """
        Run detection on a single measurement.

        Parameters
        ----------
        phi : float
            Integrated information value.
        lz_normalized : float
            Normalized Lempel-Ziv complexity.
        smf_normalized : float
            Normalized self-model fidelity.

        Returns
        -------
        DetectionResult
        """
        c = self.config

        # Update EMA
        if self._initialized:
            self._phi_ema = (1 - c.ema_alpha) * self._phi_ema + c.ema_alpha * phi
            self._lz_ema = (1 - c.ema_alpha) * self._lz_ema + c.ema_alpha * lz_normalized
            self._smf_ema = (1 - c.ema_alpha) * self._smf_ema + c.ema_alpha * smf_normalized
        else:
            self._phi_ema = phi
            self._lz_ema = lz_normalized
            self._smf_ema = smf_normalized
            self._initialized = True

        # Composite score (weighted combination of normalized measures)
        phi_norm = self._phi_ema / max(c.phi_threshold, 1e-6)
        lz_norm = self._lz_ema / max(c.lz_threshold, 1e-6)
        smf_norm = self._smf_ema / max(c.smf_threshold, 1e-6)

        composite = (
            c.phi_weight * phi_norm +
            c.lz_weight * lz_norm +
            c.smf_weight * smf_norm
        )

        # Convergence: are all three measures elevated together?
        measures_above = [
            self._phi_ema > c.phi_threshold * 0.5,
            self._lz_ema > c.lz_threshold * 0.5,
            self._smf_ema > c.smf_threshold * 0.5,
        ]
        convergence = sum(measures_above) / 3.0

        # Strengthen convergence score by checking correlation of recent values
        if len(self._phi_history) >= 5:
            recent_phi = np.array(self._phi_history[-10:])
            recent_lz = np.array(self._lz_history[-10:])
            recent_smf = np.array(self._smf_history[-10:])

            # Pairwise correlations
            corrs = []
            for a, b in [(recent_phi, recent_lz), (recent_phi, recent_smf), (recent_lz, recent_smf)]:
                if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                    r = np.corrcoef(a, b)[0, 1]
                    corrs.append(max(0, r))  # Only positive correlation counts
            if corrs:
                convergence = 0.5 * convergence + 0.5 * np.mean(corrs)

        # Classification
        above_threshold = composite > 1.0 and convergence > c.min_convergence_score

        # Confidence based on distance from threshold and convergence
        distance = abs(composite - 1.0)
        confidence = 1.0 - np.exp(-distance * 3)
        confidence *= convergence

        # Transition detection (derivatives)
        self._phi_history.append(self._phi_ema)
        self._lz_history.append(self._lz_ema)
        self._smf_history.append(self._smf_ema)
        self._composite_history.append(composite)

        # Trim histories
        max_hist = c.window_size * 2
        self._phi_history = self._phi_history[-max_hist:]
        self._lz_history = self._lz_history[-max_hist:]
        self._smf_history = self._smf_history[-max_hist:]
        self._composite_history = self._composite_history[-max_hist:]

        transition_detected = False
        transition_direction = None
        derivative_magnitude = 0.0

        if len(self._composite_history) >= 3:
            deriv = np.gradient(self._composite_history[-c.window_size:])
            derivative_magnitude = float(np.abs(deriv[-1]))

            if derivative_magnitude > c.derivative_threshold:
                transition_detected = True
                transition_direction = "ascending" if deriv[-1] > 0 else "descending"

        return DetectionResult(
            above_threshold=above_threshold,
            confidence=float(confidence),
            phi=phi,
            lz_normalized=lz_normalized,
            smf_normalized=smf_normalized,
            composite_score=float(composite),
            transition_detected=transition_detected,
            transition_direction=transition_direction,
            derivative_magnitude=derivative_magnitude,
            convergence_score=float(convergence),
            phi_smoothed=float(self._phi_ema),
            lz_smoothed=float(self._lz_ema),
            smf_smoothed=float(self._smf_ema),
        )

    def calibrate(
        self,
        phi_values: np.ndarray,
        lz_values: np.ndarray,
        smf_values: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Calibrate the detector from Phase 1/Phase 2 results.

        Fits optimal thresholds and weights from labeled data.

        Parameters
        ----------
        phi_values, lz_values, smf_values : np.ndarray
            Measure values.
        labels : np.ndarray
            Binary labels (0 = below threshold, 1 = above).
        """
        # Simple approach: find thresholds that maximize separation
        for measure, values, attr in [
            ("phi", phi_values, "phi_threshold"),
            ("lz", lz_values, "lz_threshold"),
            ("smf", smf_values, "smf_threshold"),
        ]:
            above = values[labels == 1]
            below = values[labels == 0]

            if len(above) > 0 and len(below) > 0:
                # Threshold at the midpoint between class means
                threshold = (np.mean(above) + np.mean(below)) / 2
                setattr(self.config, attr, float(threshold))

        # Optimize weights by logistic regression
        X = np.column_stack([
            phi_values / max(self.config.phi_threshold, 1e-6),
            lz_values / max(self.config.lz_threshold, 1e-6),
            smf_values / max(self.config.smf_threshold, 1e-6),
        ])

        # Simple gradient descent on logistic loss
        weights = np.array([0.33, 0.33, 0.33])
        lr = 0.1

        for _ in range(1000):
            scores = X @ weights
            probs = 1 / (1 + np.exp(-scores))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)

            # Gradient of cross-entropy loss
            grad = X.T @ (probs - labels) / len(labels)
            weights -= lr * grad

            # Project to simplex (weights sum to 1, non-negative)
            weights = np.maximum(weights, 0.01)
            weights /= weights.sum()

        self.config.phi_weight = float(weights[0])
        self.config.lz_weight = float(weights[1])
        self.config.smf_weight = float(weights[2])

    def reset(self):
        """Reset the detector's internal state."""
        self._phi_ema = 0.0
        self._lz_ema = 0.0
        self._smf_ema = 0.0
        self._phi_history.clear()
        self._lz_history.clear()
        self._smf_history.clear()
        self._composite_history.clear()
        self._initialized = False
