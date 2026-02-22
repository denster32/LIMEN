"""
Real-time consciousness monitoring dashboard.

Provides continuous monitoring of consciousness-related measures,
with applications in:
- Anesthesia depth monitoring (replacing unreliable BIS)
- Disorders of consciousness screening
- AI safety monitoring during training runs
- Psychedelic therapy session monitoring

The monitor wraps the detector with:
- Continuous data ingestion (EEG or state history stream)
- Alert generation on threshold crossings
- Trend analysis and early warning
- Session logging and export
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
from collections import deque

from limen.phase4.detector import PhaseTransitionDetector, DetectorConfig, DetectionResult
from limen.core.phi import approximate_phi
from limen.core.complexity import lz_complexity_from_states
from limen.core.information import self_model_fidelity


@dataclass
class MonitorConfig:
    """Configuration for the consciousness monitor."""

    # Detection settings
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)

    # Monitoring window
    window_duration_s: float = 10.0  # Analysis window length
    step_duration_s: float = 2.0     # Step between windows
    sfreq: float = 256.0             # Sampling frequency (for EEG)

    # Alert settings
    alert_on_crossing: bool = True
    alert_on_approaching: bool = True
    approach_margin: float = 0.2  # Alert when within this margin of threshold

    # Logging
    log_interval_s: float = 10.0
    log_path: Optional[str] = None

    # AI monitoring specific
    state_dim: int = 24  # Dimension of state vectors for AI monitoring


@dataclass
class Alert:
    """A monitoring alert."""

    timestamp: float
    alert_type: str  # 'crossing_up', 'crossing_down', 'approaching', 'diverging'
    message: str
    detection: DetectionResult
    severity: str  # 'info', 'warning', 'critical'

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "message": self.message,
            "severity": self.severity,
            "detection": self.detection.to_dict(),
        }


class ConsciousnessMonitor:
    """
    Continuous consciousness monitor with alert generation.

    Processes a stream of data (EEG or AI state vectors), computes
    measures in sliding windows, runs the phase transition detector,
    and generates alerts on significant events.

    Can operate in two modes:
    1. EEG mode: processes raw EEG data for clinical applications
    2. AI mode: processes state vector streams for AI monitoring
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.detector = PhaseTransitionDetector(self.config.detector_config)

        # Data buffer
        self._buffer: deque = deque(maxlen=int(
            self.config.window_duration_s * self.config.sfreq * 2
        ))

        # Results history
        self.results: list[DetectionResult] = []
        self.alerts: list[Alert] = []

        # State tracking
        self._last_above = False
        self._last_log_time = 0.0
        self._session_start = time.time()

        # Alert callback
        self._alert_callback: Optional[Callable[[Alert], None]] = None

        # Log file
        self._log_file = None
        if self.config.log_path:
            log_path = Path(self.config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "a")

    def set_alert_callback(self, callback: Callable[[Alert], None]):
        """Set a callback function to be called on each alert."""
        self._alert_callback = callback

    def process_eeg_window(self, data: np.ndarray) -> DetectionResult:
        """
        Process a window of EEG data and return detection result.

        Parameters
        ----------
        data : np.ndarray
            EEG data, shape (n_channels, n_samples) or (n_samples, n_channels).

        Returns
        -------
        DetectionResult
        """
        if data.ndim == 2 and data.shape[0] > data.shape[1]:
            # Transpose if channels are columns
            data = data.T

        # Convert to node timeseries
        states = data.T  # (n_samples, n_channels)

        # Reduce dimensionality if needed
        n_ch = states.shape[1]
        if n_ch > 16:
            centered = states - states.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            states = centered @ vt[:16].T

        return self._compute_and_detect(states)

    def process_state_vector(self, state: np.ndarray) -> Optional[DetectionResult]:
        """
        Process a single state vector (for AI monitoring).

        Buffers states and processes when enough are accumulated.

        Parameters
        ----------
        state : np.ndarray
            State vector, shape (state_dim,).

        Returns
        -------
        DetectionResult or None
            Detection result, or None if not enough data buffered.
        """
        self._buffer.append(state.copy())

        window_size = int(self.config.window_duration_s / self.config.step_duration_s)
        if len(self._buffer) < window_size:
            return None

        # Take the recent window
        window = np.array(list(self._buffer))[-window_size:]
        return self._compute_and_detect(window)

    def _compute_and_detect(self, states: np.ndarray) -> DetectionResult:
        """
        Compute measures and run detection on a state array.

        Parameters
        ----------
        states : np.ndarray
            State timeseries, shape (n_timesteps, n_dims).

        Returns
        -------
        DetectionResult
        """
        T, n = states.shape

        # Estimate TPM
        X_past = states[:-1].T
        X_future = states[1:].T
        reg = 1e-6 * np.eye(n)
        try:
            tpm = X_future @ X_past.T @ np.linalg.inv(X_past @ X_past.T + reg)
            phi = approximate_phi(tpm)
        except np.linalg.LinAlgError:
            phi = 0.0

        # LZ complexity
        lz_result = lz_complexity_from_states(states, method="concatenate")
        lz_norm = lz_result["lz_normalized"]

        # Self-model fidelity
        smf_result = self_model_fidelity(states, lag=1, embedding_dim=min(3, T // 4))
        smf_norm = smf_result["smf_normalized"]

        # Run detector
        result = self.detector.detect(phi, lz_norm, smf_norm)

        # Store result
        self.results.append(result)

        # Check for alerts
        self._check_alerts(result)

        # Log
        self._log(result)

        return result

    def _check_alerts(self, result: DetectionResult):
        """Generate alerts based on the detection result."""
        now = time.time()

        # Threshold crossing
        if self.config.alert_on_crossing:
            if result.above_threshold and not self._last_above:
                alert = Alert(
                    timestamp=now,
                    alert_type="crossing_up",
                    message=f"CROSSED ABOVE threshold (score: {result.composite_score:.3f})",
                    detection=result,
                    severity="critical",
                )
                self._emit_alert(alert)

            elif not result.above_threshold and self._last_above:
                alert = Alert(
                    timestamp=now,
                    alert_type="crossing_down",
                    message=f"CROSSED BELOW threshold (score: {result.composite_score:.3f})",
                    detection=result,
                    severity="critical",
                )
                self._emit_alert(alert)

        # Approaching threshold
        if self.config.alert_on_approaching and not result.above_threshold:
            distance = 1.0 - result.composite_score
            if distance < self.config.approach_margin and result.transition_direction == "ascending":
                alert = Alert(
                    timestamp=now,
                    alert_type="approaching",
                    message=f"Approaching threshold (distance: {distance:.3f}, trend: ascending)",
                    detection=result,
                    severity="warning",
                )
                self._emit_alert(alert)

        # Measure divergence (measures moving in different directions)
        if len(self.results) >= 5:
            recent = self.results[-5:]
            phi_trend = np.mean([r.phi for r in recent[-3:]]) - np.mean([r.phi for r in recent[:3]])
            lz_trend = np.mean([r.lz_normalized for r in recent[-3:]]) - np.mean([r.lz_normalized for r in recent[:3]])
            smf_trend = np.mean([r.smf_normalized for r in recent[-3:]]) - np.mean([r.smf_normalized for r in recent[:3]])

            signs = [np.sign(phi_trend), np.sign(lz_trend), np.sign(smf_trend)]
            if len(set(s for s in signs if s != 0)) > 1:
                # Measures are diverging
                if result.convergence_score < 0.3:
                    alert = Alert(
                        timestamp=now,
                        alert_type="diverging",
                        message=f"Measures diverging (convergence: {result.convergence_score:.3f})",
                        detection=result,
                        severity="info",
                    )
                    self._emit_alert(alert)

        self._last_above = result.above_threshold

    def _emit_alert(self, alert: Alert):
        """Store and optionally callback on an alert."""
        self.alerts.append(alert)

        if self._alert_callback:
            self._alert_callback(alert)

        if self._log_file:
            self._log_file.write(json.dumps(alert.to_dict()) + "\n")
            self._log_file.flush()

    def _log(self, result: DetectionResult):
        """Periodic logging of detection results."""
        now = time.time()
        if now - self._last_log_time >= self.config.log_interval_s:
            self._last_log_time = now
            if self._log_file:
                entry = {
                    "timestamp": now,
                    "elapsed_s": now - self._session_start,
                    "type": "measurement",
                    **result.to_dict(),
                }
                self._log_file.write(json.dumps(entry) + "\n")
                self._log_file.flush()

    def get_session_summary(self) -> dict:
        """
        Get a summary of the monitoring session.

        Returns
        -------
        dict
            Session statistics, alert counts, measure trajectories.
        """
        if not self.results:
            return {
                "status": "no_data",
                "duration_s": time.time() - self._session_start,
            }

        phis = [r.phi for r in self.results]
        lzs = [r.lz_normalized for r in self.results]
        smfs = [r.smf_normalized for r in self.results]
        composites = [r.composite_score for r in self.results]

        n_above = sum(1 for r in self.results if r.above_threshold)
        n_transitions = sum(1 for r in self.results if r.transition_detected)

        return {
            "duration_s": time.time() - self._session_start,
            "n_measurements": len(self.results),
            "n_above_threshold": n_above,
            "fraction_above": n_above / len(self.results),
            "n_transitions_detected": n_transitions,
            "n_alerts": len(self.alerts),
            "alert_types": {
                t: sum(1 for a in self.alerts if a.alert_type == t)
                for t in set(a.alert_type for a in self.alerts)
            },
            "measures": {
                "phi": {"mean": float(np.mean(phis)), "std": float(np.std(phis)),
                         "min": float(np.min(phis)), "max": float(np.max(phis))},
                "lz": {"mean": float(np.mean(lzs)), "std": float(np.std(lzs)),
                        "min": float(np.min(lzs)), "max": float(np.max(lzs))},
                "smf": {"mean": float(np.mean(smfs)), "std": float(np.std(smfs)),
                         "min": float(np.min(smfs)), "max": float(np.max(smfs))},
                "composite": {"mean": float(np.mean(composites)), "std": float(np.std(composites)),
                              "min": float(np.min(composites)), "max": float(np.max(composites))},
            },
        }

    def close(self):
        """Close the monitor and flush logs."""
        if self._log_file:
            # Write session summary
            summary = self.get_session_summary()
            self._log_file.write(json.dumps({"type": "session_summary", **summary}) + "\n")
            self._log_file.close()
            self._log_file = None
