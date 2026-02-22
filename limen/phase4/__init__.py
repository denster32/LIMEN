"""
Phase 4: Applications.

Real-time monitoring and detection tools built on the phase transition
framework. If the detector works, these are the practical applications:
anesthesia monitoring, disorders of consciousness screening, AI safety
monitoring, and psychedelic therapy dosing.
"""

from limen.phase4.detector import PhaseTransitionDetector, DetectorConfig
from limen.phase4.monitor import ConsciousnessMonitor, MonitorConfig

__all__ = [
    "PhaseTransitionDetector",
    "DetectorConfig",
    "ConsciousnessMonitor",
    "MonitorConfig",
]
