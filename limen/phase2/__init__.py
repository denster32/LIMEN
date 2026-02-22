"""
Phase 2: Biological Validation.

Test whether the same mathematical signatures found in synthetic networks
(Phase 1) appear in known consciousness transitions from EEG data:
anesthesia, sleep, seizures, and psychedelic states.
"""

from limen.phase2.eeg_loader import EEGLoader, EEGSegment
from limen.phase2.biological import BiologicalMeasures
from limen.phase2.validate import CrossSubstrateValidator

__all__ = [
    "EEGLoader",
    "EEGSegment",
    "BiologicalMeasures",
    "CrossSubstrateValidator",
]
