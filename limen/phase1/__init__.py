"""
Phase 1: The $20 Experiment.

Small recurrent neural networks (8-256 nodes) with parameter sweeps
measuring integrated information, algorithmic complexity, and self-model
fidelity to detect phase transitions in information processing.
"""

from limen.phase1.network import RecurrentNetwork
from limen.phase1.sweep import ParameterSweep, SweepResult
from limen.phase1.measures import compute_all_measures
from limen.phase1.visualize import plot_phase_transition, plot_sweep_results

__all__ = [
    "RecurrentNetwork",
    "ParameterSweep",
    "SweepResult",
    "compute_all_measures",
    "plot_phase_transition",
    "plot_sweep_results",
]
