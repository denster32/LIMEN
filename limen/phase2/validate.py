"""
Cross-substrate validation: comparing synthetic and biological signatures.

The central test of Phase 2: do the phase transition signatures found
in synthetic recurrent networks (Phase 1) match the transitions observed
in biological neural data during known consciousness changes?

If yes: the signature generalizes across substrates, strengthening the
hypothesis that it captures something fundamental about consciousness.

If no: the synthetic model is missing something essential about biological
substrate — which is itself an informative result.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from limen.phase2.eeg_loader import EEGLoader, EEGSegment
from limen.phase2.biological import BiologicalMeasures, BiologicalMeasurement
from limen.phase1.measures import MeasurementResult


@dataclass
class ValidationResult:
    """Results from cross-substrate validation."""

    # Correlation between synthetic and biological measure profiles
    phi_correlation: float
    lz_correlation: float
    smf_correlation: float

    # Critical exponent comparison
    # (slope of the transition in log-log space)
    synthetic_exponents: dict
    biological_exponents: dict
    exponent_match_score: float

    # Convergence analysis
    synthetic_convergence: float  # Do synthetic measures converge?
    biological_convergence: float  # Do biological measures converge?

    # Overall validation score
    validation_score: float

    # Raw data for plotting
    synthetic_profile: dict
    biological_profile: dict

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        lines = [
            "=" * 60,
            "LIMEN PHASE 2: Cross-Substrate Validation Results",
            "=" * 60,
            "",
            "Measure Correlations (synthetic vs. biological):",
            f"  Φ:   {self.phi_correlation:+.3f}",
            f"  LZ:  {self.lz_correlation:+.3f}",
            f"  SMF: {self.smf_correlation:+.3f}",
            "",
            "Convergence Scores:",
            f"  Synthetic:  {self.synthetic_convergence:.3f}",
            f"  Biological: {self.biological_convergence:.3f}",
            "",
            f"Critical Exponent Match: {self.exponent_match_score:.3f}",
            "",
            f"OVERALL VALIDATION SCORE: {self.validation_score:.3f}",
            "",
        ]

        if self.validation_score > 0.7:
            lines.append("→ STRONG MATCH: Synthetic signatures generalize to biological data.")
            lines.append("  The phase transition hypothesis is supported.")
        elif self.validation_score > 0.4:
            lines.append("→ PARTIAL MATCH: Some signatures generalize, others diverge.")
            lines.append("  Further investigation needed on divergent measures.")
        else:
            lines.append("→ WEAK MATCH: Synthetic predictions do not match biological data.")
            lines.append("  The simulation may be missing essential substrate features.")

        lines.append("=" * 60)
        return "\n".join(lines)


class CrossSubstrateValidator:
    """
    Validates whether synthetic phase transition signatures match
    biological consciousness transitions.

    Workflow:
    1. Generate synthetic transition profile (measures vs. parameter)
    2. Compute biological transition profile (measures vs. time)
    3. Compare the shape of the transitions (correlation, exponents)
    4. Score the match
    """

    def __init__(self, n_components: int = 16):
        self.bio_measures = BiologicalMeasures(n_components=n_components)
        self.eeg_loader = EEGLoader()

    def validate_transition(
        self,
        synthetic_results: list[MeasurementResult],
        eeg_segments: list[EEGSegment],
        verbose: bool = True,
    ) -> ValidationResult:
        """
        Run the full cross-substrate validation.

        Parameters
        ----------
        synthetic_results : list[MeasurementResult]
            Phase 1 results ordered by connection density.
        eeg_segments : list[EEGSegment]
            EEG segments ordered by time (spanning a transition).
        verbose : bool
            Print progress.

        Returns
        -------
        ValidationResult
        """
        if verbose:
            print("Phase 2: Cross-Substrate Validation")
            print("=" * 50)

        # Extract synthetic profile
        syn_phi = np.array([r.phi for r in synthetic_results])
        syn_lz = np.array([r.lz_normalized for r in synthetic_results])
        syn_smf = np.array([r.self_model_fidelity_normalized for r in synthetic_results])
        syn_densities = np.array([r.connection_density for r in synthetic_results])

        # Normalize synthetic profiles
        syn_phi_norm = self._normalize(syn_phi)
        syn_lz_norm = self._normalize(syn_lz)
        syn_smf_norm = self._normalize(syn_smf)

        if verbose:
            print(f"  Synthetic: {len(synthetic_results)} configurations")

        # Compute biological profile
        if verbose:
            print(f"  Computing biological measures on {len(eeg_segments)} segments...")

        bio_results = self.bio_measures.measure_transition(eeg_segments)

        bio_phi = np.array([r.phi for r in bio_results])
        bio_lz = np.array([r.lz_normalized for r in bio_results])
        bio_smf = np.array([r.self_model_fidelity_normalized for r in bio_results])
        bio_times = np.array([
            r.window_start_s if r.window_start_s is not None else i
            for i, r in enumerate(bio_results)
        ])

        # Normalize biological profiles
        bio_phi_norm = self._normalize(bio_phi)
        bio_lz_norm = self._normalize(bio_lz)
        bio_smf_norm = self._normalize(bio_smf)

        if verbose:
            print(f"  Biological: {len(bio_results)} windows measured")

        # Interpolate to common grid for comparison
        n_points = min(len(syn_phi_norm), len(bio_phi_norm), 50)
        common_grid = np.linspace(0, 1, n_points)

        syn_phi_interp = np.interp(common_grid, np.linspace(0, 1, len(syn_phi_norm)), syn_phi_norm)
        syn_lz_interp = np.interp(common_grid, np.linspace(0, 1, len(syn_lz_norm)), syn_lz_norm)
        syn_smf_interp = np.interp(common_grid, np.linspace(0, 1, len(syn_smf_norm)), syn_smf_norm)

        bio_phi_interp = np.interp(common_grid, np.linspace(0, 1, len(bio_phi_norm)), bio_phi_norm)
        bio_lz_interp = np.interp(common_grid, np.linspace(0, 1, len(bio_lz_norm)), bio_lz_norm)
        bio_smf_interp = np.interp(common_grid, np.linspace(0, 1, len(bio_smf_norm)), bio_smf_norm)

        # Compute correlations
        phi_corr = float(np.corrcoef(syn_phi_interp, bio_phi_interp)[0, 1])
        lz_corr = float(np.corrcoef(syn_lz_interp, bio_lz_interp)[0, 1])
        smf_corr = float(np.corrcoef(syn_smf_interp, bio_smf_interp)[0, 1])

        # Handle NaN correlations
        phi_corr = 0.0 if np.isnan(phi_corr) else phi_corr
        lz_corr = 0.0 if np.isnan(lz_corr) else lz_corr
        smf_corr = 0.0 if np.isnan(smf_corr) else smf_corr

        if verbose:
            print(f"\n  Correlations:")
            print(f"    Φ:   {phi_corr:+.3f}")
            print(f"    LZ:  {lz_corr:+.3f}")
            print(f"    SMF: {smf_corr:+.3f}")

        # Critical exponents (slope near transition in log space)
        syn_exponents = self._estimate_exponents(
            syn_densities, syn_phi, syn_lz, syn_smf
        )
        bio_exponents = self._estimate_exponents(
            bio_times, bio_phi, bio_lz, bio_smf
        )

        # Exponent match: compare ratios of exponents
        exponent_match = self._compare_exponents(syn_exponents, bio_exponents)

        # Convergence (do measures peak together?)
        syn_convergence = self._convergence_score(syn_phi_norm, syn_lz_norm, syn_smf_norm)
        bio_convergence = self._convergence_score(bio_phi_norm, bio_lz_norm, bio_smf_norm)

        # Overall validation score
        mean_corr = (abs(phi_corr) + abs(lz_corr) + abs(smf_corr)) / 3
        validation_score = (
            0.4 * mean_corr +
            0.3 * exponent_match +
            0.15 * syn_convergence +
            0.15 * bio_convergence
        )

        result = ValidationResult(
            phi_correlation=phi_corr,
            lz_correlation=lz_corr,
            smf_correlation=smf_corr,
            synthetic_exponents=syn_exponents,
            biological_exponents=bio_exponents,
            exponent_match_score=exponent_match,
            synthetic_convergence=syn_convergence,
            biological_convergence=bio_convergence,
            validation_score=validation_score,
            synthetic_profile={
                "parameter": syn_densities.tolist(),
                "phi": syn_phi_norm.tolist(),
                "lz": syn_lz_norm.tolist(),
                "smf": syn_smf_norm.tolist(),
            },
            biological_profile={
                "parameter": bio_times.tolist(),
                "phi": bio_phi_norm.tolist(),
                "lz": bio_lz_norm.tolist(),
                "smf": bio_smf_norm.tolist(),
            },
        )

        if verbose:
            print(result.summary())

        return result

    def validate_with_synthetic_data(
        self,
        synthetic_results: list[MeasurementResult],
        transition_type: str = "anesthesia",
        verbose: bool = True,
    ) -> ValidationResult:
        """
        Convenience method: validate against synthetic EEG data.

        Useful for pipeline development before real data is available.
        """
        # Generate synthetic EEG transition
        segments = self.eeg_loader.create_synthetic_transition(
            transition_type=transition_type,
            n_channels=19,
            duration_s=120,
        )

        # Window the transition
        all_windows = []
        for seg in segments:
            windows = self.eeg_loader.extract_transition_windows(
                seg, window_duration_s=10, step_s=2
            )
            all_windows.extend(windows)

        return self.validate_transition(
            synthetic_results, all_windows, verbose=verbose
        )

    def plot_validation(
        self,
        result: ValidationResult,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the cross-substrate comparison.

        Side-by-side plots of synthetic vs. biological measure profiles
        with correlation annotations.
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Limen Phase 2: Cross-Substrate Validation\n"
            f"(Score: {result.validation_score:.3f})",
            fontsize=14, fontweight="bold",
        )

        colors = {"phi": "#E63946", "lz": "#457B9D", "smf": "#2A9D8F"}
        labels = {"phi": r"$\Phi$", "lz": "LZ", "smf": "SMF"}
        corrs = {
            "phi": result.phi_correlation,
            "lz": result.lz_correlation,
            "smf": result.smf_correlation,
        }

        for i, key in enumerate(["phi", "lz", "smf"]):
            # Top row: profiles
            ax = axes[0, i]
            syn = result.synthetic_profile[key]
            bio = result.biological_profile[key]

            syn_x = np.linspace(0, 1, len(syn))
            bio_x = np.linspace(0, 1, len(bio))

            ax.plot(syn_x, syn, "-", color=colors[key], linewidth=2, label="Synthetic")
            ax.plot(bio_x, bio, "--", color=colors[key], linewidth=2, alpha=0.7, label="Biological")
            ax.set_title(f"{labels[key]} (r = {corrs[key]:+.3f})", fontsize=11)
            ax.set_xlabel("Normalized parameter")
            ax.set_ylabel("Normalized measure")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Bottom row: scatter (synthetic vs biological, interpolated)
            ax2 = axes[1, i]
            n_pts = min(len(syn), len(bio))
            syn_interp = np.interp(np.linspace(0, 1, n_pts), syn_x, syn)
            bio_interp = np.interp(np.linspace(0, 1, n_pts), bio_x, bio)

            ax2.scatter(syn_interp, bio_interp, c=colors[key], alpha=0.5, s=20)
            # Perfect correlation line
            lims = [0, 1]
            ax2.plot(lims, lims, "k--", alpha=0.3)
            ax2.set_xlabel("Synthetic")
            ax2.set_ylabel("Biological")
            ax2.set_title(f"{labels[key]} Scatter", fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-0.1, 1.1)
            ax2.set_ylim(-0.1, 1.1)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    @staticmethod
    def _estimate_exponents(
        x: np.ndarray, phi: np.ndarray, lz: np.ndarray, smf: np.ndarray
    ) -> dict:
        """
        Estimate critical exponents (slope near the steepest point).

        In a true phase transition, the order parameter diverges as
        |p - p_c|^(-α) near the critical point p_c. The exponent α
        characterizes the universality class of the transition.
        """
        exponents = {}
        for name, values in [("phi", phi), ("lz", lz), ("smf", smf)]:
            # Find the steepest point
            if len(values) < 3:
                exponents[name] = 0.0
                continue

            grad = np.gradient(values, x)
            peak_idx = np.argmax(np.abs(grad))

            # Fit a power law near the peak
            window = max(3, len(values) // 5)
            start = max(0, peak_idx - window)
            end = min(len(values), peak_idx + window)

            x_local = x[start:end] - x[peak_idx]
            v_local = values[start:end]

            # Avoid log(0)
            x_pos = np.abs(x_local) + 1e-10
            v_pos = np.abs(v_local - v_local.min()) + 1e-10

            try:
                coeffs = np.polyfit(np.log(x_pos), np.log(v_pos), 1)
                exponents[name] = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                exponents[name] = 0.0

        return exponents

    @staticmethod
    def _compare_exponents(syn: dict, bio: dict) -> float:
        """
        Compare critical exponents between synthetic and biological.

        If the exponent ratios are similar, the transitions belong to
        the same "universality class" — a strong validation result.
        """
        scores = []
        for key in ["phi", "lz", "smf"]:
            s, b = syn.get(key, 0), bio.get(key, 0)
            if abs(s) > 0.01 and abs(b) > 0.01:
                ratio = min(abs(s), abs(b)) / max(abs(s), abs(b))
                scores.append(ratio)
            else:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _convergence_score(phi: np.ndarray, lz: np.ndarray, smf: np.ndarray) -> float:
        """
        Measure how well the three measures converge (peak at the same point).
        """
        if len(phi) < 3:
            return 0.0

        grad_phi = np.abs(np.gradient(phi))
        grad_lz = np.abs(np.gradient(lz))
        grad_smf = np.abs(np.gradient(smf))

        peak_phi = np.argmax(grad_phi) / len(grad_phi)
        peak_lz = np.argmax(grad_lz) / len(grad_lz)
        peak_smf = np.argmax(grad_smf) / len(grad_smf)

        spread = np.std([peak_phi, peak_lz, peak_smf])
        return float(np.exp(-spread / 0.1))
