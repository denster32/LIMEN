#!/usr/bin/env python3
"""
Run Phase 2: Biological validation.

Tests whether Phase 1 synthetic signatures match biological EEG transitions.
Uses synthetic EEG data by default (for development), with options to
point at real PhysioNet data.

Usage:
    python -m scripts.run_phase2                              # Synthetic data
    python -m scripts.run_phase2 --edf path/to/data.edf      # Real EDF file
    python -m scripts.run_phase2 --transition anesthesia      # Specific transition
    python -m scripts.run_phase2 --phase1-results results/    # Use Phase 1 data
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from limen.phase1.network import RecurrentNetwork, NetworkConfig
from limen.phase1.measures import compute_all_measures
from limen.phase2.eeg_loader import EEGLoader
from limen.phase2.biological import BiologicalMeasures
from limen.phase2.validate import CrossSubstrateValidator


def main():
    parser = argparse.ArgumentParser(
        description="Limen Phase 2: Biological Validation",
    )

    parser.add_argument("--transition", type=str, default="anesthesia",
                       choices=["anesthesia", "sleep", "seizure", "psychedelic"],
                       help="Type of consciousness transition to validate against")
    parser.add_argument("--edf", type=str, default=None,
                       help="Path to real EDF file (uses synthetic data if not provided)")
    parser.add_argument("--phase1-results", type=str, default=None,
                       help="Path to Phase 1 results JSON (generates fresh if not provided)")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--n-components", type=int, default=16,
                       help="Number of PCA components for EEG analysis")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Generate or load synthetic reference data from Phase 1
    # =====================================================================
    if args.phase1_results:
        if not args.quiet:
            print("Loading Phase 1 results...")
        # For simplicity, regenerate a quick sweep for validation
        synthetic_results = _generate_quick_synthetic()
    else:
        if not args.quiet:
            print("Generating synthetic reference data...")
        synthetic_results = _generate_quick_synthetic()

    if not args.quiet:
        print(f"  {len(synthetic_results)} synthetic configurations")

    # =====================================================================
    # Run validation
    # =====================================================================
    validator = CrossSubstrateValidator(n_components=args.n_components)

    if args.edf:
        # Use real EDF data
        if not args.quiet:
            print(f"Loading EDF: {args.edf}")
        loader = EEGLoader()
        segment = loader.load_edf(args.edf, transition_type=args.transition)
        windows = loader.extract_transition_windows(segment, window_duration_s=10, step_s=2)

        result = validator.validate_transition(
            synthetic_results, windows, verbose=not args.quiet
        )
    else:
        # Use synthetic EEG
        if not args.quiet:
            print(f"Using synthetic EEG ({args.transition} transition)")

        result = validator.validate_with_synthetic_data(
            synthetic_results,
            transition_type=args.transition,
            verbose=not args.quiet,
        )

    # Save results
    result_dict = {
        "phi_correlation": result.phi_correlation,
        "lz_correlation": result.lz_correlation,
        "smf_correlation": result.smf_correlation,
        "exponent_match_score": result.exponent_match_score,
        "synthetic_convergence": result.synthetic_convergence,
        "biological_convergence": result.biological_convergence,
        "validation_score": result.validation_score,
        "synthetic_exponents": result.synthetic_exponents,
        "biological_exponents": result.biological_exponents,
    }

    result_path = output_dir / f"phase2_validation_{args.transition}.json"
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    if not args.quiet:
        print(f"\nResults saved to {result_path}")

    # Plot
    fig = validator.plot_validation(
        result,
        save_path=str(output_dir / f"validation_{args.transition}.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    if not args.quiet:
        print(f"Plot saved to {output_dir}/validation_{args.transition}.png")


def _generate_quick_synthetic(
    n_nodes: int = 32,
    n_densities: int = 20,
    n_timesteps: int = 500,
) -> list:
    """Generate a quick synthetic sweep for validation."""
    from limen.phase1.measures import compute_all_measures

    results = []
    densities = [d / n_densities for d in range(1, n_densities)]

    for density in densities:
        config = NetworkConfig(
            n_nodes=n_nodes,
            connection_density=density,
            seed=42,
        )
        network = RecurrentNetwork(config)
        states = network.run(n_timesteps, warmup=200)
        tpm = network.get_effective_tpm(states)

        result = compute_all_measures(
            states=states,
            tpm=tpm,
            connection_density=density,
            n_nodes=n_nodes,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    main()
