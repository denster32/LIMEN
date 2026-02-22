#!/usr/bin/env python3
"""
Run the Phase 1 experiment: sweep connection density across network sizes
and measure Φ, LZ complexity, and self-model fidelity.

This is the $20 experiment — the one that tests whether the three measures
converge at the same critical density.

Usage:
    python -m scripts.run_phase1                    # Quick test (8-32 nodes)
    python -m scripts.run_phase1 --full             # Full experiment (8-128 nodes)
    python -m scripts.run_phase1 --sizes 8 16 32    # Custom sizes
    python -m scripts.run_phase1 --output results/  # Custom output directory
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from limen.phase1.sweep import ParameterSweep, SweepConfig
from limen.phase1.visualize import plot_phase_transition, plot_sweep_results, plot_multiscale_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Limen Phase 1: The $20 Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test run:       python -m scripts.run_phase1
  Full experiment:      python -m scripts.run_phase1 --full
  Custom network sizes: python -m scripts.run_phase1 --sizes 8 16 32 64
  More density points:  python -m scripts.run_phase1 --density-steps 100
        """,
    )

    parser.add_argument("--full", action="store_true",
                       help="Run full experiment (8-128 nodes, 50 density steps, 5 trials)")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                       help="Network sizes to test")
    parser.add_argument("--density-steps", type=int, default=None,
                       help="Number of density steps")
    parser.add_argument("--trials", type=int, default=None,
                       help="Number of trials per configuration")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Simulation timesteps per run")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results and plots")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")

    args = parser.parse_args()

    # Build configuration
    if args.full:
        config = SweepConfig(
            network_sizes=[8, 16, 32, 64, 128],
            density_steps=50,
            n_trials=5,
            n_timesteps=2000,
            warmup_steps=500,
        )
    else:
        config = SweepConfig(
            network_sizes=args.sizes or [8, 16, 32],
            density_steps=args.density_steps or 25,
            n_trials=args.trials or 3,
            n_timesteps=args.timesteps or 1000,
            warmup_steps=300,
        )

    if args.sizes:
        config.network_sizes = args.sizes
    if args.density_steps:
        config.density_steps = args.density_steps
    if args.trials:
        config.n_trials = args.trials
    if args.timesteps:
        config.n_timesteps = args.timesteps

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir = str(output_dir / "checkpoints")

    # Run the sweep
    sweep = ParameterSweep(config)
    result = sweep.run(verbose=not args.quiet)

    # Save results
    result_path = output_dir / "phase1_results.json"
    result.save(str(result_path))
    if not args.quiet:
        print(f"\nResults saved to {result_path}")

    # Generate plots
    if not args.quiet:
        print("\nGenerating plots...")

    # Individual network size plots
    for n_nodes in config.network_sizes:
        if n_nodes in result.results:
            fig = plot_phase_transition(
                result, n_nodes,
                save_path=str(output_dir / f"phase_transition_n{n_nodes}.png"),
            )
            plt_close(fig)

    # Comprehensive sweep plot
    fig = plot_sweep_results(
        result,
        save_path=str(output_dir / "sweep_results.png"),
    )
    plt_close(fig)

    # Multiscale analysis for largest network
    largest = max(config.network_sizes)
    if largest in result.results:
        fig = plot_multiscale_analysis(
            result, largest,
            save_path=str(output_dir / f"multiscale_n{largest}.png"),
        )
        plt_close(fig)

    # Print summary
    summary = result.summary()
    if not args.quiet:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total time: {summary['total_time_hours']:.2f} hours")
        print(f"Network sizes: {summary['network_sizes']}")
        print(f"\nConvergence scores (do the measures spike together?):")
        for n, score in summary["convergence_scores"].items():
            transition = summary["candidate_transitions"][n]
            verdict = "YES" if score > 0.5 else "MAYBE" if score > 0.2 else "NO"
            print(f"  n={n:4d}: score={score:.3f} ({verdict}), transition at d={transition:.3f}")

        best_size = max(summary["convergence_scores"], key=summary["convergence_scores"].get)
        best_score = summary["convergence_scores"][best_size]
        print(f"\nBest convergence: n={best_size} (score={best_score:.3f})")

        if best_score > 0.5:
            print("\n→ The measures converge. A candidate phase transition has been found.")
            print(f"  Critical density: d ≈ {summary['candidate_transitions'][best_size]:.3f}")
        elif best_score > 0.2:
            print("\n→ Partial convergence. More data or larger networks may be needed.")
        else:
            print("\n→ No convergence detected. The phase transition hypothesis is not supported")
            print("  by this data. Consider adjusting parameters or the model.")

        print(f"\nPlots saved to {output_dir}/")
        print("=" * 60)


def plt_close(fig):
    """Close a matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
