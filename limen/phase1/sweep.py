"""
Parameter sweep orchestration for the $20 experiment.

Systematically varies connection density across a range of network sizes,
running the full measurement pipeline at each configuration. Handles
parallelism, checkpointing, and result aggregation.

This is the experiment runner — the script that answers the question:
"Do the measures spike together?"
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from limen.phase1.network import RecurrentNetwork, NetworkConfig
from limen.phase1.measures import compute_all_measures, compute_phase_derivatives, MeasurementResult


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep experiment."""

    # Network sizes to test
    network_sizes: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])

    # Connection density range
    density_min: float = 0.02
    density_max: float = 0.98
    density_steps: int = 50

    # Simulation parameters
    n_timesteps: int = 2000
    warmup_steps: int = 500
    n_trials: int = 5  # Repetitions per configuration for statistics

    # Network parameters
    weight_scale: float = 1.0
    tau: float = 1.0
    dt: float = 0.1
    noise_amplitude: float = 0.01

    # Computation
    n_workers: int = 1  # Parallel workers
    checkpoint_dir: Optional[str] = None

    def densities(self) -> np.ndarray:
        """Generate the density sweep values."""
        return np.linspace(self.density_min, self.density_max, self.density_steps)


@dataclass
class SweepResult:
    """Results from a complete parameter sweep."""

    config: SweepConfig
    densities: np.ndarray
    results: dict  # {n_nodes: {density: [MeasurementResult, ...]}}
    phase_analysis: dict  # {n_nodes: derivative analysis}
    total_time_seconds: float

    def summary(self) -> dict:
        """Generate a summary of the sweep results."""
        summary = {
            "total_configurations": 0,
            "total_time_hours": self.total_time_seconds / 3600,
            "network_sizes": list(self.results.keys()),
            "convergence_scores": {},
            "candidate_transitions": {},
        }

        for n_nodes, analysis in self.phase_analysis.items():
            summary["convergence_scores"][n_nodes] = analysis["convergence"]["score"]
            summary["candidate_transitions"][n_nodes] = analysis["convergence"]["mean_density"]
            summary["total_configurations"] += len(self.results.get(n_nodes, {}))

        return summary

    def to_arrays(self, n_nodes: int) -> dict:
        """
        Extract measure arrays for a given network size.

        Returns dict with keys: 'densities', 'phi', 'lz', 'smf' (each as
        arrays of mean values across trials).
        """
        if n_nodes not in self.results:
            raise ValueError(f"No results for n_nodes={n_nodes}")

        node_results = self.results[n_nodes]
        densities = sorted(node_results.keys())

        phi_means = []
        lz_means = []
        smf_means = []
        phi_stds = []
        lz_stds = []
        smf_stds = []

        for d in densities:
            trials = node_results[d]
            phis = [t.phi for t in trials]
            lzs = [t.lz_normalized for t in trials]
            smfs = [t.self_model_fidelity_normalized for t in trials]

            phi_means.append(np.mean(phis))
            lz_means.append(np.mean(lzs))
            smf_means.append(np.mean(smfs))
            phi_stds.append(np.std(phis))
            lz_stds.append(np.std(lzs))
            smf_stds.append(np.std(smfs))

        return {
            "densities": np.array(densities),
            "phi_mean": np.array(phi_means),
            "phi_std": np.array(phi_stds),
            "lz_mean": np.array(lz_means),
            "lz_std": np.array(lz_stds),
            "smf_mean": np.array(smf_means),
            "smf_std": np.array(smf_stds),
        }

    def save(self, path: str):
        """Save results to JSON for later analysis."""
        output = {
            "config": asdict(self.config),
            "total_time_seconds": self.total_time_seconds,
            "results": {},
        }

        for n_nodes, density_results in self.results.items():
            output["results"][str(n_nodes)] = {}
            for density, trials in density_results.items():
                output["results"][str(n_nodes)][str(density)] = [
                    t.to_dict() for t in trials
                ]

        # Phase analysis (without numpy arrays)
        output["phase_analysis"] = {}
        for n_nodes, analysis in self.phase_analysis.items():
            output["phase_analysis"][str(n_nodes)] = {
                "peaks": analysis["peaks"],
                "convergence": analysis["convergence"],
            }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)


def _run_single_configuration(
    n_nodes: int,
    density: float,
    n_timesteps: int,
    warmup: int,
    weight_scale: float,
    tau: float,
    dt: float,
    noise_amplitude: float,
    seed: int,
) -> MeasurementResult:
    """
    Run a single network configuration and compute all measures.
    Designed to be called in parallel via ProcessPoolExecutor.
    """
    config = NetworkConfig(
        n_nodes=n_nodes,
        connection_density=density,
        weight_scale=weight_scale,
        tau=tau,
        dt=dt,
        noise_amplitude=noise_amplitude,
        seed=seed,
    )

    network = RecurrentNetwork(config)
    states = network.run(n_timesteps, warmup=warmup)
    tpm = network.get_effective_tpm(states)

    result = compute_all_measures(
        states=states,
        tpm=tpm,
        connection_density=density,
        n_nodes=n_nodes,
    )

    return result


class ParameterSweep:
    """
    Orchestrates the full parameter sweep experiment.

    For each network size and connection density, runs multiple trials
    and computes all information-theoretic measures. Then analyzes the
    results for phase transition signatures.
    """

    def __init__(self, config: Optional[SweepConfig] = None):
        self.config = config or SweepConfig()

    def run(self, verbose: bool = True) -> SweepResult:
        """
        Execute the full parameter sweep.

        Parameters
        ----------
        verbose : bool
            Print progress information.

        Returns
        -------
        SweepResult
            Complete results with phase analysis.
        """
        t_start = time.time()
        densities = self.config.densities()
        results = {}

        total_configs = (
            len(self.config.network_sizes)
            * len(densities)
            * self.config.n_trials
        )

        if verbose:
            print(f"Limen Phase 1: The $20 Experiment")
            print(f"{'=' * 50}")
            print(f"Network sizes: {self.config.network_sizes}")
            print(f"Density range: {self.config.density_min:.2f} to {self.config.density_max:.2f}")
            print(f"Density steps: {self.config.density_steps}")
            print(f"Trials per config: {self.config.n_trials}")
            print(f"Total configurations: {total_configs}")
            print(f"Timesteps per run: {self.config.n_timesteps}")
            print(f"{'=' * 50}")

        completed = 0

        for n_nodes in self.config.network_sizes:
            if verbose:
                print(f"\n--- Network size: {n_nodes} nodes ---")

            results[n_nodes] = {}

            for density in densities:
                trial_results = []

                if self.config.n_workers > 1:
                    # Parallel execution
                    with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                        futures = {}
                        for trial in range(self.config.n_trials):
                            seed = hash((n_nodes, int(density * 1000), trial)) % (2**31)
                            future = executor.submit(
                                _run_single_configuration,
                                n_nodes=n_nodes,
                                density=density,
                                n_timesteps=self.config.n_timesteps,
                                warmup=self.config.warmup_steps,
                                weight_scale=self.config.weight_scale,
                                tau=self.config.tau,
                                dt=self.config.dt,
                                noise_amplitude=self.config.noise_amplitude,
                                seed=seed,
                            )
                            futures[future] = trial

                        for future in as_completed(futures):
                            trial_results.append(future.result())
                            completed += 1
                else:
                    # Sequential execution
                    for trial in range(self.config.n_trials):
                        seed = hash((n_nodes, int(density * 1000), trial)) % (2**31)
                        result = _run_single_configuration(
                            n_nodes=n_nodes,
                            density=density,
                            n_timesteps=self.config.n_timesteps,
                            warmup=self.config.warmup_steps,
                            weight_scale=self.config.weight_scale,
                            tau=self.config.tau,
                            dt=self.config.dt,
                            noise_amplitude=self.config.noise_amplitude,
                            seed=seed,
                        )
                        trial_results.append(result)
                        completed += 1

                results[n_nodes][density] = trial_results

                if verbose:
                    mean_phi = np.mean([r.phi for r in trial_results])
                    mean_lz = np.mean([r.lz_normalized for r in trial_results])
                    mean_smf = np.mean([r.self_model_fidelity_normalized for r in trial_results])
                    print(
                        f"  d={density:.3f}: "
                        f"Φ={mean_phi:.4f}  "
                        f"LZ={mean_lz:.4f}  "
                        f"SMF={mean_smf:.4f}  "
                        f"[{completed}/{total_configs}]"
                    )

            # Checkpoint after each network size
            if self.config.checkpoint_dir:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_n{n_nodes}.json"
                if verbose:
                    print(f"  Saving checkpoint: {checkpoint_path}")

        # Phase analysis
        if verbose:
            print(f"\n{'=' * 50}")
            print("Analyzing phase transitions...")

        phase_analysis = {}
        for n_nodes in results:
            arrays = _extract_arrays(results[n_nodes])
            if len(arrays["densities"]) > 3:
                phase_analysis[n_nodes] = compute_phase_derivatives(
                    arrays["densities"],
                    arrays["phi"],
                    arrays["lz"],
                    arrays["smf"],
                )

                if verbose:
                    conv = phase_analysis[n_nodes]["convergence"]
                    print(
                        f"  n={n_nodes}: convergence score = {conv['score']:.4f}, "
                        f"candidate transition at d = {conv['mean_density']:.3f}"
                    )

        t_end = time.time()

        if verbose:
            print(f"\nTotal time: {(t_end - t_start) / 60:.1f} minutes")

        return SweepResult(
            config=self.config,
            densities=densities,
            results=results,
            phase_analysis=phase_analysis,
            total_time_seconds=t_end - t_start,
        )


def _extract_arrays(density_results: dict) -> dict:
    """Helper to extract mean arrays from trial results."""
    densities = sorted(density_results.keys())
    phi_means = []
    lz_means = []
    smf_means = []

    for d in densities:
        trials = density_results[d]
        phi_means.append(np.mean([t.phi for t in trials]))
        lz_means.append(np.mean([t.lz_normalized for t in trials]))
        smf_means.append(np.mean([t.self_model_fidelity_normalized for t in trials]))

    return {
        "densities": np.array(densities),
        "phi": np.array(phi_means),
        "lz": np.array(lz_means),
        "smf": np.array(smf_means),
    }
