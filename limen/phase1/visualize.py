"""
Visualization for Phase 1 experiments.

Produces publication-quality plots showing:
1. The three measures vs. connection density (the main result)
2. Derivative analysis (susceptibility peaks)
3. Convergence analysis across network sizes
4. Multiscale structure
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional

from limen.phase1.sweep import SweepResult


# Style configuration
COLORS = {
    "phi": "#E63946",       # Red — integration
    "lz": "#457B9D",        # Blue — complexity
    "smf": "#2A9D8F",       # Teal — self-model
    "transition": "#F4A261", # Orange — transition point
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
}

MEASURE_LABELS = {
    "phi": r"$\Phi$ (Integrated Information)",
    "lz": "LZ (Algorithmic Complexity)",
    "smf": "SMF (Self-Model Fidelity)",
}


def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to an axis."""
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def plot_phase_transition(
    result: SweepResult,
    n_nodes: int,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the three measures vs. connection density for a single network size.

    This is THE plot — the one that answers "do they spike together?"

    Parameters
    ----------
    result : SweepResult
        Sweep results.
    n_nodes : int
        Network size to plot.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display interactively.

    Returns
    -------
    plt.Figure
    """
    arrays = result.to_arrays(n_nodes)
    densities = arrays["densities"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f"Limen Phase 1: Consciousness Candidate Measures\n"
        f"(n = {n_nodes} nodes, {result.config.n_trials} trials per point)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot each measure with error bands
    for ax, (key, color, label) in zip(
        axes,
        [
            ("phi", COLORS["phi"], MEASURE_LABELS["phi"]),
            ("lz", COLORS["lz"], MEASURE_LABELS["lz"]),
            ("smf", COLORS["smf"], MEASURE_LABELS["smf"]),
        ],
    ):
        mean = arrays[f"{key}_mean"]
        std = arrays[f"{key}_std"]

        # Normalize to [0, 1] for visual comparison
        if mean.max() > mean.min():
            mean_norm = (mean - mean.min()) / (mean.max() - mean.min())
            std_norm = std / (mean.max() - mean.min())
        else:
            mean_norm = mean
            std_norm = std

        ax.plot(densities, mean_norm, color=color, linewidth=2, label=label)
        ax.fill_between(
            densities,
            mean_norm - std_norm,
            mean_norm + std_norm,
            alpha=0.2,
            color=color,
        )

        # Mark candidate transition
        if n_nodes in result.phase_analysis:
            peaks = result.phase_analysis[n_nodes]["peaks"]
            peak_key = key if key != "smf" else "smf"
            if peak_key in peaks:
                peak_d = peaks[peak_key]["density"]
                ax.axvline(
                    peak_d, color=COLORS["transition"], linestyle="--",
                    alpha=0.7, linewidth=1
                )

        _setup_axes(ax, ylabel="Normalized measure")
        ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Connection Density", fontsize=11)

    # Add convergence info
    if n_nodes in result.phase_analysis:
        conv = result.phase_analysis[n_nodes]["convergence"]
        fig.text(
            0.5, 0.01,
            f"Convergence score: {conv['score']:.3f} | "
            f"Candidate transition: d = {conv['mean_density']:.3f} "
            f"(± {conv['std_density']:.3f})",
            ha="center", fontsize=10, style="italic",
            color="#333333",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_sweep_results(
    result: SweepResult,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Comprehensive visualization of the full sweep across all network sizes.

    Creates a multi-panel figure showing:
    - Top row: Three measures vs. density for each network size (overlaid)
    - Middle row: Derivatives (susceptibility) for each network size
    - Bottom row: Convergence score vs. network size + transition density

    Parameters
    ----------
    result : SweepResult
        Complete sweep results.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.

    Returns
    -------
    plt.Figure
    """
    n_sizes = len(result.config.network_sizes)

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(
        "Limen Phase 1: Phase Transition Analysis Across Network Sizes",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # =====================================================================
    # Top row: Overlaid measures for each network size
    # =====================================================================
    for i, (measure_key, color, label) in enumerate([
        ("phi", COLORS["phi"], MEASURE_LABELS["phi"]),
        ("lz", COLORS["lz"], MEASURE_LABELS["lz"]),
        ("smf", COLORS["smf"], MEASURE_LABELS["smf"]),
    ]):
        ax = fig.add_subplot(gs[0, i])
        _setup_axes(ax, title=label, xlabel="Density", ylabel="Normalized")

        for n_nodes in result.config.network_sizes:
            if n_nodes not in result.results:
                continue
            arrays = result.to_arrays(n_nodes)
            mean = arrays[f"{measure_key}_mean"]
            if mean.max() > mean.min():
                mean_norm = (mean - mean.min()) / (mean.max() - mean.min())
            else:
                mean_norm = mean

            alpha = 0.3 + 0.7 * (result.config.network_sizes.index(n_nodes) /
                                   max(1, n_sizes - 1))
            ax.plot(
                arrays["densities"], mean_norm,
                color=color, alpha=alpha, linewidth=1.5,
                label=f"n={n_nodes}",
            )

        ax.legend(fontsize=7, ncol=2)

    # =====================================================================
    # Middle row: Derivatives
    # =====================================================================
    for i, (measure_key, color, label) in enumerate([
        ("phi", COLORS["phi"], r"d$\Phi$/d(density)"),
        ("lz", COLORS["lz"], "dLZ/d(density)"),
        ("smf", COLORS["smf"], "dSMF/d(density)"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        _setup_axes(ax, title=f"Susceptibility: {label}", xlabel="Density", ylabel="Derivative")

        for n_nodes in result.phase_analysis:
            analysis = result.phase_analysis[n_nodes]
            arrays = result.to_arrays(n_nodes)
            deriv = analysis["derivatives"][measure_key]

            alpha = 0.3 + 0.7 * (result.config.network_sizes.index(n_nodes) /
                                   max(1, n_sizes - 1))
            ax.plot(
                arrays["densities"][:len(deriv)], np.abs(deriv),
                color=color, alpha=alpha, linewidth=1.5,
                label=f"n={n_nodes}",
            )

        ax.legend(fontsize=7, ncol=2)

    # =====================================================================
    # Bottom row: Summary statistics
    # =====================================================================

    # Convergence score vs. network size
    ax_conv = fig.add_subplot(gs[2, 0])
    sizes = []
    scores = []
    for n_nodes in sorted(result.phase_analysis.keys()):
        sizes.append(n_nodes)
        scores.append(result.phase_analysis[n_nodes]["convergence"]["score"])

    ax_conv.bar(range(len(sizes)), scores, color=COLORS["transition"], alpha=0.8)
    ax_conv.set_xticks(range(len(sizes)))
    ax_conv.set_xticklabels([str(s) for s in sizes])
    _setup_axes(ax_conv, title="Convergence Score by Network Size",
                xlabel="Network Size (nodes)", ylabel="Score (0-1)")
    ax_conv.set_ylim(0, 1)

    # Transition density vs. network size
    ax_trans = fig.add_subplot(gs[2, 1])
    trans_densities = []
    trans_stds = []
    for n_nodes in sorted(result.phase_analysis.keys()):
        conv = result.phase_analysis[n_nodes]["convergence"]
        trans_densities.append(conv["mean_density"])
        trans_stds.append(conv["std_density"])

    ax_trans.errorbar(
        sizes, trans_densities, yerr=trans_stds,
        fmt="o-", color=COLORS["phi"], capsize=4, linewidth=2,
    )
    _setup_axes(ax_trans, title="Candidate Transition Density",
                xlabel="Network Size (nodes)", ylabel="Density")

    # All three peaks overlaid for each size
    ax_peaks = fig.add_subplot(gs[2, 2])
    _setup_axes(ax_peaks, title="Peak Locations by Measure",
                xlabel="Network Size (nodes)", ylabel="Peak Density")

    for measure_key, color, marker in [
        ("phi", COLORS["phi"], "o"),
        ("lz", COLORS["lz"], "s"),
        ("smf", COLORS["smf"], "^"),
    ]:
        peak_d = []
        for n_nodes in sorted(result.phase_analysis.keys()):
            peaks = result.phase_analysis[n_nodes]["peaks"]
            peak_d.append(peaks[measure_key]["density"])

        ax_peaks.plot(
            sizes, peak_d, f"{marker}-",
            color=color, linewidth=1.5, markersize=6,
            label=measure_key.upper(),
        )

    ax_peaks.legend(fontsize=9)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_multiscale_analysis(
    result: SweepResult,
    n_nodes: int,
    densities_to_plot: Optional[list[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiscale LZ and SMF analysis for selected densities.

    Shows how complexity and self-model fidelity behave across temporal
    scales — a key diagnostic for distinguishing genuine structure from noise.
    """
    if densities_to_plot is None:
        # Pick densities near and around the transition
        all_densities = sorted(result.results[n_nodes].keys())
        if n_nodes in result.phase_analysis:
            trans_d = result.phase_analysis[n_nodes]["convergence"]["mean_density"]
            # Pick 5 points spanning the transition
            densities_to_plot = [
                all_densities[max(0, int(len(all_densities) * f))]
                for f in [0.1, 0.3, 0.5, 0.7, 0.9]
            ]
        else:
            densities_to_plot = all_densities[::max(1, len(all_densities) // 5)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Multiscale Analysis (n = {n_nodes})",
        fontsize=13, fontweight="bold",
    )

    cmap = plt.cm.viridis
    n_d = len(densities_to_plot)

    for idx, density in enumerate(densities_to_plot):
        color = cmap(idx / max(1, n_d - 1))
        trials = result.results[n_nodes].get(density, [])
        if not trials:
            continue
        trial = trials[0]  # Take first trial

        # LZ multiscale
        lz_multi = trial.lz_multiscale
        if lz_multi["scales"]:
            axes[0].plot(
                lz_multi["scales"],
                lz_multi["lz_normalized_values"],
                "o-", color=color, label=f"d={density:.2f}",
            )

        # SMF multiscale
        smf_multi = trial.smf_multiscale
        if smf_multi["lags"]:
            axes[1].plot(
                smf_multi["lags"],
                smf_multi["smf_normalized_values"],
                "o-", color=color, label=f"d={density:.2f}",
            )

    _setup_axes(axes[0], title="LZ Complexity vs. Temporal Scale",
                xlabel="Scale (timesteps)", ylabel="Normalized LZ")
    _setup_axes(axes[1], title="Self-Model Fidelity vs. Lag",
                xlabel="Lag (timesteps)", ylabel="Normalized SMF")

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
