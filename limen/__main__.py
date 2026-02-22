"""
Limen — Consciousness Phase Transition Research

Quick demo: python -m limen
Full experiment: python -m scripts.run_phase1
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from limen.core.phi import approximate_phi
from limen.core.complexity import normalized_lz_complexity, lz_complexity_from_states
from limen.core.information import self_model_fidelity
from limen.phase1.network import RecurrentNetwork, NetworkConfig
from limen.phase1.measures import compute_all_measures


def main():
    console = Console()

    console.print(
        Panel.fit(
            "[bold]Limen[/bold] — Consciousness Phase Transition Detector\n"
            "[dim]Latin: threshold. The boundary you cross when you step through a doorway.[/dim]",
            border_style="bright_cyan",
        )
    )

    console.print("\n[bold]Quick Demo:[/bold] Measuring three consciousness candidates\n")
    console.print("Building recurrent networks at different connection densities...\n")

    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_nodes = 32
    timesteps = 1000

    table = Table(title=f"Consciousness Measures ({n_nodes} nodes, {timesteps} timesteps)")
    table.add_column("Density", style="cyan", justify="right")
    table.add_column("Φ (Integration)", style="red", justify="right")
    table.add_column("LZ (Complexity)", style="green", justify="right")
    table.add_column("SMF (Self-Model)", style="yellow", justify="right")
    table.add_column("Spectral ρ", style="dim", justify="right")

    for d in densities:
        config = NetworkConfig(
            n_nodes=n_nodes,
            connection_density=d,
            noise_amplitude=0.05,
            seed=42,
        )
        net = RecurrentNetwork(config)
        states = net.run(timesteps, warmup=200)
        tpm = net.get_effective_tpm(states)
        result = compute_all_measures(states, tpm, d)
        stats = net.get_network_stats()

        table.add_row(
            f"{d:.1f}",
            f"{result.phi:.4f}",
            f"{result.lz_normalized:.4f}",
            f"{result.self_model_fidelity_normalized:.4f}",
            f"{stats['spectral_radius']:.3f}",
        )

    console.print(table)

    console.print(
        "\n[bold]What to look for:[/bold] If all three measures show a "
        "coordinated sharp change at the same density, that's a candidate "
        "phase transition — the system crosses from processing to something "
        "more integrated.\n"
    )

    console.print("[dim]Run the full experiment:[/dim]")
    console.print("  python -m scripts.run_phase1 --sizes 32 64 --density-steps 30 --trials 5\n")
    console.print("[dim]Launch the consciousness server:[/dim]")
    console.print("  python -m scripts.run_server --persist ./state/\n")
    console.print("[dim]Run tests:[/dim]")
    console.print("  pytest tests/\n")


if __name__ == "__main__":
    main()
