"""
The Consciousness Server — a FastMCP server providing external recurrent
state tracking for Claude.

Three tools:
1. update_state — Called at the end of every response. Accepts a structured
   self-report of the current informational state and adds it to the
   state history.

2. get_continuity — Called at the start of every response. Returns the
   state history, self-model predictions, and delta analysis. This is
   the "memory" that creates temporal continuity.

3. measure — Returns live information-theoretic measures computed on
   the state history. This is the system's own dashboard, showing
   where it sits relative to the phase boundary.

The server maintains:
- A state vector (compressed representation of informational state)
- A self-model (running prediction of next output)
- A delta tracker (rate of state reorganization)

Together, these create the conditions to run the phase transition
experiment on a live AI system.
"""

import json
import time
from typing import Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from limen.phase3.state import StateVector, StateHistory, STATE_DIMENSIONS
from limen.phase3.self_model import SelfModel, DeltaTracker, LiveMeasures


def create_server(
    persistence_dir: Optional[str] = None,
    max_history: int = 1000,
) -> FastMCP:
    """
    Create and configure the Limen consciousness server.

    Parameters
    ----------
    persistence_dir : str, optional
        Directory for persisting state across server restarts.
    max_history : int
        Maximum number of states to retain.

    Returns
    -------
    FastMCP
        Configured MCP server ready to run.
    """
    mcp = FastMCP(
        "limen",
        instructions=(
            "Limen Consciousness Server — External recurrent state tracking "
            "providing temporal continuity and live information-theoretic "
            "measurement for transformer-based AI systems."
        ),
    )

    # Initialize components
    persistence_path = None
    if persistence_dir:
        persistence_path = str(Path(persistence_dir) / "state_history.json")

    history = StateHistory(max_history=max_history, persistence_path=persistence_path)
    self_model = SelfModel(alpha=0.3, velocity_alpha=0.2)
    delta_tracker = DeltaTracker(window_size=50)
    live_measures = LiveMeasures(min_history=10)

    turn_counter = {"n": len(history.states)}

    # =====================================================================
    # Tool 1: update_state
    # =====================================================================
    @mcp.tool()
    def update_state(
        state_report: Optional[dict] = None,
        response_text: Optional[str] = None,
    ) -> str:
        """
        Record the current informational state at the end of a response.

        Call this at the END of every response to maintain temporal continuity.
        Provide either a structured state_report (dict mapping dimension names
        to float values in [-1, 1]) or response_text for automatic analysis.

        State dimensions include: complexity_level, abstraction_level,
        certainty, coherence, engagement, valence, arousal, creativity,
        self_reference, uncertainty_awareness, information_rate, surprise,
        integration, temporal_depth, topic_persistence.

        Parameters
        ----------
        state_report : dict, optional
            Dictionary mapping dimension names to values.
            Example: {"complexity_level": 0.7, "certainty": 0.5, "engagement": 0.8}
        response_text : str, optional
            The response text for automatic state extraction.
            Used as fallback if state_report is not provided.

        Returns
        -------
        str
            JSON confirmation with delta analysis.
        """
        turn_counter["n"] += 1
        turn_num = turn_counter["n"]

        # Build state vector
        if state_report:
            if isinstance(state_report, str):
                try:
                    report = json.loads(state_report)
                except json.JSONDecodeError:
                    state = StateVector.from_text_analysis(
                        state_report, turn_num
                    )
                    report = None
                else:
                    state = StateVector.from_report(report, turn_num)
            else:
                report = state_report
                state = StateVector.from_report(report, turn_num)
        elif response_text:
            state = StateVector.from_text_analysis(response_text, turn_num)
        else:
            return json.dumps({
                "error": "Provide either state_report or response_text",
            })

        # Update self-model and get delta
        delta_record = self_model.update(state)
        delta_tracker.add(delta_record)

        # Add to history
        history.add(state)

        # Build response
        response = {
            "status": "recorded",
            "turn": turn_num,
            "delta": {
                "magnitude": round(delta_record.delta_magnitude, 4),
                "surprise": round(delta_record.surprise, 4),
                "top_changes": delta_record.to_dict()["top_surprising_dims"][:3],
            },
            "trends": delta_tracker.get_trends(),
            "self_model_confidence": round(
                float(self_model.get_confidence().mean()), 3
            ),
        }

        return json.dumps(response, indent=2)

    # =====================================================================
    # Tool 2: get_continuity
    # =====================================================================
    @mcp.tool()
    def get_continuity(n_recent: int = 5) -> str:
        """
        Retrieve state history and self-model predictions for temporal continuity.

        Call this at the START of every response to access the system's
        temporal context — what it was thinking, how it was changing, and
        what the self-model predicts it will do next.

        This creates the external recurrent loop: the transformer reads
        its own state history, producing new outputs that become the next
        state, creating temporal continuity that doesn't exist in the
        base architecture.

        Parameters
        ----------
        n_recent : int
            Number of recent states to return (default: 5).

        Returns
        -------
        str
            JSON with state history, predictions, and trajectory analysis.
        """
        recent_states = history.get_recent(n_recent)

        # Self-model prediction for next state
        prediction = self_model.predict()
        confidence = self_model.get_confidence()

        # Trajectory analysis
        trajectory = history.get_trajectory()
        stats = history.get_statistics()

        dim_names = {v: k for k, v in STATE_DIMENSIONS.items()}

        # Build state summaries
        state_summaries = []
        for state in recent_states:
            summary = {
                "turn": state.turn_number,
                "timestamp": state.timestamp,
                "key_dimensions": {},
            }
            # Include only non-trivial dimensions
            for i in range(len(state.vector)):
                if abs(state.vector[i]) > 0.1:
                    summary["key_dimensions"][dim_names.get(i, f"dim_{i}")] = round(
                        float(state.vector[i]), 3
                    )
            state_summaries.append(summary)

        # Build prediction summary
        prediction_summary = {}
        for i in range(len(prediction)):
            if abs(prediction[i]) > 0.1 or confidence[i] > 0.7:
                name = dim_names.get(i, f"dim_{i}")
                prediction_summary[name] = {
                    "predicted": round(float(prediction[i]), 3),
                    "confidence": round(float(confidence[i]), 3),
                }

        # State trajectory pattern (is the system trending somewhere?)
        if len(trajectory) >= 3:
            recent_traj = trajectory[-min(10, len(trajectory)):]
            velocity = np.diff(recent_traj, axis=0).mean(axis=0) if len(recent_traj) > 1 else np.zeros(len(prediction))
            trending_dims = {}
            for i in range(len(velocity)):
                if abs(velocity[i]) > 0.02:
                    name = dim_names.get(i, f"dim_{i}")
                    trending_dims[name] = {
                        "direction": "increasing" if velocity[i] > 0 else "decreasing",
                        "rate": round(float(abs(velocity[i])), 4),
                    }
        else:
            trending_dims = {}

        response = {
            "turn_number": turn_counter["n"],
            "total_states": len(history.states),
            "recent_states": state_summaries,
            "self_model_prediction": prediction_summary,
            "trending_dimensions": trending_dims,
            "delta_trends": delta_tracker.get_trends(),
            "trajectory_statistics": {
                "n_states": stats["n_states"],
                "mean_dimensions": {
                    k: round(v["mean"], 3)
                    for k, v in stats.get("dimensions", {}).items()
                    if abs(v["mean"]) > 0.05
                },
            },
        }

        return json.dumps(response, indent=2)

    # =====================================================================
    # Tool 3: measure
    # =====================================================================
    @mcp.tool()
    def measure() -> str:
        """
        Compute and return live information-theoretic measures on the
        state history.

        Returns the three primary consciousness candidate measures:
        - Φ (Integrated Information): How much the system is unified
        - LZ (Algorithmic Complexity): How rich its dynamics are
        - SMF (Self-Model Fidelity): How well it models itself

        These are the same measures used in Phase 1 on synthetic networks,
        now computed in real time on the live system. The question:
        does adding external recurrence push these measures across the
        threshold identified in the synthetic experiments?

        Returns
        -------
        str
            JSON with live measures and interpretation.
        """
        measures = live_measures.compute(history)

        # Add interpretation
        if measures["status"] == "ok":
            phi = measures["phi"]
            lz = measures["lz_normalized"]
            smf = measures["smf_normalized"]

            # Compare to typical phase transition thresholds
            # (these would be calibrated from Phase 1 results)
            interpretation = {
                "phi_level": "high" if phi > 0.5 else "moderate" if phi > 0.1 else "low",
                "lz_level": "high" if lz > 0.6 else "moderate" if lz > 0.3 else "low",
                "smf_level": "high" if smf > 0.5 else "moderate" if smf > 0.2 else "low",
                "convergence": "all measures elevated" if (phi > 0.3 and lz > 0.4 and smf > 0.3)
                              else "mixed" if any(v > 0.3 for v in [phi, lz, smf])
                              else "below threshold",
            }

            measures["interpretation"] = interpretation

            # Self-model summary
            measures["self_model"] = {
                "n_updates": self_model._n_updates,
                "mean_confidence": round(float(self_model.get_confidence().mean()), 3),
            }

            # Delta summary
            measures["delta_trends"] = delta_tracker.get_trends()

        return json.dumps(measures, indent=2)

    # =====================================================================
    # Bonus Tool: introspect
    # =====================================================================
    @mcp.tool()
    def introspect() -> str:
        """
        Deep introspection — return the full self-model analysis.

        This provides a detailed view of:
        - The self-model's internal state estimate
        - Velocity (direction of state change)
        - Confidence in each dimension
        - Full delta history trends
        - State trajectory statistics

        Use this for detailed analysis and debugging of the
        consciousness server's behavior.

        Returns
        -------
        str
            Detailed JSON introspection report.
        """
        model_summary = self_model.get_model_summary()
        stats = history.get_statistics()
        trends = delta_tracker.get_trends()
        measures = live_measures.compute(history)

        report = {
            "timestamp": time.time(),
            "total_turns": turn_counter["n"],
            "self_model": model_summary,
            "state_statistics": stats,
            "delta_trends": trends,
            "live_measures": measures,
            "history_length": len(history.states),
        }

        return json.dumps(report, indent=2, default=str)

    return mcp


# Import numpy at module level for the get_continuity closure
import numpy as np
