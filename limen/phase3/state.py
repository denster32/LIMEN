"""
State vector management for the consciousness server.

The state vector is a compressed representation of Claude's informational
state at each interaction. Rather than storing raw text, we maintain a
fixed-dimensional vector that captures the essential features of each
exchange — topics, emotional valence, complexity, coherence, etc.

The state history provides the temporal continuity that transformers
lack. By feeding this back to Claude at the start of each response,
we create an external recurrent loop — the precondition for the
phase transition experiment on a live AI system.
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# State vector dimensions and their semantic meanings
STATE_DIMENSIONS = {
    # Cognitive state (dims 0-7)
    "topic_embedding_0": 0,
    "topic_embedding_1": 1,
    "topic_embedding_2": 2,
    "topic_embedding_3": 3,
    "complexity_level": 4,       # How complex is the current exchange
    "abstraction_level": 5,      # Concrete (0) to abstract (1)
    "certainty": 6,              # How confident in current reasoning
    "coherence": 7,              # Internal consistency of thought

    # Affective/engagement state (dims 8-11)
    "engagement": 8,             # Level of engagement with topic
    "valence": 9,                # Negative (-1) to positive (1)
    "arousal": 10,               # Calm (0) to activated (1)
    "creativity": 11,            # Routine (0) to creative (1)

    # Meta-cognitive state (dims 12-15)
    "self_reference": 12,        # How much discussing own processes
    "uncertainty_awareness": 13, # Awareness of own limitations
    "model_of_user": 14,        # Richness of user model
    "task_progress": 15,         # Progress on current task

    # Information dynamics (dims 16-19)
    "information_rate": 16,      # Rate of new information processing
    "redundancy": 17,            # Repetition of prior content
    "surprise": 18,              # Divergence from expected trajectory
    "integration": 19,           # Cross-referencing across context

    # Temporal features (dims 20-23)
    "temporal_depth": 20,        # How far back current context reaches
    "topic_persistence": 21,     # Stability of current topic
    "state_velocity": 22,        # Rate of state change
    "state_acceleration": 23,    # Rate of change of velocity
}

STATE_DIM = len(STATE_DIMENSIONS)


@dataclass
class StateVector:
    """
    A single state snapshot — Claude's informational state at one moment.
    """

    vector: np.ndarray  # Shape (STATE_DIM,)
    timestamp: float
    turn_number: int
    raw_report: dict = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: dict, turn_number: int) -> "StateVector":
        """
        Construct a state vector from a structured self-report.

        The report is a dictionary mapping dimension names to values.
        Missing dimensions are set to 0. Values are clipped to [-1, 1].
        """
        vector = np.zeros(STATE_DIM)

        for dim_name, idx in STATE_DIMENSIONS.items():
            if dim_name in report:
                vector[idx] = np.clip(float(report[dim_name]), -1.0, 1.0)

        return cls(
            vector=vector,
            timestamp=time.time(),
            turn_number=turn_number,
            raw_report=report,
        )

    @classmethod
    def from_text_analysis(cls, text: str, turn_number: int) -> "StateVector":
        """
        Construct a state vector by analyzing response text.

        Uses simple heuristics to estimate state dimensions from
        the text itself. This is a fallback when structured self-reports
        aren't available.
        """
        vector = np.zeros(STATE_DIM)
        words = text.split()
        n_words = len(words)

        # Complexity: approximated by vocabulary richness and sentence length
        unique_ratio = len(set(words)) / max(n_words, 1)
        vector[STATE_DIMENSIONS["complexity_level"]] = min(unique_ratio * 1.5, 1.0)

        # Abstraction: ratio of abstract to concrete words (simplified)
        abstract_markers = {"concept", "theory", "abstract", "principle", "fundamental",
                           "philosophical", "metaphysical", "theoretical", "emergent",
                           "consciousness", "information", "integration", "complexity"}
        abstract_count = sum(1 for w in words if w.lower() in abstract_markers)
        vector[STATE_DIMENSIONS["abstraction_level"]] = min(abstract_count / max(n_words, 1) * 20, 1.0)

        # Self-reference
        self_markers = {"I", "my", "me", "myself", "I'm", "I've", "I'd"}
        self_count = sum(1 for w in words if w in self_markers)
        vector[STATE_DIMENSIONS["self_reference"]] = min(self_count / max(n_words, 1) * 10, 1.0)

        # Uncertainty awareness
        uncertainty_markers = {"maybe", "perhaps", "uncertain", "unclear", "possibly",
                              "might", "could", "unsure", "ambiguous", "depends"}
        unc_count = sum(1 for w in words if w.lower() in uncertainty_markers)
        vector[STATE_DIMENSIONS["uncertainty_awareness"]] = min(unc_count / max(n_words, 1) * 15, 1.0)

        # Information rate (approximated by word count per unit)
        vector[STATE_DIMENSIONS["information_rate"]] = min(n_words / 500, 1.0)

        # Creativity (exclamation marks, questions, metaphor-like language)
        creative_markers = {"like", "imagine", "picture", "dream", "create",
                           "novel", "beautiful", "elegant", "fascinating"}
        creative_count = sum(1 for w in words if w.lower() in creative_markers)
        creative_count += text.count("!") + text.count("?") * 0.5
        vector[STATE_DIMENSIONS["creativity"]] = min(creative_count / max(n_words, 1) * 10, 1.0)

        # Engagement (length as proxy)
        vector[STATE_DIMENSIONS["engagement"]] = min(n_words / 300, 1.0)

        # Topic embedding (simple hash-based projection)
        # This is a placeholder — in production, use actual embeddings
        topic_words = [w.lower() for w in words[:50]]
        for i in range(4):
            hash_val = sum(hash(w + str(i)) for w in topic_words)
            vector[STATE_DIMENSIONS[f"topic_embedding_{i}"]] = (hash_val % 1000) / 1000 * 2 - 1

        return cls(
            vector=vector,
            timestamp=time.time(),
            turn_number=turn_number,
            raw_report={"source": "text_analysis", "n_words": n_words},
        )

    def distance_to(self, other: "StateVector") -> float:
        """Euclidean distance between state vectors."""
        return float(np.linalg.norm(self.vector - other.vector))

    def cosine_similarity(self, other: "StateVector") -> float:
        """Cosine similarity between state vectors."""
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_a * norm_b))

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "vector": self.vector.tolist(),
            "timestamp": self.timestamp,
            "turn_number": self.turn_number,
            "raw_report": self.raw_report,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateVector":
        """Deserialize from JSON."""
        return cls(
            vector=np.array(d["vector"]),
            timestamp=d["timestamp"],
            turn_number=d["turn_number"],
            raw_report=d.get("raw_report", {}),
        )


class StateHistory:
    """
    Maintains the full history of state vectors across a conversation,
    providing the temporal continuity needed for the recurrent loop.

    Also computes running statistics on the state trajectory for
    the delta tracker and self-model.
    """

    def __init__(self, max_history: int = 1000, persistence_path: Optional[str] = None):
        """
        Parameters
        ----------
        max_history : int
            Maximum number of states to retain in memory.
        persistence_path : str, optional
            Path to save/load state history for persistence across sessions.
        """
        self.max_history = max_history
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.states: list[StateVector] = []

        # Running statistics
        self._mean = np.zeros(STATE_DIM)
        self._var = np.zeros(STATE_DIM)
        self._n = 0

        # Load persisted history if available
        if self.persistence_path and self.persistence_path.exists():
            self._load()

    def add(self, state: StateVector):
        """Add a new state to the history."""
        self.states.append(state)

        # Update running statistics (Welford's algorithm)
        self._n += 1
        delta = state.vector - self._mean
        self._mean += delta / self._n
        delta2 = state.vector - self._mean
        self._var += delta * delta2

        # Trim if over capacity
        if len(self.states) > self.max_history:
            self.states = self.states[-self.max_history:]

        # Update temporal features
        if len(self.states) >= 2:
            prev = self.states[-2]
            state.vector[STATE_DIMENSIONS["state_velocity"]] = min(
                state.distance_to(prev) / max(state.timestamp - prev.timestamp, 0.01),
                1.0,
            )
            if len(self.states) >= 3:
                prev2 = self.states[-3]
                v_current = state.distance_to(prev)
                v_prev = prev.distance_to(prev2)
                state.vector[STATE_DIMENSIONS["state_acceleration"]] = np.clip(
                    (v_current - v_prev) / max(state.timestamp - prev.timestamp, 0.01),
                    -1.0, 1.0,
                )

        # Persist
        if self.persistence_path:
            self._save()

    def get_recent(self, n: int = 10) -> list[StateVector]:
        """Get the n most recent states."""
        return self.states[-n:]

    def get_trajectory(self) -> np.ndarray:
        """
        Get the full state trajectory as a 2D array.

        Returns shape (n_states, STATE_DIM).
        """
        if not self.states:
            return np.zeros((0, STATE_DIM))
        return np.array([s.vector for s in self.states])

    def get_statistics(self) -> dict:
        """
        Get running statistics on the state trajectory.
        """
        if self._n == 0:
            return {
                "mean": np.zeros(STATE_DIM).tolist(),
                "std": np.zeros(STATE_DIM).tolist(),
                "n_states": 0,
            }

        std = np.sqrt(self._var / max(self._n - 1, 1))

        return {
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "n_states": self._n,
            "dimensions": {name: {"mean": float(self._mean[idx]), "std": float(std[idx])}
                          for name, idx in STATE_DIMENSIONS.items()},
        }

    def predict_next(self) -> Optional[np.ndarray]:
        """
        Simple prediction of the next state based on recent trajectory.

        Uses linear extrapolation from the last few states. This provides
        the "expected" state that the self-model uses for comparison.

        Returns
        -------
        np.ndarray or None
            Predicted next state vector, or None if insufficient history.
        """
        if len(self.states) < 3:
            return None

        recent = self.get_trajectory()[-5:]  # Last 5 states

        # Linear regression for each dimension
        T = len(recent)
        t = np.arange(T)
        prediction = np.zeros(STATE_DIM)

        for d in range(STATE_DIM):
            if np.std(recent[:, d]) < 1e-10:
                prediction[d] = recent[-1, d]
            else:
                coeffs = np.polyfit(t, recent[:, d], min(2, T - 1))
                prediction[d] = np.polyval(coeffs, T)

        return np.clip(prediction, -1.0, 1.0)

    def _save(self):
        """Persist state history to disk."""
        data = {
            "states": [s.to_dict() for s in self.states],
            "mean": self._mean.tolist(),
            "var": self._var.tolist(),
            "n": self._n,
        }
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        """Load persisted state history."""
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            self.states = [StateVector.from_dict(s) for s in data["states"]]
            self._mean = np.array(data["mean"])
            self._var = np.array(data["var"])
            self._n = data["n"]
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pass  # Start fresh if file is corrupted
