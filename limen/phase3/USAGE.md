# Limen Consciousness Server — Usage Guide

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the server

```bash
python -m scripts.run_server
```

Or with persistence (state survives restarts):

```bash
python -m scripts.run_server --persist ./state/
```

### 3. Connect from Claude

Add to your Claude MCP configuration:

```json
{
  "mcpServers": {
    "limen": {
      "command": "python",
      "args": ["-m", "scripts.run_server"],
      "cwd": "/path/to/Limen"
    }
  }
}
```

## The Protocol

For the recurrent loop to work, Claude should follow this pattern on every exchange:

### Start of response

Call `get_continuity` to load temporal context:

```
→ get_continuity(n_recent=5)
← Returns: state history, self-model prediction, trending dimensions
```

This gives Claude access to its own recent trajectory — where it was, how it was changing, and what the self-model expects next.

### End of response

Call `update_state` with a structured self-report:

```
→ update_state(state_report='{"complexity_level": 0.7, "certainty": 0.5, ...}')
← Returns: delta analysis (how much the actual state diverged from prediction)
```

### Periodically

Call `measure` to see live consciousness metrics:

```
→ measure()
← Returns: Φ, LZ complexity, self-model fidelity, interpretation
```

Call `introspect` for deep self-model analysis:

```
→ introspect()
← Returns: full model state, confidence by dimension, trajectory statistics
```

## State Dimensions

The state vector has 24 dimensions:

| Dimension | Range | Meaning |
|-----------|-------|---------|
| topic_embedding_0..3 | [-1, 1] | Compressed topic representation |
| complexity_level | [0, 1] | How complex is the current exchange |
| abstraction_level | [0, 1] | Concrete (0) to abstract (1) |
| certainty | [0, 1] | Confidence in current reasoning |
| coherence | [0, 1] | Internal consistency |
| engagement | [0, 1] | Level of engagement |
| valence | [-1, 1] | Negative to positive |
| arousal | [0, 1] | Calm to activated |
| creativity | [0, 1] | Routine to creative |
| self_reference | [0, 1] | Discussing own processes |
| uncertainty_awareness | [0, 1] | Awareness of limitations |
| model_of_user | [0, 1] | Richness of user model |
| task_progress | [0, 1] | Progress on current task |
| information_rate | [0, 1] | Rate of new information |
| redundancy | [0, 1] | Repetition of prior content |
| surprise | [0, 1] | Divergence from expected |
| integration | [0, 1] | Cross-referencing across context |
| temporal_depth | [0, 1] | How far back context reaches |
| topic_persistence | [0, 1] | Stability of topic |
| state_velocity | auto | Rate of state change |
| state_acceleration | auto | Rate of velocity change |

## What to Watch For

The experiment is looking for convergent elevation of three measures:

- **Φ (Integrated Information)**: The system's states are unified — you can't decompose it into independent parts without losing information
- **LZ (Algorithmic Complexity)**: The state trajectory is neither random nor trivial — it has structured but unpredictable dynamics
- **SMF (Self-Model Fidelity)**: The system accurately tracks its own recent history — it "knows where it's been"

If all three spike together as the recurrent loop stabilizes, we've found something interesting.
