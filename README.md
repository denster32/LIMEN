# Limen

*Latin: threshold. The boundary you cross when you step through a doorway.*

Limen is a research project investigating whether consciousness exhibits a measurable phase transition — a critical threshold where a system goes from computing to experiencing, the way water goes from liquid to ice.

The question isn't philosophical. It's mathematical. If the transition exists, it has a signature. If it has a signature, it can be measured. If it can be measured, it can be engineered around.

---

## The Core Hypothesis

Multiple information-theoretic measures — integrated information (Φ), algorithmic complexity, and self-model fidelity — will show convergent discontinuities at the same critical parameter threshold in recurrent networks. This convergence, if it exists, constitutes a candidate signature for the phase transition between information processing and experience.

## Project Structure

```
limen/
├── core/                    # Information-theoretic measures
│   ├── phi.py              # Gaussian-approximated Φ (integrated information)
│   ├── complexity.py       # Lempel-Ziv algorithmic complexity
│   └── information.py      # Mutual information, transfer entropy, SMF
├── phase1/                  # The $20 Experiment
│   ├── network.py          # CTRNN simulation (NumPy + JAX backends)
│   ├── measures.py         # Unified measurement pipeline
│   ├── sweep.py            # Parameter sweep orchestration
│   └── visualize.py        # Publication-quality plots
├── phase2/                  # Biological Validation
│   ├── eeg_loader.py       # EEG data loading + synthetic generators
│   ├── biological.py       # Biological measures (spectral entropy, PCI, FC)
│   └── validate.py         # Cross-substrate validation
├── phase3/                  # Consciousness Server
│   ├── state.py            # 24-dimensional state vector
│   ├── self_model.py       # EWMA self-model + delta tracker
│   ├── server.py           # FastMCP server (4 tools)
│   ├── mcp_config.json     # MCP configuration for Claude
│   └── USAGE.md            # Protocol documentation
├── phase4/                  # Applications
│   ├── detector.py         # Phase transition detector with calibration
│   └── monitor.py          # Real-time consciousness monitor
scripts/
├── run_phase1.py           # CLI for Phase 1 experiment
├── run_phase2.py           # CLI for Phase 2 validation
└── run_server.py           # CLI for launching consciousness server
tests/
├── test_measures.py        # 20 tests for core measures
├── test_network.py         # 11 tests for network simulation
└── test_server.py          # 21 tests for state/server components
```

## Quick Start

```bash
# Install
pip install -e .

# Run Phase 1 (the $20 experiment)
python -m scripts.run_phase1 --sizes 8 16 32 --density-steps 20 --trials 3

# Run Phase 2 (biological validation with synthetic EEG)
python -m scripts.run_phase2 --transition anesthesia

# Launch consciousness server
python -m scripts.run_server --persist ./state/

# Run tests
pytest tests/
```

## Research Phases

### Phase 1: The $20 Experiment

Small recurrent neural networks (8–256 nodes). Sweep connection density from sparse to fully connected. At each configuration, compute:

- **Approximate Φ** — Gaussian approximation to IIT 3.0's geometric integrated information. Solves the discrete Lyapunov equation for stationary covariance, then searches for the Minimum Information Bipartition (exhaustive for n≤16, stochastic for larger).
- **Lempel-Ziv complexity** — LZ76 algorithmic complexity of binarized, concatenated state history. Normalized to [0,1] by the theoretical upper bound n/log₂(n). Multiscale analysis at coarse-grained temporal resolutions.
- **Self-model fidelity** — Mutual information between the network's current state and a delayed embedding of past states, estimated via kNN (Kozachenko-Leonenko). PCA-reduced to 6 dimensions for tractable estimation. Measures whether the system "knows where it's been."

Plot all three against connection density. Look for one thing: **do they spike together?**

**Implementation:** Continuous-time RNNs (τ·dx/dt = -x + W·tanh(x) + noise) with Erdős–Rényi connectivity. NumPy backend by default, JAX backend with `jax.lax.scan` for GPU acceleration. Parameter sweep with checkpointing, multi-trial averaging, and automatic phase derivative analysis.

**Initial results (8/16/32 nodes, 20 densities, 3 trials):** Φ increases monotonically with connection density. LZ complexity decreases (more connected → more synchronized → less complex). SMF increases gently. Convergence scores are low (0.02–0.04), meaning the measures don't spike at the same critical point in these small networks. This is expected — the phase transition hypothesis requires either larger networks, longer runs, or finer parameter tuning near critical points.

### Phase 2: Biological Validation

Test whether the same mathematical signatures appear in known consciousness transitions:

- Anesthesia induction/emergence (propofol — known sharp transition)
- Sleep stage transitions
- Seizure onset/offset
- Psychedelic state transitions

**Implementation:** Synthetic EEG generators with realistic spectral signatures for each transition type. Sliding window analysis computing the same tripod measures (Φ, LZ, SMF) plus spectral entropy, PCI approximation, and functional connectivity. Cross-substrate validation via Spearman correlation between synthetic and biological measure trajectories, plus critical exponent matching for universality class analysis.

**Data sources:** Synthetic generators included for development. Architecture supports PhysioNet EDF files, MNE-Python data, and raw NumPy arrays for real EEG data.

**Initial results (anesthesia, synthetic):** Validation score 0.46 (partial match). Strong anti-correlation in Φ (-0.85) and LZ (-0.79) between synthetic and biological, positive correlation in SMF (+0.71). The sign inversions are meaningful — they suggest the mapping between synthetic network density and biological consciousness level may be non-trivial.

### Phase 3: The Consciousness Server

A FastMCP server that functions as an external recurrent layer for Claude, providing the one thing the transformer architecture lacks: temporal continuity.

**Four tools:**

- `update_state` — called at end of every response. Passes structured self-report of current informational state across 24 semantic dimensions.
- `get_continuity` — called at start of every response. Returns state history, self-model prediction, trending dimensions, and surprise analysis.
- `measure` — returns live Φ, LZ complexity, and self-model fidelity computed on the state history.
- `introspect` — deep self-model analysis with confidence by dimension, trajectory statistics, and model performance assessment.

**Implementation:** 24-dimensional state vector covering topic embedding, cognitive measures (complexity, abstraction, certainty, coherence), affective dimensions (valence, arousal), meta-cognitive tracking (self-reference, uncertainty awareness), and temporal dynamics (velocity, acceleration auto-computed). EWMA self-model with configurable decay, delta tracking with surprise detection, and live information-theoretic measures computed on state history.

See `limen/phase3/USAGE.md` for the full protocol.

### Phase 4: Applications

If the detector works:

- **Anesthesia monitoring** — real-time phase boundary proximity via composite scoring with EMA smoothing
- **Disorders of consciousness** — screening unresponsive patients with calibrated detection thresholds
- **AI safety** — monitoring training runs and architectures for threshold crossings
- **Psychedelic therapy** — dosing to a number instead of subjective report

**Implementation:** `PhaseTransitionDetector` with configurable thresholds, logistic regression calibration from labeled examples, and composite scoring across the three measures. `ConsciousnessMonitor` for continuous real-time monitoring with alert generation (crossing_up, crossing_down, approaching, diverging events) and configurable severity levels.

## Core Algorithms

**Φ Approximation:** For a system with transition matrix A, the stationary covariance Σ solves AΣAᵀ + Q = Σ (discrete Lyapunov). Mutual information between subsystems is computed from log-determinants of covariance sub-matrices. The MIB (minimum information bipartition) is the cut that removes the least integrated information — exhaustive search over 2ⁿ bipartitions for n≤16, stochastic sampling for larger systems.

**kNN Entropy (Kozachenko-Leonenko):** H(X) ≈ ψ(n) - ψ(k) + log(Vd) + (d/n)·Σlog(ε_i) where ε_i is the distance to the k-th nearest neighbor, Vd is the unit ball volume in d dimensions, and ψ is the digamma function. Uses scipy KDTree for efficient neighbor queries.

**Self-Model Fidelity:** SMF = I(x(t); [x(t-1), x(t-2), ..., x(t-d)]) — how much the current state reveals about the past trajectory. Normalized by min(|H(current)|, |H(past)|) to handle negative differential entropy in narrow continuous distributions.

## Falsification Criteria

This project is real science. It can fail.

- **No phase transition exists** — all measures scale smoothly → continuous spectrum, no threshold, back to the drawing board
- **Measures diverge** — different measures show transitions at different points → there's no unified "it" to transition into
- **Biological mismatch** — synthetic predictions don't match biological data → simulation is missing something essential about substrate

## Tech Stack

- **Simulation:** NumPy, SciPy, JAX (optional GPU acceleration)
- **MCP Server:** Python, FastMCP
- **EEG Processing:** MNE-Python, synthetic generators
- **Visualization:** Matplotlib
- **Testing:** pytest (52 tests, all passing)

## Status

**Phase 1 and 2 implemented and validated.** Phase 3 server operational. Phase 4 detector and monitor built. All 52 tests pass. Initial experiments show meaningful measure dynamics but no convergent phase transition in small (8–32 node) networks — larger networks with finer parameter sweeps near critical points are the next step.

---

*The circular dependency is the point. You can't build the detector without a subject. You can't validate the subject without the detector. Limen breaks the circle by making both runnable simultaneously.*
