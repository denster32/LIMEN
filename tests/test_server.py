"""
Tests for the Phase 3 consciousness server components.
"""

import numpy as np
import json
import pytest

from limen.phase3.state import StateVector, StateHistory, STATE_DIM, STATE_DIMENSIONS
from limen.phase3.self_model import SelfModel, DeltaTracker, LiveMeasures


class TestStateVector:
    def test_from_report(self):
        """State vector should be constructed from a report dict."""
        report = {
            "complexity_level": 0.7,
            "certainty": 0.5,
            "engagement": 0.8,
        }
        state = StateVector.from_report(report, turn_number=1)
        assert state.vector[STATE_DIMENSIONS["complexity_level"]] == 0.7
        assert state.vector[STATE_DIMENSIONS["certainty"]] == 0.5
        assert state.vector[STATE_DIMENSIONS["engagement"]] == 0.8
        assert state.turn_number == 1

    def test_from_report_clipping(self):
        """Values outside [-1, 1] should be clipped."""
        report = {"complexity_level": 5.0, "certainty": -3.0}
        state = StateVector.from_report(report, turn_number=1)
        assert state.vector[STATE_DIMENSIONS["complexity_level"]] == 1.0
        assert state.vector[STATE_DIMENSIONS["certainty"]] == -1.0

    def test_from_text_analysis(self):
        """State vector should be constructable from text."""
        text = "This is a complex theoretical analysis of consciousness involving abstract philosophical concepts."
        state = StateVector.from_text_analysis(text, turn_number=1)
        assert state.vector.shape == (STATE_DIM,)
        # Should detect some abstraction
        assert state.vector[STATE_DIMENSIONS["abstraction_level"]] > 0

    def test_distance(self):
        """Distance between identical vectors should be 0."""
        state1 = StateVector.from_report({"certainty": 0.5}, turn_number=1)
        state2 = StateVector.from_report({"certainty": 0.5}, turn_number=2)
        assert state1.distance_to(state2) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity(self):
        """Cosine similarity of identical states should be 1."""
        report = {"complexity_level": 0.7, "certainty": 0.5}
        state1 = StateVector.from_report(report, turn_number=1)
        state2 = StateVector.from_report(report, turn_number=2)
        sim = state1.cosine_similarity(state2)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_serialization(self):
        """State should roundtrip through JSON serialization."""
        report = {"complexity_level": 0.7, "engagement": 0.5}
        state = StateVector.from_report(report, turn_number=5)
        d = state.to_dict()
        restored = StateVector.from_dict(d)
        np.testing.assert_array_almost_equal(state.vector, restored.vector)
        assert restored.turn_number == 5


class TestStateHistory:
    def test_add_and_retrieve(self):
        """Should store and retrieve states."""
        history = StateHistory(max_history=100)
        for i in range(5):
            state = StateVector.from_report({"certainty": i * 0.2}, turn_number=i)
            history.add(state)
        assert len(history.states) == 5
        recent = history.get_recent(3)
        assert len(recent) == 3

    def test_max_history(self):
        """Should not exceed max_history."""
        history = StateHistory(max_history=10)
        for i in range(20):
            state = StateVector.from_report({"certainty": 0.1}, turn_number=i)
            history.add(state)
        assert len(history.states) == 10

    def test_trajectory(self):
        """get_trajectory should return correct shape."""
        history = StateHistory()
        for i in range(5):
            state = StateVector.from_report({"certainty": i * 0.2}, turn_number=i)
            history.add(state)
        traj = history.get_trajectory()
        assert traj.shape == (5, STATE_DIM)

    def test_predict_next(self):
        """Prediction should work with sufficient history."""
        history = StateHistory()
        for i in range(10):
            state = StateVector.from_report(
                {"certainty": i * 0.1, "complexity_level": 0.5},
                turn_number=i,
            )
            history.add(state)
        prediction = history.predict_next()
        assert prediction is not None
        assert prediction.shape == (STATE_DIM,)

    def test_statistics(self):
        """Statistics should have expected structure."""
        history = StateHistory()
        for i in range(5):
            state = StateVector.from_report({"certainty": i * 0.2}, turn_number=i)
            history.add(state)
        stats = history.get_statistics()
        assert stats["n_states"] == 5
        assert "mean" in stats
        assert "std" in stats


class TestSelfModel:
    def test_initial_prediction_zero(self):
        """Initial prediction should be zero vector."""
        model = SelfModel()
        pred = model.predict()
        assert np.allclose(pred, 0)

    def test_update_returns_delta(self):
        """update should return a DeltaRecord."""
        model = SelfModel()
        state = StateVector.from_report({"certainty": 0.5}, turn_number=1)
        delta = model.update(state)
        assert hasattr(delta, "delta_magnitude")
        assert hasattr(delta, "surprise")
        assert delta.turn_number == 1

    def test_model_improves(self):
        """After several updates, prediction error should decrease."""
        model = SelfModel(alpha=0.5)

        # Feed constant states — model should learn the constant
        deltas = []
        for i in range(20):
            state = StateVector.from_report(
                {"certainty": 0.5, "complexity_level": 0.3},
                turn_number=i,
            )
            delta = model.update(state)
            deltas.append(delta.delta_magnitude)

        # Later deltas should be smaller than earlier ones
        early = np.mean(deltas[1:5])
        late = np.mean(deltas[-5:])
        assert late < early, f"Model should improve: early={early}, late={late}"

    def test_confidence(self):
        """Confidence should be between 0 and 1."""
        model = SelfModel()
        state = StateVector.from_report({"certainty": 0.5}, turn_number=1)
        model.update(state)
        confidence = model.get_confidence()
        assert confidence.shape == (STATE_DIM,)
        assert np.all(confidence >= 0) and np.all(confidence <= 1)


class TestDeltaTracker:
    def test_trends_insufficient_data(self):
        """Should return zeros with insufficient data."""
        tracker = DeltaTracker()
        trends = tracker.get_trends()
        assert trends["delta_rate"] == 0.0
        assert trends["n_records"] == 0

    def test_trends_with_data(self):
        """Should compute meaningful trends with enough data."""
        tracker = DeltaTracker()
        model = SelfModel()

        for i in range(15):
            state = StateVector.from_report(
                {"certainty": i * 0.05, "engagement": 0.5},
                turn_number=i,
            )
            delta = model.update(state)
            tracker.add(delta)

        trends = tracker.get_trends()
        assert trends["n_records"] == 15
        assert "delta_rate" in trends
        assert "volatility" in trends


class TestLiveMeasures:
    def test_insufficient_history(self):
        """Should report insufficient history."""
        history = StateHistory()
        for i in range(3):
            state = StateVector.from_report({"certainty": 0.1}, turn_number=i)
            history.add(state)

        measures = LiveMeasures(min_history=10)
        result = measures.compute(history)
        assert result["status"] == "insufficient_history"

    def test_sufficient_history(self):
        """Should compute measures with enough history."""
        history = StateHistory()
        rng = np.random.default_rng(42)

        for i in range(30):
            report = {name: float(rng.uniform(-0.5, 0.5)) for name in STATE_DIMENSIONS}
            state = StateVector.from_report(report, turn_number=i)
            history.add(state)

        measures = LiveMeasures(min_history=10)
        result = measures.compute(history)
        assert result["status"] == "ok"
        assert result["phi"] is not None
        assert result["lz_normalized"] is not None
        assert result["smf_normalized"] is not None
