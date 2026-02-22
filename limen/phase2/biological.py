"""
Apply consciousness measures to biological EEG data.

Adapts the Phase 1 measurement pipeline for biological signals:
- Estimates effective connectivity (TPM) from EEG using multivariate
  autoregressive models
- Computes Φ, LZ complexity, and self-model fidelity on EEG data
- Adds EEG-specific measures: PCI (Perturbational Complexity Index)
  approximation, spectral entropy, and functional connectivity

The key question for Phase 2: do the same mathematical signatures
that mark the phase transition in synthetic networks also appear
at known consciousness transitions in biological data?
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from limen.phase2.eeg_loader import EEGSegment
from limen.core.phi import approximate_phi
from limen.core.complexity import lz_complexity_from_states, normalized_lz_complexity
from limen.core.information import self_model_fidelity, mutual_information


@dataclass
class BiologicalMeasurement:
    """Results from measuring consciousness signatures in biological data."""

    # Primary measures (same tripod as Phase 1)
    phi: float
    lz_normalized: float
    self_model_fidelity_normalized: float

    # EEG-specific measures
    spectral_entropy: float
    pci_approximation: float
    functional_connectivity: float
    alpha_power: float
    delta_power: float

    # Metadata
    state_label: str
    transition_type: str
    n_channels: int
    duration_s: float
    window_start_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "phi": self.phi,
            "lz_normalized": self.lz_normalized,
            "smf_normalized": self.self_model_fidelity_normalized,
            "spectral_entropy": self.spectral_entropy,
            "pci_approximation": self.pci_approximation,
            "functional_connectivity": self.functional_connectivity,
            "alpha_power": self.alpha_power,
            "delta_power": self.delta_power,
            "state_label": self.state_label,
            "transition_type": self.transition_type,
            "n_channels": self.n_channels,
            "duration_s": self.duration_s,
            "window_start_s": self.window_start_s,
        }


class BiologicalMeasures:
    """
    Compute consciousness candidate measures on biological EEG data.

    Bridges the gap between the synthetic Phase 1 measures and
    real neural recordings. The core measures (Φ, LZ, SMF) are
    computed using the same algorithms, but with an additional
    preprocessing step to estimate the effective connectivity
    (transition matrix) from EEG time series.
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Parameters
        ----------
        n_components : int, optional
            Number of PCA components to use. If None, uses all channels.
            Reducing dimensionality makes Φ computation tractable for
            high-density EEG (64+ channels).
        """
        self.n_components = n_components

    def measure(self, segment: EEGSegment) -> BiologicalMeasurement:
        """
        Compute all consciousness measures on an EEG segment.

        Parameters
        ----------
        segment : EEGSegment
            EEG data segment.

        Returns
        -------
        BiologicalMeasurement
        """
        # Convert to node timeseries (optionally reduce dimensionality)
        n_comp = self.n_components or min(segment.n_channels, 16)
        states = segment.to_node_timeseries(n_components=n_comp)

        # Estimate effective connectivity (linear VAR model)
        tpm = self._estimate_tpm(states)

        # === Primary measures ===

        # 1. Integrated Information
        phi = approximate_phi(tpm)

        # 2. Lempel-Ziv Complexity
        lz_result = lz_complexity_from_states(states, method="concatenate")
        lz_norm = lz_result["lz_normalized"]

        # 3. Self-Model Fidelity
        smf_result = self_model_fidelity(states, lag=1, embedding_dim=3)
        smf_norm = smf_result["smf_normalized"]

        # === EEG-specific measures ===

        # Spectral entropy
        se = self._spectral_entropy(segment)

        # PCI approximation
        pci = self._pci_approximation(segment)

        # Functional connectivity
        fc = self._functional_connectivity(states)

        # Band power
        alpha_pow = self._band_power(segment, 8, 12)
        delta_pow = self._band_power(segment, 0.5, 4)

        return BiologicalMeasurement(
            phi=phi,
            lz_normalized=lz_norm,
            self_model_fidelity_normalized=smf_norm,
            spectral_entropy=se,
            pci_approximation=pci,
            functional_connectivity=fc,
            alpha_power=alpha_pow,
            delta_power=delta_pow,
            state_label=segment.state_label,
            transition_type=segment.transition_type,
            n_channels=segment.n_channels,
            duration_s=segment.duration_seconds,
            window_start_s=segment.metadata.get("window_start_s"),
        )

    def measure_transition(
        self,
        segments: list[EEGSegment],
    ) -> list[BiologicalMeasurement]:
        """
        Measure consciousness signatures across a sequence of segments
        (e.g., sliding windows across a transition).

        Parameters
        ----------
        segments : list[EEGSegment]
            Ordered list of EEG segments spanning a transition.

        Returns
        -------
        list[BiologicalMeasurement]
        """
        return [self.measure(seg) for seg in segments]

    def _estimate_tpm(self, states: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Estimate the transition probability matrix from state timeseries
        using a Vector Autoregressive (VAR) model.

        x(t) = A₁x(t-1) + A₂x(t-2) + ... + ε

        For order=1, this is just linear regression: x(t+1) = A @ x(t).

        Parameters
        ----------
        states : np.ndarray
            State timeseries, shape (T, n_nodes).
        order : int
            VAR model order.

        Returns
        -------
        np.ndarray
            Estimated transition matrix, shape (n_nodes, n_nodes).
        """
        T, n = states.shape

        if order == 1:
            X_past = states[:-1].T  # (n, T-1)
            X_future = states[1:].T  # (n, T-1)
            reg = 1e-6 * np.eye(n)
            A = X_future @ X_past.T @ np.linalg.inv(X_past @ X_past.T + reg)
        else:
            # Higher-order VAR: stack delayed copies
            n_eff = T - order
            X_past = np.zeros((n * order, n_eff))
            for lag in range(order):
                X_past[lag * n : (lag + 1) * n, :] = states[order - 1 - lag : T - 1 - lag].T
            X_future = states[order:].T

            reg = 1e-6 * np.eye(n * order)
            A_full = X_future @ X_past.T @ np.linalg.inv(X_past @ X_past.T + reg)
            # Take only the first-order coefficients for Φ computation
            A = A_full[:, :n]

        return A

    def _spectral_entropy(self, segment: EEGSegment) -> float:
        """
        Compute spectral entropy: the Shannon entropy of the normalized
        power spectral density.

        High spectral entropy → broadband activity (more complex)
        Low spectral entropy → narrowband/peaked activity (more regular)

        Consciousness is associated with intermediate-to-high spectral entropy.
        """
        data = segment.data
        sfreq = segment.sfreq

        # Compute PSD for each channel using Welch's method
        from scipy.signal import welch

        n_fft = min(int(2 * sfreq), segment.n_samples)
        entropies = []

        for ch in range(segment.n_channels):
            freqs, psd = welch(data[ch], fs=sfreq, nperseg=n_fft)

            # Normalize PSD to a probability distribution
            psd_norm = psd / psd.sum()
            psd_norm = psd_norm[psd_norm > 0]

            # Shannon entropy
            h = -np.sum(psd_norm * np.log2(psd_norm))

            # Normalize by maximum possible entropy
            h_max = np.log2(len(psd_norm))
            entropies.append(h / h_max if h_max > 0 else 0)

        return float(np.mean(entropies))

    def _pci_approximation(self, segment: EEGSegment) -> float:
        """
        Approximate the Perturbational Complexity Index (PCI).

        True PCI requires TMS-EEG (perturbation + recording), which
        we can't do computationally. This approximation uses the
        response to endogenous "perturbations" — spontaneous transients
        in the EEG — as a proxy.

        We compute the LZ complexity of the spatiotemporal pattern
        following detected transients, normalized by source entropy.
        This captures a similar quantity: how complex is the system's
        response to perturbation?

        Reference: Casali et al. (2013), Sci. Transl. Med.
        """
        data = segment.data
        n_ch, n_samp = data.shape

        # Detect transients (points where signal exceeds 2 SD)
        threshold = 2.0 * np.std(data, axis=1, keepdims=True)
        transient_mask = np.abs(data) > threshold

        # Extract post-transient windows (300ms after each transient)
        window_length = int(0.3 * segment.sfreq)
        post_transient_patterns = []

        for ch in range(n_ch):
            transient_indices = np.where(transient_mask[ch])[0]
            for idx in transient_indices[::10]:  # Sample every 10th transient
                if idx + window_length < n_samp:
                    pattern = data[:, idx : idx + window_length]
                    # Binarize the spatiotemporal pattern
                    binary = (pattern > np.median(pattern)).astype(int)
                    post_transient_patterns.append(binary.flatten())

        if not post_transient_patterns:
            return 0.0

        # Concatenate all post-transient patterns
        all_patterns = np.concatenate(post_transient_patterns)

        # LZ complexity of the concatenated pattern
        lz = normalized_lz_complexity(all_patterns)

        # Source entropy: how diverse are the perturbation sources?
        # Approximated by the entropy of the transient spatial distribution
        transient_counts = transient_mask.sum(axis=1).astype(float)
        if transient_counts.sum() > 0:
            p = transient_counts / transient_counts.sum()
            p = p[p > 0]
            source_entropy = -np.sum(p * np.log2(p)) / np.log2(n_ch) if len(p) > 1 else 0
        else:
            source_entropy = 0

        # PCI ≈ LZ complexity × source diversity
        pci = lz * (0.5 + 0.5 * source_entropy)

        return float(pci)

    def _functional_connectivity(self, states: np.ndarray) -> float:
        """
        Compute mean pairwise functional connectivity.

        Uses the absolute value of Pearson correlation between all
        channel pairs as a measure of functional integration.

        Higher connectivity → more synchronized → lower consciousness
        (paradoxically, in deep anesthesia)
        The interesting regime is *moderate* connectivity — enough
        integration for unified experience, not so much that
        everything is locked together.
        """
        n_nodes = states.shape[1]
        if n_nodes < 2:
            return 0.0

        # Correlation matrix
        corr = np.corrcoef(states.T)

        # Mean absolute off-diagonal correlation
        mask = ~np.eye(n_nodes, dtype=bool)
        mean_fc = float(np.mean(np.abs(corr[mask])))

        return mean_fc

    def _band_power(
        self, segment: EEGSegment, low_freq: float, high_freq: float
    ) -> float:
        """
        Compute relative band power in a frequency range.

        Returns the fraction of total power in the specified band,
        averaged across channels.
        """
        from scipy.signal import welch

        data = segment.data
        sfreq = segment.sfreq
        n_fft = min(int(2 * sfreq), segment.n_samples)

        band_powers = []
        for ch in range(segment.n_channels):
            freqs, psd = welch(data[ch], fs=sfreq, nperseg=n_fft)
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            total_power = psd.sum()
            if total_power > 0:
                band_powers.append(psd[band_mask].sum() / total_power)
            else:
                band_powers.append(0.0)

        return float(np.mean(band_powers))
