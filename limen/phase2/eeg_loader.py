"""
EEG data loading and preprocessing for biological validation.

Handles multiple data sources:
- PhysioNet open EEG datasets (anesthesia, sleep staging)
- MNE sample datasets (for development/testing)
- Raw EDF/BDF files from any source
- Custom NumPy arrays

Preprocessing follows standard EEG analysis practices:
bandpass filtering, artifact rejection, re-referencing, and epoching
around consciousness transitions.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import warnings


@dataclass
class EEGSegment:
    """
    A segment of EEG data representing a specific consciousness state.

    Attributes
    ----------
    data : np.ndarray
        EEG data, shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz.
    channel_names : list[str]
        Names of EEG channels.
    state_label : str
        Consciousness state label (e.g., 'awake', 'anesthetized', 'sleep_N3').
    transition_type : str
        Type of transition (e.g., 'anesthesia', 'sleep', 'seizure', 'psychedelic').
    metadata : dict
        Additional information (subject ID, timestamp, drug, dosage, etc.).
    """

    data: np.ndarray
    sfreq: float
    channel_names: list[str]
    state_label: str
    transition_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sfreq

    def to_node_timeseries(self, n_components: Optional[int] = None) -> np.ndarray:
        """
        Convert EEG channels to a node timeseries format compatible
        with the Phase 1 measurement pipeline.

        Optionally reduces dimensionality via PCA.

        Returns shape (n_samples, n_components).
        """
        # Transpose to (n_samples, n_channels)
        data_t = self.data.T

        if n_components is not None and n_components < self.n_channels:
            # PCA dimensionality reduction
            centered = data_t - data_t.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            data_t = centered @ vt[:n_components].T

        return data_t


class EEGLoader:
    """
    Unified EEG data loader supporting multiple formats and sources.

    Wraps MNE-Python for file I/O and provides a simplified interface
    for extracting consciousness-transition segments.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Parameters
        ----------
        data_dir : str, optional
            Root directory for EEG data files.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._mne = None

    def _get_mne(self):
        """Lazy import of MNE."""
        if self._mne is None:
            try:
                import mne
                mne.set_log_level("WARNING")
                self._mne = mne
            except ImportError:
                raise ImportError(
                    "MNE-Python is required for EEG loading. "
                    "Install with: pip install mne"
                )
        return self._mne

    def load_edf(
        self,
        filepath: str,
        state_label: str = "unknown",
        transition_type: str = "unknown",
        preprocess: bool = True,
        bandpass: tuple[float, float] = (0.5, 45.0),
        resample_hz: Optional[float] = None,
    ) -> EEGSegment:
        """
        Load an EDF/BDF file and return as an EEGSegment.

        Parameters
        ----------
        filepath : str
            Path to the EDF/BDF file.
        state_label : str
            Consciousness state label.
        transition_type : str
            Type of transition.
        preprocess : bool
            Apply standard preprocessing (filtering, re-referencing).
        bandpass : tuple
            Bandpass filter range in Hz.
        resample_hz : float, optional
            Target sampling rate for resampling.

        Returns
        -------
        EEGSegment
        """
        mne = self._get_mne()
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

        if preprocess:
            # Pick EEG channels only
            raw.pick_types(eeg=True, exclude="bads")

            # Bandpass filter
            raw.filter(bandpass[0], bandpass[1], verbose=False)

            # Re-reference to average
            raw.set_eeg_reference("average", verbose=False)

        if resample_hz is not None and raw.info["sfreq"] != resample_hz:
            raw.resample(resample_hz, verbose=False)

        data = raw.get_data()  # (n_channels, n_samples)
        sfreq = raw.info["sfreq"]
        ch_names = raw.ch_names

        return EEGSegment(
            data=data,
            sfreq=sfreq,
            channel_names=ch_names,
            state_label=state_label,
            transition_type=transition_type,
            metadata={
                "filepath": str(filepath),
                "duration_s": data.shape[1] / sfreq,
                "bandpass": bandpass,
            },
        )

    def load_from_array(
        self,
        data: np.ndarray,
        sfreq: float,
        channel_names: Optional[list[str]] = None,
        state_label: str = "unknown",
        transition_type: str = "unknown",
    ) -> EEGSegment:
        """
        Create an EEGSegment from a NumPy array.

        Parameters
        ----------
        data : np.ndarray
            EEG data, shape (n_channels, n_samples).
        sfreq : float
            Sampling frequency.
        channel_names : list[str], optional
            Channel names. Auto-generated if not provided.
        state_label, transition_type : str
            Labels for the segment.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(data.shape[0])]

        return EEGSegment(
            data=data,
            sfreq=sfreq,
            channel_names=channel_names,
            state_label=state_label,
            transition_type=transition_type,
        )

    def load_mne_sample(self) -> list[EEGSegment]:
        """
        Load MNE sample dataset for development and testing.

        Returns segments from the MNE sample dataset, which includes
        auditory and visual stimulation paradigms.
        """
        mne = self._get_mne()

        sample_data_folder = mne.datasets.sample.data_path()
        raw_fname = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"

        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
        raw.pick_types(eeg=True, exclude="bads")
        raw.filter(0.5, 45.0, verbose=False)

        # Split into segments
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        segment_length = int(30 * sfreq)  # 30-second segments

        segments = []
        for i in range(0, data.shape[1] - segment_length, segment_length):
            seg_data = data[:, i : i + segment_length]
            segments.append(
                EEGSegment(
                    data=seg_data,
                    sfreq=sfreq,
                    channel_names=raw.ch_names,
                    state_label=f"segment_{i // segment_length}",
                    transition_type="sample",
                    metadata={"source": "mne_sample", "offset_s": i / sfreq},
                )
            )

        return segments

    def create_synthetic_transition(
        self,
        transition_type: str = "anesthesia",
        n_channels: int = 19,
        sfreq: float = 256.0,
        duration_s: float = 120.0,
        transition_point: float = 0.5,
        seed: int = 42,
    ) -> list[EEGSegment]:
        """
        Generate synthetic EEG data simulating a consciousness transition.

        Uses known spectral signatures of different consciousness states:
        - Awake: dominant alpha (8-12 Hz), low delta
        - Anesthetized: dominant delta (0.5-4 Hz), loss of alpha, burst suppression
        - Sleep N3: dominant delta, sleep spindles (12-14 Hz)
        - Seizure: hypersynchronous fast activity, then postictal suppression

        This is for pipeline development and testing before running on
        real data.

        Parameters
        ----------
        transition_type : str
            Type of transition to simulate.
        n_channels : int
            Number of EEG channels.
        sfreq : float
            Sampling frequency.
        duration_s : float
            Total duration in seconds.
        transition_point : float
            Fraction of duration where transition occurs (0-1).
        seed : int
            Random seed.

        Returns
        -------
        list[EEGSegment]
            List of [pre_transition, post_transition] segments.
        """
        rng = np.random.default_rng(seed)
        n_samples = int(duration_s * sfreq)
        t = np.arange(n_samples) / sfreq
        transition_sample = int(n_samples * transition_point)

        # Generate frequency components
        def make_oscillation(freq, amplitude, phase_noise=0.1):
            phases = rng.uniform(0, 2 * np.pi, n_channels)
            osc = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                phase_drift = phase_noise * np.cumsum(rng.normal(0, 1, n_samples)) / sfreq
                osc[ch] = amplitude * np.sin(2 * np.pi * freq * t + phases[ch] + phase_drift)
            return osc

        if transition_type == "anesthesia":
            # Pre: awake (alpha dominant, some beta)
            pre_alpha = make_oscillation(10, 20)  # Alpha
            pre_beta = make_oscillation(20, 8)    # Beta
            pre_noise = rng.normal(0, 5, (n_channels, n_samples))
            pre_signal = pre_alpha + pre_beta + pre_noise

            # Post: anesthetized (delta dominant, burst suppression)
            post_delta = make_oscillation(2, 40)   # Delta
            post_noise = rng.normal(0, 3, (n_channels, n_samples))
            # Burst suppression: intermittent silence
            burst_mask = np.ones(n_samples)
            burst_duration = int(0.5 * sfreq)
            for start in range(0, n_samples, int(3 * sfreq)):
                end = min(start + burst_duration, n_samples)
                burst_mask[start:end] = 0.1
            post_signal = (post_delta + post_noise) * burst_mask

            pre_label = "awake"
            post_label = "anesthetized"

        elif transition_type == "sleep":
            # Pre: awake/light sleep
            pre_alpha = make_oscillation(10, 15)
            pre_noise = rng.normal(0, 5, (n_channels, n_samples))
            pre_signal = pre_alpha + pre_noise

            # Post: deep sleep (N3) — delta waves + sleep spindles
            post_delta = make_oscillation(1.5, 50)
            post_spindle = make_oscillation(13, 10)
            # Spindles are intermittent
            spindle_envelope = np.zeros(n_samples)
            for start in range(0, n_samples, int(5 * sfreq)):
                end = min(start + int(1.5 * sfreq), n_samples)
                spindle_envelope[start:end] = np.hanning(end - start)
            post_signal = post_delta + post_spindle * spindle_envelope + rng.normal(0, 3, (n_channels, n_samples))

            pre_label = "awake"
            post_label = "sleep_N3"

        elif transition_type == "seizure":
            # Pre: normal
            pre_alpha = make_oscillation(10, 15)
            pre_noise = rng.normal(0, 5, (n_channels, n_samples))
            pre_signal = pre_alpha + pre_noise

            # Post: seizure (hypersynchronous, increasing amplitude and frequency)
            seizure_freq = np.linspace(4, 20, n_samples)  # Frequency sweep
            post_signal = np.zeros((n_channels, n_samples))
            amplitude_ramp = np.linspace(20, 100, n_samples)
            for ch in range(n_channels):
                phase = np.cumsum(2 * np.pi * seizure_freq / sfreq)
                post_signal[ch] = amplitude_ramp * np.sin(phase + rng.uniform(0, 0.5))
            post_signal += rng.normal(0, 5, (n_channels, n_samples))

            pre_label = "normal"
            post_label = "seizure"

        elif transition_type == "psychedelic":
            # Pre: normal awake
            pre_alpha = make_oscillation(10, 20)
            pre_beta = make_oscillation(20, 8)
            pre_noise = rng.normal(0, 5, (n_channels, n_samples))
            pre_signal = pre_alpha + pre_beta + pre_noise

            # Post: psychedelic — reduced alpha, increased broadband complexity
            # Desynchronization + increased entropy
            post_components = []
            for freq in np.linspace(1, 40, 20):
                amp = 8 * rng.exponential(1)
                post_components.append(make_oscillation(freq, amp, phase_noise=0.5))
            post_signal = sum(post_components) + rng.normal(0, 8, (n_channels, n_samples))

            pre_label = "baseline"
            post_label = "psychedelic"

        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

        # Build smooth transition
        exponent = np.clip(-0.1 * sfreq * (t - t[transition_sample]), -500, 500)
        sigmoid = 1 / (1 + np.exp(exponent))
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            data[ch] = (1 - sigmoid) * pre_signal[ch] + sigmoid * post_signal[ch]

        # Split into pre and post segments
        margin = int(5 * sfreq)  # 5-second margin around transition
        pre_data = data[:, : transition_sample - margin]
        post_data = data[:, transition_sample + margin :]

        ch_names = [f"EEG{i+1:02d}" for i in range(n_channels)]

        segments = [
            EEGSegment(
                data=pre_data,
                sfreq=sfreq,
                channel_names=ch_names,
                state_label=pre_label,
                transition_type=transition_type,
                metadata={"synthetic": True, "period": "pre"},
            ),
            EEGSegment(
                data=post_data,
                sfreq=sfreq,
                channel_names=ch_names,
                state_label=post_label,
                transition_type=transition_type,
                metadata={"synthetic": True, "period": "post"},
            ),
        ]

        return segments

    def extract_transition_windows(
        self,
        segment: EEGSegment,
        window_duration_s: float = 10.0,
        step_s: float = 2.0,
    ) -> list[EEGSegment]:
        """
        Slide a window across a segment, creating sub-segments for
        tracking measures over time.

        Parameters
        ----------
        segment : EEGSegment
            Source segment.
        window_duration_s : float
            Window length in seconds.
        step_s : float
            Step size in seconds.

        Returns
        -------
        list[EEGSegment]
            List of windowed segments.
        """
        window_samples = int(window_duration_s * segment.sfreq)
        step_samples = int(step_s * segment.sfreq)

        windows = []
        for start in range(0, segment.n_samples - window_samples, step_samples):
            end = start + window_samples
            win_data = segment.data[:, start:end]

            windows.append(
                EEGSegment(
                    data=win_data,
                    sfreq=segment.sfreq,
                    channel_names=segment.channel_names,
                    state_label=segment.state_label,
                    transition_type=segment.transition_type,
                    metadata={
                        **segment.metadata,
                        "window_start_s": start / segment.sfreq,
                        "window_end_s": end / segment.sfreq,
                    },
                )
            )

        return windows
