# SPDX-FileCopyrightText: Idiap Research Institute
# SPDX-FileContributor: Enno Hermann <enno.hermann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Functions for speech analysis."""

import importlib.resources
import math
from collections.abc import Callable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.signal import lfilter

plt.rcParams["figure.facecolor"] = "w"

SAMPLING_FREQUENCY = 16000

FIG_WIDTH = 15
FIG_HEIGHT = 5


def load_signal(filename: str) -> npt.NDArray[np.float64]:
    """Load a speech signal from the given file."""
    included_fn = importlib.resources.files("idiap_spe").joinpath(
        f"data/speech_analysis/{filename}"
    )
    if included_fn.is_file():
        filename = str(included_fn)
    with open(filename) as f:
        lines = [int(line.rstrip()) for line in f]
        return np.array(lines[16:], dtype="int64")  # Actual data starts at line 17


def _plot_grid(ax: Axes) -> None:
    """Add a grid to the plot."""
    ax.minorticks_on()
    ax.grid()
    ax.grid(which="minor", ls=":")


def plot_amplitude(
    signal: npt.NDArray[np.float64], ax: Axes, title: str = "", *, grid: bool = False
) -> None:
    """Plot amplitude of a signal."""
    if grid:
        _plot_grid(ax)
    ax.set_title(title)
    ax.set_xlabel("Time in samples")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(signal))
    ax.set_ylim(-32768, 32767)
    ax.plot(signal, lw=1)


def plot_frequency(
    frequencies: npt.ArrayLike,
    amplitudes: npt.NDArray[np.float64],
    ax: Axes,
    title: str = "",
) -> None:
    """Plot frequency of a signal."""
    _plot_grid(ax)
    ax.set_title(title)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Log amplitude")
    ax.set_xlim(0, max(np.atleast_1d(frequencies)))
    ax.plot(frequencies, amplitudes, lw=1)


def speech_signal_observation(
    filename: str, title: str = "Speech signal"
) -> npt.NDArray[np.float64]:
    """Load a speech signal from the given file and plot amplitude and log energy."""
    data = load_signal(filename)
    _, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
    plt.subplots_adjust(hspace=0.3)

    plot_amplitude(data, axs[0], title)

    # Plot energy
    frame_shift = 64
    frame_size = 256
    max_frames = math.floor((len(data) - frame_size + frame_shift) / frame_shift)
    energy = np.zeros(max_frames)

    for i in range(max_frames):
        loc = i * frame_shift
        window = data[loc : loc + frame_size]
        energy[i] = np.dot(window.T, window)

    log_energy = 10 * np.log10(energy)
    log_energy_norm = log_energy - np.mean(log_energy)

    axs[1].set_title("Short-time energy plot of the above utterance")
    axs[1].set_xlabel("Time in terms of frames")
    axs[1].set_ylabel("Log energy")
    axs[1].set_xlim(0, len(log_energy_norm))
    axs[1].set_ylim(0, max(log_energy_norm))
    axs[1].plot(log_energy_norm, ".")

    return data


def select_speech(
    data: npt.NDArray[np.float64],
    begin: int,
    end: int,
    title: str = "Windowed signal",
    *,
    plot: bool = True,
) -> npt.NDArray[np.float64]:
    """Return a window of the given speech signal and plot it.

    The data between frame numbers `begin` (inclusive) and `end` (exclusive) is
    returned.
    """
    window = data[begin:end]
    if plot:
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax = fig.gca()
        plot_amplitude(window, ax, title, grid=True)
    return window


def autocorrelation(
    data: npt.NDArray[np.float64],
    max_lags: int,
    title: str = "Autocorrelation",
    *,
    plot: bool = True,
) -> npt.NDArray[np.float64]:
    """Return and plot the autocorrelation of the given signal."""
    # Compute autocorrelation
    n = len(data)
    correlation = np.correlate(data, data, mode="full")
    correlation = correlation[n - 1 - max_lags : n + max_lags]

    # Plot autocorrelation
    if plot:
        fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        plt.subplots_adjust(hspace=0.3)
        for i, x in enumerate([correlation, correlation[max_lags + 1 :]]):
            axs[i].minorticks_on()
            axs[i].grid()
            axs[i].grid(which="minor", ls=":")
            axs[i].set_xlabel("Lag number")
            axs[i].set_ylabel("Amplitude")
            axs[i].set_xlim((0, len(x)))
            axs[i].plot(x)

        axs[0].set_title(title)
        axs[1].set_title("Autocorrelation (right half)")

    return correlation


def _compute_log_fourier(
    data: npt.NDArray[np.float64], order: int = 512
) -> npt.NDArray[np.float64]:
    """Compute the log Fourier spectrum."""
    log_fourier: npt.NDArray[np.float64] = 20 * np.log10(
        np.abs(np.fft.fft(data, n=order))
    )
    return log_fourier


def fourier_spectrum(
    data: npt.NDArray[np.float64],
    order: int = 512,
    sf: int = SAMPLING_FREQUENCY,
    title: str = "Fourier spectrum",
) -> None:
    """Compute and plot the fourier spectrum of the given signal."""
    # Compute fourier spectrum
    log_fourier = _compute_log_fourier(data, order)
    frequencies = [n * sf / (order * 1000) for n in range(order)]

    # Plot fourier spectrum
    _fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
    plt.subplots_adjust(hspace=0.3)
    for i, (freq, amplitudes, plot_title) in enumerate(
        [
            (frequencies, log_fourier, title),
            (
                frequencies[: order // 2],
                log_fourier[: order // 2],
                "Fourier spectrum (left half)",
            ),
        ]
    ):
        plot_frequency(freq, amplitudes, axs[i], plot_title)


def spectrogram(
    data: npt.NDArray[np.float64],
    order: int,
    window: Callable[[int], npt.NDArray[np.float64]] = np.hanning,
    sf: int = SAMPLING_FREQUENCY,
    title: str = "Spectrogram",
) -> None:
    """Compute and plot the spectrogram of the given signal."""

    def _window_fun(x: npt.ArrayLike) -> npt.ArrayLike:
        a: npt.NDArray[np.float64] = np.array(x)
        return window(len(a)) * a

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.specgram(data, NFFT=order, window=_window_fun, Fs=sf, cmap="magma")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")


def preemphasize(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pre-emphasize to remove the spectral tilt due to glottal pulse spectrum."""
    diff_data = np.empty_like(data)
    diff_data[0] = data[0]
    diff_data[1:] = np.diff(data)
    return diff_data


def lp_spectrum(
    data: npt.NDArray[np.float64],
    lp_order: int,
    order: int,
    window: Callable[[int], npt.NDArray[np.float64]] = np.hanning,
    *,
    sf: int = SAMPLING_FREQUENCY,
    plot: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute and plot the LP spectrum."""
    windowed_data = preemphasize(data)
    windowed_data = window(len(windowed_data)) * windowed_data

    frequencies = [n * sf / (order * 1000) for n in range(order // 2)]

    # Computation of LP and Fourier spectrum
    coefficients = librosa.lpc(windowed_data, order=lp_order)
    lp_spec: npt.NDArray[np.float64] = -20.0 * np.log10(
        np.abs(np.fft.fft(coefficients, order))
    )
    log_fourier = _compute_log_fourier(data, order)

    if plot:
        _fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        plt.subplots_adjust(hspace=0.3)
        for i, (spectrum, title) in enumerate(
            [(log_fourier, "Fourier spectrum"), (lp_spec, "LP spectrum")]
        ):
            plot_frequency(frequencies, spectrum[: order // 2], axs[i], title)
    return lp_spec


def lp_residual(
    data: npt.NDArray[np.float64],
    lp_order: int,
    window: Callable[[int], npt.NDArray[np.float64]] = np.hanning,
    *,
    plot: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute and plot the LP residual."""
    # Pre-emphasize, window the signal and compute LPC
    windowed_data = preemphasize(data)
    windowed_data = window(len(windowed_data)) * windowed_data
    coefficients = librosa.lpc(windowed_data, order=lp_order)

    residual = np.empty_like(data)
    padded_data = np.pad(data, (lp_order, 0))

    for i in range(len(data)):
        predict = 0
        for j in range(1, lp_order + 1):
            predict = predict + coefficients[j] * padded_data[i + lp_order - j]
        residual[i] = padded_data[i + lp_order] + predict

    # Plot signal and residual
    if plot:
        _fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        plt.subplots_adjust(hspace=0.3)
        for i, (signal, title) in enumerate(
            [(data, "Original signal"), (residual, "LP residual signal")]
        ):
            plot_amplitude(signal, axs[i], title, grid=True)
    return residual


def speaker_variation(
    utterances: list[tuple[str, int]],
    lp_order: int = 16,
    sample_length: int = 480,
    fft_order: int = 512,
    sf: int = SAMPLING_FREQUENCY,
) -> None:
    """Plot the LP spectra for a list of utterances."""
    frequencies = [n * sf / (fft_order * 1000) for n in range(fft_order // 2)]
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.gca()
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Log amplitude")
    ax.set_title("Comparison of LP spectra")
    ax.set_xlim(0, sf // 2000)

    for filename, start in utterances:
        st_data = load_signal(filename)[start : start + sample_length]
        spectrum = lp_spectrum(st_data, lp_order, fft_order, plot=False)
        ax.plot(frequencies, spectrum[: fft_order // 2], label=filename)
    ax.legend()


def sift(
    filename: str,
    lp_order: int = 10,
    frame_size: int = 480,
    frame_shift: int = 160,
    sf: int = SAMPLING_FREQUENCY,
) -> list[float]:
    """Compute the pitch contour using the SIFT algorithm."""
    max_lags = 256

    b = [0.0357081667, -0.0069956244, -0.0069956244, 0.0357081667]
    a = [1.0, -2.34036589, 2.01190019, -0.61419218]

    data = load_signal(filename)
    max_frames = int(np.floor((len(data) - frame_size + frame_shift) / frame_shift))
    divisor = 1.8

    pitch = []
    for i in range(max_frames):
        idx = i * frame_shift
        frame = data[idx : idx + frame_size]
        frame = lfilter(b, a, frame)
        residual = lp_residual(frame, lp_order, plot=False)
        residual = autocorrelation(residual, max_lags, plot=False)

        max_residual = residual[max_lags] / divisor
        max_idx = max_lags
        for j in range(max_lags + (sf // 400), max_lags + (sf // 80)):
            if residual[j] > max_residual:
                max_residual = residual[j]
                max_idx = j

        max_idx = max_idx - max_lags
        if max_idx > 0:
            pitch.append(sf / max_idx)
        else:
            pitch.append(0.0)

    _fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
    plt.subplots_adjust(hspace=0.3)

    # Plot signal
    plot_amplitude(data, axs[0], "Speech signal")

    # Plot pitch
    axs[1].set_title("Pitch contour")
    axs[1].set_xlabel("Time in samples")
    axs[1].set_ylabel("Pitch frequency (Hz)")
    axs[1].set_xlim(0, len(pitch))
    axs[1].plot(pitch, ".")

    return pitch
