"""
Signal-processing utilities for the MEMS sensor analysis.

This module contains reusable functions for time-domain and
frequency-domain analysis of simulated MEMS signals.
"""

import numpy as np

from scipy import signal



def calculate_rms(
    signal_values: np.ndarray,
) -> float:
    """
    Calculate the root mean square (RMS) value of a signal.

    RMS provides a measure of the overall magnitude of a
    time-varying signal.

    Parameters
    ----------
    signal_values:
        One-dimensional array containing the signal samples.

    Returns
    -------
    float
        Root mean square value of the signal.
    """

    rms_value = np.sqrt(
        np.mean(
            signal_values ** 2
        )
    )

    return float(rms_value)




def calculate_fft_spectrum(
    signal_values: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the one-sided Fast Fourier Transform (FFT)
    magnitude spectrum of a real-valued signal.

    Parameters
    ----------
    signal_values:
        One-dimensional array containing the signal samples.

    sampling_frequency:
        Number of signal samples recorded per second.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequencies and corresponding one-sided FFT magnitudes.
    """

    # Number of samples contained in the signal.
    number_of_samples = len(signal_values)


    # Calculate the discrete Fourier transform.
    fft_values = np.fft.rfft(
        signal_values
    )


    # Calculate the frequency corresponding to every
    # FFT coefficient.
    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sampling_frequency,
    )


    # Convert the complex FFT coefficients into magnitudes.
    #
    # The factor 2 / N provides a one-sided amplitude scaling
    # for the positive-frequency spectrum.
    magnitudes = (
        2.0
        / number_of_samples
        * np.abs(fft_values)
    )


    return frequencies, magnitudes



def calculate_power_spectral_density(
    signal_values: np.ndarray,
    sampling_frequency: float,
    segment_length: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Power Spectral Density (PSD) of a signal
    using Welch's method.

    Parameters
    ----------
    signal_values:
        One-dimensional array containing the signal samples.

    sampling_frequency:
        Number of signal samples recorded per second.

    segment_length:
        Number of samples used in each Welch segment.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequencies and corresponding power spectral density values.
    """

    frequencies, power_spectral_density = signal.welch(
        signal_values,
        fs=sampling_frequency,
        nperseg=segment_length,
    )

    return frequencies, power_spectral_density


def find_dominant_frequency(
    frequencies: np.ndarray,
    spectral_values: np.ndarray,
    minimum_frequency: float = 0.0,
) -> tuple[float, float]:
    """
    Find the dominant frequency in a frequency-domain spectrum.

    The dominant frequency is defined as the frequency
    corresponding to the largest spectral value above the
    specified minimum frequency.

    Parameters
    ----------
    frequencies:
        Array containing frequency values.

    spectral_values:
        Array containing FFT magnitudes or PSD values.

    minimum_frequency:
        Frequencies below this value are ignored.

    Returns
    -------
    tuple[float, float]
        Dominant frequency and corresponding spectral value.
    """

    # Select only frequencies at or above the chosen
    # minimum frequency.
    valid_mask = frequencies >= minimum_frequency

    valid_frequencies = frequencies[
        valid_mask
    ]

    valid_spectral_values = spectral_values[
        valid_mask
    ]


    # Find the position of the largest spectral value.
    dominant_index = np.argmax(
        valid_spectral_values
    )


    # Obtain the corresponding frequency and magnitude/power.
    dominant_frequency = valid_frequencies[
        dominant_index
    ]

    dominant_spectral_value = valid_spectral_values[
        dominant_index
    ]


    return (
        float(dominant_frequency),
        float(dominant_spectral_value),
    )


