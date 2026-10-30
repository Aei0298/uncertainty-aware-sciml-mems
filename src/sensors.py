"""

Measurement and sensor models for the simplified MEMS gyroscope study.

The physical dynamical response is generated elsewhere. This module models
how the true physical signal is transformed into a measured signal through
sensor gain, bias, and measurement noise.

Generic measurement model:

    y_measured(t) = gain * y_true(t) + bias + noise(t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorParameters:
    """
    Parameters describing the measurement system.

    Parameters
    ----------
    gain : float
        Multiplicative sensor gain.
    bias : float
        Additive sensor bias.
    noise_standard_deviation : float
        Standard deviation of additive Gaussian measurement noise.
    """

    gain: float = 1.0
    bias: float = 0.0
    noise_standard_deviation: float = 0.0

    def validate(self) -> None:
        """
        Validate sensor parameters.

        Raises
        ------
        ValueError
            If invalid sensor values are provided.
        """

        if self.gain <= 0:
            raise ValueError("Sensor gain must be greater than zero.")

        if self.noise_standard_deviation < 0:
            raise ValueError(
                "Noise standard deviation must be non-negative."
            )


def apply_gain(
    true_signal: np.ndarray,
    gain: float,
) -> np.ndarray:
    """
    Apply multiplicative sensor gain.

    Parameters
    ----------
    true_signal : np.ndarray
        Ground-truth physical signal.
    gain : float
        Multiplicative gain.

    Returns
    -------
    np.ndarray
        Gain-adjusted signal.
    """

    if gain <= 0:
        raise ValueError("Gain must be greater than zero.")

    return gain * np.asarray(true_signal, dtype=float)


def apply_bias(
    signal: np.ndarray,
    bias: float,
) -> np.ndarray:
    """
    Add sensor bias to a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    bias : float
        Additive sensor bias.

    Returns
    -------
    np.ndarray
        Bias-adjusted signal.
    """

    return np.asarray(signal, dtype=float) + bias


def add_gaussian_noise(
    signal: np.ndarray,
    standard_deviation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add zero-mean Gaussian measurement noise.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    standard_deviation : float
        Standard deviation of Gaussian noise.
    rng : np.random.Generator
        NumPy random-number generator.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """

    if standard_deviation < 0:
        raise ValueError(
            "Noise standard deviation must be non-negative."
        )

    signal_array = np.asarray(signal, dtype=float)

    if standard_deviation == 0:
        return signal_array.copy()

    noise = rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=signal_array.shape,
    )

    return signal_array + noise


def measure_signal(
    true_signal: np.ndarray,
    parameters: SensorParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Transform a true physical signal into a measured sensor signal.

    Measurement model:

        y_measured = gain * y_true + bias + noise

    Parameters
    ----------
    true_signal : np.ndarray
        Ground-truth physical signal.
    parameters : SensorParameters
        Sensor-model parameters.
    rng : np.random.Generator
        NumPy random-number generator.

    Returns
    -------
    np.ndarray
        Simulated sensor measurement.
    """

    parameters.validate()

    measured_signal = apply_gain(
        true_signal=true_signal,
        gain=parameters.gain,
    )

    measured_signal = apply_bias(
        signal=measured_signal,
        bias=parameters.bias,
    )

    measured_signal = add_gaussian_noise(
        signal=measured_signal,
        standard_deviation=parameters.noise_standard_deviation,
        rng=rng,
    )

    return measured_signal