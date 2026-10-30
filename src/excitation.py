"""
Excitation-signal definitions for the MEMS dynamical-system simulations.

Excitation generation is kept separate from the physical dynamics so that
different test-signal strategies can be evaluated without changing the
governing system model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def sinusoidal_excitation(
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
) -> Callable[[float], float]:
    """
    Create a sinusoidal excitation function.

    F(t) = A * sin(2*pi*f*t + phase)

    Parameters
    ----------
    amplitude : float
        Excitation amplitude.
    frequency : float
        Excitation frequency in Hz.
    phase : float, optional
        Phase offset in radians.

    Returns
    -------
    Callable[[float], float]
        Excitation function F(t).
    """

    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")

    def excitation(time: float) -> float:
        return amplitude * np.sin(
            2.0 * np.pi * frequency * time + phase
        )

    return excitation


def chirp_excitation(
    amplitude: float,
    start_frequency: float,
    end_frequency: float,
    duration: float,
) -> Callable[[float], float]:
    """
    Create a linear-frequency chirp excitation.

    The instantaneous frequency changes linearly from start_frequency
    to end_frequency over the specified duration.

    Parameters
    ----------
    amplitude : float
        Excitation amplitude.
    start_frequency : float
        Initial frequency in Hz.
    end_frequency : float
        Final frequency in Hz.
    duration : float
        Sweep duration in seconds.

    Returns
    -------
    Callable[[float], float]
        Chirp excitation function F(t).
    """

    if duration <= 0:
        raise ValueError("Duration must be greater than zero.")

    if start_frequency < 0 or end_frequency < 0:
        raise ValueError("Frequencies must be non-negative.")

    sweep_rate = (
        end_frequency - start_frequency
    ) / duration

    def excitation(time: float) -> float:
        phase = 2.0 * np.pi * (
            start_frequency * time
            + 0.5 * sweep_rate * time**2
        )

        return amplitude * np.sin(phase)

    return excitation


def multi_sine_excitation(
    amplitudes: list[float],
    frequencies: list[float],
    phases: list[float] | None = None,
) -> Callable[[float], float]:
    """
    Create a multi-sine excitation function.

    F(t) = sum_i A_i * sin(2*pi*f_i*t + phi_i)

    Parameters
    ----------
    amplitudes : list[float]
        Excitation amplitudes.
    frequencies : list[float]
        Frequencies in Hz.
    phases : list[float] or None, optional
        Phase offsets in radians. If omitted, all phases are zero.

    Returns
    -------
    Callable[[float], float]
        Multi-sine excitation function F(t).
    """

    if len(amplitudes) != len(frequencies):
        raise ValueError(
            "amplitudes and frequencies must have the same length."
        )

    if any(frequency < 0 for frequency in frequencies):
        raise ValueError("Frequencies must be non-negative.")

    if phases is None:
        phases = [0.0] * len(frequencies)

    if len(phases) != len(frequencies):
        raise ValueError(
            "phases and frequencies must have the same length."
        )

    amplitudes_array = np.asarray(amplitudes, dtype=float)
    frequencies_array = np.asarray(frequencies, dtype=float)
    phases_array = np.asarray(phases, dtype=float)

    def excitation(time: float) -> float:
        terms = amplitudes_array * np.sin(
            2.0 * np.pi * frequencies_array * time
            + phases_array
        )

        return float(np.sum(terms))

    return excitation