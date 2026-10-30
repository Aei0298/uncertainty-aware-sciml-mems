"""
Core dynamical models for the simplified MEMS gyroscope study.

The initial model is a forced mass-spring-damper system with an optional
cubic stiffness term:

    m*x_ddot + c*x_dot + k*x + alpha*x^3 = F(t)

This module contains only the physical system dynamics. Excitation signals,
sensor effects, simulation orchestration, and uncertainty are handled in
separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MEMSParameters:
    """
    Physical parameters of the simplified MEMS dynamical system.

    Parameters
    ----------
    mass : float
        Effective mass m.
    damping : float
        Damping coefficient c.
    stiffness : float
        Linear stiffness coefficient k.
    nonlinear_stiffness : float, optional
        Cubic stiffness coefficient alpha. Set to zero for the linear model.
    """

    mass: float
    damping: float
    stiffness: float
    nonlinear_stiffness: float = 0.0

    def validate(self) -> None:
        """
        Validate the physical parameter values.

        Raises
        ------
        ValueError
            If physically required positive parameters are invalid.
        """

        if self.mass <= 0:
            raise ValueError("Mass must be greater than zero.")

        if self.damping < 0:
            raise ValueError("Damping must be non-negative.")

        if self.stiffness <= 0:
            raise ValueError("Stiffness must be greater than zero.")


def natural_angular_frequency(parameters: MEMSParameters) -> float:
    """
    Calculate the undamped natural angular frequency.

    omega_n = sqrt(k / m)

    Returns
    -------
    float
        Natural angular frequency in rad/s.
    """

    parameters.validate()

    return np.sqrt(parameters.stiffness / parameters.mass)


def natural_frequency(parameters: MEMSParameters) -> float:
    """
    Calculate the undamped natural frequency.

    f_n = (1 / 2*pi) * sqrt(k / m)

    Returns
    -------
    float
        Natural frequency in Hz.
    """

    omega_n = natural_angular_frequency(parameters)

    return omega_n / (2.0 * np.pi)


def damping_ratio(parameters: MEMSParameters) -> float:
    """
    Calculate the damping ratio.

    zeta = c / (2 * sqrt(k*m))

    Returns
    -------
    float
        Dimensionless damping ratio.
    """

    parameters.validate()

    denominator = 2.0 * np.sqrt(
        parameters.stiffness * parameters.mass
    )

    return parameters.damping / denominator


def restoring_force(
    displacement: float,
    parameters: MEMSParameters,
) -> float:
    """
    Calculate the restoring force contribution.

    Restoring force:

        F_restore = k*x + alpha*x^3

    Parameters
    ----------
    displacement : float
        Current displacement x.
    parameters : MEMSParameters
        Physical system parameters.

    Returns
    -------
    float
        Restoring-force contribution.
    """

    linear_force = parameters.stiffness * displacement

    nonlinear_force = (
        parameters.nonlinear_stiffness * displacement**3
    )

    return linear_force + nonlinear_force


def mems_state_derivative(
    time: float,
    state: np.ndarray,
    parameters: MEMSParameters,
    excitation_function: Callable[[float], float],
) -> np.ndarray:
    """
    Compute the state derivative of the MEMS system.

    State vector:

        state = [x, v]

    where:

        x = displacement
        v = velocity

    Governing equation:

        m*x_ddot + c*x_dot + k*x + alpha*x^3 = F(t)

    State-space representation:

        x_dot = v

        v_dot = (
            F(t)
            - c*v
            - k*x
            - alpha*x^3
        ) / m

    Parameters
    ----------
    time : float
        Current simulation time.
    state : np.ndarray
        Current state [displacement, velocity].
    parameters : MEMSParameters
        Physical model parameters.
    excitation_function : Callable[[float], float]
        Function returning the excitation force at the current time.

    Returns
    -------
    np.ndarray
        State derivative [x_dot, v_dot].
    """

    parameters.validate()

    displacement, velocity = state

    excitation_force = excitation_function(time)

    restoring = restoring_force(
        displacement=displacement,
        parameters=parameters,
    )

    damping_force = parameters.damping * velocity

    acceleration = (
        excitation_force
        - damping_force
        - restoring
    ) / parameters.mass

    return np.array(
        [velocity, acceleration],
        dtype=float,
    )