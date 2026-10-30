"""

Simulation orchestration for the simplified MEMS gyroscope study.

This module connects:

- the physical dynamical model,
- excitation signals,
- numerical ODE integration,
- derived physical quantities,
- the sensor measurement model.

It is intended to provide a reproducible interface for one deterministic
simulation run. Monte Carlo logic is implemented separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import MEMSParameters, mems_state_derivative
from .sensors import SensorParameters, measure_signal


@dataclass
class SimulationResult:
    """
    Container for the output of one MEMS simulation.

    Attributes
    ----------
    time : np.ndarray
        Simulation time vector.
    displacement : np.ndarray
        True displacement response.
    velocity : np.ndarray
        True velocity response.
    acceleration : np.ndarray
        True acceleration response.
    excitation : np.ndarray
        Applied excitation force.
    measured_signal : np.ndarray
        Simulated sensor measurement.
    """

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    excitation: np.ndarray
    measured_signal: np.ndarray


def create_time_vector(
    start_time: float,
    end_time: float,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Create a uniformly sampled time vector.

    Parameters
    ----------
    start_time : float
        Initial simulation time in seconds.
    end_time : float
        Final simulation time in seconds.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Uniform time vector.
    """

    if end_time <= start_time:
        raise ValueError(
            "end_time must be greater than start_time."
        )

    if sampling_frequency <= 0:
        raise ValueError(
            "sampling_frequency must be greater than zero."
        )

    duration = end_time - start_time

    number_of_samples = int(
        np.floor(duration * sampling_frequency)
    ) + 1

    return np.linspace(
        start_time,
        end_time,
        number_of_samples,
    )


def calculate_acceleration(
    time: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    parameters: MEMSParameters,
    excitation_function: Callable[[float], float],
) -> np.ndarray:
    """
    Calculate true acceleration from the governing equation.

    Parameters
    ----------
    time : np.ndarray
        Simulation time vector.
    displacement : np.ndarray
        True displacement response.
    velocity : np.ndarray
        True velocity response.
    parameters : MEMSParameters
        Physical model parameters.
    excitation_function : Callable[[float], float]
        Excitation force function.

    Returns
    -------
    np.ndarray
        True acceleration response.
    """

    acceleration = np.empty_like(
        displacement,
        dtype=float,
    )

    for index, current_time in enumerate(time):

        excitation_force = excitation_function(
            current_time
        )

        restoring_force = (
            parameters.stiffness * displacement[index]
            + parameters.nonlinear_stiffness
            * displacement[index] ** 3
        )

        damping_force = (
            parameters.damping * velocity[index]
        )

        acceleration[index] = (
            excitation_force
            - damping_force
            - restoring_force
        ) / parameters.mass

    return acceleration


def evaluate_excitation(
    time: np.ndarray,
    excitation_function: Callable[[float], float],
) -> np.ndarray:
    """
    Evaluate the excitation signal on the simulation time vector.

    Parameters
    ----------
    time : np.ndarray
        Simulation time vector.
    excitation_function : Callable[[float], float]
        Excitation force function.

    Returns
    -------
    np.ndarray
        Excitation values at each simulation time.
    """

    return np.array(
        [
            excitation_function(current_time)
            for current_time in time
        ],
        dtype=float,
    )


def simulate_mems_system(
    parameters: MEMSParameters,
    excitation_function: Callable[[float], float],
    sensor_parameters: SensorParameters,
    start_time: float,
    end_time: float,
    sampling_frequency: float,
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
    solver_method: str = "RK45",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    random_seed: int | None = None,
    sensor_output: str = "acceleration",
) -> SimulationResult:
    """
    Simulate one deterministic MEMS experiment.

    Parameters
    ----------
    parameters : MEMSParameters
        Physical model parameters.
    excitation_function : Callable[[float], float]
        Applied excitation force.
    sensor_parameters : SensorParameters
        Measurement-system parameters.
    start_time : float
        Simulation start time in seconds.
    end_time : float
        Simulation end time in seconds.
    sampling_frequency : float
        Output sampling frequency in Hz.
    initial_displacement : float, optional
        Initial displacement.
    initial_velocity : float, optional
        Initial velocity.
    solver_method : str, optional
        ODE solver used by scipy.integrate.solve_ivp.
    rtol : float, optional
        Relative solver tolerance.
    atol : float, optional
        Absolute solver tolerance.
    random_seed : int or None, optional
        Seed used for sensor noise generation.
    sensor_output : str, optional
        Physical quantity observed by the simulated sensor.

        Supported values:

        - "displacement"
        - "velocity"
        - "acceleration"

    Returns
    -------
    SimulationResult
        Complete output of one simulation run.
    """

    parameters.validate()
    sensor_parameters.validate()

    time = create_time_vector(
        start_time=start_time,
        end_time=end_time,
        sampling_frequency=sampling_frequency,
    )

    initial_state = np.array(
        [
            initial_displacement,
            initial_velocity,
        ],
        dtype=float,
    )

    def ode_function(
        current_time: float,
        current_state: np.ndarray,
    ) -> np.ndarray:

        return mems_state_derivative(
            time=current_time,
            state=current_state,
            parameters=parameters,
            excitation_function=excitation_function,
        )

    solution = solve_ivp(
        fun=ode_function,
        t_span=(start_time, end_time),
        y0=initial_state,
        t_eval=time,
        method=solver_method,
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            "ODE solver failed: "
            f"{solution.message}"
        )

    displacement = solution.y[0]
    velocity = solution.y[1]

    acceleration = calculate_acceleration(
        time=time,
        displacement=displacement,
        velocity=velocity,
        parameters=parameters,
        excitation_function=excitation_function,
    )

    excitation = evaluate_excitation(
        time=time,
        excitation_function=excitation_function,
    )

    if sensor_output == "displacement":
        true_sensor_signal = displacement

    elif sensor_output == "velocity":
        true_sensor_signal = velocity

    elif sensor_output == "acceleration":
        true_sensor_signal = acceleration

    else:
        raise ValueError(
            "sensor_output must be one of: "
            "'displacement', 'velocity', or 'acceleration'."
        )

    rng = np.random.default_rng(random_seed)

    measured_signal = measure_signal(
        true_signal=true_sensor_signal,
        parameters=sensor_parameters,
        rng=rng,
    )

    return SimulationResult(
        time=time,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        excitation=excitation,
        measured_signal=measured_signal,
    )