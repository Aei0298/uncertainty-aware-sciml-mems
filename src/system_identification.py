"""
Classical system-identification utilities.

This module contains reusable functions for estimating the
physical parameters of the linear MEMS dynamical system

    m*x_ddot + c*x_dot + k*x = F(t)

from time-series data.
"""

import numpy as np


def build_regression_matrix(
    acceleration,
    velocity,
    displacement,
):
    """
    Construct the regression matrix for the linear
    mass-damper-stiffness model.

    Each row has the form:

        [acceleration, velocity, displacement]

    corresponding to:

        F = m*a + c*v + k*x
    """

    return np.column_stack(
        (
            acceleration,
            velocity,
            displacement,
        )
    )


def estimate_linear_parameters(
    acceleration,
    velocity,
    displacement,
    excitation,
):
    """
    Estimate mass, damping, and stiffness using
    ordinary least squares.

    Returns
    -------
    estimated_parameters : numpy.ndarray
        Parameter vector:

            [mass, damping, stiffness]

    residuals : numpy.ndarray
        Least-squares residual information.

    rank : int
        Rank of the regression matrix.

    singular_values : numpy.ndarray
        Singular values of the regression matrix.
    """

    regression_matrix = build_regression_matrix(
        acceleration=acceleration,
        velocity=velocity,
        displacement=displacement,
    )

    estimated_parameters, residuals, rank, singular_values = (
        np.linalg.lstsq(
            regression_matrix,
            excitation,
            rcond=None,
        )
    )

    return (
        estimated_parameters,
        residuals,
        rank,
        singular_values,
    )


def reconstruct_force(
    mass,
    damping,
    stiffness,
    acceleration,
    velocity,
    displacement,
):
    """
    Reconstruct the excitation force using identified
    physical parameters.

    The reconstructed force is

        F_hat = m*a + c*v + k*x
    """

    return (
        mass * acceleration
        + damping * velocity
        + stiffness * displacement
    )


def calculate_condition_number(
    regression_matrix,
    normalize_columns=False,
):
    """
    Calculate the condition number of a regression matrix.

    If normalize_columns is True, each column is divided by
    its Euclidean norm before the condition number is calculated.
    This helps separate numerical scaling effects from genuine
    dependence between the regressors.
    """

    matrix = np.asarray(
        regression_matrix,
        dtype=float,
    )


    if normalize_columns:

        column_norms = np.linalg.norm(
            matrix,
            axis=0,
        )

        if np.any(column_norms == 0.0):
            raise ValueError(
                "Regression matrix contains a zero-norm column."
            )

        matrix = (
            matrix
            / column_norms
        )


    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )


    condition_number = (
        singular_values[0]
        / singular_values[-1]
    )


    return condition_number