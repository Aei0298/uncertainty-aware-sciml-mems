"""
Equation-discovery utilities for the MEMS dynamical system.

This module contains reusable functions for constructing
candidate-function libraries and performing sparse regression
for governing-equation discovery.
"""

import numpy as np


def build_candidate_library(
    displacement,
    velocity,
    excitation,
):
    """
    Construct the candidate-function library used for
    sparse equation discovery.

    The current library contains:

        1
        x
        x_dot
        F
        x^2
        x*x_dot
        x_dot^2
        x^3

    Returns
    -------
    candidate_library : numpy.ndarray
        Matrix containing the candidate functions.

    candidate_names : list[str]
        Human-readable names of the candidate terms.
    """

    candidate_library = np.column_stack(
        (
            np.ones_like(displacement),
            displacement,
            velocity,
            excitation,
            displacement ** 2,
            displacement * velocity,
            velocity ** 2,
            displacement ** 3,
        )
    )

    candidate_names = [
        "1",
        "x",
        "x_dot",
        "F",
        "x^2",
        "x*x_dot",
        "x_dot^2",
        "x^3",
    ]

    return candidate_library, candidate_names


def normalize_candidate_library(
    candidate_library,
):
    """
    Normalize every candidate-library column by its
    Euclidean norm.

    Returns
    -------
    normalized_library : numpy.ndarray
        Candidate matrix with unit-norm columns.

    column_norms : numpy.ndarray
        Original Euclidean norm of every column.
    """

    column_norms = np.linalg.norm(
        candidate_library,
        axis=0,
    )

    if np.any(column_norms == 0.0):
        raise ValueError(
            "Candidate library contains a zero-norm column."
        )

    normalized_library = (
        candidate_library
        / column_norms
    )

    return (
        normalized_library,
        column_norms,
    )


def sequential_threshold_least_squares(
    normalized_library,
    target,
    threshold,
    max_iterations=20,
):
    """
    Perform Sequentially Thresholded Least Squares (STLSQ).

    Small coefficients are removed iteratively and the
    remaining candidate terms are refitted.

    Returns
    -------
    coefficients : numpy.ndarray
        Sparse coefficients on the normalized feature scale.

    active_terms : numpy.ndarray
        Boolean mask showing which candidate terms remain.
    """

    coefficients = np.linalg.lstsq(
        normalized_library,
        target,
        rcond=None,
    )[0]

    active_terms = np.ones(
        normalized_library.shape[1],
        dtype=bool,
    )

    for _ in range(max_iterations):

        new_active_terms = (
            np.abs(coefficients)
            >= threshold
        )

        if np.array_equal(
            new_active_terms,
            active_terms,
        ):
            break

        active_terms = new_active_terms

        if not np.any(active_terms):
            break

        reduced_coefficients = np.linalg.lstsq(
            normalized_library[:, active_terms],
            target,
            rcond=None,
        )[0]

        coefficients = np.zeros(
            normalized_library.shape[1]
        )

        coefficients[active_terms] = (
            reduced_coefficients
        )

    return coefficients, active_terms


def convert_to_physical_coefficients(
    normalized_coefficients,
    column_norms,
):
    """
    Convert normalized regression coefficients back to
    the original physical feature scale.
    """

    return (
        normalized_coefficients
        / column_norms
    )



def standardize_data(
    values,
    axis=0,
):
    """
    Standardize numerical data to zero mean and unit
    standard deviation.

    Returns
    -------
    standardized_values : numpy.ndarray
        Standardized data.

    mean_values : numpy.ndarray
        Mean used for standardization.

    standard_deviation_values : numpy.ndarray
        Standard deviation used for standardization.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    mean_values = np.mean(
        values,
        axis=axis,
    )

    standard_deviation_values = np.std(
        values,
        axis=axis,
    )

    if np.any(
        standard_deviation_values == 0.0
    ):
        raise ValueError(
            "Cannot standardize data with zero standard deviation."
        )

    standardized_values = (
        values
        - mean_values
    ) / standard_deviation_values

    return (
        standardized_values,
        mean_values,
        standard_deviation_values,
    )


def inverse_standardize(
    standardized_values,
    mean_values,
    standard_deviation_values,
):
    """
    Transform standardized values back to their
    original numerical scale.
    """

    return (
        standardized_values
        * standard_deviation_values
        + mean_values
    )


def refine_linear_structure(
    input_matrix,
    target,
):
    """
    Estimate the optimal coefficients of a fixed
    linear symbolic structure using least squares.
    """

    coefficients = np.linalg.lstsq(
        input_matrix,
        target,
        rcond=None,
    )[0]

    return coefficients