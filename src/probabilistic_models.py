"""
Reusable probabilistic machine-learning functions
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)


def create_gp_training_subset(
    input_data,
    target,
    time,
    number_of_samples=800,
):
    """
    Create an evenly spaced Gaussian Process training subset.

    Parameters
    ----------
    input_data : numpy.ndarray
        Full training input matrix.

    target : numpy.ndarray
        Full training target vector.

    time : numpy.ndarray
        Full training time vector.

    number_of_samples : int
        Number of evenly spaced samples to retain.

    Returns
    -------
    subset_input : numpy.ndarray

    subset_target : numpy.ndarray

    subset_time : numpy.ndarray

    subset_indices : numpy.ndarray
    """

    subset_indices = np.linspace(
        0,
        len(input_data) - 1,
        number_of_samples,
        dtype=int,
    )

    subset_input = input_data[
        subset_indices
    ]

    subset_target = target[
        subset_indices
    ]

    subset_time = time[
        subset_indices
    ]

    return (
        subset_input,
        subset_target,
        subset_time,
        subset_indices,
    )


def standardize_gp_data(
    X_train,
    X_test,
    y_train,
):
    """
    Standardize Gaussian Process inputs and target.

    Scalers are fitted only on the training data.
    """

    input_scaler = StandardScaler()

    target_scaler = StandardScaler()


    X_train_scaled = input_scaler.fit_transform(
        X_train
    )

    X_test_scaled = input_scaler.transform(
        X_test
    )

    y_train_scaled = (
        target_scaler
        .fit_transform(
            y_train.reshape(-1, 1)
        )
        .ravel()
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        input_scaler,
        target_scaler,
    )


def create_gaussian_process(
    initial_noise_level=1e-6,
    noise_level_bounds=(1e-10, 1e-1),
    random_state=9512,
):
    """
    Create the Gaussian Process model used in the project.
    """

    kernel = (
        ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3),
        )
        *
        RBF(
            length_scale=[
                1.0,
                1.0,
                1.0,
            ],
            length_scale_bounds=(1e-2, 1e3),
        )
        +
        WhiteKernel(
            noise_level=initial_noise_level,
            noise_level_bounds=noise_level_bounds,
        )
    )


    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=False,
        n_restarts_optimizer=2,
        random_state=random_state,
    )

    return model


def convert_gp_prediction_to_physical_units(
    predictive_mean_scaled,
    predictive_std_scaled,
    target_scaler,
):
    """
    Convert Gaussian Process predictions from standardized
    target units back to physical units.
    """

    predictive_mean = (
        target_scaler
        .inverse_transform(
            predictive_mean_scaled.reshape(-1, 1)
        )
        .ravel()
    )

    predictive_std = (
        predictive_std_scaled
        * target_scaler.scale_[0]
    )

    return (
        predictive_mean,
        predictive_std,
    )


def calculate_prediction_interval(
    predictive_mean,
    predictive_std,
    confidence_multiplier=1.96,
):
    """
    Construct a symmetric Gaussian prediction interval.
    """

    lower_bound = (
        predictive_mean
        - confidence_multiplier
        * predictive_std
    )

    upper_bound = (
        predictive_mean
        + confidence_multiplier
        * predictive_std
    )

    return (
        lower_bound,
        upper_bound,
    )


def calculate_interval_coverage(
    observed_values,
    lower_bound,
    upper_bound,
):
    """
    Calculate the empirical fraction of observations
    contained inside a prediction interval.
    """

    coverage = np.mean(
        (observed_values >= lower_bound)
        &
        (observed_values <= upper_bound)
    )

    return coverage


def extract_gp_noise_std(
    gaussian_process_model,
    target_scaler,
):
    """
    Extract the optimized WhiteKernel noise standard
    deviation and convert it to physical target units.
    """

    noise_variance_scaled = (
        gaussian_process_model
        .kernel_
        .k2
        .noise_level
    )

    noise_std_scaled = np.sqrt(
        noise_variance_scaled
    )

    noise_std_physical = (
        noise_std_scaled
        * target_scaler.scale_[0]
    )

    return noise_std_physical