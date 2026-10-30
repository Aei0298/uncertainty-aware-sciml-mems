"""
Reusable machine-learning functions.
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def create_time_aware_split(
    input_data,
    target,
    time,
    train_fraction=0.80,
):
    """
    Split time-series data without random shuffling.

    The first portion is used for training and the
    remaining portion is used for testing.
    """

    split_index = int(
        train_fraction
        * len(input_data)
    )

    return (
        input_data[:split_index],
        input_data[split_index:],
        target[:split_index],
        target[split_index:],
        time[:split_index],
        time[split_index:],
    )


def train_linear_regression(
    X_train,
    y_train,
):
    """
    Train an ordinary linear regression model.
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_polynomial_regression(
    X_train,
    y_train,
    degree=2,
):
    """
    Train polynomial regression using a scikit-learn pipeline.
    """

    model = Pipeline(
        steps=[
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False,
                ),
            ),
            (
                "linear_regression",
                LinearRegression(),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_random_forest(
    X_train,
    y_train,
    n_estimators=200,
    random_state=9512,
):
    """
    Train a Random Forest regression model.
    """

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def calculate_regression_metrics(
    y_true,
    y_prediction,
):
    """
    Calculate common regression-performance metrics.
    """

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_prediction,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_prediction,
    )

    r2 = r2_score(
        y_true,
        y_prediction,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }