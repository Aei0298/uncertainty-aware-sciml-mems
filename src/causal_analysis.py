import numpy as np

from sklearn.ensemble import RandomForestRegressor


def residualize_treatment_and_outcome(
    treatment,
    outcome,
    adjustment_variables,
    random_state=9512,
    n_estimators=300,
):
    """
    Residualize both treatment and outcome with respect to
    the selected adjustment variables.

    This implements a simple machine-learning-based
    residualization approach.

    Parameters
    ----------
    treatment : numpy.ndarray
        Treatment variable.

    outcome : numpy.ndarray
        Outcome variable.

    adjustment_variables : numpy.ndarray
        Matrix of adjustment variables.

    random_state : int
        Random seed for reproducibility.

    n_estimators : int
        Number of trees used in the nuisance models.

    Returns
    -------
    treatment_residual : numpy.ndarray
        Treatment variation not explained by the
        adjustment variables.

    outcome_residual : numpy.ndarray
        Outcome variation not explained by the
        adjustment variables.

    treatment_model : RandomForestRegressor

    outcome_model : RandomForestRegressor
    """

    adjustment_variables = np.asarray(
        adjustment_variables,
        dtype=float,
    )

    if adjustment_variables.ndim == 1:
        adjustment_variables = (
            adjustment_variables.reshape(-1, 1)
        )


    treatment_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    outcome_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


    treatment_model.fit(
        adjustment_variables,
        treatment,
    )

    outcome_model.fit(
        adjustment_variables,
        outcome,
    )


    predicted_treatment = treatment_model.predict(
        adjustment_variables
    )

    predicted_outcome = outcome_model.predict(
        adjustment_variables
    )


    treatment_residual = (
        treatment
        - predicted_treatment
    )

    outcome_residual = (
        outcome
        - predicted_outcome
    )


    return (
        treatment_residual,
        outcome_residual,
        treatment_model,
        outcome_model,
    )


def estimate_residualized_causal_slope(
    treatment_residual,
    outcome_residual,
):
    """
    Estimate the slope relating residualized treatment
    variation to residualized outcome variation.

    The regression is fitted through the origin because
    both residualized variables are centered around zero.
    """

    numerator = np.sum(
        treatment_residual
        * outcome_residual
    )

    denominator = np.sum(
        treatment_residual ** 2
    )


    if denominator == 0.0:
        raise ValueError(
            "Treatment residual has zero variation."
        )


    causal_slope = (
        numerator
        / denominator
    )

    return causal_slope


def calculate_physics_intervention_effect(
    stiffness,
    mass,
    stiffness_change,
):
    """
    Calculate the average physical intervention effect
    of changing stiffness by a fixed amount.

    The natural frequency is evaluated before and after
    the intervention

        do(k = k + stiffness_change).

    Parameters
    ----------
    stiffness : numpy.ndarray
        Original stiffness values.

    mass : numpy.ndarray
        Mass values.

    stiffness_change : float
        Fixed intervention applied to stiffness.

    Returns
    -------
    baseline_frequency : numpy.ndarray
        Natural frequency before intervention.

    intervened_frequency : numpy.ndarray
        Natural frequency after intervention.

    individual_effect : numpy.ndarray
        Frequency change for each simulated system.

    average_effect : float
        Mean intervention effect over the population.
    """

    stiffness = np.asarray(
        stiffness,
        dtype=float,
    )

    mass = np.asarray(
        mass,
        dtype=float,
    )


    baseline_frequency = (
        1.0
        / (2.0 * np.pi)
        * np.sqrt(
            stiffness / mass
        )
    )


    intervened_stiffness = (
        stiffness
        + stiffness_change
    )


    if np.any(
        intervened_stiffness <= 0.0
    ):
        raise ValueError(
            "Intervened stiffness must remain positive."
        )


    intervened_frequency = (
        1.0
        / (2.0 * np.pi)
        * np.sqrt(
            intervened_stiffness / mass
        )
    )


    individual_effect = (
        intervened_frequency
        - baseline_frequency
    )


    average_effect = np.mean(
        individual_effect
    )


    return (
        baseline_frequency,
        intervened_frequency,
        individual_effect,
        average_effect,
    )