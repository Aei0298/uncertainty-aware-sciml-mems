"""
Utilities for Monte Carlo sampling and uncertainty analysis.

This module contains functions used to generate uncertain physical,
environmental, and sensor parameters for repeated MEMS simulations.
"""

from typing import Any

import numpy as np
import pandas as pd


def sample_bounded_normal(
    rng: np.random.Generator,
    mean: float,
    standard_deviation: float,
    lower_bound: float,
    upper_bound: float,
    size: int,
) -> np.ndarray:
    """
    Sample values from a normal distribution while enforcing
    lower and upper physical bounds.

    Parameters
    ----------
    rng:
        NumPy random-number generator.

    mean:
        Mean of the normal distribution.

    standard_deviation:
        Standard deviation of the normal distribution.

    lower_bound:
        Smallest allowed value.

    upper_bound:
        Largest allowed value.

    size:
        Number of random samples to generate.

    Returns
    -------
    np.ndarray
        Array containing bounded random samples.
    """

    # First generate samples from the requested normal distribution.
    samples = rng.normal(
        loc=mean,
        scale=standard_deviation,
        size=size,
    )

    # Identify all samples that fall outside the allowed range.
    invalid_mask = (
        (samples < lower_bound)
        | (samples > upper_bound)
    )

    # Resample only the invalid values.
    #
    # This loop continues until every generated value satisfies
    # the specified physical bounds.
    while np.any(invalid_mask):

        number_of_invalid_samples = np.sum(invalid_mask)

        samples[invalid_mask] = rng.normal(
            loc=mean,
            scale=standard_deviation,
            size=number_of_invalid_samples,
        )

        invalid_mask = (
            (samples < lower_bound)
            | (samples > upper_bound)
        )

    return samples


def sample_uniform(
    rng: np.random.Generator,
    lower_bound: float,
    upper_bound: float,
    size: int,
) -> np.ndarray:
    """
    Sample values from a uniform distribution.

    Every value between the lower and upper bounds has the
    same probability of being sampled.

    Parameters
    ----------
    rng:
        NumPy random-number generator.

    lower_bound:
        Smallest possible sampled value.

    upper_bound:
        Largest possible sampled value.

    size:
        Number of random samples to generate.

    Returns
    -------
    np.ndarray
        Array containing uniformly distributed samples.
    """

    samples = rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=size,
    )

    return samples


def sample_from_distribution(
    rng: np.random.Generator,
    distribution_config: dict[str, Any],
    size: int,
) -> np.ndarray:
    """
    Generate random samples using the probability distribution
    described in a configuration dictionary.

    Supported distributions:
    - normal
    - uniform

    Parameters
    ----------
    rng:
        NumPy random-number generator.

    distribution_config:
        Dictionary containing the distribution type and
        its required parameters.

    size:
        Number of random samples to generate.

    Returns
    -------
    np.ndarray
        Array containing the generated random samples.
    """

    # Read which probability distribution should be used.
    distribution_type = distribution_config["distribution"]


    # NORMAL DISTRIBUTION

    if distribution_type == "normal":

        return sample_bounded_normal(
            rng=rng,
            mean=distribution_config["mean"],
            standard_deviation=distribution_config[
                "standard_deviation"
            ],
            lower_bound=distribution_config["minimum"],
            upper_bound=distribution_config["maximum"],
            size=size,
        )


    # UNIFORM DISTRIBUTION

    if distribution_type == "uniform":

        return sample_uniform(
            rng=rng,
            lower_bound=distribution_config["minimum"],
            upper_bound=distribution_config["maximum"],
            size=size,
        )


    # UNSUPPORTED DISTRIBUTION

    raise ValueError(
        f"Unsupported distribution type: {distribution_type}"
    )


def generate_monte_carlo_samples(
    rng: np.random.Generator,
    uncertain_parameters: dict[str, Any],
    measurement_uncertainty: dict[str, Any],
    number_of_simulations: int,
) -> dict[str, np.ndarray]:
    """
    Generate all uncertain physical, environmental, and
    measurement parameters required for a Monte Carlo study.

    Each returned array contains one sampled value for every
    Monte Carlo simulation.

    Parameters
    ----------
    rng:
        NumPy random-number generator.

    uncertain_parameters:
        Configuration dictionary for uncertain physical and
        environmental parameters.

    measurement_uncertainty:
        Configuration dictionary for uncertain sensor and
        measurement parameters.

    number_of_simulations:
        Number of Monte Carlo samples to generate.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing one array for each uncertain
        parameter.
    """

    samples = {}


    # PHYSICAL AND ENVIRONMENTAL PARAMETERS

    for parameter_name, parameter_config in (
        uncertain_parameters.items()
    ):

        samples[parameter_name] = sample_from_distribution(
            rng=rng,
            distribution_config=parameter_config,
            size=number_of_simulations,
        )


    # MEASUREMENT PARAMETERS

    for parameter_name, parameter_config in (
        measurement_uncertainty.items()
    ):

        samples[parameter_name] = sample_from_distribution(
            rng=rng,
            distribution_config=parameter_config,
            size=number_of_simulations,
        )


    return samples


def calculate_uncertainty_summary(
    data,
    columns,
    lower_percentile=2.5,
    upper_percentile=97.5,
):
    """
    Calculate summary uncertainty statistics for selected
    Monte Carlo output variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Monte Carlo result table.

    columns : list of str
        Output variables to summarize.

    lower_percentile : float
        Lower percentile of the empirical uncertainty interval.

    upper_percentile : float
        Upper percentile of the empirical uncertainty interval.

    Returns
    -------
    pandas.DataFrame
        Table containing the mean, standard deviation,
        coefficient of variation, minimum, maximum,
        and empirical uncertainty interval.
    """

    import numpy as np
    import pandas as pd


    summary_rows = []


    for column_name in columns:

        values = data[
            column_name
        ].to_numpy()


        mean_value = np.mean(
            values
        )

        standard_deviation = np.std(
            values,
            ddof=1,
        )


        # Coefficient of variation expresses the
        # standard deviation relative to the mean.
        if mean_value != 0.0:

            coefficient_of_variation = (
                standard_deviation
                / abs(mean_value)
                * 100.0
            )

        else:

            coefficient_of_variation = np.nan


        lower_interval = np.percentile(
            values,
            lower_percentile,
        )

        upper_interval = np.percentile(
            values,
            upper_percentile,
        )


        summary_rows.append(
            {
                "output": column_name,
                "mean": mean_value,
                "standard_deviation":
                    standard_deviation,
                "coefficient_of_variation_percent":
                    coefficient_of_variation,
                "minimum": np.min(values),
                "lower_95_percentile":
                    lower_interval,
                "upper_95_percentile":
                    upper_interval,
                "maximum": np.max(values),
            }
        )


    return pd.DataFrame(
        summary_rows
    )




def calculate_spearman_sensitivity(
    data,
    input_columns,
    output_columns,
):
    """
    Calculate Spearman rank correlations between uncertain
    inputs and Monte Carlo outputs.

    Absolute correlation magnitude provides a simple
    global sensitivity indicator for monotonic relationships.

    Parameters
    ----------
    data : pandas.DataFrame
        Monte Carlo result table.

    input_columns : list of str
        Uncertain input parameters.

    output_columns : list of str
        Quantities of interest.

    Returns
    -------
    pandas.DataFrame
        Sensitivity matrix with inputs as rows and
        outputs as columns.
    """

    sensitivity_matrix = pd.DataFrame(
        index=input_columns,
        columns=output_columns,
        dtype=float,
    )


    for input_name in input_columns:

        for output_name in output_columns:

            sensitivity_matrix.loc[
                input_name,
                output_name,
            ] = data[
                input_name
            ].corr(
                data[output_name],
                method="spearman",
            )


    return sensitivity_matrix