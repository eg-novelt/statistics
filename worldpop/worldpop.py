
from collections.abc import Callable
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import arviz as az
from arviz import InferenceData


def sample_or_load(
    sample_func: Callable[[], InferenceData],
    cache_file: Path,
    clean : bool = False 
) -> InferenceData:
    """Load cached inference results or run sampling and cache them.

    Args:
        sample_func: Zero-argument function that runs sampling and
            returns an ArviZ InferenceData object.
        cache_file: Path to the NetCDF file used to cache the results.

    Returns:
        The loaded or newly sampled InferenceData.
    """
    cache_file = Path(cache_file)

    # Reuse existing results to avoid rerunning expensive sampling.
    if cache_file.exists() and not clean:
        return az.from_netcdf(cache_file)

    # Run sampling only when no cached results exist.
    idata = sample_func()

    # Persist the results so they can be reused in later runs.
    idata.to_netcdf(cache_file)

    return idata


def score_metrics(df, N_pred):

    # micro census population
    N = df["N"].to_numpy()

    # Residual = predicted population - observed population.
    # Positive: overprediction; negative: underprediction.
    residual = N_pred - N

    # Bias: mean signed residual.
    # Ideal value: 0.
    bias = residual.mean()

    # Imprecision: standard deviation of residuals.
    # Smaller values indicate less variation in prediction errors.
    imprecision = residual.std(ddof=0)

    # Inaccuracy: mean absolute residual (MAE).
    # Measures average absolute error in number of people.
    inaccuracy = np.abs(residual).mean()

    # R²: squared Pearson correlation between observed and predicted counts.
    # Measures association, not necessarily calibration or absolute accuracy.
    r_squared = pearsonr(N, N_pred).statistic**2

    # Scaled residuals: residual divided by predicted population.
    # These express errors relative to the prediction.
    # Exclude zero or near-zero predictions to avoid division by zero
    # and unstable ratios.
    valid = N_pred > 0
    scaled_residual = residual[valid] / N_pred[valid]

    # Scaled bias, imprecision, and inaccuracy.
    scaled_bias = scaled_residual.mean()
    scaled_imprecision = scaled_residual.std(ddof=0)
    scaled_inaccuracy = np.abs(scaled_residual).mean()

    # Display all metrics in a table.
    metrics = pd.Series({
        "Bias (people)": bias,
        "Imprecision (people)": imprecision,
        "Inaccuracy / MAE (people)": inaccuracy,
        "R²": r_squared,
        "Scaled bias": scaled_bias,
        "Scaled imprecision": scaled_imprecision,
        "Scaled inaccuracy": scaled_inaccuracy,
    }, name="Value")

    return metrics

def prepare_hierachy_indices(df):

    # Region index: each unique region gets an integer index.
    # Original region IDs may be non-contiguous or start at 1.
    # Zero-based region indices.
    df["region_idx"], region_ids = pd.factorize(df["region"], sort=False)

    # Zero-based state indices, unique within each region.
    state_keys = pd.Series(list(zip(df["region"], df["state"])))
    df["state_idx"], state_ids = pd.factorize(state_keys, sort=False)

    # Zero-based local indices, unique within each state and region.
    local_keys = pd.Series(list(zip(df["region"], df["state"], df["local"])))
    df["local_idx"], local_ids = pd.factorize(local_keys, sort=False)

    
