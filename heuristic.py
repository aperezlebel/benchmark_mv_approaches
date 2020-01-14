"""Implement heuristics to determine what are the missing values in the DB."""

import pandas as pd
import numpy as np


def NHIS(series):
    """Implement the heuritic for detecting NHIS missing values.

    Parameters
    ----------
    series : pandas.Series
        One column of the NHIS dataframe, stored as a pandas.Series object.

    Returns
    -------
    pandas.Series
        A series with same name and index as input series but having values
        in [0, 1, 2] encoding respectively: Not a missing value,
        Not applicable, Not available.

    """
    # The series storing the type of missing values
    series_mv = pd.Series(0, index=series.index, name=series.name)

    # Type 1 missing values
    series_mv[series.isna()] = 1  # Type 1 missing values

    # Type 2 missing values
    # This rule is not applicable for mixed types columns
    if series.dtype == np.object:
        return series_mv

    val_max = series.max()

    if val_max < 10:  # Type 1 column
        series_mv[series == 7] = 2
        series_mv[series == 8] = 2
        series_mv[series == 9] = 2

    if val_max < 100:  # Type 2 column
        series_mv[series == 97] = 2
        series_mv[series == 98] = 2
        series_mv[series == 99] = 2

    return series_mv
