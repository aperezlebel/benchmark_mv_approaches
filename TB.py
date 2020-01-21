"""Gather all TraumaBase related functions."""

import pandas as pd

from encode import ordinal_encode
from missing_values import get_missing_values


def load_database(encode=True):
    """Load the TraumaBase database."""
    data_folder = 'TraumaBase/'

    TB = {
        '20000': pd.read_csv(data_folder+'Traumabase_20000.csv', sep=';')
    }

    # if encode:
    #     df = TB['20000'].copy()
    #     df_mv = get_missing_values(df, TB_heuristic)
    #     df.where(df_mv == 0, other=-1, inplace=True)
    #     df_encoded, _ = ordinal_encode(df)
    #     df_encoded[df_mv != 0] = np.nan
    #     TB['20000'] = df_encoded

    return TB


def heuristic(series):
    """Implement the heuristic for detecting TraumaBase missing values.

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

    series_mv[series.isna()] = 2
    series_mv[series == 'NA'] = 2
    series_mv[series == 'ND'] = 2

    return series_mv


# TraumaBase parameters
db = load_database(encode=True)
