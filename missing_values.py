"""Function for detecting missing values."""
import pandas as pd

from database import NHIS


def get_missing_values(df, heuristic):
    """Determine the type of missing value present in the given data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame storing the input table from which to determine the type
        of missing values.
    heuristic : function with pandas.Series -> pandas.Series signature
        The heuristic according to which are determined the type of missing
        values. Given a column of df stored as a pandas.Series, the heuristic
        returns a pandas.Series storing the type of missing values encountered.

    Returns
    -------
    pandas.DataFrame
        A data frame with same index and columns as the input one but storing
        the type of missing values encountered (0: Not a missing value,
        1: Not applicable, 2: Not available).

    """
    # Compute the Series storing the types of missing values
    columns = [heuristic(df.iloc[:, index]) for index in range(df.shape[1])]
    # Concat the Series into a data frame
    df_mv = pd.concat(columns, axis=1, sort=False)

    return df_mv


if __name__ == '__main__':
    print(get_missing_values(NHIS['family'], NHIS.heuristic))
    print(get_missing_values(NHIS['child'], NHIS.heuristic))
    print(get_missing_values(NHIS['adult'], NHIS.heuristic))
    print(get_missing_values(NHIS['person'], NHIS.heuristic))
    print(get_missing_values(NHIS['injury'], NHIS.heuristic))
    print(get_missing_values(NHIS['household'], NHIS.heuristic))
