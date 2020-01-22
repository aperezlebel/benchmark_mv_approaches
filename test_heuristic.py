"""Manually test the accuracy of the heuristics."""

import numpy as np

from database import NHIS
from missing_values import get_missing_values


def test_missing_values(df, df_mv, N=10):
    """Manually check if the missing value type is accurate.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame storing the original values.
    df_mv : pandas.DataFrame
        Data frame storing the missing values types (0: not a missing value,
        1: Not applicable, 2: Not available).
    N : int
        Number of random draws of cells for each type of missing values.

    Returns
    -------
    list
        The scores obtained based on the manual checks done by the user, i.e
        the number of good guess for each type of missing values. 0: everything
        is wrong, N: everything is right.

    """
    scores = [0, 0, 0]

    for mv_value in [0, 1, 2]:
        # Get coordinates of every cells matching the wanted value.
        row_ids, col_ids = np.where(df_mv == mv_value)

        # Skip if no missing values found of this type
        if len(row_ids) == 0:
            continue

        # Draw a subset of this coordinates
        rand_range = np.random.choice(range(len(row_ids)), N, replace=False)

        for i in rand_range:
            row_id, col_id = row_ids[i], col_ids[i]  # Selected coordinates
            value = df.iloc[row_id, col_id]  # Value in the original data frame
            col_name = df.columns[col_id]  # Name of the selected column

            print(
                f'\nFound: {value}\n'
                f'At: {col_name} ({row_id}, {col_id})\n'
                f'Assigned: {mv_value}\n'
            )

            if input('Correct?') == '':
                scores[mv_value] += 1

    print(f'\nScores with N={N}:\n{scores}')

    return scores


if __name__ == '__main__':
    df_mv = get_missing_values(NHIS['family'], NHIS.heuristic)
    test_missing_values(NHIS['family'], df_mv)
