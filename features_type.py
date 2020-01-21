"""Functions to load or set the type of features of the databases."""

import pandas as pd

import NHIS


def _ask_feature_type_df(df):
    """Ask to the user the types of the feature of the data frame.

    Available types:
        0 - Categorical
        1 - Ordinal
        2 - Continue

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame from which to determine the feature (=columns) types.

    Returns:
    --------
    pandas.Series
        Series with df.columns as index and integer type as values (0, 1 or 2).

    """
    types = pd.Series(0, index=df.columns)

    print(
        '\n'
        ' -------------------------------------------------------\n'
        '|Set the type of features in the data frame.            |\n'
        '|-------------------------------------------------------|\n'
        '|Type an integer in {0, 1, 2} to set a category.        |\n'
        '|Leave empty to select default choice [bracketed].      |\n'
        '|Type "end" to exit and set all unaswered to default.   |\n'
        '--------------------------------------------------------'
    )

    for feature in df.columns:
        # Ask the feature type to the user
        while True:
            t = input(
                f'\n\n'
                f'Feature: {feature}\n\n'
                f'Type? [0 - Categorical]\n'
                f'       1 - Ordinal\n'
                f'       2 - Continue\n'
            )

            # By typing 'end', the unanswered are set to default
            if t == 'end':
                return types

            # Convert empty (default) to 0 - Categorical
            if t == '':
                t = 0

            # Try to convert user's input to integer
            try:
                t = int(t)
            except ValueError:
                pass

            # Check if the integer is in the good range
            if isinstance(t, int) and t <= 2 and t >= 0:
                break  # t matchs all conditions, so break the loop

            print('\nError: enter an integer in {0, 1, 2} or type \'end\'.')

        types[feature] = t

    return types


if __name__ == '__main__':
    print(_ask_feature_type_df(NHIS.db['family']))
