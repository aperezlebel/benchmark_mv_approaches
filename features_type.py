"""Functions to load or set the type of features of the databases."""

import pandas as pd
import os
from time import time

import NHIS
import TB


backup_dir = 'backup/'
os.makedirs(backup_dir, exist_ok=True)


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


def ask_feature_type_helper():
    """Implement helper for asking feature type to the user."""
    available_db_names = [db.acronym for db in [NHIS, TB]]

    while True:
        # Prevent from asking again when user failed on second input
        if 'db_name' not in locals():
            db_name = input(
                f'\n'
                f'Which database do you want to set the features\' types?\n'
                f'Available choices: {available_db_names}\n'
                f'Type "exit" to end.\n'
            )

        # Load appropiate database
        if db_name == NHIS.acronym:
            db = NHIS
        elif db_name == TB.acronym:
            db = TB
        elif db_name == 'exit':
            return
        else:
            print(f'\nAnswer must be in {available_db_names}')
            del db_name
            continue

        available_df_names = list(db.db.keys())

        df_name = input(
            f'\n'
            f'Which data frame of {db_name} do you want to set the '
            f'features\' types?\n'
            f'Available: {available_df_names}\n'
            f'Type "none" to change database.\n'
            f'Type "exit" to end.\n'
        )

        if df_name == 'none':
            del db_name
            continue

        if df_name == 'exit':
            return

        if df_name not in available_df_names:
            print(f'\nAnswer must be in {available_df_names}')
            continue

        df = db.db[df_name]
        types = _ask_feature_type_df(df)
        _dump_feature_types(types, db, df_name)


def _dump_feature_types(types, db, df_name):
    """Dump the feature types anonymizing the features' names."""
    dir_path = f'metadata/features_types/{db.acronym}/'

    # Anonymize features' names
    anonymized_types = pd.Series(types.values, index=range(len(types.index)))

    # Creates directories if doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save to csv
    filepath = dir_path+df_name
    anonymized_types.to_csv(f'{filepath}.csv', header=False)

    # Backup all dumps in the same folder
    backup_tag = f'{filepath.replace("/", "_")}_{time():.1f}'
    anonymized_types.to_csv(f'{backup_dir}{backup_tag}.csv', header=False)


def _load_feature_types(db, df_name):
    """Load the feature types deanonymizing the features' names."""
    filepath = f'metadata/features_types/{db.acronym}/{df_name}.csv'

    # Load types series
    anonymized_types = pd.read_csv(filepath, index_col=0,
                                   header=None, squeeze=True)

    # Deanonymize features' names and return
    return pd.Series(anonymized_types.values, index=db.db[df_name].columns)


if __name__ == '__main__':
    ask_feature_type_helper()
    # types = _ask_feature_type_df(NHIS.db['family'])
    # _dump_feature_types(types, NHIS, 'family')
    # print(_load_feature_types(NHIS, 'family'))
    # print(_load_feature_types(TB, '20000'))
