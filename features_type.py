"""Functions to load or set the type of features of the databases."""

import pandas as pd
import os
from time import time

import database
from database.constants import METADATA_PATH


backup_dir = 'backup/'
os.makedirs(backup_dir, exist_ok=True)


def _ask_feature_type_df(df):
    """Ask to the user the types of the feature of the data frame.

    Available types:
        0 - Categorical
        1 - Ordinal
        2 - Continue • real
        3 - Continue • integer
        4 - Date timestamp
        5 - Date exploded
        6 - Binary
        -1 - Not a feature

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame from which to determine the feature (=columns) types.

    Returns:
    --------
    pandas.Series
        Series with df.columns as index and integers as values.

    """
    types = pd.Series(0, index=df.columns)

    print(
        '\n'
        ' -------------------------------------------------------\n'
        '|Set the type of features in the data frame.            |\n'
        '|-------------------------------------------------------|\n'
        '|Type an integer in [-1, 6] to set a category.          |\n'
        '|Leave empty to select default choice [bracketed].      |\n'
        '|Type "end" to exit and set all unaswered to default.   |\n'
        '--------------------------------------------------------'
    )

    for i, feature in enumerate(df.columns):
        # Ask the feature type to the user
        while True:
            t = input(
                f'\n\n'
                f'Feature (ID: {i}): {feature}\n\n'
                f'Type? [0 - Categorical]\n'
                f'       1 - Ordinal\n'
                f'       2 - Continue • real\n'
                f'       3 - Continue • integer\n'
                f'       4 - Date timestamp\n'
                f'       5 - Date exploded\n'
                f'       6 - Binary\n'
                f'      -1 - Not a feature\n'
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
            if isinstance(t, int) and t <= 6 and t >= -1:
                break  # t matchs all conditions, so break the loop

            print('\nError: enter an integer in [-1, 6] or type "end".')

        types[feature] = t

    return types


def ask_feature_type_helper():
    """Implement helper for asking feature type to the user."""
    NHIS, TB = database.NHIS(), database.TB()
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

        available_df_names = db.df_names()

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

        df = db[df_name]
        types = _ask_feature_type_df(df)
        _dump_feature_types(types, db, df_name, anonymize=False)


def _dump_feature_types(types, db, df_name, anonymize=True):
    """Dump the features' types anonymizing the features' names.

    Parameters:
    -----------
    types: pandas.Series
        Series with features' names as index and features' types as values.
    db : Database object
        Used to dump results in the right folder.
    df_name : string
        Name or path of the data frame from which has been computed the types.
        Used to dump the results in the right folder.
    anonymize : bool
        Whether to anonymize feature names or not when dumping.
        False: features' name is dumped. True: only id is dumped.

    """
    if df_name in db.frame_paths:
        path = db.frame_paths[df_name]
    else:
        path = df_name
    filename = os.path.basename(path)
    basename, _ = os.path.splitext(filename)
    dir_path = f'{METADATA_PATH}features_types/{db.acronym}/'

    # Anonymize features' names
    if anonymize:
        types = pd.Series(types.values, index=range(len(types.index)))

    # Creates directories if doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save to csv
    filepath = dir_path+basename
    types.to_csv(f'{filepath}.csv', header=False)

    # Backup all dumps in the same folder
    backup_tag = f'{filepath.replace("/", "_")}_{time():.1f}'
    types.to_csv(f'{backup_dir}{backup_tag}.csv', header=False)


def _load_feature_types(db, df_name, anonymized=True):
    """Load the features' types deanonymizing the features' names.

    Parameters:
    -----------
    db : Database class
        The features' database.
    df_name : string
        Name of the features' data frame.
    anonymized : bool
        Whether the features have been anonymized before being dumped
        (i.e no feature names but only their id).

    Returns:
    --------
    pandas.Series
        Series with features' names as index and features' types as values.

    """
    filename = os.path.basename(db.frame_paths[df_name])
    basename, _ = os.path.splitext(filename)
    filepath = f'{METADATA_PATH}features_types/{db.acronym}/{basename}.csv'

    # Load types series
    types = pd.read_csv(filepath, index_col=0,
                        header=None, squeeze=True)

    # Deanonymize features' names
    if anonymized:
        types = pd.Series(types.values, index=db[df_name].columns)

    return types

if __name__ == '__main__':
    from database import NHIS, TB
    ask_feature_type_helper()
    # types = _ask_feature_type_df(NHIS['family'])
    # _dump_feature_types(types, NHIS, 'family')
    # print(_load_feature_types(NHIS, 'family'))
    # print(_load_feature_types(TB, '20000'))
