"""Operations on pandas data frame."""

import pandas as pd
import numpy as np


def split_features(df, groups):
    """Split the columns of a df according to their given group.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame to be splitted.
    groups : pandas.Series
        Series with the features' names or indexs as index and the group as
        values.

    Returns:
    --------
    dict
        Dictionnary with groups as keys and sub data frame as values.

    """
    sub_df = dict()

    for group_id in groups.unique():
        # Get the names of the features to drop
        features_to_drop = groups[groups != group_id].index
        # Get the data frame without those features
        if isinstance(df, pd.Series):
            sub_df[group_id] = df.drop(features_to_drop)
        else:
            sub_df[group_id] = df.drop(features_to_drop, 1)

    return sub_df


def fill_df(df, b, value, keys=None):

    def fill(df, b, value):
        for col in b.columns:
            df[col][b[col]] = value

        return df

    if isinstance(df, dict):
        if keys is None:
            return {k: fill(df[k], b[k], value) for k in df.keys()}

        df_encoded = dict()
        for k in keys:
            df_encoded[k] = fill(df[k], b[k], value)

        for k in df.keys():
            if k not in df_encoded:
                df_encoded[k] = df[k].copy()

        return df_encoded

    if isinstance(df, list):
        return [fill(v, b[k], value) for k, v in enumerate(df)]

    return fill(df, b, value)


def rint_features(df, features):
    """Inplace round elements of selected columns to nearest integer.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame with columns to be rounded.
    groups : pandas.Series
        Series with the features' names as index and boolean values where True
        encode the features to round.

    Returns:
    --------
    pandas.DataFrame
        Data frame with rounded columns.

    """
    features_to_round = features[features].index

    df_rounded = df.copy()

    for feature_name in features_to_round:
        df_rounded[feature_name] = np.rint(df[feature_name])

    return df_rounded


if __name__ == '__main__':
    from database import TB
    print(split_features(TB['20000'], TB.feature_types['20000']))
