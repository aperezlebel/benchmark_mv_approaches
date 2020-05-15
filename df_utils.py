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
        return df.mask(b, value)


    if isinstance(df, dict):
        if keys is None:
            return {k: fill(df[k], b[k], value) for k in df.keys()}

        df_filled = dict()
        for k in keys:
            df_filled[k] = fill(df[k], b[k], value)

        for k in df.keys():
            if k not in df_filled:
                df_filled[k] = df[k].copy()

        return df_filled

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


def set_dtypes_features(df, groups, dtypes):
    """Split the columns of a df according to their given group.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame to be splitted.
    groups : pandas.Series
        Series with the features' names or indexs as index and the group as
        values.
    group : scalar
        Scalar to look for in the groups series. The dtype will be set on the
        matched features.
    dtypes: python type or np.dtype
        Matched features will be set to this type

    Returns:
    --------
    pandas.DataFrame
        Data frame with casted dtypes.

    """
    _dtypes = dict()

    for group, dtype in dtypes.items():
        # Get the names of the features to set the dtype
        features = groups[groups == group].index
        for f in features:
            _dtypes[f] = dtype

    return df.astype(_dtypes)


def dtype_from_types(types, type_to_dtype):
    dtype = dict()

    for f, t in types.items():
        if t in type_to_dtype:
            dtype[f] = type_to_dtype[t]

    return dtype


def get_columns(df):
    if isinstance(df, pd.DataFrame):
        return df.columns
    elif isinstance(df, pd.Series):
        return df.index
    else:
        raise ValueError(f'non supported type {type(df)}')
