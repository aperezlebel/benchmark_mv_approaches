"""Functions to encode a data frame (OrdinalEncode, OneHotEncode)..."""

import yaml
from time import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from datetime import datetime

from database.constants import NOT_MISSING, BINARY, CONTINUE_I, MV_PLACEHOLDER
from df_utils import fill_df


def _df_type_handler(function, df_seq, keys=None, **kwargs):
    if not isinstance(keys, list):
        keys = [keys]

    if not df_seq:  # Empty df_seq
        return function(**kwargs)

    n_df = len(df_seq)
    if isinstance(df_seq[0], dict):
        res = tuple([dict() for k in range(n_df)])

        for k in df_seq[0].keys():
            if keys is None or (keys is not None and k in keys):
                r = function(*(df_seq[i][k] for i in range(n_df)), **kwargs)
                for i in range(n_df):
                    res[i][k] = r[i]
            else:
                for i in range(n_df):
                    res[i][k] = df_seq[i][k].copy()

        return res

    if isinstance(df_seq[0], list):
        new_df_seq = [dict(enumerate(df)) for df in df_seq]
        df_encoded, mv_encoded = _df_type_handler(function,
                                                  new_df_seq,
                                                  keys=keys,
                                                  **kwargs)
        return list(df_encoded.values()), list(mv_encoded.values())

    return function(*df_seq, **kwargs)


def ordinal_encode(df, mv, keys=None, order=None):

    def encode(df, mv, order=None):
        categories = 'auto'

        if order is not None:
            categories = []
            for feature_name in df.columns:
                if feature_name in order:
                    feature_order = order[feature_name]
                else:
                    print(
                        f'INFO: ordinal order for {feature_name} not found. '
                        f'Derived from unique values found.')
                    feature_order = list(np.unique(df[feature_name].values))
                categories.append(feature_order)

        enc = OrdinalEncoder(categories=categories)

        df = fill_df(df, mv != NOT_MISSING, MV_PLACEHOLDER)
        # df = df.fillna(MV_PLACEHOLDER).astype(str)  # Cast to str
        # Fit transform the encoder
        data_encoded = enc.fit_transform(df)
        df = fill_df(df, mv != NOT_MISSING, np.nan)

        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index, columns=df.columns)

        # print(enc.categories_)
        # order_list = []
        # for i, feature_name in enumerate(df.columns):
        #     order_list.append(list(enc.categories_[i]))
        #     print(f'{feature_name}\n\t{enc.categories_[i]}')

        # document = yaml.dump(order_list, allow_unicode=True)
        # print(document)
        # with open(f'order_{time()}.yml', 'w') as file:
        #     file.write(document)

        return df_encoded, mv

    return _df_type_handler(encode, (df, mv), keys, order=order)


def one_hot_encode(df, mv, types, parent, keys=None):

    def encode(df, mv, types, parent):
        enc = OneHotEncoder(sparse=False)

        # Cast to str to prevent: "argument must be a string or number" error
        # which occurs when mixed types floats and str
        df = df.astype(str)

        # Fill missing values with a placeholder
        df = fill_df(df, mv != NOT_MISSING, MV_PLACEHOLDER)

        # Fit transform the encoder
        data_encoded = enc.fit_transform(df)

        df = fill_df(df, mv != NOT_MISSING, np.nan)

        feature_names = list(enc.get_feature_names(list(df.columns)))

        parent = pd.Series()

        for i, c in enumerate(df.columns):
            for suffix in enc.categories_[i]:
                parent[f'{c}_{suffix}'] = c

        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index,
                                  columns=feature_names
                                  )

        mv_encoded = pd.DataFrame(NOT_MISSING*np.ones(data_encoded.shape),
                                  index=df.index,
                                  columns=feature_names)

        types_encoded = pd.Series(BINARY, index=feature_names)

        return df_encoded, mv_encoded, types_encoded, parent

    return _df_type_handler(encode, (df, mv, types, parent), keys=keys)


def date_encode(df, mv, types, parent, keys=None, method='timestamp', dayfirst=False):

    def encode(df, mv, types, parent, method='timestamp', dayfirst=False):
        df = fill_df(df, mv != NOT_MISSING, np.nan)

        if method == 'timestamp':
            data = dict()

            for feature_name in df.columns:
                dt_series = pd.to_datetime(df[feature_name], dayfirst=dayfirst)
                dt_min = np.datetime64(dt_series.min())
                tdt = np.timedelta64(1, 'D')
                data[feature_name] = np.subtract(dt_series.values, dt_min)/tdt

            df_encoded = pd.DataFrame(data, index=df.index)
            mv_encoded = mv
            types_encoded = pd.Series(CONTINUE_I, index=df_encoded.columns)

        elif method == 'explode':
            df_data = dict()
            mv_data = dict()
            parent = pd.Series()

            for feature_name in df.columns:
                dt = pd.to_datetime(df[feature_name], dayfirst=dayfirst).dt

                df_data[f'{feature_name}_year'] = dt.year
                df_data[f'{feature_name}_month'] = dt.month
                df_data[f'{feature_name}_day'] = dt.day

                mv_data[f'{feature_name}_year'] = mv[feature_name]
                mv_data[f'{feature_name}_month'] = mv[feature_name]
                mv_data[f'{feature_name}_day'] = mv[feature_name]

                parent[f'{feature_name}_year'] = feature_name
                parent[f'{feature_name}_month'] = feature_name
                parent[f'{feature_name}_day'] = feature_name

            df_encoded = pd.DataFrame(df_data, index=df.index)
            mv_encoded = pd.DataFrame(mv_data, index=df.index)
            types_encoded = pd.Series(CONTINUE_I, index=df_encoded.columns)

        return df_encoded, mv_encoded, types_encoded, parent

    return _df_type_handler(encode, (df, mv, types, parent), keys=keys, method=method,
                            dayfirst=dayfirst)
