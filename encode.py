"""Functions to encode a data frame (OrdinalEncode, OneHotEncode)..."""

import yaml
from time import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from database.constants import NOT_MISSING, BINARY


def _df_type_handler(function, df_seq, keys=None, **kwargs):
    if not df_seq:
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

    return function(df_seq, **kwargs)


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

        # Fit transform the encoder
        data_encoded = enc.fit_transform(df)

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


def one_hot_encode(df, mv, types, keys=None):

    def encode(df, mv, types):
        enc = OneHotEncoder(sparse=False)

        # Fit transform the encoder
        data_encoded = enc.fit_transform(df)

        feature_names = list(enc.get_feature_names(list(df.columns)))

        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index,
                                  columns=feature_names
                                  )

        mv_encoded = pd.DataFrame(NOT_MISSING*np.ones(data_encoded.shape),
                                  index=df.index,
                                  columns=feature_names)

        types = pd.Series(BINARY, index=feature_names)

        return df_encoded, mv_encoded, types

    return _df_type_handler(encode, (df, mv, types), keys=keys)
