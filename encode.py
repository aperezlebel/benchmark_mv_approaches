"""Functions to encode a data frame (OrdinalEncode, OneHotEncode)..."""

import yaml
from time import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from database.constants import NOT_MISSING


def _df_type_handler(function, df, mv, keys, **kwargs):
    if isinstance(df, dict):
        df_encoded, mv_encoded = dict(), dict()
        for k in df.keys():
            if keys is None or (keys is not None and k in keys):
                df_encoded[k], mv_encoded[k] = function(df[k], mv[k], **kwargs)
            else:
                df_encoded[k], mv_encoded[k] = df[k].copy(), mv[k].copy()

        return df_encoded, mv_encoded

    if isinstance(df, list):
        df_encoded, mv_encoded = _df_type_handler(function,
                                                  dict(enumerate(df)),
                                                  dict(enumerate(mv)),
                                                  keys=keys,
                                                  **kwargs)
        return list(df_encoded.values()), list(mv_encoded.values())

    return function(df, mv, **kwargs)


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

    return _df_type_handler(encode, df, mv, keys, order=order)


def one_hot_encode(df, mv, keys=None):

    def encode(df, mv):
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

        return df_encoded, mv_encoded

    return _df_type_handler(encode, df, mv, keys)
