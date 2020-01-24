"""Functions to encode a data frame (OrdinalEncode, OneHotEncode)..."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def _df_type_handler(function, df, mv, keys):
    if isinstance(df, dict):
        df_encoded, mv_encoded = dict(), dict()
        for k in df.keys():
            if keys is None or (keys is not None and k in keys):
                df_encoded[k], mv_encoded[k] = function(df[k], mv[k])
            else:
                df_encoded[k], mv_encoded[k] = df[k].copy(), mv[k].copy()

        return df_encoded, mv_encoded

    if isinstance(df, list):
        df_encoded, mv_encoded = _df_type_handler(function,
                                                  dict(enumerate(df)),
                                                  dict(enumerate(mv)),
                                                  keys=keys)
        return list(df_encoded.values()), list(mv_encoded.values())

    return function(df, mv)


def ordinal_encode(df, mv, keys=None):

    def encode(df, mv):
        enc = OrdinalEncoder()
        # Fit transform the encode
        data_encoded = enc.fit_transform(df)

        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index, columns=df.columns)

        return df_encoded, mv

    return _df_type_handler(encode, df, mv, keys)


def one_hot_encode(df, mv, keys=None):

    def encode(df, mv):
        enc = OneHotEncoder(sparse=False)

        # Fit transform the encode
        data_encoded = enc.fit_transform(df)

        feature_names = list(enc.get_feature_names(list(df.columns)))

        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index,
                                  columns=feature_names
                                  )

        mv_encoded = pd.DataFrame(np.zeros(data_encoded.shape),
                                  index=df.index,
                                  columns=feature_names)

        return df_encoded, mv_encoded

    return _df_type_handler(encode, df, mv, keys)
