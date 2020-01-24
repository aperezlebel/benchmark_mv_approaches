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

        # Fill missing values otherwise the fit raises and error cause of Nans
        # edited_df = df.copy()
        # print(edited_df.dtypes)
        # for col in mv.columns:
        #     edited_df[col][mv[col]] = 'MISSING_VALUE'

        # Fit transform the encode
        enc.fit(df)
        data_encoded = enc.transform(df)
        # del edited_df
        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index, columns=df.columns)

        # Set back the missing values
        # df_encoded[mv] = df[mv]

        return df_encoded, mv

    return _df_type_handler(encode, df, mv, keys)

    # if isinstance(df, dict):
    #     df_encoded, mv_encoded = dict(), dict()
    #     for k in df.keys():
    #         if keys is None or (keys is not None and k in keys):
    #             df_encoded[k], mv_encoded[k] = encode(df[k], mv[k])
    #         else:
    #             df_encoded[k], mv_encoded[k] = df[k].copy(), mv[k].copy()

    #     return df_encoded, mv_encoded

    # if isinstance(df, list):
    #     df_encoded, mv_encoded = ordinal_encode(dict(enumerate(df)),
    #                                             dict(enumerate(mv)),
    #                                             keys=keys)
    #     return list(df_encoded.values()), list(mv_encoded.values())

    # return encode(df, mv)


def one_hot_encode(df, mv, keys=None):
    # return df

    def encode(df, mv):
        enc = OneHotEncoder(sparse=False)

        # Fill missing values otherwise the fit raises and error cause of Nans
        # edited_df = df.copy()
        # print(edited_df.dtypes)
        # print(edited_df.index)
        # print(mv.index)
        # # edited_df.loc[mv] = 'MISSING_VALUE'
        # for col in mv.columns:
        #     edited_df[col][mv[col]] = 'MISSING_VALUE'

        # Fit transform the encode
        enc.fit(df)
        data_encoded = enc.transform(df)
        # del edited_df
        feature_names = list(enc.get_feature_names(list(df.columns)))
        # print(feature_names)
        # print(len(feature_names))
        # print(data_encoded.shape)
        # print(len(df.index))
        df_encoded = pd.DataFrame(data_encoded,
                                  index=df.index,
                                  columns=feature_names
                                  )

        mv_encoded = pd.DataFrame(np.zeros(data_encoded.shape),
                                  index=df.index,
                                  columns=feature_names)
        # print(df_encoded.shape)
        # print(df_encoded)
        # print(data_encoded)
        # exit()
        return df_encoded, mv_encoded

    return _df_type_handler(encode, df, mv, keys)

    # if isinstance(df, dict):
    #     df_encoded, mv_encoded = dict(), dict()
    #     for k in df.keys():
    #         if keys is None or (keys is not None and k in keys):
    #             df_encoded[k], mv_encoded[k] = encode(df[k], mv[k])
    #         else:
    #             df_encoded[k], mv_encoded[k] = df[k].copy(), mv[k].copy()

    #     return df_encoded, mv_encoded

    # if isinstance(df, list):
    #     df_encoded, mv_encoded = ordinal_encode(dict(enumerate(df)),
    #                                             dict(enumerate(mv)),
    #                                             keys=keys)
    #     return list(df_encoded.values()), list(mv_encoded.values())

    # return encode(df, mv)
