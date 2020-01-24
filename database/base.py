"""Implement abstract class for the databases."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from missing_values import get_missing_values
from features_type import _load_feature_types
from df_utils import split_features, fill_df
from encode import ordinal_encode, one_hot_encode


class Database(ABC):

    @abstractmethod
    def __init__(self, name='', acronym=''):
        self.dataframes = dict()
        self.missing_values = dict()
        self.features_types = dict()

        self.encoded_dataframes = dict()
        self.name = name
        self.acronym = acronym
        self._load_db()
        self._load_feature_types()
        self._find_missing_values()

    def __getitem__(self, name):
        """Get data frame giving its name."""
        return self.dataframes[name]

    def df_names(self):
        """Get data frames' names."""
        return list(self.dataframes.keys())

    @abstractmethod
    def _load_db(self):
        pass

    @abstractmethod
    def heuristic(self, series):
        """Implement the heuristic for detecting missing values.

        Parameters
        ----------
        series : pandas.Series
            One column of the NHIS dataframe, stored as a pandas.Series object.

        Returns
        -------
        pandas.Series
            A series with same name and index as input series but having values
            in [0, 1, 2] encoding respectively: Not a missing value,
            Not applicable, Not available.

        """
        pass

    def _load_feature_types(self):
        for name in self.df_names():
            try:
                self.features_types[name] = _load_feature_types(self, name)
            except FileNotFoundError:
                print(f'{name}: features types not found. Ignored.')
            except ValueError:
                print(
                    f'{name}: error while loading feature type. '
                    f'Check if lengths match. Ignored.'
                )

    def _find_missing_values(self):
        for name, df in self.dataframes.items():
            self.missing_values[name] = get_missing_values(df, self.heuristic)

    @staticmethod
    def _encode_df(df, mv, types):
        # Split the data frame according to the types of the features
        splitted_df = split_features(df, types)

        # df_categorical = splitted_df[0]
        # df_ordinal = splitted_df[1]
        # df_continue = splitted_df[2]
        # df_binary = splitted_df[3]

        # Split the missing values in the same way
        splitted_mv = split_features(mv, types)

        # mv_categorical = splitted_mv[0]
        # mv_ordinal = splitted_mv[1]
        # mv_continue = splitted_mv[2]
        # mv_binary = splitted_mv[3]

        to_ordinal_encode_ids = [1, 3]
        to_one_hot_encode_ids = [0]
        to_delete_ids = [-1]

        for k in to_delete_ids:
            del splitted_df[k]
            del splitted_mv[k]


        # Fill missing values otherwise the fit raises and error cause of Nans
        # df_categorical[mv_categorical] = 'MISSING_VALUE'
        # df_ordinal[mv_ordinal] = 'MISSING_VALUE'
        # df_continue[mv_continue] = 'MISSING_VALUE'
        # df_binary[mv_binary] = 'MISSING_VALUE'

        # fill_df(df_categorical, mv_categorical, 'MISSING_VALUE')
        # fill_df(df_ordinal, mv_ordinal, 'MISSING_VALUE')
        # fill_df(df_continue, mv_continue, 'MISSING_VALUE')
        # fill_df(df_binary, mv_binary, 'MISSING_VALUE')

        # fill_df([
        #     df_categorical,
        #     df_ordinal,
        #     df_continue,
        #     df_binary
        # ],
        #     [
        #     mv_categorical,
        #     mv_ordinal,
        #     mv_continue,
        #     mv_binary
        # ], 'MISSING_VALUE')

        fill_df(splitted_df, splitted_mv, 'MISSING_VALUE')

        # Df to ordinal encode:
        # to_ordinal_encode_df = {k: splitted_df[k] for k in to_ordinal_encode_ids}
        # to_ordinal_encode_mv = {k: splitted_mv[k] for k in to_ordinal_encode_ids}

        # Df to one hot encode:
        # to_one_hot_encode_df = {k: splitted_df[k] for k in to_one_hot_encode_ids}
        # to_one_hot_encode_mv = {k: splitted_mv[k] for k in to_one_hot_encode_ids}

        # Ordinal encode the ordinal and binary ones
        # ord_encoded = ordinal_encode([
        #     # df_categorical,
        #     df_ordinal,
        #     df_binary
        # ], [mv_ordinal, mv_binary])

        # ord_encoded_df = ordinal_encode(to_ordinal_encode_df, to_ordinal_encode_mv)

        # oe_df = [ordinal_encode(splitted_df[k], splitted_mv[k]) for k in to_ordinal_encode_ids]

        splitted_df, splitted_mv = ordinal_encode(splitted_df, splitted_mv, keys=to_ordinal_encode_ids)

        # One hot encode the categorical one
        # one_hot_encoded = one_hot_encode([df_categorical], [mv_categorical])
        # one_hot_encoded_df = one_hot_encode(to_one_hot_encode_df, to_one_hot_encode_mv)
        # ohe_df = [one_hot_encode(splitted_df[k], splitted_mv[k]) for k in to_one_hot_encode_ids]
        splitted_df, splitted_mv = one_hot_encode(splitted_df, splitted_mv, keys=to_one_hot_encode_ids)



        # encoded_df_list = [df_continue]+oe_df+ohe_df

        encoded_df = pd.concat(splitted_df.values(), axis=1)
        encoded_mv = pd.concat(splitted_mv.values(), axis=1)

        # Set missing values to blank
        fill_df(encoded_df, encoded_mv, np.nan)
        # encoded_df[mv] = np.nan  # Clean the missing values with blank

        return encoded_df

    def _encode(self):
        for name, df in self.dataframes.items():
            if name not in self.features_types:
                print(f'{name}: feature types missing. Encoding ignored.')
            elif name not in self.missing_values:
                print(f'{name}: missing values df missing. Encoding ignored.')
            else:
                types = self.features_types[name]
                mv = self.missing_values[name]
                self.encoded_dataframes[name] = self._encode_df(df, mv != 0,
                                                                types)
