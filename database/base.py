"""Implement abstract class for the databases."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import yaml
import os

from missing_values import get_missing_values
from features_type import _load_feature_types
from df_utils import split_features, fill_df, set_dtypes_features
from encode import ordinal_encode, one_hot_encode, date_encode
from .constants import CATEGORICAL, ORDINAL, BINARY, CONTINUE_R, CONTINUE_I, \
    NOT_A_FEATURE, NOT_MISSING, DATE_TIMESTAMP, DATE_EXPLODED, METADATA_PATH


class Database(ABC):

    @abstractmethod
    def __init__(self, name='', acronym='', paths=dict(), sep=',', load=None):
        self.dataframes = dict()
        self.missing_values = dict()
        self.feature_types = dict()
        self.ordinal_orders = dict()

        self.encoded_dataframes = dict()
        self.encoded_missing_values = dict()
        self.encoded_feature_types = dict()

        self.name = name
        self.acronym = acronym
        self.frame_paths = paths
        self._sep = sep

        if load is not None:
            self.load(load)

    @property
    def available_paths(self):
        return {n: p for n, p in self.frame_paths.items() if os.path.exists(p)}

    def load(self, load):
        if isinstance(load, str):
            load = [load]

        if not isinstance(load, list):
            raise ValueError('Table names to load must be list or str.')

        self._load_db(load)
        self._load_feature_types()
        self._drop()
        self._load_ordinal_orders()
        self._find_missing_values()
        self._encode()

    def __getitem__(self, name):
        """Get data frame giving its name."""
        return self.dataframes[name]

    def df_names(self):
        """Get data frames' names."""
        return list(self.dataframes.keys())

    @abstractmethod
    def _to_drop(self, df_name):
        pass

    def _drop(self):
        for name, df in self.dataframes.items():
            to_drop = self._to_drop(name)
            print(f'{name}: Dropping {len(to_drop)} cols out of {df.shape[1]}')
            df.drop(to_drop, axis=1, inplace=True)

    def _load_db(self, load):
        available_paths = self.available_paths
        for n in load:
            if n not in available_paths:
                raise ValueError(
                    f'{n} not an available name.\n'
                    f'Available name and paths are {available_paths}.'
                )
            p = self.frame_paths[n]
            self.dataframes[n] = pd.read_csv(p, sep=self._sep)

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
                self.feature_types[name] = _load_feature_types(self, name,
                                                               anonymized=False)
            # except FileNotFoundError:
            #     print(f'{name}: features types not found. Ignored.')
            except ValueError:
                print(
                    f'{name}: error while loading feature type. '
                    f'Check if lengths match. Ignored.'
                )

    def _load_ordinal_orders(self):
        for df_name in self.dataframes.keys():
            filepath = f'{METADATA_PATH}/ordinal_orders/{self.acronym}/{df_name}.yml'

            if not os.path.exists(filepath):
                print(f'Order file not found. No order loaded for {df_name}.')
                continue

            with open(filepath, 'r') as file:
                try:
                    self.ordinal_orders[df_name] = yaml.safe_load(file)
                except yaml.YAMLError as exc:
                    print(f'{exc}. No order loaded for {df_name}.')

    def _find_missing_values(self):
        for name, df in self.dataframes.items():
            self.missing_values[name] = get_missing_values(df, self.heuristic)

    @staticmethod
    def _encode_df(df, mv, types, order=None):
        # Split the data frame according to the types of the features
        splitted_df = split_features(df, types)

        # Split the missing values in the same way
        splitted_mv = split_features(mv, types)

        # Split the feature types in the same way
        splitted_types = split_features(types, types)

        # Choose which tables go in which pipeline
        to_ordinal_encode_ids = [ORDINAL, BINARY]
        to_one_hot_encode_ids = [CATEGORICAL]
        to_delete_ids = [NOT_A_FEATURE]

        # Delete unwanted tables
        for k in to_delete_ids:
            del splitted_df[k]
            del splitted_mv[k]

        # Fill missing values otherwise the fit raises an error cause of Nans
        splitted_mv_bool = {k: mv != NOT_MISSING for k, mv in splitted_mv.items()}
        fill_df(splitted_df, splitted_mv_bool, 'z MISSING_VALUE')

        # Ordinal encode
        splitted_df, splitted_mv = ordinal_encode(splitted_df, splitted_mv, keys=to_ordinal_encode_ids, order=order)

        # One hot encode
        splitted_df, splitted_mv, splitted_types = one_hot_encode(splitted_df, splitted_mv, splitted_types, keys=to_one_hot_encode_ids)

        # Set missing values to blank
        splitted_mv_bool = {k: mv != NOT_MISSING for k, mv in splitted_mv.items()}
        fill_df(splitted_df, splitted_mv_bool, np.nan)

        # Date encode
        splitted_df, splitted_mv, splitted_types = date_encode(splitted_df, splitted_mv, splitted_types, keys=DATE_EXPLODED, method='explode', dayfirst=True)
        splitted_df, splitted_mv, splitted_types = date_encode(splitted_df, splitted_mv, splitted_types, keys=DATE_TIMESTAMP, method='timestamp', dayfirst=True)

        # Merge encoded df
        encoded_df = pd.concat(splitted_df.values(), axis=1)
        encoded_mv = pd.concat(splitted_mv.values(), axis=1)
        encoded_types = pd.concat(splitted_types.values())

        # Set types on encoded df
        encoded_df = set_dtypes_features(encoded_df, encoded_types, {
            CONTINUE_R: float,
            CONTINUE_I: float})

        return encoded_df, encoded_mv, encoded_types

    @abstractmethod
    def _encode(self):
        for name, df in self.dataframes.items():
            if name not in self.feature_types:
                print(f'{name}: feature types missing. Encoding ignored.')
            elif name not in self.missing_values:
                print(f'{name}: missing values df missing. Encoding ignored.')
            else:
                types = self.feature_types[name]
                mv = self.missing_values[name]
                order = None
                if name in self.ordinal_orders:
                    order = self.ordinal_orders[name]
                encoded = self._encode_df(df, mv, types, order=order)
                self.encoded_dataframes[name] = encoded[0]
                self.encoded_missing_values[name] = encoded[1]
                self.encoded_feature_types[name] = encoded[2]
