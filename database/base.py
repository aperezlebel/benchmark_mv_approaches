"""Implement abstract class for the databases."""

from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
import yaml
import os

from missing_values import get_missing_values
from features_type import _load_feature_types
from df_utils import split_features, fill_df, set_dtypes_features, \
    dtype_from_types
from encode import ordinal_encode, one_hot_encode, date_encode
from .constants import CATEGORICAL, ORDINAL, BINARY, CONTINUE_R, CONTINUE_I, \
    NOT_A_FEATURE, NOT_MISSING, DATE_TIMESTAMP, DATE_EXPLODED, METADATA_PATH


logger = logging.getLogger(__name__)


type_to_dtype = {
    CATEGORICAL: 'category',
    ORDINAL: 'category',
    BINARY: 'category',
    CONTINUE_R: np.float32,
    CONTINUE_I: 'Int32',
    # DATE_TIMESTAMP: 'datetime64',
    # DATE_EXPLODED: 'datetime64'
}


class Database(ABC):

    @abstractmethod
    def __init__(self, name='', acronym='', paths=dict(), sep=',', load=None,
                 encoding='utf-8', encode=None):
        self.dataframes = dict()
        self.missing_values = dict()
        self.feature_types = dict()
        self.ordinal_orders = dict()

        self.encoded_dataframes = dict()
        self.encoded_missing_values = dict()
        self.encoded_feature_types = dict()
        self.encoded_parent = dict()

        self.name = name
        self.acronym = acronym
        self.frame_paths = paths
        self._sep = sep
        self._encoding = encoding
        self.encode = encode
        self._dtype = None

        if load is not None:
            self.load(load)

    @property
    def available_paths(self):
        return {n: p for n, p in self.frame_paths.items() if os.path.exists(p)}

    def is_loaded(self, df_name):
        return df_name in self.dataframes

    def load(self, meta):
        self._load_feature_types(meta)
        self._load_db(meta)
        self._load_ordinal_orders(meta)
        self._find_missing_values(meta)
        self._encode(meta)

    @staticmethod
    def get_drop_and_keep_meta(features, meta):
        """Give which feature to keep and which to drop from a TaskMeta.

        Prameters
        ---------
        features : list or pd.Series or pd.DataFrame
            Gives the universal set of features, i.e the features to consider.
        meta : TaskMeta object
            Object having keep, drop, keep_contains, drop_contains parameters.

        Returns
        -------
        to_keep : set
            Set containing the features to keep
        to_drop : set
            Set containing the feature to drop

        """
        return Database.get_drop_and_keep(
            features=features,
            keep=meta.keep,
            keep_contains=meta.keep_contains,
            drop=meta.drop,
            drop_contains=meta.drop_contains,
            predict=meta.predict
        )

    @staticmethod
    def get_drop_and_keep(features, keep=None, keep_contains=None, drop=None,
                          drop_contains=None, predict=None):
        """Give which feature to keep and which to drop from a TaskMeta.

        Prameters
        ---------
        features : list or pd.Series or pd.DataFrame
            Gives the universal set of features, i.e the features to consider.
        keep : list of str
            List of features to keep.
        keep_contains : list of str
            List of patterns of features to keep.
        drop : list of str
            List of features to drop.
        drop_contains : list of str
            List of patterns of features to drop.

        Returns
        -------
        to_keep : set
            Set containing the features to keep
        to_drop : set
            Set containing the feature to drop

        """
        # Check features
        if isinstance(features, pd.DataFrame):
            features = features.columns

        elif isinstance(features, pd.Series):
            features = features.index

        elif isinstance(features, list):
            features = pd.Index(features)

        else:
            raise ValueError('features must be df, series or list.')

        # Check keep and drop
        if any((keep, keep_contains)) and any((drop, drop_contains)):
            raise ValueError('Cannot use keep and drop at the same time.')

        # Case where neither keep nor drop is given
        if not any((keep, keep_contains, drop, drop_contains)):
            return set(features), set()

        if any((keep, keep_contains)):
            method = 'keep'
            select = [] if keep is None else keep
            if predict is not None:
                select += [predict]
            select_contains = [] if keep_contains is None else keep_contains

        elif any((drop, drop_contains)):
            method = 'drop'
            select = [] if drop is None else drop
            select_contains = [] if drop_contains is None else drop_contains

        # Convert select_contains based on patterns to explicit selection
        select_array = np.logical_or.reduce(
            np.array([features.str.contains(p) for p in select_contains])
        )
        select_series = pd.Series(select_array, index=features)
        select_extra = list(select_series[select_series].index)

        # Merge select and select_extra
        select = set(select + select_extra)

        # Keep only the features present in features (non existing features
        # may have been given in keep/drop...)
        select = select.intersection(set(features))

        # Transform drop selection into keep selection
        select_complement = set(features) - select
        # select_complement = Database.get_complement(select, list(features))

        if method == 'drop':
            if predict is not None and predict in select:
                raise ValueError('Feature to predict is in features to drop.')

            return select_complement, select

        assert method == 'keep'
        return select, select_complement

    def __getitem__(self, name):
        """Get data frame giving its name."""
        return self.dataframes[name]

    def df_names(self):
        """Get data frames' names."""
        return list(self.dataframes.keys())

    def _load_db(self, meta):
        if isinstance(meta, str):
            df_name, tag = meta, meta
        else:
            df_name, tag = meta.df_name, meta.tag

        logger.info(f'Loading {df_name} data frame.')
        available_paths = self.available_paths

        if df_name not in available_paths:
            raise ValueError(
                f'{df_name} not an available name.\n'
                f'Available name and paths are {available_paths}.'
            )
        p = self.frame_paths[df_name]

        # dtype = None
        # if self._dtype is not None:
        #     dtype = self._dtype.get(df_name, None)

        if not isinstance(meta, str):
            # Load only the features of the database (avoid load time)
            features = pd.read_csv(p, sep=self._sep, encoding=self._encoding,
                                   nrows=0)

            # Compute index where feature to predict is Nan
            if meta.predict is not None and meta.predict in features:
                df_predict = pd.read_csv(p, sep=self._sep, encoding=self._encoding,
                                         usecols=[meta.predict], squeeze=True)
                logger.info(
                    f'Raw DB of shape [{df_predict.size} x {features.shape[1]}]')
                df_predict_mv = get_missing_values(df_predict, self.heuristic)
                index_to_drop = df_predict.index[df_predict_mv != 0]+1
                logger.info(
                    f'Rows to drop because NA in predict {len(index_to_drop)}')
            else:
                index_to_drop = None

            # Compute the features to keep
            to_keep, to_drop = self.get_drop_and_keep_meta(features, meta)
            logger.info(
                f'Features to drop as specified in meta {len(to_drop)}')

        else:
            to_keep = None
            index_to_drop = None

        # Load only the features needed: save a lot of time and space
        df = pd.read_csv(p, sep=self._sep, encoding=self._encoding,
                         usecols=to_keep, skiprows=index_to_drop)

        logger.info(f'df {tag} loaded with shape {df.shape}')
        # dtype=dtype)

        # Replace potential infinite values by Nans
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.dataframes[tag] = df

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

    def _load_feature_types(self, meta):
        logger.info(f'Loading feature types for {self.acronym}.')

        if isinstance(meta, str):
            df_name, tag = meta, meta
        else:
            df_name, tag = meta.df_name, meta.tag

        try:
            types = _load_feature_types(self, df_name, anonymized=False)
        except ValueError:
            print(
                f'{df_name}: error while loading feature type. '
                f'Check if lengths match. Ignored.'
            )
            return

        self.feature_types[tag] = types

    def _set_dtypes(self, meta):
        logger.info(f'Setting dtypes for {self.acronym}.')
        if self._dtype is None:
            self._dtype = dict()

        logger.info(f'Setting dtypes of {meta.tag}.')
        types = self.feature_types[meta.tag]
        self._dtype[meta.tag] = dtype_from_types(types, type_to_dtype)

    def _load_ordinal_orders(self, meta):
        logger.info(f'Loading ordinal orders for {self.acronym}.')

        if isinstance(meta, str):
            df_name, tag = meta, meta
        else:
            df_name, tag = meta.df_name, meta.tag

        logger.info(f'Loading ordinal orders of {df_name}.')
        filepath = f'{METADATA_PATH}/ordinal_orders/{self.acronym}/{df_name}.yml'

        if not os.path.exists(filepath):
            print(f'Order file not found. No order loaded for {df_name}.')
            return

        with open(filepath, 'r') as file:
            try:
                order = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f'{exc}. No order loaded for {df_name}.')
                return

        self.ordinal_orders[tag] = order

    def _find_missing_values(self, meta):
        tag = meta if isinstance(meta, str) else meta.tag
        logger.info(f'Finding missing values of {tag}.')
        df = self.dataframes[tag]
        self.missing_values[tag] = get_missing_values(df, self.heuristic)

    @staticmethod
    def _encode_df(df, mv, types, order=None, encode=None):
        logger.info(f'Encode mode: {encode}')

        common_features = [f for f in df.columns if f in types.index]
        types = types[common_features]

        # Assign a defaut type for extra features in type
        extra_features = [f for f in df.columns if f not in types.index]
        extra_types = pd.Series(CONTINUE_R, index=extra_features)

        types = pd.concat([types, extra_types])
        parent = pd.Series(df.columns, index=df.columns)

        # Split the data frame according to the types of the features
        splitted_df = split_features(df, types)

        # Split the missing values in the same way
        splitted_mv = split_features(mv, types)

        # Split the feature types in the same way
        splitted_types = split_features(types, types)

        # Split the parent features
        splitted_parent = split_features(parent, types)

        # Choose which tables go in which pipeline
        to_ordinal_encode_ids = []
        to_one_hot_encode_ids = []
        to_delete_ids = [NOT_A_FEATURE]

        if not isinstance(encode, list):
            encode = [encode]

        if encode is not None and ('ordinal' in encode or 'all' in encode):
            to_ordinal_encode_ids = [ORDINAL, BINARY]

        if encode is not None and ('one_hot' in encode or 'all' in encode):
            to_one_hot_encode_ids = [CATEGORICAL]

        if encode is not None and ('date' in encode or 'all' in encode):
            to_date_encode_exp = [DATE_EXPLODED]
            to_date_encode_tim = [DATE_TIMESTAMP]

        logger.info(f'Keys, ordinal encode: {to_ordinal_encode_ids}')
        logger.info(f'Keys, one hot encode: {to_one_hot_encode_ids}')
        logger.info(f'Keys, date encode exp: {to_date_encode_exp}')
        logger.info(f'Keys, date encode tim: {to_date_encode_tim}')
        logger.info(f'Keys, to delete: {to_delete_ids}')

        # Delete unwanted tables
        for k in to_delete_ids:
            splitted_df.pop(k, None)
            splitted_mv.pop(k, None)

        # Fill missing values otherwise the fit raises an error cause of Nans
        # splitted_mv_bool = {k: mv != NOT_MISSING for k, mv in splitted_mv.items()}
        # splitted_df = fill_df(splitted_df, splitted_mv_bool, 'z MISSING_VALUE')
        # Set missing values to blank
        logger.info('Encoding: Fill missing values.')
        splitted_mv_bool = {k: mv != NOT_MISSING for k, mv in splitted_mv.items()}
        splitted_df = fill_df(splitted_df, splitted_mv_bool, np.nan)

        # Ordinal encode
        logger.info('Encoding: Ordinal encode.')
        splitted_df, splitted_mv = ordinal_encode(splitted_df, splitted_mv, keys=to_ordinal_encode_ids, order=order)

        # One hot encode
        logger.info('Encoding: One hot encode.')
        splitted_df, splitted_mv, splitted_types, splitted_parent = one_hot_encode(splitted_df, splitted_mv, splitted_types, splitted_parent, keys=to_one_hot_encode_ids)

        # Date encode
        logger.info('Encoding: Date encode.')
        splitted_df, splitted_mv, splitted_types, splitted_parent = date_encode(splitted_df, splitted_mv, splitted_types, splitted_parent, keys=to_date_encode_exp, method='explode', dayfirst=True)
        splitted_df, splitted_mv, splitted_types, splitted_parent = date_encode(splitted_df, splitted_mv, splitted_types, splitted_parent, keys=to_date_encode_tim, method='timestamp', dayfirst=True)

        logger.info('Encoding: Fill missing values.')
        splitted_mv_bool = {k: mv != NOT_MISSING for k, mv in splitted_mv.items()}
        splitted_df = fill_df(splitted_df, splitted_mv_bool, np.nan)

        # Merge encoded df
        logger.info('Encoding: Merge df.')
        encoded_df = pd.concat(splitted_df.values(), axis=1)
        encoded_mv = pd.concat(splitted_mv.values(), axis=1)
        encoded_types = pd.concat(splitted_types.values())
        encoded_parent = pd.concat(splitted_parent.values())

        # Set types on encoded df
        encoded_df = set_dtypes_features(encoded_df, encoded_types, {
            CONTINUE_R: float,
            CONTINUE_I: float,
        })

        return encoded_df, encoded_mv, encoded_types, encoded_parent

    def _encode(self, meta):
        tag = meta if isinstance(meta, str) else meta.tag
        df = self.dataframes[tag]
        logger.info(f'Encoding {tag}.')

        if tag not in self.feature_types:
            print(f'{tag}: feature types missing. Encoding ignored.')

        elif tag not in self.missing_values:
            print(f'{tag}: missing values df missing. Encoding ignored.')

        else:
            types = self.feature_types[tag]
            mv = self.missing_values[tag]
            order = self.ordinal_orders.get(tag, None)

            encoded = self._encode_df(df, mv, types, order=order, encode=self.encode)

            self.encoded_dataframes[tag] = encoded[0]
            self.encoded_missing_values[tag] = encoded[1]
            self.encoded_feature_types[tag] = encoded[2]
            self.encoded_parent[tag] = encoded[3]

            logger.info(f'df {tag} encoded with shape {encoded[0].shape}')

    def _rename(self, obj, rename):
        rename_from = rename.keys()

        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
            cols_to_rename = [c for c in df.columns if c in set(rename_from)]

            for c in cols_to_rename:
                df[rename[c]] = df[c]

            df.drop(cols_to_rename, axis=1, inplace=True)
            return df

        if isinstance(obj, pd.Series):
            series = obj.copy()
            cols_to_rename = [c for c in series.index if c in set(rename_from)]

            for c in cols_to_rename:
                series[rename[c]] = series[c]

            series.drop(cols_to_rename, inplace=True)

            return series

        if isinstance(obj, dict):
            d = obj.copy()

            cols_to_rename = [c for c in obj.keys() if c in set(rename_from)]

            for c in cols_to_rename:
                d[rename[c]] = obj[c]
                d.pop(c)

            return d

    def rename_encode(self, tag, rename, encode='all'):
        df = self.dataframes[tag]

        mv = self.missing_values[tag]
        types = self.feature_types[tag]
        order = None
        if tag in self.ordinal_orders:
            order = self.ordinal_orders[tag]

        df = self._rename(df, rename)
        mv = self._rename(mv, rename)
        types = self._rename(types, rename)
        if order is not None:
            order = self._rename(order, rename)

        df, _, _, parent = self._encode_df(df, mv, types, order=order, encode=encode)

        return df, parent
