"""Implement the new way of coding Tasks."""
import pandas as pd
from dataclasses import dataclass
import logging

from missing_values import get_missing_values
from features_type import _load_feature_types
from database import dbs
from .transform import Transform


@dataclass
class TaskMeta(object):
    """Store the metadata of a task."""

    name: str
    db: str
    df_name: str
    classif: bool

    predict: Transform
    transform: Transform = None
    idx_selection: Transform = None
    select: Transform = None
    encode: str = None

    def get_infos(self):
        """Return a dict containing infos on the object."""
        return {
            'name': self.name,
            'db': self.db,
            'df_name': self.df_name,
            'classif': self.classif,
            'idx_selection': self.idx_selection.get_infos(),
            'predict': self.predict.get_infos(),
            'encode': self.encode,
        }

    @property
    def tag(self):
        """Return db_name/task_name."""
        return f'{self.db}/{self.name}'


class Task(object):
    """Gather a TaskMeta and a dataframe."""

    def __init__(self, meta):
        """Init."""
        self.meta = meta

        # Store the features availables in each dataframe
        self._f_init = None
        self._f_y = None
        self._f_transform = None

        # Store the dataframes
        self._X_base = None
        self._X_extra = None
        self._y = None

        self._rows_to_drop = None

    @property
    def X(self):
        """Input dataset."""
        if not all((self._X_base, self._X_extra)):
            self._load()

        if self._X_base is None:
            return self._X_extra

        if self._X_extra is None:
            return self._X_base

        return pd.concat((self._X_base, self._X_extra), axis=1)

    @property
    def y(self):
        """Feature to predict."""
        if self._y is not None:
            self._load()

        return self._y[self._f_y[0]]

    def _features_to_load(self, features):
        """From a set of features to load, find where they are."""
        f_init, f_y, f_transform = set(), set(), set()

        if self._f_init:
            f_init = set(self._f_init).intersection(features)

        if self._f_y:
            f_y = set(self._f_y).intersection(features)

        if self._f_transform:
            f_transform = set(self._f_transform).intersection(features)

        diff = set(features) - f_init.union(f_y).union(f_transform)
        if diff:
            raise ValueError(f'Some asked features where not found in any df. '
                             f'Diff: {diff}')

        return f_init, f_y, f_transform

    def _load_and_merge(self, f_init, f_y, f_transform):
        """Load asked features from each dataframe and merge them."""
        df_init = pd.DataFrame()
        df_y = pd.DataFrame()
        df_transform = pd.DataFrame()

        if f_init:
            db = dbs[self.meta.db]
            df_name = self.meta.df_name
            df_path = db.frame_paths[df_name]
            sep = db._sep
            encoding = db._encoding

            df_init = pd.read_csv(df_path, sep=sep, encoding=encoding,
                                  usecols=f_init, skiprows=self._rows_to_drop)

        if f_y:
            if self._y is None:
                raise ValueError('Asked to load feature from y but y is None.')

            df_y = self._y[f_y]

        if f_transform:
            if self._X_extra is None:
                raise ValueError('Asked to load feature from transformed df '
                                 'which id None.')

            df_transform = self._X_extra[f_transform]

        return pd.concat((df_init, df_y, df_transform), axis=1)

    def _load(self):
        """Load a dataframe from taskmeta."""
        # Step 0: get dataframe's path and load infos
        logging.debug('Get df path and load infos')
        db = dbs[self.meta.db]
        df_name = self.meta.df_name
        df_path = db.frame_paths[df_name]
        sep = db._sep
        encoding = db._encoding

        # Step 1: Load available features from initial df
        df = pd.read_csv(df_path, sep=sep, encoding=encoding, nrows=0)
        self._f_init = set(df.columns)

        # Step 2: Derive indexes to drop if any
        idx_transformer = self.meta.idx_selection
        if idx_transformer:
            logging.debug('Derive indexes to drop.')
            features_to_load = idx_transformer.input_features
            df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                             usecols=features_to_load)
            idx = df.index
            logging.debug(f'Loaded df of shape {df.shape}.')
            df = idx_transformer.transform(df)
            idx_to_keep = df.index
            idx_to_drop = idx.difference(idx_to_keep)
            self._rows_to_drop = idx_to_drop + 1  # Rows start to 1 wih header

        # Step 3: Derive the feature to predict y
        logging.debug('Derive the feature to predict y.')
        features_to_load = self.meta.predict.input_features
        df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                         usecols=features_to_load, skiprows=self._rows_to_drop)
        logging.debug(f'Loaded df of shape {df.shape}.')

        if len(self.meta.predict.output_features) != 1:
            raise ValueError('Expected only one item in output features '
                             'for deriving predict.')

        df = self.meta.predict.transform(df)
        y_name = self.meta.predict.output_features[0]
        self._y = df[[y_name]]
        self._f_y = [y_name]  # Store the name of the feature to predict

        # Step 4: Add NAN values of y to index to drop and drop them from y
        y_mv = get_missing_values(self._y[y_name], db.heuristic)
        idx_to_drop_y = self._y[y_name].index[y_mv != 0]

        idx_to_drop = idx_to_drop.union(idx_to_drop_y)  # merge the indexes
        self._rows_to_drop = idx_to_drop + 1  # Rows start to 1 with header
        self._y = self._y.drop(idx_to_drop_y, axis=0)

        # Step 5: Derive new set of features if any
        transform = self.meta.transform
        if transform:
            asked_features = transform.input_features
            f_init, f_y, f_transform = self._features_to_load(asked_features)
            df = self._load_and_merge(f_init, f_y, f_transform)
            logging.debug(f'Loaded df of shape {df.shape}.')
            df = transform.transform(df)
            features = set(df.columns)
            features_to_keep = set(transform.output_features)
            features_to_drop = features - features_to_keep
            df.drop(features_to_drop, axis=1, inplace=True)
            self._X_extra = df

        # Step 6: Load asked features
        select = self.meta.select
        if select:
            features_to_load = select.get_parent(select.output_features)
            df = pd.read_csv(df_path, sep=sep, usecols=features_to_load,
                             encoding=encoding, skiprows=self._rows_to_drop)

            # Step 7: Encode loaded features
            if self.meta.encode:
                mv = get_missing_values(df, db.heuristic)
                db._load_feature_types(self.meta)
                types = _load_feature_types(db, df_name, anonymized=False)
                db._load_ordinal_orders(self.meta)
                order = db.ordinal_orders[self.meta.tag]
                df, _, _, _ = db._encode_df(df, mv, types, order=order,
                                            encode=self.meta.encode)

            # Step 8: Drop unwanted features
            features_to_keep = select.output_features
            features = set(df.columns)
            features_to_drop = features - set(features_to_keep)
            df.drop(features_to_drop, axis=1, inplace=True)
            self._X_base = df
