"""Implement the new way of coding Tasks."""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Set
import logging

from missing_values import get_missing_values
from df_utils import fill_df
from database import dbs, _load_feature_types
from .transform import Transform
from encode import ordinal_encode


@dataclass
class TaskMeta(object):
    """Store the metadata of a task."""

    name: str
    db: str
    df_name: str
    classif: bool

    predict: Transform
    transform: Transform = None
    idx_column: str = None
    idx_selection: Transform = None
    select: Transform = None
    encode_select: str = None
    encode_transform: str = None
    drop: Set[str] = field(default_factory=set)
    encode_y: bool = True

    def __post_init__(self):
        if isinstance(self.drop, list):
            self.drop = set(self.drop)

        if self.idx_column is None:
            self.idx_column = []
        elif not isinstance(self.idx_column, list):
            self.idx_column = [self.idx_column]

        if self.drop.intersection(self.predict.output_features):
            raise ValueError('Some predict output features are in drop.')

    def get_infos(self):
        """Return a dict containing infos on the object."""
        data = {
            'name': self.name,
            'db': self.db,
            'df_name': self.df_name,
            'classif': self.classif,
            'encode_select': self.encode_select,
            'encode_transform': self.encode_transform,
        }

        if self.idx_selection is not None:
            data['idx_selection'] = self.idx_selection.get_infos()

        if self.predict is not None:
            data['predict']: self.predict.get_infos()

        if self.transform is not None:
            data['transform']: self.transform.get_infos()

        if self.select is not None:
            data['select']: self.select.get_infos()

        return data

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
        self._X_select_base = None
        self._X_select_unenc = None
        self._X_select = None
        self._X_extra_base = None
        self._X_extra_unenc = None
        self._X_extra = None
        self._y = None

        self._file_index = None

        self._rows_to_drop = None

    @property
    def X(self):
        """Input dataset."""
        if self._X_select is None and self._X_extra is None:
            self._load_X_y()

        if self._X_select is None:
            return self._X_extra

        if self._X_extra is None:
            return self._X_select

        return pd.concat((self._X_select, self._X_extra), axis=1)

    @property
    def y(self):
        """Feature to predict."""
        if self._y is None:
            self._load_y()

        return self._y[self._f_y[0]]

    def is_classif(self):
        """Tell if the task is a classification or a regression."""
        return self.meta.classif

    def get_infos(self):
        """Get infos on the task."""
        infos = self.meta.get_infos()
        infos['_X_select.shape'] = repr(getattr(self._X_select, 'shape', None))
        infos['_X_extra.shape'] = repr(getattr(self._X_extra, 'shape', None))
        infos['X.shape'] = repr(getattr(self.X, 'shape', None))
        infos['_y.shape'] = repr(getattr(self._y, 'shape', None))
        return infos

    def _idx_to_rows(self, idx):
        if self.meta.idx_column is None:
            return idx + 1  # Rows start to 1 wih header

        if self._file_index is None:
            self._load_index()

        rows = []
        index = self._file_index
        for i in idx:
            rows.append(index.get_loc(i)+1)

        return rows

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
            df_init = self._X_extra_base
            features = set(df_init.columns)
            features_to_drop = features - set(f_init)
            df_init = df_init.drop(features_to_drop, axis=1)

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

    def _load_index(self):
        db = dbs[self.meta.db]
        df_name = self.meta.df_name
        df_path = db.frame_paths[df_name]
        sep = db._sep
        encoding = db._encoding
        index_col = self.meta.idx_column

        if index_col:
            df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                             usecols=index_col, index_col=index_col)
            self._file_index = df.index

    def _load_y(self):
        """Load a dataframe from taskmeta (only y)."""
        # Step 0: get dataframe's path and load infos
        logging.debug('Get df path and load infos')
        db = dbs[self.meta.db]
        df_name = self.meta.df_name
        df_path = db.frame_paths[df_name]
        sep = db._sep
        encoding = db._encoding
        index_col = self.meta.idx_column

        # Step 1: Load available features from initial df
        df = pd.read_csv(df_path, sep=sep, encoding=encoding, nrows=0,
                         index_col=index_col)
        self._f_init = {s for s in set(df.columns) if s not in self.meta.drop}
        self._f_init.update(index_col)

        # Step 1.2: load index
        self._load_index()

        # Step 2: Derive indexes to drop if any
        idx_transformer = self.meta.idx_selection
        idx_to_drop = pd.Index([])  # At start, no indexes to drop
        if idx_transformer:
            logging.debug('Derive indexes to drop.')
            features_to_load = set(idx_transformer.input_features+index_col)
            features_to_load = features_to_load.intersection(self._f_init)
            df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                             usecols=features_to_load, index_col=index_col)
            idx = df.index
            logging.debug(f'Loaded df of shape {df.shape}.')
            df = idx_transformer.transform(df)
            idx_to_keep = df.index
            idx_to_drop = idx.difference(idx_to_keep)
            self._rows_to_drop = self._idx_to_rows(idx_to_drop)

        # Step 3: Derive the feature to predict y
        logging.debug('Derive the feature to predict y.')
        features_to_load = set(self.meta.predict.input_features+index_col)
        features_to_load = features_to_load.intersection(self._f_init)
        df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                         usecols=features_to_load, skiprows=self._rows_to_drop,
                         index_col=index_col)
        logging.debug(f'Loaded df of shape {df.shape}.')

        if len(self.meta.predict.output_features) != 1:
            raise ValueError('Expected only one item in output features '
                             'for deriving predict.')

        df = self.meta.predict.transform(df)
        y_name = self.meta.predict.output_features[0]
        self._y = df[[y_name]]
        self._f_y = [y_name]  # Store the name of the feature to predict

        # Drop the feature to predict from _f_init
        self._f_init.discard(y_name)

        # Step 4: Add NAN values of y to index to drop and drop them from y
        y_mv = get_missing_values(self._y[y_name], db.heuristic)
        idx_to_drop_y = self._y[y_name].index[y_mv != 0]

        idx_to_drop = idx_to_drop.union(idx_to_drop_y)  # merge the indexes
        self._rows_to_drop = self._idx_to_rows(idx_to_drop)
        self._y = self._y.drop(idx_to_drop_y, axis=0)

        # Step 5: Encode y if needed
        if self.is_classif() and self.meta.encode_y:
            y_mv = get_missing_values(self._y, db.heuristic)
            self._y, _ = ordinal_encode(self._y, y_mv)
        elif self.meta.encode_y:  # cast to float for regression
            self._y = self._y.astype(float)

        self._y.sort_index(inplace=True)  # to have consistent order with X

    def _load_X_base(self):
        if self._y is None:
            self._load_y()

        # Step 0: get dataframe's path and load infos
        logging.debug('Get df path and load infos')
        db = dbs[self.meta.db]
        df_name = self.meta.df_name
        df_path = db.frame_paths[df_name]
        sep = db._sep
        encoding = db._encoding
        index_col = self.meta.idx_column

        # Step 5.1: Load asked features
        features_to_load = set(index_col)
        select = self.meta.select
        if select:
            if select.output_features and select.input_features:
                raise ValueError('Cannot specify both input and output '
                                 'features for select transform.')

            if select.output_features:
                select_f = select.get_parent(select.output_features)
            else:
                select_f = select.input_features
            select_f = set(select_f).intersection(self._f_init)
            features_to_load.update(select_f)
            features_to_load = features_to_load.intersection(self._f_init)

        transform = self.meta.transform
        if transform:
            transform_f = set(transform.input_features).intersection(self._f_init)
            features_to_load.update(transform_f)
            features_to_load = features_to_load.intersection(self._f_init)

        # If nothing specified, we load everything
        if not select and not transform:
            features_to_load = None

        df = pd.read_csv(df_path, sep=sep, usecols=features_to_load,
                         encoding=encoding, skiprows=self._rows_to_drop,
                         index_col=index_col, low_memory=False)
        # We add low_memory=False because if True, types are inferred by chunk
        # and some mixed types may happen (eg 1 and 1.0) which lead to an
        # error when ordinal encoding (2 categories instead of one).
        mv = get_missing_values(df, db.heuristic)
        df = fill_df(df, mv != 0, np.nan)

        df.sort_index(inplace=True)  # to have consistent order with y

        # Step 5.2: save the results
        if select:
            self._X_select_base = df[select_f]
            self._X_select_unenc = df[select_f]

        if transform:
            self._X_extra_base = df[transform_f]
            self._X_extra_unenc = df[transform_f]

        if not select and not transform:
            self._X_select_base = df
            self._X_select_unenc = df

        # Step 5.3: Encode both dataframes
        if self.meta.encode_transform and self._X_extra_base is not None:
            df = self._X_extra_base
            mv = get_missing_values(df, db.heuristic)
            types = _load_feature_types(db, df_name, anonymized=False)
            db._load_ordinal_orders(self.meta)
            order = db.ordinal_orders.get(self.meta.tag, None)
            df, _, _, _ = db._encode_df(df, mv, types, order=order,
                                        encode=self.meta.encode_transform)
            self._X_extra_base = df
            self._X_extra_base.sort_index(inplace=True)

        if self.meta.encode_select and self._X_select_base is not None:
            df = self._X_select_base
            mv = get_missing_values(df, db.heuristic)
            types = _load_feature_types(db, df_name, anonymized=False)
            db._load_ordinal_orders(self.meta)
            order = db.ordinal_orders.get(self.meta.tag, None)
            df, _, _, _ = db._encode_df(df, mv, types, order=order,
                                        encode=self.meta.encode_select)
            self._X_select_base = df
            self._X_select_base.sort_index(inplace=True)

        self.check_index_consistency()

    def _load_X_y(self):
        """Load a dataframe from taskmeta (X and y)."""
        if self._X_extra_base is None and self._X_select_base is None:
            self._load_X_base()

        # Step 6: Derive new set of features if any
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
            df.sort_index(inplace=True)
            self._X_extra = df

        # Step 7: Drop unwanted features if output specified
        select = self.meta.select
        if select and select.output_features:
            df = self._X_select_base
            features_to_keep = select.output_features
            features = set(df.columns)
            features_to_drop = features - set(features_to_keep)
            df = df.drop(features_to_drop, axis=1)
            df.sort_index(inplace=True)
            self._X_select = df
        else:
            self._X_select = self._X_select_base

        self.check_index_consistency()

    def check_index_consistency(self):
        """Check whether all indexes are equal."""
        dfs = [self._y, self._X_extra, self._X_extra_base, self._X_extra_unenc,
               self._X_select, self._X_select_base, self._X_select_unenc]

        indexes = [df.index for df in dfs if df is not None]

        for i in range(len(indexes)-1):
            idx1 = indexes[i]
            idx2 = indexes[i+1]
            assert idx1.equals(idx2)
