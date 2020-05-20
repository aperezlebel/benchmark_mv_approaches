"""Implement the new way of coding Tasks."""
import pandas as pd
from dataclass import dataclass
import logging

from missing_values import get_missing_values
from features_type import _load_feature_types
from ...database import dbs
from .transform import Transform


@dataclass
class TaskMeta(object):

    name: str
    db: str
    df_name: str
    classif: bool

    predict: Transform
    transform: Transform = None
    ids_selection: Transform = None
    encoding: Transform = None
    encode: str = None

    def get_infos(self):
        """Return a dict containing infos on the object."""
        return {
            'name': self.name,
            'db': self.db,
            'df_name': self.df_name,
            'classif': self.classif,
            'ids_selection': self.ids_selection.get_infos(),
            'predict': self.predict.get_infos(),
            'encode': self.encode,
        }

    @property
    def tag(self):
        return f'{self.db}/{self.name}'


class Task(object):
    """Gather a TaskMeta and a dataframe."""

    def __init__(self, meta):
        """Init."""
        self.meta = meta
        self._X_base = None
        self._X_extra = None
        self._y = None

    @property
    def X(self):
        """Input dataset."""
        if not all((self._X_base, self._X_extra)):
            self._load()

        if not self._X_base:
            return self._X_extra

        if not self._X_extra:
            return self._X_base

        return pd.concat((self._X_base, self._X_extra), axis=1)

    @property
    def y(self):
        """Feature to predict."""
        if not self._y:
            self._load()
        return self._y

    def _load(self):
        """Load a dataframe from taskmeta."""
        # Step 1: get dataframe's path and load infos
        logging.debug('Get df path and load infos')
        db = dbs[self.meta.db]
        df_name = self.meta.df_name
        df_path = db.frame_paths[df_name]
        sep = db.sep
        encoding = db._encoding

        # Step 2: Derive indexes to drop if any
        ids_transformer = self.meta.ids_selection
        if ids_transformer:
            logging.debug('Derive indexes to drop.')
            features_to_load = self.meta.ids_selection.input_features
            df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                             usecols=features_to_load)
            logging.debug(f'Loaded df of shape {df.shape}.')
            df = self.meta.ids_selection.transform(df)
            idx_to_drop = df.index
            rows_to_drop = idx_to_drop + 1  # Rows start to 1 wih header

        # Step 3: Derive the feature to predict y
        logging.debug('Derive the feature to predict y.')
        features_to_load = self.meta.predict.input_features
        df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                         usecols=features_to_load, skiprows=rows_to_drop)
        logging.debug(f'Loaded df of shape {df.shape}.')

        if not len(self.meta.predict.output_features) != 1:
            raise ValueError('Expected only one item in output features'
                             'for deriving predict.')

        predict_name = self.meta.predict.output_features[0]
        self._y = df[predict_name]

        # Step 4: Add NAN values of y to index to drop
        y_mv = get_missing_values(self._y, db.heuristic)
        idx_to_drop_y = self._y.index[y_mv != 0]
        idx_to_drop += idx_to_drop_y
        rows_to_drop = idx_to_drop + 1  # Rows start to 1 with header

        # Step 5: Derive new set of features if any
        transform = self.meta.transform
        if transform:
            features_to_load = transform.input_features
            df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                             usecols=features_to_load, skiprows=rows_to_drop)
            logging.debug(f'Loaded df of shape {df.shape}.')
            df = transform.transform(df)
            features = set(df.columns)
            features_to_keep = set(transform.output_features)
            features_to_drop = features - features_to_keep
            df.drop(features_to_drop, index=1, inplace=True)
            self._X_extra = df

        # Step 6: Load asked features
        encoding = self.meta.encoding
        if encoding:
            features_to_load = encoding.get_parent(encoding.input_features)
            df = pd.read_csv(df_path, sep=sep, encoding=encoding,
                             usecols=features_to_load, skiprows=rows_to_drop)

            # Step 7: Encode loaded features
            mv = get_missing_values(df, db.heuristic)
            db._load_feature_types(self.meta)
            types = _load_feature_types(db, df_name, anonymized=False)
            db._load_ordinal_orders(self.meta)
            order = db.ordinal_orders[self.meta.tag]
            df, _, _, _ = db._encode_df(df, mv, types, order=order,
                                        encode=self.meta.encode)

            # Step 8: Drop unwanted features
            features_to_keep = encoding.get_encoded(encoding.output_features)
            features = set(df.columns)
            features_to_drop = features - set(features_to_keep)
            df.drop(features_to_drop, axis=1, inplace=True)
            self._X_base = df
