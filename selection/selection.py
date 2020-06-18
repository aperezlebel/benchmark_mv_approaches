"""Run the Anova feature selection of scikit-learn."""
import argparse
import pandas as pd
import logging
from sklearn.feature_selection import f_classif, f_regression
from joblib import Parallel, delayed
import numpy as np
import csv
import os
from sklearn.preprocessing import OneHotEncoder
import functools

from missing_values import get_missing_values
from df_utils import fill_df
from prediction.tasks import tasks
from database import dbs
from database.constants import CATEGORICAL, CONTINUE_R, CONTINUE_I, BINARY, ORDINAL
from prediction.tasks.transform import Transform
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

# Parser config
parser = argparse.ArgumentParser(description='Prediction program.')
parser.add_argument('program')
parser.add_argument('task_name', nargs='?', default=None)
parser.add_argument('--RS', dest='RS', default=0, nargs='?',
                    help='The random state to use.')
parser.add_argument('--T', dest='T', default=0, nargs='?',
                    help='The trial #.')
parser.add_argument('--TMAX', dest='TMAX', default=5, nargs='?',
                    help='The max # of trials.')

sep = ','
encoding = 'ISO-8859-1'


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""
    args = parser.parse_args(argv)

    print('Retrieving task')
    RS = int(args.RS)
    T = int(args.T)
    TMAX = int(args.TMAX)
    print(f'RS {RS} T {T} TMAX {TMAX}')
    task_name = args.task_name
    task = tasks.get(task_name, n_top_pvals=None)

    temp_dir = f'selected/{task.meta.tag}/temp/'

    print('Retreiving db')
    db = dbs[task.meta.db]

    print('Retrieving y')
    y = task.y
    print(f'y loaded with shape {y.shape}')

    if task.is_classif():
        logger.info('Classification, using f_classif')
        f_callable = f_classif
        ss = StratifiedShuffleSplit(n_splits=TMAX,
                                    test_size=2/3,
                                    random_state=RS)
    else:
        logger.info('Regression, using f_regression')
        f_callable = f_regression
        ss = ShuffleSplit(n_splits=TMAX, test_size=2/3,
                          random_state=RS)

    index = y.index

    assert T >= 0

    # Alter the task to select only 1/3 for selection
    split_iter = ss.split(y, y)
    for _ in range(T+1):
        keep_idx, drop_idx = next(split_iter)

    # Convert to index
    keep_index = [index[i] for i in keep_idx]
    drop_index = [index[i] for i in drop_idx]

    def select_idx(df):
        """Define the idx to keep from the database."""
        return df.drop(drop_index, axis=0)

    task.meta.idx_selection = Transform(
        input_features=[],
        transform=select_idx,
    )

    series = pd.Series(keep_index)
    dump_path = f'pvals/{task.meta.tag}/RS{RS}-T{T}-used_idx.csv'
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    series.to_csv(dump_path, header=None, index=False)
    print(f'Idx used of shape {series.size}')

    # Ignore existing pvals selection
    task.meta.select = None
    task.meta.encode_select = 'ordinal'

    # Force reload y to take into account previous change
    task._load_y()
    y = task.y
    print(f'y reloaded with shape {y.shape}')

    index = y.index

    temp_df_transposed_path = temp_dir+'X_transposed.csv'

    if not os.path.exists(temp_df_transposed_path):
        print('Retrieving X')
        X = task.X
        print(f'X loaded with shape {X.shape}')

        os.makedirs(temp_dir, exist_ok=True)
        X_t = X.transpose()
        X_t.to_csv(temp_df_transposed_path, quoting=csv.QUOTE_ALL)

    X_t = pd.read_csv(temp_df_transposed_path, iterator=True, chunksize=1,
                      index_col=0)

    # Load types
    print('Loading types')
    db._load_feature_types(task.meta)
    types = db.feature_types[task.meta.tag]

    def pval_one_feature(x, y):
        # Drop rows wih missing values both in f and y
        x = pd.Series(x, index=index)
        x.replace(to_replace='', value=np.nan, inplace=True)
        x = x.astype(float)
        idx_to_drop = set(x.index[x.isna()])
        x = x.drop(idx_to_drop, axis=0)
        y_dropped = y.drop(idx_to_drop, axis=0)

        x = x.to_numpy().reshape(-1, 1)
        y_dropped = y_dropped.to_numpy().reshape(-1)

        assert x.shape[0] == y_dropped.shape[0]

        if x.shape[0] < 0.01*index.size:  # Not enough sample, skipping
            return None

        _, pval = f_callable(x, y_dropped)

        return pval[0]

    def handler(row, y):
        name = row.index[0]
        x = np.squeeze(np.transpose(row.to_numpy()))
        print(name)

        if name == '':
            return

        t = types[name]

        if t == CATEGORICAL or t == BINARY:
            # categorical encode
            df = pd.DataFrame({name: x})
            df = df.astype(str)
            df.replace(to_replace='', value=np.nan, inplace=True)

            enc = OneHotEncoder(sparse=False)

            # Cast to str to prevent: "argument must be a string or number"
            # error which occurs when mixed types floats and str

            # Fill missing values with a placeholder
            df.fillna('MISSING_VALUE', inplace=True)

            # Fit transform the encoder
            data_encoded = enc.fit_transform(df)

            feature_names = list(enc.get_feature_names(list(df.columns)))

            df_encoded = pd.DataFrame(data_encoded,
                                      index=df.index,
                                      columns=feature_names
                                      )
            L = []
            for f in df_encoded:
                print(f'\t{f}')
                L.append((f, pval_one_feature(df_encoded[f], y)))
            return L

        elif t == CONTINUE_R or t == CONTINUE_I or t == ORDINAL:
            return [(name, pval_one_feature(x, y))]

        print(f'"{name}" ignored ')

    res = Parallel(n_jobs=-1, require='sharedmem')(delayed(handler)
                                                   (row, y) for row in X_t)

    res = [r for r in res if r is not None]

    res = functools.reduce(lambda x, y: x+y, res)
    print(res)

    names, pvals = zip(*res)

    pvals = pd.Series(pvals, index=names)
    print(pvals)
    dump_path = f'pvals/{task.meta.tag}/RS{RS}-T{T}-pvals.csv'
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    pvals.to_csv(dump_path, header=False)
