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

from prediction.tasks import tasks
from database import dbs
from df_utils import dtype_from_types
from database.constants import CATEGORICAL, ORDINAL, BINARY, CONTINUE_R, \
    CONTINUE_I, NOT_A_FEATURE, NOT_MISSING, DATE_TIMESTAMP, DATE_EXPLODED, \
    METADATA_PATH, MV_PLACEHOLDER


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

# Parser config
parser = argparse.ArgumentParser(description='Prediction program.')
parser.add_argument('program')
parser.add_argument('task_name', nargs='?', default=None)
parser.add_argument('--RS', dest='RS', default=None, nargs='?',
                    help='The random state to use.')

sep = ','
encoding = 'ISO-8859-1'


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""
    args = parser.parse_args(argv)

    print('Retrieving task')
    task_name = args.task_name
    task = tasks[task_name]

    # print('Setting dump helper')
    # dh = DumpHelper(task, None)  # Used to dump results

    if task.is_classif():
        logger.info('Classification, using f_classif')
        f_callable = f_classif
    else:
        logger.info('Regression, using f_regression')
        f_callable = f_regression

    print('Retreiving db')
    db = dbs[task.meta.db]
    df_path = db.frame_paths[task.meta.df_name]

    print('Loading y')
    # ICD9 = '41271-0.0'
    # ICD9_main = '41203-0.0'
    # ICD9_sec = '41205-0.0'

    # ICD10 = '41270-0.0'
    # ICD10_main = '41202-0.0'
    # ICD10_sec = '41204-0.0'
    # cancer10 = '40006-0.0'
    # cancer9 = '40013-0.0'
    # sex = '31-0.0'

    df = pd.read_csv(df_path, usecols=task.meta.keep, sep=sep, encoding=encoding)
    # df = pd.read_csv(df_path, usecols=['48-0.0'], sep=sep, encoding=encoding)
    index = df.index
    print('Transforming')
    df = task.meta.transform(df)
    print('Getting y')
    y = df[task.meta.predict]
    # y = df['48-0.0']
    # idx_to_drop_y = set(y.index[y.isna()])
    idx_to_keep_y = set(y.index)
    idx_to_drop_y = set(index) - idx_to_keep_y
    # df = df.dropna(axis=0, subset=['48-0.0'])
    # y = y.drop(idx_to_drop_y)
    print(y.shape)

    base, ext = os.path.splitext(df_path)
    df_transposed_path = f'{base}_transposed{ext}'

    # df_t = pd.read_csv(df_transposed_path, sep=sep, encoding=encoding, nrows=0)

    # print(df_t.shape)
    # exit()

    # Load types
    print('Loading types')
    db._load_feature_types(task.meta)
    types = db.feature_types[task.meta.tag]

    # type_to_dtype = {
    #     CATEGORICAL: 'category',
    #     # ORDINAL: 'category',
    #     BINARY: 'category',
    #     CONTINUE_R: np.float32,
    #     CONTINUE_I: 'Int32',
    #     NOT_A_FEATURE: 'object',
    #     DATE_TIMESTAMP: 'object',
    #     DATE_EXPLODED: 'object'
    # }

    # print('Converting to dtypes')
    # dtype = dtype_from_types(types, type_to_dtype)

    # print('Dask reading')
    # df = dd.read_csv(df_path, sep=sep, encoding=encoding, dtype=dtype)
    # df = dd.read_csv(df_path, sep=sep, encoding=encoding, dtype=dtype)

    file = open(df_transposed_path, 'r', newline='')

    reader = csv.reader(file)

    # df = pd.read_csv(df_path, usecols=['eid'], sep=sep, encoding=encoding)
    # print(df)
    # y = df['eid']

    # def my_gen():
    #     i = 0
    #     while i < 10:
    #         yield next(reader)
    #         i+=1

    # for row in my_gen():
    #     print(len(row))
    #     print(len(row[1:]))
    #     print(y.shape)
    #     print(f'0: {row[0]}')
    #     print(f'1: {row[1]}')
    #     print(f'2: {row[2]}')
    #     print(f'-1: {row[-1]}')
    #     print(f'-2: {row[-2]}')

    # exit()


    # for i in range(10):
    #     row = next(reader)
    #     feature_name = row[0]
    #     feature_content = row[1:]
    #     print(feature_name)
    # exit()
    # q = queue.Queue()

    def pval_one_feature(x, y):
        # Drop rows wih missing values both in f and y
        # print(x)
        x = pd.Series(x, index=index)
        x.replace(to_replace='', value=np.nan, inplace=True)
        x = x.astype(float)
        idx_to_drop_x = set(x.index[x.isna()])
        idx_to_drop = idx_to_drop_x.union(idx_to_drop_y)
        # print(idx_to_drop)
        x = x.drop(idx_to_drop, axis=0)
        # return x
        y_dropped = y.drop(set(y.index).intersection(idx_to_drop_x), axis=0)

        y = x.to_numpy()
        x = x.to_numpy().reshape(-1, 1)
        y_dropped = y_dropped.to_numpy().reshape(-1)

        F, pval = f_callable(x, y_dropped)

        return pval[0]

    def handler(row, y):
        name, x = row[0], row[1:]
        print(name)
        # print()
        # print(len(row))
        # print(len(x))
        # print(len(y))
        # print()
        if name == '':
            return
        # if name not in types:
        #     print(f'"{name}"" not in types, skipping')
        #     return
        t = types[name]

        if t == CATEGORICAL:
            # categorical encode
            df = pd.DataFrame({name: x})
            df = df.astype(str)
            df.replace(to_replace='', value=np.nan, inplace=True)

            enc = OneHotEncoder(sparse=False)

            # Cast to str to prevent: "argument must be a string or number" error
            # which occurs when mixed types floats and str

            # Fill missing values with a placeholder
            df.fillna('MISSING_VALUE', inplace=True)

            # Fit transform the encoder
            data_encoded = enc.fit_transform(df)

            feature_names = list(enc.get_feature_names(list(df.columns)))

            # print(feature_names)

            df_encoded = pd.DataFrame(data_encoded,
                                      index=df.index,
                                      columns=feature_names
                                      )
            L = []
            for f in df_encoded:
                print(f'\t{f}')
                L.append((f, pval_one_feature(df_encoded[f], y)))
            return L
            # return [(f, pval_one_feature(df_encoded[f], y)) for f in df_encoded]

        elif t == CONTINUE_R or t == CONTINUE_I:
            return [(name, pval_one_feature(x, y))]

        print(f'"{name}"" ignored ')


    def my_gen():
        i = 0
        while i < 10:
            yield next(reader)
            i+=1

    res = Parallel(n_jobs=1, require='sharedmem')(delayed(handler)
                                                  (row, y) for row in my_gen())

    res = [r for r in res if r is not None]

    res = functools.reduce(lambda x, y: x+y, res)
    print(res)

    names, pvals = zip(*res)
    # print(names)
    # print(pvals)
    pvals = pd.Series(pvals, index=names)
    print(pvals)
    dump_path = f'selected/{task.meta.tag}/pvals.csv'
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    pvals.to_csv(dump_path, header=False)
    file.close()
    exit()


    # print('Start loop')
    # for i, f in enumerate(df):# if types[f] in [CONTINUE_I, CONTINUE_R]
    #     print(f'{i}: {f}')
    #     handler(df[f], q)

    # file.close()
    # exit()

    # # def pval_one_feature(X, f, y, pvals):
    # #     logger.info(f'Feature {f}')
    # #     feature = X[f]

    # #     # Drop rows wih missing values both in f and y
    # #     idx_to_drop = feature.index[feature.isna()]
    # #     feature = feature.drop(idx_to_drop, axis=0)
    # #     y_dropped = y.drop(idx_to_drop, axis=0)

    # #     feature = feature.to_numpy().reshape(-1, 1)
    # #     y_dropped = y_dropped.to_numpy().reshape(-1)

    # #     F, pval = f_callable(feature, y_dropped)

    # #     pvals[f] = pval[0]

    # pvals = dict()

    # series = pd.Series(pvals)
    # dh.dump_pvals(series)
