
import logging
import argparse
from time import time
import numpy as np
import pandas as pd
import os
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, \
    HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split

from prediction.tasks import tasks


logger = logging.getLogger(__name__)

# Parser config
parser = argparse.ArgumentParser(description='Test fit time.')
parser.add_argument('program')
parser.add_argument('task_name', nargs='?', default=None)
parser.add_argument('est', nargs='?', default=None)
parser.add_argument('imp', nargs='?', default=None)
parser.add_argument('--max-iter', default=10,
                    dest='max_iter', type=int)


def run(argv=None):
    """Emulate a HP search and monitor fit time."""
    args = parser.parse_args(argv)

    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'Mean+mask': SimpleImputer(strategy='mean', add_indicator=True),
        'Med': SimpleImputer(strategy='median'),
        'Med+mask': SimpleImputer(strategy='median', add_indicator=True),
        'Iterative': IterativeImputer(max_iter=args.max_iter),
        'Iterative+mask': IterativeImputer(add_indicator=True,
                                           max_iter=args.max_iter),
        'KNN': KNNImputer(),
        'KNN+mask': KNNImputer(add_indicator=True),

    }

    task_name = args.task_name
    est = args.est
    imp = imputers.get(args.imp, None)

    if task_name is None or est is None:
        logger.info('No argv given.')
        task_name = 'TB/shock_hemo'
        est = 'HGBC'

    task = tasks[task_name]
    logger.info(f'Argv given. Task {task.meta.tag}. est {est}.')

    t0 = time()
    logger.info('Getting X.')
    X = task.X
    logger.info('Getting y.')
    y = task.y

    logger.info(f'X shape before splits: {X.shape}')

    # Simulate the outer CV (the one of KFold)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    # Simulate the inner CV (the one of RandomSearchCV)
    X_train2, X_test2, y_train2, _ = train_test_split(X_train, y_train, test_size=0.2)

    # Now X has the same shape as in real experiment
    logger.info(f'X shape: {X_train2.shape}')

    t_X_ready = time()

    if imp is not None:
        logger.info(f'Fitting imputer {args.imp}')
        imp.fit(X_train2, y_train2)
        t_fit_imp = time()
        logger.info('Imputer fitted.')

        logger.info('Transforming X_train')
        imp.transform(X_train2)
        t_tra1_imp = time()
        logger.info('X_train transformed')

        logger.info('Transforming X_test')
        imp.transform(X_test2)
        t_tra2_imp = time()
        logger.info('X_test transformed')

    t_fits = [time()]
    for learning_rate in [0.05, 0.1, 0.3]:
        for max_depth in [3, 6, 9]:
            if est == 'HGBC':
                estimator = HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )
            elif est == 'HGBR':
                estimator = HistGradientBoostingRegressor(
                    loss='least_absolute_deviation',
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )
            else:
                raise ValueError(f'Unknown estimator {est}')

            logger.info(f'Params: LR {learning_rate} MD {max_depth}')
            logger.info('Fitting estimator.')
            estimator.fit(X_train2, y_train2)
            t_fits.append(time())
            logger.info('Estimator fitted.')

    t_fits = np.diff(t_fits)

    data = {
        'task_tag': [task.meta.tag],
        'imp': [args.imp],
        'imp_params': [repr({'max_iter': args.max_iter})],
        'X_shape': [repr(X.shape)],
        'X_train_shape': [repr(X_train2.shape)],
        'X_test_shape': [repr(X_test2.shape)],
        'time_X_ready': [t_X_ready-t0],
        'time_fit_imp': np.around([0 if imp is None else t_fit_imp-t_X_ready], 2),
        'time_tra1_imp': np.around([0 if imp is None else t_tra1_imp-t_X_ready], 2),
        'time_tra2_imp': np.around([0 if imp is None else t_tra2_imp-t_tra1_imp], 2),
        'time_fits': [repr(np.around(t_fits.tolist(), 2))],
        'time_fits_mean': [np.around(t_fits.mean(), 2)]
    }

    new_df = pd.DataFrame(data)

    df = None
    filepath = 'results/fit_time.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0)

    if df is not None:
        new_df = pd.concat([df, new_df])

    new_df.to_csv(filepath)
