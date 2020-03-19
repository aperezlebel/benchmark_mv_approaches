import os
import logging
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


from prediction.tasks import tasks


logger = logging.getLogger(__name__)


def run(argv=None):
    if argv is None or len(argv) < 2:
        logger.info('No argv given.')
        task = tasks['TB/shock_hemo']
        imp = 'iterative'
    else:
        task = tasks[argv[1]]
        imp = argv[2]
        logger.info(f'Argv given. Task {task.meta.tag}. Imp {imp}.')

    logger.info('Getting X.')
    X = task.X
    logger.info('Getting y.')
    y = task.y

    logger.info(f'X shape before splits: {X.shape}')

    # Simulate the outer CV (the one of KFold)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    # Simulate the inner CV (the one of RandomSearchCV)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2)

    # Now X has the same shape as in real experiment
    logger.info(f'X shape: {X_train2.shape}')

    if imp == 'iterative':
        imp = IterativeImputer()
    elif imp == 'knn':
        imp = KNNImputer()

    t0 = time()

    logger.info('Fitting imputer.')
    imp.fit(X_train2)
    t1 = time()
    logger.info('Imputer fitted.')

    logger.info('Transforming X_train.')
    imp.transform(X_train2)
    t2 = time()
    logger.info('X_train transformed.')

    logger.info('Transforming X_test.')
    imp.transform(X_test2)
    t3 = time()
    logger.info('X_test transformed.')

    data = {
        'task_tag': [task.meta.tag],
        'imp': [imp.__class__.__name__],
        'X_shape': [repr(X.shape)],
        'X_train_shape': [repr(X_train2.shape)],
        'X_test_shape': [repr(X_test2.shape)],
        'fit_time': [t1-t0],
        'transform_time_train': [t2-t1],
        'transform_time_test': [t3-t2]
    }

    new_df = pd.DataFrame(data)

    df = None
    filepath = 'results/impute_time.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0)

    if df is not None:
        new_df = pd.concat([df, new_df])

    new_df.to_csv(filepath)

    print(new_df)

