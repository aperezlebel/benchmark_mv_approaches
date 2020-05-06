"""Run the Anova feature selection of scikit-learn."""
import argparse
import pandas as pd
import logging
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from prediction.tasks import tasks
from prediction.DumpHelper import DumpHelper


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

# Parser config
parser = argparse.ArgumentParser(description='Prediction program.')
parser.add_argument('program')
parser.add_argument('task_name', nargs='?', default=None)
parser.add_argument('--RS', dest='RS', default=None, nargs='?',
                    help='The random state to use.')


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""
    args = parser.parse_args(argv)

    task_name = args.task_name
    task = tasks[task_name]

    X, y = task.X, task.y

    dh = DumpHelper(task, None)  # Used to dump results

    pvals = dict()

    n_tot = X.shape[1]

    if task.is_classif():
        logger.info('Classification, using f_classif')
        f_callable = f_classif
    else:
        logger.info('Regression, using f_regression')
        f_callable = f_regression

    for i, feature_name in enumerate(X):
        logger.info(f'Feature {i+1} out of {n_tot}')
        feature = X[feature_name]

        # Drop rows wih missing values both in f and y
        idx_to_drop = feature.index[feature.isna()]
        feature = feature.drop(idx_to_drop, axis=0)
        y_dropped = y.drop(idx_to_drop, axis=0)

        feature = feature.to_numpy().reshape(-1, 1)
        y_dropped = y_dropped.to_numpy().reshape(-1)

        F, pval = f_callable(feature, y_dropped)

        pvals[feature_name] = pval[0]

    dh.dump_pvals(pd.Series(pvals))
