"""Pipeline to train model, find best parameters, give results."""
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
import numpy as np

from .DumpHelper import DumpHelper
from .FakeStep import FakeStep


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


def train(task, strategy, RS=None):
    X, y = task.X, task.y

    logger.info(f'Started task "{task.meta.tag}" '
                f'using "{strategy.name}" strategy on "{task.meta.db}".')
    logger.info(f'X shape: {X.shape}')
    logger.info(f'y shape: {y.shape}')

    logger.info(f'Resetting strategy RS to {RS}')
    strategy.reset_RS(RS)  # Must be done before init DumpHelper

    dh = DumpHelper(task, strategy, RS=RS)  # Used to dump results

    # Create pipeline
    if strategy.imputer is not None:
        logger.info('Creating pipeline with imputer.')
        steps = [
            ('log1', FakeStep('imputer')),
            ('imputer', strategy.imputer),
            ('log2', FakeStep('searchCV_estimator')),
            ('searchCV_estimator', strategy.search)
        ]
    else:
        logger.info('Creating pipeline without imputer.')
        steps = [
            ('log1', FakeStep('searchCV_estimator')),
            ('searchCV_estimator', strategy.search)
        ]

    estimator = Pipeline(steps)

    # Size of the train set
    for n in strategy.train_set_steps:
        n_tot = X.shape[0]
        if n_tot - n < strategy.min_test_set*n_tot:
            # Size of the train set too small, skipping
            continue

        sss = StratifiedShuffleSplit(n_splits=5, test_size=n_tot-n,
                                     random_state=RS)

        for i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            logger.info(f'Fold {i}: Started fitting the estimator')
            estimator.fit(X_train, y_train)
            logger.info('Ended fitting the estimator')

            if strategy.is_classification() and strategy.roc:
                probas = estimator.predict_proba(X_test)
                logger.info('Started predict_proba')
                y_pred = np.argmax(probas, axis=1)
                dh.dump_probas(y_test, probas, fold=i, tag=str(n))
            else:
                if not strategy.is_classification():
                    logger.info('ROC: not a classification.')
                elif not strategy.roc:
                    logger.info('ROC: not wanted.')

                logger.info('Started predict')
                y_pred = estimator.predict(X_test)

            logger.info(f'Fold {i}: Ended predict.')
            dh.dump_prediction(y_pred, y_test, fold=i, tag=str(n))

