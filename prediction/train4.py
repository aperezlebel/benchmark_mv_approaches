"""Pipeline to train model, find best parameters, give results."""
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_predict, \
    StratifiedShuffleSplit
import numpy as np
import time

from .DumpHelper import DumpHelper
from .TimerStep import TimerStep


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


def train(task, strategy, RS=None, **kwargs):
    if task.is_classif() != strategy.is_classification():
        raise ValueError('Task and strategy mix classif and regression.')

    assert 'T' in kwargs
    T = kwargs['T']

    X, y = task.X, task.y

    logger.info(f'Started task "{task.meta.tag}" '
                f'using "{strategy.name}" strategy on "{task.meta.db}".')
    logger.info(f'X shape: {X.shape}')
    logger.info(f'y shape: {y.shape}')

    if RS is not None:
        logger.info(f'Resetting strategy RS to {RS}')
        strategy.reset_RS(RS)  # Must be done before init DumpHelper

    dh = DumpHelper(task, strategy, RS=RS, T=T)  # Used to dump results

    if RS is None:
        RS = 42

    # Create pipeline
    timer_start = TimerStep('start')
    timer_mid = TimerStep('mid')
    if strategy.imputer is not None:
        logger.info('Creating pipeline with imputer.')
        steps = [
            # ('log1', FakeStep('imputer')),
            ('timer_start', timer_start),
            ('imputer', strategy.imputer),
            ('timer_mid', timer_mid),
            # ('log2', FakeStep('searchCV_estimator')),
            ('searchCV_estimator', strategy.search),
        ]
    else:
        logger.info('Creating pipeline without imputer.')
        steps = [
            # ('log1', FakeStep('searchCV_estimator')),
            ('timer_mid', timer_mid),
            ('searchCV_estimator', strategy.search),
        ]

    estimator = Pipeline(steps)

    # Size of the train set
    for n in strategy.train_set_steps:
        n_tot = X.shape[0]
        if n_tot - n < strategy.min_test_set*n_tot:
            # Size of the train set too small, skipping
            continue

        if strategy.is_classification():
            ss = StratifiedShuffleSplit(n_splits=strategy.n_splits,
                                        test_size=n_tot-n,
                                        random_state=RS)
        else:
            ss = ShuffleSplit(n_splits=strategy.n_splits, test_size=n_tot-n,
                              random_state=RS)

        for i, (train_idx, test_idx) in enumerate(ss.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            logger.info(f'Fold {i}: Started fitting the estimator')
            logger.info(f'Value count for y_train:\n{y_train.value_counts()}')
            logger.info(f'Value count for y_test:\n{y_test.value_counts()}')
            estimator.fit(X_train, y_train)
            logger.info('Ended fitting the estimator')

            # Store fit times
            end_ts = time.time()
            start_ts = timer_start.last_fit_timestamp
            mid_ts = timer_mid.last_fit_timestamp

            end_pt = time.process_time()
            start_pt = timer_start.last_fit_pt
            mid_pt = timer_mid.last_fit_pt

            imputation_time = round(mid_ts - start_ts, 6) if start_ts else None
            tuning_time = round(end_ts - mid_ts, 6)
            imputation_pt = round(mid_pt - start_pt, 6) if start_pt else None
            tuning_pt = round(end_pt - mid_pt, 6)
            dh.dump_times(imputation_time, tuning_time,
                          imputation_pt, tuning_pt,
                          fold=i, tag=str(n))

            # Predict
            if strategy.is_classification() and strategy.roc:
                probas = estimator.predict_proba(X_test)
                logger.info('Started predict_proba')
                # y_pred = np.argmax(probas, axis=1)
                dh.dump_probas(y_test, probas, fold=i, tag=str(n))
                y_pred = estimator.predict(X_test)
            else:
                if not strategy.is_classification():
                    logger.info('ROC: not a classification.')
                elif not strategy.roc:
                    logger.info('ROC: not wanted.')

                logger.info('Started predict')
                y_pred = estimator.predict(X_test)

            logger.info(f'Fold {i}: Ended predict.')
            dh.dump_prediction(y_pred, y_test, fold=i, tag=str(n))
