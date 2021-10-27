"""Pipeline to train model, find best parameters, give results."""
import logging
import os
import time
from os.path import join, relpath

import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from .DumpHelper import DumpHelper
from .TimerStep import TimerStep

logger = logging.getLogger(__name__)


def train(task, strategy, RS=None, dump_idx_only=False, T=0, n_bagging=None,
          train_size=None, n_permutation=None):
    """Train a model (strategy) on some data (task) and dump results.

    Parameters
    ----------
    task : Task object
        Define a prediction task. Used to retrieve the data of the wanted task.
    strategy : Strategy object
        Define the method (imputation + model) to use.
    RS : int
        Define a random state.
    T : int
        Trial number for the ANOVA selection step, from 1 to 5 if 5 trials for
        the ANOVA selection.
        Used only for names of folder when dumping results.
    n_bagging : bool
        Whether to use bagging.

    """
    if task.is_classif() != strategy.is_classification() and not dump_idx_only:
        raise ValueError('Task and strategy mix classif and regression.')

    X, y = task.X, task.y  # Expensive data retrieval is hidden here

    logger.info(f'Started task "{task.meta.tag}" '
                f'using "{strategy.name}" strategy on "{task.meta.db}".')
    logger.info(f'X shape: {X.shape}')
    logger.info(f'y shape: {y.shape}')

    if RS is not None:
        logger.info(f'Resetting strategy RS to {RS}')
        strategy.reset_RS(RS)  # Must be done before init DumpHelper

    dh = DumpHelper(task, strategy, RS=RS, T=T, n_bagging=n_bagging)  # Used to dump results

    # Create timer steps used in the pipeline to time training time
    timer_start = TimerStep('start')
    timer_mid = TimerStep('mid')

    # Create pipeline with imputation and hyper-parameters tuning
    if strategy.imputer is not None:  # Has an imputation step
        logger.info('Creating pipeline with imputer.')
        steps = [
            ('timer_start', timer_start),
            ('imputer', strategy.imputer),  # Imputation step
            ('timer_mid', timer_mid),
            ('searchCV_estimator', strategy.search),  # HP tuning step
        ]
    else:
        logger.info('Creating pipeline without imputer.')
        steps = [
            ('timer_mid', timer_mid),
            ('searchCV_estimator', strategy.search),  # HP tuning step
        ]

    estimator = Pipeline(steps)

    if n_bagging is not None:
        global_timer_start = TimerStep('global_start')
        Bagging = BaggingClassifier if strategy.is_classification() else BaggingRegressor
        estimator = Bagging(estimator, n_estimators=n_bagging, random_state=RS)
        estimator = Pipeline([
            ('global_timer_start', global_timer_start),
            ('bagged_estimator', estimator),
        ])
        print(f'Using {Bagging} with {n_bagging} estimators and RS={RS}.')

    logger.info('Before size loop')
    # Size of the train set
    train_set_steps = strategy.train_set_steps if train_size is None else [train_size]
    for n in train_set_steps:
        print(f'SIZE {n}')
        logger.info(f'Size {n}')
        n_tot = X.shape[0]
        if n_tot - n < strategy.min_test_set*n_tot:
            # Size of the test set too small, skipping
            continue

        # Choose right splitter depending on classification or regression
        if task.is_classif():
            ss = StratifiedShuffleSplit(n_splits=strategy.n_splits,
                                        test_size=n_tot-n,
                                        random_state=RS)
        else:
            ss = ShuffleSplit(n_splits=strategy.n_splits, test_size=n_tot-n,
                              random_state=RS)

        # Repetedly draw train and test sets
        for i, (train_idx, test_idx) in enumerate(ss.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f'FOLD {i}')

            # Used to save the IDs of the sub-sampled dataset.
            if dump_idx_only:
                logger.info(f'Dumped IDs of {task.meta.tag}, size={n}, trial={T}, fold={i}')
                folder = relpath('ids/')
                os.makedirs(folder, exist_ok=True)
                name = task.meta.name.replace('pvals', 'screening')
                trial = int(T) + 1
                fold = i + 1
                common = f'{task.meta.db}-{name}-size{n}-trial{trial}-fold{fold}'
                filepath_idx_train = join(folder, f'{common}-train-idx.csv')
                filepath_idx_test = join(folder, f'{common}-test-idx.csv')
                filepath_col_train = join(folder, f'{common}-train-col.csv')
                filepath_col_test = join(folder, f'{common}-test-col.csv')
                pd.Series(X_train.index).to_csv(filepath_idx_train, index=False)
                pd.Series(X_test.index).to_csv(filepath_idx_test, index=False)
                pd.Series(X_train.columns).to_csv(filepath_col_train, index=False, header=False)
                pd.Series(X_test.columns).to_csv(filepath_col_test, index=False, header=False)
                continue  # when dumping IDs, we skip prediction

            logger.info(f'Fold {i}: Started fitting the estimator')
            estimator.fit(X_train, y_train)
            logger.info('Ended fitting the estimator')

            def compute_times(start, mid, end):
                return {
                    'imputation': round(mid - start, 6) if start else None,
                    'tuning': round(end - mid, 6),
                }

            if n_bagging is None:
                # Retrieve fit times from timestamps
                end_ts = time.time()  # Wall-clock time
                start_ts = timer_start.last_fit_timestamp
                mid_ts = timer_mid.last_fit_timestamp

                end_pt = time.process_time()  # Process time (!= Wall-clock time)
                start_pt = timer_start.last_fit_pt
                mid_pt = timer_mid.last_fit_pt

                times = compute_times(start_ts, mid_ts, end_ts)
                pts = compute_times(start_pt, mid_pt, end_pt)
                imputation_time = times['imputation']
                tuning_time = times['tuning']
                imputation_pt = pts['imputation']
                tuning_pt = pts['tuning']

            else:
                end_ts = time.time()  # Wall-clock time
                start_ts = global_timer_start.last_fit_timestamp

                end_pt = time.process_time()  # Process time (!= Wall-clock time)
                start_pt = global_timer_start.last_fit_pt

                # No mid_timer for bagged estimator
                times = compute_times(start_ts, start_ts, end_ts)
                pts = compute_times(start_pt, start_pt, end_pt)
                imputation_time = times['imputation']
                tuning_time = times['tuning']
                imputation_pt = pts['imputation']
                tuning_pt = pts['tuning']

            # Dump fit times
            dh.dump_times(imputation_time, tuning_time,
                          imputation_pt, tuning_pt,
                          fold=i, tag=str(n))

            # Predict
            if strategy.is_classification() and strategy.roc:  # ROC asked
                # Compute probas for retrieving ROC curve
                probas = estimator.predict_proba(X_test)
                logger.info('Started predict_proba')
                # y_pred = np.argmax(probas, axis=1)
                dh.dump_probas(y_test, probas, fold=i, tag=str(n))
                y_pred = estimator.predict(X_test)
            else:
                # No need for probas, only predictions
                if not strategy.is_classification():
                    logger.info('ROC: not a classification.')
                elif not strategy.roc:
                    logger.info('ROC: not wanted.')

                logger.info('Started predict')
                y_pred = estimator.predict(X_test)

            # Dump results
            logger.info(f'Fold {i}: Ended predict.')
            dh.dump_prediction(y_pred, y_test, fold=i, tag=str(n))

            if n_permutation is not None:
                scoring = 'roc_auc' if strategy.is_classification() else 'r2'
                r = permutation_importance(estimator, X_test, y_test,
                                           n_repeats=n_permutation,
                                           random_state=RS, scoring=scoring)

                importances = pd.DataFrame(r.importances.T, columns=X_train.columns)
                importances.index.rename('repeat', inplace=True)
                importances = importances.reindex(sorted(importances.columns), axis=1)

                dh.dump_importances(importances, fold=i, tag=str(n))

                mv_props = X_test.isna().sum(axis=0)/X_test.shape[0]
                mv_props.rename(i, inplace=True)
                mv_props = mv_props.to_frame().T
                mv_props = mv_props.reindex(sorted(mv_props.columns), axis=1)
                dh.dump_mv_props(mv_props, fold=i, tag=str(n))
