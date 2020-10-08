"""Pipeline to train model, find best parameters, give results."""
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import time

from .DumpHelper import DumpHelper
from .TimerStep import TimerStep


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


def train(task, strategy, RS=None, **kwargs):
    """Train a model (strategy) on some data (task) and dump results.

    Parameters
    ----------
    task : Task object
        Define a prediction task. Used to retrieve the data of the wanted task.
    strategy : Strategy object
        Define the method (imputation + model) to use.
    RS : int
        Define a random state.
    **kwargs : dict
        T expected: trial number for the ANOVA selection step, from 1 to 5
            if 5 trials for the ANOVA selection. Used only for names of folder
            when dumping results.

    """
    if task.is_classif() != strategy.is_classification():
        raise ValueError('Task and strategy mix classif and regression.')

    assert 'T' in kwargs
    T = kwargs['T']  # Trial number (ANOVA), only used for dumping names here

    X, y = task.X, task.y  # Expensive data retrieval is hidden here

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

    logger.info('Before size loop')
    # Size of the train set
    for n in strategy.train_set_steps:
        logger.info(f'Size {n}')
        n_tot = X.shape[0]
        if n_tot - n < strategy.min_test_set*n_tot:
            # Size of the test set too small, skipping
            continue

        # Choose right splitter depending on classification or regression
        if strategy.is_classification():
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

            logger.info(f'Fold {i}: Started fitting the estimator')
            estimator.fit(X_train, y_train)
            logger.info('Ended fitting the estimator')

            # Retrieve fit times from timestamps
            end_ts = time.time()  # Wall-clock time
            start_ts = timer_start.last_fit_timestamp
            mid_ts = timer_mid.last_fit_timestamp

            end_pt = time.process_time()  # Process time (!= Wall-clock time)
            start_pt = timer_start.last_fit_pt
            mid_pt = timer_mid.last_fit_pt

            imputation_time = round(mid_ts - start_ts, 6) if start_ts else None
            tuning_time = round(end_ts - mid_ts, 6)
            imputation_pt = round(mid_pt - start_pt, 6) if start_pt else None
            tuning_pt = round(end_pt - mid_pt, 6)

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
