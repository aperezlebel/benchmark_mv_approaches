"""Pipeline to train model, find best parameters, give results."""

# from sklearn.experimental import enable_hist_gradient_boosting
import logging
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

from .DumpHelper import DumpHelper


logger = logging.getLogger(__name__)


def impute(df, imputer):
    """Impute missing values given an already fitted imputer.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with missing values to impute
    imputer : sklearn imputer (instance of _BaseImputer)
        The already fitted imputer.

    Returns
    -------
    pd.DataFrame
        Data frame with imputed missing values. Extra columns might have
        been added depending on the imputer.

    """
    # Columns containing only missing values are discarded by the imputer
    discared_columns = df.columns[np.isnan(imputer.statistics_)]

    data_imputed = imputer.transform(df)

    # Manage the case where an indicator was used to add binary columns for MV
    indicator = imputer.indicator_
    # If any, get feature ids for which an indicator column has been created
    features_with_mv = indicator.features_ if indicator is not None else []
    # If any, create names for these extra features.
    extra_columns = [f'indicator_{df.columns[id]}' for id in features_with_mv]

    base_columns = [c for c in df.columns if c not in discared_columns]

    columns = base_columns+extra_columns

    return pd.DataFrame(data_imputed, index=df.index, columns=columns)


def train(task, strategy):
    """Train a model following a strategy on prediction task.

    Parameters:
    -----------
    task : Task object
        Contain the dataframe and the task metadata.
    strategy : Strategy object
        Describe the estimator and the strategy to train and find the best
        parameters.

    Returns:
    --------
    dict
        Stores the results of the training.

    """
    logger.info(f'Started task "{task.meta.tag}" '
                f'using "{strategy.name}" strategy on "{task.meta.db}".')
    dh = DumpHelper(task, strategy)  # Used to dump results

    X, y = task.X, task.y

    # Non nested CV
    # X_train, X_test, y_train, y_test = strategy.split(X, y)

    # Nested CV
    Estimators = []

    for i, (train_index, test_index) in enumerate(strategy.outer_cv.split(X)):
        logger.info(f'Started fold {i}.')

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Imputation
        if strategy.imputer is not None:
            logger.info('Fitting imputation.')
            strategy.imputer.fit(X_train)
            logger.info('Imputing X_train.')
            X_train = impute(X_train, strategy.imputer)
            logger.info('Imputing X_test.')
            X_test = impute(X_test, strategy.imputer)

        # Hyper-parameters search
        logger.info('Searching best hyper-parameters.')
        estimator = deepcopy(strategy.search)
        estimator.fit(X_train, y_train)
        Estimators.append(estimator)

        logger.info('Predicting on X_test using best fitted estimator.')
        y_pred = estimator.predict(X_test)

        if strategy.compute_importance:  # Compute feature importance
            logger.info('Computing feature importance using permutations.')
            importance = permutation_importance(estimator, X_test, y_test, **strategy.importance_params)
            dh.dump_importance(importance, fold=i)
        else:
            logger.info('Skipping feature importance.')

        dh.dump_best_params(estimator.best_params_, fold=i)
        dh.dump_prediction(y_pred, y_test, fold=i)
        dh.dump_cv_results(estimator.cv_results_, fold=i)

        if strategy.is_classification():
            logger.info('Computing y_score for ROC.')
            y_score = estimator.decision_function(X_test)
            dh.dump_roc(y_score, y_test, fold=i)

        # Learning curve
        if strategy.learning_curve:
            logger.info('Computing learning curve.')
            curve = learning_curve(estimator, X_train, y_train,
                                   cv=strategy.inner_cv, return_times=True,
                                   **strategy.learning_curve_params)
            dh.dump_learning_curve({
                'train_sizes_abs': curve[0],
                'train_scores': curve[1],
                'test_scores': curve[2],
                'fit_times': curve[3],
                'score_times': curve[4],
            }, fold=i)

    return Estimators
