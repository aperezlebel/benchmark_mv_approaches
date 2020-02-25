"""Pipeline to train model, find best parameters, give results."""

# from sklearn.experimental import enable_hist_gradient_boosting
import pandas as pd
import numpy as np
from copy import deepcopy

from .DumpHelper import DumpHelper


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
    dh = DumpHelper(task, strategy)  # Used to dump results

    X, y = task.X, task.y

    # Non nested CV
    # X_train, X_test, y_train, y_test = strategy.split(X, y)

    # Nested CV
    Estimators = []

    for i, (train_index, test_index) in enumerate(strategy.outer_cv.split(X)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Imputation
        if strategy.imputer is not None:
            strategy.imputer.fit(X_train)
            X_train = impute(X_train, strategy.imputer)
            X_test = impute(X_test, strategy.imputer)

        # Hyper-parameters search
        estimator = deepcopy(strategy.search)
        estimator.fit(X_train, y_train)
        Estimators.append(estimator)

        y_pred = estimator.predict(X_test)

        dh.dump_best_params(estimator.best_params_, fold=i)
        dh.dump_prediction(y_pred, y_test, fold=i)
        dh.dump_cv_results(estimator.cv_results_, fold=i)

        if strategy.is_classification():
            y_score = estimator.decision_function(X_test)
            dh.dump_roc(y_score, y_test, fold=i)

    return Estimators
