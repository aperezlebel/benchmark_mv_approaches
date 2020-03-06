"""Pipeline to train model, find best parameters, give results."""

# from sklearn.experimental import enable_hist_gradient_boosting
import logging
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, cross_val_predict
from sklearn.pipeline import Pipeline

from .DumpHelper import DumpHelper


logger = logging.getLogger(__name__)


# def impute(df, imputer):
#     """Impute missing values given an already fitted imputer.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Data frame with missing values to impute
#     imputer : sklearn imputer (instance of _BaseImputer)
#         The already fitted imputer.

#     Returns
#     -------
#     pd.DataFrame
#         Data frame with imputed missing values. Extra columns might have
#         been added depending on the imputer.

#     """
#     # Columns containing only missing values are discarded by the imputer
#     discared_columns = df.columns[np.isnan(imputer.statistics_)]

#     data_imputed = imputer.transform(df)

#     # Manage the case where an indicator was used to add binary columns for MV
#     indicator = imputer.indicator_
#     # If any, get feature ids for which an indicator column has been created
#     features_with_mv = indicator.features_ if indicator is not None else []
#     # If any, create names for these extra features.
#     extra_columns = [f'indicator_{df.columns[id]}' for id in features_with_mv]

#     base_columns = [c for c in df.columns if c not in discared_columns]

#     columns = base_columns+extra_columns

#     return pd.DataFrame(data_imputed, index=df.index, columns=columns)


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
    X, y = task.X, task.y

    logger.info(f'Started task "{task.meta.tag}" '
                f'using "{strategy.name}" strategy on "{task.meta.db}".')

    dh = DumpHelper(task, strategy)  # Used to dump results

    # Create pipeline
    if strategy.imputer is not None:
        logger.info('Creating pipeline with imputer.')
        steps = [
            ('imputer', strategy.imputer),
            ('searchCV_estimator', strategy.search)
        ]
    else:
        logger.info('Creating pipeline without imputer.')
        steps = [('searchCV_estimator', strategy.search)]

    estimator = Pipeline(steps)

    # CV prediction
    logger.info(f'Started cross_val_predict with {strategy.outer_cv.n_splits} folds.')
    y_pred = cross_val_predict(estimator, X, y, cv=strategy.outer_cv, n_jobs=1, verbose=1000)
    dh.dump_prediction(y_pred, y)

    # Learning curve
    if strategy.learning_curve:
        logger.info(f'Learning curve: starting with {strategy.outer_cv.n_splits} folds.')
        curve = learning_curve(estimator, X, y,
                               cv=strategy.outer_cv, return_times=True,
                               **strategy.learning_curve_params)
        dh.dump_learning_curve({
            'train_sizes_abs': curve[0],
            'train_scores': curve[1],
            'test_scores': curve[2],
            'fit_times': curve[3],
            'score_times': curve[4],
        }, fold=None)
    else:
        logger.info('Learning curve: skipping.')


    # ROC curve
    if strategy.is_classification():
        logger.info('ROC: computing curve.')
        y_score = cross_val_predict(estimator, X, y, cv=strategy.outer_cv,
                                    n_jobs=1, method='decision_function',
                                    verbose=1000)
        # probas = cross_val_predict(estimator, X, y, cv=strategy.outer_cv, method='predict_proba')
        # print(probas)

        # y_score = estimator.decision_function(X_test)
        # dh.dump_probas(probas[:, 1], y, fold=None)
        dh.dump_roc(y_score, y, fold=None)
    else:
        logger.info('ROC: not a classification, skipping.')

    # # Feature importance
    # if strategy.compute_importance:
    #     logger.info(f'Feature importance: starting with {strategy.outer_cv.n_splits} folds.')
    #     for i, (train_index, test_index) in enumerate(strategy.outer_cv.split(X)):
    #         logger.info(f'Started fold {i}.')

    #         X_test = X.iloc[test_index]
    #         y_test = y.iloc[test_index]

    #         importance = permutation_importance(estimator, X_test, y_test, **strategy.importance_params)
    #         dh.dump_importance(importance, fold=i)
    # else:
    #     logger.info('Feature importance: skipping.')

