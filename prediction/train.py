"""Pipeline to train model, find best parameters, give results."""

from sklearn.metrics import classification_report, roc_auc_score


def train(task, strategy):
    """Train a model following a strategy on prediction task.

    Parameters:
    -----------
    task : PredictionTask object
        Describe the prediction task.
    strategy : Strategy object
        Describe the estimator and the strategy to train and find the best
        parameters.

    Returns:
    --------
    dict
        Stores the results of the training.

    """
    X, y = task.X, task.y
    X_train, X_test, y_train, y_test = strategy.split(X, y)
    estimator = strategy.search.fit(X_train, y_train)

    y_true, y_pred = y_test, estimator.predict(X_test)
    print(classification_report(y_true, y_pred))

    print(f'Best parameters:\n{estimator.best_params_}\n')
    print(estimator.cv_results_)

    y_score = estimator.decision_function(X_test)
    print(f'AUC score:\n{roc_auc_score(y_test, y_score)}')

    return estimator
