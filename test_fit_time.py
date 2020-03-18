
import logging
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, \
    HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from prediction.tasks import tasks


logger = logging.getLogger(__name__)


def run(argv=None):
    if argv is None or len(argv) < 2:
        logger.info('No argv given.')
        task = tasks['TB/shock_hemo']
        clf = 'HGBC'
    else:
        task = tasks[argv[1]]
        clf = argv[2]
        logger.info(f'Argv given. Task {task.meta.tag}. Clf {clf}.')

    logger.info('Getting X.')
    X = task.X
    logger.info('Getting y.')
    y = task.y

    logger.info(f'X shape before splits: {X.shape}')

    # Simulate the outer CV (the one of KFold)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    # Simulate the inner CV (the one of RandomSearchCV)
    X_train2, _, y_train2, _ = train_test_split(X_train, y_train, test_size=0.2)

    # Now X has the same shape as in real experiment
    logger.info(f'X shape: {X_train2.shape}')

    for learning_rate in [0.05, 0.1, 0.3]:
        for max_depth in [3, 6, 9]:
            if clf == 'HGBC':
                estimator = HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )
            elif clf == 'HGBR':
                estimator = HistGradientBoostingRegressor(
                    loss='least_absolute_deviation',
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )

            logger.info(f'Params: LR {learning_rate} MD {max_depth}')
            logger.info('Fitting estimator.')
            estimator.fit(X_train2, y_train2)
            logger.info('Estimator fitted.')




