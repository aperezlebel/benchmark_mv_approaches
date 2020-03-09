
import logging
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from prediction.tasks import tasks


logger = logging.getLogger(__name__)


def run(argv=None):
    task = tasks['UKBB/fluid_intelligence_light']

    logger.info('Getting X.')
    X = task.X
    logger.info('Getting y.')
    y = task.y

    # Simulate the outer CV (the one of KFold)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Simulate the inner CV (the one of RandomSearchCV)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2)

    # Now X has the same shape as in real experiment
    logger.info(f'X shape: {X_train2.shape}')

    estimator = HistGradientBoostingRegressor(
        loss='least_absolute_deviation',
        learning_rate=1e-5,
        max_depth=11
    )

    logger.info('Fitting estimator.')
    estimator.fit(X_train2, y_train2)
    logger.info('Estimator fitted.')




