
import logging
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from prediction.tasks import tasks


logger = logging.getLogger(__name__)


def run():
    task = tasks['UKBB/fluid_intelligence_light']

    logger.info('Getting X.')
    X = task.X
    logger.info('Getting y.')
    y = task.y

    logger.info(f'X shape: {X.shape}')

    estimator = HistGradientBoostingRegressor(
        loss='least_absolute_deviation',
        learning_rate=1e-5,
        max_depth=11
    )

    logger.info('Fitting estimator.')
    estimator.fit(X, y)
    logger.info('Estimator fitted.')




