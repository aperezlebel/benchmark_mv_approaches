"""Init strategies to be used for running jobs."""
import yaml
import os
import numpy as np
import logging
from sklearn.model_selection import ShuffleSplit, GridSearchCV, \
    RandomizedSearchCV, KFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, \
    HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from copy import deepcopy

from scipy.stats import uniform
from sklearn.utils.fixes import loguniform

from .strategy import Strategy


logger = logging.getLogger(__name__)

RS = 42
strategies = list()

# Load some params from custom file
filepath = 'custom/strategy_params.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
else:
    params = dict()

# Load or defaults
n_outer_splits = params.get('n_outer_splits', 2)
n_inner_splits = params.get('n_inner_splits', 2)
n_jobs = params.get('n_jobs', 1)
n_iter = params.get('n_iter', 1)
n_repeats = params.get('n_repeats', 1)
compute_importance = params.get('compute_importance', False)
learning_curve = params.get('learning_curve', False)
n_learning_trains = params.get('n_learning_trains', 5)
iterative_imputer_max_iter = params.get('iterative_imputer_max_iter', 10)
roc = params.get('roc', False)

logger.info(f'Loaded strategy_params.yml with following parameters:')
logger.info(f'n_outer_splits: {n_outer_splits}')
logger.info(f'n_inner_splits: {n_inner_splits}')
logger.info(f'n_jobs: {n_jobs}')
logger.info(f'n_iter: {n_iter}')
logger.info(f'n_repeats: {n_repeats}')
logger.info(f'compute_importance: {compute_importance}')
logger.info(f'learning_curve: {learning_curve}')
logger.info(f'n_learning_trains: {n_learning_trains}')
logger.info(f'iterative_imputer_max_iter: {iterative_imputer_max_iter}')
logger.info(f'roc: {roc}')

# A strategy to run a classification
strategies.append(Strategy(
    name='Classification',
    estimator=HistGradientBoostingClassifier(),
    inner_cv=ShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS),
    search=GridSearchCV,
    param_space={
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [3, 6, 9]
    },
    search_params={
        'scoring': 'roc_auc_ovr_weighted',
        'verbose': 1000,
        'n_jobs': n_jobs,
        'return_train_score': True,
    },
    # search=RandomizedSearchCV,
    # param_space={
    #     'learning_rate': uniform(1e-5, 1),
    #     'max_iter': range(10, 500)
    # },
    # search_params={
    #     'scoring': 'recall',
    #     'verbose': 1000,
    #     'n_jobs': n_jobs,
    #     'return_train_score': True,
    #     'n_iter': n_iter
    # },
    outer_cv=KFold(n_splits=n_outer_splits, shuffle=True, random_state=RS),
    compute_importance=compute_importance,
    importance_params={
        'n_jobs': n_jobs,
        'n_repeats': n_repeats,
    },
    learning_curve=learning_curve,
    learning_curve_params={
        'scoring': 'roc_auc_ovr_weighted',
        'train_sizes': np.linspace(0.1, 1, n_learning_trains)
    },
    roc=roc
))


strategies.append(Strategy(
    name='Classification_RFC',
    estimator=RandomForestClassifier(n_jobs=1),
    inner_cv=ShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS),
    search=GridSearchCV,
    param_space={
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 9]
    },
    search_params={
        'scoring': 'roc_auc_ovr_weighted',
        'verbose': 1000,
        'n_jobs': n_jobs,
        'return_train_score': True,
    },
    # search=RandomizedSearchCV,
    # param_space={
    #     'learning_rate': uniform(1e-5, 1),
    #     'max_iter': range(10, 500)
    # },
    # search_params={
    #     'scoring': 'recall',
    #     'verbose': 1000,
    #     'n_jobs': n_jobs,
    #     'return_train_score': True,
    #     'n_iter': n_iter
    # },
    outer_cv=KFold(n_splits=n_outer_splits, shuffle=True, random_state=RS),
    compute_importance=compute_importance,
    importance_params={
        'n_jobs': n_jobs,
        'n_repeats': n_repeats,
    },
    learning_curve=learning_curve,
    learning_curve_params={
        'scoring': 'roc_auc_ovr_weighted',
        'train_sizes': np.linspace(0.1, 1, n_learning_trains)
    },
    roc=roc
))

# A strategy to run a regression
strategies.append(Strategy(
    name='Regression',
    estimator=HistGradientBoostingRegressor(loss='least_absolute_deviation'),
    inner_cv=ShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS),
    search=GridSearchCV,
    param_space={
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [3, 6, 9]
    },
    search_params={
        'scoring': ['r2', 'neg_mean_absolute_error'],
        'refit': 'r2',
        'verbose': 1000,
        'n_jobs': n_jobs,
        'return_train_score': True,
    },
    # search=RandomizedSearchCV,
    # param_space={
    #     'learning_rate': uniform(1e-5, 1),
    #     'max_depth': range(3, 11)
    # },
    # search_params={
    #     'scoring': ['r2', 'neg_mean_absolute_error'],
    #     'refit': 'r2',
    #     'verbose': 1000,
    #     'return_train_score': True,
    #     'n_iter': n_iter,
    #     'n_jobs': n_jobs
    # },
    outer_cv=KFold(n_splits=n_outer_splits, shuffle=True, random_state=RS),
    compute_importance=compute_importance,
    importance_params={
        'n_jobs': n_jobs,
        'n_repeats': n_repeats,
    },
    learning_curve=learning_curve,
    learning_curve_params={
        'scoring': 'r2',
        'train_sizes': np.linspace(0.1, 1, n_learning_trains)
    },
    roc=roc
))


# Add imputation to the previous strategies
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'Mean+mask': SimpleImputer(strategy='mean', add_indicator=True),
    'Med': SimpleImputer(strategy='median'),
    'Med+mask': SimpleImputer(strategy='median', add_indicator=True),
    'Iterative': IterativeImputer(max_iter=iterative_imputer_max_iter),
    'Iterative+mask': IterativeImputer(add_indicator=True,
                                       max_iter=iterative_imputer_max_iter),
}

# Add imputed versions of the previosu strategies
imputed_strategies = list()

for imputer_name, imputer in imputers.items():
    for strategy in strategies:
        strategy = deepcopy(strategy)
        strategy.imputer = imputer
        strategy.name = f'{strategy.name}_imputed_{imputer_name}'
        imputed_strategies.append(strategy)

strategies = strategies + imputed_strategies
strategies = {strategy.name: strategy for strategy in strategies}

