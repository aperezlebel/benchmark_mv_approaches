"""Init strategies to be used for running jobs."""
import logging
import os
from copy import deepcopy

import numpy as np
import yaml
from sklearn.experimental import (enable_hist_gradient_boosting,
                                  enable_iterative_imputer)
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import (GridSearchCV, KFold, ShuffleSplit,
                                     StratifiedShuffleSplit)

from .strategy import Strategy

logger = logging.getLogger(__name__)

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
param_space = params.get('param_space', None)
train_set_steps = params.get('train_set_steps', [])
min_test_set = params.get('min_test_set', 0.2)
n_splits = params.get('n_splits', 5)

# Default RS
RS = 42

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
logger.info(f'param_space: {param_space}')
logger.info(f'RS: {RS}')
logger.info(f'train_set_steps: {train_set_steps}')
logger.info(f'min_test_set: {min_test_set}')

if param_space is None:
    param_space = {
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [3, 6, 9]
    }

# A strategy to run a classification
strategies.append(Strategy(
    name='Classification',
    estimator=HistGradientBoostingClassifier(random_state=RS),
    inner_cv=StratifiedShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS),
    search=GridSearchCV,
    param_space=param_space,
    search_params={
        'scoring': 'roc_auc_ovr_weighted',
        'verbose': 0,
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
    #     'verbose': 0,
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
    roc=roc,
    train_set_steps=train_set_steps,
    min_test_set=min_test_set,
    n_splits=n_splits,
))

strategies.append(Strategy(
    name='Classification_Logit',
    estimator=LogisticRegressionCV(random_state=RS, cv=StratifiedShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS),),
    inner_cv=None,
    search=None,
    param_space=None,
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
    roc=roc,
    train_set_steps=train_set_steps,
    min_test_set=min_test_set,
    n_splits=n_splits,
))


# A strategy to run a regression
strategies.append(Strategy(
    name='Regression',
    estimator=HistGradientBoostingRegressor(loss='least_absolute_deviation', random_state=RS),
    inner_cv=ShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS),
    search=GridSearchCV,
    param_space=param_space,
    search_params={
        'scoring': ['r2', 'neg_mean_absolute_error'],
        'refit': 'r2',
        'verbose': 0,
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
    #     'verbose': 0,
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
    roc=roc,
    train_set_steps=train_set_steps,
    min_test_set=min_test_set,
    n_splits=n_splits,
))

strategies.append(Strategy(
    name='Regression_Ridge',
    estimator=RidgeCV(cv=ShuffleSplit(n_splits=n_inner_splits, train_size=0.8, random_state=RS)),
    inner_cv=None,
    search=None,
    param_space=None,
    search_params=None,
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
    roc=roc,
    train_set_steps=train_set_steps,
    min_test_set=min_test_set,
    n_splits=n_splits,
))


# Add imputation to the previous strategies
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'Mean+mask': SimpleImputer(strategy='mean', add_indicator=True),
    'Med': SimpleImputer(strategy='median'),
    'Med+mask': SimpleImputer(strategy='median', add_indicator=True),
    'Iterative': IterativeImputer(max_iter=iterative_imputer_max_iter,
                                  random_state=RS),
    'Iterative+mask': IterativeImputer(add_indicator=True,
                                       max_iter=iterative_imputer_max_iter,
                                       random_state=RS),
    'KNN': KNNImputer(),
    'KNN+mask': KNNImputer(add_indicator=True),

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

