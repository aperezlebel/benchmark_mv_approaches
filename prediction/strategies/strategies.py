"""Init strategies to be used for running jobs."""
import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, \
    HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from copy import deepcopy

from .strategy import Strategy


RS = 42
strategies = list()

# A strategy to run a classification
strategies.append(Strategy(
    name='Classification',
    estimator=HistGradientBoostingClassifier(),
    inner_cv=ShuffleSplit(n_splits=2, train_size=0.8, random_state=RS),
    param_space={
        'learning_rate': [0.01],#np.linspace(0.01, 0.15, 3),
        'max_iter': [100]#[100, 500, 1000]
    },
    search=GridSearchCV,
    search_params={
        'scoring': 'recall',
        'verbose': 2,
        'n_jobs': -1,
        'return_train_score': True
    },
    outer_cv=KFold(n_splits=2, shuffle=True, random_state=RS),
    compute_importance=True,
    importance_params={
        'n_jobs': -1,
        'n_repeats': 1,
    },
    learning_curve=True,
    learning_curve_params={
        'scoring': 'roc_auc_ovr_weighted',
        'n_jobs': -1
    }
))

# A strategy to run a regression
strategies.append(Strategy(
    name='Regression',
    estimator=HistGradientBoostingRegressor(loss='least_absolute_deviation'),
    inner_cv=ShuffleSplit(n_splits=2, train_size=0.8, random_state=RS),
    param_space={
        'learning_rate': np.array([0.1]),#np.linspace(0.001, 0.1, 5),#[0.1, 0.15, 0.2, 0.25],
        'max_depth': [3]#[3, 6, 8]
    },
    search=GridSearchCV,
    search_params={
        'scoring': ['r2', 'neg_mean_absolute_error'],
        'refit': 'r2',
        'verbose': 2,
        'return_train_score': True
    },
    outer_cv=KFold(n_splits=2, shuffle=True, random_state=RS),
    compute_importance=True,
    importance_params={
        'n_jobs': -1,
        'n_repeats': 1,
    },
    learning_curve=True,
    learning_curve_params={
        'scoring': 'r2',
        'n_jobs': -1
    }
))


# Add imputation to the previous strategies
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'Mean+mask': SimpleImputer(strategy='mean', add_indicator=True),
    'Med': SimpleImputer(strategy='median'),
    'Med+mask': SimpleImputer(strategy='median', add_indicator=True),
    'Iterative': IterativeImputer(),
    'Iterative+mask': IterativeImputer(add_indicator=True),
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

