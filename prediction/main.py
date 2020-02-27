"""Set the strategies and run the predicitons."""
import numpy as np
from copy import deepcopy
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, \
    HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

import database
from .tasks import Task, UKBB_tasks_meta, TB_tasks_meta
from .strategy import Strategy
from .train import train


def main():
    dbs = {
        'TB': database.TB(),
        'UKBB': database.UKBB()
    }

    RS = 42

    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'Mean+mask': SimpleImputer(strategy='mean', add_indicator=True),
        'Med': SimpleImputer(strategy='median'),
        'Med+mask': SimpleImputer(strategy='median', add_indicator=True),
        'Iterative': IterativeImputer(),
        'Iterative+mask': IterativeImputer(add_indicator=True),
    }

    strategies = dict()

    strategies['Classification'] = Strategy(
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
    )


    # strat_imp = deepcopy(strategies[''])
    # strat_imp.imputer = imputers['Mean']
    # strat_imp.name = strat.name+'_imputed'

    strategies['Regression'] = Strategy(
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
    )

    imputed_strategies = dict()

    for name, strat in strategies.items():
        strat_imp = deepcopy(strat)
        strat_imp.imputer = imputers['Mean']
        strat_imp.name = strat.name+'_imputed'
        imputed_strategies[name] = strat_imp

    # meta = TB_tasks_meta['death']
    # TB.load(meta.df_name)
    # task = Task(TB.encoded_dataframes[meta.df_name], meta)

    # _ = train(task, strat_imp)

    to_run = [
        (TB_tasks_meta['death'], strategies['Classification']),
        (TB_tasks_meta['death'], imputed_strategies['Classification'])
    ]

    for meta, strategy in to_run:
        db = dbs[meta.db]
        df_name = meta.df_name
        db.load(df_name)
        df = db.encoded_dataframes[df_name]
        task = Task(df, meta)

        _ = train(task, strategy)



    # meta2 = UKBB_tasks_meta['fluid_intelligence']
    # UKBB.load(meta2.df_name)
    # task2 = Task(UKBB.encoded_dataframes[meta2.df_name], meta2)
    # # _ = train(task2, strat2_imp)

    # _ = train(task2, strat2)

    # plt.show()
