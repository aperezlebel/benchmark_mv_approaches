import os
import re
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from custom.const import get_fig_folder

db_order = [
    'TB',
    'UKBB',
    'MIMIC',
    'NHIS',
]


def run_feature_importance(graphics_folder, results_folder, n, average_folds=True):
    filenames = [
        f'{n}_importances.csv',
        f'{n}_mv_props.csv',
    ]

    dfs = []

    # Aggregate feature importances of all tasks
    for root, subdirs, files in os.walk(results_folder):
        print(root)

        res = re.search(join(results_folder, '/(.*)/RS'), root)

        if res is None:
            continue

        for filename in filenames:

            if filename not in files:
                raise OSError(f'{filename} not found at path {root}')
        task = res.group(1)
        db = task.split('/')[0]

        importance = pd.read_csv(join(root, f'{n}_importances.csv'), index_col=0)
        mv_props = pd.read_csv(join(root, f'{n}_mv_props.csv'), index_col=0)
        mv_props.set_index('fold', inplace=True)

        importance.reset_index(inplace=True)
        importance.set_index(['fold', 'repeat'], inplace=True)
        importance_avg = importance.groupby(level='fold').mean()

        if average_folds:
            importance_avg = importance_avg.mean()
            importance_avg = importance_avg.to_frame().T
            mv_props = mv_props.mean()
            mv_props = mv_props.to_frame().T
            id_vars = None
            index = ['feature']

        else:
            importance_avg.reset_index(inplace=True)
            mv_props.reset_index(inplace=True)
            id_vars = ['fold']
            index = ['fold', 'feature']

        importance_avg = pd.melt(importance_avg, id_vars=id_vars,
                                 var_name='feature', value_name='importance')
        importance_avg.set_index(index, inplace=True)

        mv_props = pd.melt(mv_props, id_vars=id_vars, var_name='feature', value_name='mv_prop')
        mv_props.set_index(index, inplace=True)

        df = pd.concat([importance_avg, mv_props], axis=1)
        assert not pd.isna(df).any().any()

        df['db'] = db
        df = pd.concat({task: df}, names=['task'])

        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    print(df)

    fig = plt.figure()
    ax = plt.gca()
    sns.set_palette(sns.color_palette('colorblind'))
    sns.scatterplot(x='mv_prop', y='importance', hue='db', data=df, ax=ax, s=15, hue_order=db_order)
    ax.set_xlabel('Proportion of missing values in features')
    ax.set_ylabel('Feature importance (score drop)')
    ax.legend(title='Database')

    fig_name = 'importance_avg' if average_folds else 'importance'
    fig_folder = get_fig_folder(graphics_folder)
    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig.savefig(join(fig_folder, f'{fig_name}.jpg'), bbox_inches='tight')
