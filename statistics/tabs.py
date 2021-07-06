"""Create tabs of the article"""
import os
from os.path import join
import pandas as pd

from prediction.PlotHelper import PlotHelper
from .tests import tasks_to_drop, db_rename, db_order
from prediction.df_utils import get_scores_tab, get_ranks_tab
from custom.const import get_tab_folder


def run_scores(graphics_folder, linear, csv=False):
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        df.drop(index=df[(df['db'] == db) & (df['task'] == task)].index, inplace=True)

    df['task'] = df['task'].str.replace('_pvals', '_screening')

    if linear:
        method_order = [
            'MIA',
            'Linear+Mean',
            'Linear+Mean+mask',
            'Linear+Med',
            'Linear+Med+mask',
            'Linear+Iter',
            'Linear+Iter+mask',
            'Linear+KNN',
            'Linear+KNN+mask',
        ]

    else:
        method_order = [
            'MIA',
            'Mean',
            'Mean+mask',
            'Med',
            'Med+mask',
            'Iter',
            'Iter+mask',
            'KNN',
            'KNN+mask',
        ]

    db_order = [
        'TB',
        'UKBB',
        'MIMIC',
        'NHIS',
    ]

    scores = get_scores_tab(df, method_order=method_order, db_order=db_order, relative=True)
    ranks = get_ranks_tab(df, method_order=method_order, db_order=db_order)

    rename = {
        'Med': 'Median',
        'Med+mask': 'Median+mask',
        'Iter': 'Iterative',
        'Iter+mask': 'Iterative+mask',
    }

    if linear:
        rename['MIA'] = 'Boosted trees+MIA'
    scores.rename(rename, axis=0, inplace=True)
    ranks.rename(rename, axis=0, inplace=True)

    # Rename DBs
    scores.rename(db_rename, axis=1, inplace=True)
    ranks.rename(db_rename, axis=1, inplace=True)

    print(scores)
    print(ranks)

    # Turn bold the best ranks
    best_ranks_by_task = ranks.loc['Average'].astype(float).idxmin(axis=0, skipna=True)
    for (db, task), best_method in best_ranks_by_task.iteritems():
        ranks.loc[('Average', best_method), (db, task)] = f"\\textbf{{{ranks.loc[('Average', best_method), (db, task)]}}}"

    best_ranks_by_size = ranks['Average'].drop('Average', axis=0, level=0).astype(float).groupby(['Size']).idxmin(axis=0, skipna=True)
    for db in best_ranks_by_size.columns:
        for size, value in best_ranks_by_size[db].iteritems():
            if pd.isnull(value):
                continue
            best_method = value[1]
            ranks.loc[(size, best_method), ('Average', db)] = f"\\textbf{{{ranks.loc[(size, best_method), ('Average', db)]}}}"

    # Preprocessing for latex dump
    tasks = scores.columns.get_level_values(1)
    rename = {k: k.replace("_", r"\_") for k in tasks}
    rename = {k: f'\\rot{{{v}}}' for k, v in rename.items()}

    scores.rename(columns=rename, inplace=True)
    ranks.rename(columns=rename, inplace=True)

    # Rotate dbs on ranks
    ranks.rename({v: f'\\rot{{{v}}}' for v in ranks['Average'].columns}, axis=1, level=1, inplace=True)

    # Turn bold the reference scores
    def boldify(x):
        if pd.isnull(x):
            return ''
        else:
            return f'\\textbf{{{x}}}'

    smallskip = '0.15in'
    bigskip = '0.3in'
    medskip = '0.23in'
    index_rename = {}

    for size in [2500, 10000, 25000, 100000, 'Average']:
        scores.loc[(size, 'Reference score')] = scores.loc[(size, 'Reference score')].apply(boldify)
        if size == 2500:
            continue
        skip = bigskip if size == 'Average' else smallskip
        index_rename[size] = f'\\rule{{0pt}}{{{skip}}}{size}'

    scores.rename(index_rename, axis=0, level=0, inplace=True)
    ranks.rename(index_rename, axis=0, level=0, inplace=True)

    n_latex_columns = len(ranks.columns)+2
    column_format = 'l'*(n_latex_columns-5)+f'@{{\\hskip {smallskip}}}'+'l'*4+f'@{{\\hskip {medskip}}}'+'l'

    tab_folder = get_tab_folder(graphics_folder)
    tab1_name = 'scores_linear' if linear else 'scores'
    tab2_name = 'ranks_linear' if linear else 'ranks'

    scores.to_latex(join(tab_folder, f'{tab1_name}.tex'), na_rep='', escape=False, table_env='tabularx') #, column_format='L'*scores.shape[1])
    ranks.to_latex(join(tab_folder, f'{tab2_name}.tex'), na_rep='', escape=False,
    table_env='tabularx', column_format=column_format)

    if csv:
        scores.to_csv(join(tab_folder, f'{tab1_name}.csv'))
        ranks.to_csv(join(tab_folder, f'{tab2_name}.csv'))


def run_desc(graphics_folder):
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        df.drop(index=df[(df['db'] == db) & (df['task'] == task)].index, inplace=True)

    df = PlotHelper.get_task_description(df)

    time_columns = {
        'Imputation time (s)': 'Time - Imputation (s)',
        'Tuning time (s)': 'Time - Tuning (s)',
        'Total time (s)': 'Time - Total (s)',
    }

    df.rename(columns=time_columns, inplace=True)

    for db, task in tasks_to_drop.items():
        df = df.drop((db, task), axis=0)

    df = df.reset_index()
    df['Task'] = df['Task'].str.replace('_pvals', '_screening')
    df = df.set_index(['Database', 'Task'])

    df = df.reindex(db_order, level=0, axis=0)

    tab_folder = get_tab_folder(graphics_folder)

    # Rename DBs
    df.rename(db_rename, axis=0, inplace=True)
    # Rotate Traumabase
    df.rename({'Traumabase': '\\rotsmash{Traumabase}'}, axis=0, inplace=True)

    df.rename({'n': 'Number of samples', 'p': 'Number of features'}, axis=1, inplace=True)

    # Custom escape content of table
    df['Target'] = df['Target'].map(lambda x: str(x).replace('_', r'\_'))

    # Round floats to int
    def to_int(x):  # Convert to int and robust to NaN
        try:
            f = float(x)
            if f < 1:
                return f'{f:.2f}'
            return str(int(x))
        except:
            return x

    df = df.applymap(to_int)

    # Preprocessing for latex dump
    tasks = df.index.get_level_values(1)
    rename = {k: k.replace("_", r"\_") for k in tasks}
    df.rename(index=rename, inplace=True)

    rename = {v: f'\\rot{{{v}}}' for v in df.columns if v not in ['Target', 'Description']}
    df.rename(columns=rename, level=0, inplace=True)

    skip = '0.3in'
    index_rename = {}
    for i, v in enumerate(pd.unique(df.index.get_level_values(0))):
        if i == 0:
            continue
        index_rename[v] = f'\\rule{{0pt}}{{{skip}}}{v}'

    df.rename(index_rename, axis=0, level=0, inplace=True)

    n = len(df.columns)
    column_format = 'l'*df.index.nlevels+'l'*(n-2)+'X'*2

    with pd.option_context("max_colwidth", None):
        df.to_latex(join(tab_folder, 'task_description.tex'),
                    table_env='tabularx',
                    bold_rows=False, na_rep=None, escape=False,
                    column_format=column_format,
                    multirow=False)
