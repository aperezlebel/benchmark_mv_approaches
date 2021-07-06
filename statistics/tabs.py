"""Create tabs of the article"""
import os
from os.path import join
import pandas as pd

from prediction.PlotHelper import PlotHelper
from .tests import tasks_to_drop, db_rename, db_order
from custom.const import get_tab_folder


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
