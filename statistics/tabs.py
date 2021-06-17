"""Create tabs of the article"""
import os
from os.path import join
import pandas as pd

from prediction.PlotHelper import PlotHelper
from .tests import tasks_to_drop, db_rename, db_order
from custom.const import get_tab_folder


def run_desc(graphics_folder):
    path = os.path.abspath('scores/scores.csv')
    scores = pd.read_csv(path, index_col=0)

    # Drop tasks
    scores = scores.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        # print(scores.shape)
        scores = scores.drop((db, task), axis=0)
    scores = scores.reset_index()
    # exit()

    df = PlotHelper.get_task_description(scores)

    print(df)
    exit()

    print(list(df.columns))
    time_columns = {
        'Imputation time (s)': 'Time - Imputation (s)',
        'Tuning time (s)': 'Time - Tuning (s)',
        'Total time (s)': 'TIme - Total (s)',
    }

    df.rename(columns=time_columns, inplace=True)

    # time_columns = {
    #     'Imputation time (s)': '\\rot{{Imputation}}',
    #     'Tuning time (s)': '\\rot{{Tuning}}',
    #     'Total time (s)': '\\rot{{Total}}',
    # }
    # # df_time = df[time_columns.keys()]
    # # df_time.columns = pd.MultiIndex.from_product([['Time'], df_time.columns])
    # # df_time.rename(time_columns, axis=1, inplace=True)
    # # print(df_time)

    # empty_char = '\\hphantom{{ }}'
    # df.columns = pd.MultiIndex.from_product([df.columns, [empty_char]])
    # # print(df)

    # renamed_cols = []
    # for lvl1, lvl2 in df.columns:
    #     if lvl1 in time_columns.keys():
    #         lvl1_out = 'Time'
    #         lvl2_out = time_columns[lvl1]

    #     else:
    #         lvl1_out = lvl1
    #         lvl2_out = lvl2

    #     print(lvl1, lvl2, lvl1_out, lvl2_out)
    #     renamed_cols.append((lvl1_out, lvl2_out))

    # print(pd.MultiIndex.from_tuples(renamed_cols))

    # df.columns = pd.MultiIndex.from_tuples(renamed_cols)

    # print(df)

    # df['Time'] = df_time

    # print(df)

    # exit()

    # df = df.set_index(['Database', 'Task'])
    print(df.shape)
    for db, task in tasks_to_drop.items():
        df = df.drop((db, task), axis=0)
        print(df.shape)
    exit()
    df = df.reset_index()
    df = df.set_index(['Database', 'Task'])

    df = df.reindex(db_order, level=0, axis=0)

    tab_folder = get_tab_folder(graphics_folder)

    # Rename DBs
    df.rename(db_rename, axis=0, inplace=True)
    # Rotate Traumabase
    df.rename({'Traumabase': '\\rotsmash{Traumabase}'}, axis=0, inplace=True)

    # Custom escape content of table
    df['Target'] = df['Target'].map(lambda x: str(x).replace('_', r'\_'))
    # df[('Target', empty_char)] = df[('Target', empty_char)].map(lambda x: str(x).replace('_', r'\_'))

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
    # rename = {k: f'\\rot{{{v}}}' for k, v in rename.items()}
    # df.rename(columns=rename, inplace=True)

    # rename = {v: "\\makecell{"+'\\\\'.join(v)+"}" for v in df.index.get_level_values(0)}
    # rename = {v: f'\\rotatebox{{-90}}{{{v}}}' for v in df.index.get_level_values(0)}
    # df.rename(index=rename, level=0, inplace=True)

    rename = {v: f'\\rot{{{v}}}' for v in df.columns if v not in ['Target', 'Description']}
    # rename = {v: f'\\rot{{{v}}}' for v in ['Score', 'Scorer', 'Selection']}
    df.rename(columns=rename, level=0, inplace=True)
    # df = df.transpose()
    print(df)

    n = len(df.columns)
    column_format = 'l'*df.index.nlevels+'l'*(n-2)+'X'*2
    # df.to_csv(join(tab_folder, 'task_description.csv'))
    with pd.option_context("max_colwidth", None):
        df.to_latex(join(tab_folder, 'task_description.tex'), table_env='tabularx',
        bold_rows=False, na_rep=None, escape=False, column_format=column_format,
        multirow=False)

    print(df)