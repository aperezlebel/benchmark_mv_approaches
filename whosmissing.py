import pandas as pd
import numpy as np

from prediction.PlotHelperV4 import PlotHelperV4


df = pd.read_csv('scores/scores.csv', index_col=0)


methods =[
    'MIA',
    'Mean',
    'Mean+mask',
    'Med',
    'Med+mask',
    'Iter',
    'Iter+mask',
    'KNN',
    'KNN+mask',
    'Linear+Iter',
    'Linear+Iter+mask',
]

translate_methods = {
    'MIA': 0,
    'Mean': 3,
    'Mean+mask': 6,
    'Med': 9,
    'Med+mask': 12,
    'Iter': 15,
    'Iter+mask': 18,
    'KNN': 27,
    'KNN+mask': 30,
}


def missing_pt_imputation(df):
    idx = df.loc[df['tuning_PT'].isna(), 'tuning_PT'].index

    df = df.loc[idx]

    print(df)

    dfgb = df.groupby(by=['db', 'task', 'method'])
    df = dfgb.agg({'imputation_PT': 'first'})
    df = df.reset_index()

    print(df)

    for id, subdf in df.groupby(by=['db', 'task']):
        methods = list(subdf['method'])
        t_methods = [translate_methods[m] for m in methods]
        t_methods.sort()
        db = id[0]
        task = id[1]
        print(f'{db}/{task}: {methods}')


def missing_scores(df, expected_methods):
    df = PlotHelperV4.aggregate(df, 'tuning_PT')

    for id, subdf in df.groupby(by=['size', 'db', 'task']):
        methods = list(subdf['method'])
        size = id[0]
        db = id[1]
        task = id[2]
        for m in expected_methods:
            if m not in methods:
                print(f'{size}/{db}/{task}: {m} missing')
            else:
                n_t = subdf.loc[subdf['method'] == m, 'n_trials']
                assert len(n_t) == 1
                n_t = n_t.iloc[0]
                if n_t != 5:
                    print(f'{size}/{db}/{task}: {m} missing trial: {n_t}/5')

                if pd.isnull(subdf.loc[subdf['method'] == m, 'tuning_PT'].iloc[0]):
                    print(subdf)


if __name__ == '__main__':
    missing_scores(df, expected_methods=methods)
