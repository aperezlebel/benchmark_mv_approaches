"""Run some statistical tests on the results."""
import os
import pandas as pd
from scipy.stats import wilcoxon

from prediction.PlotHelper import PlotHelper


def run_wilcoxon():
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Agregate accross trials by averaging
    df = df.reset_index()
    df['n_trials'] = 1  # Add a count column to keep track of # of trials
    dfgb = df.groupby(['size', 'db', 'task', 'method', 'fold'])
    df = dfgb.agg({
        'score': 'mean',
        'n_trials': 'sum',
        'scorer': PlotHelper.assert_equal,  # first and assert equal
        'selection': PlotHelper.assert_equal,
        'n': PlotHelper.assert_equal,
        'p': 'mean',  #PlotHelper.assert_equal,
        'type': PlotHelper.assert_equal,
        'imputation_WCT': 'mean',
        'tuning_WCT': 'mean',
        'imputation_PT': 'mean',
        'tuning_PT': 'mean',
    })

    # Reset index to addlevel of the multi index to the columns of the df
    df = df.reset_index()
    df = df.set_index(['size', 'db', 'task', 'method', 'fold'])

    MIA = df.iloc[df.index.get_level_values('method') == 'MIA']
    MIA.reset_index(level='method', drop=True, inplace=True)
    MIA_scores = MIA['score']
    MIA_index = MIA.index

    methods = df.index.get_level_values('method').unique()
    methods = [m for m in methods if m != 'MIA']

    rows = []
    for method in methods:
        m = df.iloc[df.index.get_level_values('method') == method]
        m.reset_index(level='method', drop=True, inplace=True)
        m_scores = m['score']
        ref_scores = MIA_scores.loc[m_scores.index]
        w_double = wilcoxon(x=ref_scores, y=m_scores, alternative='two-sided')
        w_greater = wilcoxon(x=ref_scores, y=m_scores, alternative='greater')
        rows.append((method, w_double[0], w_double[1], w_greater[0], w_greater[1]))

    W_test = pd.DataFrame(rows, columns=[
        'method',
        'two-sided_stat',
        'two-sided_pval',
        'greater_stat',
        'greater_pval',
        ]).set_index('method')

    half1 = ['Mean',
        'Mean+mask',
        'Med',
        'Med+mask',
        'Iter',
        'Iter+mask',
        'KNN',
        'KNN+mask',
    ]

    half2 = ['Linear+Mean',
        'Linear+Mean+mask',
        'Linear+Med',
        'Linear+Med+mask',
        'Linear+Iter',
        'Linear+Iter+mask',
        'Linear+KNN',
        'Linear+KNN+mask',
    ]

    W_test = W_test.reindex(half1 + half2)

    W_test['two-sided_pval'] = [f'{w:.2g}' for w in W_test['two-sided_pval']]
    W_test['greater_pval'] = [f'{w:.2g}' for w in W_test['greater_pval']]

    print(W_test)

    W_test.to_csv('scores/wilcoxon.csv')
    W_test.to_latex('scores/wilcoxon.tex')

    W_test1 = W_test.loc[half1]
    W_test2 = W_test.loc[half2]
