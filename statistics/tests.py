"""Run some statistical tests on the results."""
import os
import numpy as np
import pandas as pd
from scipy.stats import f, chi2, wilcoxon, friedmanchisquare
import scikit_posthocs

from prediction.PlotHelper import PlotHelper
from prediction.df_utils import get_scores_tab, get_ranks_tab


def friedman_statistic(ranks, N):
    """Compute the Friedman statistic.

    See (Demsar, 2006).

    Parameters
    ----------
        ranks : np.array of shape (k,)
            Mean ranks of the k algorithms averaged over N datasets
        N : int
            Number of datasets ranks were averaged on

    Returns
    -------
        XF2 : float
            Friedman statistic (chi 2 with k-1 degrees of freedom)
        XF2_pval : flaot
            p-value associated to XF2
        FF : float
            Corrected statistic by Iman and Davenport. F-distribution with
            k-1 and (k-1)*(N-1) degrees of freedom
        FF_pval : float
            p-value associated to FF

    """
    ranks = np.squeeze(np.array(ranks))
    if np.isnan(ranks).any():
        raise ValueError(f'NaN found in given ranks: {ranks}')
    k = ranks.shape[0]

    XF2 = 12*N/(k*(k+1))*(np.sum(np.square(ranks)) - 1/4*k*(k+1)**2)
    FF = (N-1)*XF2/(N*(k-1) - XF2)

    XF2_pval = chi2.sf(XF2, k)
    FF_pval = f.sf(FF, k-1, (k-1)*(N-1))

    return XF2, XF2_pval, FF, FF_pval


def run_wilcoxon():
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # # Agregate accross trials by averaging
    # df = df.reset_index()
    # df['n_trials'] = 1  # Add a count column to keep track of # of trials
    # dfgb = df.groupby(['size', 'db', 'task', 'method', 'fold'])
    # df = dfgb.agg({
    #     'score': 'mean',
    #     'n_trials': 'sum',
    #     'scorer': PlotHelper.assert_equal,  # first and assert equal
    #     'selection': PlotHelper.assert_equal,
    #     'n': PlotHelper.assert_equal,
    #     'p': 'mean',  #PlotHelper.assert_equal,
    #     'type': PlotHelper.assert_equal,
    #     'imputation_WCT': 'mean',
    #     'tuning_WCT': 'mean',
    #     'imputation_PT': 'mean',
    #     'tuning_PT': 'mean',
    # })

    # Aggregate both trials and folds
    df = PlotHelper.aggregate(df, 'score')

    # Reset index to addlevel of the multi index to the columns of the df
    df = df.reset_index()
    df = df.set_index(['size', 'db', 'task', 'method'])

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

    half1 = [
        'Mean',
        'Mean+mask',
        'Med',
        'Med+mask',
        'Iter',
        'Iter+mask',
        'KNN',
        'KNN+mask',
    ]

    half2 = [
        'Linear+Mean',
        'Linear+Mean+mask',
        'Linear+Med',
        'Linear+Med+mask',
        'Linear+Iter',
        'Linear+Iter+mask',
        'Linear+KNN',
        'Linear+KNN+mask',
    ]

    W_test = W_test.reindex(half1 + half2)

    W_test['two-sided_pval'] = [f'{w:.1g}' for w in W_test['two-sided_pval']]
    W_test['greater_pval'] = [f'{w:.1g}' for w in W_test['greater_pval']]

    print(W_test)

    W_test.columns = pd.MultiIndex.from_tuples([s.split('_') for s in W_test.columns])

    print(W_test)

    W_test.rename({
        'two-sided': 'Two-sided',
        'greater': 'Greater',
        'pval': 'p-value',
        'stat': 'Statistic',
    }, axis=1, inplace=True)

    W_test.rename({
        # 'Mean': 'Mean',
        # 'Mean+mask': 'Mean+mask',
        'Med': 'Median',
        'Med+mask': 'Median+mask',
        'Iter': 'Iterative',
        'Iter+mask': 'Iterative+mask',
        # 'KNN': 'KNN',
        # 'KNN+mask': 'KNN+mask',
        # 'Linear+Mean': 'Linear+Mean',
        # 'Linear+Mean+mask': 'Linear+Mean+mask',
        # 'Linear+Med': 'Linear+Med',
        # 'Linear+Med+mask': 'Linear+Med+mask',
        # 'Linear+Iter': 'Linear+Iterative',
        # 'Linear+Iter+mask': 'Linear+Iterative+mask',
        # 'Linear+KNN': 'Linear+KNN',
        # 'Linear+KNN+mask': 'Linear+KNN+mask',
        # 'method': 'Method',
    }, axis=0, inplace=True)

    W_test.index.rename('Method', inplace=True)

    print(W_test)

    # Delete two-sided
    W_test.drop('Two-sided', axis=1, level=0, inplace=True)
    W_test.columns = W_test.columns.droplevel(0)
    W_test.drop('Statistic', axis=1, inplace=True)

    print(W_test)


    W_test.to_csv('scores/wilcoxon.csv')
    W_test.to_latex('scores/wilcoxon.tex')

    # W_test1 = W_test.loc[half1]
    # W_test2 = W_test.loc[half2]


def run_pairwise_wilcoxon():
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # # Agregate accross trials by averaging
    # df = df.reset_index()
    # df['n_trials'] = 1  # Add a count column to keep track of # of trials
    # dfgb = df.groupby(['size', 'db', 'task', 'method', 'fold'])
    # df = dfgb.agg({
    #     'score': 'mean',
    #     'n_trials': 'sum',
    #     'scorer': PlotHelper.assert_equal,  # first and assert equal
    #     'selection': PlotHelper.assert_equal,
    #     'n': PlotHelper.assert_equal,
    #     'p': 'mean',  #PlotHelper.assert_equal,
    #     'type': PlotHelper.assert_equal,
    #     'imputation_WCT': 'mean',
    #     'tuning_WCT': 'mean',
    #     'imputation_PT': 'mean',
    #     'tuning_PT': 'mean',
    # })

    # Aggregate both trials and folds
    df = PlotHelper.aggregate(df, 'score')

    dfgb = df.groupby(['size', 'db', 'task'])
    df['rank'] = dfgb['score'].rank(method='dense', ascending=False).astype(int)
    print(df)

    return

    # Reset index to addlevel of the multi index to the columns of the df
    df = df.reset_index()
    # df = df.set_index(['size', 'db', 'task', 'method'])

    print(df)

    # scikit_posthocs.posthoc_wilcoxon(df, val_col='score', group_col='method')

    res = scikit_posthocs.posthoc_nemenyi(df, val_col='score', group_col='method')

    print(res)


def run_friedman():
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

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

    df = get_ranks_tab(df, method_order=method_order, db_order=db_order, average_sizes=False)
    sizes = df.index.get_level_values(0).unique()

    N = len(df.drop('AVG', level=0, axis=1).columns)

    rows = []
    for size in sizes:
        ranks = df.loc[size, ('AVG', 'All')]

        XF2, XF2_pval, FF, FF_pval = friedman_statistic(ranks, N)
        rows.append([XF2, XF2_pval, FF, FF_pval])

    df_statistic = pd.DataFrame(rows, columns=['XF2', 'XF2_pval', 'FF', 'FF_pval'], index=sizes)

    def myround(x):
        if np.isnan(x):
            return x
        else:
            return f'{x:.2g}'

    df_statistic = df_statistic.applymap(myround)
    print(df_statistic)

    # df = get_scores_tab(df, method_order=method_order, db_order=db_order, relative=True)
    # # df = get_ranks_tab(df, method_order=method_order, db_order=db_order)
    # # df.to_csv('ranks.csv')
    # # df.to_latex('ranks.tex', na_rep='')
    # print(df)
    # df.to_latex('scores.tex', na_rep='', formatters={'Method': {'AVG': r'\textbf{AVG}'}})

    # exit()

    # # Aggregate both trials and folds
    # # df = PlotHelper.aggregate(df, 'score')


    # # df = df[['size', 'db', 'task', 'method', 'score', 'trial', 'fold']]
    # PlotHelper.ranks(df, method_order=method_order)
    # exit()

    # r = PlotHelper.mean_rank('scores/scores.csv', method_order=method_order)



    # df = df[df['method'].isin(method_order)]

    # dfgb = df.groupby(['size', 'db', 'task', 'trial', 'fold'])
    # df['rank'] = dfgb['score'].rank(method='dense', ascending=False)
    # print(df)

    # # Agregate across foldss by averaging
    # dfgb = df.groupby(['size', 'db', 'task', 'method', 'trial'])
    # df = dfgb.agg({'rank': 'mean', 'selection': 'first'})

    # # Agregate across trials by averaging
    # df = df.reset_index()
    # df['n_trials'] = 1  # Add a count column to keep track of # of trials
    # dfgb = df.groupby(['size', 'db', 'task', 'method'])
    # df = dfgb.agg({'rank': 'mean', 'selection': 'first', 'n_trials': 'sum'})

    # # We only take into account full results (n_trials == 5)
    # df = df.reset_index()
    # idx_valid = df.index[(df['selection'] == 'manual') | (
    #     (df['selection'] != 'manual') & (df['n_trials'] == 5))]
    # df = df.loc[idx_valid]

    # dfgb = df.groupby(['db', 'task'])
    # N = len(dfgb)

    # # Average across tasks
    # dfgb = df.groupby(['size', 'method'])
    # df = dfgb.agg({'rank': 'mean'})

    # # Reset index to addlevel of the multi index to the columns of the df
    # df = df.reset_index()

    # # print(df)
    # # return

    # # dfgb = df.groupby(['size', 'db', 'task'])
    # # df['rank'] = dfgb['score'].rank(method='dense', ascending=False).astype(int)

    # # print(df)

    # # dfgb = df.groupby(['db', 'task', 'method'])
    # # df = dfgb.agg({
    # #     'rank': 'mean',
    # # })

    # # df = df.reset_index()
    # # df = df.set_index(['size', 'db', 'task', 'method'])
    # # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # #     print(df)




    # dfgb = df.groupby(['method'])
    # df = dfgb.agg({
    #     'rank': 'mean',
    # })

    # print(df)

    # R = np.squeeze(np.array(df))
    # print(R)

    # k = len(R)

    # XF2 = 12*N/(k*(k+1))*(np.sum(np.square(R)) - 1./4*k*(k+1)**2)

    # print(XF2)

    # FF = (N-1)*XF2/(N*(k-1) - XF2)
    # print(FF)
