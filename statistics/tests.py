"""Run some statistical tests on the results."""
import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.stats import f, chi2, wilcoxon
import matplotlib.pyplot as plt
# from adjustText import adjust_text

from prediction.PlotHelper import PlotHelper
from prediction.df_utils import get_scores_tab, get_ranks_tab
from custom.const import get_fig_folder, get_tab_folder


tasks_to_drop = {
    'TB': 'platelet',
    'NHIS': 'bmi_pvals',
}

db_rename = {
    'TB': 'Traumabase',
}

db_order = [
    'TB',
    'UKBB',
    'MIMIC',
    'NHIS',
]


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

    print(f'k={k}, N={N}')
    CD = critical_distance(k, N)

    return XF2, XF2_pval, FF, FF_pval, CD


def critical_distance(k, N):
    """Compute the critical difference for the Nemenyi test.

    Parameters
    ----------
        k : Number of algorithms to compare (2 <= k <= 10)
        N : Number of datasets algorithms were tested on

    Returns
    -------
        CD : float
            Critical difference

    """
    assert 2 <= k <= 10
    q_05 = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
    CD = q_05[k-2]*np.sqrt(k*(k+1)/(6*N))

    return CD


def run_wilcoxon_():
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
    W_test.to_latex('scores/wilcoxon.tex', na_rep='')

    # W_test1 = W_test.loc[half1]
    # W_test2 = W_test.loc[half2]


def run_wilcoxon_mia(graphics_folder, csv=False):
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Drop tasks
    df = df.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        df = df.drop((db, task), axis=0)
    df = df.reset_index()

    df['task'] = df['task'].str.replace('_pvals', '_screening')

    method_order1 = [
        'Mean',
        'Mean+mask',
        'Med',
        'Med+mask',
        'Iter',
        'Iter+mask',
        'KNN',
        'KNN+mask',
    ]

    method_order2 = [
        'Linear+Mean',
        'Linear+Mean+mask',
        'Linear+Med',
        'Linear+Med+mask',
        'Linear+Iter',
        'Linear+Iter+mask',
        'Linear+KNN',
        'Linear+KNN+mask',
    ]

    method_order = ['MIA'] + method_order1 + method_order2

    db_order = [
        'TB',
        'UKBB',
        'MIMIC',
        'NHIS',
    ]

    df = get_scores_tab(df, method_order=method_order, db_order=db_order,
                        average_sizes=False, formatting=False)
    sizes = df.index.get_level_values(0).unique()

    rows = []
    for size in sizes:
        # print(f'Size={size}: ', end='\t')

        scores = df.loc[size]
        ref_scores = scores.loc['MIA']

        methods = scores.index.unique()
        methods = [m for m in methods if m != 'MIA']

        for method in methods:
            m_scores = scores.loc[method]
            idx = m_scores.notnull()

            x = ref_scores[idx]
            y = m_scores[idx]

            print(x)

            w_double = wilcoxon(x=x, y=y, alternative='two-sided')
            w_greater = wilcoxon(x=x, y=y, alternative='greater')

            rows.append([size, method, w_double[0], w_double[1], w_greater[0], w_greater[1]])

    W_test = pd.DataFrame(rows, columns=[
        'size',
        'method',
        'two-sided_stat',
        'two-sided_pval',
        'greater_stat',
        'greater_pval',
        ]).set_index(['size', 'method'])

    # W_test = W_test.reindex(method_order)

    # W_test['two-sided_pval'] = [f'{w:.1g}' for w in W_test['two-sided_pval']]
    # W_test['greater_pval'] = [f'{w:.1g}' for w in W_test['greater_pval']]


    W_test.drop(['two-sided_pval', 'two-sided_stat'], axis=1, inplace=True)

    W_test.rename({
        'greater_pval': 'p-value',
        'greater_stat': 'Statistic',
    }, axis=1, inplace=True)

    W_test.index.rename('Size', level=0, inplace=True)
    W_test.index.rename('Method', level=1, inplace=True)

    W_test.drop(['Statistic'], axis=1, inplace=True)

    W_test = pd.pivot_table(W_test, values='p-value', index='Method', columns='Size')

    W_test = W_test.reindex(method_order1 + method_order2)

    W_test.rename({
        'Med': 'Median',
        'Med+mask': 'Median+mask',
        'Iter': 'Iterative',
        'Iter+mask': 'Iterative+mask',
    }, axis=0, inplace=True)

    def pvalue_formatter(x, alpha, n_bonferroni):
        if np.isnan(x):
            return x
        else:
            if x < alpha/n_bonferroni:  # below bonferroni corrected alpha level
                return f'$\\text{{{x:.1e}}}^{{\\star\\star}}$'

            if x < alpha:  # below alpha level but above bonferroni
                return f'$\\text{{{x:.1e}}}^{{\\star}}$'

            return f'{x:.1e}'

    print(W_test)

    tab_folder = get_tab_folder(graphics_folder)

    if csv:
        W_test.to_csv(join(tab_folder, 'wilcoxon_greater.csv'))

    print(f'Apply Bonferroni correction with {W_test.shape[0]} values.')
    W_test = W_test.applymap(lambda x: pvalue_formatter(x, alpha=0.05, n_bonferroni=W_test.shape[0]))
    W_test.to_latex(join(tab_folder, 'wilcoxon_greater.tex'), na_rep='', escape=False)


def run_wilcoxon_linear(graphics_folder, csv=False):
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Drop tasks
    df = df.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        df = df.drop((db, task), axis=0)
    df = df.reset_index()

    df['task'] = df['task'].str.replace('_pvals', '_screening')

    method_order1 = [
        'Mean',
        'Mean+mask',
        'Med',
        'Med+mask',
        'Iter',
        'Iter+mask',
        'KNN',
        'KNN+mask',
    ]

    method_order2 = [
        'Linear+Mean',
        'Linear+Mean+mask',
        'Linear+Med',
        'Linear+Med+mask',
        'Linear+Iter',
        'Linear+Iter+mask',
        'Linear+KNN',
        'Linear+KNN+mask',
    ]

    method_order = method_order1 + method_order2

    db_order = [
        'TB',
        'UKBB',
        'MIMIC',
        'NHIS',
    ]

    df = get_scores_tab(df, method_order=method_order, db_order=db_order,
                        average_sizes=False, formatting=False)
    sizes = df.index.get_level_values(0).unique()

    rows = []
    for size in sizes:
        # print(f'Size={size}: ', end='\t')

        scores = df.loc[size]
        for method1, method2 in zip(method_order1, method_order2):
            try:
                scores1 = scores.loc[method1]
                scores2 = scores.loc[method2]

            # methods = scores.index.unique()
            # methods = [m for m in methods if m != 'MIA']

            # for method in methods:
                # m_scores = scores.loc[method]
                # idx = m_scores.notnull()

                # x = ref_scores[idx]
                # y = m_scores[idx]

                # print(x)
                x, y = scores1, scores2

                w_double = wilcoxon(x=x, y=y, alternative='two-sided')
                w_greater = wilcoxon(x=x, y=y, alternative='greater')

            except KeyError:
                w_double_= (np.nan, np.nan)
                w_greater = (np.nan, np.nan)

            rows.append([size, method1, w_double[0], w_double[1], w_greater[0], w_greater[1]])

    W_test = pd.DataFrame(rows, columns=[
        'size',
        'imputer',
        'two-sided_stat',
        'two-sided_pval',
        'greater_stat',
        'greater_pval',
        ]).set_index(['size', 'imputer'])

    # W_test = W_test.reindex(method_order)

    # W_test['two-sided_pval'] = [f'{w:.1g}' for w in W_test['two-sided_pval']]
    # W_test['greater_pval'] = [f'{w:.1g}' for w in W_test['greater_pval']]


    W_test.drop(['two-sided_pval', 'two-sided_stat'], axis=1, inplace=True)

    W_test.rename({
        'greater_pval': 'p-value',
        'greater_stat': 'Statistic',
    }, axis=1, inplace=True)

    W_test.index.rename('Size', level=0, inplace=True)
    W_test.index.rename('Imputer', level=1, inplace=True)

    W_test.drop(['Statistic'], axis=1, inplace=True)

    W_test = pd.pivot_table(W_test, values='p-value', index='Imputer', columns='Size')

    W_test = W_test.reindex(method_order1)

    W_test.rename({
        'Med': 'Median',
        'Med+mask': 'Median+mask',
        'Iter': 'Iterative',
        'Iter+mask': 'Iterative+mask',
    }, axis=0, inplace=True)

    def pvalue_formatter(x, alpha, n_bonferroni):
        if np.isnan(x):
            return x
        else:
            if x < alpha/n_bonferroni:  # below bonferroni corrected alpha level
                return f'$\\text{{{x:.1e}}}^{{\\star\\star}}$'

            if x < alpha:  # below alpha level but above bonferroni
                return f'$\\text{{{x:.1e}}}^{{\\star}}$'

            return f'{x:.1e}'

    print(W_test)

    tab_folder = get_tab_folder(graphics_folder)

    if csv:
        W_test.to_csv(join(tab_folder, 'wilcoxon_linear_greater.csv'))

    print(f'Apply Bonferroni correction with {W_test.shape[0]} values.')
    W_test = W_test.applymap(lambda x: pvalue_formatter(x, alpha=0.05, n_bonferroni=W_test.shape[0]))
    W_test.to_latex(join(tab_folder, 'wilcoxon_linear_greater.tex'), na_rep='', escape=False, table_env='tabularx')


def run_wilcoxon(graphics_folder, linear=False, csv=False):
    if linear:
        run_wilcoxon_linear(graphics_folder, csv=csv)
    else:
        run_wilcoxon_mia(graphics_folder, csv=csv)
    

def run_friedman(graphics_folder, linear=False, csv=False):
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Drop tasks
    df = df.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        df = df.drop((db, task), axis=0)
    df = df.reset_index()

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


        # method_order = [
        #     'MIA',
        #     'Mean',
        #     'Mean+mask',
        #     'Med',
        #     'Med+mask',
        #     'Iter',
        #     'Iter+mask',
        #     'KNN',
        #     'KNN+mask',
        #     'Linear+Mean',
        #     'Linear+Mean+mask',
        #     'Linear+Med',
        #     'Linear+Med+mask',
        #     'Linear+Iter',
        #     'Linear+Iter+mask',
        #     'Linear+KNN',
        #     'Linear+KNN+mask',
        # ]

    db_order = [
        'TB',
        'UKBB',
        'MIMIC',
        'NHIS',
    ]

    df = get_ranks_tab(df, method_order=method_order, db_order=db_order, average_sizes=False)
    sizes = df.index.get_level_values(0).unique()

    ranks_by_db = df.drop('AVG', level=0, axis=1)

    rows = []
    for size in sizes:
        print(f'Size={size}: ', end='\t')

        ranks = df.loc[size, ('AVG', 'All')]
        N = (~ranks_by_db.loc[size].isna().all(axis=0)).sum()

        XF2, XF2_pval, FF, FF_pval, CD = friedman_statistic(ranks, N)
        rows.append([XF2, XF2_pval, FF, FF_pval, CD, N])

    df_statistic = pd.DataFrame(rows, columns=['XF2', 'XF2_pval', 'FF', 'FF_pval', 'CD', 'N'], index=sizes)

    def myround(x):
        if np.isnan(x):
            return x
        else:
            return f'{x:.2g}'

    df_statistic = df_statistic.applymap(myround)

    fig, axes = plt.subplots(2, 2, figsize=(7, 8))

    if linear:
        print(df)
        df.rename({'MIA': 'Boosted trees+MIA'}, axis=0, inplace=True)
        print(df)

    for i, ax in enumerate(axes.reshape(-1)):
        size = sizes[i]
        ranks = df.loc[size, ('AVG', 'All')]
        critical_distances = df_statistic['CD'].astype(float)
        plot_ranks(ranks, critical_distances[size], ax)
        N = df_statistic.loc[size, 'N']
        ax.set_title(f'Size={size}, N={N}')

    fig_folder = get_fig_folder(graphics_folder)
    tab_folder = get_tab_folder(graphics_folder)

    tab_name = 'friedman_linear' if linear else 'friedman'
    fig_name = 'critical_distance_linear' if linear else 'critical_distance'

    plt.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')

    print(df_statistic)

    if csv:
        df_statistic.to_csv(join(tab_folder, f'{tab_name}.csv'))

    # Preprocessing for latex dump

    df_statistic.rename({
        'XF2': r'$\chi^2_F$',
        'XF2_pval': r'$\chi^2_F$ p-value',
        'FF': r'$F_F$',
        'FF_pval': r'$F_F$ p-value',
    }, axis=1, inplace=True)
    df_statistic.to_latex(join(tab_folder, f'{tab_name}.tex'), na_rep='', escape=False)

    return df_statistic


def plot_ranks(average_ranks, critical_distance, ax):
    average_ranks = average_ranks.sort_values()
    min_rank = np.min(average_ranks)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_color('none')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_visible(False)#.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim(1, 9)
    ax.set_xlim(-.3, .3)
    ax.invert_yaxis()

    cd1 = min_rank
    cd2 = min_rank + critical_distance
    colors = ['red' if r < cd2 else 'black' for r in average_ranks]
    ax.scatter(np.zeros_like(average_ranks), average_ranks, color=colors, marker='.', clip_on=False, zorder=10)
    ax.plot(-.1*np.ones(2), [cd1, cd2], color='red', marker='_', markeredgewidth=1.5)
    ax.text(-.12, (cd1+cd2)/2, 'critical distance', rotation=90, ha='center', va='center', color='red')
    # ax.annotate('Critical difference', xy=(-.1, (cd1+cd2)/2), textcoords="offset points",
            # horizontalalignment="right", verticalalignment="bottom")

    texts = []
    # for i, y in enumerate(np.linspace(1.5, 8.5, len(average_ranks))):
    y_pos = np.linspace(1.5, 8.5, len(average_ranks))
    for i, (method, rank) in enumerate(average_ranks.iteritems()):
        t = ax.text(.12, y_pos[i], method, va='center')
        texts.append(t)
        ax.plot([0, .12], [rank, y_pos[i]], color='black', ls=':', lw=0.25)

    # adjust_text(texts, ax=ax, autoalign=False, ha='left', va='center')

    # for text in texts:
    #     x, y = text.get_position()
    #     text.set_position((.12, y))
        # text.set_horizontalalignment('left')

def run_scores(graphics_folder, linear, csv=False):
    path = os.path.abspath('scores/scores.csv')
    df = pd.read_csv(path, index_col=0)

    # Drop tasks
    df = df.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        df = df.drop((db, task), axis=0)
    df = df.reset_index()

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

    # Preprocessing for latex dump
    tasks = scores.columns.get_level_values(1)
    rename = {k: k.replace("_", r"\_") for k in tasks}
    rename = {k: f'\\rot{{{v}}}' for k, v in rename.items()}

    scores.rename(columns=rename, inplace=True)
    ranks.rename(columns=rename, inplace=True)

    tab_folder = get_tab_folder(graphics_folder)
    tab1_name = 'scores_linear' if linear else 'scores'
    tab2_name = 'ranks_linear' if linear else 'ranks'

    scores.to_latex(join(tab_folder, f'{tab1_name}.tex'), na_rep='', escape=False) #, column_format='L'*scores.shape[1])
    ranks.to_latex(join(tab_folder, f'{tab2_name}.tex'), na_rep='', escape=False)# , column_format='L'*ranks.shape[1])

    if csv:
        scores.to_csv(join(tab_folder, f'{tab1_name}.csv'))
        ranks.to_csv(join(tab_folder, f'{tab2_name}.csv'))
