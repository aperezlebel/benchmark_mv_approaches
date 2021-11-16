from os.path import join

import pandas as pd
from custom.const import get_fig_folder
from prediction.PlotHelper import PlotHelper

from .tests import run_wilcoxon

rename = {
    '': 'MIA',
    '_imputed_Mean': 'Mean',
    '_imputed_Mean+mask': 'Mean+mask',
    '_imputed_Med': 'Med',
    '_imputed_Med+mask': 'Med+mask',
    '_imputed_Iterative': 'Iter',
    '_imputed_Iterative+mask': 'Iter+mask',
    '_imputed_KNN': 'KNN',
    '_imputed_KNN+mask': 'KNN+mask',
    '_imputed_Iterative_Bagged100': 'MI',
    '_imputed_Iterative+mask_Bagged100': 'MI+mask',
    # '_Bagged100': 'MIA+Bagging',
    '_Logit_imputed_Mean': 'Linear+Mean',
    '_Logit_imputed_Mean+mask': 'Linear+Mean+mask',
    '_Logit_imputed_Med': 'Linear+Med',
    '_Logit_imputed_Med+mask': 'Linear+Med+mask',
    '_Logit_imputed_Iterative': 'Linear+Iter',
    '_Logit_imputed_Iterative+mask': 'Linear+Iter+mask',
    '_Logit_imputed_KNN': 'Linear+KNN',
    '_Logit_imputed_KNN+mask': 'Linear+KNN+mask',
    '_Ridge_imputed_Mean': 'Linear+Mean',
    '_Ridge_imputed_Mean+mask': 'Linear+Mean+mask',
    '_Ridge_imputed_Med': 'Linear+Med',
    '_Ridge_imputed_Med+mask': 'Linear+Med+mask',
    '_Ridge_imputed_Iterative': 'Linear+Iter',
    '_Ridge_imputed_Iterative+mask': 'Linear+Iter+mask',
    '_Ridge_imputed_KNN': 'Linear+KNN',
    '_Ridge_imputed_KNN+mask': 'Linear+KNN+mask',
}

# rename_on_plot = {
#     'relative_score': 'Relative prediction score',
#     'absolute_score': 'Prediction score difference',
#     'relative_total_PT': 'Relative total training time',
#     'relative_total_WCT': 'Relative wall-clock time',
#     'TB': 'Traumabase',
#     'Mean+mask': 'Mean\n+mask',
#     'Med': 'Median',
#     'Med+mask': 'Median\n+mask',
#     'Iter': 'Iterative',
#     'Iter+mask': 'Iterative\n+mask',
#     'KNN+mask': 'KNN+mask',
#     'Linear+Mean+mask': 'Linear+Mean\n+mask',
#     'Linear+Med+mask': 'Linear+Med\n+mask',
#     'Linear+Iter+mask': 'Linear+Iter\n+mask',
#     'Linear+KNN+mask': 'Linear+KNN\n+mask',
#     'MI+mask': 'MI\n+mask',
#     'MIA+Bagging100': 'MIA+Bagging',
#     'MIA+bagging': 'MIA+Bagging',
#     'MI': 'Iterative\n+Bagging (MI)',
#     'MI+mask': 'Iterative+mask\n+Bagging (MI)',
#     # 'MIA': 'Boosted trees\n+MIA'
# }
rename_on_plot = {
    'relative_score': 'Relative prediction score',
    'absolute_score': 'Prediction score difference',
    'relative_total_PT': 'Relative total training time',
    'relative_total_WCT': 'Relative wall-clock time',
    'TB': 'Traumabase',
    'Mean+mask': 'Mean+mask',
    'Med': 'Median',
    'Med+mask': 'Median+mask',
    'Iter': 'Iterative',
    'Iter+mask': 'Iterative+mask',
    'KNN+mask': 'KNN+mask',
    'Linear+Mean+mask': 'Linear+Mean+mask',
    'Linear+Med+mask': 'Linear+Med+mask',
    'Linear+Iter+mask': 'Linear+Iter+mask',
    'Linear+KNN+mask': 'Linear+KNN+mask',
    'MI+mask': 'MI+mask',
    'MIA+Bagging100': 'MIA+Bagging',
    'MIA+bagging': 'MIA+Bagging',
    'MI': 'Iterative+Bagging (MI)',
    'MI+mask': 'Iterative+mask+Bagging (MI)',
    # 'MIA': 'Boosted trees+MIA'
}

method_order_all = [
    'MIA',
    'Mean',
    'Mean+mask',
    'Med',
    'Med+mask',
    'Iter',
    'Iter+mask',
    'KNN',
    'KNN+mask',
    'MI',
    'MI+mask',
    'MIA+bagging',
]

method_order_bagging = [
    'MIA',
    'Mean+mask',
    'Mean+mask+bagging',
    # 'MI',
    'MI+mask',
    'MIA+bagging',
]

method_order_linear = [
    # 'MIA', 'Linear+Iter', 'Linear+Iter+mask',
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

db_order = [
    'TB',
    'UKBB',
    'MIMIC',
    'NHIS',
]

tasks_to_drop = {
    'TB': 'platelet',
    'NHIS': 'bmi_pvals',
}


def run_multiple_imputation(graphics_folder, n=None, bagging_only=False, linear=False):
    if bagging_only and linear:
        raise ValueError('No linear for bagging')
    elif bagging_only:
        method_order = method_order_bagging
    elif linear:
        method_order = method_order_linear
    else:
        method_order = method_order_all

    # reference_method = None if bagging_only else 'MIA'
    reference_method = 'MIA'

    filepaths = [
        'scores/scores.csv',
        'scores/scores_mi_2500.csv',
        'scores/scores_mia_2500.csv',
        'scores/scores_mi_10000.csv',
        'scores/scores_mia_10000.csv',
        'scores/scores_mia_25000.csv',
        'scores/scores_mi_25000.csv',
        'scores/scores_mia_100000.csv',
        'scores/scores_mean+mask+bagging_2500.csv',
    ]
    dfs = [pd.read_csv(path, index_col=0) for path in filepaths]
    scores = pd.concat(dfs, axis=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        scores.drop(index=scores[(scores['db'] == db) & (scores['task'] == task)].index, inplace=True)

    scores['task'] = scores['task'].str.replace('_pvals', '_screening')

    # Get Wilcoxon table for symbol annotation
    W_test_greater = run_wilcoxon(graphics_folder=None, spacing=False, no_rename=True, greater=True)
    W_test_less = run_wilcoxon(graphics_folder=None, spacing=False, no_rename=True, greater=False)

    symbols = {}

    def pvalue_to_symbol(pvalue, alpha, n_bonferroni, greater=True):
        c = '' if greater else '(>)'
        if pvalue < alpha/n_bonferroni:
            return f'$\\star\\star{c}$'
        if pvalue < alpha:
            return f'$\\star{c}$'
        return None

    alpha = 0.05
    n_bonferroni = W_test_greater.shape[0]

    for size in W_test_greater:
        symbols[size] = {}
        symbols[size]['MIA'] = '$\\rightarrow$'
        for k, v in W_test_greater[size].iteritems():
            symbols[size][k] = pvalue_to_symbol(v, alpha, n_bonferroni, greater=True)
        for k, v in W_test_less[size].iteritems():
            if symbols[size][k] is None:
                symbols[size][k] = pvalue_to_symbol(v, alpha, n_bonferroni, greater=False)

    comments = {}
    comments[100000] = {}
    comments[100000]['KNN'] = 'Intractable'
    comments[100000]['KNN+mask'] = 'Intractable'

    if n is not None:
        scores = scores.query(f'size == {n}')
        figsize = (4.5, 5.25)
        legend_bbox = (1.055, 1.075)

    else:
        figsize = (18, 6)
        # legend_bbox = (4.415, 1.015)
        legend_bbox = (2.02, 1.05)

    if len(method_order) >= 12:
        y_labelsize = 14
    else:
        y_labelsize = 18

    fig = PlotHelper.plot_scores(
        scores, method_order=method_order, db_order=db_order,
        rename=rename_on_plot, reference_method=None, symbols=symbols,
        only_full_samples=False, legend_bbox=legend_bbox, figsize=figsize,
        table_fontsize=13, y_labelsize=y_labelsize, comments=comments)

    scores['total_PT'] = scores['imputation_PT'].fillna(0) + scores['tuning_PT']
    scores['tag'] = scores['db'] + '/' + scores['task']
    print(scores)
    for index, subdf in scores.groupby(['size', 'method']):
        size, method = index
        # print(subdf)
        # print(subdf['tag'])
        # print(subdf['tag'].unique())
        # print(len(subdf['tag'].unique()))
        n_tasks = len(subdf['tag'].unique())
        total_pt_time = subdf['total_PT'].sum()/n_tasks
        # exit()
        # print(index)
        comments_size = comments.get(size, {})
        if comments_size.get(method, None) is None:
            comments_size[method] = f'{int(total_pt_time/3600):,d} hours'.replace(',', '\\,')
        comments[size] = comments_size
    print(comments)
    # exit()
    xticks = {
        # 1/10: '$\\frac{1}{10}\\times$',
        # 2/3: '$\\frac{2}{3}\\times$',
        1: '$1\\times$',
        # 3/2: '$\\frac{3}{2}\\times$',
        2: '$2\\times$',
        # 5: '$5\\times$',
        # 10: '$10\\times$',
        50: '$50\\times$',
        100: '$100\\times$',
        # 150: '$150\\times$',
        200: '$200\\times$',
        500: '$500\\times$',
    }
    legend_bbox = (4.16, 1.05)
    broken_axis = [(2.3, 55), (2.3, 55), (2.5, 25), (3.5, 25)]
    y_labelsize = 15.5
    if bagging_only:
        comments_align = {
            0: ['right']*2+['left']*3,
            1: ['right']*2+['left']*3,
            2: ['right']*2+['left']*3,
            3: ['right']*2+['left']*3,
        }
        y_labelsize = 20

    elif linear:
        comments_align = None
        broken_axis = None
        legend_bbox = (2.02, 1.05)

    else:
        comments_align = {
            0: ['right']*9+['left']*3,
            1: ['right']*9+['left']*3,
            2: ['right']*9+['left']*3,
            3: ['right']*7+['left']*5,
        }

    # figsize = (18, 6)

    fig_time = PlotHelper.plot_times(
        scores, 'PT', xticks_dict=xticks, method_order=method_order,
        db_order=db_order, rename=rename_on_plot, y_labelsize=y_labelsize,
        legend_bbox=legend_bbox, broken_axis=broken_axis,
        only_full_samples=False, reference_method=reference_method, figsize=figsize, comments=comments,
        comments_align=comments_align, comments_spacing=0.11)

    fig.subplots_adjust(wspace=0.02)
    fig_time.subplots_adjust(wspace=0.02)
    fig_folder = get_fig_folder(graphics_folder)
    if bagging_only:
        name = 'bagging'
    elif linear:
        name = 'linear'
    else:
        name = 'mi'

    fig_name = f'boxplots_{name}_scores_{n}'
    fig_time_name = f'boxplots_{name}_times_{n}'

    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig_time.savefig(join(fig_folder, f'{fig_time_name}.pdf'), bbox_inches='tight')
    fig.savefig(join(fig_folder, f'{fig_name}.jpg'), bbox_inches='tight')
    fig_time.savefig(join(fig_folder, f'{fig_time_name}.jpg'), bbox_inches='tight')
