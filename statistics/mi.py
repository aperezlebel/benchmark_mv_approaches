from os.path import join
import pandas as pd

from custom.const import get_fig_folder
from prediction.PlotHelper import PlotHelper

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

rename_on_plot = {
    'relative_score': 'Relative prediction score',
    'absolute_score': 'Prediction score difference',
    'relative_total_PT': 'Relative total training time',
    'relative_total_WCT': 'Relative wall-clock time',
    'TB': 'Traumabase',
    'Mean+mask': 'Mean\n+mask',
    'Med': 'Median',
    'Med+mask': 'Median\n+mask',
    'Iter': 'Iterative',
    'Iter+mask': 'Iterative\n+mask',
    'KNN+mask': 'KNN\n+mask',
    'Linear+Mean+mask': 'Linear+Mean\n+mask',
    'Linear+Med+mask': 'Linear+Med\n+mask',
    'Linear+Iter+mask': 'Linear+Iter\n+mask',
    'Linear+KNN+mask': 'Linear+KNN\n+mask',
    'MI+mask': 'MI\n+mask',
    # 'MIA': 'Boosted trees\n+MIA'
}

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
    'MI',
    'MI+mask',
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


def run_multiple_imputation(graphics_folder):
    # pass
    # exit()
    filepath = 'scores/scores.csv'
    filepath_mi = 'scores/scores_mi.csv'
    scores_all = pd.read_csv(filepath, index_col=0)
    scores_mi = pd.read_csv(filepath_mi, index_col=0)

    scores = pd.concat([scores_all, scores_mi], axis=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        scores.drop(index=scores[(scores['db'] == db) & (scores['task'] == task)].index, inplace=True)

    scores['task'] = scores['task'].str.replace('_pvals', '_screening')

    scores = scores.query('size == 2500')

    fig = PlotHelper.plot_scores(
        scores, method_order=method_order, db_order=db_order,
        rename=rename_on_plot, reference_method=None, symbols=None,
        only_full_samples=False, legend_bbox=(1.055, 1.075), figsize=(4.5, 5.25),
        table_fontsize=10)
    xticks = {
        2/3: '$\\frac{2}{3}\\times$',
        1: '$1\\times$',
        3/2: '$\\frac{3}{2}\\times$',
    }
    fig_time = PlotHelper.plot_times(
        scores, 'PT', xticks_dict=xticks, method_order=method_order,
        db_order=db_order, rename=rename_on_plot)

    fig_folder = get_fig_folder(graphics_folder)

    fig_name = 'boxplots_mi_scores'
    fig_time_name = 'boxplots_mi_times'

    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig_time.savefig(join(fig_folder, f'{fig_time_name}.pdf'), bbox_inches='tight')
    fig.savefig(join(fig_folder, f'{fig_name}.jpg'), bbox_inches='tight')
    fig_time.savefig(join(fig_folder, f'{fig_time_name}.jpg'), bbox_inches='tight')
