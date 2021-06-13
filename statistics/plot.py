import pandas as pd
import matplotlib.pyplot as plt

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
    'UKBB': 'UK BioBank',
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
]

linear_method_order = [
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


def run_boxplot(graphics_folder, linear):
    filepath = 'scores/scores.csv'
    scores = pd.read_csv(filepath, index_col=0)

    # Drop tasks
    scores = scores.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        scores = scores.drop((db, task), axis=0)
    scores = scores.reset_index()

    if linear:
        fig = PlotHelper.plot_MIA_linear(scores, db_order=db_order, method_order=linear_method_order, rename=rename_on_plot)
    else:
        fig = PlotHelper.plot_scores(scores, method_order=method_order, db_order=db_order, rename=rename_on_plot, reference_method=None)
    
    if fig:
        plt.show()