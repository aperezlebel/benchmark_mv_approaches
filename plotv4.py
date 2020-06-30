"""Plot the results of --train4."""
from prediction.PlotHelperV4 import PlotHelperV4
import matplotlib.pyplot as plt


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
    'rel_score': 'Relative prediction score',
    'TB': 'Traumabase',
    'UKBB': 'UK BioBank',
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

db_order = [
    'TB',
    'MIMIC',
    'NHIS',
    'UKBB',
]

ph = PlotHelperV4(root_folder='results_original/graham/results/', rename=rename)

filepath = 'scores/scores_results_graham.csv'
# ph.dump(filepath)
fig = ph.plot(filepath, method_order=method_order, db_order=db_order, rename=rename)#, reference_method='MIA')
if fig:
    plt.show()
