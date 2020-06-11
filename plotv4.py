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
    'relative_score': 'Relative prediction score',
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
]

ph = PlotHelperV4(root_folder='results_selected/FINAL_RESULTS/trial4/',
                  rename=rename, reference_method='MIA')

# print(ph.databases())
# print(ph.existing_methods())
# print(ph.existing_sizes())
# methods = ph.methods('TB', 'platelet')
# print(methods)
# print(ph.score('TB', 'platelet', 'RS0_Regression_imputed_Med+mask', '5000', mean=True))
# print(ph.relative_scores('TB', 'platelet', methods, '5000'))

# av_methods = ph.availale_methods_by_size('TB', 'platelet', '5000')
# print(av_methods)
# print(ph.relative_scores('TB', 'platelet', av_methods, '5000'))

fig = ph.plot(method_order=method_order, db_order=db_order, compute=True)
if fig:
    plt.show()
