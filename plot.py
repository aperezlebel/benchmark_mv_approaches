"""Plot the previously dumped results."""

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import sklearn

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from prediction.PlotHelper import PlotHelper

rename = {
    'Regression': 'Missing incorporate attribute',
    'Regression_imputed_Mean': 'Mean imputation',
    'Regression_imputed_Mean+mask': 'Mean+mask imputation',
    'Regression_imputed_Med': 'Med imputation',
    'Regression_imputed_Med+mask': 'Med+mask imputation',
    'Regression_imputed_Iterative': 'Iterative imputation',
    'Regression_imputed_Iterative+mask': 'Iterative+mask imputation',
    'Regression_imputed_KNN': 'KNN imputation',
    'Regression_imputed_KNN+mask': 'KNN+mask imputation',
    'Classification': 'Missing incorporate attribute',
    'Classification_imputed_Mean': 'Mean imputation',
    'Classification_imputed_Mean+mask': 'Mean+mask imputation',
    'Classification_imputed_Med': 'Med imputation',
    'Classification_imputed_Med+mask': 'Med+mask imputation',
    'Classification_imputed_Iterative': 'Iterative imputation',
    'Classification_imputed_Iterative+mask': 'Iterative+mask imputation',
    'Classification_imputed_KNN': 'KNN imputation',
    'Classification_imputed_KNN+mask': 'KNN+mask imputation',
    'TB': 'TraumaBase',
    'TB_ALL_RS_FIXED': 'TraumaBase',
    'TB_ALL_RS_FIXED_TRIALS': 'TraumaBase',
    'TB_ALL_RS_FIXED_TRIALS_TASKS_FIXED': 'TraumaBase',
    'UKBB_ALL_RS_FIXED_TRIALS_TASKS_FIXED': 'UK BioBank',
    'death': 'death',
    'shock_hemo': 'hemorrhagic shock',
    'shock_hemo_16p': 'hemorrhagic shock 16p',
    'shock_hemo_25p_5LC': 'hemorrhagic shock',
    'platelet_25p_5LC': 'platelet',
    'platelet_best_params': 'platelet',
    'shock_hemo_36p_5LC': 'hemorrhagic shock',
    'platelet_36p_5LC': 'platelet',
    'platelet_40p_5LC': 'platelet',
    'platelet_36pv2_5LC': 'platelet',
    'platelet_36pv3bis_5LC': 'platelet',
    'platelet_36pv3_RS0': 'platelet RS0',
    'platelet_36pv3_RS42g': 'platelet RS42',
    'platelet_36pv4_RS42': 'platelet',
    'platelet': 'platelet',
    'acid_40p_5LC': 'acid tranexamic',
    'r2': 'R2',
    'roc_auc_ovr_weighted': 'Area under the ROC curve',
    'platelet_36pv3_RS42_REPRO_CHECK_c': 'platelet RS42',
    'platelet_36pv3_RS42_REPRO_CHECK_g': 'platelet RS42',
    'platelet_36pv3_RS42g': 'platelet RS42',
    'platelet_36pv3_RS42c': 'platelet RS42',
    'platelet_36pv3_RS0': 'platelet RS00',
    'platelet_36pv3_RS0_REPRO_CHECK_c': 'platelet RS00',
    'platelet_36pv3_RS0_REPRO_CHECK_g': 'platelet RS00',
    'platelet_36pv3_5T': 'platelet',
    'platelet_9p_3T': 'platelet',
    'acid_36pv3_5T': 'acid tranexamic',
    'shock_hemo_36pv3_5T': 'hemorrhagic shock',
    'breast_36pv3_5T': 'breast cancer',
    'breast_100pvals_36pv3_5T': 'breast cancer with top 100 pvals',
    'parkinson_100pvals_36pv3_5T': 'parkinson with top 100 pvals',
    'skin_100pvals_36pv3_5T': 'skin cancer with top 100 pvals',
    'death_36pv3_5Tc': 'death',
}


strats = [
    'Classification',
    'Classification_imputed_Mean',
    'Classification_imputed_Mean+mask',
    'Classification_imputed_Med',
    'Classification_imputed_Med+mask',
    'Classification_imputed_KNN',
    'Classification_imputed_KNN+mask',
    # 'Classification_imputed_Iterative',
    # 'Classification_imputed_Iterative+mask',
]

strats_gathered = [
    'Classification',
    ('Classification_imputed_Mean', 'Classification_imputed_Mean+mask'),
    ('Classification_imputed_Med', 'Classification_imputed_Med+mask'),
    # ('Classification_imputed_KNN', 'Classification_imputed_KNN+mask'),
    ('Classification_imputed_Iterative', 'Classification_imputed_Iterative+mask'),
]

ph = PlotHelper('UKBB_ALL_RS_FIXED_TRIALS_TASKS_FIXED/parkinson_100pvals_36pv3_5T', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, RS=[0])

# ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS_TASKS_FIXED/shock_hemo_36pv3_5T', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve_one(None)

plt.show()
exit()


ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS_TASKS_FIXED/shock_hemo_36pv3_5T', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, RS=[0, 1, 2, 3, 4])#, truncate=(0.20, 0.05))
ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS_TASKS_FIXED/acid_36pv3_5T', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, RS=[0, 1, 2, 3, 4])#, truncate=(0.20, 0.05))

strats_gathered = [
    'Classification',
    ('Classification_imputed_Mean', 'Classification_imputed_Mean+mask'),
    ('Classification_imputed_Med', 'Classification_imputed_Med+mask'),
    ('Classification_imputed_KNN', 'Classification_imputed_KNN+mask'),
    # ('Classification_imputed_Iterative', 'Classification_imputed_Iterative+mask'),
]

ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS_TASKS_FIXED/death_36pv3_5Tc', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, RS=[0, 1, 2, 3])#, truncate=(0.20, 0.05))


# ph = PlotHelper('TB/shock_hemo_46', results_folder='results_graham/', rename=rename)
# ph.plot_learning_curve('Classification', truncate=(0.1, 0.05))

# ph = PlotHelper('TB/platelet_47', results_folder='results_graham/', rename=rename)
# ph.plot_learning_curve('Regression', truncate=(0.1, 0.05))

# ph = PlotHelper('TB/platelet_45', results_folder='results_graham/', rename=rename)
# ph.plot_learning_curve('Regression', truncate=(0.1, 0.05))

# plt.show()
# exit()

# ph = PlotHelper('TB/death_after_fix', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.1, 0.05))

# ph = PlotHelper('TB/death', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.1, 0.05))

# ph.plot_scores_accross_strats(scorer=sklearn.metrics.roc_auc_score, strats=strats)
# plt.show()
# exit()

# plt.show()
# exit()


strats = [
    'Regression',
    'Regression_imputed_Mean',
    'Regression_imputed_Mean+mask',
    'Regression_imputed_Med',
    'Regression_imputed_Med+mask',
    'Regression_imputed_KNN',
    'Regression_imputed_KNN+mask',
    'Regression_imputed_Iterative',
    'Regression_imputed_Iterative+mask',
]

strats_gathered = [
    'Regression',
    ('Regression_imputed_Mean', 'Regression_imputed_Mean+mask'),
    ('Regression_imputed_Med', 'Regression_imputed_Med+mask'),
    ('Regression_imputed_KNN', 'Regression_imputed_KNN+mask'),
    ('Regression_imputed_Iterative', 'Regression_imputed_Iterative+mask'),
]
# ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS/platelet_9p_3T', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, RS=[0, 1, 2])#, truncate=(0.20, 0.05))

ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS_TASKS_FIXED/platelet_36pv3_5T', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, RS=[0, 1, 2, 3, 4])#, truncate=(0.20, 0.05))
# ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS/platelet_36pv3_5T', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, RS=[1])#, truncate=(0.20, 0.05))
# ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS/platelet_36pv3_5T', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, RS=[2])#, truncate=(0.20, 0.05))
# ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS/platelet_36pv3_5T', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, RS=[3])#, truncate=(0.20, 0.05))
# ph = PlotHelper('TB_ALL_RS_FIXED_TRIALS/platelet_36pv3_5T', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, RS=[4])#, truncate=(0.20, 0.05))

plt.show()
exit()

# Repro check
ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS42c', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))
ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS42_REPRO_CHECK_c', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))
ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS42g', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))
ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS42_REPRO_CHECK_g', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))

ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS0', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))
ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS0_REPRO_CHECK_g', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05)
ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv3_RS0_REPRO_CHECK_c', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))

plt.show()
exit()

# ph = PlotHelper('TB_ALL_RS_FIXED/platelet_36pv4_RS42', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered)#, truncate=(0.20, 0.05))
# ph = PlotHelper('TB/platelet_36pv3_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# ph = PlotHelper('TB/platelet_36pv3bis_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# ph = PlotHelper('TB/platelet_best_params', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# ph = PlotHelper('TB/platelet_36p_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# ph = PlotHelper('TB/platelet_40p_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# ph = PlotHelper('TB/platelet_30p_RS0_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# plt.show()
# exit()
# ph = PlotHelper('TB/platelet_25p_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))

# ph = PlotHelper('TB/platelet', results_folder='results_selected/', rename=rename)
# # # ph.plot_roc('Classification_imputed_Mean')
# ph.plot_learning_curve(strats_gathered, truncate=(0.20, 0.05))
# plt.show()
# exit()

strats = [
    'Classification',
    'Classification_imputed_Mean',
    'Classification_imputed_Mean+mask',
    'Classification_imputed_Med',
    'Classification_imputed_Med+mask',
    'Classification_imputed_KNN',
    'Classification_imputed_KNN+mask',
    'Classification_imputed_Iterative',
    'Classification_imputed_Iterative+mask',
]

strats_gathered = [
    'Classification',
    ('Classification_imputed_Mean', 'Classification_imputed_Mean+mask'),
    ('Classification_imputed_Med', 'Classification_imputed_Med+mask'),
    ('Classification_imputed_KNN', 'Classification_imputed_KNN+mask'),
    ('Classification_imputed_Iterative', 'Classification_imputed_Iterative+mask'),
]

ph = PlotHelper('TB/shock_hemo_212', results_folder='results/', rename=rename)
ph.plot_learning_curve(['Classification'], truncate=(0, 0), RS=[0, 42])
ph.plot_learning_curve(['Classification'], truncate=(0, 0), RS=[0])
ph.plot_learning_curve(['Classification'], truncate=(0, 0), RS=[42])
plt.show()
exit()


ph = PlotHelper('TB/acid_40p_5LC', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, truncate=(-0.50, 0))
ph = PlotHelper('TB/acid_9pc_5LC', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, truncate=(-0.50, 0))
ph = PlotHelper('TB/acid_9pg_5LC', results_folder='results_selected/', rename=rename)
ph.plot_learning_curve(strats_gathered, truncate=(-0.50, 0))

plt.show()
exit()

ph = PlotHelper('TB/shock_hemo_36p_5LC', results_folder='results_selected/', rename=rename)
# ph.plot_roc('Classification_imputed_Mean')
ph.plot_learning_curve(strats_gathered, truncate=(0.35, 0.075))

# ph = PlotHelper('TB/shock_hemo', results_folder='results_selected/', rename=rename)
# # ph.plot_roc('Classification_imputed_Mean')
# ph.plot_learning_curve(strats_gathered, truncate=(0.35, 0.075))
# ph.plot_scores_accross_strats(scorer=sklearn.metrics.roc_auc_score, strats=strats)

plt.show()
exit()
# ph.plot_roc('Classification')
# ph = PlotHelper('UKBB/fluid_intelligence_15', results_folder='results_cedar/')
# ph.plot_learning_curve('Regression')
# ph.plot_regression('Regression')
# # ph.plot_scores_accross_strats('Regression')
# strat_infos = ph.get_strat_infos('Regression')
# print(strat_infos['outer_cv_params'])
# print(strat_infos['inner_cv_params'])
# print(strat_infos['search_params'])

# ph = PlotHelper('UKBB/fluid_intelligence_light_2')
# ph.plot_learning_curve('Regression')
# plt.show()
# exit()

rename = {
    'Regression': 'HistGradientBoostingRegressor',
    'Regression_imputed_Mean': 'HGBR_imputed_Mean',
    'Regression_imputed_Mean+mask': 'HGBR_imputed_Mean+mask',
    'Regression_imputed_Med': 'HGBR_imputed_Med',
    'Regression_imputed_Med+mask': 'HGBR_imputed_Med+mask'
}

ph = PlotHelper('UKBB/fluid_intelligence_1', results_folder='results_cedar/', rename=rename)
# _, axes = plt.subplots(1, 2)
ph.plot_regression('Regression', ax=None)
ph.plot_regression('Regression_imputed_Med', ax=None)
strats = [
    'Regression',
    'Regression_imputed_Mean',
    'Regression_imputed_Mean+mask',
    'Regression_imputed_Med',
    'Regression_imputed_Med+mask'
]
ph.plot_scores_accross_strats(strats=strats, ax=None,
                              scorer=sklearn.metrics.mean_absolute_error)
ph.plot_scores_accross_strats(strats=strats, ax=None,
                              scorer=sklearn.metrics.r2_score)
# ph.plot_learning_curve('Regression')
# ph.plot_full_results('HistGradientBoostingClassifier')
plt.show()

# ph = PlotHelper('UKBB/fluid_intelligence')
# ph.plot_full_results('HistGradientBoostingRegressor')

# import seaborn as sns

# fmri = sns.load_dataset("fmri")
# print(fmri)
# exit()

# from sklearn import metrics
# ph = PlotHelper('TB/shock_hemo')
# ph.plot_roc('HistGradientBoostingClassifier')
# ph.plot_scores('HistGradientBoostingClassifier', [
#     # metrics.accuracy_score,
#     metrics.recall_score,

# ])

plt.show()
