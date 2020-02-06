from .tasks import NHIS_tasks
# print(NHIS_tasks)
# exit()

import matplotlib
matplotlib.use('MacOSX')
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from database import TB
from .prediction_task import PredictionTask
from .strategy import Strategy
from .train import train




RS = 42
test_size = 0.33

# df_encoded = TB.encoded_dataframes['20000']
# df_imputed = pd.read_csv('imputed/TB_20000_imputed_rounded_Iterative.csv', sep=';', index_col=0)
# df = df_imputed

# feature_to_predict = "Date de décès (à l'hôpital après sortie de réanimation)"

# y = df_encoded[feature_to_predict]
# y = (~y.isna()).astype(int)
# y = y.astype(int)
# print(y)
# exit()
# print(y.value_counts())
# exit()
# print(y)
# print(df_imputed)
# print(f'START{df_imputed.columns[-1]}STOP')
# for i, col in enumerate(df_imputed.columns):
#     if col == feature_to_predict:
#         print(i)
# print(df_imputed.iloc[])
# print(df_imputed[feature_to_predict])
# df_imputed = df_imputed.rename(columns={feature_to_predict: 'is_dead'})
# print(df_imputed)

# to_drop = [
#     feature_to_predict,
#     "Décès",
#     "Cause du décès_A survécu",
#     "Cause du décès_Autre (précisez ci-dessous)",
#     "Cause du décès_Choc hémorragique",
#     "Cause du décès_Choc septique",
#     "Cause du décès_Défaillance multi-viscérale",
#     "Cause du décès_LATA",
#     "Cause du décès_Mort encéphalique",
#     "Cause du décès_Trauma cranien",
#     "Cause du décès_z MISSING_VALUE",
#     "Transfert secondaire, pourquoi ?_Pas de transfert",
#     "Transfert secondaire, pourquoi ?_Plateau technique insuffisant",
#     "Transfert secondaire, pourquoi ?_Rapprochement familial",
#     "Transfert secondaire, pourquoi ?_z MISSING_VALUE",
#     "Sortie_Autre réanimation",
#     "Sortie_Centre de rééducation",
#     "Sortie_Domicile",
#     "Sortie_Service hospitalier",
#     "Sortie_z MISSING_VALUE",
#     "Glasgow de sortie",
#     "Nombre de jours à l'hôpital",
#     "Durée de séjour en réa- si date de sortie connue, durée de séjour = (date sortie - date d entrée)- si date de sortie inconnue, d",
#     "Nombre de jours de VM",
#     "Procédure limitations de soins (LATA)",
# ]

# X = df.drop(to_drop, axis=1)


# prediction_death_imputed = PredictionTask(
#     df=df_imputed,
#     to_predict="Décès",
#     to_drop=[
#         "Date de décès (à l'hôpital après sortie de réanimation)",
#         "Cause du décès_A survécu",
#         "Cause du décès_Autre (précisez ci-dessous)",
#         "Cause du décès_Choc hémorragique",
#         "Cause du décès_Choc septique",
#         "Cause du décès_Défaillance multi-viscérale",
#         "Cause du décès_LATA",
#         "Cause du décès_Mort encéphalique",
#         "Cause du décès_Trauma cranien",
#         "Cause du décès_z MISSING_VALUE",
#         "Transfert secondaire, pourquoi ?_Pas de transfert",
#         "Transfert secondaire, pourquoi ?_Plateau technique insuffisant",
#         "Transfert secondaire, pourquoi ?_Rapprochement familial",
#         "Transfert secondaire, pourquoi ?_z MISSING_VALUE",
#         "Sortie_Autre réanimation",
#         "Sortie_Centre de rééducation",
#         "Sortie_Domicile",
#         "Sortie_Service hospitalier",
#         "Sortie_z MISSING_VALUE",
#         "Glasgow de sortie",
#         "Nombre de jours à l'hôpital",
#         "Durée de séjour en réa- si date de sortie connue, durée de séjour = (date sortie - date d entrée)- si date de sortie inconnue, d",
#         "Nombre de jours de VM",
#         "Procédure limitations de soins (LATA)",
#     ]
# )

# print(prediction_death_imputed.df)
# print(prediction_death_imputed.df_plain)
# print(prediction_death_imputed.X)
# print(prediction_death_imputed.y.value_counts())

task = NHIS_tasks['death_with_MV']

print(np.array(task._df.columns))
exit()

strat = Strategy(
    HistGradientBoostingClassifier(),
    split=lambda X, y: train_test_split(
        X, y, test_size=test_size, random_state=RS),#, stratify=y),
    cv=StratifiedShuffleSplit(n_splits=3, train_size=0.9, random_state=RS),
    param_space={
        'learning_rate': [0.1, 0.15, 0.2, 0.25],
    },
    search=lambda e, p, cv: GridSearchCV(e, p, scoring='recall', cv=cv)
)

train(task, strat)
exit()

X = prediction_death_imputed.X
y = prediction_death_imputed.y

# exit()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RS, stratify=y)
print(y_train.value_counts())
print(y_test.value_counts())
# exit()
# print(X_train)
# print(y_train)


# Estimators
# knn = KNeighborsClassifier(n_neighbors=8)
# boosting = GradientBoostingClassifier()
hist_boosting = HistGradientBoostingClassifier()
# SGD = SGDClassifier()

estimator = hist_boosting

print(estimator.get_params())
# exit()

param_grid = {
    'learning_rate': [0.1, 0.15, 0.2, 0.25],

}

cv = StratifiedShuffleSplit(n_splits=3, train_size=0.9, random_state=RS)
clf_CV = GridSearchCV(estimator, param_grid, scoring='recall', cv=cv)
clf_CV.fit(X_train, y_train)

print(f'Best parameters:\n{clf_CV.best_params_}')
means = clf_CV.cv_results_['mean_test_score']
stds = clf_CV.cv_results_['std_test_score']

print(means)
print(stds)

y_true, y_pred = y_test, clf_CV.predict(X_test)
print(classification_report(y_true, y_pred))


exit()


# Scoring
cv = StratifiedShuffleSplit(n_splits=3, train_size=0.9, random_state=RS)
scores = cross_val_score(estimator, X_train, y_train, cv=cv, n_jobs=-1, verbose=2)
print(scores)


exit()

estimator.fit(X_train, y_train)

y_pred = estimator.predict(X_test)

print(estimator.score(X_test, y_test))
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(estimator, X_test, y_test)

y_score = estimator.decision_function(X_test)

# Compute ROC curve and ROC area for each class
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
# for i in range(2):
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

# print(estimator.feature_importances_)

df_show = pd.DataFrame({
    'feature': list(X.columns),
    'importance': list(estimator.feature_importances_)
})

# print(df_show)

df_show.sort_values('importance', ascending=False, inplace=True)

# df_show = df_show.truncate(before=0, after=30)

f, ax = plt.subplots(figsize=(6, 15))

sns.set_color_codes("pastel")
sns.barplot(x="importance", y="feature", data=df_show.iloc[:50],
            label="Total", color="b")

plt.xscale('log')
plt.show()

