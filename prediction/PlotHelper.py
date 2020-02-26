"""Class for plot results from saved csv."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
import yaml

results_folder = 'results/'


class PlotHelper:

    def __init__(self, task):
        if isinstance(task, str):
            db, task_name = task.split('/')
        else:
            db, task_name = task.meta.db, task.meta.name

        self.task_folder = f'{results_folder}{db}/{task_name}/'

        self._load_infos()

    def _load_infos(self):
        self.task_infos = self._load_yaml(self.task_folder+'task_infos.yml')
        self.features = np.array(self._load_yaml(self.task_folder+'features.yml'))

    def _path_from_strat(self, strat):
        name = strat if isinstance(strat, str) else strat.name
        return f'{self.task_folder}{name}/'

    def _load_yaml(self, filepath):
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def _load_yaml_from_filename(self, strat, filename):
        strat_path = self._path_from_strat(strat)
        return self._load_yaml(strat_path+filename)

    def _is_classification(self, strat):
        strat_infos = self._load_yaml_from_filename(strat, 'strat_infos.yml')
        print(strat_infos)
        return strat_infos['classification']

    def plot_regression(self, strat):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}prediction.csv')

        y_true = df['y_true']
        y_pred = df['y_pred']

        ax = plt.axes()
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                '--r', linewidth=2)
        ax.scatter(y_true, y_pred, alpha=0.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([y_true.min(), y_true.max()])
        ax.set_ylim([y_true.min(), y_true.max()])
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

    def plot_roc(self, strat):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}roc.csv')

        FPR, TPR = [], []
        ROC_AUC = []

        df_show = pd.DataFrame()

        for fold, df_gb in df.groupby('fold'):
            y_score = df_gb['y_score']
            y_true = df_gb['y_true']

            print(y_score.shape)

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            df_show = pd.concat([
                df_show,
                pd.DataFrame({
                    'fold': fold,
                    'fpr': fpr,#np.round(fpr, decimals=2),
                    'tpr': tpr,
                    'auc': roc_auc
                })
            ])

            # FPR.append(fpr.reshape((-1, 1)))
            # TPR.append(tpr.reshape((-1, 1)))
            # ROC_AUC.append(roc_auc)

            # print(fpr.shape)

        # print(FPR[0].shape)
        # exit()
        # FPR = np.concatenate(FPR, axis=1)
        # TPR = np.concatenate(TPR, axis=1)
        # ROC_AUC = np.array(ROC_AUC)

        print(df_show)
        sns.lineplot(x='fpr', y='tpr', data=df_show, ci='sd', units='fold', estimator=None)# estimator='median')
        return
        # exit()


        # y_score, y_true = df['y_score'], df['y_true']

        # fpr, tpr, _ = roc_curve(y_true, y_score)
        # roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        for i in range(FPR.shape[0]):
            fpr, tpr = FPR[i, :], TPR[i, :]
            plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.3f)' % ROC_AUC[i])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    def classification_report(self, strat, output_dict=False):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}prediction.csv')

        y_true = df['y_true']
        y_pred = df['y_pred']

        return classification_report(y_true, y_pred, output_dict=output_dict)


    def confusion_matrix(self, strat, labels=None):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}prediction.csv')

        y_true = df['y_true']
        y_pred = df['y_pred']

        matrix = confusion_matrix(y_true, y_pred, labels=labels)

        return matrix

    def plot_confusion_matrix(self, strat, labels=None):
        matrix = self.confusion_matrix(strat, labels=labels)

        plt.figure(figsize=(6, 4))
        if labels is None:
            labels = 'auto'
        sns.heatmap(matrix, cmap='Greens', annot=True, xticklabels=labels, yticklabels=labels)

    def best_params(self, strat):
        return self._load_yaml_from_filename(strat, 'best_params.yml')

    def cv_results(self, strat):
        return self._load_yaml_from_filename(strat, 'cv_results.yml')

    def plot_full_results(self, strat, labels=None):

        if self._is_classification(strat):
            self.plot_roc(strat)
            self.plot_confusion_matrix(strat)
            print(self.classification_report(strat))

        else:
            self.plot_regression(strat)

        print(self.best_params(strat))
        print(self.cv_results(strat))

    def _scores(self, strat, scorer, fold=None):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}prediction.csv')

        scores = dict()

        for _fold, df_gb in df.groupby('fold'):
            y_true = df_gb['y_true']
            y_pred = df_gb['y_pred']
            scores[_fold] = scorer(y_true, y_pred)

        return scores

    def plot_scores(self, strat, scorers):
        if not isinstance(scorers, list):
            scorers = [scorers]

        scores = [self._scores(strat, scorer) for scorer in scorers]
        print(scores)

        scores_labels = [scorer.__name__ for scorer in scorers]
        scores_values = [list(s.values()) for s in scores]

        plt.figure(figsize=(5, 4))
        bplot = plt.boxplot(scores_values, labels=scores_labels,
                            vert=False, patch_artist=True)
        for patch, color in zip(bplot['boxes'],
                                ['C0', 'C1', 'C2', 'C3']):
            patch.set_facecolor(color)

    def plot_importances(self, strat, tol=5e-3):
        """Boxplot feature importances. Importance below tol are ignored."""
        data = self._load_yaml_from_filename(strat, 'importance.yml')

        if not data:
            raise ValueError('Missing feature importances.')

        # Retrieve the importances mean of each fold
        importances = [imp['importances_mean'] for imp in data.values()]
        importances = np.array(importances)

        # Sort according to mean
        importances_mean = np.mean(importances, axis=0)
        sorted_idx = importances_mean.argsort()

        # Keep only the significant ones (> tol)
        selected_idx = importances_mean[sorted_idx] > tol
        sorted_idx = sorted_idx[selected_idx]

        plt.boxplot(importances[:, sorted_idx], vert=False,
                    labels=self.features[sorted_idx])
        plt.title(f'Feature importances above {tol}')

