"""Class for plot results from saved csv."""
import pandas as pd
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

    def _path_from_strat(self, strat):
        name = strat if isinstance(strat, str) else strat.name
        return f'{self.task_folder}{name}/'

    def _load_yaml(self, strat, filename):
        strat_path = self._path_from_strat(strat)
        with open(strat_path+filename, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def _is_classification(self, strat):
        strat_infos = self._load_yaml(strat, 'strat_infos.yml')
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

        y_score, y_true = df['y_score'], df['y_true']

        fpr, tpr, _ = roc_curve(y_true, y_score)
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
        return self._load_yaml(strat, 'best_params.yml')

    def cv_results(self, strat):
        return self._load_yaml(strat, 'cv_results.yml')

    def plot_full_results(self, strat, labels=None):

        if self._is_classification(strat):
            self.plot_roc(strat)
            self.plot_confusion_matrix(strat)
            print(self.classification_report(strat))

        else:
            self.plot_regression(strat)

        print(self.best_params(strat))
        print(self.cv_results(strat))
