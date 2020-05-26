"""Class for plot results from saved csv."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
import yaml
import os

from .DumpHelper import get_RS_tag


default_results_folder = 'results/'


def dict_equals(dict1, dict2, exclude=None):
    return True


class PlotHelper:

    def __init__(self, task, results_folder=default_results_folder, rename=dict()):
        if isinstance(task, str):
            db, task_name = task.split('/')
        else:
            db, task_name = task.meta.db, task.meta.name

        self.db = db
        self.task_tag = task
        self.task_name = task_name
        self.results_folder = results_folder
        self.task_folder = f'{results_folder}{db}/{task_name}/'

        self._rename = rename

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
        return strat_infos['classification']

    def n_folds(self, strat):
        strat_infos = self._load_yaml_from_filename(strat, 'strat_infos.yml')
        return strat_infos['outer_cv_params']['n_splits']

    def n_inner_splits(self, strat):
        strat_infos = self._load_yaml_from_filename(strat, 'strat_infos.yml')
        return strat_infos['inner_cv_params']['n_splits']

    def rename(self, string):
        def rename_aux(string):
            return self._rename.get(string, string)

        if isinstance(string, list):
            return [rename_aux(s) for s in string]

        return rename_aux(string)

    def plot_regression(self, strat, ax=None):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}prediction.csv')

        y_true = df['y_true']
        y_pred = df['y_pred']

        if ax is None:
            plt.figure()
            plt.title(f'Prediction on {self.task_tag} using\n{self.rename(strat)}')
            ax = plt.gca()

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

    def plot_roc(self, strat, true_class='1'):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}probas.csv')

        FPR, TPR = [], []
        ROC_AUC = []

        df_show = pd.DataFrame()

        if True:#not df.groupby('fold'):
            # units = None
            # y_score = df['y_score']
            # y_true = df['y_true']

            # fpr, tpr, _ = roc_curve(y_true, y_score)
            # roc_auc = auc(fpr, tpr)

            # df_show = pd.DataFrame({
            #         'fpr': fpr,#np.round(fpr, decimals=2),
            #         'tpr': tpr,
            #         'auc': roc_auc
            #     })

            units = None
            probas = df[f'proba_{true_class}']
            y_true = df['y_true']

            fpr, tpr, _ = roc_curve(y_true, probas)
            roc_auc = auc(fpr, tpr)

            df_show = pd.DataFrame({
                    'fpr': fpr,#np.round(fpr, decimals=2),
                    'tpr': tpr,
                    'auc': roc_auc
                })
        else:
            pass
            # units = 'fold'
            # for fold, df_gb in df.groupby('fold'):
            #     y_score = df_gb['y_score']
            #     y_true = df_gb['y_true']

            #     print(y_score.shape)

            #     fpr, tpr, _ = roc_curve(y_true, y_score)
            #     roc_auc = auc(fpr, tpr)

            #     df_show = pd.concat([
            #         df_show,
            #         pd.DataFrame({
            #             'fold': fold,
            #             'fpr': fpr,#np.round(fpr, decimals=2),
            #             'tpr': tpr,
            #             'auc': roc_auc
            #         })
            #     ])

            #     print(df_show)

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
        sns.lineplot(x='fpr', y='tpr', data=df_show, ci='sd', units=units, estimator=None)# estimator='median')
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

        scores = dict()

        try:
            df = pd.read_csv(f'{strat_path}prediction.csv')
        except FileNotFoundError:
            return scores

        # for _fold, df_gb in df.groupby('fold'):
        #     y_true = df_gb['y_true']
        #     y_pred = df_gb['y_pred']
        #     scores[_fold] = scorer(y_true, y_pred)
        #     print(scores[_fold])

        scores[None] = scorer(df['y_true'], df['y_pred'])

        return scores

    def plot_scores(self, strat, scorers, ax=None):
        if not isinstance(scorers, list):
            scorers = [scorers]

        scores = [self._scores(strat, scorer) for scorer in scorers]

        scores_labels = self.rename([scorer.__name__ for scorer in scorers])
        scores_values = [list(s.values()) for s in scores]

        if ax is None:
            plt.figure()
            ax = plt.gca()

        bplot = ax.boxplot(scores_values, labels=scores_labels,
                           vert=False, patch_artist=True)
        for patch, color in zip(bplot['boxes'],
                                ['C0', 'C1', 'C2', 'C3', 'C4']):
            patch.set_facecolor(color)

    def plot_scores_accross_strats(self, scorer, strats=None, ax=None):
        if strats is None:  # Get all strats of the given folder
            strats = next(os.walk(self.task_folder))[1]

        if not isinstance(strats, list):
            strats = [strats]

        strats = strats[:]
        strats.reverse()

        if ax is None:
            plt.figure()
            plt.title(f'Scores on {self.task_tag}')
            ax = plt.gca()

        scores = [self._scores(strat, scorer) for strat in strats]

        scores_labels = self.rename(strats)
        scores_values = [list(s.values()) for s in scores]
        # meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
        # medianprops = dict(linestyle='--', linewidth=2.5, color='firebrick')
        bplot = ax.boxplot(scores_values, labels=scores_labels,
                           vert=False, patch_artist=True, medianprops={'color': 'black'})
        if isinstance(scorer, str):
            scorer_name = scorer
        else:
            scorer_name = scorer.__name__

        ax.set_xlabel(scorer_name)
        ax.set_ylabel('Strategy')
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        colors.reverse()
        for patch, color in zip(bplot['boxes'],
                                colors):
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

    def retrieve_strat_folders(self, strat_name, RS, check=True):
        """
        Parameters
        ----------
        strat_name : str
            Name of the strategy
        RS : int or list of int
            Only the folders storing a strategy matching strat_name and having
            a random state matching the ones given in RS are retrieved.

        Returns
        -------
        list of str
            The name of the folders containing the strategy for the given
            random states.

        """
        if RS is None:
            return [strat_name]

        if not isinstance(RS, list):
            RS = [RS]

        if RS == []:
            return []

        strats = [f'{get_RS_tag(rs)}{strat_name}' for rs in RS]

        if not check:
            return strats

        # Check if those folders exists
        for s in strats:
            path = self._path_from_strat(s)
            if not os.path.exists(path):
                raise ValueError(f'{path} not found.')

        # Check if have same strat infos

        data1 = None
        data2 = self._load_yaml_from_filename(strats[0], 'strat_infos.yml')

        for i in range(1, len(strats)-1):
            data1 = data2
            data2 = self._load_yaml_from_filename(strats[i], 'strat_infos.yml')

            if not dict_equals(data1, data2, exclude='random_state'):
                raise ValueError(f'{strats[i-1]} and {strats[i]} have'
                                 f'different infos in strats_infos.yml')

        return strats

    def plot_learning_curve_one(self, strat, axes=None, color=None,
                                marker=None, line='-', RS=None):
        train_scores = []
        test_scores = []
        fit_times = []

        if strat:
            for s in self.retrieve_strat_folders(strat, RS):
                data = self._load_yaml_from_filename(s, 'learning_curve.yml')
                strat_infos = self._load_yaml_from_filename(s, 'strat_infos.yml')

                scoring = self.rename(strat_infos['learning_curve_params']['scoring'])

                if not data:
                    raise ValueError('No data found for learning curve.')

                for curve_data in data.values():
                    train_scores.append(np.array(curve_data['train_scores']))
                    test_scores.append(np.array(curve_data['test_scores']))
                    fit_times.append(np.array(curve_data['fit_times']))

            # Concatenate scores from different outter and inner folds
            train_scores = np.concatenate(train_scores, axis=1)
            test_scores = np.concatenate(test_scores, axis=1)
            fit_times = np.concatenate(fit_times, axis=1)

            train_sizes = np.array(curve_data['train_sizes_abs'])

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_xlabel("Training examples")

        if not strat:
            axes[0].set_ylim(0, 1)
            axes[0].set_xlim(0, 20000)

            axes[1].set_ylim(0, 100)
            axes[1].set_xlim(0, 20000)

            axes[2].set_ylim(0, 1)
            axes[2].set_xlim(0, 100)

        if strat:
            axes[0].set_ylabel(f"Score ({scoring})")
        else:
            axes[0].set_ylabel(f"Score")

        # Plot learning curve
        axes[0].grid()
        # axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
        #                      train_scores_mean + train_scores_std, alpha=0.1,
        #                      color=color)
        if strat:
            axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1,
                                color=color)
            # axes[0].plot(train_sizes, train_scores_mean, 'd-', color=color,
            #              label=f"T {strat}")
            axes[0].plot(train_sizes, test_scores_mean, f'{marker}{line}', color=color,
                        label=self.rename(strat))

        if strat:
            axes[0].legend(title='Gradient boosted trees', loc="best")

        axes[0].set_title(f"Prediction performance", x=0.5, y=0.94)
        # axes[0].title("my title", x=0.5, y=0.6)
        # axes[0].text(.5, .95, f'Prediction performance: {scoring}',
        #              horizontalalignment='center', transform=axes[0].transAxes)

        # Plot n_samples vs fit_times
        axes[1].grid()

        if strat:
            axes[1].plot(train_sizes, fit_times_mean, f'{marker}{line}', color=color)
            axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                                fit_times_mean + fit_times_std, alpha=0.1, color=color)

        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("Fit time (seconds)")
        # axes[1].set_title("Scalability of the model")
        axes[1].set_title("Computational cost of the models", x=0.5, y=0.94)
        # axes[1].set_title("Scalability of the model", x=0.5, y=0.94)
        # axes[]

        # Plot fit_time vs score
        axes[2].grid()

        if strat:
            axes[2].plot(fit_times_mean, test_scores_mean, f"{marker}{line}", color=color)
            axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1, color=color)

        axes[2].set_xlabel("Fit time (seconds)")

        if strat:
            axes[2].set_ylabel(f"Score ({scoring})")
        else:
            axes[2].set_ylabel(f"Score")

        # axes[2].set_title(f"Performance of the model: {scoring}")
        axes[2].set_title("Cost to benefit: time versus prediction performance", x=0.5, y=0.94)
        # axes[2].set_title(f"Performance of the model: {scoring}", x=0.5, y=0.94)
        # axes[]

        # fig = plt.gcf()
        # plt.suptitle("Title centered above all subplots")
        # fig.tight_layout()

        # plt.subplots_adjust(top=0.94)

        return plt

    def plot_learning_curve(self, strats, axes=None, truncate=(0., 0.), RS=None):
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
            # fig.tight_layout(rect=[0, 0.03, 1, 0.8])
            # fig.suptitle(f'Prediction of {self.rename(self.task_name)} '
            #              f'on {self.rename(self.db)}', y=1)

            # fig.tight_layout(rect=[0, 0.03, 1, 0.9])

            axes[1].annotate(f'Prediction of {self.rename(self.task_name)} '
                             f'on {self.rename(self.db)}', (0.5, 0.93),
                             xycoords='figure fraction', ha='center',
                             fontsize=13
                             )

        if not isinstance(strats, list):
            strats = [strats]

        colors = [f'C{i}' for i in range(10)]
        markers = ['o', '*', '.', 'x', '+', 'X', 'P', 'v', '^', '<', '>']
        for strat, c, m in zip(strats, colors, markers):
            if isinstance(strat, tuple) and len(strat) == 2:
                strat, strat_masked = strat
                if strat is not None:
                    self.plot_learning_curve_one(strat, axes=axes, color=c, marker=m, line='-', RS=RS)
                if strat_masked is not None:
                    self.plot_learning_curve_one(strat_masked, axes=axes, color=c, marker=m, line='--', RS=RS)
            else:
                self.plot_learning_curve_one(strat, axes=axes, color=c, marker=m, RS=RS)

        ylim0_inf, ylim0_sup = axes[0].get_ylim()
        delta0 = ylim0_sup-ylim0_inf
        ylim2_inf, ylim2_sup = axes[2].get_ylim()
        delta2 = ylim2_sup-ylim2_inf



        axes[0].set_ylim((ylim0_inf+truncate[0]*delta0, ylim0_sup-truncate[1]*delta0))
        axes[2].set_ylim((ylim2_inf+truncate[0]*delta2, ylim2_sup-truncate[1]*delta2))

    def get_strat_infos(self, strat):
        return self._load_yaml_from_filename(strat, 'strat_infos.yml')


