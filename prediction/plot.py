"""Class for plot results from saved csv."""
import pandas as pd
import matplotlib.pyplot as plt


results_folder = 'results/'


class PlotHelper:

    def __init__(self, task):
        self.task = task

        self.task_folder = (f'{results_folder}{self.task.meta.db}/'
                            f'{self.task.meta.name}/')

    def _path_from_strat(self, strat):
        name = strat if isinstance(strat, str) else strat.name
        return f'{self.task_folder}{name}/'

    def scatter_regression(self, strat):
        strat_path = self._path_from_strat(strat)

        df = pd.read_csv(f'{strat_path}regression_results.csv')

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
