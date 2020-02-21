"""Class to dump task info and fit results into csv/yaml files."""
import pandas as pd
import numpy as np
import os
import yaml

results_folder = 'results/'

def _dump_yaml(data, filepath):
    data = listify(data)
    with open(filepath, 'w') as file:
        file.write(yaml.dump(data, allow_unicode=True))

def _dump_infos(item, filepath):
    """Dump the infos at the given place.

    Parameters
    ----------
    item : object
        Object implementing  a get_infos method returning the infos to dump.
    filepath : string
        Path to the place to store the infos.
    """
    data = item.get_infos()
    _dump_yaml(data, filepath)

def listify(d):
    """Convert all numpy arrays contained in the dict to lists.

    Parameters
    ----------
    d : dict

    Returns
    -------
    dict
        Same dict with numpy arrays converted to lists

    """
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, dict):
            d[k] = listify(v)
    return d


class DumpHelper:


    def __init__(self, task, strat):
        self.task = task
        self.strat = strat

        self.task_folder = (f'{results_folder}{self.task.meta.db}/'
                            f'{self.task.meta.name}/')
        self.strat_folder = f'{self.task_folder}{strat.name}/'

        self.dump_infos()

    def dump_infos(self):
        """Dump the infos of the task and strategy used."""

        os.makedirs(self.strat_folder, exist_ok=True)

        _dump_infos(self.task, f'{self.task_folder}task_infos.yml')
        _dump_infos(self.strat, f'{self.strat_folder}strat_infos.yml')

    def dump_prediction(self, y_pred, y_true):
        df = pd.DataFrame({
            'y_pred': y_pred,
            'y_true': y_true,
        })

        df.to_csv(self.strat_folder+'prediction.csv')

    def dump_best_params(self, best_params):
        _dump_yaml(best_params, self.strat_folder+'best_params.yml')

    def dump_cv_results(self, cv_results):
        _dump_yaml(cv_results, self.strat_folder+'cv_results.yml')

    def dump_roc(self, y_score, y_true):
        df = pd.DataFrame({
            'y_score': y_score,
            'y_true': y_true
        })

        df.to_csv(self.strat_folder+'roc.csv')

    # def dump_classification_report(self, report):
    #     _dump_yaml(report, self.strat_folder+'classification_report.yml')
