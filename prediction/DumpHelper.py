"""Class to dump task info and fit results into csv/yaml files."""
import pandas as pd
import numpy as np
import os
import yaml
import shutil

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

        # Create all necessary folders and ignore if already exist
        os.makedirs(self.strat_folder, exist_ok=True)

        # Clear strategy folder before dumping
        shutil.rmtree(self.strat_folder)

        # Create again an empty strategy folder
        os.makedirs(self.strat_folder, exist_ok=True)

        _dump_infos(self.task, f'{self.task_folder}task_infos.yml')
        _dump_infos(self.strat, f'{self.strat_folder}strat_infos.yml')

    def _filepath(self, filename):
        return f'{self.strat_folder}{filename}'

    @staticmethod
    def _load_content(filepath):
        """Used to load a yaml or csv file as dict or df respectively.

        Parameters
        ----------
        filepath : str
            Path of the file to load

        Returns
        -------
        dict or df
            Depends on the extension present in the path:
            If csv: returns df. If file doesn't exist, returns empty df.
            If yml: returns dict. If file doesn't exist, returns empty dict.

        """
        _, ext = os.path.splitext(filepath)

        if ext == '.yml':
            if not os.path.exists(filepath):
                return dict()
            # If exists
            with open(filepath, 'r') as file:
                return yaml.safe_load(file)

        elif ext == '.csv':
            if not os.path.exists(filepath):
                return pd.DataFrame()
            # If exists
            return pd.read_csv(filepath, index_col=0)

        else:
            raise ValueError(f'Extension {ext} not supported.')

    @staticmethod
    def _append_fold(filepath, data, fold=None):
        """Dump data (df or dict) in a file while keeping track of the fold #.

        Open or create a file and dump the data in it. If previous data has
        been dumped for the same fold number, the newer one will replace it.
        To keep track of the fold number, the data is stored in a dict with
        fold number as key (if data is dict). If data is dataframe, an extra
        fold columns is added and the dataframe is concatenated to the existing
        after removing the existing data having the same fold number.

        Parameters
        ----------
        filepath : str
            Path of the file to dump the data in.
        data : dict or pandas.DatFrame
            The data to dump.
        fold : int or None
            The fold number of the data.

        """
        content = DumpHelper._load_content(filepath)

        if isinstance(content, dict):
            content[fold] = data
            _dump_yaml(content, filepath)

        elif isinstance(content, pd.DataFrame):
            if not isinstance(data, pd.DataFrame):
                raise ValueError('Dumping to csv require pandas df as data.')

            # Remove previous results of same fold number
            if not content.empty:
                # Df is supposed to have fold column if not empty
                content = content[content.fold != fold]

            # Add new results
            data = data.copy()
            data['fold'] = fold
            content = pd.concat([content, data])

            content.to_csv(filepath)

    def _dump(self, data, filename, fold=None):
        """Wraper to dump data (dict or df) in a file"""
        filepath = self._filepath(filename)
        DumpHelper._append_fold(filepath, data, fold=fold)

    def dump_prediction(self, y_pred, y_true, fold=None):
        df = pd.DataFrame({
            'y_pred': y_pred,
            'y_true': y_true,
        })

        self._dump(df, 'prediction.csv', fold=fold)

    def dump_best_params(self, best_params, fold=None):
        self._dump(best_params, 'best_params.yml', fold=fold)


    def dump_cv_results(self, cv_results, fold=None):
        self._dump(cv_results, 'cv_results.yml', fold=fold)

    def dump_roc(self, y_score, y_true, fold=None):
        df = pd.DataFrame({
            'y_score': y_score,
            'y_true': y_true
        })

        self._dump(df, 'roc.csv', fold=fold)

    # def dump_classification_report(self, report):
    #     _dump_yaml(report, self.strat_folder+'classification_report.yml')
