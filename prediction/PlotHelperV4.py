"""Implement  PlotHelper for train4 results."""
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import seaborn as sns


class PlotHelperV4(object):
    """Plot the train4 results."""

    def __init__(self, root_folder, rename):
        """Init."""
        # Stepe 1: Check and register root_path
        root_folder = root_folder.rstrip('/')  # Remove trailing '/'
        self.root_folder = root_folder
        self._rename = rename

        if not os.path.isdir(self.root_folder):
            raise ValueError(f'No dir at specified path: {self.root_folder}')

        # Step 2: Get the relative path of all subdirs in the structure
        walk = os.walk(self.root_folder)

        abs_dirpaths = []
        abs_filepaths = []
        for root, dirnames, filenames in walk:
            if dirnames:
                for dirname in dirnames:
                    abs_dirpaths.append(f'{root}/{dirname}')
            if filenames:
                for filename in filenames:
                    abs_filepaths.append(f'{root}/{filename}')

        prefix = os.path.commonprefix(abs_dirpaths)
        rel_dir_paths = [os.path.relpath(p, prefix) for p in abs_dirpaths]
        rel_file_paths = [os.path.relpath(p, prefix) for p in abs_filepaths]

        # Step 3.1: Convert relative paths to nested python dictionnary (dirs)
        nested_dir_dict = {}

        for rel_dir_path in rel_dir_paths:
            d = nested_dir_dict
            for x in rel_dir_path.split('/'):
                d = d.setdefault(x, {})

        # Step 3.2: Convert relative paths to nested python dictionnary (files)
        nested_file_dict = {}

        for rel_file_path in rel_file_paths:
            d = nested_file_dict
            for x in rel_file_path.split('/'):
                d = d.setdefault(x, {})

        # Step 4: Fill the class attributes with the nested dicts
        self._nested_dir_dict = nested_dir_dict
        self._nested_file_dict = nested_file_dict

    def databases(self):
        """Return the databases found in the root folder."""
        ndd = self._nested_dir_dict
        return [db for db in ndd.keys() if self.tasks(db)]

    def tasks(self, db):
        """Return the tasks related to a given database."""
        ndd = self._nested_dir_dict
        return [t for t in ndd[db].keys() if self.methods(db, t)]

    def methods(self, db, t):
        """Return the methods used by a given task."""
        ndd = self._nested_dir_dict
        return [m for m in ndd[db][t] if self._is_valid_method(db, t, m)]

    def _is_valid_method(self, db, t, m):
        # Must contain either Classification or Regression
        if 'Regression' not in m and 'Classification' not in m:
            return False

        path = f'{self.root_folder}/{db}/{t}/{m}/'
        if not os.path.exists(path):
            return False

        _, _, filenames = next(os.walk(path))

        return len(filenames) > 1  # always strat_infos.yml in m folder

    def short_method_name(self, m):
        """Return the suffix from the method name."""
        s = m.split('Regression')
        if len(s) == 1:
            s = m.split('Classification')
        if len(s) == 1:
            raise ValueError(f'Unable to find short method name of {m}')

        return s[1]

    def rename(self, x):
        """Rename a string."""
        return self._rename.get(x, x)

    def existing_methods(self):
        """Return the existing methods used by the tasks in the root_folder."""
        methods = set()

        for db in self.databases():
            for t in self.tasks(db):
                for m in self.methods(db, t):
                    s = m.split('Regression')
                    if len(s) == 1:
                        s = m.split('Classification')
                    suffix = s[1]
                    methods.add(suffix)

        return methods

    def existing_sizes(self):
        """Return the existing training sizes found in the root_folder."""
        sizes = set()
        nfd = self._nested_file_dict

        for db in self.databases():
            for t in self.tasks(db):
                for m in self.methods(db, t):
                    for filename in nfd[db][t][m].keys():
                        s = filename.split('_prediction.csv')
                        if len(s) > 1:  # Pattern found
                            size = s[0]  # size is the first part
                            sizes.add(size)

        sizes = list(sizes)
        sizes.sort(key=int)

        return sizes

    def score(self, db, t, m, size, true_class='1', mean=False):
        """Compute score of a given db, task, method, size.

        Parameters
        ----------
        db : str
            Name of db folder.
        t : str
            Name of task folder.
        m : str
            Name of method folder.
        size : str
            Size of the train set to load.
        true_class : str
            Name of the true class (if classification).
        mean : bool
            Whether to compute the mean of the score or return all scores.

        Return
        ------
        scores : dict or float
            If mean is False: return dict of scores of each fold.
            Else, return a float, mean of scores on all folds.

        """
        print(f'Compute score of {db}/{t}/{m}/{size}')
        method_path = f'{self.root_folder}/{db}/{t}/{m}/'
        strat_infos_path = method_path+'strat_infos.yml'

        if not os.path.exists(strat_infos_path):
            raise ValueError(f'Path {strat_infos_path} doesn\'t exist.')

        with open(strat_infos_path, 'r') as file:
            strat_infos = yaml.safe_load(file)

        is_classif = strat_infos['classification']

        if is_classif:
            scorer = roc_auc_score
            df_path = f'{method_path}{size}_probas.csv'
            y_true_col = 'y_true'
            y_col = f'proba_{true_class}'
        else:
            scorer = r2_score
            df_path = f'{method_path}{size}_prediction.csv'
            y_true_col = 'y_true'
            y_col = 'y_pred'

        df = pd.read_csv(df_path)

        scores = dict()

        for fold, df_gb in df.groupby('fold'):
            y_true = df_gb[y_true_col]
            y = df_gb[y_col]

            score = scorer(y_true, y)
            scores[fold] = score

        if mean:
            scores = np.mean(list(scores.values()))

        return scores

    def relative_scores(self, db, t, methods, size):
        """Get relative scores of given methods for (db, task, size).

        Size and methods must exist (not check performed).
        """
        scores = {m: self.score(db, t, m, size, mean=True) for m in methods}
        mean = np.mean(list(scores.values()))

        relative_scores = {m: (s - mean)/mean for m, s in scores.items()}

        return relative_scores

    def availale_methods_by_size(self, db, t, size):
        """Get the methods available for a given size."""
        methods = self.methods(db, t)
        nfd = self._nested_file_dict
        available_methods = set()

        for m in methods:
            for filename in nfd[db][t][m]:
                if f'{size}_' in filename:
                    available_methods.add(m)

        return available_methods

    def _y(self, m, db, n_m, n_db):
        """Get y-axis position given # db and # m and n_db and n_m."""
        assert 0 <= m < n_m
        assert 0 <= db < n_db
        return n_m - m - (db+1)/(n_db+1)

    def plot(self, method_order=None, db_order=None):
        """Plot the full available results."""
        dbs = self.databases()
        existing_methods = self.existing_methods()
        existing_sizes = self.existing_sizes()

        print(existing_sizes)

        # Convert existing methods from short name to renamed name
        existing_methods = {self.rename(m) for m in existing_methods}

        # Reorder the methods if asked
        temp = []
        if method_order:
            # Add methods with order in order
            for m in method_order:
                if m in existing_methods:
                    temp.append(m)

        # Add methods without order
        for m in existing_methods:
            if m not in temp:
                temp.append(m)

        method_order_list = temp
        method_order = {m: i for i, m in enumerate(method_order_list)}

        # Reorder the db if asked
        temp = []
        if db_order:
            for db in db_order:
                if db in db_order:
                    temp.append(db)

        for db in dbs:
            if db not in temp:
                temp.append(db)

        db_order_list = temp
        db_order_list_renamed = [self.rename(db) for db in db_order_list]
        db_order = {db: i for i, db in enumerate(db_order_list)}

        n_db = len(dbs)
        n_m = len(existing_methods)
        n_sizes = len(existing_sizes)

        rows = []
        for i, size in enumerate(existing_sizes):
            for db in dbs:
                for t in self.tasks(db):
                    methods = self.availale_methods_by_size(db, t, size)
                    rel_scores = self.relative_scores(db, t, methods, size)
                    for m, s in rel_scores.items():
                        short_m = self.short_method_name(m)
                        renamed_m = self.rename(short_m)
                        priority = method_order[renamed_m]
                        db_id = db_order[db]
                        rdb = self.rename(db)
                        y = self._y(priority, db_id, n_m, n_db)
                        rows.append(
                            (size, db, rdb, t, priority, m, renamed_m, s, y)
                        )

        cols = ['n', 'db', 'Databases', 't', 'p', 'm', 'rm', 'relative_score', 'y']

        df = pd.DataFrame(rows, columns=cols)
        print(df)

        fig, axes = plt.subplots(nrows=1, ncols=n_sizes, figsize=(20, 6))
        plt.subplots_adjust(
            left=0.075,
            right=0.95,
            bottom=0.1,
            top=0.95,
            wspace=0.05
        )

        for i, size in enumerate(existing_sizes):
            ax = axes[i]
            idx = df.index[df['n'] == size]
            df_gb = df.loc[idx]

            twinx = ax.twinx()
            twinx.set_ylim(0, n_m)
            twinx.yaxis.set_visible(False)

            ax.axvline(0, ymin=0, ymax=n_m, color='gray', zorder=0)

            # Boxplot
            sns.set_palette(sns.color_palette('gray'))
            sns.boxplot(x='relative_score', y='rm', data=df_gb, orient='h',
                        ax=ax, order=method_order_list)

            # Scatter plot
            sns.set_palette(sns.color_palette('colorblind'))
            sns.scatterplot(x='relative_score', y='y', hue='Databases',
                            data=df_gb, ax=twinx,
                            hue_order=db_order_list_renamed,
                            style='Databases',
                            markers=['o', '^', 'v', 's'],
                            s=75,
                            )

            if i > 0:  # if not the first axis
                ax.yaxis.set_visible(False)
                twinx.get_legend().remove()
            ax.set_title(f'n={size}')
            ax.set_xlabel(self.rename(ax.get_xlabel()))
            ax.set_ylabel(None)
            ax.set_axisbelow(True)
            ax.grid(True, axis='x')

        return fig
