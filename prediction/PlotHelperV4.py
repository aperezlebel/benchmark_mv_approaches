"""Implement  PlotHelper for train4 results."""
import os
import yaml
import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score, roc_auc_score
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import seaborn as sns


class PlotHelperV4(object):
    """Plot the train4 results."""

    def __init__(self, root_folder, rename, reference_method=None):
        """Init."""
        # Stepe 1: Check and register root_path
        root_folder = root_folder.rstrip('/')  # Remove trailing '/'
        self.root_folder = root_folder
        self._rename = rename
        self._reference_method = reference_method

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

        rel_dir_paths = [os.path.relpath(p, root_folder) for p in abs_dirpaths]
        rel_file_paths = [os.path.relpath(p, root_folder) for p in abs_filepaths]

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

        # Step 5: Compute scores for reference method
        if self._reference_method:
            self._reference_score = dict()
            for size in self.existing_sizes():
                scores_size = self._reference_score.get(size, dict())
                for db in self.databases():
                    scores = scores_size.get(db, dict())
                    for t in self.tasks(db):
                        score = None
                        for m in self.availale_methods_by_size(db, t, size):
                            if self._is_reference_method(m):
                                score = self.score(db, t, m, size, mean=True)
                                break
                        scores[t] = score
                    scores_size[db] = scores

                self._reference_score[size] = scores_size

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

    def _is_reference_method(self, m):
        if not hasattr(self, '_reference_method'):
            return False

        m = self.short_method_name(m)
        return self.rename(m) == self._reference_method

    def short_method_name(self, m):
        """Return the suffix from the method name."""
        s = m.split('Regression')
        if len(s) == 1:
            s = m.split('Classification')
        if len(s) == 1:
            raise ValueError(f'Unable to find short method name of {m}')

        return s[1]

    @staticmethod
    def rename_str(rename_dict, s):
        return rename_dict.get(s, s)

    def rename(self, s):
        """Rename a string."""
        return self.rename_str(self._rename, s)

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
            scorer_name = 'roc_auc_score'
            df_path = f'{method_path}{size}_probas.csv'
            y_true_col = 'y_true'
            y_col = f'proba_{true_class}'
        else:
            scorer = r2_score
            scorer_name = 'r2_score'
            df_path = f'{method_path}{size}_prediction.csv'
            y_true_col = 'y_true'
            y_col = 'y_pred'

        try:
            df = pd.read_csv(df_path)
        except pd.errors.EmptyDataError:
            if mean:
                return None
            return dict()

        scores = dict()

        for fold, df_gb in df.groupby('fold'):
            y_true = df_gb[y_true_col]
            y = df_gb[y_col]

            score = scorer(y_true, y)
            scores[fold] = score

        if mean:
            scores = np.mean(list(scores.values()))

        return scores, scorer_name

    def absolute_scores(self, db, t, methods, size, mean=True):
        """Get absolute scores of given methods for (db, task, size).

        Size and methods must exist (not check performed).
        """
        return {m: self.score(db, t, m, size, mean=mean) for m in methods}

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

    @staticmethod
    def _y(m, db, n_m, n_db):
        """Get y-axis position given # db and # m and n_db and n_m."""
        assert 0 <= m < n_m
        assert 0 <= db < n_db
        return n_m - m - (db+1)/(n_db+1)

    @staticmethod
    def _add_relative_score(df, reference_method=None):
        dfgb = df.groupby(['size', 'db', 'task'])

        def rel_score(df):
            if reference_method is None:  # use mean
                ref_score = df['score'].mean()
            else:
                methods = df['method']
                ref_score = float(df.loc[methods == reference_method, 'score'])
                df['reference'] = reference_method
                df['referece_score'] = ref_score

            df['relative_score'] = df['score'] - ref_score

            return df

        return dfgb.apply(rel_score)

    def dump(self, filepath):
        """Scan results in result_folder and compute scores."""
        existing_sizes = self.existing_sizes()
        rows = []
        for i, size in enumerate(existing_sizes):
            for db in self.databases():
                for t in self.tasks(db):
                    methods = self.availale_methods_by_size(db, t, size)
                    abs_scores = self.absolute_scores(db, t, methods, size,
                                                      mean=False)
                    for m, (scores, scorer) in abs_scores.items():
                        for fold, s in scores.items():
                            if s is None:
                                print(f'Skipping {db}/{t}/{m}')
                                continue
                            if 'Regression' in m:
                                tag = m.split('Regression')[0]
                            elif 'Classification' in m:
                                tag = m.split('Classification')[0]
                            else:
                                tag = 'Error while retrieving tag'
                                print(tag)
                            params = re.search('RS(.+?)_T(.+?)_', tag)
                            T = params.group(2)
                            short_m = self.short_method_name(m)
                            renamed_m = self.rename(short_m)
                            rows.append(
                                (size, db, t, renamed_m, T, fold, s, scorer)
                            )

        cols = ['size', 'db', 'task', 'method', 'trial', 'fold', 'score', 'scorer']

        df = pd.DataFrame(rows, columns=cols).astype({
            'size': int,
            'trial': int,
            'fold': int,
            })
        df.sort_values(by=['size', 'db', 'task', 'method', 'trial', 'fold'],
                       inplace=True, ignore_index=True)
        print(df)

        df.to_csv(filepath)

    @staticmethod
    def plot(filepath, db_order=None, method_order=None, reference_method=None, rename=dict()):
        """Plot the full available results."""
        df = pd.read_csv(filepath)

        sizes = list(df['size'].unique())
        n_sizes = len(sizes)
        dbs = list(df['db'].unique())
        n_dbs = len(dbs)
        methods = list(df['method'].unique())
        n_methods = len(methods)

        # Check db_order
        if db_order is None:
            db_order = dbs
            # db_order = {db: i for i, db in enumerate(dbs)}
        elif set(dbs) != set(db_order):
            raise ValueError(f'Db order missmatch existing ones {dbs}')

        # Check method order
        if method_order is None:
            method_order = methods
            # method_order = {m: i for i, m in enumerate(method_order_list)}
        elif set(methods) != set(method_order):
            raise ValueError(f'Method order missmatch existing ones {methods}')

        # Agregate accross folds by averaging
        dfgb = df.groupby(['size', 'db', 'task', 'method', 'trial'])
        df = dfgb.agg({'score': 'mean'})

        # Agregate accross trials by averaging
        dfgb = df.groupby(['size', 'db', 'task', 'method'])
        df = dfgb.agg({'score': 'mean'})

        # Reset index to addlevel of the multi index to the columns of the df
        df = df.reset_index()

        # Compute and add relative score
        df = PlotHelperV4._add_relative_score(df, reference_method=reference_method)

        # Add y position for plotting
        def _add_y(row):
            method_idx = method_order.index(row['method'])
            db_idx = db_order.index(row['db'])
            return PlotHelperV4._y(method_idx, db_idx, n_methods, n_dbs)

        df['y'] = df.apply(_add_y, axis=1)

        # Add a renamed column for databases for plotting
        df['Database'] = df.apply(lambda row: PlotHelperV4.rename_str(rename, row['db']), axis=1)

        # Print df with all its edits
        print(df)

        fig, axes = plt.subplots(nrows=1, ncols=n_sizes, figsize=(20, 6))
        plt.subplots_adjust(
            left=0.075,
            right=0.95,
            bottom=0.1,
            top=0.95,
            wspace=0.05
        )

        markers = ['o', '^', 'v', 's']
        db_markers = {db: markers[i] for i, db in enumerate(db_order)}


        for i, size in enumerate(sizes):
            ax = axes[i]
            idx = df.index[df['size'] == size]
            df_gb = df.loc[idx]

            twinx = ax.twinx()
            twinx.set_ylim(0, n_methods)
            twinx.yaxis.set_visible(False)

            ax.axvline(0, ymin=0, ymax=n_methods, color='gray', zorder=0)

            # Boxplot
            sns.set_palette(sns.color_palette('gray'))
            sns.boxplot(x='relative_score', y='method', data=df_gb, orient='h',
                        ax=ax, order=method_order, showfliers=False)

            # Scatter plot
            sns.set_palette(sns.color_palette('colorblind'))
            sns.scatterplot(x='relative_score', y='y', hue='Database',
                            data=df_gb, ax=twinx,
                            hue_order=db_order,
                            style='Database',
                            markers=db_markers,#['o', '^', 'v', 's'],
                            s=75,
                            )

            if i > 0:  # if not the first axis
                ax.yaxis.set_visible(False)
                twinx.get_legend().remove()
            ax.set_title(f'n={size}')
            ax.set_xlabel(rename.get(ax.get_xlabel(), ax.get_xlabel()))
            ax.set_ylabel(None)
            ax.set_axisbelow(True)
            ax.grid(True, axis='x')

        return fig
