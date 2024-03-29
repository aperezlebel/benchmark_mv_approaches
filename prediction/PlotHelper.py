"""Implement  PlotHelper for train4 results."""
import os
import re
import shutil
from decimal import Decimal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import r2_score, roc_auc_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

from prediction.df_utils import aggregate, assert_equal, get_ranks_tab


class PlotHelper(object):
    """Plot the train4 results."""

    def __init__(self, root_folder, rename={}, reference_method=None):
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
                return None, None
            return dict()
        except FileNotFoundError:
            if mean:
                return None, None
            return dict(), None

        scores = dict()

        for fold, df_gb in df.groupby('fold'):
            y_true = df_gb[y_true_col]
            y = df_gb[y_col]

            score = scorer(y_true, y)
            scores[fold] = score

        if mean:
            scores = np.mean(list(scores.values()))

        return scores, scorer_name

    def times(self, db, t, m, size):
        """Compute time of a given db, task, method, size.

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

        Return
        ------
        imputation_times : dict
            Dict of imputation time of each fold.
        tuning_times : dict
            Dict of tuning time of each fold.


        """
        print(f'Compute time of {db}/{t}/{m}/{size}')
        df_path = f'{self.root_folder}/{db}/{t}/{m}/{size}_times.csv'
        try:
            df = pd.read_csv(df_path)
        except pd.errors.EmptyDataError:
            return None, None

        cols = df.columns
        imputation_wct = dict()
        tuning_wct = dict()
        imputation_pt = dict()
        tuning_pt = dict()

        for fold, df_gb in df.groupby('fold'):

            if 'imputation_PT' in cols and 'imputation_WCT' in cols:
                imputation_wct[fold] = float(df_gb['imputation_WCT'])
                tuning_wct[fold] = float(df_gb['tuning_WCT'])
                imputation_pt[fold] = float(df_gb['imputation_PT'])
                tuning_pt[fold] = float(df_gb['tuning_PT'])

            else:
                assert len(df_gb['imputation']) == 1
                assert len(df_gb['tuning']) == 1
                imputation_wct[fold] = float(df_gb['imputation'])
                tuning_wct[fold] = float(df_gb['tuning'])

        return {
            'imputation_WCT': imputation_wct,
            'tuning_WCT': tuning_wct,
            'imputation_PT': imputation_pt,
            'tuning_PT': tuning_pt
        }

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
    def _add_relative_value(df, value, how, reference_method=None):
        assert value in df.columns
        dfgb = df.groupby(['size', 'db', 'task'])

        def rel_value(df):
            if reference_method is None:  # use mean
                ref_value = df[value].mean()
            else:
                methods = df['method']
                ref_value = float(df.loc[methods == reference_method, value])
                df['reference'] = reference_method
                df[f'referece_{value}'] = ref_value

            if how == 'mean':
                normalization = df[value].mean()
            elif how == 'std':
                normalization = df[value].std()
            elif how == 'no-norm':
                normalization = 1
            elif how == 'abs':
                ref_value = 0
                normalization = 1
            elif how == 'log':
                normalization = ref_value
                ref_value = 0

            df[f'relative_{value}'] = (df[value] - ref_value)/normalization

            return df

        return dfgb.apply(rel_value)

    def _export(self, db, id):
        sizes = self.existing_sizes()
        dump_dir = f'sandbox/compare_{id}/{db}/'
        if os.path.isdir(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir, exist_ok=True)

        for t in self.tasks(db):
            for size in sizes:
                methods = self.availale_methods_by_size(db, t, size)
                for m in methods:
                    subpath = f'{db}/{t}/{m}/{size}_prediction.csv'
                    df_path = f'{self.root_folder}/{subpath}'
                    print(df_path)
                    r_subpath = subpath.replace('/', '_')
                    shutil.copyfile(df_path, dump_dir+r_subpath)

    def dump(self, filepath, n=None):
        """Scan results in result_folder and compute scores."""
        existing_sizes = self.existing_sizes()
        if n is not None and str(n) not in existing_sizes:
            raise ValueError(f'Asked n={n} not in existing sizes: {existing_sizes}')
        elif n is not None:
            sizes = [str(n)]
        else:
            sizes = existing_sizes

        rows = []
        for i, size in enumerate(sizes):
            for db in self.databases():
                for t in self.tasks(db):
                    methods = self.availale_methods_by_size(db, t, size)
                    abs_scores = self.absolute_scores(db, t, methods, size,
                                                      mean=False)
                    for m, (scores, scorer) in abs_scores.items():
                        times = self.times(db, t, m, size)

                        method_path = f'{self.root_folder}/{db}/{t}/{m}/'

                        # Load strat info
                        strat_infos_path = method_path+'strat_infos.yml'
                        if not os.path.exists(strat_infos_path):
                            raise ValueError(f'Path {strat_infos_path} doesn\'t exist.')
                        with open(strat_infos_path, 'r') as file:
                            strat_infos = yaml.safe_load(file)
                        is_classif = strat_infos['classification']
                        task_type = 'Classification' if is_classif else 'Regression'

                        # Load task info
                        task_infos_path = method_path+'task_infos.yml'
                        if not os.path.exists(task_infos_path):
                            raise ValueError(f'Path {task_infos_path} doesn\'t exist.')
                        with open(task_infos_path, 'r') as file:
                            task_infos = yaml.safe_load(file)
                        X_shape = task_infos['X.shape']
                        # Convert representation of tuple (str) to tuple
                        X_shape = X_shape.replace('(', '')
                        X_shape = X_shape.replace(')', '')
                        X_shape = X_shape.replace(' ', '')
                        n, p = X_shape.split(',')

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
                            selection = 'ANOVA' if '_pvals' in t else 'manual'
                            imp_wct = times['imputation_WCT'][fold]
                            tun_wct = times['tuning_WCT'][fold]
                            imp_pt = times['imputation_PT'].get(fold, None)
                            tun_pt = times['tuning_PT'].get(fold, None)

                            rows.append(
                                (size, db, t, renamed_m, T, fold, s, scorer, selection, n, p, task_type, imp_wct, tun_wct, imp_pt, tun_pt)
                            )

        cols = ['size', 'db', 'task', 'method', 'trial', 'fold', 'score', 'scorer', 'selection', 'n', 'p', 'type', 'imputation_WCT', 'tuning_WCT', 'imputation_PT', 'tuning_PT']

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
    def get_task_description(filepath):
        """Build and dump a csv that will explain each task once completed."""
        if not isinstance(filepath, pd.DataFrame):
            df = pd.read_csv(filepath, index_col=0)
        else:
            df = filepath

        df = aggregate(df, 'score')

        print(df)

        dfgb = df.groupby(['db', 'task'])
        df = dfgb.agg({
            'score': 'mean',
            # 'n_trials': 'sum',
            # 'n_folds': 'sum',
            'scorer': assert_equal,  # first and assert equal
            'selection': assert_equal,
            'n': assert_equal,
            'p': assert_equal,
            'type': assert_equal,
            'imputation_WCT': 'mean',
            'tuning_WCT': 'mean',
            'imputation_PT': 'mean',
            'tuning_PT': 'mean',
        })

        df = df.reset_index()

        # Sum times
        df['total_PT'] = df['imputation_PT'].fillna(0) + df['tuning_PT']
        df['total_WCT'] = df['imputation_WCT'].fillna(0) + df['tuning_WCT']

        # Round scores
        df['imputation_PT'] = df['imputation_PT'].astype(int)
        df['imputation_WCT'] = df['imputation_WCT'].astype(int)
        df['tuning_PT'] = df['tuning_PT'].astype(int)
        df['tuning_WCT'] = df['tuning_WCT'].astype(int)
        df['score'] = df['score'].round(2)
        df['total_PT'] = df['total_PT'].astype(int)
        df['total_WCT'] = df['total_WCT'].astype(int)

        df = df.drop(['imputation_WCT', 'tuning_WCT', 'total_WCT'], axis=1)

        # Rename values in columns
        df['selection'] = df['selection'].replace({
            'ANOVA': 'A',
            'manual': 'M',
        })
        df['scorer'] = df['scorer'].replace({
            'roc_auc_score': 'AUC',
            'r2_score': 'R2',
        })
        df['type'] = df['type'].replace({
            'Classification': 'C',
            'Regression': 'R',
        })

        # Rename columns
        rename_dict = {
            'db': 'Database',
            'imputation_PT': 'Imputation time (s)',
            'tuning_PT': 'Tuning time (s)',
            'total_PT': 'Total time (s)',
            'n': 'n',
            'p': 'p',
        }

        # Capitalize
        for f in df.columns:
            if f not in rename_dict:
                rename_dict[f] = f.capitalize()

        df = df.rename(rename_dict, axis=1)

        # Create multi index
        df = df.set_index(['Database', 'Task'])

        # Read desciptions from file
        description_filepath = 'scores/descriptions.csv'
        if os.path.exists(description_filepath):
            desc = pd.read_csv(description_filepath, index_col=[0, 1])
            df = pd.concat([df, desc], axis=1)
        else:
            desc = pd.DataFrame(
                {'Target': 'Explain target here.', 'Description': 'Write task description here.'},
                index=df.index, columns=['Target', 'Description']
            )
            desc.to_csv(description_filepath)

        return df

    @staticmethod
    def _plot(filepath, value, how, xticks_dict=None, xlims=None, db_order=None,
              method_order=None, rename=dict(), reference_method=None,
              figsize=None, legend_bbox=None, xlabel=None, symbols=None, comments=None,
              only_full_samples=True, y_labelsize=18, broken_axis=None, comments_align=None,
              comments_spacing=0.025, colors=None, ref_vline=None, non_ref_vline=False):
        """Plot the full available results."""
        if not isinstance(filepath, pd.DataFrame):
            df = pd.read_csv(filepath, index_col=0)
        else:
            df = filepath

        assert value in df.columns

        sizes = list(df['size'].unique())
        n_sizes = len(sizes)
        dbs = list(df['db'].unique())
        n_dbs = len(dbs)
        methods = list(df['method'].unique())

        # Check db_order
        if db_order is None:
            db_order = dbs
        elif set(dbs) != set(db_order):
            raise ValueError(f'Db order missmatch existing ones {dbs}')

        # Check method order
        if method_order is None:
            method_order = methods
        elif set(method_order).issubset(set(methods)):
            df = df[df['method'].isin(method_order)]
        else:
            raise ValueError(f'Method order missmatch existing ones {methods}')

        methods = list(df['method'].unique())
        n_methods = len(methods)

        df = aggregate(df, value)

        # Compute and add relative value
        df = PlotHelper._add_relative_value(df, value, how,
                                              reference_method=reference_method)

        # Add y position for plotting
        def _add_y(row):
            method_idx = method_order.index(row['method'])
            db_idx = db_order.index(row['db'])
            return PlotHelper._y(method_idx, db_idx, n_methods, n_dbs)

        df['y'] = df.apply(_add_y, axis=1)

        # Add a renamed column for databases for plotting
        df['Database'] = df.apply(lambda row: PlotHelper.rename_str(rename, row['db']), axis=1)

        # Print df with all its edits
        print(df)

        matplotlib.rcParams.update({
            'font.size': 10,
            'legend.fontsize': 16,
            'legend.title_fontsize': 18,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': y_labelsize,
            # 'mathtext.fontset': 'stixsans',
            'font.family': 'STIXGeneral',
            'text.usetex': True,
        })

        if figsize is None:
            figsize = (17, 5.25)

        fig, axes = plt.subplots(nrows=1, ncols=n_sizes, figsize=figsize)
        plt.subplots_adjust(
            left=0.075,
            right=0.95,
            bottom=0.1,
            top=0.95,
            wspace=0.05
        )

        if n_sizes == 1:
            axes = [axes]

        markers = ['o', '^', 'v', 's']
        renamed_db_order = [PlotHelper.rename_str(rename, db) for db in db_order]
        db_markers = {db: markers[i] for i, db in enumerate(renamed_db_order)}

        def round_extrema(min_x, max_x):
            if how == 'log':
                return min_x, max_x

            min_delta = min(abs(min_x), abs(max_x))
            min_delta_tuple = min_delta.as_tuple()
            n_digits = len(min_delta_tuple.digits)
            e = min_delta_tuple.exponent

            e_unit = n_digits + e - 1
            mult = Decimal(str(10**e_unit))

            # Round to first significant digit
            min_x = mult*np.floor(min_x/mult)
            max_x = mult*np.ceil(max_x/mult)

            return min_x, max_x

        # Compute the xticks
        def xticks_params(values, xticks_dict=None):
            true_min_x = Decimal(str(values.min()))
            true_max_x = Decimal(str(values.max()))

            if xlims is not None:
                true_min_x, true_max_x = xlims

            if xticks_dict is None:  # Automatic xticks

                min_x, max_x = round_extrema(true_min_x, true_max_x)
                max_delta = float(max(abs(min_x), abs(max_x)))

                xticks = list(np.linspace(-max_delta, max_delta, 5))
                del xticks[0]
                del xticks[-1]
                xtick_labels = None

            else:  # Manual xticks
                assert isinstance(xticks_dict, dict)
                xticks = list(xticks_dict.keys())
                xtick_labels = list(xticks_dict.values())

                # min_x = Decimal(min(xticks))
                # max_x = Decimal(max(xticks))

                min_x = true_min_x
                max_x = true_max_x

                max_delta = float(max(abs(min_x), abs(max_x)))

            # Set limits
            if how == 'log':
                xlim_min = .9*float(true_min_x)
                xlim_max = 1.1*float(true_max_x)

            else:
                # Symetric constraint on xlims: use max absolute value
                # xlim_min = -max_delta
                # xlim_max = max_delta

                # Asymetric constraint: add margin to max and min
                margin = max_delta*0.05
                xlim_min = float(true_min_x) - margin
                xlim_max = float(true_max_x) + margin

            return xlim_min, xlim_max, xticks, xtick_labels

        # Uncomment this line to use same xlims constraint for all subplots
        # xlim_min, xlim_max, xticks, xtick_labels = xticks_params(df[f'relative_{value}'], xticks_dict=xticks_dict)

        if broken_axis is not None:
            axes_bg, axes_left, axes_right = [], [], []

        for i, size in enumerate(sizes):
            ax = axes[i]

            ax_bg = ax
            # ax_bg.axis('off')
            # break
            if broken_axis is not None:
                divider = make_axes_locatable(ax)
                # ax_right = divider.new_horizontal(size="2000%", pad=0.01)
                # ax = divider.new_horizontal(size="2000%", pad=0.01, pack_start=True)
                # ax.axis('off')
                # ax_bg.axis('off')
                ax_bg.spines['right'].set_visible(False)
                ax_bg.spines['bottom'].set_visible(False)
                ax_bg.spines['top'].set_visible(False)
                ax_bg.spines['left'].set_visible(False)
                ax_bg.tick_params(bottom=False, left=False, labelleft=False)#, labelbottom=False)
                ax_bg.xaxis.set_ticks([0, 1])
                ax_bg.xaxis.set_ticklabels([' ', ' '])
                # ax_bg.set_xticks(xticks, minor=False)
                ax_bg.set_xticklabels([' ', ' '], minor=False)
                ax_right = divider.append_axes('right', size='2000%', pad=0.0)
                ax_left = divider.append_axes('left', size='2000%', pad=0.0)
                fig.add_axes(ax_right)
                # fig.add_axes(ax)
                fig.add_axes(ax_left)
                ax = ax_left
                ax_left.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)
                ax_right.tick_params(left=False, labelleft=False)

                axes_bg.append(ax_bg)
                axes_left.append(ax_left)
                axes_right.append(ax_right)

                d = .01  # how big to make the diagonal lines in axes coordinates
                # arguments to pass to plot, just so we don't keep repeating them
                kwargs = dict(transform=ax_right.transAxes, color='k', clip_on=False)
                ax_right.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
                ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax_right.transAxes)  # switch to the bottom axes
                ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                kwargs.update(transform=ax_left.transAxes)  # switch to the bottom axes
                ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                # ax_left.axis('off')
                # ax_right.axis('off')
                # ax_bg.axis('off')

                # ax_left.spines['right'].set_visible(False)
                # ax_left.spines['bottom'].set_visible(False)
                # ax_left.spines['top'].set_visible(False)
                # ax_left.spines['left'].set_visible(False)

                # ax_right.spines['right'].set_visible(False)
                # ax_right.spines['bottom'].set_visible(False)
                # ax_right.spines['top'].set_visible(False)
                # ax_right.spines['left'].set_visible(False)

                # locator = divider.new_locator(nx=0, ny=1)
                # ax.set_axes_locator(locator)

            # break

            # Select the rows of interest
            subdf = df[df['size'] == size]

            # Split in valid and invalid data
            # idx_valid = subdf.index[(subdf['selection'] == 'manual') | (
            #     (subdf['selection'] != 'manual') & (subdf['n_trials'] == 5))]
            if only_full_samples:
                idx_valid = subdf.index[subdf['n_folds'] == 25]
                idx_invalid = subdf.index.difference(idx_valid)
                df_valid = subdf.loc[idx_valid]
                df_invalid = subdf.loc[idx_invalid]

            else:
                df_valid = subdf
                df_invalid = pd.DataFrame(columns=subdf.columns)

            # Update parameters for plotting invalids
            dbs_having_invalids = list(df_invalid['Database'].unique())
            n_dbs_invalid = len(dbs_having_invalids)
            db_invalid_markers = {db: m for db, m in db_markers.items() if db in dbs_having_invalids}
            renamed_db_order_invalid = [x for x in renamed_db_order if x in dbs_having_invalids]

            twinx = ax.twinx()
            twinx.set_ylim(0, n_methods)
            twinx.yaxis.set_visible(False)

            if broken_axis is not None:
                twinx_right = ax_right.twinx()
                twinx_right.set_ylim(0, n_methods)
                twinx_right.yaxis.set_visible(False)
                twinx_right.spines['left'].set_visible(False)
                twinx.spines['right'].set_visible(False)
                twinx_right.tick_params(left=False, labelleft=False)

            # Add gray layouts in the background every other rows
            for k in range(0, n_methods, 2):
                ax.axhspan(k-0.5, k+0.5, color='.93', zorder=0)
                if broken_axis is not None:
                    ax_right.axhspan(k-0.5, k+0.5, color='.93', zorder=0)
                    ax_bg.axhspan(k-0.5, k+0.5, color='.93', zorder=0)
                    ax_bg.set_ylim(-0.5-((n_methods+1) % 2), n_methods-0.5-((n_methods+1) % 2))

            mid = 1 if how == 'log' else 0
            ax.axvline(mid, ymin=0, ymax=n_methods, color='gray', zorder=0)
            if broken_axis is not None:
                ax.axvline(mid, ymin=0, ymax=n_methods, color='gray', zorder=0)

            # Build the color palette for the boxplot
            if colors is None:
                paired_colors = sns.color_palette('Paired').as_hex()
                # del paired_colors[10]
                paired_colors[10] = sns.color_palette("Set2").as_hex()[5]
                boxplot_palette = sns.color_palette(['#525252']+paired_colors)

            else:
                boxplot_palette = sns.color_palette(colors)

            # Add axvline for reference method
            if ref_vline is not None:
                ref_med = df_valid.query('method == @ref_vline')[f'relative_{value}'].median()
                ax.axvline(ref_med, ymin=0, ymax=n_methods, color='gray', zorder=0, ls='--', lw=1)

            # Add mean of methods other than reference
            if non_ref_vline:
                non_ref_mean = df_valid.query('method != @ref_vline and method != "MI" and method != "MI+mask" and method != "MIA+mask"')[f'relative_{value}'].median()
                ax.axvline(non_ref_mean, ymin=0, ymax=n_methods, color='gray', zorder=0, ls='--', lw=1)

            # Boxplot
            sns.set_palette(boxplot_palette)
            sns.boxplot(x=f'relative_{value}', y='method', data=df_valid, orient='h',
                        ax=ax, order=method_order, showfliers=False)
            if broken_axis is not None:
                sns.boxplot(x=f'relative_{value}', y='method', data=df_valid, orient='h',
                            ax=ax_right, order=method_order, showfliers=False)
                # sns.boxplot(x=f'relative_{value}', y='method', data=df_valid, orient='h',
                #             ax=ax_bg, order=method_order, showfliers=False)

            # Scatter plot for valid data points
            sns.set_palette(sns.color_palette('colorblind'))
            g2 = sns.scatterplot(x=f'relative_{value}', y='y', hue='Database',
                                 data=df_valid, ax=twinx,
                                 hue_order=renamed_db_order,
                                 style='Database',
                                 markers=db_markers,
                                 s=75,
                                 )
            if broken_axis is not None:
                g2_2 = sns.scatterplot(x=f'relative_{value}', y='y', hue='Database',
                                     data=df_valid, ax=twinx_right,
                                     hue_order=renamed_db_order,
                                     style='Database',
                                     markers=db_markers,
                                     s=75,
                                     )

            if legend_bbox:
                # g2.legend(loc='upper left', bbox_to_anchor=legend_bbox, ncol=1, title='Database')
                handles, labels = g2.get_legend_handles_labels()
                r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False)
                handles = [r] + handles
                labels = ['\\textbf{{Database}}'] + labels
                # g2.legend(loc='lower center', bbox_to_anchor=legend_bbox, ncol=4, title='\\textbf{{Database}}')
                # g2.get_legend().get_title().set_position((-250, -20))
                g2.legend(loc='lower center', bbox_to_anchor=legend_bbox, ncol=5, handles=handles, labels=labels)
                # g2.legend(loc='lower center', bbox_to_anchor=legend_bbox, ncol=4, title='Database')

            # Scatter plot for invalid data points
            if n_dbs_invalid > 0:
                sns.set_palette(sns.color_palette(n_dbs_invalid*['lightgray']))
                g3 = sns.scatterplot(x=f'relative_{value}', y='y', hue='Database',
                                     data=df_invalid, ax=twinx,
                                     hue_order=renamed_db_order_invalid,
                                     style='Database',
                                     markers=db_invalid_markers,
                                     s=75,
                                     legend=False,
                                     )
            # g3.legend(title='title')

            if not legend_bbox and i < len(sizes)-1:
                twinx.get_legend().remove()
                if broken_axis is not None:
                    twinx_right.get_legend().remove()

            elif legend_bbox and i > 0:
                twinx.get_legend().remove()

            if broken_axis is not None:
                twinx_right.get_legend().remove()

            if broken_axis is not None:
                ax_right.yaxis.set_visible(False)

            if i > 0:  # if not the first axis
                ax.yaxis.set_visible(False)
                # twinx.get_legend().remove()
            else:
                # Get yticks labels and rename them according to given dict
                labels = [item.get_text() for item in ax.get_yticklabels()]
                r_labels = [PlotHelper.rename_str(rename, l) for l in labels]
                ax.set_yticklabels(r_labels)
                # ax.text(1.1, 1.1, '\\textbf{{Database}}', fontsize='x-large', ha='left', va='center', transform=ax.transAxes, zorder=10)
                # if broken_axis is not None:
                #     ax_right.set_yticklabels(r_labels)

            if how == 'log':
                ax.set_xscale('log')
                twinx.set_xscale('log')

                if broken_axis is not None:
                    ax_right.set_xscale('log')
                    twinx_right.set_xscale('log')

            # Comment this line to use same xlims constraint for all subplots
            xlim_min, xlim_max, xticks, xtick_labels = xticks_params(subdf[f'relative_{value}'], xticks_dict=xticks_dict)

            if xtick_labels is not None:
                ax.set_xticks(xticks, minor=False)
                ax.set_xticklabels(xtick_labels, minor=False)
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

                twinx.set_xticks(xticks, minor=False)
                twinx.set_xticklabels(xtick_labels, minor=False)
                twinx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

            else:
                ax.set_xticks(xticks)
                twinx.set_xticks(xticks)

            ax.set_xlim(left=xlim_min, right=xlim_max)
            # twinx.set_xlim(left=xlim_min, right=xlim_max)

            if broken_axis is not None:
                if xtick_labels is not None:
                    xlim = ax_right.get_xlim()
                    ax_right.set_xticks(xticks, minor=False)
                    ax_right.set_xticklabels(xtick_labels, minor=False)
                    ax_right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                    ax_right.set_xlim(xlim)

                    xlim = twinx_right.get_xlim()
                    twinx_right.set_xticks(xticks, minor=False)
                    twinx_right.set_xticklabels(xtick_labels, minor=False)
                    twinx_right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                    twinx_right.set_xlim(xlim)

                else:
                    ax_right.set_xticks(xticks)
                    twinx_right.set_xticks(xticks)

                ax.set_xlim(left=xlim_min, right=xlim_max)

            ax_bg.set_title(f'n={size:,d}'.replace(',', '\\,'))
            if xlabel is None:
                xlabel = ax.get_xlabel()
            ax_bg.set_xlabel(PlotHelper.rename_str(rename, xlabel))
            ax.set_ylabel(None)
            ax.set_axisbelow(True)
            ax.grid(True, axis='x')
            if broken_axis is not None:
                if xlabel is None:
                    xlabel = ax_right.get_xlabel()
                # ax_right.set_xlabel(PlotHelper.rename_str(rename, xlabel))
                ax_right.set_xlabel(None)
                ax_left.set_xlabel(None)
                ax_right.set_ylabel(None)
                ax_right.set_axisbelow(True)
                ax_right.grid(True, axis='x')

            if broken_axis is not None:
                xlim = ax.get_xlim()
                xlim2 = ax_right.get_xlim()
                if isinstance(broken_axis, list):
                    ba_lims = broken_axis[i]
                else:
                    ba_lims = broken_axis
                ax.set_xlim((xlim[0], ba_lims[0]))
                ax_right.set_xlim((ba_lims[1], xlim2[1]))

                ax_bg.set_ylim(ax.get_ylim())

            # break

            # Optionally adds symbols on each line (for significance)
            if symbols is not None:
                method_symbols = symbols.get(size, None)
                if method_symbols is None:
                    continue

                xmin, xmax = ax.get_xlim()
                ax.set_xlim((xmin - 0.08*(xmax - xmin), xmax))

                for i, method in enumerate(method_order):
                    symbol = method_symbols.get(method, None)

                    if symbol is None:
                        continue

                    ax.annotate(symbol, xy=(0.025, 1-(i+0.5)/n_methods), color='black',
                                xycoords='axes fraction', fontsize='x-large', va='center')

            # Optionally adds comments on each line (for untractable)
            if comments is not None:
                method_comments = comments.get(size, None)
                if method_comments is None:
                    continue

                for m, method in enumerate(method_order):
                    comment = method_comments.get(method, None)

                    if comment is None:
                        continue

                    x = comments_spacing
                    ha = 'left'
                    ax_comment = ax_left if broken_axis is not None else ax_bg
                    if comments_align is not None:
                        if isinstance(comments_align, dict):
                            align = comments_align[i][m]
                        else:
                            align = comments_align[m]

                        if align == 'right':
                            x = 1 - comments_spacing
                            ha = 'right'
                            ax_comment = ax_right if broken_axis is not None else ax_bg

                    ax_comment.text(x, 1-(m+0.5)/n_methods, comment,
                                    color='.4', fontsize='x-large', ha=ha,
                                    va='center', transform=ax_comment.transAxes)

        if broken_axis is not None:
            return fig, axes_bg, axes_left, axes_right

        return fig, axes, None, None

    @staticmethod
    def mean_rank(filepath, method_order=None):
        if not isinstance(filepath, pd.DataFrame):
            df = pd.read_csv(filepath, index_col=0)
        else:
            df = filepath

        # Check method order
        methods = list(df['method'].unique())
        if method_order is None:
            method_order = methods
        elif set(method_order).issubset(set(methods)):
            # Keep only the rows having methods in method_order
            df = df[df['method'].isin(method_order)]
        else:
            raise ValueError(f'Method order missmatch existing ones {methods}')

        dfgb = df.groupby(['size', 'db', 'task', 'trial', 'fold'])
        df['rank'] = dfgb['score'].rank(method='dense', ascending=False)
        print(df)

        # Agregate across foldss by averaging
        dfgb = df.groupby(['size', 'db', 'task', 'method', 'trial'])
        df = dfgb.agg({'rank': 'mean', 'selection': 'first'})

        # Agregate across trials by averaging
        df = df.reset_index()
        df['n_trials'] = 1  # Add a count column to keep track of # of trials
        dfgb = df.groupby(['size', 'db', 'task', 'method'])
        df = dfgb.agg({'rank': 'mean', 'selection': 'first', 'n_trials': 'sum'})

        # We only take into account full results (n_trials == 5)
        df = df.reset_index()
        idx_valid = df.index[(df['selection'] == 'manual') | (
            (df['selection'] != 'manual') & (df['n_trials'] == 5))]
        df = df.loc[idx_valid]

        # Average across tasks
        dfgb = df.groupby(['size', 'db', 'method'])
        df = dfgb.agg({'rank': 'mean'})

        # Reset index to addlevel of the multi index to the columns of the df
        df = df.reset_index()

        # Compute average by size
        dfgb = df.groupby(['size', 'method'])
        df_avg_by_size = dfgb.agg({'rank': 'mean'})
        df_avg_by_size = df_avg_by_size.reset_index()
        df_avg_by_size = pd.pivot_table(df_avg_by_size, values='rank', index=['method'], columns=['size'])

        # Compute average on all data
        dfgb = df.groupby(['method'])
        df_avg = dfgb.agg({'rank': 'mean'})

        # Create a pivot table of the rank accross methods
        df_pt = pd.pivot_table(df, values='rank', index=['method'], columns=['size', 'db'])

        df_pt.sort_values(by=['size', 'db'], axis=1, inplace=True)

        # Add average by size columns
        for size in df['size'].unique():
            df_pt[(size, 'AVG')] = df_avg_by_size[size]

        df_pt.sort_values(by=['size'], axis=1, inplace=True)

        # Add global order column
        df_pt[('Global', 'AVG')] = df_avg
        df_pt[('Global', 'Rank')] = df_avg.rank().astype(int)

        # Round mean ranks
        df_pt = df_pt.round(2)

        # Reorder the method index
        if method_order:
            assert len(set(method_order)) == len(set(df['method'].unique()))
            df_pt = df_pt.reindex(method_order)

        print(df_pt)

        return df_pt

    @staticmethod
    def ranks(filepath, method_order=None):
        if not isinstance(filepath, pd.DataFrame):
            df = pd.read_csv(filepath, index_col=0)
        else:
            df = filepath.copy()

        # Check method order
        methods = list(df['method'].unique())
        if method_order is None:
            method_order = methods
        elif set(method_order).issubset(set(methods)):
            # Keep only the rows having methods in method_order
            df = df[df['method'].isin(method_order)]
        else:
            raise ValueError(f'Method order missmatch existing ones {methods}')

        # Compute ranks on each fold, trial
        dfgb = df.groupby(['size', 'db', 'task', 'trial', 'fold'])
        df['rank'] = dfgb['score'].rank(method='dense', ascending=False)

        # Average accross folds and trials
        mean_scores = aggregate(df, 'score')
        mean_ranks = aggregate(df, 'rank')

        dfgb = mean_scores.groupby(['size', 'db', 'task'])
        mean_scores['rank'] = dfgb['score'].rank(method='dense', ascending=False)

        mean_scores = mean_scores.set_index(['size', 'db', 'task', 'method'])
        mean_ranks = mean_ranks.set_index(['size', 'db', 'task', 'method'])

        rank_of_mean_scores = mean_scores['rank']
        mean_scores = mean_scores['score']
        mean_ranks = mean_ranks['rank']

        rank_of_mean_scores = rank_of_mean_scores.reset_index()
        mean_scores = mean_scores.reset_index()
        mean_ranks = mean_ranks.reset_index()

        # print(mean_scores)
        # print(rank_of_mean_scores)
        # print(mean_ranks)

        # Average on dataset size
        dfgb = mean_scores.groupby(['db', 'task', 'method'])
        mean_scores_on_sizes = dfgb.agg({'score': 'mean'})

        dfgb = rank_of_mean_scores.groupby(['db', 'task', 'method'])
        mean_rank_of_mean_scores_on_sizes = dfgb.agg({'rank': 'mean'})

        dfgb = mean_ranks.groupby(['db', 'task', 'method'])
        mean_ranks_on_sizes = dfgb.agg({'rank': 'mean'})

        dfgb = mean_scores_on_sizes.groupby(['db', 'task'])
        rank_of_mean_scores_on_sizes = dfgb['score'].rank(method='dense', ascending=False)

        print(mean_scores_on_sizes)
        print(rank_of_mean_scores_on_sizes)
        print(mean_rank_of_mean_scores_on_sizes)
        print(mean_ranks_on_sizes)

        ranks_on_sizes = mean_ranks_on_sizes.copy().rename({'rank': 'mean_ranks'}, axis=1)
        ranks_on_sizes['rank_of_mean_scores'] = rank_of_mean_scores_on_sizes
        ranks_on_sizes['mean_rank_of_mean_scores'] = mean_rank_of_mean_scores_on_sizes

        print(ranks_on_sizes)

    @staticmethod
    def plot_scores(filepath, db_order=None, method_order=None, rename=dict(),
                    reference_method=None, symbols=None, comments=None, only_full_samples=True,
                    legend_bbox=(4.22, 1.075), figsize=(18, 5.25), table_fontsize=13,
                    y_labelsize=18, pos_arrow=None, w_bag=None, w_const=None,
                    w_cond=None, colors=None, hline_pos=None, ref_vline=None):
        if not isinstance(filepath, pd.DataFrame):
            scores = pd.read_csv(filepath, index_col=0)
        else:
            scores = filepath

        fig, axes, _, _ = PlotHelper._plot(scores, 'score', how='no-norm',
                                           method_order=method_order,
                                           db_order=db_order, rename=rename,
                                           reference_method=reference_method,
                                           figsize=figsize,
                                           legend_bbox=legend_bbox,
                                           symbols=symbols,
                                           comments=comments,
                                           only_full_samples=only_full_samples,
                                           y_labelsize=y_labelsize,
                                           colors=colors,
                                           ref_vline=ref_vline,
                                           )

        df_ranks = get_ranks_tab(scores, method_order=method_order, db_order=db_order, average_sizes=True)

        global_avg_ranks = df_ranks[('Average', 'All')].loc['Average']
        argmin = global_avg_ranks.argmin()
        global_avg_ranks.iloc[argmin] = f"\\textbf{{{global_avg_ranks.iloc[argmin]}}}"
        cellText = np.transpose([list(global_avg_ranks.astype(str))])
        rowLabels = list(global_avg_ranks.index)
        rowLabels = [PlotHelper.rename_str(rename, s) for s in rowLabels]
        n_methods = cellText.shape[0]
        cellColours = [['white']]*n_methods
        for i in range(0, n_methods, 2):
            cellColours[i] = ['.93']

        table = axes[-1].table(cellText=cellText, loc='right',
                               rowLabels=None,
                               colLabels=['Mean\nrank'],
                            #    bbox=[1.32, -0.11, .19, .87],
                               bbox=[1.02, 0, .14, (n_methods+1)/n_methods],
                               #    bbox=[1.3, 0, .2, .735],
                               colWidths=[0.14],
                               cellColours=cellColours,
                               )
        table.set_fontsize(table_fontsize)

        n_methods = 9 if method_order is None else len(method_order)

        # Add brackets
        ax = axes[0]

        w_bag = 45 if w_bag is None else w_bag
        n_bag = 1.5
        bag_subsize = 'small'

        l_tail = 0.03
        dh = 1./n_methods
        lw = 1.3
        fs = 18

        if n_methods <= 8:
            w_const = 70 if w_const is None else w_const
            w_cond = 70 if w_cond is None else w_cond
            w_bag = 110 if w_bag is None else w_bag
            pos_arrow = -0.94 if pos_arrow is None else pos_arrow
            n_cond = None
            n_const = None
            n_bag = 1.5
            bag_subsize = 'Large'
        if n_methods == 9:
            w_const = 70 if w_const is None else w_const
            w_cond = 70 if w_cond is None else w_cond
            pos_arrow = -0.3 if pos_arrow is None else pos_arrow
            n_cond = 2
            n_const = 6
        elif n_methods == 10:
            w_const = 55 if w_const is None else w_const
            w_cond = 70 if w_cond is None else w_cond
            pos_arrow = -0.3 if pos_arrow is None else pos_arrow
            n_cond = 3
            n_const = 7
        elif n_methods == 11:
            w_const = 55 if w_const is None else w_const
            w_cond = 86 if w_cond is None else w_cond
            pos_arrow = -0.3 if pos_arrow is None else pos_arrow
            n_cond = 3
            n_const = 8
        elif n_methods == 12:
            w_const = 60 if w_const is None else w_const
            w_cond = 60 if w_cond is None else w_cond
            w_bag = 45 if w_bag is None else w_bag
            pos_arrow = -0.74 if pos_arrow is None else pos_arrow
            n_cond = 5
            n_const = 9
            n_bag = 1.5

        # Here is the label and arrow code of interest
        if n_const is not None:
            ax.annotate('Constant\nimputation\n\n', xy=(pos_arrow, n_const*dh), xytext=(pos_arrow-l_tail, n_const*dh), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='center',
                        bbox=None,#dict(boxstyle='square', fc='white'),
                        arrowprops=dict(arrowstyle=f'-[, widthB={w_const/fs}, lengthB=0.5', lw=lw),
                        rotation=90,
                        )

        if n_cond is not None:
            ax.annotate('Conditional\nimputation\n\n', xy=(pos_arrow, n_cond*dh), xytext=(pos_arrow-l_tail, n_cond*dh), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='center',
                        bbox=None,#dict(boxstyle='square', fc='white'),
                        arrowprops=dict(arrowstyle=f'-[, widthB={w_cond/fs}, lengthB=0.5', lw=lw),
                        rotation=90,
                        )

        if w_bag != 0:
            ax.annotate(f'Bagging\n\\{bag_subsize}{{(multiple imputation)}}\n\n', xy=(pos_arrow, n_bag*dh), xytext=(pos_arrow-l_tail, n_bag*dh), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='center',
                        bbox=None,#dict(boxstyle='square', fc='white'),
                        arrowprops=dict(arrowstyle=f'-[, widthB={w_bag/fs}, lengthB=0.5', lw=lw),
                        rotation=90,
                        )

        plt.subplots_adjust(right=.88)

        if hline_pos is not None:
            for ax in axes:
                for pos in hline_pos:
                    ax.axhline(pos-0.5, color='black', lw=1)

        return fig

    @staticmethod
    def plot_times(filepath, which, xticks_dict=None, xlims=None, db_order=None,
                   method_order=None, rename=dict(), reference_method=None,
                   linear=False, only_full_samples=True, y_labelsize=18, comments=None, figsize=(18, 5.25),
                   legend_bbox=(4.22, 1.075), broken_axis=None, comments_align=None, comments_spacing=0.025,
                   table_fontsize=13, pos_arrow=None, w_bag=None, w_const=None,
                   w_cond=None, colors=None, hline_pos=None, non_ref_vline=False):
        if not isinstance(filepath, pd.DataFrame):
            scores = pd.read_csv(filepath, index_col=0)
        else:
            scores = filepath

        if which == 'PT':
            scores['total_PT'] = scores['imputation_PT'].fillna(0) + scores['tuning_PT']
            value = 'total_PT'
        elif which == 'WCT':
            scores['total_WCT'] = scores['imputation_WCT'].fillna(0) + scores['tuning_WCT']
            value = 'total_WCT'
        else:
            raise ValueError(f'Unknown argument {which}')
        fig, axes_bg, axes_left, axes_right = PlotHelper._plot(scores, value, how='log',
                                     xticks_dict=xticks_dict,
                                     xlims=xlims,
                                     method_order=method_order,
                                     db_order=db_order, rename=rename,
                                     reference_method=reference_method,
                                     figsize=figsize,
                                     only_full_samples=only_full_samples,
                                     y_labelsize=y_labelsize,
                                     comments=comments,
                                     legend_bbox=legend_bbox,
                                     broken_axis=broken_axis,
                                     comments_align=comments_align,
                                     comments_spacing=comments_spacing,
                                     colors=colors,
                                     non_ref_vline=non_ref_vline,
                                     )

        # df_ranks = get_ranks_tab(scores, method_order=method_order, db_order=db_order, average_sizes=True)

        # global_avg_ranks = df_ranks[('Average', 'All')].loc['Average']
        # argmin = global_avg_ranks.argmin()
        # global_avg_ranks.iloc[argmin] = f"\\textbf{{{global_avg_ranks.iloc[argmin]}}}"

        times = scores.groupby(['method']).aggregate({'tuning_PT': 'sum'})
        # print(times)
        cellText = [f'{int(times.loc[m]/3600/24):,d}'.replace(',', '\,') for m in method_order]
        # print(cellText)
        # exit()

        cellText = np.transpose([cellText])
        # cellText = np.transpose([list(global_avg_ranks.astype(str))])
        rowLabels = method_order
        rowLabels = [PlotHelper.rename_str(rename, s) for s in rowLabels]
        n_methods = len(cellText)
        cellColours = [['white']]*n_methods
        for i in range(0, n_methods, 2):
            cellColours[i] = ['.93']

        axes_table = axes_bg if broken_axis is None else axes_right

        if broken_axis:
            bbox = [1.042, 0, .28, (n_methods+1)/n_methods]
            colWidths = [0.28]

        else:
            bbox = [1.02, 0, .14, (n_methods+1)/n_methods]
            colWidths = [0.14]

        table = axes_table[-1].table(cellText=cellText, loc='right',
                               rowLabels=None,
                               colLabels=['CPU\ndays'],
                               bbox=bbox,
                               colWidths=colWidths,
                               cellColours=cellColours,
                               )
        table.set_fontsize(table_fontsize)

        # Add brackets
        # fs = 18
        # lw = 1.3
        # dh = 1./9
        # l_tail = 0.03
        n_methods = 9 if method_order is None else len(method_order)

        w_bag = 45 if w_bag is None else w_bag
        n_bag = 1.5
        bag_subsize = 'small'

        l_tail = 0.03
        dh = 1./n_methods
        lw = 1.3
        fs = 18

        if n_methods <= 8:
            w_const = 70 if w_const is None else w_const
            w_cond = 70 if w_cond is None else w_cond
            w_bag = 110 if w_bag is None else w_bag
            n_cond = 2
            n_const = 6
            pos_arrow = -1.84 if pos_arrow is None else pos_arrow
            n_cond = None
            n_const = None
            n_bag = 1.5
            bag_subsize = 'Large'
        if n_methods == 9:
            w_const = 70 if w_const is None else w_const
            w_cond = 70 if w_cond is None else w_cond
            pos_arrow = -0.3 if pos_arrow is None else pos_arrow
            n_cond = 2
            n_const = 6
        elif n_methods == 10:
            w_const = 55 if w_const is None else w_const
            w_cond = 70 if w_cond is None else w_cond
            pos_arrow = -0.3 if pos_arrow is None else pos_arrow
            n_cond = 3
            n_const = 7
        elif n_methods == 11:
            w_const = 55 if w_const is None else w_const
            w_cond = 86 if w_cond is None else w_cond
            pos_arrow = -0.3 if pos_arrow is None else pos_arrow
            n_cond = 3
            n_const = 8
        elif n_methods == 12:
            w_const = 60 if w_const is None else w_const
            w_cond = 60 if w_cond is None else w_cond
            w_bag = 45 if w_bag is None else w_bag
            pos_arrow = -1.575 if pos_arrow is None else pos_arrow
            n_cond = 5
            n_const = 9
            n_bag = 1.5

        ax = axes_bg[0] if broken_axis is None else axes_left[0]

        # Here is the label and arrow code of interest
        if n_const is not None:
            ax.annotate('Constant\nimputation\n\n', xy=(pos_arrow, n_const*dh), xytext=(pos_arrow-l_tail, n_const*dh), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='center',
                        bbox=None,#dict(boxstyle='square', fc='white'),
                        arrowprops=dict(arrowstyle=f'-[, widthB={w_const/fs}, lengthB=0.5', lw=lw),
                        rotation=90,
                        )

        if n_cond is not None:
            ax.annotate('Conditional\nimputation\n\n', xy=(pos_arrow, n_cond*dh), xytext=(pos_arrow-l_tail, n_cond*dh), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='center',
                        bbox=None,#dict(boxstyle='square', fc='white'),
                        arrowprops=dict(arrowstyle=f'-[, widthB={w_cond/fs}, lengthB=0.5', lw=lw),
                        rotation=90,
                        )

        if w_bag != 0:
            ax.annotate(f'Bagging\n\\{bag_subsize}{{(multiple imputation)}}\n\n', xy=(pos_arrow, n_bag*dh), xytext=(pos_arrow-l_tail, n_bag*dh), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='center',
                        bbox=None,#dict(boxstyle='square', fc='white'),
                        arrowprops=dict(arrowstyle=f'-[, widthB={w_bag/fs}, lengthB=0.5', lw=lw),
                        rotation=90,
                        )

        # Add arrow on top right comment
        ax = axes_bg[-1] if broken_axis is None else axes_right[-1]
        xpos = 0.7 if broken_axis is not None else 0.85
        ypos = 0.94 if n_methods < 7 else 0.964
        ax.annotate('Mean time\nper task', xy=(xpos, ypos), xytext=(xpos, 1.1),
                    xycoords='axes fraction', ha='center', va='center',
                    fontsize=fs,
                    arrowprops=dict(arrowstyle=f'->', lw=1, color='gray'),
                    )

        if hline_pos is not None:
            for axes in [axes_bg, axes_left, axes_right]:
                if axes is None:
                    continue
                for ax in axes:
                    for pos in hline_pos:
                        ax.axhline(pos-0.5, color='black', lw=1)

        return fig

    @staticmethod
    def plot_MIA_linear(filepath, db_order, method_order, rename=dict(), symbols=None):
        if not isinstance(filepath, pd.DataFrame):
            scores = pd.read_csv(filepath, index_col=0)
        else:
            scores = filepath
        # Select methods of interest
        scores = scores.loc[scores['method'].isin(method_order)]

        fig, axes = PlotHelper._plot(scores, 'score', how='no-norm',
                                       rename=rename,
                                       db_order=db_order,
                                       method_order=method_order,

                                       #    xlabel='absolute_score',
                                       #    xticks_dict={
                                       #        0: '0',
                                       #        1: '1',
                                       #    },
                                       #    xlims=(0, 1.1)
                                       xticks_dict={
                                           #    -0.05: '-0.05',
                                           0: '0',
                                           0.05: '0.05',
                                           .1: '0.1',
                                       },
                                       xlims=(-0.04, 0.14),
                                       #    figsize=(17, 3.25),
                                       figsize=(18, 5.25),
                                       legend_bbox=(4.30, 1.075),
                                       symbols=symbols,
                                       )

        df_ranks = get_ranks_tab(scores, method_order=method_order, db_order=db_order, average_sizes=True)

        global_avg_ranks = df_ranks[('Average', 'All')].loc['Average']
        argmin = global_avg_ranks.argmin()
        global_avg_ranks.iloc[argmin] = f"\\textbf{{{global_avg_ranks.iloc[argmin]}}}"
        cellText = np.transpose([list(global_avg_ranks.astype(str))])
        rowLabels = list(global_avg_ranks.index)
        rowLabels = [PlotHelper.rename_str(rename, s) for s in rowLabels]

        table = axes[-1].table(cellText=cellText, loc='right',
                       rowLabels=rowLabels,
                       colLabels=['Mean\nrank'],
                    #    bbox=[1.37, 0, .2, .735],
                       bbox=[1.41, -0.11, .19, .87],
                       colWidths=[0.2],
                       )
        table.set_fontsize(13)

        # Add brackets
        ax = axes[0]
        fs = 18
        lw = 1.3
        dh = 1./9
        l_tail = 0.03
        pos_arrow = -0.45
        # Here is the label and arrow code of interest
        ax.annotate('Constant\nimputation\n\n', xy=(pos_arrow, 6*dh), xytext=(pos_arrow-l_tail, 6*dh), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='center',
                    bbox=None,#dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle=f'-[, widthB={70/fs}, lengthB=0.5', lw=lw),
                    rotation=90,
                    )

        ax.annotate('Conditional\nimputation\n\n', xy=(pos_arrow, 2*dh), xytext=(pos_arrow-l_tail, 2*dh), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='center',
                    bbox=None,#dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle=f'-[, widthB={70/fs}, lengthB=0.5', lw=lw),
                    rotation=90,
                    )

        plt.subplots_adjust(right=.88, left=.09)

        return fig
