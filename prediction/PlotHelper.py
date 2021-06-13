"""Implement  PlotHelper for train4 results."""
import os
import yaml
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import r2_score, roc_auc_score
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
import shutil


class PlotHelper(object):
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

    def get_task_description(self, filepath):
        """Build and dump a csv that will explain each task once completed."""
        df = pd.read_csv(filepath)

        df = self.aggregate(df, 'score')

        print(df)

        dfgb = df.groupby(['db', 'task'])
        df = dfgb.agg({
            'score': 'mean',
            # 'n_trials': 'sum',
            # 'n_folds': 'sum',
            'scorer': PlotHelper.assert_equal,  # first and assert equal
            'selection': PlotHelper.assert_equal,
            'n': PlotHelper.assert_equal,
            'p': PlotHelper.assert_equal,
            'type': PlotHelper.assert_equal,
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
    def assert_equal(s):
        """Check if all value of the series are equal and return the value."""
        if s.empty:
            return 0

        if not (s.iloc[0] == s).all():
            raise ValueError(
                f'Values differ but supposed to be constant. Col: {s.name}.'
            )
        return s.iloc[0]

    @staticmethod
    def aggregate(df, value):
        # Agregate accross folds by averaging
        df['n_folds'] = 1
        dfgb = df.groupby(['size', 'db', 'task', 'method', 'trial'])
        df = dfgb.agg({
            value: 'mean',
            'n_folds': 'sum',
            'scorer': PlotHelper.assert_equal,  # first and assert equal
            'selection': PlotHelper.assert_equal,
            'n': PlotHelper.assert_equal,
            'p': PlotHelper.assert_equal,
            'type': PlotHelper.assert_equal,
            'imputation_WCT': 'mean',
            'tuning_WCT': 'mean',
            'imputation_PT': 'mean',
            'tuning_PT': 'mean',
        })

        # Agregate accross trials by averaging
        df = df.reset_index()
        df['n_trials'] = 1  # Add a count column to keep track of # of trials
        dfgb = df.groupby(['size', 'db', 'task', 'method'])
        df = dfgb.agg({
            value: 'mean',
            'n_trials': 'sum',
            'n_folds': 'sum',
            'scorer': PlotHelper.assert_equal,  # first and assert equal
            'selection': PlotHelper.assert_equal,
            'n': PlotHelper.assert_equal,
            'p': 'mean',  #PlotHelper.assert_equal,
            'type': PlotHelper.assert_equal,
            'imputation_WCT': 'mean',
            'tuning_WCT': 'mean',
            'imputation_PT': 'mean',
            'tuning_PT': 'mean',
        })

        # Reset index to addlevel of the multi index to the columns of the df
        df = df.reset_index()

        return df

    @staticmethod
    def _plot(filepath, value, how, xticks_dict=None, xlims=None, db_order=None,
              method_order=None, rename=dict(), reference_method=None,
              figsize=None, legend_bbox=None, xlabel=None):
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

        df = PlotHelper.aggregate(df, value)

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
            'legend.fontsize': 10,
            'axes.titlesize': 17,
            'axes.labelsize': 15,
            'xtick.labelsize': 12,
            'ytick.labelsize': 18,
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

        for i, size in enumerate(sizes):
            ax = axes[i]

            # Select the rows of interest
            subdf = df[df['size'] == size]

            # Split in valid and invalid data
            # idx_valid = subdf.index[(subdf['selection'] == 'manual') | (
            #     (subdf['selection'] != 'manual') & (subdf['n_trials'] == 5))]
            idx_valid = subdf.index[subdf['n_folds'] == 25]
            idx_invalid = subdf.index.difference(idx_valid)
            df_valid = subdf.loc[idx_valid]
            df_invalid = subdf.loc[idx_invalid]

            # Update parameters for plotting invalids
            dbs_having_invalids = list(df_invalid['Database'].unique())
            n_dbs_invalid = len(dbs_having_invalids)
            db_invalid_markers = {db: m for db, m in db_markers.items() if db in dbs_having_invalids}
            renamed_db_order_invalid = [x for x in renamed_db_order if x in dbs_having_invalids]

            twinx = ax.twinx()
            twinx.set_ylim(0, n_methods)
            twinx.yaxis.set_visible(False)

            # Add gray layouts in the background every other rows
            for k in range(0, n_methods, 2):
                ax.axhspan(k-0.5, k+0.5, color='.93', zorder=0)

            mid = 1 if how == 'log' else 0
            ax.axvline(mid, ymin=0, ymax=n_methods, color='gray', zorder=0)

            # Build the color palette for the boxplot
            paired_colors = sns.color_palette('Paired').as_hex()
            boxplot_palette = sns.color_palette(['#525252']+paired_colors)

            # Boxplot
            sns.set_palette(boxplot_palette)
            sns.boxplot(x=f'relative_{value}', y='method', data=df_valid, orient='h',
                        ax=ax, order=method_order, showfliers=False)

            # Scatter plot for valid data points
            sns.set_palette(sns.color_palette('colorblind'))
            g2 = sns.scatterplot(x=f'relative_{value}', y='y', hue='Database',
                                 data=df_valid, ax=twinx,
                                 hue_order=renamed_db_order,
                                 style='Database',
                                 markers=db_markers,
                                 s=75,
                                 )

            if legend_bbox:
                g2.legend(loc='upper left', bbox_to_anchor=legend_bbox, ncol=1)

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

            if i > 0:  # if not the first axis
                ax.yaxis.set_visible(False)
                twinx.get_legend().remove()
            else:
                # Get yticks labels and rename them according to given dict
                labels = [item.get_text() for item in ax.get_yticklabels()]
                r_labels = [PlotHelper.rename_str(rename, l) for l in labels]
                ax.set_yticklabels(r_labels)

            if how == 'log':
                ax.set_xscale('log')
                twinx.set_xscale('log')

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

            ax.set_title(f'n={size}')
            if xlabel is None:
                xlabel = ax.get_xlabel()
            ax.set_xlabel(PlotHelper.rename_str(rename, xlabel))
            ax.set_ylabel(None)
            ax.set_axisbelow(True)
            ax.grid(True, axis='x')

        return fig, axes

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
        mean_scores = PlotHelper.aggregate(df, 'score')
        mean_ranks = PlotHelper.aggregate(df, 'rank')

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
                    reference_method=None,):
        fig, axes = PlotHelper._plot(filepath, 'score', how='no-norm',
                                       method_order=method_order,
                                       db_order=db_order, rename=rename,
                                       reference_method=reference_method,
                                       figsize=(17, 5.25),
                                       legend_bbox=(4.22, 1.015))

        df_ranks = PlotHelper.mean_rank(filepath, method_order=method_order)

        global_avg_ranks = df_ranks[('Global', 'AVG')]
        cellText = np.transpose([list(global_avg_ranks.astype(str))])
        rowLabels = list(global_avg_ranks.index)
        rowLabels = [PlotHelper.rename_str(rename, s) for s in rowLabels]

        axes[-1].table(cellText=cellText, loc='right',
                       rowLabels=rowLabels,
                       colLabels=['Mean\nrank'],
                       bbox=[1.3, 0, .2, .735],
                       colWidths=[0.2],
                       )

        plt.subplots_adjust(right=.88)

        return fig

    @staticmethod
    def plot_times(filepath, which, xticks_dict=None, xlims=None, db_order=None,
                   method_order=None, rename=dict(), reference_method=None):
        df = pd.read_csv(filepath, index_col=0)
        if which == 'PT':
            df['total_PT'] = df['imputation_PT'].fillna(0) + df['tuning_PT']
            value = 'total_PT'
        elif which == 'WCT':
            df['total_WCT'] = df['imputation_WCT'].fillna(0) + df['tuning_WCT']
            value = 'total_WCT'
        else:
            raise ValueError(f'Unknown argument {which}')
        fig, _ = PlotHelper._plot(df, value, how='log',
                                    xticks_dict=xticks_dict,
                                    xlims=xlims,
                                    method_order=method_order,
                                    db_order=db_order, rename=rename,
                                    reference_method=reference_method,
                                    figsize=None)
        return fig

    # @staticmethod
    # def plot_MIA_linear_diff(filepath, db_order, rename=dict()):
    #     df = pd.read_csv(filepath, index_col=0)

    #     # Select methods of interest
    #     # df = df.loc[df['method'].isin(['MIA', 'Linear+Iter', 'Linear+Iter+mask'])]

    #     # Compute the difference of scores between MIA and Linear model
    #     dfgb = df.groupby(['method'])

    #     MIA = dfgb.get_group('MIA').set_index(['size', 'db', 'task', 'trial', 'fold'])
    #     Linear = dfgb.get_group('Linear+Iter').set_index(['size', 'db', 'task', 'trial', 'fold'])
    #     Linear_mask = dfgb.get_group('Linear+Iter+mask').set_index(['size', 'db', 'task', 'trial', 'fold'])

    #     # Diff between MIA and Linear+Iterative
    #     diff1 = MIA.copy()
    #     diff1['method'] = 'Linear\n - MIA'

    #     # Diff between MIA and Linear+Iterative+mask
    #     diff2 = MIA.copy()
    #     diff2['method'] = 'Linear+mask\n - MIA'

    #     scores_diff1 = Linear['score'] - MIA['score']
    #     scores_normalized1 = scores_diff1.divide(MIA['score'])

    #     scores_diff2 = Linear_mask['score'] - MIA['score']
    #     scores_normalized2 = scores_diff2.divide(MIA['score'])

    #     diff1['score'] = scores_diff1
    #     diff1.reset_index(inplace=True)

    #     diff2['score'] = scores_diff2
    #     diff2.reset_index(inplace=True)

    #     diff = pd.concat([diff1, diff2], axis=0)

    #     diff.to_csv('sandbox/diff2.csv')

    #     fig, axes = PlotHelper._plot(diff, 'score', how='abs',
    #                                    rename=rename,
    #                                    db_order=db_order,
    #                                    xlabel='absolute_score',
    #                                    xticks_dict={
    #                                        0: '0',
    #                                        -0.1: '-0.1',
    #                                    },
    #                                    xlims=(-0.11, 0.04)
    #                                    )

    #     return fig

    @staticmethod
    def plot_MIA_linear(filepath, db_order, method_order, rename=dict()):
        df = pd.read_csv(filepath, index_col=0)
        # Select methods of interest
        df = df.loc[df['method'].isin(method_order)]

        fig, axes = PlotHelper._plot(df, 'score', how='no-norm',
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
                                       figsize=(17, 5.25),
                                       legend_bbox=(4.22, 1.015),
                                       )

        df_ranks = PlotHelper.mean_rank(filepath, method_order=method_order)

        global_avg_ranks = df_ranks[('Global', 'AVG')]
        cellText = np.transpose([list(global_avg_ranks.astype(str))])
        rowLabels = list(global_avg_ranks.index)
        rowLabels = [PlotHelper.rename_str(rename, s) for s in rowLabels]

        axes[-1].table(cellText=cellText, loc='right',
                       rowLabels=rowLabels,
                       colLabels=['Mean\nrank'],
                       bbox=[1.37, 0, .2, .735],
                       colWidths=[0.2],
                       )

        plt.subplots_adjust(right=.88, left=.09)

        return fig
