"""Implement  PlotHelper for train4 results."""
import os


class PlotHelperV4(object):
    """Plot the train4 results."""

    def __init__(self, root_folder):
        """Init."""
        # Stepe 1: Check and register root_path
        root_folder = root_folder.rstrip('/')  # Remove trailing '/'
        self.root_folder = root_folder

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

        return sizes
