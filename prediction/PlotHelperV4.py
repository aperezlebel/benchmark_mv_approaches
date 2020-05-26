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
        for root, dirnames, filenames in walk:
            if dirnames:
                for dirname in dirnames:
                    # print(f'{root}/{dirname}')
                    abs_dirpaths.append(f'{root}/{dirname}')

        prefix = os.path.commonprefix(abs_dirpaths)
        rel_dir_paths = [os.path.relpath(p, prefix) for p in abs_dirpaths]
        # print(rel_dir_paths)

        # Step 3: Convert relative paths to nested python dictionnary
        nested_dict = {}

        for rel_dir_path in rel_dir_paths:
            d = nested_dict
            for x in rel_dir_path.split('/'):
                d = d.setdefault(x, {})

        print(nested_dict)

        # Step 4: Fill the class attributes with the nested dict
        self._nested_dict = nested_dict

    def databases(self):
        """Return the databases found in the root folder."""
        nd = self._nested_dict
        return [db for db in nd.keys() if self.tasks(db)]

    def tasks(self, db):
        """Return the tasks related to a given database."""
        nd = self._nested_dict
        return [t for t in nd[db].keys() if nd[db][t]]

    def methods(self, db, t):
        """Return the methods used by a given task."""
        nd = self._nested_dict
        return [m for m in nd[db][t] if self._is_valid_method(db, t, m)]

    def _is_valid_method(self, db, t, m):
        path = f'{self.root_folder}/{db}/{t}/{m}/'

        if not os.path.exists(path):
            return False

        _, _, filenames = next(os.walk(path))

        return len(filenames) > 1  # there is always strat_infos.yml in m folder

