

from abc import ABC, abstractmethod


class Database(ABC):

    @abstractmethod
    def __init__(self):
        self.tables = dict()
        self.name = ''
        self.acronym = ''
        self._load()

    def __getitem__(self, key):
        return self.tables[key]

    def tables_names(self):
        return list(self.tables.keys())

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def heuristic(self):
        pass
