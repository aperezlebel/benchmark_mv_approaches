"""Implement the Strategy class."""
import sklearn
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class Strategy():
    """Elements to run cross-validated ML estimator with hyperparms tuning."""

    _count = 0

    def __init__(self, estimator, split, cv, param_space, search,
                 imputer=None, split_params=dict(), search_params=dict(),
                 name=None):
        self.estimator = estimator
        self.cv = cv
        self.param_space = param_space
        self._name = name
        self.imputer = imputer

        self._split_function = split
        self.split = lambda X, y: split(X, y, **split_params)
        self.split_params = split_params

        search_params['cv'] = self.cv
        self.search = search(estimator, param_space, **search_params)

        Strategy._count += 1
        self.count = Strategy._count

        if not all(p in estimator.get_params().keys() for p in param_space.keys()):
            raise ValueError('Given parmameters must be params of estimator.')

    @property
    def name(self):
        if self._name is None:
            return self.estimator_class()
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def is_regression(self):
        """Return True if the estimator is a regressor."""
        return isinstance(self.estimator, RegressorMixin)

    def is_classification(self):
        """Return True if the estimator is a regressor."""
        return isinstance(self.estimator, ClassifierMixin)

    def estimator_class(self):
        return self.estimator.__class__.__name__

    def cv_class(self):
        return self.cv.__class__.__name__

    def search_class(self):
        return self.search.__class__.__name__

    def imputer_class(self):
        return self.imputer.__class__.__name__

    def get_infos(self):
        # Remove redondant params in the dump
        estimator_params = {
            k: v for k, v in self.estimator.__dict__.items() if k not in self.param_space
        }
        # Remove redondant params in the dump
        search_params = {
            k: v for k, v in self.search.__dict__.items() if k not in ['estimator', 'cv']
        }
        imputer_params = None if self.imputer is None else self.imputer.__dict__

        return {
            'name': self.name,
            'estimator': self.estimator_class(),
            'estimator_params': estimator_params,
            'split_function': self._split_function.__name__,
            'split_params': self.split_params,
            'cv': self.cv_class(),
            'cv_params': self.cv.__dict__,
            'search': self.search_class(),
            'search_params': search_params,
            'classification': self.is_classification(),
            'param_space': self.param_space,
            'imputer': self.imputer_class(),
            'imputer_params': imputer_params,
            'sklearn_version': sklearn.__version__
        }
