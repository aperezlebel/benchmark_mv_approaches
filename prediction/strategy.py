"""Implement the Strategy class."""
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


@dataclass()
class Strategy():
    """Elements to run cross-validated ML estimator with hyperparms tuning."""

    estimator: BaseEstimator
    split: Any
    cv: Any
    param_space: dict
    search: Callable[[BaseEstimator, dict, Any], Any]
    _name: str = None
    _count: int = field(default=0, init=False)

    def __post_init__(self):
        """Check params and intialize search."""
        Strategy._count += 1
        self.count = Strategy._count
        # Check
        e = self.estimator
        p = self.param_space
        if not all(p in e.get_params().keys() for p in p.keys()):
            raise ValueError('Given parmameters must be params of estimator.')

        # Intitialize search function with given parameters
        self.search = self.search(self.estimator, self.param_space, self.cv)

    @property
    def name(self):
        if self._name is None:
            return self.estimator_class()
        return self._name

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

    def get_infos(self):
        estimator_params = {
            k: v for k, v in self.estimator.__dict__.items() if k not in self.param_space
        }
        return {
            'name': self.name,
            'estimator': self.estimator_class(),
            'estimator_params': estimator_params,
            'cv': self.cv_class(),
            'cv_params': self.cv.__dict__,
            'classification': self.is_classification(),
            'param_space': self.param_space
        }
