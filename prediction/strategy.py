"""Implement the Strategy class."""
from dataclasses import dataclass
from typing import Any, Callable
from sklearn.base import BaseEstimator


@dataclass()
class Strategy():
    """Elements to run cross-validated ML estimator with hyperparms tuning."""

    estimator: BaseEstimator
    split: Any
    cv: Any
    param_space: dict
    search: Callable[[BaseEstimator, dict, Any], Any]

    def __post_init__(self):
        """Check params and intialize search."""
        # Check
        e = self.estimator
        p = self.param_space
        if not all(p in e.get_params().keys() for p in p.keys()):
            raise ValueError('Given parmameters must be params of estimator.')

        # Intitialize search function with given parameters
        self.search = self.search(self.estimator, self.param_space, self.cv)
