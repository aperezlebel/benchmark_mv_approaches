"""Implement the Strategy class."""
import logging

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Strategy():
    """Elements to run cross-validated ML estimator with hyperparms tuning."""

    def __init__(self, estimator, inner_cv, outer_cv, param_space, search,
                 imputer=None, search_params=dict(), compute_importance=False,
                 importance_params=dict(), learning_curve=False,
                 learning_curve_params=dict(), roc=False, name=None,
                 train_set_steps=None, min_test_set=None, n_splits=None):
        self.estimator = estimator
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.param_space = param_space
        self._name = name
        self.imputer = imputer
        self.compute_importance = compute_importance
        self.importance_params = importance_params
        self.learning_curve = learning_curve
        self.learning_curve_params = learning_curve_params
        self.roc = roc
        self.train_set_steps = train_set_steps
        self.min_test_set = min_test_set
        self.n_splits = n_splits

        if search is None:
            self.search = self.estimator

        else:
            if not all(p in estimator.get_params().keys() for p in self.param_space.keys()):
                self.param_space = dict()

            search_params['cv'] = self.inner_cv
            estimator = Pipeline([
                ('model', estimator)
            ])
            param_space = {f'model__{k}': v for k, v in self.param_space.items()}
            self.search = search(estimator, param_space, **search_params)

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

    def inner_cv_class(self):
        return self.inner_cv.__class__.__name__

    def outer_cv_class(self):
        return self.inner_cv.__class__.__name__

    def search_class(self):
        return self.search.__class__.__name__

    def imputer_class(self):
        return self.imputer.__class__.__name__

    def get_infos(self):
        # Remove redondant params in the dump
        if self.param_space:
            estimator_params = {
                k: v for k, v in self.estimator.__dict__.items() if k not in self.param_space
            }
        else:
            estimator_params = None
        # Remove redondant params in the dump
        search_params = {
            k: v for k, v in self.search.__dict__.items() if k not in ['estimator', 'cv']
        }
        imputer_params = None if self.imputer is None else self.imputer.__dict__

        # Retrieving params
        inner_cv = None if self.inner_cv is None else self.inner_cv.__dict__

        return {
            'name': self.name,
            'estimator': self.estimator_class(),
            'estimator_params': estimator_params,
            'inner_cv': self.inner_cv_class(),
            'inner_cv_params': inner_cv,
            'outer_cv': self.outer_cv_class(),
            'outer_cv_params': self.outer_cv.__dict__,
            'search': self.search_class(),
            'search_params': search_params,
            'classification': self.is_classification(),
            'param_space': self.param_space,
            'imputer': self.imputer_class(),
            'imputer_params': imputer_params,
            'compute_importance': self.compute_importance,
            'importance_params': self.importance_params,
            'learning_curve_params': self.learning_curve_params,
            'sklearn_version': sklearn.__version__
        }

    def reset_RS(self, RS):
        if RS is None:
            return  # Nothing to do

        RS = int(RS)

        for obj in [self.estimator, self.inner_cv, self.outer_cv, self.imputer]:
            if obj is not None and hasattr(obj, 'random_state'):
                logger.info(f'Reset RS for {obj.__class__.__name__} to {RS}.')
                obj.random_state = RS
