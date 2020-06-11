"""Test if tasks are correctly loaded."""
import os
import yaml
from unittest import TestCase

from prediction.tasks import tasks


class test_TB(TestCase):
    """Test TraumaBase tasks."""

    def test_shock_hemo(self):
        """Test shock hemo task."""
        task = tasks['TB/shock_hemo']
        meta = task.meta

        assert task.is_classif()

        X = task.X
        y = task.y

        n_rows = 19569
        n_features = 12

        assert X.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features
        L1 = list(X.columns)
        L2 = meta.transform.output_features
        self.assertCountEqual(L1, L2)

    def test_acid(self):
        """Test acid tranexamic task."""
        task = tasks['TB/acid']
        meta = task.meta

        assert task.is_classif()

        X = task.X
        y = task.y

        n_rows = 1770
        n_features = 62

        assert X.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features

    def test_platelet(self):
        """Test platelet task."""
        task = tasks['TB/platelet']
        meta = task.meta

        assert not task.is_classif()

        X = task.X
        y = task.y

        n_rows = 19042
        n_features = 15

        assert X.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features

    def test_death(self):
        """Test death task."""
        task = tasks['TB/death_pvals']
        meta = task.meta

        assert task.is_classif()

        X = task.X
        y = task.y

        n_rows = 12341
        # n_features = 697

        assert X.shape[0] == n_rows
        # assert X.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features


# Load some params from custom file
filepath = 'custom/strategy_params.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
else:
    params = dict()

n_top_pvals = params.get('n_top_pvals', 100)


class test_UKBB(TestCase):
    """Test UKBB tasks."""

    def test_breast(self):
        """Test breast cancer task."""
        task = tasks['UKBB/breast_pvals']
        meta = task.meta

        assert task.is_classif()

        X = task.X
        y = task.y

        n_rows = 273384
        n_features = n_top_pvals

        assert X.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features
        L1 = list(X.columns)
        L2 = meta.select.output_features
        self.assertCountEqual(L1, L2)
