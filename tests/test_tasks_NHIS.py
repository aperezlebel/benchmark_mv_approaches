"""Test if tasks are correctly loaded."""
import os
import yaml
from unittest import TestCase

from prediction.tasks import tasks


# Load some params from custom file
filepath = 'custom/strategy_params.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
else:
    params = dict()

n_top_pvals = params.get('n_top_pvals', 100)


class test_NHIS(TestCase):
    """Test NHIS tasks."""

    def test_income(self):
        """Test income cancer task."""
        task = tasks['NHIS/income_pvals']
        meta = task.meta

        assert not task.is_classif()

        X = task.X
        y = task.y

        n_rows = 20987

        assert X.shape[0] == n_rows
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features
        # L1 = list(X.columns)
        # L2 = meta.select.output_features
        # self.assertCountEqual(L1, L2)

    def test_bmi(self):
        """Test bmi cancer task."""
        task = tasks['NHIS/bmi_pvals']
        meta = task.meta

        assert not task.is_classif()

        X = task.X
        y = task.y

        n_rows = 11247

        assert X.shape[0] == n_rows
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features
        # L1 = list(X.columns)
        # L2 = meta.select.output_features
        # self.assertCountEqual(L1, L2)
