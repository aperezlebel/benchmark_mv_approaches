"""Test if tasks are correctly loaded."""
from unittest import TestCase

from prediction.tasks import tasks


class test_TB(TestCase):
    """Test TraumaBase tasks."""

    def test_hemo(self):
        """Test shock hemo task."""
        task = tasks['TB/hemo']
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

        assert X.shape == (n_rows, 99)
        assert y.shape == (n_rows,)
        assert not y.isna().any()

        assert task._f_y == meta.predict.output_features
