"""Test the Database class."""

from pytest import raises

from database.base import Database
from prediction.tasks.taskMeta import TaskMeta


def test_get_drop_and_keep_meta_drop():
    meta = TaskMeta(
        name='name',
        db='db',
        df_name='df_name',
        predict='predict',
        drop=['drop'],
        drop_contains=['drop_contains']
    )

    features = ['predict', 'drop', 'drop_contains_1', 'drop_contains_2']

    to_keep, to_drop = Database.get_drop_and_keep_meta(features, meta)

    assert to_keep == set(['predict'])
    assert to_drop == set(['drop', 'drop_contains_1', 'drop_contains_2'])


def test_get_drop_and_keep_meta_keep():
    meta = TaskMeta(
        name='name',
        db='db',
        df_name='df_name',
        predict='predict',
        keep=['keep'],
        keep_contains=['keep_contains']
    )

    features = ['predict', 'keep', 'keep_contains_1', 'drop']

    to_keep, to_drop = Database.get_drop_and_keep_meta(features, meta)

    assert to_keep == set(['predict', 'keep', 'keep_contains_1'])
    assert to_drop == set(['drop'])


def test_get_drop_and_keep_meta_keepand_drop():
    meta = TaskMeta(
        name='name',
        db='db',
        df_name='df_name',
        predict='predict',
        keep=['keep'],
        drop=['drop'],
        keep_contains=['keep_contains']
    )

    features = ['predict', 'keep', 'keep_contains_1', 'drop']

    with raises(ValueError):
        _, _ = Database.get_drop_and_keep_meta(features, meta)


if __name__ == '__main__':
    test_get_drop_and_keep_meta_drop()
